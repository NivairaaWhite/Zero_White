"""checkpoint.py - save/resume a whole Zero White run.

Everything (weights, optimizer rule + momentum, curiosity model, self-model
policy + journal, curriculum position, RNG streams) goes into ONE npz written
atomically, so weights and state can never disagree after a crash. v2
checkpoints (weights.npz + state.json) still load.
"""

import json
import os

import numpy as np

from . import fileio
from .zero_white import ZeroWhite
from .self_model import SelfModel

FILE = "weights.npz"
_OPT_KEYS = ("W1", "b1", "W2", "b2")


def save(zw, path, curriculum=None, extra=None):
    os.makedirs(path, exist_ok=True)
    net, arrays = zw.net, {}
    for i, l in enumerate(net.layers):
        arrays[f"L{i}.W"], arrays[f"L{i}.b"] = l["W"], l["b"]
    if net.patch:
        for k, v in net.patch.items():
            arrays[f"patch.{k}"] = v
    for k in _OPT_KEYS:
        arrays[f"opt.{k}"] = getattr(zw.opt, k)
        arrays[f"icm.{k}"] = getattr(zw.curiosity, k)
    for name, m in zw.opt.momentum.items():
        arrays[f"mom.{name}"] = m
    meta = dict(
        version=3, in_dim=net.in_dim, out_dim=net.out_dim, num_chunks=net.num_chunks,
        patch_dims=[net.patch_local, net.patch_global],
        n_layers=len(net.layers), use_patch=net.use_patch, has_patch=net.patch is not None,
        task=zw.task, candidates_per_gen=zw.candidates_per_gen, lr_soft_cap=zw.lr_soft_cap,
        lr_penalty=zw.lr_penalty, lr_floor=zw.lr_floor, batch_size=zw.batch_size,
        consolidate_steps=zw.consolidate_steps, steps_trained=zw.steps_trained,
        opt=zw.opt.hyper(), icm_lr=zw.curiosity.lr,
        rng=dict(zw=fileio.rng_state(zw.rng), net=fileio.rng_state(net.rng),
                 opt=fileio.rng_state(zw.opt.rng)),
        self_model=zw.self_model.to_dict(),
        curriculum=curriculum.to_dict() if curriculum else None,
        extra=extra or {})
    arrays["__state__"] = np.frombuffer(json.dumps(meta, default=fileio._json_default).encode("utf-8"),
                                        dtype=np.uint8)
    fileio.save_npz(os.path.join(path, FILE), arrays)


def _read_meta(w, path):
    if "__state__" in w:
        return json.loads(bytes(w["__state__"]).decode("utf-8"))
    return fileio.read_json(os.path.join(path, "state.json"))      # v2 layout


def exists(path):
    return os.path.exists(os.path.join(path, FILE))


def load(path):
    from .tasks import Curriculum
    w = fileio.load_npz(os.path.join(path, FILE))       # fully read + closed: Windows can't replace an open file
    m = _read_meta(w, path)
    zw = ZeroWhite(m["in_dim"], m["out_dim"], hidden=(2,), task=m["task"], num_chunks=m["num_chunks"],
                   candidates_per_gen=m["candidates_per_gen"], lr_soft_cap=m["lr_soft_cap"],
                   lr_penalty=m["lr_penalty"], lr_floor=m.get("lr_floor", 0.01),
                   self_model=SelfModel.from_dict(m["self_model"]), batch_size=m.get("batch_size"),
                   consolidate_steps=m.get("consolidate_steps", 0),
                   patch_dims=tuple(m.get("patch_dims", (6, 8))))
    zw.net.layers = [{"W": w[f"L{i}.W"], "b": w[f"L{i}.b"]} for i in range(m["n_layers"])]
    zw.net.use_patch = m["use_patch"]
    zw.net.patch = {k: w[f"patch.{k}"] for k in ("Wl", "bl", "Wg", "bg")} if m["has_patch"] else None
    for k in _OPT_KEYS:
        setattr(zw.opt, k, w[f"opt.{k}"].copy())
        setattr(zw.curiosity, k, w[f"icm.{k}"].copy())
    zw.opt.momentum = {k[4:]: w[k].copy() for k in w if k.startswith("mom.")}
    hyper = m.get("opt") or {"base_lr": m["base_lr"]}
    for k, v in hyper.items():
        setattr(zw.opt, k, v)
    zw.curiosity.lr = m.get("icm_lr", zw.curiosity.lr)
    zw.steps_trained = m.get("steps_trained", 0)
    if "rng" in m:                                   # resume the random streams, don't replay seed 0
        zw.rng = fileio.rng_from_state(m["rng"]["zw"])
        zw.net.rng = fileio.rng_from_state(m["rng"]["net"])
        zw.opt.rng = fileio.rng_from_state(m["rng"]["opt"])
    cur = Curriculum.from_dict(m["curriculum"], zw.net) if m.get("curriculum") else None
    return zw, cur
