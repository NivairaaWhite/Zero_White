"""grow.py - make the chat model bigger WITHOUT forgetting what it has learned (Net2Net-style, function-preserving).

Until 6.3 a model kept its size for life ("use `chat train --fresh` for a bigger one"): growing meant throwing the weights away and starting
again, so a model that began small because the data was small could never use the data it later got. `grow` returns a bigger GPT that
computes EXACTLY the same function (same logits, same loss) as the one it came from, then training simply continues; the new capacity starts
idle and learns from there.

How each kind of growth keeps the function unchanged (the architecture is the Llama-style block in model.py):

  layers   a new residual block whose output projections `wo` and `wd` are ZERO adds exactly nothing to the residual stream. Its other
           weights are copied from the block below it (init="copy": it starts from useful features) or drawn at random (init="random").
           Gradients reach `wo`/`wd` immediately and the rest follows as they move away from zero.
  ffn      extra SwiGLU hidden units: new columns in `wg`/`wu`, and new ROWS in `wd` that are zero.
  heads    extra attention heads, which needs a wider residual stream (d_model = n_head * head_dim). The old channels are untouched; the new
           channels hold zeros (embedding columns zero, every write into them zero). RMSNorm divides by the root-mean-square over ALL channels,
           so padding with zeros would rescale it by sqrt(D_new / D_old); the norm gains are multiplied by the inverse factor and the norm's
           epsilon by D_old / D_new (stored in the grown model's config), which makes the correction exact rather than approximate.
           New heads read the old channels through new columns of wq/wk/wv and write to them through new rows of `wo`, which are zero.

NOT done: mixture-of-experts blocks, shrinking, changing the vocabulary or the context length, and any automatic decision of WHEN to grow
(`chat grow` is a command you run). Growing never touches
the tokenizer, the data, or the old weights file: the grown model is a NEW version and `chat rollback` returns to the old one.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from .model import GPT, GPTConfig, count_params


def _std(a: np.ndarray, default: float = 0.02) -> float:
    s = float(np.std(a))
    return s if s > 1e-8 else default


def _grow_heads(p: Dict[str, np.ndarray], c: GPTConfig, k: int, rng) -> tuple:
    hd = c.head_dim
    D, D2 = c.d_model, c.d_model + k * hd
    q = dict()
    scale = math.sqrt(D / D2)
    for name, a in p.items():
        if name == "tok_emb" or name == "lm_head":
            q[name] = np.concatenate([a, np.zeros((a.shape[0], D2 - D), a.dtype)], axis=1)       # new channels start empty
        elif name == "nf" or name.endswith((".n1", ".n2")):
            q[name] = np.concatenate([a * scale, np.ones(D2 - D, a.dtype)])                      # undo the RMS rescale caused by the zero padding
        elif name.endswith((".wq", ".wk", ".wv")):
            w = np.zeros((D2, D2), a.dtype)
            w[:D, :D] = a
            s = _std(a)
            w[D:, :] = rng.standard_normal((D2 - D, D2)).astype(a.dtype) * s                    # padded channels are zero: reading them is harmless
            w[:D, D:] = rng.standard_normal((D, D2 - D)).astype(a.dtype) * s                    # the new heads' own projections
            q[name] = w
        elif name.endswith(".wo"):
            w = np.zeros((D2, D2), a.dtype)                                                      # new heads write nothing yet; nothing writes the new channels
            w[:D, :D] = a
            q[name] = w
        elif name.endswith((".wg", ".wu")):
            w = np.zeros((D2, a.shape[1]), a.dtype)
            w[:D] = a
            w[D:] = rng.standard_normal((D2 - D, a.shape[1])).astype(a.dtype) * _std(a)
            q[name] = w
        elif name.endswith(".wd"):
            q[name] = np.concatenate([a, np.zeros((a.shape[0], D2 - D), a.dtype)], axis=1)
        else:
            raise KeyError(f"unknown parameter {name}")
    # RMSNorm computes x / sqrt(mean(x^2) + eps). Padding with zeros turns mean(x^2) into (D/D2) * mean(x^2); choosing eps' = eps * D/D2 makes the
    # whole denominator exactly sqrt(D/D2) times the old one, so the gain correction above is exact for every input, not just for large activations.
    cfg = GPTConfig.from_dict(dict(c.to_dict(), d_model=D2, n_head=c.n_head + k, eps=c.eps * D / D2))
    return q, cfg


def _grow_ffn(p: Dict[str, np.ndarray], c: GPTConfig, extra: int, rng) -> tuple:
    q = dict(p)
    for l in range(c.n_layer):
        wg, wu, wd = p[f"b{l}.wg"], p[f"b{l}.wu"], p[f"b{l}.wd"]
        q[f"b{l}.wg"] = np.concatenate([wg, rng.standard_normal((wg.shape[0], extra)).astype(wg.dtype) * _std(wg)], axis=1)
        q[f"b{l}.wu"] = np.concatenate([wu, rng.standard_normal((wu.shape[0], extra)).astype(wu.dtype) * _std(wu)], axis=1)
        q[f"b{l}.wd"] = np.concatenate([wd, np.zeros((extra, wd.shape[1]), wd.dtype)], axis=0)    # the new units add nothing yet
    return q, GPTConfig.from_dict(dict(c.to_dict(), ffn=c.ffn + extra))


def _grow_layers(p: Dict[str, np.ndarray], c: GPTConfig, n: int, at: int, init: str, rng) -> tuple:
    L = c.n_layer
    at = L if at is None else int(at)
    if not 0 <= at <= L:
        raise ValueError(f"cannot insert at {at}: the model has {L} layers (0..{L})")
    q = {k: v for k, v in p.items() if not k.startswith("b")}
    for l in range(L):
        new_l = l if l < at else l + n
        for key in ("n1", "wq", "wk", "wv", "wo", "n2", "wg", "wu", "wd"):
            q[f"b{new_l}.{key}"] = p[f"b{l}.{key}"]
    src = max(0, at - 1) if init == "copy" else None
    for j in range(n):
        pre = f"b{at + j}."
        for key in ("n1", "n2"):
            q[pre + key] = p[f"b{src}.{key}"].copy() if src is not None else np.ones_like(p[f"b{0}.{key}"])
        for key in ("wq", "wk", "wv", "wg", "wu"):
            ref = p[f"b{max(0, at - 1)}.{key}"]
            q[pre + key] = ref.copy() if src is not None else (rng.standard_normal(ref.shape).astype(ref.dtype) * _std(ref))
        q[pre + "wo"] = np.zeros_like(p["b0.wo"])                                                 # these two zeros are what keeps the function
        q[pre + "wd"] = np.zeros_like(p["b0.wd"])
    return q, GPTConfig.from_dict(dict(c.to_dict(), n_layer=L + n))


def to_moe(model: GPT, experts: int, top_k: int = 2, router_std: float = 0.02, aux: float = 0.01, seed: int = 0) -> GPT:
    """Turn every dense feed-forward block into a mixture of `experts` copies of it with a small random router. Function-preserving: the gates of the `top_k` chosen
    experts sum to 1 and the experts are identical, so the output is the dense block's output whatever the router says. The experts then diverge because different
    tokens go to different experts and so give them different gradients. Costs `experts` times the feed-forward parameters, but only `top_k` of them are used per token."""
    c = model.cfg
    if c.n_experts > 1:
        raise ValueError("the model is already a mixture of experts")
    if experts < 2 or not 1 <= top_k <= experts:
        raise ValueError("need experts >= 2 and 1 <= top_k <= experts")
    rng = np.random.default_rng(seed)
    p = {k: np.array(v) for k, v in model.host_params().items()}
    q = {k: v for k, v in p.items() if not k.endswith((".wg", ".wu", ".wd"))}
    for l in range(c.n_layer):
        q[f"b{l}.rt"] = (rng.standard_normal((c.d_model, experts)) * router_std).astype(p[f"b{l}.wg"].dtype)
        for j in range(experts):
            for key in ("wg", "wu", "wd"):
                q[f"b{l}.e{j}.{key}"] = p[f"b{l}.{key}"].copy()
    cfg = GPTConfig.from_dict(dict(c.to_dict(), n_experts=int(experts), top_k=int(top_k), moe_aux=float(aux)))
    return GPT(cfg, params=q, backend=model.be)


def grow(model: GPT, layers: int = 0, ffn: int = 0, heads: int = 0, at: Optional[int] = None, init: str = "copy", seed: int = 0) -> GPT:
    """A bigger model computing the same function as `model` (see the module docstring). `at` is where new layers go (default: the top)."""
    if model.cfg.n_experts > 1 and (layers or ffn or heads):
        raise ValueError("layers / ffn / heads can only be added to a dense model: grow first, then convert with to_moe")
    if min(layers, ffn, heads) < 0:
        raise ValueError("growth only adds capacity: layers, ffn and heads must be >= 0")
    if init not in ("copy", "random"):
        raise ValueError("init must be 'copy' or 'random'")
    rng = np.random.default_rng(seed)
    p = {k: np.array(v) for k, v in model.host_params().items()}
    c = model.cfg
    if heads:
        p, c = _grow_heads(p, c, int(heads), rng)
    if ffn:
        p, c = _grow_ffn(p, c, int(ffn), rng)
    if layers:
        p, c = _grow_layers(p, c, int(layers), at, init, rng)
    if not (layers or ffn or heads):
        return model.clone()
    return GPT(c, params=p, backend=model.be)


def describe(old: GPTConfig, new: GPTConfig) -> str:
    a, b = count_params(old)[0], count_params(new)[0]
    return (f"{old.n_layer} layers x d_model {old.d_model} x {old.n_head} heads x ffn {old.ffn} ({a:,} params)  ->  "
            f"{new.n_layer} x {new.d_model} x {new.n_head} x {new.ffn} ({b:,} params)")


def grow_current(lay, layers: int = 0, ffn: int = 0, heads: int = 0, at: Optional[int] = None, init: str = "copy", seed: int = 0, experts: int = 0, top_k: int = 2) -> dict:
    """Grow the live model of a chat workspace into a NEW version and make it live. The old file stays (`chat rollback` / `chat versions`).
    Refuses while a training run holds the lock, because that run would overwrite the pointer when it finishes."""
    from .model import read_meta
    from .util import FileLock, LockBusy
    path = lay.current_weights()
    if not path:
        raise RuntimeError("there is no trained model to grow: run `zerowhite chat train` first")
    lk = FileLock(lay.train_lock)
    try:
        lk.acquire()
    except LockBusy:
        raise RuntimeError("a training run is active; grow the model when it has finished") from None
    try:
        model, meta = GPT.load(path)
        grown = grow(model, layers=layers, ffn=ffn, heads=heads, at=at, init=init, seed=seed)
        if experts:
            grown = to_moe(grown, experts, top_k=top_k, seed=seed)
        meta = {k: v for k, v in meta.items() if k not in ("file", "updated")}
        n = int(meta.get("grown", 0)) + 1
        meta.update(grown=n, grown_from=lay.current()["file"], grown_ops=dict(layers=layers, ffn=ffn, heads=heads, at=at, init=init, experts=experts, top_k=top_k),
                    params_before=model.num_params(), params=grown.num_params())
        out = lay.new_weights_path("grow", n)
        grown.save(out, meta=meta)
        lay.set_current(out, **meta)
        return dict(file=out, old=model.cfg, new=grown.cfg, summary=describe(model.cfg, grown.cfg), config=read_meta(out)["config"])
    finally:
        lk.release()
