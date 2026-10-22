"""gpucheck.py - prove, on THIS machine, that training on the GPU works and is faster, before trusting it with hours of compute.

`python -m zerowhite chat gpu` runs it. The checks, each with a tolerance and a reason:

  1. kernels      a matrix product on the card matches NumPy (a broken CUDA / CuPy pairing fails here, in seconds);
  2. model        the SAME weights and batch give the same loss and gradients on the card as on the CPU, including repeated token ids
                  (the embedding-gradient scatter), dropout masks, packed documents and negative (preference) weights;
  3. optimiser    three AdamW steps from the same gradients land on the same weights, with ONE device->host transfer per step;
  4. speed        real training tokens/s of a reference model, CPU vs card, each timed with the device synchronised.

Tolerances allow for float32 summation order (cuBLAS vs the CPU's BLAS) and, if you enabled it (CUPY_TF32=1), the ~1e-3 error of TF32 matrix
units, which is reported rather than hidden. A failure names the check and the numbers; nothing here changes any workspace.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from . import backend as bk
from .model import GPT, GPTConfig
from .optim import AdamW

LOSS_RTOL = 2e-3                      # relative loss difference allowed between the card and the CPU
GRAD_RTOL = 2e-2                      # per-tensor: max |g_dev - g_cpu| / max |g_cpu|
PARAM_ATOL = 5e-4                     # after three optimiser steps at lr 1e-3


def _batch(rng, B: int, T: int, V: int) -> dict:
    ids = rng.integers(0, max(8, V // 8), (B, T)).astype(np.int64)          # a small active vocabulary: many repeated ids exercise the scatter
    seg = np.ones((B, T), np.int64)
    seg[:, T // 2:] = 2                                                     # two packed documents per row
    pos = np.concatenate([np.arange(T // 2), np.arange(T - T // 2)])[None].repeat(B, 0).astype(np.int64)
    w = (rng.random((B, T)) > 0.3).astype(np.float32)
    w[0, 3] = -0.5                                                          # a negative weight (preference / unlikelihood updates)
    return dict(ids=ids, tgt=np.roll(ids, -1, 1), w=w, pos=pos, seg=seg)


def _check(report: dict, name: str, ok: bool, detail: str, advice: str = ""):
    report["checks"].append(dict(name=name, ok=bool(ok), detail=detail, advice=advice if not ok else ""))
    report["ok"] = report["ok"] and bool(ok)


def run(be: bk.Backend, speed: bool = True, speed_rows: int = 16, speed_ctx: int = 256, say=None) -> dict:
    """Run every check against backend `be`. Returns a report dict: ok, checks=[{name, ok, detail, advice}], speed numbers, device info."""
    say = say or (lambda m: None)
    report = dict(ok=True, checks=[], device=be.describe(), backend=be.label())
    if not be.is_gpu:
        _check(report, "device", False, f"no GPU backend is active ({be.note or 'compute.backend is cpu'})",
               "install CuPy for your CUDA version (`pip install cupy-cuda12x`) and check `nvidia-smi`; then run this again")
        return report

    # 1. kernels --------------------------------------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    a = rng.standard_normal((256, 256)).astype(np.float32)
    got = be.to_host(be.asarray(a) @ be.asarray(a.T))
    ref = a @ a.T
    err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-12))
    _check(report, "kernels", err < 1e-3, f"256x256 float32 matrix product: relative error {err:.1e}",
           "the CUDA runtime and the CuPy build probably do not match (`pip show cupy-cuda12x`, `nvidia-smi`)")
    say(f"kernels    {'ok' if err < 1e-3 else 'FAIL'}  (rel err {err:.1e})")

    # 2. the model: same weights, same batch ------------------------------------------------------------------------------
    worst_loss, worst_grad, finite = 0.0, 0.0, True
    for tie in (True, False):
        cfg = GPTConfig(vocab=97, d_model=32, n_layer=2, n_head=4, ffn=64, ctx=64, tie=tie)
        cpu, dev = GPT(cfg, seed=3), GPT(cfg, seed=3, backend=be)
        b = _batch(np.random.default_rng(1), 3, 40, 97)
        lc, gc, _ = cpu.loss_and_grads(b, 0.2, np.random.default_rng(9))
        ld, gd, _ = dev.loss_and_grads(b, 0.2, np.random.default_rng(9))
        be.sync()
        worst_loss = max(worst_loss, abs(ld - lc) / (abs(lc) + 1e-12))
        for k in gc:
            g = be.to_host(gd[k])
            finite = finite and bool(np.all(np.isfinite(g)))
            worst_grad = max(worst_grad, float(np.max(np.abs(g - gc[k])) / (np.max(np.abs(gc[k])) + 1e-12)))
    ok = finite and worst_loss < LOSS_RTOL and worst_grad < GRAD_RTOL
    tf32 = os.environ.get("CUPY_TF32", "0") not in ("", "0")
    _check(report, "model", ok, f"loss rel diff {worst_loss:.1e}, worst gradient rel diff {worst_grad:.1e}" + (" (TF32 is on: larger error is expected)" if tf32 else ""),
           "the device model disagrees with the CPU model: do not train on this setup; report it with the numbers above")
    say(f"model      {'ok' if ok else 'FAIL'}  (loss {worst_loss:.1e}, grads {worst_grad:.1e})")

    # 3. the optimiser ------------------------------------------------------------------------------------------------------
    cfg = GPTConfig(vocab=97, d_model=32, n_layer=2, n_head=4, ffn=64, ctx=64)
    cpu, dev = GPT(cfg, seed=3), GPT(cfg, seed=3, backend=be)
    b = _batch(np.random.default_rng(2), 3, 40, 97)
    _, gc, _ = cpu.loss_and_grads(b)
    _, gd, _ = dev.loss_and_grads(b)
    oc, od = AdamW(cpu.p), AdamW(dev.p, backend=be)
    for _ in range(3):
        oc.step(cpu.p, gc, 1e-3, 1.0)
        od.step(dev.p, gd, 1e-3, 1.0)
    be.sync()
    pdiff = max(float(np.max(np.abs(cpu.p[k] - be.to_host(dev.p[k])))) for k in cpu.p)
    _check(report, "optimiser", pdiff < PARAM_ATOL, f"three AdamW steps: largest weight difference {pdiff:.1e}",
           "the device optimiser disagrees with the CPU one")
    say(f"optimiser  {'ok' if pdiff < PARAM_ATOL else 'FAIL'}  (max diff {pdiff:.1e})")

    # 4. speed ----------------------------------------------------------------------------------------------------------------
    if speed and report["ok"]:
        from . import autoscale as AS
        ref_cfg = GPTConfig(vocab=2000, d_model=128, n_layer=3, n_head=4, ffn=344, ctx=speed_ctx)
        t = time.time()
        tps_cpu = AS.bench(ref_cfg, max(1, speed_rows // 4), speed_ctx, 1, bk.CPU)
        tps_dev = AS.bench(ref_cfg, speed_rows, speed_ctx, 3, be)
        report.update(tokens_per_s_cpu=round(tps_cpu), tokens_per_s_gpu=round(tps_dev), speedup=round(tps_dev / max(tps_cpu, 1e-9), 1))
        _check(report, "speed", tps_dev > tps_cpu, f"reference model (128d x 3L, ctx {speed_ctx}): CPU {tps_cpu:,.0f} tokens/s, GPU {tps_dev:,.0f} tokens/s "
               f"= {tps_dev / max(tps_cpu, 1e-9):.1f}x ({time.time() - t:.0f}s to measure)",
               "the card is slower than the CPU on this reference size; check that nothing else is using it, and that the driver is not in a low-power state")
        say(f"speed      {'ok' if tps_dev > tps_cpu else 'FAIL'}  ({report['speedup']}x)")
    be.release()
    return report


def format_report(r: dict) -> str:
    d = r["device"]
    lines = [f"device : {r['backend']}" + (f"   CuPy {d['cupy']}" if d.get("cupy") else "")]
    if d.get("vram_free_gb") is not None:
        lines.append(f"memory : {d['vram_free_gb']} GB free of {d.get('vram_gb')} GB")
    for c in r["checks"]:
        lines.append(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['name']:<10} {c['detail']}")
        if c["advice"]:
            lines.append(f"        -> {c['advice']}")
    lines.append("")
    if r["ok"]:
        lines.append("The GPU is ready: `compute.backend=auto` (the default) will use it for training. Existing models keep their size; "
                     "to get a bigger one train a new model (`zerowhite chat train --fresh`).")
    else:
        lines.append("Not ready: training will stay on the CPU (auto) or stop with an explanation (compute.backend=gpu).")
    return "\n".join(lines)
