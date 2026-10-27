"""optim.py - AdamW with global-norm clipping, a cosine schedule, and exact save/restore of its state."""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from . import backend as bk


def cosine_lr(peak: float, min_ratio: float, warmup: int, step: int, frac: float) -> float:
    """Linear warm-up for `warmup` steps, then cosine decay from peak to peak*min_ratio as `frac` (0..1) of the run elapses."""
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    f = min(1.0, max(0.0, frac))
    return peak * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * f)))


class AdamW:
    """AdamW. `params` and `grads` live wherever `backend` says (the CPU by default); only plain floats and, on request, host copies
    of the state ever cross to the host."""

    def __init__(self, params: Dict[str, np.ndarray], beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1,
                 backend: Optional[bk.Backend] = None):
        self.b1, self.b2, self.eps, self.wd = beta1, beta2, eps, weight_decay
        self.be = backend or bk.CPU
        self.xp = self.be.xp
        self.m = {k: self.xp.zeros_like(v) for k, v in params.items()}
        self.v = {k: self.xp.zeros_like(v) for k, v in params.items()}
        self.t = 0

    @staticmethod
    def global_norm(grads: Dict[str, np.ndarray]) -> float:
        """Host arrays only (kept for callers that have them); the optimiser itself uses `norm`, which also serves a device."""
        return float(math.sqrt(sum(float(np.vdot(g, g)) for g in grads.values())))

    def norm(self, grads: Dict[str, np.ndarray]) -> float:
        """Global gradient norm with ONE transfer to the host: the per-array dot products are stacked on the device first. A float()
        per array would be one synchronisation each (a hundred or more per step on a deep model)."""
        xp = self.xp
        parts = [xp.vdot(g, g) for g in grads.values()]
        if not parts:
            return 0.0
        return float(math.sqrt(sum(float(x) for x in self.be.to_host(xp.stack(parts)))))

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], lr: float,
             clip: Optional[float] = None) -> Dict[str, float]:
        """One update in place. A non-finite gradient skips the step entirely (returns applied=0) so a single bad batch
        can never poison the weights."""
        gn = self.norm(grads)
        if not math.isfinite(gn):
            return dict(grad_norm=gn, applied=0.0)
        scale = 1.0
        if clip and gn > clip:
            scale = clip / (gn + 1e-12)
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k, g in grads.items():
            p = params[k]
            m, v = self.m[k], self.v[k]
            if scale != 1.0:
                g = g * scale
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * (g * g)
            upd = (m / bc1) / (self.xp.sqrt(v / bc2) + self.eps)
            if self.wd and p.ndim >= 2:
                upd += self.wd * p
            p -= (lr * upd).astype(p.dtype)
        return dict(grad_norm=gn, applied=1.0)

    # ---- state ------------------------------------------------------------------------------------------------
    def state_arrays(self, prefix: str = "opt") -> Dict[str, np.ndarray]:
        out = {f"{prefix}.m.{k}": self.be.to_host(v) for k, v in self.m.items()}
        out.update({f"{prefix}.v.{k}": self.be.to_host(v) for k, v in self.v.items()})
        out[f"{prefix}.t"] = np.array([self.t], dtype=np.int64)
        return out

    def load_state(self, arrays: Dict[str, np.ndarray], prefix: str = "opt") -> bool:
        try:
            for k in self.m:
                self.be.assign(self.m[k], arrays[f"{prefix}.m.{k}"])
                self.be.assign(self.v[k], arrays[f"{prefix}.v.{k}"])
            self.t = int(arrays[f"{prefix}.t"][0])
            return True
        except (KeyError, ValueError):
            return False
