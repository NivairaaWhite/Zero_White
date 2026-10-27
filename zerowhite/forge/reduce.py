"""reduce.py - let program agents see ALL of a high-dimensional observation, not the first 24 numbers of it.

`ProgramPolicyAgent` evolves a small Python expression over `obs[0] ... obs[n-1]`. Until 6.3 it kept `min(obs_dim, 24)` columns: the FIRST 24, by
position. On an environment with a 40-number observation the other 16 were invisible, and if the informative signal lived in them the agent
could not exceed chance however long it searched (the docs list this as a limit; the 'Perceiver in front of the program agent' idea of the
Gemini review is the same fix at larger scale).

The fix here is deliberately small and exact: a whitened PCA projector, fitted with plain numpy (an SVD) on observations gathered by a random
policy before the search starts. The program then sees k <= 24 components that together carry the variance of ALL the raw numbers, each scaled to
about unit variance so one set of mutation constants fits every component. It is only used when obs_dim > MAX_COLS, so every existing
task behaves exactly as before. The projector is stored with the agent's state so a reloaded agent sees the same components.

HONEST LIMITS: PCA is linear and unsupervised, and it is fitted on what a RANDOM policy sees; a feature that only matters in states a random
policy never reaches can be lost, and a high-variance irrelevant dimension can crowd out a low-variance relevant one. It is a much better
default than truncation, not a substitute for a learned encoder; and component signs are arbitrary, so evolved programs may read `obs[0] < 0`
where a person would write `> 0`. Image-sized observations (thousands of numbers) are beyond what 48-256 probe samples can support.
"""
from __future__ import annotations

import numpy as np

MAX_COLS = 24          # how many numbers a program agent can read (unchanged)
MIN_SAMPLES = 8


class Projector:
    def __init__(self, mean, scale, comps, sd):
        self.mean, self.scale = np.asarray(mean, float), np.asarray(scale, float)
        self.comps, self.sd = np.asarray(comps, float), np.asarray(sd, float)

    @property
    def k(self) -> int:
        return int(self.comps.shape[0])

    @classmethod
    def fit(cls, obs, k: int = MAX_COLS) -> "Projector":
        X = np.asarray(obs, dtype=float)
        if X.ndim != 2 or X.shape[0] < MIN_SAMPLES:
            raise ValueError(f"need at least {MIN_SAMPLES} observations of a fixed width to fit a projector")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale > 1e-9, scale, 1.0)                           # a constant column stays 0 after centring
        Z = (X - mean) / scale
        _, s, vt = np.linalg.svd(Z, full_matrices=False)
        var = s ** 2 / max(1, Z.shape[0] - 1)
        k = int(max(1, min(k, vt.shape[0], int(np.sum(var > 1e-9 * max(var[0], 1e-30))))))
        sd = np.sqrt(np.maximum(var[:k], 0.0))
        sd = np.maximum(sd, 1e-3 * sd[0])                                    # do not blow up near-empty components
        return cls(mean, scale, vt[:k], sd)

    def __call__(self, o) -> np.ndarray:
        z = (np.nan_to_num(np.asarray(o, dtype=float), nan=0.0, posinf=0.0, neginf=0.0) - self.mean) / self.scale
        return (self.comps @ z) / self.sd

    def to_dict(self) -> dict:
        return dict(mean=self.mean.tolist(), scale=self.scale.tolist(), comps=self.comps.tolist(), sd=self.sd.tolist())

    @classmethod
    def from_dict(cls, d: dict) -> "Projector":
        return cls(d["mean"], d["scale"], d["comps"], d["sd"])


def needed(obs_dim: int) -> bool:
    return int(obs_dim) > MAX_COLS
