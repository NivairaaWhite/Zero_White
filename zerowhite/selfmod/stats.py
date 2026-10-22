"""stats.py - is this improvement real, or is it seed luck?

Until 6.3 a self-written patch or hook was accepted when its MEAN score on unseen seeds beat the incumbent's by a fixed margin. A mean over a
handful of seeds can clear a margin because ONE seed went well; the margin says how big the gain is, not how much to believe it. This module
adds the second half: a paired, distribution-free significance test on the per-seed differences.

  signflip_p(diffs)   one-sided p-value for "the mean difference is > 0". Under the null hypothesis that the patch changed nothing, each
                      paired difference is as likely to be +d as -d, so the observed mean is compared with the means of every
                      re-signing of the differences. Exact (all 2**n signings) for n <= EXACT_MAX, else Monte Carlo with the +1 correction
                      that keeps p > 0. No normality assumption, nothing but numpy.
  bootstrap_ci(diffs) percentile interval for the mean difference (reported, not used to decide).
  gate(base, new)     accept iff n >= min_n AND mean gain > margin AND p < alpha. Returns a dict that says which condition failed.

Smallest possible p with n pairs is 1 / 2**n (exact) so the gate needs n >= 5 pairs to ever reach alpha = 0.05 (1/32 = 0.031): fewer seeds
than that cannot establish significance, and the gate says so instead of guessing.

Part of the CONSTITUTION: the self-modifier must not be able to loosen the test that judges it (see CONSTITUTION.sha256).
"""
from __future__ import annotations

import itertools

import numpy as np

EXACT_MAX = 16


def signflip_p(diffs, n_resamples: int = 20000, seed: int = 0) -> float:
    d = np.asarray(diffs, dtype=np.float64).ravel()
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return 1.0
    obs = d.mean()
    tol = 1e-12 * max(1.0, float(np.abs(d).max()))
    if n <= EXACT_MAX:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))      # (2**n, n)
        means = (signs * d).mean(axis=1)
        return float((means >= obs - tol).sum() / means.shape[0])
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(int(n_resamples), n))
    means = (signs * d).mean(axis=1)
    return float((1 + (means >= obs - tol).sum()) / (1 + int(n_resamples)))


def bootstrap_ci(diffs, level: float = 0.95, n_resamples: int = 4000, seed: int = 0):
    d = np.asarray(diffs, dtype=np.float64).ravel()
    d = d[np.isfinite(d)]
    if d.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(int(n_resamples), d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return (float(lo), float(hi))


def gate(base, new, margin: float = 0.0, alpha: float = 0.05, min_n: int = 5, seed: int = 0) -> dict:
    """Judge `new` against `base`, paired by position (same seed / same function in the same slot)."""
    b, n = np.asarray(base, dtype=np.float64).ravel(), np.asarray(new, dtype=np.float64).ravel()
    if b.shape != n.shape:
        raise ValueError(f"paired scores must have the same length ({b.size} vs {n.size})")
    diffs = n - b
    ok = np.isfinite(diffs)
    diffs = diffs[ok]
    out = dict(n=int(diffs.size), margin=float(margin), alpha=None if alpha is None else float(alpha))
    if diffs.size == 0:
        return dict(out, accept=False, mean_gain=float("nan"), p=1.0, ci=(float("nan"), float("nan")), positive=0, reason="no comparable scores")
    gain = float(diffs.mean())
    p = signflip_p(diffs, seed=seed)
    ci = bootstrap_ci(diffs, seed=seed)
    out.update(mean_gain=gain, p=p, ci=ci, positive=int((diffs > 0).sum()))
    if alpha is None:                                                          # the statistical half switched off: the old behaviour
        return dict(out, accept=gain > margin, reason="" if gain > margin else f"mean gain {gain:+.4f} does not exceed the margin {margin:g}")
    if gain <= margin:
        return dict(out, accept=False, reason=f"mean gain {gain:+.4f} over {diffs.size} paired scores does not exceed the margin {margin:g}")
    if diffs.size < min_n:
        return dict(out, accept=False, reason=f"only {diffs.size} paired scores: fewer than {min_n} cannot reach p < {alpha:g} however large the gain")
    if p >= alpha:
        return dict(out, accept=False, reason=f"gain {gain:+.4f} is not significant: p = {p:.3f} >= {alpha:g} "
                                              f"({out['positive']} of {diffs.size} paired scores improved)")
    return dict(out, accept=True, reason="")
