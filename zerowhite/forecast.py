"""forecast.py - what will the next attempt cost, and how wrong has that kind of guess been?

Until 6.3 the organism predicted an attempt's wall-clock cost as (seconds per unit of budget measured on the previous attempt) x budget,
padded by a fixed `time_margin` of 1.5. Both halves were guesses: the cost of an attempt is not linear in its budget (the docs record 2 s
predicted / 8 s taken, 56 s predicted / 86 s taken), and a constant pad cannot be right for every task and every machine.

This module replaces the guesses by two things learned from the organism's own history, per task:

  fit_power / predict   seconds ~ a * units**gamma, fitted robustly (Theil-Sen on log-log, gamma clipped to [0.7, 1.6]) on the last
                        WINDOW attempts for the exponent and the last FIT_WINDOW (12) for the level, so that a long run of same-size attempts does
                        not erase what is known about how cost scales, while throttling or other load is followed quickly. Needs MIN_FIT attempts at >= 2 different sizes; until then the caller keeps the old linear formula.
  overrun_margin        the pad. Every finished attempt whose prediction came from the fitted curve contributes one ratio actual / predicted (what
                        the organism said when it decided). Ratios of the earlier linear probe are NOT mixed in: they come from a different, much
                        cruder predictor, and its large early overruns would inflate the pad for the next 24 attempts (seen in a real run: x3.4).
                        The pad is the split-conformal quantile of those ratios: with n ratios, take the ceil((n + 1) * q)-th smallest.
                        Under the (modest) assumption that the next ratio is exchangeable with the recent ones, the next attempt then fits
                        inside prediction x pad with probability >= q. With too few ratios to say that (n < q / (1 - q) - 1) the pad is the
                        old fixed one, or the largest overrun seen if that is bigger. The pad is bounded to [1, 10].

Pure Python on purpose: it is called once per decision with at most a few dozen numbers.
"""
from __future__ import annotations

import math

WINDOW = 24            # attempts remembered per task (and overrun ratios kept for the pad)
FIT_WINDOW = 12        # ... of which the LEVEL of the curve is taken from the most recent ones (the exponent from all of them): follows slow drift
MIN_FIT = 4            # attempts needed before the fitted curve replaces the linear probe
GAMMA_LO, GAMMA_HI = 0.7, 1.6
PAD_LO, PAD_HI = 1.0, 10.0


def _median(v):
    s = sorted(v)
    n = len(s)
    if not n:
        raise ValueError("median of nothing")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def clean(obs):
    """Keep only well-formed [units, seconds, ...] rows with positive numbers (a damaged state file must not break scheduling)."""
    out = []
    for o in obs or []:
        try:
            u, s = float(o[0]), float(o[1])
        except (TypeError, ValueError, IndexError):
            continue
        if u > 0 and s > 0 and math.isfinite(u) and math.isfinite(s):
            out.append([u, s])
    return out[-WINDOW:]


def fit_power(obs):
    """(a, gamma) with seconds ~ a * units**gamma, or None when the history cannot support a curve.

    Theil-Sen on (log units, log seconds): the slope is the median of the pairwise slopes, so one outlier (a throttled or interrupted
    attempt) moves it very little. Pairs whose sizes differ by under 2% carry no information about the slope and are skipped."""
    pts = clean(obs)
    if len(pts) < MIN_FIT:
        return None
    lx = [math.log(u) for u, _ in pts]
    ly = [math.log(s) for _, s in pts]
    slopes = [(ly[j] - ly[i]) / (lx[j] - lx[i]) for i in range(len(pts)) for j in range(i + 1, len(pts)) if abs(lx[j] - lx[i]) > 0.02]
    if not slopes:
        return None                                  # every attempt had the same size: nothing is known about how cost scales
    gamma = min(GAMMA_HI, max(GAMMA_LO, _median(slopes)))
    b = _median([y - gamma * x for x, y in list(zip(lx, ly))[-FIT_WINDOW:]])
    return math.exp(b), gamma


def predict(obs, units):
    """Point prediction in seconds for `units`, or None while there is not enough history (the caller keeps its own formula)."""
    f = fit_power(obs)
    if f is None or units <= 0:
        return None
    a, g = f
    return a * units ** g


def overrun_margin(ratios, q=0.9, fallback=1.5):
    """(pad, basis, n): the multiplier that the next attempt's cost stays under with probability >= q (see the module docstring).

    basis is 'calibrated' when the history supports the claim, else 'fixed' (the old pad, raised to the worst overrun seen)."""
    rs = [float(r) for r in (ratios or []) if isinstance(r, (int, float)) and math.isfinite(r) and r > 0][-WINDOW:]
    n = len(rs)
    fb = max(PAD_LO, float(fallback))
    k = math.ceil((n + 1) * q)
    if n and k <= n:
        pad = sorted(rs)[k - 1]
        return min(PAD_HI, max(PAD_LO, pad)), "calibrated", n
    worst = max(rs) if rs else 0.0
    return min(PAD_HI, max(fb, worst)), "fixed", n


def push(history, item, cap=WINDOW):
    """Append to a bounded history list (returns a new list)."""
    return (list(history or []) + [item])[-cap:]


def affordable_units(cost_fn, cap_s, pad, hi, tol=1e-4):
    """The largest u in (0, hi] with cost_fn(u) * pad <= cap_s, by bisection (cost_fn must be increasing). 0.0 when even a tiny u does not fit."""
    if cost_fn(hi) * pad <= cap_s:
        return hi
    lo = 0.0
    top = hi
    for _ in range(60):
        mid = 0.5 * (lo + top)
        if mid <= 0:
            break
        if cost_fn(mid) * pad <= cap_s:
            lo = mid
        else:
            top = mid
        if top - lo <= tol * max(hi, 1e-9):
            break
    return lo
