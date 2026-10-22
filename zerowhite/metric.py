"""metric.py - one currency for 'how much did this attempt teach the system': bits.

The organism's measure of progress is `skill`, normalised PER TASK (accuracy against the majority class, R^2, return against a random policy,
1 - loss / ln(vocab)). That is right for comparing candidates on ONE task and only roughly comparable across tasks: +0.1 skill on a 3-row table and +0.1 on
a language model are not the same amount of learning. This module converts a skill into an information measure that is comparable, with the model of each
kind stated, so the scheduler can ask "which task gives the most bits per second" (`organism --metric bits`; the default stays `skill`).

  classification   a k-class classifier of accuracy a is treated as a symmetric channel: I(a) = log2 k + a log2 a + (1 - a) log2((1 - a) / (k - 1)) bits per example
                   (the mutual information when errors are spread evenly; a lower bound otherwise, by Fano). Bits of the task = n * (I(a) - I(majority-class rate)),
                   n = validation examples.
  regression       Gaussian channel: 0.5 * log2(1 / (1 - R^2)) bits per sample, times n validation samples.
  lm (chat)        (ln(vocab) - nll) / ln 2 bits per token, i.e. compression below a uniform guess, times the token count the run reports.
                   ASSUMED: the count used is the training tokens (the trainer does not report the validation token count), so chat bits are an order of magnitude
                   statement, not a measurement.
  env / other      no information-theoretic reading exists for a return. They are converted at `bits_per_skill` (default 50 bits per unit of skill) and flagged
                   'assumed'. That exchange rate is a judgement, not a measurement: it is how the scheduler is told how much a control task is worth next to a table.

All functions are pure. Bits are monotone in skill, so ordering candidates on ONE task is unchanged by the conversion.
"""
from __future__ import annotations

import math

LN2 = math.log(2.0)


def _xlog2x(p: float) -> float:
    return 0.0 if p <= 0.0 else p * math.log2(p)


def symmetric_channel_bits(acc: float, k: int) -> float:
    """Mutual information (bits per example) of a k-ary symmetric channel with accuracy `acc`; 0 at chance (1/k)."""
    k = max(2, int(k))
    a = min(1.0, max(0.0, float(acc)))
    if a <= 1.0 / k:
        return 0.0
    return math.log2(k) + _xlog2x(a) + (0.0 if a >= 1.0 else (1 - a) * math.log2((1 - a) / (k - 1)))


def classification_bits(skill: float, k: int, chance: float, n: float) -> float:
    s = min(1.0, max(0.0, float(skill)))
    c = min(0.999999, max(1.0 / max(2, int(k)), float(chance)))
    acc = c + s * (1.0 - c)
    return float(n) * max(0.0, symmetric_channel_bits(acc, k) - symmetric_channel_bits(c, k))


def regression_bits(r2: float, n: float) -> float:
    r = min(0.999999, max(0.0, float(r2)))
    return float(n) * 0.5 * math.log2(1.0 / (1.0 - r))


def lm_bits(skill: float, vocab: int, n_tokens: float) -> float:
    s = min(1.0, max(0.0, float(skill)))
    return float(n_tokens) * s * math.log(max(2, int(vocab))) / LN2


def task_meta(task) -> dict:
    """What the conversion needs to know about a task object (any of the forge task classes). Unknown things are left out, not guessed."""
    kind = getattr(task, "kind", None)
    meta = dict(kind=kind if kind in ("classification", "regression") else "other")
    if meta["kind"] == "classification":
        meta.update(k=int(task.n_classes), chance=float(task.chance()), n=int(len(task.Xval)))
    elif meta["kind"] == "regression":
        meta.update(n=int(len(task.Xval)))
    return meta


def skill_to_bits(meta: dict, skill: float, bits_per_skill: float = 50.0):
    """(bits, basis): basis is 'measured' for the kinds with an information model, 'assumed' for the exchange-rate fallback."""
    meta = meta or {}
    kind = meta.get("kind")
    if kind == "classification" and meta.get("n"):
        return classification_bits(skill, meta.get("k", 2), meta.get("chance", 0.5), meta["n"]), "measured"
    if kind == "regression" and meta.get("n"):
        return regression_bits(skill, meta["n"]), "measured"
    if kind == "lm" and meta.get("vocab") and meta.get("n"):
        return lm_bits(skill, meta["vocab"], meta["n"]), "assumed"
    return max(0.0, min(1.0, float(skill))) * float(bits_per_skill), "assumed"
