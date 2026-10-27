"""preference.py - learning from "this answer is better than that one" (DPO), shared by offline training and online learning.

For a pair (prompt, chosen, rejected) and a frozen reference model, with  z = beta * [(lp_c - ref_c) - (lp_r - ref_r)]
(lp = summed log-probability of the answer tokens), the loss is  -log sigmoid(z).  Its gradient is
    -beta * sigmoid(-z) * (grad lp_c - grad lp_r)
which is exactly what the model's weighted cross-entropy head computes when the chosen answer's tokens get weight
+beta*sigmoid(-z) and the rejected answer's tokens weight -beta*sigmoid(-z). No second backward pass is needed.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .model import GPT

Pair = Tuple[np.ndarray, np.ndarray, np.ndarray, float]          # (prompt_ids, chosen_ids, rejected_ids, weight)


def _row(prompt: np.ndarray, answer: np.ndarray, T: int):
    ids = np.concatenate((prompt, answer)).astype(np.int64)
    if len(ids) > T + 1:                                           # keep the answer, drop the oldest prompt tokens
        cut = len(ids) - (T + 1)
        ids = ids[cut:]
        n_ans = min(len(answer), len(ids) - 1)
    else:
        n_ans = len(answer)
    return ids, n_ans


def make_batch(seqs: Sequence[Tuple[np.ndarray, np.ndarray]], coefs: Sequence[float], T: int, pad: int) -> dict:
    """One row per (prompt, answer); weights `coef` on the answer tokens only."""
    rows = [_row(p, a, T) for p, a in seqs]
    L = max(len(r[0]) - 1 for r in rows)
    B = len(rows)
    ids = np.full((B, L), pad, np.int64)
    tgt = np.zeros((B, L), np.int64)
    w = np.zeros((B, L), np.float32)
    pos = np.zeros((B, L), np.int64)
    seg = np.zeros((B, L), np.int64)
    for b, (r, n_ans) in enumerate(rows):
        n = len(r) - 1
        ids[b, :n] = r[:-1]
        tgt[b, :n] = r[1:]
        w[b, n - n_ans:n] = coefs[b]
        pos[b, :n] = np.arange(n)
        seg[b, :n] = 1
    return dict(ids=ids, tgt=tgt, w=w, pos=pos, seg=seg)


def sequence_logprobs(model: GPT, pairs: Sequence[Pair], T: int, pad: int) -> Tuple[np.ndarray, np.ndarray]:
    """Summed answer log-probability of every chosen and every rejected answer under `model` (no gradients)."""
    out = []
    for which in (1, 2):
        b = make_batch([(p[0], p[which]) for p in pairs], [1.0] * len(pairs), T, pad)
        out.append(model.token_logprobs(b).sum(axis=1))
    return out[0], out[1]


def dpo_grads(model: GPT, ref_c: np.ndarray, ref_r: np.ndarray, pairs: Sequence[Pair], beta: float, T: int, pad: int,
              pair_weight: float = 1.0):
    """Gradients of the mean DPO loss over `pairs` w.r.t. `model` (the policy).
    Returns (grads, stats) with stats = loss, accuracy (chosen margin > rejected), margin."""
    lp_c, lp_r = sequence_logprobs(model, pairs, T, pad)
    z = beta * ((lp_c - ref_c) - (lp_r - ref_r))
    sig_neg = 1.0 / (1.0 + np.exp(np.clip(z, -30, 30)))                        # sigmoid(-z)
    pw = np.array([p[3] for p in pairs], dtype=np.float64) * pair_weight
    coef = beta * sig_neg * pw / max(1, len(pairs))
    seqs = [(p[0], p[1]) for p in pairs] + [(p[0], p[2]) for p in pairs]
    coefs = list(coef) + list(-coef)
    batch = make_batch(seqs, coefs, T, pad)
    _, grads, _ = model.loss_and_grads(batch)
    loss = float(np.mean(np.logaddexp(0.0, -z)))                              # -log sigmoid(z)
    return grads, dict(loss=loss, accuracy=float(np.mean(z > 0)), margin=float(np.mean(z)))
