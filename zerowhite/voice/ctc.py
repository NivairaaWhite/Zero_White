"""ctc.py - the built-in speech recogniser: a dilated-convolution acoustic model with CTC loss, hand-written forward and backward
passes in numpy (no ML library).

    log-mel frames (T, n_mels)
      -> conv k, stride 2, ReLU                     (50 frames/s at a 10 ms hop)
      -> L x [ x + ReLU(conv k, dilation d_i) ]     (residual; dilations cycle, so context grows quickly)
      -> linear -> characters + blank
and Connectionist Temporal Classification to learn the alignment, so training data needs only audio + transcript.
Decoding is greedy best-path. The alphabet is whatever characters occur in your transcripts: any language.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

NEG = -1e30


class CTCConfig:
    def __init__(self, n_in: int, hidden: int, layers: int, vocab: int, kernel: int = 5, dilations: Sequence[int] = (1, 2, 4, 8)):
        self.n_in, self.hidden, self.layers, self.vocab = int(n_in), int(hidden), int(layers), int(vocab)
        self.kernel, self.dilations = int(kernel), [int(d) for d in dilations] or [1]

    def to_dict(self) -> dict:
        return dict(n_in=self.n_in, hidden=self.hidden, layers=self.layers, vocab=self.vocab, kernel=self.kernel, dilations=self.dilations)

    @classmethod
    def from_dict(cls, d: dict) -> "CTCConfig":
        return cls(d["n_in"], d["hidden"], d["layers"], d["vocab"], d.get("kernel", 5), d.get("dilations", [1, 2, 4, 8]))

    def dilation(self, i: int) -> int:
        return self.dilations[i % len(self.dilations)]


def _conv_fwd(x: np.ndarray, W: np.ndarray, b: np.ndarray, k: int, d: int, s: int) -> Tuple[np.ndarray, tuple]:
    """x (B, T, C) -> (B, ceil(T/s), Cout), 'same' padding. Returns the output and what backward needs."""
    B, T, C = x.shape
    pad = (k - 1) * d // 2
    pad_r = (k - 1) * d - pad
    xp = np.pad(x, ((0, 0), (pad, pad_r), (0, 0)))
    To = -(-T // s)
    cols = np.concatenate([xp[:, j * d: j * d + (To - 1) * s + 1: s, :] for j in range(k)], axis=2)       # (B, To, k*C)
    y = cols.reshape(B * To, k * C) @ W + b
    return y.reshape(B, To, -1), (cols, xp.shape, T)


def _conv_bwd(dy: np.ndarray, W: np.ndarray, cache: tuple, k: int, d: int, s: int, C: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols, xp_shape, T = cache
    B, To, Co = dy.shape
    dyf = dy.reshape(B * To, Co)
    dW = cols.reshape(B * To, -1).T @ dyf
    db = dyf.sum(axis=0)
    dcols = (dyf @ W.T).reshape(B, To, k, C)
    dxp = np.zeros(xp_shape, dtype=dy.dtype)
    for j in range(k):
        dxp[:, j * d: j * d + (To - 1) * s + 1: s, :] += dcols[:, :, j, :]
    pad = (k - 1) * d // 2
    return dxp[:, pad: pad + T, :], dW, db


class CTCModel:
    def __init__(self, cfg: CTCConfig, rng: Optional[np.random.Generator] = None, params: Optional[Dict[str, np.ndarray]] = None):
        self.cfg = cfg
        if params is not None:
            self.p = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
        else:
            rng = rng or np.random.default_rng(0)
            k, H = cfg.kernel, cfg.hidden
            self.p = {"in.W": (rng.standard_normal((k * cfg.n_in, H)) * math.sqrt(2.0 / (k * cfg.n_in))).astype(np.float32),
                      "in.b": np.zeros(H, np.float32)}
            for i in range(cfg.layers):
                self.p[f"l{i}.W"] = (rng.standard_normal((k * H, H)) * math.sqrt(2.0 / (k * H)) * 0.5).astype(np.float32)
                self.p[f"l{i}.b"] = np.zeros(H, np.float32)
            self.p["out.W"] = (rng.standard_normal((H, cfg.vocab)) * math.sqrt(1.0 / H)).astype(np.float32)
            self.p["out.b"] = np.zeros(cfg.vocab, np.float32)

    def num_params(self) -> int:
        return int(sum(v.size for v in self.p.values()))

    @staticmethod
    def out_lengths(lens: np.ndarray) -> np.ndarray:
        return -(-np.asarray(lens) // 2)

    # ---- forward / backward ---------------------------------------------------------------------------------------
    def forward(self, X: np.ndarray, lens: np.ndarray, train: bool = False):
        """X (B, T, n_in) float32, lens (B,) valid frames. Returns logits (B, T', V), output lengths, cache."""
        c, p = self.cfg, self.p
        lo = self.out_lengths(lens)
        X = X * (np.arange(X.shape[1])[None, :] < np.asarray(lens)[:, None])[:, :, None]       # padding is silence, whatever the caller put there
        x, cache0 = _conv_fwd(X, p["in.W"], p["in.b"], c.kernel, 1, 2)
        mask = (np.arange(x.shape[1])[None, :] < lo[:, None]).astype(np.float32)[:, :, None]
        pre0 = x
        x = np.maximum(x, 0.0) * mask
        caches = []
        for i in range(c.layers):
            h, ci = _conv_fwd(x, p[f"l{i}.W"], p[f"l{i}.b"], c.kernel, c.dilation(i), 1)
            r = np.maximum(h, 0.0) * mask
            caches.append((ci, h))
            x = x + r
        logits = x @ p["out.W"] + p["out.b"]
        return logits, lo, (X.shape, cache0, pre0, mask, caches, x)

    def backward(self, dlogits: np.ndarray, cache) -> Dict[str, np.ndarray]:
        c, p = self.cfg, self.p
        xshape, cache0, pre0, mask, caches, xo = cache
        g: Dict[str, np.ndarray] = {}
        B, T2, H = xo.shape
        g["out.W"] = xo.reshape(-1, H).T @ dlogits.reshape(-1, dlogits.shape[-1])
        g["out.b"] = dlogits.sum(axis=(0, 1))
        dx = dlogits @ p["out.W"].T
        for i in reversed(range(c.layers)):
            ci, h = caches[i]
            dh = dx * mask * (h > 0)
            dxi, dW, db = _conv_bwd(dh, p[f"l{i}.W"], ci, c.kernel, c.dilation(i), 1, H)
            g[f"l{i}.W"], g[f"l{i}.b"] = dW, db
            dx = dx + dxi
        d0 = dx * mask * (pre0 > 0)
        _, dW, db = _conv_bwd(d0, p["in.W"], cache0, c.kernel, 1, 2, c.n_in)
        g["in.W"], g["in.b"] = dW, db
        return g


# ---------------------------------------------------------------------------------------------------------------------
# CTC loss
# ---------------------------------------------------------------------------------------------------------------------
def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def ctc_loss_grad(logp: np.ndarray, labels: Sequence[int], blank: int = 0) -> Tuple[float, Optional[np.ndarray]]:
    """Negative log-likelihood of `labels` given per-frame log-probabilities (T, V) and its gradient w.r.t. the *logits*
    (softmax - occupancy). Returns (inf, None) when the label sequence cannot fit in T frames."""
    T, V = logp.shape
    lab = list(labels)
    L = len(lab)
    repeats = sum(1 for i in range(1, L) if lab[i] == lab[i - 1])
    if T < L + repeats or L == 0:
        return float("inf"), None
    ext = np.full(2 * L + 1, blank, dtype=np.int64)
    ext[1::2] = lab
    S = len(ext)
    lp = logp[:, ext]                                                       # (T, S)
    skip = np.zeros(S, bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    alpha = np.full((T, S), NEG)
    alpha[0, 0] = lp[0, 0]
    if S > 1:
        alpha[0, 1] = lp[0, 1]
    for t in range(1, T):
        a = alpha[t - 1]
        c = np.logaddexp(a, np.concatenate([[NEG], a[:-1]]))
        if S > 2:
            c = np.logaddexp(c, np.where(skip, np.concatenate([[NEG, NEG], a[:-2]]), NEG))
        alpha[t] = c + lp[t]
    logP = np.logaddexp(alpha[T - 1, S - 1], alpha[T - 1, S - 2] if S > 1 else NEG)
    if not np.isfinite(logP) or logP < -1e20:
        return float("inf"), None
    skip_f = np.zeros(S, bool)
    skip_f[:-2] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    beta = np.full((T, S), NEG)
    beta[T - 1, S - 1] = lp[T - 1, S - 1]
    if S > 1:
        beta[T - 1, S - 2] = lp[T - 1, S - 2]
    for t in range(T - 2, -1, -1):
        b = beta[t + 1]
        c = np.logaddexp(b, np.concatenate([b[1:], [NEG]]))
        if S > 2:
            c = np.logaddexp(c, np.where(skip_f, np.concatenate([b[2:], [NEG, NEG]]), NEG))
        beta[t] = c + lp[t]
    ab = alpha + beta - logP - lp                                           # log of (alpha*beta / (P * y)); <= 0 for valid paths
    occ = np.zeros((V, T))
    np.add.at(occ, ext, np.exp(np.minimum(ab, 0.0)).T)
    grad = np.exp(logp) - occ.T
    return float(-logP), grad


def batch_loss_grad(logits: np.ndarray, out_lens: np.ndarray, targets: List[List[int]], blank: int = 0) -> Tuple[float, np.ndarray, int, int]:
    """Summed CTC loss over the batch and d(loss)/d(logits). Returns (loss_sum, dlogits, n_ok, n_skipped).
    Utterances whose transcript cannot be aligned (too long for the audio) are skipped, never fatal."""
    B = logits.shape[0]
    dl = np.zeros_like(logits)
    total, ok, skipped = 0.0, 0, 0
    for b in range(B):
        T = int(out_lens[b])
        lp = log_softmax(logits[b, :T].astype(np.float64))
        loss, g = ctc_loss_grad(lp, targets[b], blank)
        if g is None or not math.isfinite(loss):
            skipped += 1
            continue
        total += loss
        dl[b, :T] = g
        ok += 1
    return total, dl, ok, skipped


# ---------------------------------------------------------------------------------------------------------------------
# decoding / metrics
# ---------------------------------------------------------------------------------------------------------------------
def greedy_decode(logits: np.ndarray, blank: int = 0) -> List[int]:
    """Best path: argmax per frame, collapse repeats, drop blanks. logits (T, V)."""
    best = logits.argmax(axis=-1)
    out, prev = [], -1
    for t in best:
        if t != prev and t != blank:
            out.append(int(t))
        prev = int(t)
    return out


def edit_distance(a: Sequence, b: Sequence) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def error_rate(refs: Sequence[str], hyps: Sequence[str], words: bool = False) -> float:
    """Character (or word) error rate over a set: total edit distance / total reference length."""
    err = tot = 0
    for r, h in zip(refs, hyps):
        rr, hh = (r.split(), h.split()) if words else (list(r), list(h))
        err += edit_distance(rr, hh)
        tot += len(rr)
    return err / max(1, tot)
