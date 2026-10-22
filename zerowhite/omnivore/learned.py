"""learned.py - a small learned model that decides which delimiter lays out a file's columns, where the hand-written rule is unsure or silent.

`tables.find_delimiter` picks the delimiter that makes rows the same width, scored by `consistency^2 * log2(width)`. That score is a guess, and it has known blind
spots: a space-delimited file with decimal commas splits "consistently" on the commas. This module LEARNS the scoring function. For each candidate delimiter it
computes the same dozen pieces of evidence (consistency, width, how numeric the resulting cells look, how many contain spaces, empty cells, decimal-comma density,
the old rule's score itself, ...) and a tiny network (shared weights, ~600 parameters) is trained to give the true delimiter the top score. Because every
candidate is described the same way, it learns what columns look like, not which character is usual.

HOW IT IS USED (devour, `--learned-sniffer`): the rules run first, unchanged.
  rescue (default)   when the rules find no table, the ranker's pick is offered and the file is accepted as a table only if it then has at least one numeric column.
  arbitrate          the ranker may also replace the rules' delimiter, but only if the resulting table has strictly MORE numeric columns than the rules' table.
  off                rules only.

WHAT WAS TRIED FIRST, AND WHY IT IS NOT HERE. A byte-level network (byte / byte-pair histograms and line statistics -> delimiter, header, decimal mark, encoding)
scored ~100% on held-out files from the training generator and 12 of 16 on files written independently of it (the rules: 14 of 16), with probability 1.0 on its wrong
answers. A model that is confident and wrong off-distribution must not overrule rules, so it was removed rather than shipped. The ranker, trained on the SAME
generator, scores 16 of 16 on those independent files, and the tests keep that comparison.

HONEST SCOPE: 16 independent files is a small sample, and I wrote both the generator and those files. This is evidence that the ranker generalises better than a model of
raw bytes, not that it beats the rules on your data. It chooses among five delimiters; it does not repair corrupted files (ragged rows, unbalanced quotes, cut-off tails,
preambles stay rule-based) and it does not decide headers, decimal marks or encodings (the rules do). Weights (`ranker.npz`, a few KB) ship with the package; retrain with
`python -m zerowhite.omnivore.learned train`.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

from . import synth

DELIMS = synth.DELIMS
_DEC_COMMA = re.compile(r"\d,\d")


# ------------------------------------------------------------------------------------------------ the delimiter ranker
RANK_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ranker.npz")
_NUMCELL = re.compile(r"^\s*[-+]?(\d+([.,]\d+)?|[.,]\d+)([eE][-+]?\d+)?\s*%?\s*$")
N_RANK = len(DELIMS) + 11


def delim_features(text: str) -> np.ndarray:
    """(len(DELIMS), N_RANK): one row of evidence per candidate delimiter. Every row is computed the same way for every candidate, so the model learns a
    SCORING FUNCTION over candidates (what makes a split look like columns), not the identity of a character."""
    lines = [ln for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()][:200]
    body = lines[1:] if len(lines) > 3 else lines
    comma_digit = min(1.0, len(_DEC_COMMA.findall(text[:20000])) / max(1.0, len(lines)))
    rows = []
    for di, d in enumerate(DELIMS):
        counts = [ln.count(d) for ln in lines] or [0]
        c = np.array(counts)
        modal = int(np.bincount(c).argmax())
        cons = float(np.mean(c == modal))
        width = modal + 1
        cells = [x for ln in body for x in ln.split(d)] if modal > 0 else []
        ne = [x for x in cells if x.strip()]
        num = float(np.mean([bool(_NUMCELL.match(x)) for x in ne])) if ne else 0.0
        words = float(np.mean([(" " in x.strip()) for x in ne])) if (ne and d != " ") else 0.0
        mean_len = min(1.0, float(np.mean([len(x) for x in ne])) / 30.0) if ne else 0.0
        empty = float(np.mean([not x.strip() for x in cells])) if cells else 0.0
        one_hot = [1.0 if i == di else 0.0 for i in range(len(DELIMS))]
        rows.append(one_hot + [cons, np.log2(width) / 4.0, float(width >= 2), cons * cons * np.log2(max(width, 1)) / 4.0, num, words, mean_len, empty,
                               comma_digit if d == "," else 0.0, float(c.std() / (c.mean() + 1e-9)) if c.mean() > 0 else 1.0, float(modal == 0)])
    return np.array(rows, dtype=float)


class Ranker:
    def __init__(self, W1, b1, w2, b2, mu, sd):
        self.W1, self.b1, self.w2, self.b2, self.mu, self.sd = W1, b1, w2, b2, mu, sd

    def scores(self, F):
        H = np.maximum(0.0, ((F - self.mu) / self.sd) @ self.W1 + self.b1)
        return H @ self.w2 + self.b2

    def rank(self, text: str):
        sc = self.scores(delim_features(text))
        e = np.exp(sc - sc.max())
        p = e / e.sum()
        return DELIMS[int(p.argmax())], float(p.max())

    def save(self, path):
        np.savez_compressed(path, **{k: np.asarray(v, dtype=np.float32) for k, v in dict(W1=self.W1, b1=self.b1, w2=self.w2, b2=self.b2, mu=self.mu, sd=self.sd).items()})

    @classmethod
    def load(cls, path=RANK_WEIGHTS):
        z = np.load(path)
        return cls(z["W1"], z["b1"], z["w2"], z["b2"], z["mu"], z["sd"])


def train_ranker(n: int = 20000, seed: int = 0, hidden: int = 24, epochs: int = 25, lr: float = 3e-3, verbose: bool = True) -> Ranker:
    data = synth.dataset(n, seed + 11)
    Fs = np.stack([delim_features(l["text"]) for _, l in data])                   # (n, 5, F)
    y = np.array([DELIMS.index(l["delim"]) for _, l in data])
    flat = Fs.reshape(-1, N_RANK)
    mu, sd = flat.mean(axis=0), flat.std(axis=0) + 1e-6
    Z = (Fs - mu) / sd
    rng = np.random.default_rng(seed + 2)
    W1, b1 = rng.normal(0, np.sqrt(2.0 / N_RANK), (N_RANK, hidden)), np.zeros(hidden)
    w2, b2 = rng.normal(0, np.sqrt(1.0 / hidden), hidden), np.zeros(1)
    params = [W1, b1, w2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    step = 0
    for ep in range(epochs):
        order = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, 128):
            idx = order[i:i + 128]
            Zb = Z[idx]
            Hb = np.maximum(0.0, Zb @ W1 + b1)
            sc = Hb @ w2 + b2[0]
            sc -= sc.max(axis=1, keepdims=True)
            p = np.exp(sc)
            p /= p.sum(axis=1, keepdims=True)
            tot += float(-np.log(p[np.arange(len(idx)), y[idx]] + 1e-12).sum())
            d = p.copy()
            d[np.arange(len(idx)), y[idx]] -= 1.0
            d /= len(idx)
            gw2 = np.einsum("bc,bch->h", d, Hb)
            gb2 = np.array([d.sum()])
            dH = d[:, :, None] * w2[None, None, :] * (Hb > 0)
            gW1 = np.einsum("bcf,bch->fh", Zb, dH)
            gb1 = dH.sum(axis=(0, 1))
            step += 1
            for j, (p_, g) in enumerate(zip(params, [gW1, gb1, gw2, gb2])):
                m[j] = 0.9 * m[j] + 0.1 * g
                v[j] = 0.999 * v[j] + 0.001 * g * g
                p_ -= lr * (m[j] / (1 - 0.9 ** step)) / (np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
        if verbose and (ep % 5 == 4 or ep == epochs - 1):
            print(f"ranker epoch {ep + 1}/{epochs}  loss {tot / n:.4f}", file=sys.stderr)
    return Ranker(W1, b1, w2, b2, mu, sd)


_rcache = {}


def get_ranker():
    if "r" not in _rcache:
        try:
            _rcache["r"] = Ranker.load()
        except (OSError, KeyError, ValueError):
            _rcache["r"] = None
    return _rcache["r"]


def hint(data: bytes, min_p: float = 0.8):
    """{'delim': ...} when the ranker is at least `min_p` sure which delimiter lays out the file's columns, else None."""
    r = get_ranker()
    if r is None:
        return None
    from . import sniff
    dec = sniff.decode(data)
    if dec is None:
        return None
    d, p = r.rank(dec.text)
    return {"delim": d} if p >= min_p else None


def parse_with_hint(text: str, data: bytes, mode: str = "rescue", max_rows: int = 50_000):
    """The table devour makes from a text file: the rules first, then (mode 'rescue' / 'arbitrate') the ranker's delimiter, accepted only when it is believable."""
    from . import tables
    tab = tables.parse_delimited(text, max_rows)
    if mode == "off":
        return tab
    arb = mode == "arbitrate"
    h = hint(data, min_p=0.97 if arb else 0.8) if (tab is None or arb) else None
    if h and (tab is None or h["delim"] != tab.delimiter):
        tab2 = tables.parse_delimited(text, max_rows, hint=dict(h, arbitrate=arb))
        if tab2 is not None and tables.table_quality(tab2) > tables.table_quality(tab):        # strictly more table-like than what the rules made (prose scores 0)
            return tab2
    return tab


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        rk = train_ranker(n=int(sys.argv[2]) if len(sys.argv) > 2 else 20000)
        rk.save(RANK_WEIGHTS)
        print("saved", RANK_WEIGHTS)
