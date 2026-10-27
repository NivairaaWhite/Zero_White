"""episodic.py - a long-term, vectorised episodic memory: the system remembers which agents won on which tasks, and starts a new task from the nearest memories instead of from nothing.

Until 6.4 the only memory of past work was the zoo (one champion per task, found by NAME) and the learned prior (per-family running means). A task the system had never seen
under that name started from uniform-random genomes and the trivial program, however much it resembled an earlier one. This module adds the missing piece:

  RECORDS    one per task profile, per champion (family, genome, skill, program source), per failed candidate and per remembered observation window: `<workspace>/forge/episodic/`.
  ENCODER    a small two-layer numpy network trained with a CONTRASTIVE loss (multi-positive InfoNCE, the SupCon form): two noisy views of the same record, and records of the same
             task (or of champions of the same family on same-sized tasks), are pulled together; everything else in the batch is pushed apart. Its input is a hashed bag of words and
             character trigrams of the record's description plus a vector of numbers (log size, kind, simple data statistics, a linear-probe skill). Gradients are written out by hand
             (the engine has no transpose) and checked against finite differences in the tests.
  RECALL     cosine similarity of the query task against every stored champion vector: the k nearest with their genomes and (for program agents) their SOURCE CODE.
  SPLICE     for program agents the recalled programs are re-indexed to the new observation width, compiled under the same safety gate as any evolved program, and CROSSED OVER
             pairwise (astmut.crossover); the results join the evolver's starting population. For the other families the recalled genomes (and a blend of two of them) become
             candidates of the race, replacing some of the uniform-random proposals.

What it does not do: it does not change a task's champion rules (a recalled candidate must still win the race on this task's validation data), and it cannot know a task
is similar in a way the profile does not show (a profile is the data's shape, its simple statistics and its name). The encoder is only as good as the number of tasks it has seen;
with few records the untrained encoder is a random projection of the same features, which is still a usable similarity (the tests compare both). Records are summaries, not data.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import time
import zlib

import numpy as np

from .. import fileio
from ..selfmod import astmut as A

BUCKETS = 256                       # hashed text features
NUM = 24                            # numeric profile features
F_IN = BUCKETS + NUM
KINDS = ("classification", "regression", "env", "text", "image_gen")
MAX_RECORDS = 3000
SRC_CHARS = 4000


# ======================================================================================================================================= features
def _h(token: str) -> int:
    return zlib.crc32(token.encode("utf-8"))


def hash_text(text: str, buckets: int = BUCKETS) -> np.ndarray:
    """Signed feature hashing of word unigrams, word bigrams and character trigrams, sublinear term frequency, unit length."""
    v = np.zeros(buckets)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]", text.lower())
    grams = list(words) + [a + "_" + b for a, b in zip(words, words[1:])]
    s = " ".join(words)
    grams += [s[i:i + 3] for i in range(max(0, len(s) - 2))]
    counts: dict = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    for g, c in counts.items():
        hv = _h(g)
        v[hv % buckets] += (1.0 if (hv >> 16) & 1 else -1.0) * (1.0 + math.log(c))
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def array_stats(X, limit: int = 600) -> np.ndarray:
    """8 numbers describing a data matrix or a window of observations: scale, spread, skew, tails, sparsity, column correlation, discreteness, and how spread the rows are."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    X = np.nan_to_num(X.reshape(len(X), -1)[:limit], nan=0.0, posinf=0.0, neginf=0.0)
    if X.size == 0 or len(X) < 2:
        return np.zeros(8)
    sd = X.std(axis=0)
    mu = X.mean(axis=0)
    z = (X - mu) / np.where(sd > 1e-9, sd, 1.0)
    skew = float(np.mean(np.abs(np.mean(z ** 3, axis=0))))
    kurt = float(np.mean(np.abs(np.mean(z ** 4, axis=0) - 3.0)))
    zero = float(np.mean(X == 0.0))
    cols = X[:, :20]
    cor = 0.0
    if cols.shape[1] > 1:
        c = np.corrcoef(cols.T)
        c = np.nan_to_num(c)
        cor = float((np.abs(c).sum() - cols.shape[1]) / max(1, cols.shape[1] * (cols.shape[1] - 1)))
    discrete = float(np.mean([len(np.unique(X[:, j])) <= 6 for j in range(min(X.shape[1], 20))]))
    return np.array([math.tanh(float(np.mean(np.abs(mu)))), math.tanh(float(sd.mean())), math.tanh(skew), math.tanh(kurt / 4.0), zero, cor, discrete,
                     math.tanh(float(np.mean(np.abs(X - np.median(X, axis=0)))))])


def linear_probe(task) -> float:
    """How far a plain linear model gets on the task, in [0, 1] (a cheap hint of whether a nonlinear agent is needed); 0 when it cannot be measured."""
    try:
        if getattr(task, "kind", "") not in ("classification", "regression") or not hasattr(task, "Xtr"):
            return 0.0
        Xtr, ytr = np.asarray(task.Xtr, float)[:600], np.asarray(task.ytr)[:600]
        Xva, yva = np.asarray(task.Xval, float)[:400], np.asarray(task.yval)[:400]
        A_ = np.c_[Xtr, np.ones(len(Xtr))]
        B_ = np.c_[Xva, np.ones(len(Xva))]
        reg = 1e-2 * np.eye(A_.shape[1])
        if task.kind == "classification":
            Y = np.eye(int(task.n_classes))[ytr.astype(int)]
            W = np.linalg.solve(A_.T @ A_ + reg, A_.T @ Y)
            acc = float(np.mean((B_ @ W).argmax(1) == yva))
            ch = float(task.chance())
            return float(np.clip((acc - ch) / max(1e-9, 1 - ch), 0.0, 1.0))
        w = np.linalg.solve(A_.T @ A_ + reg, A_.T @ ytr.astype(float))
        p = B_ @ w
        return float(np.clip(1.0 - np.mean((p - yva) ** 2) / (np.var(yva) + 1e-12), 0.0, 1.0))
    except Exception:                                                                      # noqa: BLE001 - a profile feature must never stop a solve
        return 0.0


def _bucket(v: float) -> int:
    return int(round(math.log2(max(1.0, float(v)))))


def probe_obs(task, n: int = 120, seed: int = 1) -> np.ndarray:
    """Observations a random policy sees in an environment task: (n, obs_dim). Used for the profile and for the remembered state windows."""
    rng = np.random.default_rng(seed)
    obs = []
    o = task.env.reset(rng)
    for _ in range(n):
        obs.append(np.asarray(o, float).ravel())
        o, _, done, _ = task.env.step(int(rng.integers(task.n_actions)))
        if done:
            o = task.env.reset(rng)
    return np.array(obs)


def task_profile(task):
    """(text, numeric vector) for a task: what the memory embeds. Cheap (a few hundred rows at most) and side-effect free."""
    sig = task.signature() if hasattr(task, "signature") else {}
    kind = getattr(task, "kind", sig.get("kind", "?"))
    name = str(getattr(task, "name", "task"))
    dim, cls, n = int(sig.get("dim", 0)), int(sig.get("classes", 0)), int(sig.get("n", 0))
    stats = np.zeros(8)
    probe = 0.0
    try:
        if kind in ("classification", "regression"):
            stats = array_stats(task.Xtr)
            probe = linear_probe(task)
        elif kind == "env":
            stats = array_stats(probe_obs(task))
        elif kind == "text":
            codes = np.frombuffer(str(getattr(task, "text", ""))[:20000].encode("utf-8", "ignore"), np.uint8)
            if len(codes) > 8:
                p = np.bincount(codes, minlength=256) / len(codes)
                ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
                stats[0], stats[1] = ent / 8.0, math.tanh(len(codes) / 20000.0)
        elif kind == "image_gen" and hasattr(task, "sup"):
            stats = array_stats(task.sup.Xtr)
    except Exception:                                                                      # noqa: BLE001
        pass
    words = " ".join(re.split(r"[^A-Za-z0-9]+", name))
    text = f"task {words} kind_{kind} dim_b{_bucket(dim)} cls_b{_bucket(cls)} n_b{_bucket(n)} image_{bool(sig.get('image'))} probe_{int(probe * 4)} " \
           f"sparse_{int(stats[4] * 4)} corr_{int(stats[5] * 4)} discrete_{int(stats[6] * 2)}"
    num = np.zeros(NUM)
    num[0], num[1], num[2] = math.log1p(dim) / 8.0, math.log1p(cls) / 6.0, math.log1p(n) / 12.0
    if kind in KINDS:
        num[3 + KINDS.index(kind)] = 1.0
    num[8] = float(bool(sig.get("image")))
    num[9:17] = stats
    num[17] = probe
    return text, num


def featurize(text: str, num: np.ndarray) -> np.ndarray:
    return np.concatenate([hash_text(text), np.asarray(num, float)])


# ======================================================================================================================================= the contrastive encoder
class Encoder:
    """x (F) -> tanh(x W1 + b1) -> z = h W2 + b2 -> z / |z|. Trained with multi-positive InfoNCE; every gradient is explicit."""

    def __init__(self, f_in: int = F_IN, hidden: int = 96, out: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.p = dict(W1=rng.standard_normal((f_in, hidden)) / math.sqrt(f_in) * 2.0, b1=np.zeros(hidden), W2=rng.standard_normal((hidden, out)) / math.sqrt(hidden), b2=np.zeros(out))
        self.trained_steps = 0

    def forward(self, X, keep=False):
        p = self.p
        h = np.tanh(X @ p["W1"] + p["b1"])
        z = h @ p["W2"] + p["b2"]
        nz = np.sqrt((z ** 2).sum(1, keepdims=True) + 1e-9)
        zn = z / nz
        return (zn, (X, h, z, nz, zn)) if keep else zn

    def embed(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return self.forward(X)

    @staticmethod
    def loss_and_dS(S, pos, tau):
        """Multi-positive InfoNCE over the rows of the similarity matrix S (B x B): L_i = -log( sum_{j in P(i)} p_ij ), p = softmax(S / tau). Returns (mean loss, dL/dS)."""
        B = S.shape[0]
        lg = S / tau
        lg = lg - lg.max(1, keepdims=True)
        e = np.exp(lg)
        P = e / e.sum(1, keepdims=True)
        sp = (P * pos).sum(1, keepdims=True) + 1e-12
        loss = float(-np.log(sp).mean())
        dS = (P - pos * P / sp) / (tau * B)
        return loss, dS

    def loss_grads(self, XA, XB, pos, tau):
        zA, cA = self.forward(XA, keep=True)
        zB, cB = self.forward(XB, keep=True)
        S = zA @ zB.T
        l1, d1 = self.loss_and_dS(S, pos, tau)
        l2, d2 = self.loss_and_dS(S.T, pos.T, tau)
        dS = 0.5 * (d1 + d2.T)
        dzA, dzB = dS @ zB, dS.T @ zA
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        for dzn, (X, h, z, nz, zn) in ((dzA, cA), (dzB, cB)):
            dz = (dzn - zn * (zn * dzn).sum(1, keepdims=True)) / nz
            g["W2"] += h.T @ dz
            g["b2"] += dz.sum(0)
            dh = (dz @ self.p["W2"].T) * (1 - h ** 2)
            g["W1"] += X.T @ dh
            g["b1"] += dh.sum(0)
        return 0.5 * (l1 + l2), g

    def fit(self, X, pos_fn, steps: int = 200, batch: int = 48, tau: float = 0.2, lr: float = 3e-3, rng=None, noise: float = 0.05, drop: float = 0.15):
        """`pos_fn(idx) -> bool matrix` says which batch members are positives of each other (the diagonal is always one). Adam; returns the mean loss of the first and of the last tenth of the steps."""
        rng = rng or np.random.default_rng(0)
        X = np.asarray(X, float)
        n = len(X)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(x) for k, x in self.p.items()}
        losses = []
        for t in range(1, steps + 1):
            idx = rng.choice(n, size=min(batch, n), replace=False)
            base = X[idx]

            def view():
                a = base.copy()
                a[:, :BUCKETS] *= (rng.random((len(a), BUCKETS)) > drop)
                a[:, BUCKETS:] += rng.normal(0.0, noise, size=a[:, BUCKETS:].shape)
                return a
            pos = np.asarray(pos_fn(idx), dtype=float)
            np.fill_diagonal(pos, 1.0)
            loss, g = self.loss_grads(view(), view(), pos, tau)
            if not np.isfinite(loss):
                continue
            losses.append(loss)
            for k in self.p:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                self.p[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            self.trained_steps += 1
        if not losses:
            return None, None
        k = max(1, len(losses) // 10)                                                      # averages over the first and last tenth of the steps: single batches are noisy
        return float(np.mean(losses[:k])), float(np.mean(losses[-k:]))

    def to_arrays(self):
        return dict(self.p, steps=np.array([self.trained_steps]))

    @classmethod
    def from_arrays(cls, a):
        e = cls(a["W1"].shape[0], a["W1"].shape[1], a["W2"].shape[1])
        e.p = {k: np.asarray(a[k], float) for k in ("W1", "b1", "W2", "b2")}
        e.trained_steps = int(a["steps"][0]) if "steps" in a else 0
        return e


# ======================================================================================================================================= programs: adapt and splice
_OBS_SUB = re.compile(r"\b(obs|x)\[(\d+)\]")


def adapt_program(src: str, vocab) -> str | None:
    """A recalled program, rewritten to read the columns the NEW task has (indices wrap modulo the new width), and checked against the new vocabulary's safety gate. None if it cannot be used."""
    if not src or len(src) > SRC_CHARS or not vocab.n_cols:
        return None
    try:
        tree = ast.parse(src)
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if len(fns) != 1:
            return None
        fns[0].name = vocab.fn_name
        if [a.arg for a in fns[0].args.args] != list(vocab.args):
            return None
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == vocab.col_var:
                sl = node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, int):
                    node.slice = ast.Constant(sl.value % vocab.n_cols)
                elif isinstance(sl, ast.Tuple) and len(sl.elts) == 2 and isinstance(sl.elts[1], ast.Constant) and isinstance(sl.elts[1].value, int):
                    sl.elts[1] = ast.Constant(sl.elts[1].value % vocab.n_cols)         # x[:, 3] of a table-reading (vector) program
        out = ast.unparse(tree)
        if A.violations(out, **vocab.safety) or A.count_nodes(out) > vocab.max_nodes:
            return None
        A.compile_program(out, vocab.fn_name, **vocab.safety)
        return out
    except (SyntaxError, ValueError, TypeError, AttributeError, RecursionError):
        return None
    except Exception:                                                                      # noqa: BLE001 - a recalled program that fails to compile is simply not used
        return None


# ======================================================================================================================================= the memory
class EpisodicMemory:
    def __init__(self, ws_root: str, seed: int = 0):
        self.dir = os.path.join(os.path.abspath(ws_root), "forge", "episodic")
        self.seed = seed
        self.records: list = []
        self.feats = np.zeros((0, F_IN))
        self.enc = Encoder(seed=seed)
        self.since_fit = 0
        self._vecs = None
        self._load()

    # ---- persistence ------------------------------------------------------------------------------------------------
    def _load(self):
        try:
            recs = fileio.read_json(os.path.join(self.dir, "records.json"))
            z = fileio.load_npz(os.path.join(self.dir, "memory.npz"))
            feats = np.asarray(z["feats"], float)
            if len(recs) == len(feats) and feats.shape[1] == F_IN:
                self.records, self.feats = recs, feats
            if "W1" in z and z["W1"].shape[0] == F_IN:
                self.enc = Encoder.from_arrays(z)
            self.since_fit = int(z["since_fit"][0]) if "since_fit" in z else 0
        except (OSError, ValueError, KeyError):
            pass                                                                           # no memory yet, or an unreadable one: start empty (never crash a solve)

    def save(self):
        os.makedirs(self.dir, exist_ok=True)
        fileio.write_json(os.path.join(self.dir, "records.json"), self.records)
        fileio.save_npz(os.path.join(self.dir, "memory.npz"), dict(self.enc.to_arrays(), feats=self.feats.astype(np.float32), since_fit=np.array([self.since_fit])))

    def __len__(self):
        return len(self.records)

    # ---- adding ------------------------------------------------------------------------------------------------------
    def add(self, kind: str, text: str, num, **meta) -> int:
        rec = dict(id=int(time.time() * 1000) % 10 ** 12 * 1000 + len(self.records) % 1000, kind=kind, text=text[:300], t=time.time(), **meta)
        for k in ("src",):
            if isinstance(rec.get(k), str):
                rec[k] = rec[k][:SRC_CHARS]
        self.records.append(rec)
        self.feats = np.vstack([self.feats, featurize(text, num)[None]])
        self.since_fit += 1
        self._vecs = None
        if len(self.records) > MAX_RECORDS:
            self._forget()
        return rec["id"]

    def _forget(self):
        order = sorted(range(len(self.records)), key=lambda i: ({"state": 0, "failure": 1, "code": 2, "task": 3, "champion": 4}.get(self.records[i]["kind"], 3), self.records[i]["t"]))
        drop = set(order[:len(self.records) - MAX_RECORDS])
        keep = [i for i in range(len(self.records)) if i not in drop]
        self.records = [self.records[i] for i in keep]
        self.feats = self.feats[keep]
        self._vecs = None

    def remember_task(self, task, family: str | None = None, genome=None, skill: float | None = None, src: str | None = None, fn_name: str | None = None,
                      outcome: str = "success", origin: str = ""):
        """One record per (task, agent): a champion (outcome 'success') or a candidate that lost / failed. The text and numbers embedded are the TASK's profile only, so a query task is
        compared with the tasks the agents were built for, not with the agents' own descriptions."""
        text, num = task_profile(task)
        kind = "champion" if outcome == "success" else "failure"
        return self.add(kind, text, num, task=str(getattr(task, "name", "task")), task_kind=str(getattr(task, "kind", "?")), dim=int(task.signature().get("dim", 0)) if hasattr(task, "signature") else 0,
                        family=family, genome=dict(genome) if genome else None, skill=None if skill is None else float(skill), src=src, fn_name=fn_name, outcome=outcome, origin=origin)

    def remember_states(self, task, obs, windows: int = 4):
        """Embed windows of observations (what situations this environment produces), so a later task can be matched on its dynamics' look as well as on its description."""
        obs = np.asarray(obs, float)
        if obs.ndim != 2 or len(obs) < 8:
            return 0
        text, num = task_profile(task)
        n = 0
        for w in np.array_split(np.arange(len(obs)), windows)[:windows]:
            if len(w) < 4:
                continue
            num2 = num.copy()
            num2[9:17] = array_stats(obs[w])
            self.add("state", text, num2, task=str(getattr(task, "name", "task")), task_kind=str(getattr(task, "kind", "env")))
            n += 1
        return n

    def remember_code(self, task_name: str, src: str, skill: float, family: str = ""):
        """A program (success or failure, by its score) embedded by its own tokens: `similar_code(snippet)` finds structurally similar earlier programs."""
        return self.add("code", src, np.zeros(NUM), task=str(task_name), family=family, skill=float(skill), src=src, outcome="success" if skill > 0.05 else "failure")

    # ---- the encoder ---------------------------------------------------------------------------------------------------
    def vectors(self) -> np.ndarray:
        if self._vecs is None or len(self._vecs) != len(self.records):
            self._vecs = self.enc.embed(self.feats) if len(self.records) else np.zeros((0, self.enc.p["W2"].shape[1]))
        return self._vecs

    def _positives(self, idx):
        rs = [self.records[i] for i in idx]
        B = len(rs)
        pos = np.zeros((B, B))
        for a in range(B):
            for b in range(B):
                ra, rb = rs[a], rs[b]
                if ra["kind"] != rb["kind"]:
                    continue
                if ra.get("task") == rb.get("task"):
                    pos[a, b] = 1.0
                elif ra["kind"] == "champion" and ra.get("family") and ra.get("family") == rb.get("family") and ra.get("task_kind") == rb.get("task_kind") \
                        and abs(math.log2((ra.get("dim", 0) + 1.0) / (rb.get("dim", 0) + 1.0))) < 1.0:
                    pos[a, b] = 1.0
        return pos

    def retrieval_accuracy(self, vecs=None, kind: str | None = None) -> float:
        """Leave-one-out: how often a record's nearest OTHER record (same kind) belongs to the same task. 1.0 = perfect; the figure `fit` reports before and after training."""
        V = self.vectors() if vecs is None else vecs
        ids = [i for i, r in enumerate(self.records) if kind is None or r["kind"] == kind]
        if len(ids) < 3:
            return 0.0
        S = V[ids] @ V[ids].T
        np.fill_diagonal(S, -9)
        tasks = [self.records[i].get("task") for i in ids]
        eligible = [a for a in range(len(ids)) if tasks.count(tasks[a]) > 1]
        if not eligible:
            return 0.0
        return float(np.mean([tasks[int(S[a].argmax())] == tasks[a] for a in eligible]))

    def fit(self, steps: int = 300, batch: int = 48, tau: float = 0.2, lr: float = 3e-3, seed: int | None = None) -> dict:
        n = len(self.records)
        if n < 6:
            return dict(trained=False, reason=f"only {n} record(s): nothing to contrast yet")
        before = self.retrieval_accuracy(self.enc.embed(self.feats))
        rng = np.random.default_rng(self.seed if seed is None else seed)
        first, last = self.enc.fit(self.feats, self._positives, steps=steps, batch=batch, tau=tau, lr=lr, rng=rng)
        self._vecs = None
        self.since_fit = 0
        return dict(trained=True, records=n, loss_first=first, loss_last=last, retrieval_before=before, retrieval_after=self.retrieval_accuracy())

    def maybe_fit(self, every: int = 8) -> dict | None:
        if len(self.records) >= 6 and self.since_fit >= max(every, int(0.25 * len(self.records))):
            return self.fit(steps=min(400, 80 + 4 * len(self.records)))
        return None

    # ---- recall ----------------------------------------------------------------------------------------------------------
    def _query(self, task):
        text, num = task_profile(task)
        return self.enc.embed(featurize(text, num))[0]

    def recall(self, task, k: int = 5, kinds=("champion",), min_skill: float = 0.0, exclude_task: bool = False) -> list:
        """The k stored records of the given kinds nearest to `task` by cosine similarity, best first: [{'score', 'task', 'family', 'genome', 'skill', 'src', ...}]. Only a champion that
        reached `min_skill` is returned. A task's own earlier champion is returned too unless exclude_task (the Generalist re-scores its incumbent separately)."""
        if not self.records:
            return []
        q = self._query(task)
        V = self.vectors()
        name = str(getattr(task, "name", ""))
        sc = V @ q
        out = []
        for i in np.argsort(-sc):
            r = self.records[int(i)]
            if r["kind"] not in kinds or (exclude_task and r.get("task") == name):
                continue
            if r["kind"] == "champion" and (r.get("skill") is None or r["skill"] < min_skill):
                continue
            out.append(dict(r, score=float(sc[int(i)])))
            if len(out) >= k:
                break
        return out

    def similar_code(self, snippet: str, k: int = 3) -> list:
        if not self.records:
            return []
        q = self.enc.embed(featurize(snippet, np.zeros(NUM)))[0]
        sc = self.vectors() @ q
        out = []
        for i in np.argsort(-sc):
            r = self.records[int(i)]
            if r["kind"] == "code":
                out.append(dict(r, score=float(sc[int(i)])))
                if len(out) >= k:
                    break
        return out

    # ---- reuse -------------------------------------------------------------------------------------------------------------
    def genomes_for(self, task, family: str, space: dict, k: int = 2, rng=None, recalled=None) -> list:
        """[(genome, label)]: genomes of the nearest champions of `family`, clipped into the family's gene ranges, plus (when two are available) a blend of the two: a spliced candidate.
        `label` names the task each one came from."""
        rng = rng or np.random.default_rng(0)
        recs = [r for r in (recalled if recalled is not None else self.recall(task, k=8, exclude_task=True, min_skill=0.1)) if r.get("family") == family and r.get("genome")]
        gs, labels = [], []
        for r in recs[:k]:
            g = {}
            for key, gene in space.items():
                if key in r["genome"]:
                    try:
                        g[key] = gene.clip(r["genome"][key])
                    except Exception:                                                      # noqa: BLE001
                        pass
            if len(g) == len(space):
                gs.append(g)
                labels.append(str(r.get("task")))
        if len(gs) >= 2:
            child = {}
            for key, gene in space.items():
                a, b = gs[0][key], gs[1][key]
                w = float(rng.uniform(0.25, 0.75))
                child[key] = gene.clip(a * w + b * (1 - w)) if gene.kind != "log" else gene.clip(math.exp(math.log(a) * w + math.log(b) * (1 - w)))
            gs.append(child)
            labels.append(f"{labels[0]}+{labels[1]}")
        return list(zip(gs, labels))

    def seeds_for(self, task, vocab, rng, k: int = 3, recalled=None) -> list:
        """Starting programs for a program agent: recalled champions' sources adapted to this task, and crossovers of pairs of them."""
        recs = [r for r in (recalled if recalled is not None else self.recall(task, k=8, exclude_task=True, min_skill=0.1)) if r.get("src") and r.get("fn_name") == vocab.fn_name]
        progs = []
        for r in recs:
            a = adapt_program(r["src"], vocab)
            if a and a not in progs:
                progs.append(a)
            if len(progs) >= k:
                break
        spliced = []
        for i in range(len(progs)):
            for j in range(i + 1, len(progs)):
                c = A.crossover(progs[i], progs[j], vocab, rng)
                if c and c not in progs and c not in spliced:
                    spliced.append(c)
        return progs + spliced[:2]

    def describe(self) -> str:
        kinds: dict = {}
        for r in self.records:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        tasks = len({r.get("task") for r in self.records})
        return f"episodic memory: {len(self.records)} record(s) over {tasks} task(s) ({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))}); encoder trained {self.enc.trained_steps} step(s)"
