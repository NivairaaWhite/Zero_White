"""obsenc.py - a NON-LINEAR, DISCRETE-CODE observation compressor for program agents, and a way to choose between compressors by what the TASK needs.

`reduce.py` (6.3) gives a program agent a whitened PCA of a wide observation; `latent.py` (6.4) a Perceiver-style attention bottleneck. Both are fitted to RECONSTRUCT the observation, and a reconstruction
objective keeps whatever varies most, relevant to the task or not. This module adds two things:

  VQEncoder     a small autoencoder (observation -> k numbers -> observation, tanh MLPs, gradients written out by hand and checked against finite differences) whose k-number latent is then
                RESIDUALLY VECTOR-QUANTISED: stage 1 is a k-means codebook of the latents, stage 2 a codebook of what stage 1 left over, ... The observation becomes a few integers
                (`tokens(obs)`: one per stage, the 'VQ tokens'), the program reads the dequantised code (k standardised numbers), and the decoder is fine-tuned on QUANTISED latents (with stage dropout) so that
                any prefix of the stages is a usable, coarser code - the same recipe as the audio codec in voice/codec.py. A curved set of observations (a blob at 2-D positions in a picture) is described by a few
                tokens where PCA needs dozens of directions.
  choose_encoder  fit every candidate (PCA, the attention encoder, the VQ encoder) on observations from a random policy and score each by what the TASK can do with its code, on rollouts it was NOT fitted
                on: how well the code predicts the (discounted) RETURN-TO-GO and, with the action, the IMMEDIATE REWARD (both by ridge regression on the code and its squares, held-out R^2; the quality of
                next-observation prediction is reported too but not scored, because it is dominated by whatever varies most). The best candidate wins. This is the 'task-aware' part: a reconstruction-optimal code that discards the one variable that decides the return scores badly here.

HONEST LIMITS. All three are fitted on what a RANDOM policy sees (a feature that only matters in states a random policy never reaches can be lost); the task-aware score uses the return-to-go and the reward of the
random policy as proxies for 'what the policy needs', and a ridge on the code and its squares as the probe, so a code whose useful information is more tangled than that scores low; the choice costs fitting all
candidates (seconds to a minute for wide observations); pictures with thousands of pixels are beyond a 300-1000 sample fit. `ProgramPolicyAgent.encoder = "auto"` uses `choose_encoder`; the default stays PCA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .latent import LatentEncoder
from .reduce import MAX_COLS, Projector


def _kmeans(X: np.ndarray, k: int, rng, iters: int = 25) -> np.ndarray:
    k = max(1, min(int(k), len(X)))
    C = X[rng.choice(len(X), size=k, replace=False)].copy()
    for _ in range(iters):
        d = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            C[j] = X[m].mean(0) if m.any() else X[rng.integers(len(X))]
    return C


def _adam(P, G, m, v, step, lr):
    for j, (p, g) in enumerate(zip(P, G)):
        m[j] = 0.9 * m[j] + 0.1 * g
        v[j] = 0.999 * v[j] + 0.001 * g * g
        p -= lr * (m[j] / (1 - 0.9 ** step)) / (np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)


class VQEncoder:
    NAMES = ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4")

    def __init__(self, params, books, mean, scale, zmean, zscale):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4 = [np.asarray(p, float) for p in params]
        self.books = [np.asarray(b, float) for b in books]
        self.mean, self.scale = np.asarray(mean, float), np.asarray(scale, float)
        self.zmean, self.zscale = np.asarray(zmean, float), np.asarray(zscale, float)

    # ---- structure
    @property
    def k(self) -> int:
        return int(self.W2.shape[1])

    @property
    def stages(self) -> int:
        return len(self.books)

    @property
    def codewords(self) -> int:
        return int(self.books[0].shape[0])

    def bits(self, stages: Optional[int] = None) -> float:
        return (stages or self.stages) * float(np.log2(max(2, self.codewords)))

    # ---- network (also used by the finite-difference test)
    @staticmethod
    def loss_grads(P, X, Z_override=None):
        """MSE of decode(encode(X)) and its gradient for every parameter; `Z_override` replaces the latent (decoder-only fine-tuning on quantised latents)."""
        W1, b1, W2, b2, W3, b3, W4, b4 = P
        h1 = np.tanh(X @ W1 + b1)
        Z = h1 @ W2 + b2 if Z_override is None else Z_override
        h2 = np.tanh(Z @ W3 + b3)
        Y = h2 @ W4 + b4
        loss = float(np.mean((Y - X) ** 2))
        dY = 2.0 * (Y - X) / X.size
        gW4, gb4 = h2.T @ dY, dY.sum(0)
        dh2 = (dY @ W4.T) * (1 - h2 ** 2)
        gW3, gb3 = Z.T @ dh2, dh2.sum(0)
        if Z_override is not None:
            z = np.zeros_like
            return loss, [z(W1), z(b1), z(W2), z(b2), gW3, gb3, gW4, gb4]
        dZ = dh2 @ W3.T
        gW2, gb2 = h1.T @ dZ, dZ.sum(0)
        dh1 = (dZ @ W2.T) * (1 - h1 ** 2)
        return loss, [X.T @ dh1, dh1.sum(0), gW2, gb2, gW3, gb3, gW4, gb4]

    def _standardise(self, X):
        return (np.nan_to_num(np.asarray(X, float)) - self.mean) / self.scale

    def _latent(self, Xs):
        return np.tanh(Xs @ self.W1 + self.b1) @ self.W2 + self.b2

    def _decode(self, Z):
        return np.tanh(Z @ self.W3 + self.b3) @ self.W4 + self.b4

    # ---- quantisation
    def quantise(self, Z, stages: Optional[int] = None):
        res, toks, q = Z.copy(), [], np.zeros_like(Z)
        for book in self.books[:stages or self.stages]:
            d = (res ** 2).sum(1)[:, None] - 2 * res @ book.T + (book ** 2).sum(1)[None, :]
            a = d.argmin(1)
            toks.append(a)
            q += book[a]
            res = Z - q
        return np.stack(toks, axis=1), q

    def dequantise(self, toks) -> np.ndarray:
        toks = np.asarray(toks, dtype=np.int64)
        if toks.ndim == 1:
            toks = toks[None]
        q = np.zeros((len(toks), self.k))
        for s in range(toks.shape[1]):
            q += self.books[s][toks[:, s]]
        return q

    # ---- fitting
    @classmethod
    def fit(cls, obs, k: int = 8, hidden: int = 64, stages: int = 2, codewords: int = 24, epochs: int = 80, finetune_epochs: int = 12, seed: int = 0, lr: float = 3e-3,
            batch: int = 64) -> "VQEncoder":
        X = np.nan_to_num(np.asarray(obs, dtype=float))
        if X.ndim != 2 or len(X) < 16:
            raise ValueError("need at least 16 observations of a fixed width to fit an encoder")
        mean, scale = X.mean(0), X.std(0)
        scale = np.where(scale > 1e-9, scale, 1.0)
        Xs = (X - mean) / scale
        D = Xs.shape[1]
        rng = np.random.default_rng(seed)
        P = [rng.normal(0, np.sqrt(1.0 / D), (D, hidden)), np.zeros(hidden), rng.normal(0, np.sqrt(1.0 / hidden), (hidden, k)), np.zeros(k),
             rng.normal(0, np.sqrt(1.0 / k), (k, hidden)), np.zeros(hidden), rng.normal(0, np.sqrt(1.0 / hidden), (hidden, D)), np.zeros(D)]
        m, v = [np.zeros_like(p) for p in P], [np.zeros_like(p) for p in P]
        step = 0

        def run(n_epochs, Q=None, step_lr=lr):
            nonlocal step
            loss = 0.0
            for _ in range(n_epochs):
                order = rng.permutation(len(Xs))
                for i in range(0, len(order), batch):
                    idx = order[i:i + batch]
                    loss, G = cls.loss_grads(P, Xs[idx], None if Q is None else Q[idx])
                    step += 1
                    _adam(P, G, m, v, step, step_lr)
            return loss

        run(epochs)
        Z = np.tanh(Xs @ P[0] + P[1]) @ P[2] + P[3]
        books, res = [], Z.copy()
        for _ in range(int(stages)):
            C = _kmeans(res, codewords, rng)
            d = (res ** 2).sum(1)[:, None] - 2 * res @ C.T + (C ** 2).sum(1)[None, :]
            res = res - C[d.argmin(1)]
            books.append(C)
        tmp = cls(P, books, mean, scale, np.zeros(k), np.ones(k))
        Qs = [tmp.quantise(Z, s)[1] for s in range(1, len(books) + 1)]             # the latent as seen after 1, 2, ... S stages
        for _ in range(int(finetune_epochs)):                                       # the decoder learns to undo the quantisation noise at EVERY depth (quantiser dropout)
            for s in rng.permutation(len(books)):
                run(1, Q=Qs[int(s)], step_lr=lr * 0.5)
        Qfull = Qs[-1]
        return cls(P, books, mean, scale, Qfull.mean(0), np.maximum(Qfull.std(0), 1e-3))

    # ---- use
    def tokens(self, o, stages: Optional[int] = None) -> np.ndarray:
        """The observation as `stages` integers (the VQ tokens)."""
        Z = self._latent(self._standardise(np.asarray(o, float))[None])
        return self.quantise(Z, stages)[0][0].astype(np.int64)

    def __call__(self, o) -> np.ndarray:
        """What a program agent reads: the dequantised latent, standardised to about unit variance (k numbers)."""
        return (self.dequantise(self.tokens(o))[0] - self.zmean) / self.zscale

    def recon_mse(self, obs, stages: Optional[int] = None) -> float:
        """Error (in standardised units) of decode(dequantise(tokens)): what the discrete code keeps of the observation."""
        Xs = self._standardise(obs)
        toks, q = self.quantise(self._latent(Xs), stages)
        return float(np.mean((self._decode(q) - Xs) ** 2))

    def to_dict(self) -> dict:
        return dict(params=[getattr(self, n).tolist() for n in self.NAMES], books=[b.tolist() for b in self.books], mean=self.mean.tolist(), scale=self.scale.tolist(),
                    zmean=self.zmean.tolist(), zscale=self.zscale.tolist())

    @classmethod
    def from_dict(cls, d: dict) -> "VQEncoder":
        return cls(d["params"], d["books"], d["mean"], d["scale"], d["zmean"], d["zscale"])


# =========================================================================== task-aware choice
@dataclass
class Probe:
    """Transitions of a random policy: obs[i] -> action[i] -> nxt[i], with the discounted return still to come from obs[i]."""
    obs: np.ndarray
    nxt: np.ndarray
    act: np.ndarray
    ret: np.ndarray
    rew: np.ndarray
    n_actions: int
    episodes: int


def collect(task, n: int = 500, seed: int = 0, gamma: float = 0.97) -> Probe:
    rng = np.random.default_rng(seed)
    O, N, A, R, W, eps = [], [], [], [], [], 0
    while len(O) < n:
        o, done, ep_o, ep_n, ep_a, ep_r = task.env.reset(rng), False, [], [], [], []
        while not done and len(O) + len(ep_o) < n + 200:
            a = int(rng.integers(task.n_actions))
            o2, r, done, _ = task.env.step(a)
            ep_o.append(np.asarray(o, float).ravel())
            ep_n.append(np.asarray(o2, float).ravel())
            ep_a.append(a)
            ep_r.append(float(r))
            o = o2
        g, rets = 0.0, []
        for r in reversed(ep_r):
            g = r + gamma * g
            rets.append(g)
        O += ep_o
        N += ep_n
        A += ep_a
        R += rets[::-1]
        W += ep_r
        eps += 1
    return Probe(np.array(O), np.array(N), np.array(A), np.array(R), np.array(W), int(task.n_actions), eps)


def _quad(F: np.ndarray) -> np.ndarray:
    return np.hstack([F, F ** 2, np.ones((len(F), 1))])


def _ridge_r2(Ftr, ytr, Fva, yva, lam: float = 1.0) -> float:
    """Held-out R^2 (clipped to [-1, 1]) of a ridge regression; 0 when the target does not vary."""
    ytr, yva = np.atleast_2d(ytr.T).T, np.atleast_2d(yva.T).T
    if float(np.var(yva)) < 1e-12 or float(np.var(ytr)) < 1e-12:
        return 0.0
    A = Ftr.T @ Ftr + lam * np.eye(Ftr.shape[1])
    W = np.linalg.solve(A, Ftr.T @ ytr)
    err = ((Fva @ W - yva) ** 2).sum(0)
    tot = ((yva - yva.mean(0)) ** 2).sum(0)
    r2 = 1.0 - err / np.maximum(tot, 1e-12)
    return float(np.clip(r2.mean(), -1.0, 1.0))


def task_score(code: Callable[[np.ndarray], np.ndarray], pr: Probe, split: float = 0.7, max_targets: int = 64) -> dict:
    """How useful is `code` for THIS task? Fit on the first `split` of the transitions (whole episodes, in order), score on the rest, all by ridge regression on the code and its squares:
      value     held-out R^2 of the return-to-go
      reward    held-out R^2 of the immediate reward, from code + action (and code x action)
      dynamics  held-out R^2 of the NEXT RAW OBSERVATION (standardised, up to `max_targets` of its columns, the same for every candidate) from code + action. REPORTED, NOT SCORED: raw next-observation
                prediction is dominated by whatever varies most (a loud distractor is easy to predict), which is exactly what a task-aware choice must not reward.
      score     the mean of `value` and `reward` (each floored at 0) for those whose target varies in the held-out part; `dynamics` only when neither does (a task with constant reward).
                `reward` is the sturdy signal on a short probe; `value` of a RANDOM policy is noisy (it mostly reflects future random actions), so give `n` a few thousand when delayed reward is what matters."""
    n = len(pr.obs)
    cut = int(n * split)
    F = np.nan_to_num(np.array([np.asarray(code(o), float) for o in pr.obs]))
    F = np.clip((F - F[:cut].mean(0)) / np.where(F[:cut].std(0) > 1e-9, F[:cut].std(0), 1.0), -3.0, 3.0)      # train-part statistics, clipped: a code that drifts out of its training range must not make the squares explode
    Fq = _quad(F)
    value = _ridge_r2(Fq[:cut], pr.ret[:cut], Fq[cut:], pr.ret[cut:])
    onehot = np.eye(pr.n_actions)[pr.act]
    inter = (F[:, :, None] * onehot[:, None, :]).reshape(len(F), -1) if F.shape[1] * pr.n_actions <= 64 else np.zeros((len(F), 0))      # code x action: what each action does to each code number
    Fd = np.hstack([Fq, onehot, inter])
    reward = _ridge_r2(Fd[:cut], pr.rew[:cut], Fd[cut:], pr.rew[cut:])
    mu, sd = pr.nxt[:cut].mean(0), np.where(pr.nxt[:cut].std(0) > 1e-9, pr.nxt[:cut].std(0), 1.0)
    T = (pr.nxt - mu) / sd
    if T.shape[1] > max_targets:
        T = T[:, np.random.default_rng(0).choice(T.shape[1], max_targets, replace=False)]
    dyn = _ridge_r2(Fd[:cut], T[:cut], Fd[cut:], T[cut:])
    parts = [max(0.0, v) for v, y in ((value, pr.ret), (reward, pr.rew)) if float(np.var(y[cut:])) > 1e-12]          # a negative R^2 means 'nothing usable' (however negative): it counts as 0
    return dict(value=value, reward=reward, dynamics=dyn, score=float(np.mean(parts)) if parts else max(0.0, dyn))


@dataclass
class Choice:
    name: str
    encoder: object
    scores: Dict[str, dict] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def report(self) -> dict:
        return dict(chosen=self.name, scores={n: {k: round(v, 4) for k, v in s.items()} for n, s in self.scores.items()}, errors=self.errors)


def choose_encoder(task, candidates=("pca", "latent", "vq"), k: int = 8, seed: int = 0, n: int = 500, fit_kwargs: Optional[dict] = None) -> Choice:
    """Fit each candidate on the first part of a random-policy probe and pick the one whose code best predicts the task's value and dynamics on the rest. Ties go to the earlier candidate (PCA first:
    the cheapest, exact one). A candidate that fails to fit is reported in `errors` and skipped; if none fits, PCA's ValueError propagates."""
    pr = collect(task, n=max(n, 120), seed=seed)
    cut = int(len(pr.obs) * 0.7)
    fk = fit_kwargs or {}
    best: Optional[Choice] = None
    out = Choice("pca", None)
    for name in candidates:
        try:
            if name == "pca":
                enc = Projector.fit(pr.obs[:cut], k)
            elif name == "latent":
                enc = LatentEncoder.fit(pr.obs[:cut], k=k, seed=seed, **fk.get("latent", dict(epochs=40)))
            elif name == "vq":
                enc = VQEncoder.fit(pr.obs[:cut], k=k, seed=seed, **fk.get("vq", {}))
            else:
                raise KeyError(name)
            sc = task_score(enc, pr)
        except (ValueError, KeyError, np.linalg.LinAlgError, FloatingPointError) as ex:
            out.errors[name] = f"{type(ex).__name__}: {str(ex)[:100]}"
            continue
        out.scores[name] = sc
        if best is None or sc["score"] > out.scores[best.name]["score"] + 1e-9:
            best = Choice(name, enc)
    if best is None:
        raise ValueError(f"no encoder could be fitted ({out.errors})")
    best.scores, best.errors = out.scores, out.errors
    return best
