"""moa.py - Mixture-of-Agents routing: a LEARNED router hands each input to a few micro-agents and blends what they say, in plain numpy.

The chat model gained Mixture-of-Experts feed-forward blocks in 6.4. This module lifts the idea one level up, to whole agents, in three forms that share one router design:

  1. `moa`          (tables: classification / regression) K heterogeneous micro-agents trained TOGETHER with a router: plain MLPs, linear heads, and perception experts (an autoencoder whose
                    code feeds a small head; its reconstruction loss keeps the code a faithful description of the input), and, for regression, optionally a FROZEN evolved program (the
                    `program_formula` family: logic as source code) whose output the router can use but cannot change. The router is a linear softmax gate with noise during training and
                    hard top-k at inference: only the chosen experts run on a row (`predict` groups rows by expert), and a load-balancing term (the Switch-Transformer one: K * sum_k f_k * P_k,
                    f = share of rows whose top choice is k, P = mean gate probability) keeps it from sending everything to one expert. Usage and entropy are MEASURED and reported, not assumed.
  2. `moa_policy`   (control: environments) the same router over K micro-policies inside the actor-critic, trained by the same policy gradient (agents.PGAgent grew three hook methods; the
                    defaults are the old behaviour, so existing agents are unchanged).
  3. `moa_blend`    (the Autonomy layer) after a race, a router is trained over the FINISHED winners of different families (an MLP, a self-modifying net, an evolved formula...), so a task can be
                    handed to a swarm instead of one family. The members are frozen; only the gate learns. The gate is fitted on one half of the validation rows and the blend is SCORED on
                    the other half, which it has never seen, so stacking cannot flatter itself; it is only kept if it beats the single winner on those held-out rows.

HONEST LIMITS: routing helps when different regions of the input want different experts (the tests build such a task and compare against a single expert of the same total size); on a task
one MLP already solves, the mixture costs more parameters for no gain and the race simply does not pick it. A linear gate partitions the input space by hyperplanes, no more. A frozen program
expert can only be blended at the output (programs are not differentiable), and only for regression here. The blend's gain is measured on half the validation set, so it is a noisier estimate
than the full-set skill the other candidates report.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .. import fileio
from ..agents import PGAgent
from ..engine import Tensor, mse_loss, softmax_cross_entropy
from ..evolve import Gene
from ..nn import Adam, Params, init_mlp, mlp, mlp_raw
from .families import FAMILIES, Agent, Family, PGAgentWrap, _clf_skill, _r2


# ======================================================================================================================================= the router
def topk_mask(logits: np.ndarray, k: int) -> np.ndarray:
    """(B, K) 0/1 mask of each row's k largest logits (ties broken by index, deterministic)."""
    K = logits.shape[1]
    k = max(1, min(int(k), K))
    idx = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    m = np.zeros_like(logits)
    np.put_along_axis(m, idx, 1.0, axis=1)
    return m


def gate_forward(leaves, X: Tensor, k: int, noise: float = 0.0, rng=None, prefix: str = "g"):
    """Differentiable noisy top-k gate. Returns (G (B,K) rows summing to 1 over the chosen experts, full softmax P (B,K), mask)."""
    gl = mlp(leaves, prefix, X)
    if noise > 0.0 and rng is not None:
        gl = gl + Tensor(rng.normal(size=gl.shape) * noise)
    mask = topk_mask(gl.data, k)
    shifted = gl - Tensor(gl.data.max(axis=1, keepdims=True))
    e = shifted.exp()
    full = e / e.sum(axis=1, keepdims=True)
    if mask.sum(axis=1).max() <= 1.0:                                               # top-1: the gate VALUE is the expert's softmax probability (Switch Transformer); renormalising a single
        return full * Tensor(mask), full, mask                                        # expert would make its weight a constant 1 and the router would receive no gradient from the task
    em = e * Tensor(mask)
    return em / em.sum(axis=1, keepdims=True), full, mask


def balance_loss(full: Tensor, mask: np.ndarray) -> Tensor:
    """Switch-Transformer load balance: K * sum_k f_k * P_k, with f the share of rows whose TOP choice is k (a constant) and P the mean router probability (differentiable).
    It equals 1 for a perfectly even router and approaches K when one expert takes everything."""
    K = mask.shape[1]
    top1 = np.zeros_like(mask)
    top1[np.arange(len(mask)), full.data.argmax(axis=1)] = 1.0
    f = top1.mean(axis=0)
    return (full.mean(axis=0) * Tensor(f * K)).sum()


def usage_stats(full: np.ndarray) -> dict:
    """What the router actually does on a batch of gate probabilities: each expert's share of first choices, the normalised entropy of that share (1 = even, 0 = one expert does everything), and how many experts matter."""
    K = full.shape[1]
    share = np.bincount(full.argmax(axis=1), minlength=K) / max(1, len(full))
    p = share[share > 0]
    ent = float(-(p * np.log(p)).sum() / math.log(K)) if K > 1 else 1.0
    return dict(share=[round(float(s), 3) for s in share], entropy=round(ent, 3), active=int((share >= 0.02).sum()))


def _gate_raw(arrays, X, k, prefix="g"):
    gl = mlp_raw(arrays, prefix, X)
    mask = topk_mask(gl, k)
    e = np.exp(gl - gl.max(axis=1, keepdims=True))
    full = e / e.sum(axis=1, keepdims=True)
    if mask.sum(axis=1).max() <= 1.0:
        return full * mask, full, mask
    em = e * mask
    return em / em.sum(axis=1, keepdims=True), full, mask


# ======================================================================================================================================= tables
EXPERT_TYPES = ("mlp", "ae", "linear")


def expert_types(K: int, mix: float) -> list:
    shift = int(min(2, max(0, mix * 3)))
    return [EXPERT_TYPES[(k + shift) % 3] for k in range(K)]


class MoAAgent(Agent):
    family = "moa"
    _PROG_GENOME = dict(n_islands=2, island_size=6, tune_steps=8, max_nodes=60, n_mut=1)

    def __init__(self, task, genome, seed):
        g = self.genome = dict(genome)
        self.kind = task.kind
        K, h = int(g["n_experts"]), int(g["hidden"])
        self.K, self.top_k = K, max(1, min(int(g["top_k"]), K))
        rng = np.random.default_rng(seed)
        self.n_in = int(task.in_dim)
        self.n_out = int(task.n_classes) if task.kind == "classification" else 1
        self.types = expert_types(K, float(g["mix"]))
        self.p = Params()
        z = max(2, h // 2)
        for k, t in enumerate(self.types):
            if t == "mlp":
                init_mlp(self.p, f"e{k}", [self.n_in, h, self.n_out], rng, out_scale=0.5)
            elif t == "linear":
                init_mlp(self.p, f"e{k}", [self.n_in, self.n_out], rng, out_scale=0.5)
            else:                                                                           # a perception expert: encode -> code -> head, plus a decoder that keeps the code honest
                init_mlp(self.p, f"e{k}.enc", [self.n_in, h, z], rng)
                init_mlp(self.p, f"e{k}.dec", [z, h, self.n_in], rng)
                init_mlp(self.p, f"e{k}.head", [z, h, self.n_out], rng, out_scale=0.5)
        self.use_prog = bool(int(g.get("use_program", 0))) and task.kind == "regression"
        n_gate = K + (1 if self.use_prog else 0)
        self.n_gate = n_gate
        init_mlp(self.p, "g", [self.n_in, n_gate], rng, out_scale=0.1)
        self.opt = Adam(lr=g["lr"], weight_decay=g["wd"])
        self.ymu = self.ysd = None
        if task.kind == "regression":
            self.ymu, self.ysd = float(task.ytr.mean()), float(task.ytr.std() + 1e-9)
        self.prog = None                                                                    # (src, vocab kwargs, cols) of the frozen program expert, built on the first train()
        self._prog_fn = None
        self.spent = 0.0

    # ---- the frozen program expert ------------------------------------------------------------------------------------------------
    def _build_program(self, task, rng):
        from .families import ProgramRegAgent
        a = ProgramRegAgent(task, dict(self._PROG_GENOME), int(rng.integers(1 << 30)))
        a.train(task, 0.25, rng)
        a.val_score(task)
        self.prog = dict(src=a.best_src(), cols=list(a.cols), max_nodes=self._PROG_GENOME["max_nodes"])
        self._load_program()

    def _load_program(self):
        if not self.prog:
            self._prog_fn = None
            return
        from ..selfmod import astmut as A
        v = A.Vocab("predict", ("x",), col_var="x", n_cols=len(self.prog["cols"]), vector=True, max_nodes=self.prog["max_nodes"])
        fn = A.compile_program(self.prog["src"], "predict", **v.safety)
        cols = list(self.prog["cols"])
        ymu, ysd = self.ymu, self.ysd

        def f(X):
            with np.errstate(all="ignore"):
                p = np.broadcast_to(np.asarray(fn(X[:, cols]), dtype=float), (len(X),))
            p = np.nan_to_num((p - ymu) / ysd, nan=0.0, posinf=5.0, neginf=-5.0)
            return np.clip(p, -5.0, 5.0)[:, None]
        self._prog_fn = f

    # ---- forward ------------------------------------------------------------------------------------------------------------------
    def _expert(self, leaves, k, X):
        t = self.types[k]
        if t == "ae":
            z = mlp(leaves, f"e{k}.enc", X).tanh()
            return mlp(leaves, f"e{k}.head", z), mlp(leaves, f"e{k}.dec", z)
        return mlp(leaves, f"e{k}", X), None

    def _expert_raw(self, k, X):
        a, t = self.p.a, self.types[k]
        if t == "ae":
            return mlp_raw(a, f"e{k}.head", np.tanh(mlp_raw(a, f"e{k}.enc", X)))
        return mlp_raw(a, f"e{k}", X)

    def _mix(self, leaves, X: Tensor, train: bool, rng):
        G, full, mask = gate_forward(leaves, X, self.top_k, noise=self.genome["gate_noise"] if train else 0.0, rng=rng)
        out, recons = None, []
        for j in range(self.n_gate):
            if j < self.K:
                y, rec = self._expert(leaves, j, X)
                if rec is not None:
                    recons.append(rec)
            else:                                                                           # the frozen program: a constant column the gate may weight
                y = Tensor(np.repeat(self._prog_fn(X.data), self.n_out, axis=1))
            onehot = np.zeros(self.n_gate)
            onehot[j] = 1.0
            part = (G * Tensor(onehot)).sum(axis=1, keepdims=True) * y
            out = part if out is None else out + part
        return out, full, mask, recons

    def train(self, task, units, rng):
        if self.use_prog and self.prog is None:
            self._build_program(task, rng)
        X, y = task.Xtr, task.ytr
        bs = min(int(self.genome["batch"]), len(X))
        yt = y if self.kind == "classification" else ((y - self.ymu) / self.ysd)
        bal = float(self.genome["balance"])
        for _ in range(max(5, int(150 * units))):
            idx = rng.integers(0, len(X), size=bs)
            xb = Tensor(X[idx])
            l = self.p.leaves()
            out, full, mask, recons = self._mix(l, xb, True, rng)
            loss = softmax_cross_entropy(out, yt[idx])[0] if self.kind == "classification" else mse_loss(out, yt[idx][:, None])
            loss = loss + balance_loss(full, mask) * bal
            for r in recons:
                loss = loss + mse_loss(r, X[idx]) * 0.2 / max(1, len(recons))
            loss.backward()
            self.opt.step(self.p, l)
        self.spent += units

    # ---- inference: only the chosen experts run on a row --------------------------------------------------------------------------
    def _forward_raw(self, X):
        X = np.asarray(X, dtype=float)
        G, full, mask = _gate_raw(self.p.a, X, self.top_k)
        out = np.zeros((len(X), self.n_out))
        for j in range(self.n_gate):
            rows = np.nonzero(mask[:, j] > 0)[0]
            if not len(rows):
                continue
            y = self._expert_raw(j, X[rows]) if j < self.K else np.repeat(self._prog_fn(X[rows]), self.n_out, axis=1)
            out[rows] += G[rows, j:j + 1] * y
        return out, full

    def outputs(self, X):
        out, _ = self._forward_raw(X)
        if self.kind == "classification":
            e = np.exp(out - out.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        return out[:, 0] * self.ysd + self.ymu

    def predict(self, X):
        o = self.outputs(X)
        return o.argmax(axis=1) if self.kind == "classification" else o

    def route_stats(self, X) -> dict:
        _, full = self._forward_raw(np.asarray(X)[:2000])
        return usage_stats(full)

    def _metric(self, X, y, task):
        with np.errstate(all="ignore"):
            p = self.predict(X)
        st = self.route_stats(X)
        if self.kind == "classification":
            acc = float((p == y).mean())
            return _clf_skill(acc, task.chance()), dict(acc=acc, experts_active=st["active"], router_entropy=st["entropy"])
        r2 = _r2(p, y)
        return float(np.clip(r2, -1.0, 1.0)), dict(r2=r2, experts_active=st["active"], router_entropy=st["entropy"])

    def val_score(self, task):
        return self._metric(task.Xval, task.yval, task)

    def test_metrics(self, task):
        return self._metric(task.Xte, task.yte, task)[1]

    def n_params(self):
        return self.p.count()

    def describe(self):
        g = self.genome
        counts = {t: self.types.count(t) for t in EXPERT_TYPES if t in self.types}
        prog = " + 1 frozen evolved program" if self.prog else ""
        return (f"mixture of {self.K} micro-agents ({', '.join(f'{v} {k}' for k, v in counts.items())}{prog}), hidden {g['hidden']}, top-{self.top_k} noisy linear router "
                f"(balance weight {g['balance']:.3g}), {self.p.count()} parameters")

    def save(self, d):
        arrays = {f"p.{k}": v for k, v in self.p.a.items()}
        meta = dict(genome=self.genome, kind=self.kind, ymu=self.ymu, ysd=self.ysd, spent=self.spent, prog=self.prog, n_in=self.n_in, n_out=self.n_out)
        arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8)
        fileio.save_npz(os.path.join(d, "agent.npz"), arrays)


class MoAFamily(Family):
    name, kinds = "moa", ("classification", "regression")
    space = {"n_experts": Gene("int", 2, 6, shape=True), "hidden": Gene("int", 4, 64, shape=True), "top_k": Gene("int", 1, 3), "balance": Gene("log", 1e-3, 0.3),
             "gate_noise": Gene("float", 0.0, 0.5), "mix": Gene("float", 0.0, 1.0), "use_program": Gene("int", 0, 1), "lr": Gene("log", 1e-3, 5e-2), "wd": Gene("log", 1e-6, 1e-2),
             "batch": Gene("int", 16, 128)}

    def build(self, task, genome, seed):
        return MoAAgent(task, genome, seed)

    def load(self, d, task, genome):
        w = fileio.load_npz(os.path.join(d, "agent.npz"))
        meta = json.loads(bytes(w["__meta__"]).decode("utf-8"))
        a = MoAAgent(task, meta["genome"], 0)
        a.p = Params({k[2:]: w[k] for k in w if k.startswith("p.")})
        a.ymu, a.ysd, a.spent, a.prog = meta["ymu"], meta["ysd"], meta["spent"], meta.get("prog")
        a._load_program()
        return a


# ======================================================================================================================================= control
class MoAPolicy(PGAgent):
    """An actor-critic whose POLICY is a routed mixture of micro-policies (mixed in logit space); the critic stays one MLP."""
    kind = "moa_pgagent"

    def __init__(self, obs_dim, n_actions, hidden=32, depth=1, lr=1e-2, gamma=0.97, entropy=0.02, curiosity=0.0, seed=0, n_experts=3, balance=0.05, top_k=2):
        super().__init__(obs_dim, n_actions, hidden=hidden, depth=depth, lr=lr, gamma=gamma, entropy=entropy, curiosity=curiosity, seed=seed)
        self.cfg.update(n_experts=n_experts, balance=balance, top_k=top_k)
        for k in [k for k in self.p.a if k.startswith("pi.")]:
            del self.p.a[k]                                                                  # the plain policy net the parent built is replaced by the experts
        rng = np.random.default_rng(seed + 17)
        self.K, self.top_k = int(n_experts), max(1, min(int(top_k), int(n_experts)))
        for k in range(self.K):
            init_mlp(self.p, f"pi{k}", [obs_dim] + [hidden] * depth + [n_actions], rng, out_scale=0.1)
        init_mlp(self.p, "g", [obs_dim, self.K], rng, out_scale=0.1)

    def _logits(self, leaves, X):
        G, full, mask = gate_forward(leaves, X, self.top_k)
        out = None
        for k in range(self.K):
            onehot = np.zeros(self.K)
            onehot[k] = 1.0
            part = (G * Tensor(onehot)).sum(axis=1, keepdims=True) * mlp(leaves, f"pi{k}", X)
            out = part if out is None else out + part
        self._last = (full, mask)
        return out

    def _aux_loss(self, leaves, X):
        full, mask = self._last
        return balance_loss(full, mask) * float(self.cfg["balance"])

    def _logits_raw(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        G, _, mask = _gate_raw(self.p.a, X, self.top_k)
        out = np.zeros((len(X), self.cfg["n_actions"]))
        for k in range(self.K):
            rows = np.nonzero(mask[:, k] > 0)[0]
            if len(rows):
                out[rows] += G[rows, k:k + 1] * mlp_raw(self.p.a, f"pi{k}", X[rows])
        return out

    def route_stats(self, obs) -> dict:
        _, full, _ = _gate_raw(self.p.a, np.atleast_2d(np.asarray(obs, dtype=float)), self.top_k)
        return usage_stats(full)

    def genome(self):
        return {k: self.cfg[k] for k in ("hidden", "depth", "lr", "gamma", "entropy", "curiosity", "n_experts", "balance", "top_k")}


class MoAPolicyWrap(PGAgentWrap):
    family = "moa_policy"

    def __init__(self, task, genome, seed, agent=None):
        Agent.__init__(self)
        self.genome = dict(genome)
        g = dict(genome)
        self.agent = agent or MoAPolicy(task.obs_dim, task.n_actions, seed=seed, **g)
        self.spent = 0.0
        self._probe = None

    def val_score(self, task):
        s, m = super().val_score(task)
        m.update(self._routing(task))
        return s, m

    def _routing(self, task):
        try:
            if self._probe is None:
                from .episodic import probe_obs
                self._probe = probe_obs(task, 100)
            st = self.agent.route_stats(self._probe)
            return dict(experts_active=st["active"], router_entropy=st["entropy"])
        except Exception:                                                                  # noqa: BLE001
            return {}

    def describe(self):
        g = self.genome
        return (f"mixture-of-policies actor-critic: {g['n_experts']} micro-policies ({g['depth']}x{g['hidden']}), top-{self.agent.top_k} router "
                f"(balance {g['balance']:.3g}), critic MLP, lr={g['lr']:.3g}, gamma={g['gamma']:.3f}, {self.agent.updates_trained} updates")

    def save(self, d):
        self.agent.save(os.path.join(d, "agent.npz"))


class MoAPolicyFamily(Family):
    name, kinds = "moa_policy", ("env",)
    space = {"hidden": Gene("int", 8, 64, shape=True), "depth": Gene("int", 1, 2, shape=True), "lr": Gene("log", 1e-3, 3e-2), "gamma": Gene("float", 0.9, 0.995),
             "entropy": Gene("log", 1e-3, 0.1), "curiosity": Gene("float", 0.0, 1.0), "n_experts": Gene("int", 2, 5, shape=True), "balance": Gene("log", 1e-3, 0.3), "top_k": Gene("int", 1, 3)}

    def build(self, task, genome, seed):
        return MoAPolicyWrap(task, genome, seed)

    def load(self, d, task, genome):
        a = MoAPolicyWrap.__new__(MoAPolicyWrap)
        a.agent, a.genome, a.spent, a._probe = MoAPolicy.load(os.path.join(d, "agent.npz")), dict(genome), 0.0, None
        return a


# ======================================================================================================================================= the Autonomy layer: a router over finished agents
def member_outputs(agent, X, task):
    """Soft outputs of a finished agent on X: class probabilities (n, C) for classification (a one-hot vote smoothed to 0.9 / 0.1 when the family has no probabilities), predictions (n,) for regression."""
    if hasattr(agent, "outputs"):
        return np.asarray(agent.outputs(X), dtype=float)
    p = agent.predict(X)
    if task.kind == "classification":
        C = int(task.n_classes)
        o = np.full((len(X), C), 0.1 / max(1, C - 1))
        o[np.arange(len(X)), np.asarray(p, dtype=int)] = 0.9
        return o
    return np.asarray(p, dtype=float)


class BlendAgent(Agent):
    """A gate over frozen member agents (each already trained, from any family). The gate is fitted on the even-indexed validation rows; scoring uses the odd-indexed ones only."""
    family = "moa_blend"

    def __init__(self, task, members, genome=None, seed=0, gate=None):
        self.members = list(members)                                   # [(family_name, genome, agent)]
        self.K = len(self.members)
        self.genome = dict(genome or dict(members=[f"{m[0]}" for m in self.members]))
        self.kind, self.spent = task.kind, 0.0
        self.p = Params(gate or {})
        if not gate:
            init_mlp(self.p, "g", [int(task.in_dim), self.K], np.random.default_rng(seed), out_scale=0.1)
        self.ymu = self.ysd = None
        if task.kind == "regression":
            self.ymu, self.ysd = float(task.ytr.mean()), float(task.ytr.std() + 1e-9)
        self.top_k = self.K
        self.opt = Adam(lr=5e-2, weight_decay=1e-4)

    @staticmethod
    def split(n):
        idx = np.arange(n)
        return idx[idx % 2 == 0], idx[idx % 2 == 1]                  # fit rows, scored rows

    def _stack(self, X, task):
        outs = [member_outputs(m[2], X, task) for m in self.members]
        return np.stack(outs, axis=0)                                  # (K, n, C) or (K, n)

    def fit_gate(self, task, steps=300, rng=None):
        rng = rng or np.random.default_rng(0)
        fit_rows, _ = self.split(len(task.Xval))
        X, y = task.Xval[fit_rows], task.yval[fit_rows]
        S = self._stack(X, task)
        clf = self.kind == "classification"
        yt = None if clf else (y - self.ymu) / self.ysd
        St = None if clf else (S - self.ymu) / self.ysd
        for _ in range(steps):
            l = self.p.leaves()
            G, _, _ = gate_forward(l, Tensor(X), self.K)
            if clf:
                # mixture likelihood: probability the blend gives the true class = sum_k g_k * P_k(y); maximise its log
                pk = S[:, np.arange(len(X)), y.astype(int)].T         # (n, K): each member's probability of the TRUE class
                loss = (Tensor(-1.0) * ((G * Tensor(pk)).sum(axis=1, keepdims=True) + 1e-6).log()).mean()
            else:
                pred = (G * Tensor(St.T)).sum(axis=1, keepdims=True)
                loss = mse_loss(pred, yt[:, None])
            loss.backward()
            self.opt.step(self.p, l)

    def outputs(self, X, task=None):
        task = task or self._task
        G, full, _ = _gate_raw(self.p.a, np.asarray(X, float), self.K)
        S = self._stack(X, task)
        if self.kind == "classification":
            return np.einsum("nk,knc->nc", G, S)
        return np.einsum("nk,kn->n", G, S)

    def predict(self, X):
        o = self.outputs(X)
        return o.argmax(axis=1) if self.kind == "classification" else o

    def _metric_rows(self, X, y, task):
        with np.errstate(all="ignore"):
            p = self.predict(X)
        if self.kind == "classification":
            acc = float((p == y).mean())
            return _clf_skill(acc, task.chance()), dict(acc=acc)
        r2 = _r2(p, y)
        return float(np.clip(r2, -1.0, 1.0)), dict(r2=r2)

    def val_score(self, task):
        self._task = task
        _, held = self.split(len(task.Xval))                          # the half the gate never saw
        s, m = self._metric_rows(task.Xval[held], task.yval[held], task)
        _, full, _ = _gate_raw(self.p.a, task.Xval[held], self.K)
        st = usage_stats(full)
        m.update(scored_on="held-out half of validation", members_active=st["active"], router_entropy=st["entropy"])
        return s, m

    def test_metrics(self, task):
        self._task = task
        return self._metric_rows(task.Xte, task.yte, task)[1]

    def n_params(self):
        return self.p.count() + sum(m[2].n_params() for m in self.members)

    def describe(self):
        return "router over finished agents: " + " | ".join(f"{m[0]}: {m[2].describe().splitlines()[0][:70]}" for m in self.members)

    def save(self, d):
        os.makedirs(d, exist_ok=True)
        listing = []
        for i, (fam, genome, agent) in enumerate(self.members):
            sub = os.path.join(d, f"m{i}")
            os.makedirs(sub, exist_ok=True)
            agent.save(sub)
            listing.append(dict(family=fam, genome=genome, dir=f"m{i}"))
        arrays = {f"p.{k}": v for k, v in self.p.a.items()}
        arrays["__meta__"] = np.frombuffer(json.dumps(dict(members=listing, kind=self.kind, ymu=self.ymu, ysd=self.ysd, genome=self.genome)).encode("utf-8"), np.uint8)
        fileio.save_npz(os.path.join(d, "blend.npz"), arrays)


class BlendFamily(Family):
    name, kinds, space = "moa_blend", (), {}                            # never proposed at random: it is built from a finished race (build_blend)

    def applicable(self, task):
        return False

    def load(self, d, task, genome):
        from .families import FAMILIES
        w = fileio.load_npz(os.path.join(d, "blend.npz"))
        meta = json.loads(bytes(w["__meta__"]).decode("utf-8"))
        members = []
        for m in meta["members"]:
            members.append((m["family"], m["genome"], FAMILIES[m["family"]].load(os.path.join(d, m["dir"]), task, m["genome"])))
        b = BlendAgent(task, members, genome=meta["genome"], gate={k[2:]: w[k] for k in w if k.startswith("p.")})
        b.ymu, b.ysd, b._task = meta["ymu"], meta["ysd"], task
        return b


def heldout_skill(agent, task) -> float:
    """An agent's skill on the validation rows a BlendAgent scores itself on (the odd-indexed ones), so a blend and the single winner can be compared on exactly the same unseen rows."""
    _, held = BlendAgent.split(len(task.Xval))
    X, y = task.Xval[held], task.yval[held]
    with np.errstate(all="ignore"):
        p = agent.predict(X)
    if task.kind == "classification":
        return _clf_skill(float((p == y).mean()), task.chance())
    return float(np.clip(_r2(p, y), -1.0, 1.0))


def build_blend(task, finalists, seed=0, steps=300):
    """A BlendAgent over the best agent of each DIFFERENT family among `finalists` (objects with .agent and .skill, best first), or None when fewer than two families are available.
    Supervised tasks only; the gate is trained here (on the even validation rows)."""
    if task.kind not in ("classification", "regression"):
        return None
    seen, members = set(), []
    for c in finalists:
        fam = c.agent.family
        if fam in seen or fam == "moa_blend" or c.skill <= -9:
            continue
        seen.add(fam)
        members.append((fam, dict(c.agent.genome), c.agent))
        if len(members) >= 4:
            break
    if len(members) < 2:
        return None
    b = BlendAgent(task, members, seed=seed)
    b._task = task
    b.fit_gate(task, steps=steps, rng=np.random.default_rng(seed))
    return b


for _f in (MoAFamily(), MoAPolicyFamily(), BlendFamily()):
    FAMILIES[_f.name] = _f
