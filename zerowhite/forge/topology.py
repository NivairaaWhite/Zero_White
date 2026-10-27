"""topology.py - topology INVENTION: a network whose wiring (not just its width) is part of what is searched, grown by function-preserving mutations.

`mlp` searches a width and a depth. `selfmod_net` (architecture.py) grows and prunes layers of a plain stack. This family searches over DAGS: every hidden node has its own width and its own
activation, and takes input from ANY earlier node (the raw input included), so skip connections, parallel branches and mixed activations are all reachable. The graph:

    node 0 = the input;  nodes 1..k = hidden groups, in a topological order that is a property of the graph, not of the node's number;  the last node = the output (linear)
    h_i = act_i( sum over j in src(i) of  h_j @ W[j->i]  + b_i )          (a sum of one weight matrix per edge = a dense layer over the concatenation of its sources)

Every mutation is exactly function-PRESERVING at the moment it happens, so a search step can never make the network worse by wiring alone, and the new capacity is learned from there:

    widen(i, k)          k more units in node i: random incoming weights, ZERO outgoing weights
    add_node(w, act)     a new node between two others: random incoming edges, ZERO outgoing edges to later nodes (the output included)
    add_edge(j -> i)     a new connection (a skip connection when it jumps over nodes): ZERO weights

(`tests` check that the outputs before and after are identical to 1e-12.) The agent's search: fit; then each round try `cands` mutations, each cloned, mutated and fine-tuned a little, and keep the best
only if it beats the current network on a held-out slice of the TRAINING data by a margin. The slice is not the validation split the Generalist selects agents on, so architecture selection cannot
leak into the race. A round whose best candidate does not win changes nothing.

Honest limits: tanh/relu/sigmoid are the activations the autograd engine has; there is no node removal (pruning is not function-preserving); the search is greedy, not a population; and on a task
a plain MLP already solves it will, correctly, mostly keep its initial wiring (the report says how many rounds changed anything).
"""
from __future__ import annotations

import json
import os

import numpy as np

from .. import fileio
from ..engine import Tensor, mse_loss, softmax_cross_entropy
from ..evolve import Gene
from ..nn import Adam, Params
from .families import FAMILIES, Agent, Family, _clf_skill, _r2

ACTS = ("tanh", "relu", "sigmoid")
INPUT, OUTPUT = 0, -1                                    # ids: the input is 0, the output is -1, hidden nodes are 1, 2, 3, ... in creation order


def _act(t: Tensor, name: str) -> Tensor:
    return t.tanh() if name == "tanh" else t.relu() if name == "relu" else t.sigmoid()


def _act_np(z: np.ndarray, name: str) -> np.ndarray:
    return np.tanh(z) if name == "tanh" else np.maximum(0.0, z) if name == "relu" else 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class TopologyNet:
    """The graph and its parameters. `nodes[id] = dict(w, act)`, `order` = hidden ids in topological order, `src[id]` = the ids it reads (INPUT, hidden ids; OUTPUT for the output)."""

    def __init__(self, in_dim: int, out_dim: int, hidden=(8,), acts=None, seed: int = 0):
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)
        self.rng = np.random.default_rng(seed)
        self.nodes, self.order, self.src, self.next_id = {}, [], {OUTPUT: []}, 1
        self.p = Params()
        prev = INPUT
        for k, w in enumerate(hidden):
            nid = self._new_node(int(w), (acts[k] if acts else ACTS[int(self.rng.integers(len(ACTS)))]))
            self.order.append(nid)
            self.src[nid] = [prev]
            self._add_edge_params(prev, nid, scale=1.0)
            prev = nid
        self.src[OUTPUT] = [prev]
        self._add_edge_params(prev, OUTPUT, scale=0.5)

    # ---- structure ------------------------------------------------------------------------------
    def width(self, nid: int) -> int:
        return self.in_dim if nid == INPUT else self.out_dim if nid == OUTPUT else self.nodes[nid]["w"]

    def _new_node(self, w: int, act: str) -> int:
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = dict(w=int(w), act=act)
        self.p.add(f"b.{nid}", np.zeros(int(w)))
        return nid

    def _add_edge_params(self, j: int, i: int, scale: float):
        shape = (self.width(j), self.width(i))
        if scale == 0.0:
            W = np.zeros(shape)
        else:
            W = self.rng.standard_normal(shape) * np.sqrt(2.0 / max(1, shape[0])) * scale
        self.p.add(f"W.{j}.{i}", W)
        if i == OUTPUT and f"b.{OUTPUT}" not in self.p.a:
            self.p.add(f"b.{OUTPUT}", np.zeros(self.out_dim))

    def pos(self, nid: int) -> int:
        return -1 if nid == INPUT else len(self.order) if nid == OUTPUT else self.order.index(nid)

    def edges(self):
        return [(j, i) for i in list(self.order) + [OUTPUT] for j in self.src[i]]

    def n_params(self) -> int:
        return self.p.count()

    def clone(self) -> "TopologyNet":
        c = TopologyNet.__new__(TopologyNet)
        c.in_dim, c.out_dim = self.in_dim, self.out_dim
        c.rng = np.random.default_rng(int(self.rng.integers(1 << 31)))
        c.nodes = {k: dict(v) for k, v in self.nodes.items()}
        c.order, c.next_id = list(self.order), self.next_id
        c.src = {k: list(v) for k, v in self.src.items()}
        c.p = self.p.clone()
        return c

    # ---- forward --------------------------------------------------------------------------------
    def forward(self, X, leaves=None):
        """Differentiable when `leaves` (Params.leaves()) is given, plain numpy otherwise."""
        if leaves is not None:
            h = {INPUT: Tensor(np.asarray(X, dtype=np.float64))}
            for i in list(self.order) + [OUTPUT]:
                z = leaves[f"b.{i}"]
                for j in self.src[i]:
                    z = z + h[j].matmul(leaves[f"W.{j}.{i}"])
                h[i] = z if i == OUTPUT else _act(z, self.nodes[i]["act"])
            return h[OUTPUT]
        h = {INPUT: np.asarray(X, dtype=np.float64)}
        a = self.p.a
        for i in list(self.order) + [OUTPUT]:
            z = a[f"b.{i}"].copy()
            for j in self.src[i]:
                z = z + h[j] @ a[f"W.{j}.{i}"]
            h[i] = z if i == OUTPUT else _act_np(z, self.nodes[i]["act"])
        return h[OUTPUT]

    # ---- function-preserving mutations ----------------------------------------------------------------
    def widen(self, nid: int, k: int) -> bool:
        if nid not in self.nodes or k < 1 or self.nodes[nid]["w"] + k > 96:
            return False
        a = self.p.a
        for j in self.src[nid]:                                              # incoming: random columns
            W = a[f"W.{j}.{nid}"]
            new = self.rng.standard_normal((W.shape[0], k)) * np.sqrt(2.0 / max(1, W.shape[0]))
            a[f"W.{j}.{nid}"] = np.concatenate([W, new], axis=1)
        a[f"b.{nid}"] = np.concatenate([a[f"b.{nid}"], np.zeros(k)])
        for i in list(self.order) + [OUTPUT]:                                # outgoing: ZERO rows, so nothing downstream changes
            if nid in self.src[i]:
                W = a[f"W.{nid}.{i}"]
                a[f"W.{nid}.{i}"] = np.concatenate([W, np.zeros((k, W.shape[1]))], axis=0)
        self.nodes[nid]["w"] += k
        return True

    def _known(self, nid: int) -> bool:
        return nid in (INPUT, OUTPUT) or nid in self.nodes

    def add_edge(self, j: int, i: int) -> bool:
        if not (self._known(j) and self._known(i)) or i == INPUT or j == OUTPUT or j in self.src[i] or self.pos(j) >= self.pos(i):
            return False
        self.src[i].append(j)
        self._add_edge_params(j, i, scale=0.0)
        return True

    def add_node(self, w: int, act: str, srcs, dsts) -> bool:
        """A new hidden node with random incoming edges from `srcs` and ZERO outgoing edges to `dsts`. It goes right after the latest of its sources, so every edge points forward."""
        if len(self.order) >= 12 or not srcs or not dsts or act not in ACTS or not all(self._known(n) for n in list(srcs) + list(dsts)) or INPUT in dsts or OUTPUT in srcs:
            return False
        at = max(self.pos(j) for j in srcs) + 1
        if any(self.pos(d) < at for d in dsts):
            return False
        nid = self._new_node(int(w), act)
        self.order.insert(at, nid)
        self.src[nid] = list(srcs)
        for j in srcs:
            self._add_edge_params(j, nid, scale=1.0)
        for d in dsts:
            self.src[d].append(nid)
            self._add_edge_params(nid, d, scale=0.0)
        return True

    def mutate(self, rng) -> str | None:
        """One random function-preserving mutation. Returns a short description, or None when the draw could not be applied."""
        kind = ("widen", "add_node", "add_edge")[int(rng.choice(3, p=[0.35, 0.4, 0.25]))]
        hidden = list(self.order)
        if kind == "widen" and hidden:
            nid = hidden[int(rng.integers(len(hidden)))]
            k = int(rng.integers(2, 9))
            return f"widen node {nid} by {k}" if self.widen(nid, k) else None
        if kind == "add_node":
            pool = [INPUT] + hidden
            srcs = sorted({pool[int(rng.integers(len(pool)))] for _ in range(int(rng.integers(1, 3)))}, key=self.pos)
            later = [n for n in hidden + [OUTPUT] if self.pos(n) > max(self.pos(j) for j in srcs)]
            if not later:
                return None
            dsts = sorted({later[int(rng.integers(len(later)))] for _ in range(int(rng.integers(1, 3)))}, key=self.pos)
            act, w = ACTS[int(rng.integers(len(ACTS)))], int(rng.integers(3, 13))
            return f"add a {w}-unit {act} node reading {srcs} feeding {dsts}" if self.add_node(w, act, srcs, dsts) else None
        pool = [INPUT] + hidden
        j, i = pool[int(rng.integers(len(pool)))], (hidden + [OUTPUT])[int(rng.integers(len(hidden) + 1))]
        return f"add edge {j} -> {i}" if self.add_edge(j, i) else None

    # ---- persistence ----------------------------------------------------------------------------------
    def spec(self) -> dict:
        return dict(in_dim=self.in_dim, out_dim=self.out_dim, nodes={str(k): v for k, v in self.nodes.items()}, order=self.order, next_id=self.next_id,
                    src={str(k): v for k, v in self.src.items()})

    @classmethod
    def from_spec(cls, spec: dict, arrays: dict, seed: int = 0) -> "TopologyNet":
        n = cls.__new__(cls)
        n.in_dim, n.out_dim = spec["in_dim"], spec["out_dim"]
        n.rng = np.random.default_rng(seed)
        n.nodes = {int(k): dict(v) for k, v in spec["nodes"].items()}
        n.order, n.next_id = list(spec["order"]), spec["next_id"]
        n.src = {int(k): list(v) for k, v in spec["src"].items()}
        n.p = Params(arrays)
        return n

    def describe(self) -> str:
        parts = [f"{self.in_dim} inputs"]
        for i in self.order:
            nd = self.nodes[i]
            parts.append(f"node {i}: {nd['w']} {nd['act']} units <- {self.src[i]}")
        parts.append(f"output ({self.out_dim}) <- {self.src[OUTPUT]}")
        skips = sum(1 for (j, i) in self.edges() if self.pos(i) - self.pos(j) > 1)
        return "; ".join(parts) + f" [{len(self.order)} hidden nodes, {len(self.edges())} edges, {skips} skip connection(s), {self.p.count()} parameters]"


# =========================================================================== the agent
class TopologyAgent(Agent):
    family = "topology_net"

    def __init__(self, task, genome, seed, net: TopologyNet | None = None):
        self.genome, self.kind = dict(genome), task.kind
        self.seed = int(seed)
        out = task.n_classes if task.kind == "classification" else 1
        self.net = net or TopologyNet(task.in_dim, out, hidden=(genome["init_width"],), seed=seed)
        self.ymu = self.ysd = None
        if task.kind == "regression":
            self.ymu, self.ysd = float(task.ytr.mean()), float(task.ytr.std() + 1e-9)
        self.spent, self.rounds_done, self.rounds_changed, self.history, self.loss_trail = 0.0, 0, 0, [], []
        r = np.random.default_rng(seed + 17)
        n = len(task.Xtr)
        idx = r.permutation(n)
        cut = max(8, int(0.8 * n))
        self._fit_idx, self._hold_idx = idx[:cut], idx[cut:] if n - cut >= 8 else idx[:cut]

    # ---- training
    def _targets(self, task, y):
        return y if self.kind == "classification" else (y - self.ymu) / self.ysd

    def _loss_leaves(self, net, leaves, X, yt):
        out = net.forward(X, leaves)
        return softmax_cross_entropy(out, yt)[0] if self.kind == "classification" else mse_loss(out, yt[:, None])

    def _fit(self, net, task, steps, rng, lr, wd):
        X, y = task.Xtr[self._fit_idx], self._targets(task, task.ytr[self._fit_idx])
        opt = Adam(lr=lr, weight_decay=wd)
        bs = min(64, len(X))
        for _ in range(steps):
            idx = rng.integers(0, len(X), size=bs)
            l = net.p.leaves()
            loss = self._loss_leaves(net, l, X[idx], y[idx])
            loss.backward()
            opt.step(net.p, l)

    def _holdout_loss(self, net, task) -> float:
        X, y = task.Xtr[self._hold_idx], self._targets(task, task.ytr[self._hold_idx])
        with np.errstate(all="ignore"):
            z = net.forward(X)
            if self.kind == "classification":
                z = z - z.max(axis=1, keepdims=True)
                nll = -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(axis=1)))
                return float(nll.mean())
            return float(np.mean((z[:, 0] - y) ** 2))

    def train(self, task, units, rng):
        g = self.genome
        steps = max(10, int(100 * units))
        self._fit(self.net, task, steps, rng, g["lr"], g["wd"])
        cur = self._holdout_loss(self.net, task)
        for _ in range(g["rounds"]):
            best, best_loss, best_note, before = None, cur, None, cur
            for _c in range(g["cands"]):
                c = self.net.clone()
                note = c.mutate(rng)
                if note is None:
                    continue
                self._fit(c, task, g["ft_steps"], rng, g["lr"], g["wd"])
                loss = self._holdout_loss(c, task)
                if np.isfinite(loss) and loss < best_loss - 0.01 * abs(cur) - 1e-9:
                    best, best_loss, best_note = c, loss, note
            self.rounds_done += 1
            if best is not None:
                self.net, cur = best, best_loss
                self.rounds_changed += 1
                self.history.append(best_note)
                self.loss_trail.append([float(before), float(best_loss)])
                self._fit(self.net, task, max(5, steps // 4), rng, g["lr"], g["wd"])
                cur = self._holdout_loss(self.net, task)
        self.spent += units

    # ---- scoring
    def predict(self, X):
        z = self.net.forward(X)
        return z.argmax(axis=1) if self.kind == "classification" else z[:, 0] * self.ysd + self.ymu

    def outputs(self, X):
        z = self.net.forward(X)
        if self.kind == "classification":
            e = np.exp(z - z.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        return z[:, 0] * self.ysd + self.ymu

    def _metric(self, X, y, task):
        with np.errstate(all="ignore"):
            p = self.predict(X)
        if self.kind == "classification":
            acc = float((p == y).mean())
            return _clf_skill(acc, task.chance()), dict(acc=acc)
        r2 = _r2(p, y)
        return float(np.clip(r2, -1.0, 1.0)), dict(r2=r2)

    def val_score(self, task):
        return self._metric(task.Xval, task.yval, task)

    def test_metrics(self, task):
        return self._metric(task.Xte, task.yte, task)[1]

    def n_params(self):
        return self.net.n_params()

    def describe(self):
        return f"topology network: {self.net.describe()}; {self.rounds_changed} of {self.rounds_done} architecture rounds changed it" + (f" (last: {self.history[-1]})" if self.history else "")

    def save(self, d):
        arrays = {f"p.{k}": v for k, v in self.net.p.a.items()}
        meta = dict(genome=self.genome, kind=self.kind, ymu=self.ymu, ysd=self.ysd, spent=self.spent, spec=self.net.spec(), rounds_done=self.rounds_done,
                    rounds_changed=self.rounds_changed, history=self.history[-50:], loss_trail=self.loss_trail[-50:], seed=self.seed)
        arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8)
        fileio.save_npz(os.path.join(d, "agent.npz"), arrays)


class TopologyFamily(Family):
    name, kinds = "topology_net", ("classification", "regression")
    space = {"init_width": Gene("int", 3, 24, shape=True), "lr": Gene("log", 3e-3, 5e-2), "wd": Gene("log", 1e-6, 1e-2),
             "rounds": Gene("int", 2, 6), "cands": Gene("int", 2, 4), "ft_steps": Gene("int", 15, 50)}

    def build(self, task, genome, seed):
        return TopologyAgent(task, genome, seed)

    def load(self, d, task, genome):
        w = fileio.load_npz(os.path.join(d, "agent.npz"))
        meta = json.loads(bytes(w["__meta__"]).decode("utf-8"))
        net = TopologyNet.from_spec(meta["spec"], {k[2:]: w[k] for k in w if k.startswith("p.")}, seed=meta["seed"])
        a = TopologyAgent(task, meta["genome"], meta["seed"], net=net)
        a.ymu, a.ysd, a.spent = meta["ymu"], meta["ysd"], meta["spent"]
        a.rounds_done, a.rounds_changed, a.history, a.loss_trail = meta["rounds_done"], meta["rounds_changed"], meta["history"], meta.get("loss_trail", [])
        return a


_f = TopologyFamily()
FAMILIES[_f.name] = _f
