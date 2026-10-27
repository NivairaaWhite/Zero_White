"""families.py - every kind of agent the Generalist can build.

Each family wraps a model class that already exists in Zero White (or, for the two *program* families,
one that is written as Python source and evolved). A family exposes:

    kinds     which task kinds it applies to
    space     its genome: name -> Gene (see evolve.py). THERE IS NO DEFAULT GENOME ANYWHERE: a genome comes
              from the learned prior (prior.py), which starts out uniform-random and learns from results.
    build()   genome -> a fresh, untrained agent
    load()    restore a saved agent

and every agent exposes train(task, units, rng), val_score(task) -> (skill, metrics), test_metrics(task),
save(dir), describe(). `skill` is a number in [-1, 1] that is comparable between candidates on ONE task
(0 = no better than chance/random, 1 = solved/perfect). Validation data / selection episodes pick winners;
test data / fresh episodes are only ever reported.
"""

import copy
import json
import os

import numpy as np

from .. import checkpoint, factory, fileio
from ..evolve import Gene
from ..engine import Tensor, mse_loss, softmax_cross_entropy
from ..nn import Adam, Params, init_mlp, mlp, mlp_raw
from ..selfmod import astmut as A
from ..selfmod import egraph
from ..selfmod.islands import ProgramEvolver
from . import reduce
from .reduce import Projector
from .latent import LatentEncoder
from .obsenc import VQEncoder, choose_encoder

SELECT_SEED, TEST_SEED = 12345, 777000


class Agent:
    family = ""
    genome = None
    spent = 0.0

    def train(self, task, units, rng):
        raise NotImplementedError

    def val_score(self, task):
        raise NotImplementedError

    def test_metrics(self, task):
        raise NotImplementedError

    def save(self, d):
        raise NotImplementedError

    def describe(self):
        return self.family

    def n_params(self):
        return 0


class Family:
    name = ""
    kinds = ()
    space = {}

    def applicable(self, task):
        return task.kind in self.kinds

    def build(self, task, genome, seed):
        raise NotImplementedError

    def load(self, d, task, genome):
        raise NotImplementedError


def _clf_skill(acc, chance):
    return float(np.clip((acc - chance) / max(1e-9, 1.0 - chance), -1.0, 1.0))


def _r2(pred, y):
    return float(1.0 - np.mean((np.asarray(pred) - y) ** 2) / (np.var(y) + 1e-12))


# =========================================================================== MLP (classification + regression)
class MLPAgent(Agent):
    family = "mlp"

    def __init__(self, task, genome, seed):
        self.genome, self.kind = dict(genome), task.kind
        rng = np.random.default_rng(seed)
        out = task.n_classes if task.kind == "classification" else 1
        self.p = Params()
        init_mlp(self.p, "m", [task.in_dim] + [genome["hidden"]] * genome["depth"] + [out], rng, out_scale=0.5)
        self.opt = Adam(lr=genome["lr"], weight_decay=genome["wd"])
        self.ymu = self.ysd = None
        if task.kind == "regression":
            self.ymu, self.ysd = float(task.ytr.mean()), float(task.ytr.std() + 1e-9)
        self.spent = 0.0

    def train(self, task, units, rng):
        X, y = task.Xtr, task.ytr
        bs = min(self.genome["batch"], len(X))
        yt = y if self.kind == "classification" else ((y - self.ymu) / self.ysd)
        for _ in range(max(5, int(150 * units))):
            idx = rng.integers(0, len(X), size=bs)
            l = self.p.leaves()
            out = mlp(l, "m", Tensor(X[idx]))
            loss = softmax_cross_entropy(out, yt[idx])[0] if self.kind == "classification" \
                else mse_loss(out, yt[idx][:, None])
            loss.backward()
            self.opt.step(self.p, l)
        self.spent += units

    def predict(self, X):
        z = mlp_raw(self.p.a, "m", X)
        return z.argmax(axis=1) if self.kind == "classification" else z[:, 0] * self.ysd + self.ymu

    def _metric(self, X, y, task):
        with np.errstate(all="ignore"):
            p = self.predict(X)
        if self.kind == "classification":
            acc = float((p == y).mean())
            return _clf_skill(acc, task.chance()), dict(acc=acc)
        r2 = _r2(p, y)
        return float(np.clip(r2, -1.0, 1.0)), dict(r2=r2)

    def outputs(self, X):
        """6.5: soft outputs (class probabilities, or the regression prediction) for the Autonomy layer's router over finished agents (moa.py)."""
        z = mlp_raw(self.p.a, "m", X)
        if self.kind == "classification":
            e = np.exp(z - z.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        return z[:, 0] * self.ysd + self.ymu

    def val_score(self, task):
        return self._metric(task.Xval, task.yval, task)

    def test_metrics(self, task):
        return self._metric(task.Xte, task.yte, task)[1]

    def n_params(self):
        return self.p.count()

    def describe(self):
        h, d = self.genome["hidden"], self.genome["depth"]
        n_in = self.p.a["m.W0"].shape[0]
        return f"MLP on {n_in} inputs: {d} hidden layer(s) x {h} units, lr={self.genome['lr']:.4g}, {self.p.count()} parameters"

    def save(self, d):
        arrays = {f"p.{k}": v for k, v in self.p.a.items()}
        meta = dict(genome=self.genome, kind=self.kind, ymu=self.ymu, ysd=self.ysd, spent=self.spent)
        arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8)
        fileio.save_npz(os.path.join(d, "agent.npz"), arrays)


class MLPFamily(Family):
    name, kinds = "mlp", ("classification", "regression")
    space = {"hidden": Gene("int", 4, 128, shape=True), "depth": Gene("int", 1, 3, shape=True),
             "lr": Gene("log", 1e-3, 5e-2), "wd": Gene("log", 1e-6, 1e-2), "batch": Gene("int", 16, 128)}

    def build(self, task, genome, seed):
        return MLPAgent(task, genome, seed)

    def load(self, d, task, genome):
        w = fileio.load_npz(os.path.join(d, "agent.npz"))
        meta = json.loads(bytes(w["__meta__"]).decode("utf-8"))
        a = MLPAgent(task, meta["genome"], 0)
        a.p = Params({k[2:]: w[k] for k in w if k.startswith("p.")})
        a.ymu, a.ysd, a.spent = meta["ymu"], meta["ysd"], meta["spent"]
        return a


# =========================================================================== self-modifying classifier
class SelfModAgent(Agent):
    family = "selfmod_net"

    def __init__(self, task, genome, seed, zw=None):
        self.genome = dict(genome)
        self.zw = zw or factory.make_classifier(task, seed=seed, hidden=genome["hidden0"], candidates=genome["candidates"])
        self.spent = 0.0

    def train(self, task, units, rng):
        self.zw.run(task.Xtr, task.ytr, task.Xval[:2000], task.yval[:2000], generations=max(1, int(4 * units)),
                    probation_steps=self.genome["probation"], verbose=False)
        self.spent += units

    def predict(self, X):
        return self.zw.predict(X)

    def _metric(self, X, y, task):
        acc = float(self.zw.evaluate(self.zw.net, X, y)[1])
        return _clf_skill(acc, task.chance()), dict(acc=acc)

    def val_score(self, task):
        return self._metric(task.Xval, task.yval, task)

    def test_metrics(self, task):
        return self._metric(task.Xte, task.yte, task)[1]

    def n_params(self):
        return self.zw.net.num_params()

    def describe(self):
        n = self.zw.net
        return (f"self-modifying net: hidden={n.hidden_sizes()} patch={n.use_patch} params={n.num_params()} "
                f"after {self.zw.self_model.generation} self-modification generations")

    def save(self, d):
        checkpoint.save(self.zw, d)


class SelfModFamily(Family):
    name, kinds = "selfmod_net", ("classification",)
    space = {"hidden0": Gene("int", 4, 32, shape=True), "candidates": Gene("int", 3, 6),
             "probation": Gene("int", 8, 30)}

    def build(self, task, genome, seed):
        return SelfModAgent(task, genome, seed)

    def load(self, d, task, genome):
        a = SelfModAgent.__new__(SelfModAgent)
        a.genome, a.zw, a.spent = dict(genome), checkpoint.load(d)[0], 0.0
        return a


# =========================================================================== actor-critic (environments)
class PGAgentWrap(Agent):
    family = "actor_critic"

    def __init__(self, task, genome, seed, agent=None):
        from ..agents import PGAgent
        self.genome = dict(genome)
        self.agent = agent or PGAgent(task.obs_dim, task.n_actions, seed=seed, **genome)
        self.spent = 0.0

    def train(self, task, units, rng):
        self.agent.train(task.env, updates=max(2, int(15 * units)), episodes=8, rng=rng)
        self.spent += units

    def _ret(self, task, episodes, seed):
        ret, succ = self.agent.evaluate(task.env, episodes, seed=seed)
        return ret, succ

    def val_score(self, task):
        ret, succ = self._ret(task, 20, SELECT_SEED)
        return task.skill(ret), dict(mean_return=ret, success=succ)

    def test_metrics(self, task):
        ret, succ = self._ret(task, 30, TEST_SEED)
        return dict(mean_return=ret, success=succ)

    def n_params(self):
        return self.agent.p.count()

    def describe(self):
        g = self.genome
        return (f"actor-critic: {g['depth']}x{g['hidden']} policy+value nets, lr={g['lr']:.3g}, gamma={g['gamma']:.3f}, "
                f"entropy={g['entropy']:.3g}, curiosity={g['curiosity']:.2f}, {self.agent.updates_trained} updates")

    def save(self, d):
        self.agent.save(os.path.join(d, "agent.npz"))


class PGFamily(Family):
    name, kinds = "actor_critic", ("env",)
    space = {"hidden": Gene("int", 8, 96, shape=True), "depth": Gene("int", 1, 2, shape=True),
             "lr": Gene("log", 1e-3, 3e-2), "gamma": Gene("float", 0.9, 0.995),
             "entropy": Gene("log", 1e-3, 0.1), "curiosity": Gene("float", 0.0, 1.0)}

    def build(self, task, genome, seed):
        return PGAgentWrap(task, genome, seed)

    def load(self, d, task, genome):
        from ..agents import PGAgent
        a = PGAgentWrap.__new__(PGAgentWrap)
        a.agent, a.genome, a.spent = PGAgent.load(os.path.join(d, "agent.npz")), dict(genome), 0.0
        return a


# =========================================================================== evolved PROGRAMS
_PROGRAM_SPACE = {"n_islands": Gene("int", 2, 5), "island_size": Gene("int", 4, 16), "tune_steps": Gene("int", 0, 24),
                  "max_nodes": Gene("int", 40, 140, shape=True), "n_mut": Gene("int", 1, 3)}


def _recalled_seeds(task, vocab, seed):
    """6.5: programs the episodic memory recalled from similar tasks, adapted to this task and spliced (episodic.py). [] when the task carries no memory (every older code path)."""
    mem = getattr(task, "_episodic", None)
    if mem is None:
        return []
    try:
        return mem.seeds_for(task, vocab, np.random.default_rng(seed + 31), recalled=getattr(task, "_recalled", None))
    except Exception:                                                                      # noqa: BLE001 - a memory problem must never stop a solve
        return []


def _to_action(s, n):
    try:
        s = float(s)
    except (TypeError, ValueError):
        return 0
    if s != s:
        return 0
    return int(s > 0.0) if n == 2 else int(np.floor(s)) % n


def _proj_label(proj) -> str:
    return "attention-encoder code" if isinstance(proj, LatentEncoder) else "vector-quantised code" if isinstance(proj, VQEncoder) else "whitened PCA projection"


def _proj_key(proj) -> str:
    return "latent" if isinstance(proj, LatentEncoder) else "vq" if isinstance(proj, VQEncoder) else "projector"


class ProgramBase(Agent):
    """An agent whose 'weights' are Python source. State = the island populations of a ProgramEvolver."""

    def _make_evolver(self, vocab, score_fn, probe_fn, seed):
        g = self.genome
        return ProgramEvolver(vocab, score_fn, [A.seed_program(vocab, self._seed_body)] + list(getattr(self, "_memory_seeds", [])), np.random.default_rng(seed),
                              probe_fn=probe_fn, n_islands=g["n_islands"], island_size=g["island_size"],
                              tune_steps=g["tune_steps"], max_mut=g["n_mut"])

    def best_src(self):
        pin = getattr(self, "_pinned", None)
        if pin:                                                           # 6.5: a program the architect (architect.py) validated on held-out data; see adopt()
            return pin
        return self.pe.best.src if self.pe.best else A.seed_program(self.vocab, self._seed_body)

    def adopt(self, src):
        """6.5: make `src` THE program (the architect proved it better than the incumbent on held-out pairs). It also joins every island, so the search keeps building on it, and it stays
        the answer until the search itself finds something with a strictly better selection score (`_unpin_if_beaten`)."""
        src = A.normalize(src)
        self._pinned = src
        for isl in self.pe.islands:
            self.pe._admit(isl, src)
        return self

    def _unpin_if_beaten(self):
        pin = getattr(self, "_pinned", None)
        if pin and self.pe.best is not None and self.pe.best.score > self.pe.seen.get(A.normalize(pin), -1e12) + 1e-9 and A.normalize(self.pe.best.src) != A.normalize(pin):
            self._pinned = None

    def describe(self):
        return "evolved program:\n" + "\n".join("    " + ln for ln in self._clean(self.best_src()).strip().splitlines())

    @staticmethod
    def _clean(src):
        """Constant folding and dead temporaries (astmut), then equality saturation (egraph, 6.3): shown and saved, never what runs."""
        out = A.simplify(src)
        try:
            return egraph.simplify(out)
        except Exception:                                                       # noqa: BLE001 - cosmetics must never lose a program
            return out

    def n_params(self):
        return len(A.constants_of(self.best_src()))

    def _extra_state(self):
        return {}

    def _save_state(self, d):
        fileio.write_json(os.path.join(d, "state.json"), dict(genome=self.genome, spent=self.spent, pinned=getattr(self, "_pinned", None), **self._extra_state(), **self.pe.state()))
        header = f'"""Evolved by Zero White for task {self.task_name!r} (family {self.family}). Namespace: np, math, ' \
                 f'pdiv, psqrt, plog, pexp, sel, relu, sig, sgn."""\n\n\n' \
                 + (f'# obs is the {self.proj.k}-number {_proj_label(self.proj)} of the raw observation '
                    f'(see state.json "{_proj_key(self.proj)}").\n\n\n'
                    if getattr(self, "proj", None) is not None else "")
        fileio.write_text(os.path.join(d, "program.py"), header + self._clean(self.best_src()))

    def _restore_state(self, d):
        st = fileio.read_json(os.path.join(d, "state.json"))
        self.pe.restore(st)
        self.spent = st.get("spent", 0.0)
        self._pinned = st.get("pinned")
        if st.get("projector") is not None:                               # the SAME components the program was evolved on, not a refit
            self.proj = Projector.from_dict(st["projector"])
        elif st.get("latent") is not None:
            self.proj = LatentEncoder.from_dict(st["latent"])
        elif st.get("vq") is not None:
            self.proj = VQEncoder.from_dict(st["vq"])
        self.encoder_choice = st.get("encoder_choice")


class ProgramPolicyAgent(ProgramBase):
    family = "program_policy"
    _seed_body = "0.0"
    project_high_dim = True            # 6.3: observations wider than 24 are projected (reduce.py) instead of truncated; False = the old behaviour
    encoder = "pca"                    # 6.4: "pca" (default) or "latent": a Perceiver-style attention bottleneck (latent.py) for observations that PCA describes badly.
                                       # 6.5: "vq" (a residual-vector-quantised autoencoder, obsenc.py) or "auto" (fit all three, keep the one whose code best predicts this task's value and dynamics)

    def __init__(self, task, genome, seed):
        self.genome, self.task_name = dict(genome), task.name
        self.n_actions = task.n_actions
        self.proj = None
        project = self.project_high_dim and reduce.needed(task.obs_dim)
        rng = np.random.default_rng(seed + 99)
        probe_obs = []
        o = task.env.reset(rng)
        use_latent = project and self.encoder == "latent"
        use_vq = project and self.encoder == "vq"
        self.encoder_choice = None
        for _ in range(800 if (use_latent or use_vq) else (256 if project else 48)):  # a projector needs more samples than the 48-step probe; an attention encoder more still
            probe_obs.append(o)
            o, _, done, _ = task.env.step(int(rng.integers(task.n_actions)))
            if done:
                o = task.env.reset(rng)
        if project:
            try:
                if project and self.encoder == "auto":
                    ch = choose_encoder(task, k=8, seed=seed)
                    self.proj, self.encoder_choice = ch.encoder, ch.report()
                elif use_vq:
                    self.proj = VQEncoder.fit(probe_obs, k=8, seed=seed)
                else:
                    self.proj = (LatentEncoder.fit(probe_obs, k=8, seed=seed, epochs=60) if use_latent else Projector.fit(probe_obs, reduce.MAX_COLS))
            except ValueError:
                self.proj = None                                          # too few observations: fall back to the first columns
        n_cols = self.proj.k if self.proj is not None else min(task.obs_dim, 24)
        self.vocab = A.Vocab("policy", ("obs",), col_var="obs", n_cols=n_cols, vector=False,
                             max_nodes=genome["max_nodes"], leaves=())
        self._probe_obs = probe_obs[:48]
        self._memory_seeds = _recalled_seeds(task, self.vocab, seed)
        self._sel_episodes = 5
        self._task = task

        def score_fn(fn, src):
            act = lambda ob: _to_action(fn(self._view(ob)), self.n_actions)       # noqa: E731
            rets = [task.rollout(act, 20000 + i)[0] for i in range(self._sel_episodes)]
            return task.skill(float(np.mean(rets))) - 1e-3 * A.count_nodes(src) / 10

        def probe_fn(fn):
            return tuple(_to_action(fn(self._view(o)), self.n_actions) for o in self._probe_obs)

        self.pe = self._make_evolver(self.vocab, score_fn, probe_fn, seed)

    def train(self, task, units, rng):
        self.pe.run(max(8, int(80 * units)))
        self._unpin_if_beaten()
        self.spent += units

    def _view(self, o):
        """What the program reads: the projected components when the observation is wider than 24, else the raw observation."""
        return self.proj(o) if self.proj is not None else o

    def _extra_state(self):
        if self.proj is None:
            return {}
        out = {_proj_key(self.proj): self.proj.to_dict()}
        if getattr(self, "encoder_choice", None):
            out["encoder_choice"] = self.encoder_choice
        return out

    def _act_fn(self):
        fn = A.compile_program(self.best_src(), "policy", **self.vocab.safety)
        return lambda o: _to_action(fn(self._view(o)), self.n_actions)

    def val_score(self, task):
        ret, succ = task.evaluate_fn(self._act_fn(), 20, SELECT_SEED)
        return task.skill(ret), dict(mean_return=ret, success=succ)

    def test_metrics(self, task):
        ret, succ = task.evaluate_fn(self._act_fn(), 30, TEST_SEED)
        return dict(mean_return=ret, success=succ)

    def save(self, d):
        self._save_state(d)


class ProgramPolicyFamily(Family):
    name, kinds = "program_policy", ("env",)
    space = dict(_PROGRAM_SPACE)

    def build(self, task, genome, seed):
        return ProgramPolicyAgent(task, genome, seed)

    def load(self, d, task, genome):
        a = ProgramPolicyAgent(task, genome, 0)
        a._restore_state(d)
        return a


class ProgramRegAgent(ProgramBase):
    """Symbolic regression: evolve a formula over the most informative input columns."""
    family = "program_formula"
    _seed_body = "0.0"

    def __init__(self, task, genome, seed):
        self.genome, self.task_name = dict(genome), task.name
        corr = np.abs([np.corrcoef(task.Xtr[:, j], task.ytr)[0, 1] if task.Xtr[:, j].std() > 1e-9 else 0.0
                       for j in range(task.in_dim)])
        corr = np.nan_to_num(corr)
        self.cols = sorted(int(c) for c in np.argsort(-corr)[:min(task.in_dim, 12)])
        self.vocab = A.Vocab("predict", ("x",), col_var="x", n_cols=len(self.cols), vector=True,
                             max_nodes=genome["max_nodes"])
        self._memory_seeds = _recalled_seeds(task, self.vocab, seed)
        X, y = task.Xtr[:, self.cols], task.ytr
        self._var = float(np.var(y) + 1e-12)
        rng = np.random.default_rng(seed + 5)
        pidx = rng.choice(len(X), size=min(24, len(X)), replace=False)
        px = X[pidx]

        def score_fn(fn, src):
            p = np.broadcast_to(np.asarray(fn(X), dtype=float), y.shape)
            return -float(np.mean((p - y) ** 2)) / self._var - 1e-3 * A.count_nodes(src) / 10

        def probe_fn(fn):
            return np.broadcast_to(np.asarray(fn(px), dtype=float), (len(px),))

        self.pe = self._make_evolver(self.vocab, score_fn, probe_fn, seed)

    def train(self, task, units, rng):
        self.pe.run(max(40, int(900 * units)))
        self._unpin_if_beaten()
        self.spent += units

    def predict(self, X, src=None):
        fn = A.compile_program(src or self.best_src(), "predict", **self.vocab.safety)
        with np.errstate(all="ignore"):
            return np.broadcast_to(np.asarray(fn(X[:, self.cols]), dtype=float), (len(X),))

    def _best_by_val(self, task):
        best = None
        srcs = [getattr(self, "_pinned", None)] if getattr(self, "_pinned", None) else [m.src for m in self.pe.top(3)]
        for src in srcs:                                              # guard against fitting the train split by luck (6.5: a program the architect validated stands alone)
            try:
                r2 = _r2(self.predict(task.Xval, src), task.yval)
            except Exception:                                         # noqa: BLE001
                continue
            if np.isfinite(r2) and (best is None or r2 > best[0]):
                best = (r2, src)
        return best

    def val_score(self, task):
        b = self._best_by_val(task)
        if b is None:
            return -1.0, dict(r2=float("-inf"))
        self._chosen = b[1]
        return float(np.clip(b[0], -1.0, 1.0)), dict(r2=b[0])

    def best_src(self):
        return getattr(self, "_pinned", None) or getattr(self, "_chosen", None) or super().best_src()

    def test_metrics(self, task):
        self.val_score(task)
        return dict(r2=_r2(self.predict(task.Xte), task.yte), columns=self.cols)

    def describe(self):
        return (f"evolved formula over input columns {self.cols}:\n" +
                "\n".join("    " + ln for ln in self.best_src().strip().splitlines()))

    def save(self, d):
        self._save_state(d)
        st = fileio.read_json(os.path.join(d, "state.json"))
        st["cols"] = self.cols
        fileio.write_json(os.path.join(d, "state.json"), st)


class ProgramRegFamily(Family):
    name, kinds = "program_formula", ("regression",)
    space = dict(_PROGRAM_SPACE)

    def build(self, task, genome, seed):
        return ProgramRegAgent(task, genome, seed)

    def load(self, d, task, genome):
        a = ProgramRegAgent(task, genome, 0)
        a._restore_state(d)
        return a


# =========================================================================== generative
class CharLMAgent(Agent):
    family = "char_lm"

    def __init__(self, task, genome, seed, lm=None):
        from ..generative import CharLM
        self.genome = dict(genome)
        self.lm = lm or CharLM(len(task.vocab), seed=seed, **genome)
        self.spent = 0.0

    def train(self, task, units, rng):
        self.lm.fit(task.train, max(20, int(250 * units)), rng=rng)
        self.spent += units

    def val_score(self, task):
        bpc = self.lm.bits_per_char(task.val)
        return float(np.clip(1.0 - bpc / task.chance(), -1.0, 1.0)), dict(bits_per_char=bpc, uniform=task.chance())

    def test_metrics(self, task):
        return self.val_score(task)[1]

    def n_params(self):
        return self.lm.p.count()

    def sample(self, task, prompt="", n=200, temperature=0.8, seed=0):
        return self.lm.generate(task.vocab, prompt, n, temperature, np.random.default_rng(seed))

    def describe(self):
        g = self.genome
        return f"character LM: context {g['ctx']}, embedding {g['emb']}, {g['depth']}x{g['hidden']} hidden, {self.lm.p.count()} parameters"

    def save(self, d):
        # vocab is stored by the zoo via the task; the model file carries its own copy for standalone loading
        from ..generative import CharVocab
        self.lm.save(os.path.join(d, "lm.npz"), getattr(self, "_vocab", CharVocab(["\x00"])))


class CharLMFamily(Family):
    name, kinds = "char_lm", ("text",)
    space = {"ctx": Gene("int", 4, 24, shape=True), "emb": Gene("int", 8, 48, shape=True),
             "hidden": Gene("int", 32, 256, shape=True), "depth": Gene("int", 1, 2, shape=True),
             "lr": Gene("log", 5e-4, 1e-2)}

    def build(self, task, genome, seed):
        a = CharLMAgent(task, genome, seed)
        a._vocab = task.vocab
        return a

    def load(self, d, task, genome):
        from ..generative import CharLM
        lm, vocab = CharLM.load(os.path.join(d, "lm.npz"))
        a = CharLMAgent(task, genome, 0, lm=lm)
        a._vocab = vocab
        return a


class CVAEAgent(Agent):
    family = "cvae"

    def __init__(self, task, genome, seed, model=None):
        from ..generative import CVAE
        self.genome = dict(genome)
        self.model = model or CVAE(task.sup.in_dim, task.sup.n_classes, seed=seed, **genome)
        self.spent = 0.0

    def train(self, task, units, rng):
        self.model.fit(task.sup.Xtr, task.sup.ytr, max(20, int(150 * units)), rng=rng)
        self.spent += units

    def val_score(self, task):
        from ..generative import score_generator
        _, info = score_generator(self.model, task.sup, task.classifier.predict)
        return float(0.5 * info["fidelity"] + 0.5 * info["diversity"]), dict(
            fidelity=info["fidelity"], diversity=info["diversity"], elbo=info["elbo"])

    def test_metrics(self, task):
        return self.val_score(task)[1]

    def n_params(self):
        return self.model.p.count()

    def describe(self):
        g = self.genome
        return f"class-conditional VAE: latent {g['latent']}, {g['depth']}x{g['hidden']} hidden, beta={g['beta']:.3g}"

    def save(self, d):
        self.model.save(os.path.join(d, "cvae.npz"))


class CVAEFamily(Family):
    name, kinds = "cvae", ("image_gen",)
    space = {"latent": Gene("int", 2, 32, shape=True), "hidden": Gene("int", 16, 256, shape=True),
             "depth": Gene("int", 1, 2, shape=True), "beta": Gene("log", 0.1, 2.0), "lr": Gene("log", 3e-4, 1e-2)}

    def build(self, task, genome, seed):
        return CVAEAgent(task, genome, seed)

    def load(self, d, task, genome):
        from ..generative import CVAE
        return CVAEAgent(task, genome, 0, model=CVAE.load(os.path.join(d, "cvae.npz")))


FAMILIES = {f.name: f for f in (MLPFamily(), SelfModFamily(), PGFamily(), ProgramPolicyFamily(), ProgramRegFamily(),
                                CharLMFamily(), CVAEFamily())}


# 6.5: families that live in their own modules. Each of them imports this one and registers itself in FAMILIES when it finishes loading, so the order of the imports does not matter.
from . import moa as _moa                                                              # noqa: E402,F401
from . import topology as _topology                                                  # noqa: E402,F401


def families_for(task):
    return [f for f in FAMILIES.values() if f.applicable(task)]
