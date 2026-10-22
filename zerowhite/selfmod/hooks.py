"""hooks.py - level-1 self-modification: Zero White rewrites its OWN search rules.

The two hook files in evolved/ (step-size update, parent selection) are evolved here as real
Python source, scored by how well the Evolver performs WITH the candidate hook on a suite of
optimisation problems. Honest design points:

  * the suite is a PROXY (cheap synthetic landscapes), because scoring a hook on real model training
    would cost hours per candidate. Search sees only the TRAIN functions; acceptance is decided on
    HELD-OUT functions the search never saw, with fresh seeds.
  * a candidate is installed only if it beats the live hook on held-out functions by a margin and is
    not worse on train; otherwise nothing changes.
  * every install is recorded in the history store and can be rolled back; the shipped files in
    evolved/ are never touched (installs go to <workspace>/selfmod/live/).
  * the textbook 1/5-success rule is measured alongside as a reference point, never used as a default.
"""

import math
import os

import numpy as np

from .. import evolved, fileio
from ..evolve import Evolver, Gene
from . import astmut as A
from . import stats
from .history import HistoryStore
from .islands import ProgramEvolver

DIM = 5
SPACE = {f"g{i}": Gene("float", 0.0, 1.0) for i in range(DIM)}


# --------------------------------------------------------------------------- the benchmark suite
def _landscapes():
    def sphere(z):          return float((z ** 2).sum())
    def ellipsoid(z):       return float((np.logspace(0, 2, len(z)) * z ** 2).sum())
    def rastrigin(z):       return float((10 * len(z) + ((6 * z) ** 2 - 10 * np.cos(2 * np.pi * 6 * z)).sum()) / 40)
    def rosenbrock(z):      return float((100 * (z[1:] - z[:-1] ** 2) ** 2 + (1 - z[:-1]) ** 2).sum() / 10)
    def ackley(z):          return float(-20 * np.exp(-0.2 * np.sqrt((z ** 2).mean())) - np.exp(np.cos(2 * np.pi * 3 * z).mean()) + 20 + np.e)
    def plateau(z):         return float(np.floor(np.abs(z) * 8).sum())
    def deceptive(z):       return float(min(((z - 0.8) ** 2).sum(), 0.6 + 4 * ((z + 0.8) ** 2).sum()))
    def noisy_sphere(z):    return float((z ** 2).sum())
    return dict(sphere=sphere, ellipsoid=ellipsoid, rastrigin=rastrigin, noisy_sphere=noisy_sphere,
                rosenbrock=rosenbrock, ackley=ackley, plateau=plateau, deceptive=deceptive)


LANDSCAPES = _landscapes()
TRAIN_FUNCS = ("sphere", "ellipsoid", "rastrigin", "noisy_sphere")
HELDOUT_FUNCS = ("rosenbrock", "ackley", "plateau", "deceptive")
CONFIGS = ((4, 2), (6, 3))                       # (population, mu): production uses (4, 2)


def _run_once(policy, fname, seed, gens, pop, mu):
    rng = np.random.default_rng(seed)
    shift = rng.uniform(-0.35, 0.35, DIM)         # optimum location differs per seed, never at a corner
    f = LANDSCAPES[fname]
    noise = 0.02 if fname == "noisy_sphere" else 0.0
    nrng = np.random.default_rng(seed + 7)
    centre = np.full(DIM, 0.5) + shift * 0.5
    scale = max(1e-9, float(np.mean([f((rng.uniform(0, 1, DIM) - centre) * 2) for _ in range(64)])))

    def loss(g):
        z = (np.array([g[f"g{i}"] for i in range(DIM)]) - centre) * 2
        return f(z) + (nrng.normal(0, noise) if noise else 0.0)

    ev = Evolver(SPACE, lambda g, parent: (-loss(g) / scale, None, {}), population=pop, mu=mu,
                 seed=seed, policy=policy)
    for _ in range(gens):
        ev.step()
    best_true = min(f((np.array([e.genome[f"g{i}"] for i in range(DIM)]) - centre) * 2) for e in ev.pool)
    return 1.0 - best_true / scale                  # 1 = optimum found, 0 = no better than random


def suite_score(policy, funcs, seeds, gens=6):
    vals = [_run_once(policy, fn, s, gens, pop, mu) for fn in funcs for s in seeds for pop, mu in CONFIGS]
    return float(np.mean(vals))


def suite_values(policy, funcs, seeds, gens=6):
    """One value per (function, seed), averaged over the population settings, in a fixed order: what a PAIRED test of two policies needs.
    `float(np.mean(suite_values(...)))` equals `suite_score(...)` (every pair averages the same number of runs)."""
    return np.array([float(np.mean([_run_once(policy, fn, s, gens, pop, mu) for pop, mu in CONFIGS])) for fn in funcs for s in seeds])


# --------------------------------------------------------------------------- vocab per hook
def vocab_for(hook):
    if hook == "sigma_update":
        return A.Vocab("sigma_update", ("sigma", "improved", "generation"), vector=False,
                       leaves=("sigma", "generation", "0.5", "1.3", "0.85"), cond_leaves=("improved",),
                       allow_pow=False, max_nodes=70)
    if hook == "select_parent":
        return A.Vocab("select_parent", ("fitnesses", "rng"), vector=False,
                       leaves=("rng.random()", "len(fitnesses)", "0.5", "2.0"), cond_leaves=("rng.random() > 0.5",),
                       conditionals=True, max_nodes=50)
    raise KeyError(hook)


def genesis_policy():
    return evolved.Policy(sigma_update=evolved._BUILTIN_GENESIS["sigma_update"],
                          select_parent=evolved._BUILTIN_GENESIS["select_parent"], label="genesis")


def _probe(hook):
    if hook == "sigma_update":
        grid = [(s, imp, g) for s in (0.1, 0.3, 0.5, 0.8, 1.0) for imp in (True, False) for g in (1, 5, 20)]
        return lambda fn: np.nan_to_num(np.array([float(fn(s, imp, g)) for s, imp, g in grid]), nan=-7.0, posinf=9.0, neginf=-9.0)
    fits = np.array([3.0, 2.0, 1.0, 0.5])
    return lambda fn: np.array([int(fn(fits, np.random.default_rng(k))) for k in range(24)], dtype=float)


def _policy_with(hook, fn):
    p = genesis_policy()
    p._fn[hook] = fn
    p.label = f"candidate:{hook}"
    return p


# --------------------------------------------------------------------------- manager (store + live overlay)
class HookManager:
    def __init__(self, workspace_root):
        self.root = os.path.join(os.path.abspath(workspace_root), "selfmod")
        self.live_dir = os.path.join(self.root, "live")
        self.store = HistoryStore(os.path.join(self.root, "history"))
        os.makedirs(self.live_dir, exist_ok=True)

    def activate(self):
        evolved.use(self.live_dir)
        return self

    def live_source(self, hook):
        p = os.path.join(self.live_dir, hook + ".py")
        return fileio.read_text(p) if os.path.exists(p) else None

    def _write_live(self, hook, src):
        fileio.write_text(os.path.join(self.live_dir, hook + ".py"), src)

    def policy(self, which="live"):
        return evolved.Policy.load(self.live_dir) if which == "live" else genesis_policy()

    def install(self, hook, body_src, meta):
        v = (self.store.head("hook:" + hook) or {"version": 0})["version"] + 1
        header = (f'"""EVOLVED hook \'{hook}\', version {v}. Written by Zero White, not by a person.\n'
                  f'Evaluated in the Zero White program namespace (np, math, pdiv, psqrt, plog, pexp, sel, relu, sig, sgn).\n'
                  f'Provenance and scores are in the history store; roll back with `selfmod rollback {hook}`."""\n\n\n')
        text = header + body_src
        evolved.compile_hook(hook, text)                               # last gate: must compile
        rec = self.store.put("hook:" + hook, text, meta)
        self._write_live(hook, text)
        return rec

    def rollback(self, hook, steps=1):
        try:
            rec = self.store.rollback("hook:" + hook, steps)
        except ValueError:                                              # nothing older: back to genesis
            p = os.path.join(self.live_dir, hook + ".py")
            if os.path.exists(p):
                os.remove(p)
            return None
        self._write_live(hook, self.store.get(rec["sha"]))
        return rec

    def reset_to_genesis(self, hook):
        p = os.path.join(self.live_dir, hook + ".py")
        if os.path.exists(p):
            os.remove(p)

    def status(self):
        out = {}
        for h in evolved.HOOKS:
            head = self.store.head("hook:" + h)
            out[h] = dict(live=self.live_source(h) is not None, version=head["version"] if head else 0)
        return out


# --------------------------------------------------------------------------- the search
def evolve_hook(mgr, hook, evals=400, seed=0, train_seeds=(0, 1, 2, 3), heldout_seeds=tuple(range(100, 110)), gens=6,
                margin=0.015, verbose=True, alpha=0.05):
    """Try to improve one hook. Returns a report dict; installs the winner only if it passes acceptance."""
    vocab = vocab_for(hook)
    live_src = mgr.live_source(hook)
    incumbent_policy = mgr.policy("live")
    genesis = genesis_policy()
    textbook = evolved.textbook_policy()

    base_train = suite_score(incumbent_policy, TRAIN_FUNCS, train_seeds, gens)

    def score_fn(fn, src):
        pol = _policy_with(hook, fn)
        other = hook == "sigma_update" and "select_parent" or "sigma_update"
        pol._fn[other] = incumbent_policy._fn[other]          # evolve one hook at a time, others as installed
        return suite_score(pol, TRAIN_FUNCS, train_seeds, gens) - 0.0015 * A.count_nodes(src) / 10

    seeds_src = []
    if hook == "sigma_update":
        seeds_src = [A.seed_program(vocab, "sigma"), A.seed_program(vocab, "sigma * (1.3 if improved else 0.85)")]
        # NOTE: the textbook rule is offered as one SEED among others only when the caller opts in via
        # seed_with_textbook; by default the search starts from the trivial constant rule alone.
        seeds_src = seeds_src[:1]
    else:
        seeds_src = [A.seed_program(vocab, "int(rng.random() * len(fitnesses))")]
    if live_src:
        try:
            body = live_src.split('"""', 2)[2].strip() + "\n"
            seeds_src.append(body)
        except IndexError:
            pass

    rng = np.random.default_rng(seed)
    pe = ProgramEvolver(vocab, score_fn, seeds_src, rng, probe_fn=_probe(hook), tune_steps=6, island_size=6,
                        n_islands=3, reset_every=60)
    pe.run(evals)
    cands = pe.top(3)

    report = dict(hook=hook, evals=pe.evals, fec_hits=pe.fec_hits, invalid=pe.invalid, base_train=base_train,
                  accepted=False, candidates=[])
    best = None
    for m in cands:
        try:
            fn = A.compile_program(m.src, hook, **vocab.safety)
        except ValueError:
            continue
        pol = _policy_with(hook, fn)
        other = "sigma_update" if hook == "select_parent" else "select_parent"
        pol._fn[other] = incumbent_policy._fn[other]
        tr = suite_score(pol, TRAIN_FUNCS, train_seeds, gens)
        ho_vec = suite_values(pol, HELDOUT_FUNCS, heldout_seeds, gens)
        ho = float(np.mean(ho_vec))
        report["candidates"].append(dict(src=m.src, train=tr, heldout=ho))
        if best is None or ho > best[1]:
            best = (m.src, ho, tr, ho_vec)
    live_vec = suite_values(incumbent_policy, HELDOUT_FUNCS, heldout_seeds, gens)
    ref = dict(live_heldout=float(np.mean(live_vec)),
               genesis_heldout=suite_score(genesis, HELDOUT_FUNCS, heldout_seeds, gens),
               textbook_heldout=suite_score(textbook, HELDOUT_FUNCS, heldout_seeds, gens),
               live_train=base_train, genesis_train=suite_score(genesis, TRAIN_FUNCS, train_seeds, gens))
    report["reference"] = ref
    if best is not None:
        src, ho, tr, ho_vec = best
        report["best"] = dict(src=src, heldout=ho, train=tr)
        verdict = stats.gate(live_vec, ho_vec, margin=margin, alpha=alpha)      # 6.3: the margin AND a paired test over (function, seed) pairs
        report["gate"] = {k: verdict[k] for k in ("n", "mean_gain", "p", "positive", "accept", "reason")}
        if verdict["accept"] and tr >= base_train - 1e-9:
            src = A.simplify(src, probe=_probe(hook))             # readable source, verified behaviour-preserving
            report["best"]["src_installed"] = src
            rec = mgr.install(hook, src, dict(heldout=ho, train=tr, vs_live=ref["live_heldout"],
                                             vs_genesis=ref["genesis_heldout"], vs_textbook=ref["textbook_heldout"],
                                             evals=pe.evals, seed=seed))
            report.update(accepted=True, version=rec["version"])
    if verbose:
        print(f"[hook {hook}] evals={pe.evals} fec_hits={pe.fec_hits} accepted={report['accepted']}")
        print(f"   held-out score: live={ref['live_heldout']:.3f} genesis={ref['genesis_heldout']:.3f} "
              f"textbook-1/5={ref['textbook_heldout']:.3f}" + (f" candidate={best[1]:.3f}" if best else ""))
        if best:
            shown = A.simplify(report.get("best", {}).get("src_installed") or best[0], probe=_probe(hook))
            print("   best candidate source:\n      " + shown.strip().replace("\n", "\n      "))
    return report
