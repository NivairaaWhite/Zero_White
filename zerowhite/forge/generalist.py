"""generalist.py - the general model: hand it a task, it builds the agent(s) for that task.

solve(task) does, in order:

  1. PROFILE    what kind of task is this (supervised / env / text / image generation), how big?
  2. INCUMBENT  load this task's current champion from the zoo (if any) and re-score it: it stays in the race
                and is only ever replaced by something strictly better, so a task never gets worse.
  3. TRANSFER   try champions that were built for *other* tasks of the same kind directly on this one (POET-style
                direct transfer); if their shapes fit and they do well, they join the race.
  4. PROPOSE    ask the learned prior (prior.py) for new (family, genome) pairs. Day one the prior is empty,
                so the first agents are random: there is no default agent anywhere.
  5. RACE       successive halving: everyone trains a little, the weaker half is dropped, the rest train twice as
                long, and so on. Training is cumulative, so survivors keep what they learned.
  6. KEEP       the winner is written to the zoo (weights or source code, a spec.json and a readable description),
                the prior is updated from EVERY candidate's result, and the test split / fresh episodes are measured
                once, for reporting only.
"""

import math
import time

import numpy as np

from .. import factory
from . import task as T
from .families import FAMILIES, families_for
from .prior import AgentPrior
from .zoo import Zoo, _safe

BASE_UNITS = {"classification": 1.0, "regression": 1.0, "env": 1.0, "text": 1.0, "image_gen": 1.0}


class Candidate:
    def __init__(self, agent, origin, cid):
        self.agent, self.origin, self.cid = agent, origin, cid
        self.skill, self.metrics, self.trail = -9.0, {}, []


class Generalist:
    def __init__(self, workspace, seed=0, verbose=True):
        self.ws = workspace if isinstance(workspace, factory.Workspace) else factory.Workspace(workspace)
        self.zoo = Zoo(self.ws.root)
        self.prior = AgentPrior(f"{self.ws.root}/forge/prior.json")
        self.rng = np.random.default_rng(seed)
        self.seed, self.verbose = seed, verbose
        self._n = 0
        try:                                                   # use whatever self-written search rules are live
            from ..selfmod.hooks import HookManager
            HookManager(self.ws.root).activate()
        except Exception:                                      # noqa: BLE001 - a broken overlay must not stop solving
            pass

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ---------------------------------------------------------------------------------------------
    def solve(self, task, budget=1.0, n_new=8, rungs=3):
        t0 = time.time()
        if task.kind == "image_gen" and task.classifier is None:
            self.log(f"[forge] '{task.name}': generation is judged by a classifier, building one first")
            sub = self.solve(task.sup, budget=budget, n_new=max(4, n_new // 2), rungs=rungs)
            task.classifier = sub["_agent"]
        fams = families_for(task)
        if not fams:
            raise ValueError(f"no agent family can handle a task of kind '{task.kind}'")
        sig = task.signature()
        self.log(f"[forge] task '{task.name}' kind={task.kind} dim={sig['dim']} classes={sig['classes']} "
                 f"-> families: {[f.name for f in fams]}")
        units = BASE_UNITS[task.kind] * budget
        cands, skill0 = [], None

        # 2) incumbent ------------------------------------------------------------------------------
        best = self.zoo.best(task.name)
        if best:
            spec = self.zoo.spec(task.name, best["id"])
            try:
                a = FAMILIES[spec["family"]].load(f"{self.zoo.task_dir(task.name)}/{best['id']}", task, spec["genome"])
                c = Candidate(a, "incumbent", self._cid())
                c.skill, c.metrics = a.val_score(task)
                skill0 = c.skill
                cands.append(c)
                self.log(f"[forge]   incumbent {spec['family']} re-scored: skill={c.skill:.3f}")
            except Exception as e:                              # noqa: BLE001 - corrupt incumbent: report, rebuild
                self.log(f"[forge]   incumbent unreadable ({type(e).__name__}: {e}); ignoring it")

        # 3) transfer -------------------------------------------------------------------------------
        for other in self.zoo.tasks():
            if other == _safe(task.name):
                continue
            ob = self.zoo.best(other)
            if not ob:
                continue
            try:
                spec = self.zoo.spec(other, ob["id"])
                if spec.get("kind") != task.kind or spec.get("dim") != sig["dim"] or spec.get("classes") != sig["classes"]:
                    continue
                if spec["family"] not in {f.name for f in fams}:
                    continue
                a = FAMILIES[spec["family"]].load(f"{self.zoo.task_dir(other)}/{ob['id']}", task, spec["genome"])
                c = Candidate(a, f"transfer:{other}", self._cid())
                c.skill, c.metrics = a.val_score(task)
                cands.append(c)
                self.log(f"[forge]   transfer from '{other}' ({spec['family']}): zero-shot skill={c.skill:.3f}")
            except Exception:                                   # noqa: BLE001 - shapes may not fit; that is fine
                continue

        # 4) propose --------------------------------------------------------------------------------
        names = [f.name for f in fams]
        for _ in range(n_new):
            fam = FAMILIES[self.prior.choose_family(task.kind, names, self.rng)]
            genome = self.prior.sample_genome(task.kind, fam.name, fam.space, self.rng, dim=sig["dim"])
            cid = self._cid()
            try:
                cands.append(Candidate(fam.build(task, genome, self.seed * 7919 + cid), "new", cid))
            except Exception as e:                              # noqa: BLE001 - a genome that cannot be built loses
                self.log(f"[forge]   could not build {fam.name} {genome}: {type(e).__name__}: {e}")

        # 5) race ----------------------------------------------------------------------------------
        alive = list(cands)
        for r in range(rungs):
            for c in alive:
                rng = np.random.default_rng(self.seed * 100003 + c.cid * 31 + r)
                try:
                    c.agent.train(task, units * (2 ** r), rng)
                    c.skill, c.metrics = c.agent.val_score(task)
                except Exception as e:                          # noqa: BLE001 - a crashing candidate is eliminated
                    c.skill, c.metrics = -9.0, dict(error=f"{type(e).__name__}: {e}")
                c.trail.append(round(c.skill, 3))
            alive.sort(key=lambda c: -c.skill)
            self.log(f"[forge]   rung {r + 1}/{rungs}: " + ", ".join(f"{c.agent.family}[{c.origin.split(':')[0]}]={c.skill:.2f}"
                                                                     for c in alive[:5]) + (" ..." if len(alive) > 5 else ""))
            if r < rungs - 1:
                alive = alive[:max(1, math.ceil(len(alive) / 2))]
        winner = alive[0]

        # 6) keep -------------------------------------------------------------------------------------
        for c in cands:                                         # learn from every candidate, winners and losers
            if c.origin == "new" and c.skill > -9:
                self.prior.observe(task.kind, c.agent.family, c.agent.genome, c.skill, dim=sig["dim"])
        self.prior.save()

        improved = skill0 is None or winner.skill > skill0 + 1e-9
        test = winner.agent.test_metrics(task) if improved or skill0 is None else None
        spec = dict(sig, skill=float(winner.skill), val=winner.metrics, origin=winner.origin, trail=winner.trail,
                    task=task.name, spent=float(winner.agent.spent), parent=(best or {}).get("id"))
        if improved:
            aid = self.zoo.add(task.name, winner.agent, dict(spec, test=test), make_champion=True)
            champion, champ_skill = winner, winner.skill
        else:
            aid = best["id"]
            self.log(f"[forge]   nothing beat the incumbent ({winner.skill:.3f} <= {skill0:.3f}); kept it unchanged")
            old = self.zoo.spec(task.name, aid)
            test = old.get("test")
            # hand back the STORED champion, not the (slightly worse) copy that kept training in the race
            kept = FAMILIES[old["family"]].load(f"{self.zoo.task_dir(task.name)}/{aid}", task, old["genome"])
            champion, champ_skill = Candidate(kept, "incumbent", self._cid()), skill0
        rec = dict(kind="forge", name=task.name, task_kind=task.kind, family=champion.agent.family,
                   skill=float(champ_skill), improved=bool(improved), origin=champion.origin, agent_id=aid,
                   genome=champion.agent.genome, test=test, candidates=len(cands), seconds=time.time() - t0,
                   description=champion.agent.describe())
        self.ws.log(rec)
        rec["_agent"] = champion.agent
        rec["baseline"] = task.chance() if hasattr(task, "chance") else None
        if self.verbose:
            self.log(f"[forge] '{task.name}' winner: {rec['family']} (from {rec['origin']}) skill={rec['skill']:.3f} "
                     f"{'IMPROVED' if improved else 'unchanged'} | test={_fmt(test)} | {rec['seconds']:.0f}s")
            self.log("        " + rec["description"].replace("\n", "\n        "))
        return rec

    def _cid(self):
        self._n += 1
        return self._n


def _fmt(d):
    if not d:
        return "n/a"
    return ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items() if k != "columns")
