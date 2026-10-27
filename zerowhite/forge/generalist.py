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
  5c. ARCHITECT (6.5, opt-in) when the race has plateaued, the architect (architect.py) proposes edits to the winner (program edits, genome changes), tests each against a paired control on
                held-out data, and the winner is replaced only by something that survived; nothing survives -> nothing changes.
  4b. RECALL    (6.5) the episodic memory (episodic.py) returns the champions of the nearest earlier tasks; their genomes (and a blend of two) replace some of the random
                proposals, and their program sources, adapted and spliced, join a program agent's starting population. They still have to win the race on THIS task.
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
    def __init__(self, workspace, seed=0, verbose=True, episodic=True, blend=True, architect=None):
        self.ws = workspace if isinstance(workspace, factory.Workspace) else factory.Workspace(workspace)
        self.zoo = Zoo(self.ws.root)
        self.prior = AgentPrior(f"{self.ws.root}/forge/prior.json")
        self.rng = np.random.default_rng(seed)
        self.seed, self.verbose, self.blend = seed, verbose, blend
        self._n = 0
        self.memory = None
        self.architect = None                                  # 6.5: an Architect (architect.py), or True to build the default one (heuristic proposals); None = off, as before
        if architect:
            try:
                from .architect import Architect
                self.architect = architect if not isinstance(architect, bool) else Architect(self.ws.root, seed=seed, verbose=verbose)
            except Exception:                                  # noqa: BLE001 - no architect is the old behaviour, never a failure
                self.architect = None
        if episodic:
            try:
                from .episodic import EpisodicMemory
                self.memory = EpisodicMemory(self.ws.root, seed=seed)
            except Exception:                                  # noqa: BLE001 - no memory is a slower start, never a failure
                self.memory = None
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
        recalled, n_recalled = [], 0
        if self.memory is not None and len(self.memory):
            try:
                recalled = self.memory.recall(task, k=8, exclude_task=True, min_skill=0.1)
                task._episodic, task._recalled = self.memory, recalled
            except Exception as e:                              # noqa: BLE001
                self.log(f"[forge]   episodic recall failed ({type(e).__name__}: {e}); continuing without it")
                recalled = []
        if recalled:
            pool = []
            for fam in fams:
                pool.append([(fam, g, lab) for g, lab in self.memory.genomes_for(task, fam.name, fam.space, k=2, rng=self.rng, recalled=recalled)])
            want = min(max(1, n_new // 3), sum(len(p) for p in pool))
            i = 0
            while n_recalled < want and any(pool):
                row = pool[i % len(pool)]
                i += 1
                if not row:
                    continue
                fam, genome, lab = row.pop(0)
                cid = self._cid()
                try:
                    cands.append(Candidate(fam.build(task, genome, self.seed * 7919 + cid), f"recall:{lab}", cid))
                    n_recalled += 1
                except Exception as e:                          # noqa: BLE001
                    self.log(f"[forge]   could not build recalled {fam.name} {genome}: {type(e).__name__}: {e}")
            if n_recalled:
                self.log(f"[forge]   episodic memory: {len(recalled)} similar earlier champion(s) (nearest: '{recalled[0]['task']}', cosine {recalled[0]['score']:.2f}); {n_recalled} recalled candidate(s) join the race")
        for _ in range(max(1, n_new - n_recalled)):
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

        # 5b) blend (6.5) ----------------------------------------------------------------------------
        if self.blend and task.kind in ("classification", "regression") and len({c.agent.family for c in cands if c.skill > -9}) >= 2:
            try:
                from . import moa
                b = moa.build_blend(task, sorted(cands, key=lambda c: -c.skill), seed=self.seed)
                if b is not None:
                    bs = moa.heldout_skill(b, task)
                    ws_ = moa.heldout_skill(winner.agent, task)
                    gain = bs - ws_
                    self.log(f"[forge]   router over {len(b.members)} finished agents ({', '.join(m[0] for m in b.members)}): {gain:+.3f} skill vs the winner on the validation rows its gate never saw")
                    if gain > 0.01:
                        cid = self._cid()
                        c = Candidate(b, "blend", cid)
                        c.skill = winner.skill + gain                  # the paired gain on unseen rows, added to the winner's full-validation skill
                        c.metrics = dict(b.val_score(task)[1], blend_gain=float(gain))
                        c.trail = [round(c.skill, 3)]
                        cands.append(c)
                        alive.insert(0, c)
                        winner = c
            except Exception as e:                              # noqa: BLE001 - a failed blend is a missed opportunity, not a failed solve
                self.log(f"[forge]   blend skipped ({type(e).__name__}: {e})")

        # 5c) architect (6.5) ------------------------------------------------------------------------
        if self.architect is not None and winner.skill > -9 and self._plateaued(winner):
            try:
                if self.architect.due(task.name):
                    prev_pin = getattr(winner.agent, "_pinned", None)
                    rep = self.architect.improve(task, winner.agent, plateau=True)
                    if rep.get("accepted"):
                        ag = rep["agent"]
                        cid = self._cid()
                        c = Candidate(ag, "architect:" + str(winner.origin).split(":")[0], cid)
                        c.skill, c.metrics = ag.val_score(task)
                        c.trail = list(winner.trail) + [round(c.skill, 3)]
                        self.log(f"[forge]   architect: {rep['note']} (skill {rep['skill_before']:.3f} -> {c.skill:.3f})")
                        if c.skill > winner.skill + 1e-9:
                            cands.append(c)
                            alive.insert(0, c)
                            winner = c
                        elif ag is winner.agent and hasattr(ag, "_pinned"):
                            ag._pinned = prev_pin                    # proven on the architect's gate pairs but not better on the full validation: the old program stays
                    else:
                        self.log(f"[forge]   architect: {rep['note']}")
                else:
                    self.log("[forge]   architect: waiting out its back-off for this task")
            except Exception as e:                              # noqa: BLE001 - a failed architect round is a missed opportunity, not a failed solve
                self.log(f"[forge]   architect skipped ({type(e).__name__}: {e})")

        # 6) keep -------------------------------------------------------------------------------------
        for c in cands:                                         # learn from every candidate, winners and losers
            if (c.origin == "new" or c.origin.startswith("recall:")) and c.skill > -9:
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
        self._remember(task, cands, champion if improved else None, kept_id=aid)
        rec = dict(kind="forge", name=task.name, task_kind=task.kind, family=champion.agent.family,
                   skill=float(champ_skill), improved=bool(improved), origin=champion.origin, agent_id=aid,
                   genome=champion.agent.genome, test=test, candidates=len(cands), seconds=time.time() - t0,
                   description=champion.agent.describe(), recalled=n_recalled, won_by_recall=str(champion.origin).startswith("recall:"))
        self.ws.log(rec)
        rec["_agent"] = champion.agent
        rec["baseline"] = task.chance() if hasattr(task, "chance") else None
        if self.verbose:
            self.log(f"[forge] '{task.name}' winner: {rec['family']} (from {rec['origin']}) skill={rec['skill']:.3f} "
                     f"{'IMPROVED' if improved else 'unchanged'} | test={_fmt(test)} | {rec['seconds']:.0f}s")
            self.log("        " + rec["description"].replace("\n", "\n        "))
        return rec

    @staticmethod
    def _plateaued(c, tol=0.02):
        """The winner stopped improving: its last rung gained less than `tol` of skill over the one before (a single rung counts as plateaued)."""
        t = c.trail
        return len(t) < 2 or (t[-1] - t[-2]) < tol

    def _remember(self, task, cands, champion, kept_id=None):
        """Write what this solve learned into the episodic memory: the champion (with its program source when it is one), a few candidates that clearly lost, and a window of observations for
        an environment. Never raises: a memory that cannot be written is a missed lesson, not a failed solve."""
        if self.memory is None:
            return
        try:
            from . import episodic as ep
            if champion is not None:
                a = champion.agent
                src = a.best_src() if hasattr(a, "best_src") else None
                fn = a.vocab.fn_name if hasattr(a, "vocab") else None
                self.memory.remember_task(task, family=a.family, genome=a.genome, skill=champion.skill, src=src, fn_name=fn, outcome="success", origin=champion.origin)
                if src:
                    self.memory.remember_code(task.name, src, champion.skill, a.family)
                if task.kind == "env":
                    self.memory.remember_states(task, ep.probe_obs(task))
            top = champion.skill if champion is not None else max((c.skill for c in cands), default=0.0)
            losers = sorted((c for c in cands if c is not champion and c.skill > -9 and c.skill < top - 0.2), key=lambda c: c.skill)[:3]
            for c in losers:
                self.memory.remember_task(task, family=c.agent.family, genome=c.agent.genome, skill=c.skill, outcome="failure", origin=c.origin)
            self.memory.maybe_fit()
            self.memory.save()
        except Exception as e:                                  # noqa: BLE001
            self.log(f"[forge]   episodic memory could not be updated ({type(e).__name__}: {e})")

    def _cid(self):
        self._n += 1
        return self._n


def _fmt(d):
    if not d:
        return "n/a"
    return ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items() if k != "columns")
