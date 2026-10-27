"""islands.py - evolve programs (Python source) with no language model.

Merged from three reference projects, re-implemented here from their *ideas*:

  FunSearch      an island model: several sub-populations evolve separately; periodically the weaker
                 half is wiped and re-seeded from the survivors' champions, which keeps diversity.
  AutoML-Zero    regularized evolution (tournament pick, mutate, the OLDEST non-champion dies, so the
                 population cannot stagnate on stale members) and the FEC cache: two programs that
                 behave identically on probe inputs are the same program, so the second is never
                 re-evaluated.
  PySR / DEAP    after structure changes, numeric constants are tuned locally ((1+1)-ES), and subtree
                 crossover recombines good parts.

The proposer is `astmut` (grammar-guided AST edits), not an LLM. Every candidate passes the static
safety gate before it is compiled, and a candidate that raises or returns non-finite scores is simply
scored as invalid.
"""

import math

import numpy as np

from . import astmut as A

INVALID = -1e12


class _Member:
    __slots__ = ("src", "score", "born")

    def __init__(self, src, score, born):
        self.src, self.score, self.born = src, score, born


class ProgramEvolver:
    """Search over programs for `vocab`.

    score_fn(fn, src) -> float, higher is better; `fn` is the compiled function.
    probe_fn(fn)      -> any hashable/ndarray "behaviour fingerprint" for the FEC cache (optional).
    seeds             -> list of source strings to start from (usually one trivial program).
    """

    def __init__(self, vocab, score_fn, seeds, rng, *, probe_fn=None, n_islands=3, island_size=8,
                 tournament=3, tune_steps=12, reset_every=150, p_cross=0.2, max_mut=2):
        self.vocab, self.score_fn, self.probe_fn, self.rng = vocab, score_fn, probe_fn, rng
        self.n_islands, self.island_size, self.tournament = n_islands, island_size, tournament
        self.tune_steps, self.reset_every, self.p_cross, self.max_mut = tune_steps, reset_every, p_cross, max_mut
        self.islands = [[] for _ in range(n_islands)]
        self.seen = {}                  # normalized source -> score
        self.fec = {}                   # behaviour fingerprint -> score
        self.evals = self.fec_hits = self.invalid = self.clock = 0
        self.history = []               # (evals, best score) every time the global best improves
        self.best = None
        for s in seeds:
            for isl in self.islands:
                self._admit(isl, s)

    # ---- evaluation -------------------------------------------------------------------
    def _evaluate(self, src):
        key = A.normalize(src)
        if key in self.seen:
            return self.seen[key]
        try:
            fn = A.compile_program(src, self.vocab.fn_name, **self.vocab.safety)
            fp = None
            if self.probe_fn is not None:
                fp = self.probe_fn(fn)
                if fp is not None:
                    fp = np.round(np.asarray(fp, dtype=float), 6).tobytes() if not isinstance(fp, (bytes, tuple)) else fp
                    if fp in self.fec:
                        self.fec_hits += 1
                        self.seen[key] = self.fec[fp]
                        return self.fec[fp]
            sc = float(self.score_fn(fn, src))
            if not math.isfinite(sc):
                raise ValueError("non-finite score")
            self.evals += 1
            if fp is not None:
                self.fec[fp] = sc
        except Exception:                                        # noqa: BLE001 - a bad candidate is just invalid
            self.invalid += 1
            sc = INVALID
        self.seen[key] = sc
        return sc

    def _admit(self, island, src, score=None):
        score = self._evaluate(src) if score is None else score
        self.clock += 1
        island.append(_Member(src, score, self.clock))
        if len(island) > self.island_size:                       # regularized evolution: oldest non-champion dies
            champ = max(island, key=lambda m: m.score)
            victims = [m for m in island if m is not champ]
            island.remove(min(victims, key=lambda m: m.born))
        if score > INVALID and (self.best is None or score > self.best.score + 1e-12):
            self.best = _Member(src, score, self.clock)
            self.history.append((self.evals, score))
        return score

    # ---- search ------------------------------------------------------------------------
    def _tournament(self, island):
        k = min(self.tournament, len(island))
        idx = self.rng.choice(len(island), size=k, replace=False)
        return max((island[i] for i in idx), key=lambda m: m.score)

    def step(self):
        isl = self.islands[int(self.rng.integers(self.n_islands))]
        if not isl:
            return
        parent = self._tournament(isl)
        child = None
        if len(isl) > 1 and self.rng.random() < self.p_cross:
            other = self._tournament(isl)
            if other is not parent:
                child = A.crossover(parent.src, other.src, self.vocab, self.rng)
        if child is None:
            child = A.mutate(parent.src, self.vocab, self.rng, n_mut=int(self.rng.integers(1, self.max_mut + 1)))
        if child is None:
            return
        sc = self._admit(isl, child)
        if self.tune_steps and sc > INVALID and sc >= max(m.score for m in isl) - 1e-12 and A.constants_of(child):
            def f(s):
                return self._evaluate(s)
            tuned, tsc, _ = A.tune_constants(child, f, self.rng, steps=self.tune_steps)
            if tsc > sc + 1e-12:
                self._admit(isl, tuned, tsc)
        if self.reset_every and self.clock % self.reset_every == 0:
            self._reset_weak_islands()

    def _reset_weak_islands(self):
        champs = [(max(i, key=lambda m: m.score) if i else None) for i in self.islands]
        order = sorted(range(self.n_islands), key=lambda k: champs[k].score if champs[k] else INVALID)
        for k in order[:self.n_islands // 2]:
            donors = [champs[j] for j in order[self.n_islands // 2:] if champs[j]]
            if donors:
                d = donors[int(self.rng.integers(len(donors)))]
                self.islands[k] = [_Member(d.src, d.score, self.clock)]

    def run(self, max_evals, max_steps=None):
        """Run until `max_evals` real evaluations (cache hits are free) or `max_steps` proposals."""
        target, steps = self.evals + max_evals, 0
        max_steps = max_steps or max_evals * 12
        while self.evals < target and steps < max_steps:
            self.step()
            steps += 1
        return self.best

    def top(self, k=3):
        pool = {}
        for isl in self.islands:
            for m in isl:
                pool[A.normalize(m.src)] = m
        return sorted(pool.values(), key=lambda m: m.score, reverse=True)[:k]

    # ---- persistence (population survives between sessions) -------------------------------------
    def state(self):
        return dict(islands=[[(m.src, m.score) for m in isl] for isl in self.islands],
                    best=(self.best.src, self.best.score) if self.best else None,
                    evals=self.evals, fec_hits=self.fec_hits, invalid=self.invalid, history=self.history[-300:])

    def restore(self, st):
        """Re-admit saved members (re-scored by the CURRENT score_fn: the task may have changed)."""
        for isl, saved in zip(self.islands, st.get("islands", [])):
            for src, _old in saved:
                try:
                    self._admit(isl, src)
                except Exception:                                # noqa: BLE001
                    pass
        return self
