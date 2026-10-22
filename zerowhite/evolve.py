"""
evolve.py - the model factory's search engine.

Zero White "creates" a generative model or an agent by evolving its genome: a
small dict of hyper-parameters. A (mu + lambda) evolution strategy with
1/5-success-rule step adaptation proposes children; each child is trained and
scored by a caller-supplied `evaluate(genome, parent_artifact)`.

The step-size rule and the parent-selection rule are not written here: they live in
evolved/sigma_update.py and evolved/select_parent.py, which Zero White may rewrite.

Two kinds of gene:
  * SHAPE genes (layer widths, depth, context length...) change the weight
    shapes. A child that mutates one is trained from scratch.
  * TRAINING genes (learning rate, KL weight, entropy bonus...) do not. A child
    that only mutates these INHERITS its parent's weights and keeps training,
    so improvement accumulates across generations and across sessions instead
    of restarting every time.
"""

import math

import numpy as np

from .evolved import Policy


class Gene:
    def __init__(self, kind, lo, hi, shape=False):
        assert kind in ("int", "float", "log")
        self.kind, self.lo, self.hi, self.shape = kind, lo, hi, shape

    def clip(self, v):
        v = min(max(v, self.lo), self.hi)
        return int(round(v)) if self.kind == "int" else float(v)

    def sample(self, rng):
        if self.kind == "log":
            return self.clip(math.exp(rng.uniform(math.log(self.lo), math.log(self.hi))))
        return self.clip(rng.uniform(self.lo, self.hi))

    def mutate(self, v, rng, sigma):
        if self.kind == "log":
            span = math.log(self.hi) - math.log(self.lo)
            nv = self.clip(math.exp(math.log(v) + rng.normal(0, sigma * 0.25 * span)))
        else:
            nv = self.clip(v + rng.normal(0, sigma * 0.25 * (self.hi - self.lo)))
        if self.kind == "int" and nv == v:           # an int mutation must actually move
            nv = self.clip(v + (1 if rng.random() < 0.5 else -1))
        return nv


def random_genome(space, rng):
    return {k: g.sample(rng) for k, g in space.items()}


def mutate_genome(space, genome, rng, sigma):
    child = dict(genome)
    names = list(space)
    n_change = max(1, int(rng.binomial(len(names), 0.4)))
    for k in rng.choice(names, size=min(n_change, len(names)), replace=False):
        child[k] = space[k].mutate(genome[k], rng, sigma)
    return child


class Entry:
    __slots__ = ("genome", "fitness", "artifact", "info")

    def __init__(self, genome, fitness, artifact, info):
        self.genome, self.fitness, self.artifact, self.info = genome, fitness, artifact, info


class Evolver:
    """(mu + lambda) evolution over genomes.

    evaluate(genome, parent_artifact_or_None) -> (fitness, artifact, info_dict)
    Higher fitness is better. The evaluate function must NOT mutate the parent
    artifact (clone it first)."""

    def __init__(self, space, evaluate, population=4, mu=2, seed=0, sigma=0.5, policy=None):
        self.space, self.evaluate = space, evaluate
        # step-size rule and parent choice are EVOLVABLE source files (see evolved/); the shipped
        # genesis versions are trivial, so anything smarter was discovered, not hard-coded.
        self.policy = policy or Policy.load()
        self.population, self.mu = population, min(mu, population)
        self.rng = np.random.default_rng(seed)
        self.sigma = sigma
        self.pool = []            # Entry list, best first
        self.journal = []         # one record per evaluated genome
        self.generation = 0
        self.baseline = None      # best fitness when run() began (the incumbent's re-scored fitness)

    @property
    def best(self):
        return self.pool[0] if self.pool else None

    def inject(self, genome, fitness, artifact, info=None):
        """Seed the pool with an existing model (e.g. the registry's incumbent)."""
        self.pool.append(Entry(dict(genome), fitness, artifact, info or {}))
        self.pool.sort(key=lambda e: e.fitness, reverse=True)
        del self.pool[self.mu:]

    def _eval(self, genome, parent):
        shapes_same = parent is not None and all(
            parent.genome[k] == genome[k] for k, g in self.space.items() if g.shape)
        fitness, artifact, info = self.evaluate(genome, parent.artifact if shapes_same else None)
        if not np.isfinite(fitness):
            fitness = -1e9
        self.journal.append(dict(generation=self.generation, genome=dict(genome),
                                 fitness=float(fitness), inherited=bool(shapes_same), **info))
        return Entry(genome, float(fitness), artifact, info)

    def step(self):
        self.generation += 1
        parent_best = self.best.fitness if self.pool else -np.inf
        children = []
        if not self.pool:
            children = [self._eval(random_genome(self.space, self.rng), None) for _ in range(self.population)]
        else:
            fits = np.array([e.fitness for e in self.pool])
            for _ in range(self.population):
                parent = self.pool[self.policy.pick_parent(fits, self.rng)]
                children.append(self._eval(mutate_genome(self.space, parent.genome, self.rng, self.sigma), parent))
        merged = sorted(self.pool + children, key=lambda e: e.fitness, reverse=True)
        self.pool = merged[:self.mu]
        if parent_best > -np.inf:
            improved = max(c.fitness for c in children) > parent_best
            # the hook proposes, the core disposes: whatever it returns is clipped to a safe range
            self.sigma = float(np.clip(self.policy.update_sigma(self.sigma, improved, self.generation), 0.1, 1.0))
        return self.best

    @property
    def improved(self):
        """Did this run beat the incumbent it started from (or is there no incumbent)?"""
        return self.baseline is None or self.best.fitness > self.baseline + 1e-12

    def run(self, generations, verbose=True, label="evolve"):
        self.baseline = self.best.fitness if self.pool else None
        for _ in range(generations):
            b = self.step()
            if verbose:
                print(f"[{label} gen {self.generation:02d}] best fitness={b.fitness:.4f} "
                      f"sigma={self.sigma:.2f} genome={b.genome}")
        return self.best

    def state(self):
        return dict(generation=self.generation, sigma=self.sigma, journal=self.journal[-200:],
                    pool=[dict(genome=e.genome, fitness=e.fitness) for e in self.pool])
