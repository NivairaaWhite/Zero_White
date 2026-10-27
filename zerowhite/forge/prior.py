"""prior.py - the Generalist's learned *generative model of agents*.

Given a task kind, it samples (family, genome): which kind of agent to build and with which settings.
It starts with no knowledge (uniform over families, uniform-random genomes: NO default agent exists)
and learns from every race the Generalist runs:

  * family choice   softmax over each family's running mean skill on this kind of task, with an optimistic
                    value for families not yet tried and a probability floor so none starves;
  * genome sampling a Gaussian around elite genomes of that (kind, family), in normalised gene space
                    (log genes in log space), weighted toward elites from *similar* tasks (log-dimension
                    distance); exploration decays with the number of elites but never reaches zero.

This is an estimation-of-distribution scheme (the idea behind CMA-ES / cross-entropy search in DEAP),
used here as a model that gets better at proposing agents the more tasks it has seen.
"""

import math
import os

import numpy as np

from .. import fileio
from ..evolve import random_genome

ELITES_KEPT = 12


def _to_unit(gene, v):
    if gene.kind == "log":
        return (math.log(v) - math.log(gene.lo)) / (math.log(gene.hi) - math.log(gene.lo))
    return (v - gene.lo) / (gene.hi - gene.lo)


def _from_unit(gene, u):
    u = min(1.0, max(0.0, u))
    if gene.kind == "log":
        return gene.clip(math.exp(math.log(gene.lo) + u * (math.log(gene.hi) - math.log(gene.lo))))
    return gene.clip(gene.lo + u * (gene.hi - gene.lo))


class AgentPrior:
    def __init__(self, path=None):
        self.path = path
        self.stats = {}          # kind -> family -> {n, mean}
        self.elites = {}         # "kind|family" -> [{genome, skill, dim}]
        self.observations = 0
        if path and os.path.exists(path):
            try:
                d = fileio.read_json(path)
                self.stats, self.elites, self.observations = d["stats"], d["elites"], d.get("observations", 0)
            except (ValueError, KeyError):
                pass                                          # corrupt prior: start learning again, never crash

    def save(self):
        if self.path:
            fileio.write_json(self.path, dict(stats=self.stats, elites=self.elites, observations=self.observations),
                              indent=1)

    # ---- learning --------------------------------------------------------------------------------
    def observe(self, kind, family, genome, skill, dim=0):
        st = self.stats.setdefault(kind, {}).setdefault(family, dict(n=0, mean=0.0))
        st["n"] += 1
        st["mean"] += (skill - st["mean"]) / min(st["n"], 12)          # recent results count a bit more
        el = self.elites.setdefault(f"{kind}|{family}", [])
        el.append(dict(genome=dict(genome), skill=float(skill), dim=int(dim)))
        el.sort(key=lambda e: -e["skill"])
        del el[ELITES_KEPT:]
        self.observations += 1

    # ---- proposing ------------------------------------------------------------------------------
    def family_probs(self, kind, names):
        known = self.stats.get(kind, {})
        means = [known[n]["mean"] for n in names if n in known]
        optimistic = (max(means) + 0.15) if means else 0.0
        s = np.array([known[n]["mean"] if n in known else optimistic for n in names], dtype=float)
        p = np.exp((s - s.max()) / 0.15)
        p = p / p.sum()
        p = np.maximum(p, 0.08)
        return p / p.sum()

    def choose_family(self, kind, names, rng):
        return names[int(rng.choice(len(names), p=self.family_probs(kind, names)))]

    def sample_genome(self, kind, family, space, rng, dim=0):
        el = self.elites.get(f"{kind}|{family}", [])
        p_explore = 0.2 + 0.8 * math.exp(-len(el) / 6.0)
        if not el or rng.random() < p_explore:
            return random_genome(space, rng)
        sim = np.array([math.exp(-abs(math.log2((e["dim"] + 1.0) / (dim + 1.0))) / 2.0) for e in el])
        w = sim * np.exp(-np.arange(len(el)) / 4.0) * (1.0 + np.clip([e["skill"] for e in el], 0, 1))
        w = w / w.sum()
        parent = el[int(rng.choice(len(el), p=w))]
        out = {}
        for k, gene in space.items():
            vals = np.array([_to_unit(gene, e["genome"][k]) for e in el if k in e["genome"]])
            sd = max(0.08, float(vals.std())) if len(vals) > 1 else 0.15
            base = _to_unit(gene, parent["genome"][k]) if k in parent["genome"] else rng.random()
            out[k] = _from_unit(gene, base + rng.normal(0.0, sd))
        return out
