"""
self_model.py — Zero White's editable record of its own self-modification
policy, now with a real action MASK and a policy that actually steers compute.

v1 evaluated every action every generation, so `action_weights` never changed
what happened (it only broke ties). v2 fixes that:

  * feasibility mask  — moves that cannot do anything right now (grow at the
    unit cap, add_layer at the layer cap, patch_on when it's already on, ...)
    are removed from the pool instead of burning a probation run on a no-op.
  * tabu cooldown     — a move that was tried `patience` times in a row and
    never won is benched for a few generations, then allowed back (it may
    become useful after the task or architecture changes).
  * weighted sampling — each generation tries `noop` (the control) plus the
    top-weighted/explored subset of the remaining moves.
  * advantage credit  — weights learn from (score - noop score), i.e. what a
    move gained BEYOND just continuing to train.
"""

import time

import numpy as np

from . import fileio


class SelfModel:
    ACTIONS = ("grow", "prune", "add_layer", "prune_layer", "perturb",
               "mutate_opt", "decay_lr", "patch_on", "patch_off", "noop")
    NEVER_BENCHED = ("noop",)

    def __init__(self, max_hidden_units=64, max_layers=6, temperature=0.1,
                 min_prob=0.03, patience=4, cooldown=3, max_history=2000):
        self.max_hidden_units = max_hidden_units
        self.max_layers = max_layers
        self.temperature = temperature
        self.min_prob = min_prob
        self.patience = patience
        self.cooldown = cooldown
        self.max_history = max_history          # long runs keep only the recent journal...
        self.tally = {a: {"tried": 0, "won": 0} for a in self.ACTIONS}   # ...plus lifetime totals

        self.action_weights = {a: 1.0 / len(self.ACTIONS) for a in self.ACTIONS}
        self._running_benefit = {a: 0.0 for a in self.ACTIONS}
        self.lose_streak = {a: 0 for a in self.ACTIONS}
        self.bench_until = {a: 0 for a in self.ACTIONS}
        self.generation = 0
        self.history = []

    # ---- mask + sampling ---------------------------------------------------
    def eligible(self, feasible):
        """Actions that are feasible now AND not currently benched."""
        out = []
        for a in self.ACTIONS:
            if not feasible.get(a, False):
                continue
            if a not in self.NEVER_BENCHED and self.bench_until[a] > self.generation:
                continue
            out.append(a)
        if "noop" not in out:
            out.append("noop")
        return out

    def select_actions(self, feasible, rng, k=5):
        """noop (control) + up to k-1 others drawn without replacement,
        probability ∝ weight with a floor so nothing starves forever."""
        pool = [a for a in self.eligible(feasible) if a != "noop"]
        chosen = ["noop"]
        if pool and k > 1:
            p = np.array([max(self.action_weights[a], self.min_prob) for a in pool])
            p = p / p.sum()
            n = min(k - 1, len(pool))
            idx = rng.choice(len(pool), size=n, replace=False, p=p)
            chosen += [pool[i] for i in idx]
        return chosen

    # ---- learning ----------------------------------------------------------
    def observe_outcome(self, action, advantage, decay=0.85):
        self._running_benefit[action] = decay * self._running_benefit[action] + (1 - decay) * advantage
        vals = np.array([self._running_benefit[a] for a in self.ACTIONS])
        vals = (vals - vals.max()) / self.temperature
        w = np.exp(vals)
        w = w / w.sum()
        self.action_weights = {a: float(w[i]) for i, a in enumerate(self.ACTIONS)}

    def record_result(self, tried, winner):
        """Update tabu bookkeeping after a generation."""
        for a in tried:
            self.tally[a]["tried"] += 1
            if a == winner:
                self.tally[a]["won"] += 1
            if a in self.NEVER_BENCHED:
                continue
            if a == winner:
                self.lose_streak[a] = 0
            else:
                self.lose_streak[a] += 1
                if self.lose_streak[a] >= self.patience:
                    self.bench_until[a] = self.generation + 1 + self.cooldown
                    self.lose_streak[a] = 0

    def log(self, entry: dict):
        entry = dict(entry)
        entry["generation"] = self.generation
        entry["t"] = time.time()
        self.history.append(entry)
        if len(self.history) > self.max_history:
            del self.history[:len(self.history) - self.max_history]

    # ---- persistence -------------------------------------------------------
    def to_dict(self):
        return {
            "generation": self.generation,
            "action_weights": self.action_weights,
            "max_hidden_units": self.max_hidden_units,
            "max_layers": self.max_layers,
            "temperature": self.temperature, "min_prob": self.min_prob,
            "patience": self.patience, "cooldown": self.cooldown,
            "running_benefit": self._running_benefit, "tally": self.tally,
            "max_history": self.max_history,
            "lose_streak": self.lose_streak, "bench_until": self.bench_until,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d):
        sm = cls(d["max_hidden_units"], d["max_layers"], d.get("temperature", 0.1),
                 d.get("min_prob", 0.03), d.get("patience", 4), d.get("cooldown", 3),
                 d.get("max_history", 2000))
        sm.generation = d["generation"]
        for name in ("action_weights", "running_benefit", "lose_streak", "bench_until"):
            tgt = {"running_benefit": "_running_benefit"}.get(name, name)
            if name in d:
                getattr(sm, tgt).update(d[name])
        for a, t in d.get("tally", {}).items():
            if a in sm.tally:
                sm.tally[a].update(t)
        sm.history = d["history"]
        return sm

    def save(self, path):
        fileio.write_json(path, self.to_dict(), indent=2)
