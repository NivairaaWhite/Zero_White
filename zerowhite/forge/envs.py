"""envs.py - more pure-numpy environments, plus PARAMETRISED ones.

MountainCar and a parametrised CartPole (gravity, pole length, force, ...) are the substrate for
open-ended co-evolution (poet.py, after POET): the environment's own parameters are a genome, so the
system can *invent* new tasks, not just solve given ones. Names accepted by agents.make_env:

    mountaincar      cartpole:g=9.8,len=0.5,force=10
"""

import numpy as np


class MountainCar:
    """Discrete MountainCar: 3 actions (left, none, right). Reward -1 per step, episode ends at the flag."""
    obs_dim, n_actions, name = 2, 3, "mountaincar"
    max_steps, max_return = 200, -80.0

    def reset(self, rng):
        self.x, self.v, self.t = rng.uniform(-0.6, -0.4), 0.0, 0
        return self._obs()

    def _obs(self):
        return np.array([(self.x + 0.3) / 0.9, self.v / 0.07])

    def step(self, a):
        self.v = float(np.clip(self.v + (int(a) - 1) * 0.001 + np.cos(3 * self.x) * -0.0025, -0.07, 0.07))
        self.x = float(np.clip(self.x + self.v, -1.2, 0.6))
        if self.x <= -1.2:
            self.v = 0.0
        self.t += 1
        won = self.x >= 0.5
        return self._obs(), -1.0, won or self.t >= self.max_steps, won


class ParamCartPole:
    """CartPole with a parameter dict. Defaults reproduce the textbook task."""
    obs_dim, n_actions, max_steps = 4, 2, 200
    max_return = 200.0
    DEFAULTS = dict(g=9.8, len=0.5, force=10.0, mass_pole=0.1, tau=0.02)
    BOUNDS = dict(g=(4.0, 25.0), len=(0.2, 2.0), force=(4.0, 16.0), mass_pole=(0.05, 0.5), tau=(0.01, 0.04))

    def __init__(self, **params):
        self.p = {**self.DEFAULTS, **{k: float(v) for k, v in params.items()}}
        self.name = "cartpole:" + ",".join(f"{k}={self.p[k]:.3g}" for k in sorted(self.p) if self.p[k] != self.DEFAULTS[k]) \
            if self.p != self.DEFAULTS else "cartpole"

    def reset(self, rng):
        self.s, self.t = rng.uniform(-0.05, 0.05, size=4), 0
        return self.s.copy()

    def step(self, a):
        p = self.p
        x, xd, th, thd = self.s
        force = p["force"] if int(a) == 1 else -p["force"]
        mt, ml = 1.0 + p["mass_pole"], p["mass_pole"] * p["len"]
        cos, sin = np.cos(th), np.sin(th)
        tmp = (force + ml * thd * thd * sin) / mt
        thacc = (p["g"] * sin - cos * tmp) / (p["len"] * (4.0 / 3.0 - p["mass_pole"] * cos * cos / mt))
        xacc = tmp - ml * thacc * cos / mt
        tau = p["tau"]
        self.s = np.array([x + tau * xd, xd + tau * xacc, th + tau * thd, thd + tau * thacc])
        self.t += 1
        dead = abs(self.s[0]) > 2.4 or abs(self.s[2]) > 12 * np.pi / 180
        return self.s.copy(), 1.0, dead or self.t >= self.max_steps, (self.t >= self.max_steps and not dead)


def parse_params(spec):
    out = {}
    for part in filter(None, spec.split(",")):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def make_extra_env(name):
    if name == "mountaincar":
        return MountainCar()
    if name.startswith("cartpole:"):
        return ParamCartPole(**parse_params(name.split(":", 1)[1]))
    raise KeyError(name)
