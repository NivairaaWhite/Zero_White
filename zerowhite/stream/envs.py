"""envs.py - observation wrappers for the reinforcement-learning side: STACK the last k observations, or HIDE part of the observation.

A single observation of a moving thing does not say how it is moving. CartPole's 4 numbers include the two velocities; remove them and a memoryless policy cannot balance the
pole, because the angle alone does not say whether the pole is falling or rising. Stack the last two observations and the difference is the missing velocity again.
`FrameStackEnv(env, k)` does that for ANY environment that follows the project's protocol (obs_dim, n_actions, reset(rng), step(a) -> obs, reward, done, success), so every
agent family that works from the observation vector (the policy-gradient agents, the evolved programs, the MoA policy) gets short-term memory with no change of its own.

Names (accepted by `agents.make_env`, so the CLI and the generalist can use them):

    stack3:cartpole            the last 3 observations, oldest first, concatenated
    pomdp:cartpole             CartPole without the velocities (cart position and pole angle only)
    stack2:pomdp:cartpole      both: the classical partially observable CartPole with a 2-frame memory
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

POMDP_KEEP = {"cartpole": (0, 2), "mountaincar": (0,)}                    # the observation entries that are positions (the rest are velocities)


class MaskObsEnv:
    """Show only the observation entries in `keep`."""

    def __init__(self, env, keep: Sequence[int], name: str = None):
        self.env, self.keep = env, tuple(int(i) for i in keep)
        if not self.keep or max(self.keep) >= env.obs_dim or min(self.keep) < 0:
            raise ValueError(f"keep {self.keep} does not index an observation of {env.obs_dim} entries")
        self.obs_dim, self.n_actions = len(self.keep), env.n_actions
        self.max_steps = getattr(env, "max_steps", None)
        self.max_return = getattr(env, "max_return", None)
        self.name = name or f"pomdp:{getattr(env, 'name', type(env).__name__.lower())}"

    def reset(self, rng):
        return np.asarray(self.env.reset(rng), dtype=float)[list(self.keep)]

    def step(self, a):
        o, r, done, won = self.env.step(a)
        return np.asarray(o, dtype=float)[list(self.keep)], r, done, won


class FrameStackEnv:
    """The last `k` observations as one vector (oldest first). At the start the first observation is repeated, so the vector never contains made-up zeros."""

    def __init__(self, env, k: int = 2, name: str = None):
        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1")
        self.env, self.k = env, k
        self.base_dim = int(env.obs_dim)
        self.obs_dim, self.n_actions = self.base_dim * k, env.n_actions
        self.max_steps = getattr(env, "max_steps", None)
        self.max_return = getattr(env, "max_return", None)
        self.name = name or f"stack{k}:{getattr(env, 'name', type(env).__name__.lower())}"
        self._frames = []

    def reset(self, rng):
        o = np.asarray(self.env.reset(rng), dtype=float).ravel()
        self._frames = [o] * self.k
        return np.concatenate(self._frames)

    def step(self, a):
        o, r, done, won = self.env.step(a)
        self._frames = self._frames[1:] + [np.asarray(o, dtype=float).ravel()]
        return np.concatenate(self._frames), r, done, won


def make_wrapped(name: str, make_base):
    """Parse 'stack<k>:<rest>' / 'pomdp:<rest>' (nestable) around `make_base(rest)`; raises KeyError for anything else (so agents.make_env can report it)."""
    if name.startswith("pomdp:"):
        base = name[len("pomdp:"):]
        env = make_wrapped(base, make_base)
        key = base.split(":")[0]
        if key not in POMDP_KEEP:
            raise KeyError(f"no known position entries for '{key}': pomdp: works for {sorted(POMDP_KEEP)} (use MaskObsEnv(env, keep) in code for others)")
        return MaskObsEnv(env, POMDP_KEEP[key], name=name)
    if name.startswith("stack") and ":" in name:
        head, rest = name.split(":", 1)
        try:
            k = int(head[len("stack"):])
        except ValueError:
            raise KeyError(name) from None
        return FrameStackEnv(make_wrapped(rest, make_base), k, name=name)
    return make_base(name)
