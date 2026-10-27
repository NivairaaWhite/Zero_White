"""A custom environment for `zerowhite solve --env-file examples/catch_env.py`.

The contract (this is all Zero White needs from YOUR environment):

    class Env:
        obs_dim: int            length of the observation vector
        n_actions: int          discrete actions 0..n_actions-1
        def reset(self, rng) -> obs            rng is a numpy Generator; use it for ALL randomness
        def step(self, action) -> (obs, reward, done, success)
    MAX_RETURN = ...            optional: the return that counts as solved (used to normalise skill)

NOTE: Zero White executes this file as-is, in-process, with your permissions. It is your code, not evolved code.
"""
import numpy as np

MAX_RETURN = 10.0


class Env:
    """Catch: a ball falls down a 9-wide board; move the paddle (left / stay / right) under it. 10 balls per game."""
    obs_dim, n_actions, name = 3, 3, "catch"

    def reset(self, rng):
        self.rng, self.paddle, self.balls = rng, 4, 0
        self._drop()
        return self._obs()

    def _drop(self):
        self.bx, self.by = int(self.rng.integers(0, 9)), 0

    def _obs(self):
        return np.array([self.bx / 8.0, self.by / 8.0, self.paddle / 8.0])

    def step(self, a):
        self.paddle = int(np.clip(self.paddle + (int(a) - 1), 0, 8))
        self.by += 1
        reward = 0.0
        if self.by == 8:
            reward = 1.0 if abs(self.paddle - self.bx) <= 0 else 0.0
            self.balls += 1
            self._drop()
        done = self.balls >= 10
        return self._obs(), reward, done, done and self.balls == 10
