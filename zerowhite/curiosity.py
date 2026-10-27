"""
curiosity.py — a from-scratch reimplementation of the idea at the heart of
noreward-rl's Intrinsic Curiosity Module: keep a forward model that predicts
the consequence of an action, and use its prediction error as a bonus signal.

noreward-rl uses this to reward an RL agent for visiting states its forward
model finds surprising. Zero White has no environment to explore — the thing
it explores is its own space of possible self-modifications — so this module
predicts "how much will this candidate modification help?" from a small
description of the system's current state, and its prediction error becomes
a novelty bonus in zero_white.py's candidate scoring. That keeps the system
from only ever repeating the one safe move it already knows about; it also
gets nudged toward trying modifications whose effect it can't yet predict
well, and the forward model itself keeps improving from every generation's
real outcome.
"""

import numpy as np


class IntrinsicCuriosityModule:
    def __init__(self, state_dim, hidden=8, lr=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((state_dim, hidden)) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, 1)) * 0.1
        self.b2 = np.zeros(1)
        self.lr = lr

    def predict(self, state):
        h = np.tanh(state @ self.W1 + self.b1)
        return float((h @ self.W2 + self.b2).ravel()[0])

    def update(self, state, true_outcome):
        """One hand-written gradient step on squared prediction error —
        small enough not to need the full autograd engine."""
        h_pre = state @ self.W1 + self.b1
        h = np.tanh(h_pre)
        pred = float((h @ self.W2 + self.b2).ravel()[0])
        err = pred - true_outcome

        dW2 = err * h.reshape(-1, 1)
        db2 = np.array([err])
        dh = err * self.W2.ravel()
        dpre = dh * (1 - h * h)
        dW1 = np.outer(state, dpre)
        db1 = dpre

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        return abs(err)


class ForwardDynamicsCuriosity:
    """The agent-side curiosity module, closer to noreward-rl's ICM: a forward
    model predicts the next observation from (observation, action); its
    prediction error is the intrinsic reward. States the model cannot yet
    predict pay a bonus, so an agent with sparse external reward is pulled
    toward unexplored territory. Built on Zero White's own autograd."""

    def __init__(self, obs_dim, n_actions, hidden=32, lr=3e-3, seed=0):
        from .nn import Params, Adam, init_mlp
        rng = np.random.default_rng(seed)
        self.n_actions, self.obs_dim = n_actions, obs_dim
        self.params = Params()
        init_mlp(self.params, "fwd", [obs_dim + n_actions, hidden, obs_dim], rng)
        self.opt = Adam(lr=lr)

    def _inp(self, obs, actions):
        onehot = np.eye(self.n_actions)[np.asarray(actions, dtype=np.int64)]
        return np.concatenate([obs, onehot], axis=1)

    def bonus(self, obs, actions, next_obs):
        from .nn import mlp_raw
        pred = mlp_raw(self.params.a, "fwd", self._inp(obs, actions))
        return ((pred - next_obs) ** 2).mean(axis=1)

    def update(self, obs, actions, next_obs, epochs=1):
        from .engine import Tensor, mse_loss
        from .nn import mlp
        x = self._inp(obs, actions)
        for _ in range(epochs):
            leaves = self.params.leaves()
            loss = mse_loss(mlp(leaves, "fwd", Tensor(x)), next_obs)
            loss.backward()
            self.opt.step(self.params, leaves)
        return float(loss.data)
