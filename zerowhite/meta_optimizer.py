"""
meta_optimizer.py — Zero White's learning algorithm is itself a small net,
and that net's own weights are tuned by evolution based on how well they
reduce loss. This is the "learns how to learn" piece: it isn't fixed SGD
with a fixed rule, it's a per-parameter update rule that keeps getting
edited.

Why evolution instead of meta-gradients (the classic `higher`-library way)?
`higher` differentiates *through* an unrolled inner training loop, which
assumes the inner network's structure stays fixed across that loop. Here the
network's structure can change every generation (architecture.py), which
would keep breaking an unrolled meta-gradient graph. A fitness-based update
(a simple Natural-Evolution-Strategy-style rule) doesn't care what the inner
network looks like — it only looks at the scalar outcome — so it survives
architecture changes for free. That's a genuine design tradeoff, not just an
excuse to avoid writing more autograd code.
"""

import copy
import numpy as np


MIN_LR, MAX_LR = 1e-4, 0.5


class LearnedOptimizer:
    def __init__(self, feat_dim=3, hidden=8, base_lr=0.05, seed=0, weight_decay=1e-3):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((feat_dim, hidden)) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, 1)) * 0.1
        self.b2 = np.zeros(1)
        self.base_lr = base_lr
        self.beta = 0.9
        self.weight_decay = weight_decay
        self.momentum = {}
        self.rng = rng

    # the update rule itself — a tiny hand-written MLP forward pass, no
    # autograd needed here since its own weights are updated by evolution,
    # not by backprop through this function
    def _rule(self, feats):
        h = np.tanh(feats @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2).ravel()

    def step(self, net, grads, max_norm=5.0):
        """Apply one update. Gradients are clipped to a global norm, and a
        non-finite gradient skips the step instead of poisoning the weights.
        Returns True if the step was applied."""
        sq = sum(float((t.grad ** 2).sum()) for t in grads.values() if t.grad is not None)
        gnorm = float(np.sqrt(sq))
        if not np.isfinite(gnorm):
            return False
        clip = min(1.0, max_norm / (gnorm + 1e-12))

        def update_fn(name, p, g):
            g = g * clip
            m_prev = self.momentum.get(name)
            if m_prev is None or m_prev.shape != g.shape:
                # a structural mutation resized/renumbered this parameter
                # since the last step — old momentum no longer applies
                m_prev = np.zeros_like(g)
            m = self.beta * m_prev + (1 - self.beta) * g
            self.momentum[name] = m
            flat_p, flat_g, flat_m = p.ravel(), g.ravel(), m.ravel()
            gscale = np.abs(flat_g).mean() + 1e-8
            feats = np.stack([flat_g / gscale, flat_m / gscale, flat_p], axis=1)
            # the rule net outputs a *correction* to plain gradient descent,
            # not the update itself — at init its weights are ~0 so the
            # correction is ~0 and this degenerates to ordinary SGD-with-
            # momentum. Evolution (meta_mutate) then searches for a better
            # correction on top of a baseline that's already known to work,
            # instead of searching blind from a random function.
            correction = np.tanh(self._rule(feats)).reshape(p.shape)
            delta = flat_m.reshape(p.shape) * (1.0 + correction)
            return p - self.base_lr * (delta + self.weight_decay * p)
        net.apply_grads(grads, update_fn)
        return True

    def clone(self):
        c = copy.copy(self)
        c.W1, c.b1, c.W2, c.b2 = self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()
        c.momentum = {k: v.copy() for k, v in self.momentum.items()}
        c.rng = np.random.default_rng(self.rng.integers(0, 2**31 - 1))
        return c

    def meta_mutate(self, std=0.05):
        """Propose a candidate revision of the learning algorithm itself."""
        c = self.clone()
        c.W1 += c.rng.standard_normal(c.W1.shape) * std
        c.W2 += c.rng.standard_normal(c.W2.shape) * std
        c.b1 += c.rng.standard_normal(c.b1.shape) * std
        c.base_lr = float(np.clip(c.base_lr * (1 + c.rng.standard_normal() * 0.25), MIN_LR, MAX_LR))
        c.momentum = {}  # a genuinely new algorithm starts with a clean momentum state
        return c

    def hyper(self):
        return dict(base_lr=self.base_lr, beta=self.beta, weight_decay=self.weight_decay)

    def halve_lr(self, factor=0.5):
        """Cool-down move: same learned rule, smaller step size. Momentum is
        kept (the direction is still valid, only the scale changes)."""
        c = self.clone()
        c.base_lr = float(np.clip(c.base_lr * factor, MIN_LR, MAX_LR))
        c.momentum = {k: v.copy() for k, v in self.momentum.items()}
        return c
