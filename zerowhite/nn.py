"""nn.py - parameter store, Adam, and small MLP helpers for models whose shape
stays fixed while they train (generators, language model, agents). The
shape-shifting classifier lives in architecture.py instead."""

import numpy as np

from .engine import Tensor


class Params:
    """name -> numpy array. leaves() makes fresh differentiable Tensors."""

    def __init__(self, arrays=None):
        self.a = {k: np.asarray(v, dtype=np.float64) for k, v in (arrays or {}).items()}

    def add(self, name, arr):
        self.a[name] = np.asarray(arr, dtype=np.float64)
        return self

    def leaves(self):
        return {k: Tensor(v, requires_grad=True) for k, v in self.a.items()}

    def clone(self):
        return Params({k: v.copy() for k, v in self.a.items()})

    def count(self):
        return int(sum(v.size for v in self.a.values()))


class Adam:
    """Adam with global-norm gradient clipping. A step whose gradient is not
    finite is skipped, so one bad batch can never poison the weights."""

    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, clip=5.0, weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.clip, self.weight_decay = clip, weight_decay
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params: Params, leaves: dict) -> bool:
        grads = {k: (t.grad if t.grad is not None else np.zeros_like(params.a[k]))
                 for k, t in leaves.items()}
        gnorm = float(np.sqrt(sum(float((g * g).sum()) for g in grads.values())))
        if not np.isfinite(gnorm):
            return False
        scale = min(1.0, self.clip / (gnorm + 1e-12))
        self.t += 1
        for k, g in grads.items():
            g = g * scale
            m = self.m.get(k)
            if m is None or m.shape != g.shape:
                m, v = np.zeros_like(g), np.zeros_like(g)
            else:
                v = self.v[k]
            m = self.b1 * m + (1 - self.b1) * g
            v = self.b2 * v + (1 - self.b2) * g * g
            self.m[k], self.v[k] = m, v
            mhat = m / (1 - self.b1 ** self.t)
            vhat = v / (1 - self.b2 ** self.t)
            p = params.a[k]
            params.a[k] = p - self.lr * (mhat / (np.sqrt(vhat) + self.eps) + self.weight_decay * p)
        return True


def init_mlp(params: Params, prefix, sizes, rng, out_scale=1.0):
    """He-initialised MLP weights named `{prefix}.W{i}` / `{prefix}.b{i}`."""
    for i in range(len(sizes) - 1):
        W = rng.standard_normal((sizes[i], sizes[i + 1])) * np.sqrt(2.0 / sizes[i])
        if i == len(sizes) - 2:
            W = W * out_scale
        params.add(f"{prefix}.W{i}", W)
        params.add(f"{prefix}.b{i}", np.zeros(sizes[i + 1]))


def n_layers(arrays, prefix):
    n = 0
    while f"{prefix}.W{n}" in arrays:
        n += 1
    return n


def mlp(leaves, prefix, x):
    """Differentiable MLP forward: ReLU between layers, linear output."""
    h = x
    n = n_layers(leaves, prefix)
    for i in range(n):
        h = h.matmul(leaves[f"{prefix}.W{i}"]) + leaves[f"{prefix}.b{i}"]
        if i < n - 1:
            h = h.relu()
    return h


def mlp_raw(arrays, prefix, x):
    """Same MLP in plain numpy (no graph) for acting / sampling."""
    h = np.asarray(x, dtype=np.float64)
    n = n_layers(arrays, prefix)
    for i in range(n):
        h = h @ arrays[f"{prefix}.W{i}"] + arrays[f"{prefix}.b{i}"]
        if i < n - 1:
            h = np.maximum(0.0, h)
    return h
