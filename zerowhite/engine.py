"""
engine.py - Zero White's own tensor/autograd engine.

No PyTorch, TensorFlow, JAX or tinygrad. numpy stores and multiplies the
arrays; everything else (the autodiff tape, every op, every loss) is written
here by hand, micrograd-style: a graph of Tensors with backward closures,
walked in reverse topological order.

v3 changes
  * backward() is iterative (the recursive version died on deep graphs);
  * `requires_grad` is real: constants (input batches, targets) no longer get
    gradients computed for them, which roughly halves the matmul cost of the
    first layer on wide inputs;
  * stable sigmoid, `clip`, `embedding`, and fused `bce_with_logits` and
    `policy_gradient_loss` ops (needed by the generative models and agents).
"""

import numpy as np


def _noop():
    return None


def _unbroadcast(grad, shape):
    """Sum a gradient back down to `shape` after numpy broadcasting expanded it."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


def _wrap(x):
    return x if isinstance(x, Tensor) else Tensor(x)


class Tensor:
    """A numpy array that remembers how it was built so it can backprop."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev")

    def __init__(self, data, requires_grad=False, _children=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._backward = _noop
        self._prev = tuple(_children)

    @property
    def shape(self):
        return self.data.shape

    # ---- graph plumbing -----------------------------------------------------
    def backward(self):
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            v, expanded = stack.pop()
            if expanded:
                topo.append(v)
                continue
            if id(v) in seen:
                continue
            seen.add(id(v))
            stack.append((v, True))
            for c in v._prev:
                if id(c) not in seen:
                    stack.append((c, False))
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            if v.requires_grad and v.grad is not None:
                v._backward()

    def _accumulate(self, g):
        if not self.requires_grad:
            return
        self.grad = g if self.grad is None else self.grad + g

    def _out(self, data, *others):
        rg = self.requires_grad or any(o.requires_grad for o in others)
        return Tensor(data, rg, (self,) + others)

    # ---- elementwise ----------------------------------------------------------
    def __add__(self, other):
        other = _wrap(other)
        out = self._out(self.data + other.data, other)

        def _backward():
            if self.requires_grad:
                self._accumulate(_unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(out.grad, other.data.shape))

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = _wrap(other)
        out = self._out(self.data * other.data, other)

        def _backward():
            if self.requires_grad:
                self._accumulate(_unbroadcast(out.grad * other.data, self.data.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(out.grad * self.data, other.data.shape))

        out._backward = _backward
        return out

    def __pow__(self, p):
        out = self._out(self.data ** p)

        def _backward():
            self._accumulate(out.grad * p * (self.data ** (p - 1)))

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-_wrap(other))

    def __rsub__(self, other):
        return (-self) + other

    __radd__ = __add__
    __rmul__ = __mul__

    def __truediv__(self, other):
        return self * (_wrap(other) ** -1.0)

    # ---- matmul -----------------------------------------------------------------
    def matmul(self, other):
        other = _wrap(other)
        out = self._out(self.data @ other.data, other)

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad @ other.data.T)
            if other.requires_grad:
                other._accumulate(self.data.T @ out.grad)

        out._backward = _backward
        return out

    __matmul__ = matmul

    # ---- reductions ---------------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = self._out(self.data.sum(axis=axis, keepdims=keepdims))

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accumulate(np.ones_like(self.data) * g)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- nonlinearities ---------------------------------------------------------------
    def relu(self):
        out = self._out(np.maximum(0.0, self.data))

        def _backward():
            self._accumulate(out.grad * (self.data > 0))

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = self._out(t)

        def _backward():
            self._accumulate(out.grad * (1 - t * t))

        out._backward = _backward
        return out

    def sigmoid(self):
        s = _stable_sigmoid(self.data)
        out = self._out(s)

        def _backward():
            self._accumulate(out.grad * s * (1 - s))

        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = self._out(e)

        def _backward():
            self._accumulate(out.grad * e)

        out._backward = _backward
        return out

    def log(self):
        out = self._out(np.log(self.data + 1e-12))

        def _backward():
            self._accumulate(out.grad / (self.data + 1e-12))

        out._backward = _backward
        return out

    def clip(self, lo, hi):
        out = self._out(np.clip(self.data, lo, hi))

        def _backward():
            self._accumulate(out.grad * ((self.data >= lo) & (self.data <= hi)))

        out._backward = _backward
        return out

    def reshape(self, *shape):
        orig = self.data.shape
        out = self._out(self.data.reshape(*shape))

        def _backward():
            self._accumulate(out.grad.reshape(orig))

        out._backward = _backward
        return out


def _stable_sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))),
                    np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def sigmoid_np(x):
    return _stable_sigmoid(np.asarray(x, dtype=np.float64))


def embedding(table: Tensor, idx) -> Tensor:
    """Row lookup: out.shape == idx.shape + (dim,). Backward scatter-adds."""
    idx = np.asarray(idx, dtype=np.int64)
    out = Tensor(table.data[idx], table.requires_grad, (table,))

    def _backward():
        g = np.zeros_like(table.data)
        np.add.at(g, idx.reshape(-1), out.grad.reshape(-1, table.data.shape[1]))
        table._accumulate(g)

    out._backward = _backward
    return out


# ---- fused losses ------------------------------------------------------------------
def softmax_cross_entropy(logits: Tensor, targets: np.ndarray):
    """Fused, stable softmax + NLL. targets = int class ids, shape (N,).
    Returns (loss Tensor, probs ndarray)."""
    z = logits.data - logits.data.max(axis=-1, keepdims=True)
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum(axis=-1, keepdims=True)
    n = logits.data.shape[0]
    loss_val = (-np.log(probs[np.arange(n), targets] + 1e-12)).mean()
    out = Tensor(loss_val, logits.requires_grad, (logits,))

    def _backward():
        d = probs.copy()
        d[np.arange(n), targets] -= 1.0
        d /= n
        logits._accumulate(d * out.grad)

    out._backward = _backward
    return out, probs


def mse_loss(pred: Tensor, target: np.ndarray) -> Tensor:
    diff = pred - Tensor(target)
    return (diff * diff).mean()


def bce_with_logits(logits: Tensor, targets: np.ndarray, sum_features=True) -> Tensor:
    """Stable binary cross-entropy on logits. Targets in [0,1].
    sum_features=True: sum over the last axis, mean over the batch (the form a
    VAE reconstruction term wants). False: mean over every element."""
    z, t = logits.data, np.asarray(targets, dtype=np.float64)
    per = np.maximum(z, 0) - z * t + np.log1p(np.exp(-np.abs(z)))
    denom = z.shape[0] if sum_features else z.size
    out = Tensor(per.sum() / denom, logits.requires_grad, (logits,))

    def _backward():
        logits._accumulate((_stable_sigmoid(z) - t) / denom * out.grad)

    out._backward = _backward
    return out


def policy_gradient_loss(logits: Tensor, actions, advantages, entropy_coef=0.0) -> Tensor:
    """-mean(adv * log pi(a|s)) - entropy_coef * mean(entropy), fused and stable."""
    actions = np.asarray(actions, dtype=np.int64)
    adv = np.asarray(advantages, dtype=np.float64)
    z = logits.data - logits.data.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    p = np.exp(logp)
    n = len(actions)
    ent = -(p * logp).sum(axis=-1)
    loss_val = -(adv * logp[np.arange(n), actions]).mean() - entropy_coef * ent.mean()
    out = Tensor(loss_val, logits.requires_grad, (logits,))

    def _backward():
        d = p.copy()
        d[np.arange(n), actions] -= 1.0
        g = adv[:, None] * d / n + entropy_coef * p * (logp + ent[:, None]) / n
        logits._accumulate(g * out.grad)

    out._backward = _backward
    return out
