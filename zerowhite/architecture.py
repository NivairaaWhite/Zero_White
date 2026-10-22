"""
architecture.py — the part of Zero White that can change its own shape.

Direct-encoding, NEAT-inspired structural mutations (grow a neuron, prune a
neuron, add a layer, prune a layer, perturb weights) operating on plain numpy
weight matrices — this is a reimplementation of the *ideas* in PyTorch-NEAT,
not a wrapper around it (PyTorch-NEAT itself imports torch, which is off
limits here).

Also includes an optional two-level "patch" block — a shared local MLP
applied to chunks of the input, whose outputs are pooled and fed through a
global MLP — which is a small, from-scratch echo of MEGABYTE's local/global
hierarchy (patch model + global model over patch summaries), reimplemented
without megabyte-pytorch's torch dependency. The self-modifier can turn this
on or off as one of the structural moves it is allowed to try.
"""

import copy
import numpy as np

from .engine import Tensor, softmax_cross_entropy


def _he(fan_in, fan_out, rng):
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)


class GrowableNet:
    def __init__(self, in_dim, out_dim, hidden=(8,), num_chunks=4, seed=0,
                 patch_local=6, patch_global=8):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_chunks = num_chunks
        self.patch_local = patch_local
        self.patch_global = patch_global
        self.rng = np.random.default_rng(seed)

        dims = [in_dim] + list(hidden) + [out_dim]
        self.layers = []  # list of {'W': (fan_in,fan_out), 'b': (fan_out,)}
        for i in range(len(dims) - 1):
            self.layers.append({
                "W": _he(dims[i], dims[i + 1], self.rng),
                "b": np.zeros(dims[i + 1]),
            })

        self.use_patch = False
        self.patch = None  # dict of Wl,bl,Wg,bg when enabled
        self._maybe_init_patch_params()

    # ---- bookkeeping -----------------------------------------------------
    def hidden_sizes(self):
        return [l["W"].shape[1] for l in self.layers[:-1]]

    def num_params(self):
        n = sum(l["W"].size + l["b"].size for l in self.layers)
        if self.use_patch:
            n += sum(self.patch[k].size for k in ("Wl", "bl", "Wg", "bg"))
        return n

    def summary(self):
        return {
            "hidden_sizes": self.hidden_sizes(),
            "num_layers": len(self.layers),
            "patch_enabled": self.use_patch,
            "num_params": self.num_params(),
        }

    def clone(self):
        c = copy.copy(self)
        c.layers = [{"W": l["W"].copy(), "b": l["b"].copy()} for l in self.layers]
        c.patch = {k: v.copy() for k, v in self.patch.items()} if self.patch else None
        c.rng = np.random.default_rng(self.rng.integers(0, 2**31 - 1))
        return c

    # ---- forward -----------------------------------------------------
    def forward(self, X):
        """Returns (output_tensor, params_dict): params_dict maps name -> leaf
        Tensor so the caller can read .grad after backward() and write updates
        back into this net's numpy arrays via apply_grads()."""
        return self._forward(Tensor(X))

    def predict_raw(self, X):
        """Pure-numpy forward pass (no autograd graph) - used for evaluation,
        acting in environments and generation, where no gradient is needed."""
        h = np.asarray(X, dtype=np.float64)
        if self.use_patch:
            n = h.shape[0]
            xr = h.reshape(n * self.num_chunks, self.in_dim // self.num_chunks)
            local = np.maximum(0.0, xr @ self.patch["Wl"] + self.patch["bl"])
            local = local.reshape(n, -1)
            h = np.maximum(0.0, local @ self.patch["Wg"] + self.patch["bg"])
        for i, layer in enumerate(self.layers):
            h = h @ layer["W"] + layer["b"]
            if i < len(self.layers) - 1:
                h = np.maximum(0.0, h)
        return h

    def input_gradient(self, X, y):
        """d(loss)/d(input) for classification — used to craft adversarial
        examples (FGSM) without any external library."""
        x_t = Tensor(X, requires_grad=True)
        logits, _ = self._forward(x_t)
        loss, _ = softmax_cross_entropy(logits, y)
        loss.backward()
        return x_t.grad

    def _forward(self, h):
        params = {}

        if self.use_patch:
            n = h.shape[0]
            chunk_dim = self.in_dim // self.num_chunks
            xr = h.reshape(n * self.num_chunks, chunk_dim)
            Wl = Tensor(self.patch["Wl"], requires_grad=True); params["patch.Wl"] = Wl
            bl = Tensor(self.patch["bl"], requires_grad=True); params["patch.bl"] = bl
            local = (xr.matmul(Wl) + bl).relu()
            local = local.reshape(n, self.num_chunks * self.patch["Wl"].shape[1])
            Wg = Tensor(self.patch["Wg"], requires_grad=True); params["patch.Wg"] = Wg
            bg = Tensor(self.patch["bg"], requires_grad=True); params["patch.bg"] = bg
            h = (local.matmul(Wg) + bg).relu()

        for i, layer in enumerate(self.layers):
            Wt = Tensor(layer["W"], requires_grad=True); params[f"L{i}.W"] = Wt
            bt = Tensor(layer["b"], requires_grad=True); params[f"L{i}.b"] = bt
            h = h.matmul(Wt) + bt
            if i < len(self.layers) - 1:
                h = h.relu()
        return h, params

    def apply_grads(self, params, update_fn):
        """update_fn(name, param_array, grad_array) -> new_param_array"""
        for name, t in params.items():
            g = t.grad if t.grad is not None else np.zeros_like(t.data)
            new_val = update_fn(name, t.data, g)
            if name.startswith("patch."):
                self.patch[name.split(".")[1]] = new_val
            else:
                idx, field = name.split(".")
                self.layers[int(idx[1:])][field] = new_val

    # ---- structural mutations (the NEAT-inspired part) --------------------
    def grow_neurons(self, k=None, layer_idx=None, max_units=64):
        """Grow a layer by k units (default: ~25% of its width, at least 2),
        capped at max_units. Function-preserving, like grow_neuron."""
        n_hidden = len(self.layers) - 1
        if n_hidden == 0:
            return False
        if layer_idx is None:
            room = [i for i in range(n_hidden) if self.layers[i]["W"].shape[1] < max_units]
            if not room:
                return False
            layer_idx = int(self.rng.choice(room))
        width = self.layers[layer_idx]["W"].shape[1]
        k = k or max(2, int(np.ceil(0.25 * width)))
        k = min(k, max_units - width)
        for _ in range(max(0, k)):
            self.grow_neuron(layer_idx)
        return k > 0

    def grow_neuron(self, layer_idx=None):
        """Add one hidden unit at layer_idx (0-indexed among hidden layers).
        New unit's outgoing weights start at exactly zero, so network's
        function is unchanged at the instant of growth (safe / Net2Net-style
        growth) — only future training lets the new unit start contributing."""
        n_hidden = len(self.layers) - 1
        if n_hidden == 0:
            return False
        if layer_idx is None:
            layer_idx = int(self.rng.integers(0, n_hidden))
        cur = self.layers[layer_idx]
        nxt = self.layers[layer_idx + 1]
        new_col = self.rng.standard_normal((cur["W"].shape[0], 1)) * 0.1
        cur["W"] = np.concatenate([cur["W"], new_col], axis=1)
        cur["b"] = np.concatenate([cur["b"], np.zeros(1)])
        new_row = np.zeros((1, nxt["W"].shape[1]))  # zero => no effect yet
        nxt["W"] = np.concatenate([nxt["W"], new_row], axis=0)
        return True

    def prune_neuron(self, layer_idx=None):
        """Remove the least-useful hidden unit (smallest outgoing weight
        norm) from a hidden layer, if it has more than one unit."""
        n_hidden = len(self.layers) - 1
        candidates = [i for i in range(n_hidden) if self.layers[i]["W"].shape[1] > 1]
        if not candidates:
            return False
        layer_idx = layer_idx if layer_idx in candidates else int(self.rng.choice(candidates))
        cur = self.layers[layer_idx]
        nxt = self.layers[layer_idx + 1]
        saliency = np.linalg.norm(nxt["W"], axis=1)
        victim = int(np.argmin(saliency))
        cur["W"] = np.delete(cur["W"], victim, axis=1)
        cur["b"] = np.delete(cur["b"], victim)
        nxt["W"] = np.delete(nxt["W"], victim, axis=0)
        return True

    def add_layer(self, max_layers=6):
        """Insert a hidden layer between two existing hidden/output layers.
        The incoming activations are post-ReLU (>= 0), so a near-identity
        layer (+ small noise) reproduces the old function almost exactly."""
        if len(self.layers) >= max_layers or len(self.layers) < 2:
            return False
        idx = int(self.rng.integers(1, len(self.layers)))  # before layers[idx], idx >= 1
        width = self.layers[idx]["W"].shape[0]
        W = np.eye(width) + self.rng.standard_normal((width, width)) * 0.01
        self.layers.insert(idx, {"W": W, "b": np.zeros(width)})
        return True

    def prune_layer(self, min_layers=1):
        """Remove a whole hidden layer (structural simplification). The
        following layer's input weights are re-initialised since a clean
        weight transfer isn't possible when an entire layer disappears."""
        n_hidden = len(self.layers) - 1
        if n_hidden <= min_layers:
            return False
        idx = int(self.rng.integers(0, n_hidden))
        removed = self.layers.pop(idx)
        nxt = self.layers[idx]
        fan_in = removed["W"].shape[0]
        nxt["W"] = _he(fan_in, nxt["W"].shape[1], self.rng)
        return True

    def perturb_weights(self, std=0.05):
        """NEAT-style weight mutation: gaussian jitter, no shape change."""
        for l in self.layers:
            l["W"] = l["W"] + self.rng.standard_normal(l["W"].shape) * std
            l["b"] = l["b"] + self.rng.standard_normal(l["b"].shape) * std
        return True

    def toggle_hierarchical(self):
        """Turn the local/global patch block on or off. Turning it on
        reroutes the input through it and re-shapes layer[0] to match, which
        is a disruptive change the self-modifier must validate via probation
        (see zero_white.py) before it's allowed to stick."""
        if self.in_dim % self.num_chunks != 0:
            return False
        self.use_patch = not self.use_patch
        if self.use_patch:
            self._maybe_init_patch_params(force=True)
            global_dim = self.patch["Wg"].shape[1]
            old0 = self.layers[0]
            self.layers[0] = {
                "W": _he(global_dim, old0["W"].shape[1], self.rng),
                "b": np.zeros(old0["W"].shape[1]),
            }
        else:
            old0 = self.layers[0]
            self.layers[0] = {
                "W": _he(self.in_dim, old0["W"].shape[1], self.rng),
                "b": np.zeros(old0["W"].shape[1]),
            }
        return True

    def _maybe_init_patch_params(self, force=False):
        if self.patch is not None and not force:
            return
        if self.in_dim % self.num_chunks != 0:
            self.patch = None
            return
        chunk_dim = self.in_dim // self.num_chunks
        local_dim, global_dim = self.patch_local, self.patch_global
        self.patch = {
            "Wl": _he(chunk_dim, local_dim, self.rng),
            "bl": np.zeros(local_dim),
            "Wg": _he(self.num_chunks * local_dim, global_dim, self.rng),
            "bg": np.zeros(global_dim),
        }
