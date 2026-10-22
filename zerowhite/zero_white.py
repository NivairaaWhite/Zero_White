"""
zero_white.py - Zero White itself (v3): the self-modifying classifier core.

Every generation it:
  1. builds the feasibility mask (which self-modifications can do anything
     right now) and asks its self-model which subset to try (always including
     `noop`, the control),
  2. clones itself once per chosen move and gives every clone the same short
     probation on the same batch sequence (a fair race),
  3. scores each clone on held-out data: improvement + curiosity bonus
     - complexity penalty - learning-rate penalty (divergence guard),
  4. keeps the winner, teaches its policy from each move's ADVANTAGE over the
     control, then trains the winning lineage a little further
     (`consolidate_steps`) so real datasets actually get learned,
  5. updates tabu/cooldown bookkeeping.

Selection uses the validation split only. A separate test split (see
datasets.py) is never touched by selection, so reported test accuracy is
honest.
"""

import numpy as np

from .architecture import GrowableNet
from .meta_optimizer import LearnedOptimizer
from .curiosity import IntrinsicCuriosityModule
from .self_model import SelfModel
from .engine import softmax_cross_entropy, mse_loss


class ZeroWhite:
    def __init__(self, in_dim, out_dim, hidden=(6,), task="classification",
                 num_chunks=4, seed=0, candidates_per_gen=5,
                 lr_soft_cap=0.15, lr_penalty=0.02, lr_floor=0.01,
                 self_model=None, opt=None, batch_size=None,
                 consolidate_steps=0, patch_dims=(6, 8)):
        self.task = task
        self.net = GrowableNet(in_dim, out_dim, hidden=hidden, num_chunks=num_chunks, seed=seed,
                               patch_local=patch_dims[0], patch_global=patch_dims[1])
        self.opt = opt or LearnedOptimizer(seed=seed)
        self.self_model = self_model or SelfModel()
        self.curiosity = IntrinsicCuriosityModule(state_dim=5, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.candidates_per_gen = candidates_per_gen
        self.lr_soft_cap = lr_soft_cap
        self.lr_penalty = lr_penalty
        self.lr_floor = lr_floor
        self.batch_size = batch_size            # None = full batch
        self.consolidate_steps = consolidate_steps
        self.steps_trained = 0

    # ---- loss / eval ---------------------------------------------------------
    def _forward_loss(self, net, X, y):
        logits, params = net.forward(X)
        if self.task == "classification":
            loss, probs = softmax_cross_entropy(logits, y)
            acc = float((probs.argmax(axis=1) == y).mean())
        else:
            loss, acc = mse_loss(logits, y), None
        return loss, params, acc

    def train_steps(self, net, opt, X, y, steps, rng=None):
        rng = rng or self.rng
        n, bs = len(X), self.batch_size
        loss_val, acc = None, None
        for _ in range(steps):
            if bs and bs < n:
                idx = rng.integers(0, n, size=bs)
                xb, yb = X[idx], y[idx]
            else:
                xb, yb = X, y
            loss_t, params, acc = self._forward_loss(net, xb, yb)
            loss_t.backward()
            opt.step(net, params)
            loss_val = float(loss_t.data)
        return loss_val, acc

    def evaluate(self, net, X, y):
        """(loss, accuracy) on any data, computed without building a graph."""
        with np.errstate(all="ignore"):
            logits = net.predict_raw(X)
            if self.task == "classification":
                z = logits - logits.max(axis=1, keepdims=True)
                logp = z[np.arange(len(y)), y] - np.log(np.exp(z).sum(axis=1))
                return float(-logp.mean()), float((logits.argmax(axis=1) == y).mean())
            return float(((logits - y) ** 2).mean()), None

    def predict(self, X):
        return self.net.predict_raw(X).argmax(axis=1)

    # ---- mask ----------------------------------------------------------------
    def feasible_actions(self, net=None, opt=None):
        net, opt = net or self.net, opt or self.opt
        m = self.self_model
        n_hidden = len(net.layers) - 1
        return {
            "grow": n_hidden > 0 and max(net.hidden_sizes() or [0]) < m.max_hidden_units,
            "prune": any(w > 1 for w in net.hidden_sizes()),
            "add_layer": 2 <= len(net.layers) < m.max_layers,
            "prune_layer": n_hidden > 1,
            "perturb": True,
            "mutate_opt": True,
            "decay_lr": opt.base_lr > self.lr_floor * 2,   # never "decay" into not-learning
            "patch_on": (not net.use_patch) and net.patch is not None,
            "patch_off": net.use_patch,
            "noop": True,
        }

    def _apply_action(self, act, net, opt):
        if act == "grow":
            net.grow_neurons(max_units=self.self_model.max_hidden_units)
        elif act == "prune":
            net.prune_neuron()
        elif act == "add_layer":
            net.add_layer(max_layers=self.self_model.max_layers)
        elif act == "prune_layer":
            net.prune_layer()
        elif act == "perturb":
            net.perturb_weights(std=0.05)
        elif act == "mutate_opt":
            opt = opt.meta_mutate()
        elif act == "decay_lr":
            opt = opt.halve_lr()
        elif act == "patch_on" and not net.use_patch:
            net.toggle_hierarchical()
        elif act == "patch_off" and net.use_patch:
            net.toggle_hierarchical()
        return net, opt

    def _state_features(self, net, opt, val_loss):
        # every feature is O(1) so the curiosity net's tanh units never saturate
        return np.array([len(net.layers) / 6.0, np.log1p(net.num_params()) / 10.0,
                         opt.base_lr, min(float(val_loss), 10.0) / 10.0,
                         np.log1p(self.self_model.generation) / 6.0], dtype=np.float64)

    # ---- one generation ------------------------------------------------------
    def self_improve_generation(self, Xtr, ytr, Xval, yval, probation_steps=15):
        sm = self.self_model
        sm.generation += 1
        baseline_val_loss, _ = self.evaluate(self.net, Xval, yval)
        base_params = max(1, self.net.num_params())

        feasible = self.feasible_actions()
        tried = sm.select_actions(feasible, self.rng, k=self.candidates_per_gen)
        batch_seed = int(self.rng.integers(0, 2**31 - 1))   # same batches for every candidate

        scored = []
        for act in tried:
            net_c, opt_c = self.net.clone(), self.opt.clone()
            net_c, opt_c = self._apply_action(act, net_c, opt_c)
            self.train_steps(net_c, opt_c, Xtr, ytr, probation_steps, np.random.default_rng(batch_seed))
            v_loss, v_acc = self.evaluate(net_c, Xval, yval)

            if not np.isfinite(v_loss):          # diverged - never selectable
                scored.append((act, -1e9, -1e9, net_c, opt_c, v_loss, v_acc))
                continue

            improvement = baseline_val_loss - v_loss
            surprise = self.curiosity.update(self._state_features(net_c, opt_c, v_loss), improvement)
            complexity_frac = (net_c.num_params() - base_params) / base_params
            lr_pen = self.lr_penalty * max(0.0, np.log(opt_c.base_lr / self.lr_soft_cap))
            score = improvement + 0.05 * surprise \
                    - 0.005 * max(0.0, complexity_frac) \
                    + 0.004 * max(0.0, -complexity_frac) \
                    - lr_pen
            scored.append((act, score, improvement, net_c, opt_c, v_loss, v_acc))

        control = next(s for a, s, *_ in scored if a == "noop")
        for a, s, *_ in scored:
            # advantage over plain training, clipped BOTH ways: if the control
            # diverged (-1e9) an unclipped advantage would be +1e9 and freeze the policy
            sm.observe_outcome(a, float(np.clip(s - control, -1.0, 1.0)))

        scored.sort(key=lambda t: t[1], reverse=True)
        act, score, improvement, net_c, opt_c, v_loss, v_acc = scored[0]
        sm.record_result(tried, act)
        self.net, self.opt = net_c, opt_c
        self.steps_trained += probation_steps

        if self.consolidate_steps:
            self.train_steps(self.net, self.opt, Xtr, ytr, self.consolidate_steps)
            self.steps_trained += self.consolidate_steps
            v_loss, v_acc = self.evaluate(self.net, Xval, yval)

        sm.log({
            "action": act, "tried": tried, "score": score, "improvement": improvement,
            "val_loss": v_loss, "val_acc": v_acc, **self.net.summary(),
            "opt_base_lr": self.opt.base_lr, "action_weights": dict(sm.action_weights),
            "benched": [a for a in sm.ACTIONS if sm.bench_until[a] > sm.generation],
        })
        return act, v_loss, v_acc

    def run(self, Xtr, ytr, Xval, yval, generations=40, probation_steps=15, verbose=True,
            on_generation=None):
        for g in range(generations):
            act, v_loss, v_acc = self.self_improve_generation(Xtr, ytr, Xval, yval, probation_steps)
            if verbose:
                self._print(g + 1, act, v_loss, v_acc)
            if on_generation:
                on_generation(self)
        return self.self_model

    def _print(self, g, act, v_loss, v_acc, tag=""):
        acc_s = f"{v_acc:.3f}" if v_acc is not None else "n/a"
        print(f"[gen {g:03d}]{tag} chose={act:11s} val_loss={v_loss:.4f} val_acc={acc_s} "
              f"layers={len(self.net.layers)} hidden={self.net.hidden_sizes()} "
              f"patch={self.net.use_patch} lr={self.opt.base_lr:.3f} params={self.net.num_params()}")


def run_curriculum(zw, curriculum, max_generations=150, probation_steps=20,
                   verbose=True, on_generation=None):
    """Train through a Curriculum until its last level is solved or the budget ends."""
    start = zw.self_model.generation
    for _ in range(max_generations):
        Xtr, ytr, Xval, yval = curriculum.data()
        act, v_loss, v_acc = zw.self_improve_generation(Xtr, ytr, Xval, yval, probation_steps)
        level_before = curriculum.level
        promoted = curriculum.step(v_acc, zw.net)
        zw.self_model.history[-1]["level"] = level_before
        if verbose:
            zw._print(zw.self_model.generation, act, v_loss, v_acc,
                      tag=f"[{curriculum.levels[level_before]['name']}]")
            if promoted:
                print(f"   >>> promoted to {curriculum.cfg['name']}")
        if on_generation:
            on_generation(zw, curriculum)
        if curriculum.finished:
            break
    return zw.self_model.generation - start
