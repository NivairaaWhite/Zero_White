"""
test_basic.py — plain-assert test suite. No pytest, no unittest even; just
functions that raise if something's wrong, run top to bottom. Same
zero-dependency spirit as the rest of Zero White.

Run with:  python3 tests/test_basic.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zerowhite.engine import Tensor, mse_loss, softmax_cross_entropy  # noqa: E402
from zerowhite.architecture import GrowableNet  # noqa: E402
from zerowhite.meta_optimizer import LearnedOptimizer  # noqa: E402
from zerowhite.curiosity import IntrinsicCuriosityModule  # noqa: E402
from zerowhite.self_model import SelfModel  # noqa: E402
from zerowhite.zero_white import ZeroWhite  # noqa: E402


def test_engine_linear_regression_converges():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 1))
    y = 3 * X + 2
    W, b = rng.standard_normal((1, 1)) * 0.1, np.zeros((1, 1))
    for _ in range(300):
        Wt, bt = Tensor(W, requires_grad=True), Tensor(b, requires_grad=True)
        pred = Tensor(X).matmul(Wt) + bt
        loss = mse_loss(pred, y)
        loss.backward()
        W = W - 0.1 * Wt.grad
        b = b - 0.1 * bt.grad
    assert abs(W.item() - 3.0) < 0.05
    assert loss.data < 1e-3, f"expected convergence, got loss={loss.data}"


def test_engine_softmax_xent_grad_shape_and_sign():
    logits = Tensor(np.array([[2.0, 0.0, 0.0]]), requires_grad=True)
    loss, probs = softmax_cross_entropy(logits, np.array([0]))
    loss.backward()
    assert logits.grad.shape == (1, 3)
    # gradient wrt the correct class's logit should be negative (increasing
    # it further would lower the loss)
    assert logits.grad[0, 0] < 0


def test_engine_matmul_backward_shapes():
    a = Tensor(np.random.randn(4, 3), requires_grad=True)
    b = Tensor(np.random.randn(3, 5), requires_grad=True)
    c = a.matmul(b).sum()
    c.backward()
    assert a.grad.shape == (4, 3)
    assert b.grad.shape == (3, 5)


def test_grow_neuron_is_function_preserving_at_the_instant_of_growth():
    net = GrowableNet(in_dim=5, out_dim=2, hidden=(4,), seed=0)
    X = np.random.default_rng(0).standard_normal((7, 5))
    before, _ = net.forward(X)
    net.grow_neuron(0)
    after, _ = net.forward(X)
    assert np.allclose(before.data, after.data, atol=1e-8), \
        "growing a neuron with a zero outgoing row must not change the output"


def test_prune_then_forward_shapes_stay_consistent():
    net = GrowableNet(in_dim=6, out_dim=3, hidden=(5, 4), seed=1)
    X = np.random.default_rng(1).standard_normal((3, 6))
    assert net.prune_neuron(0)
    out, _ = net.forward(X)
    assert out.data.shape == (3, 3)


def test_add_then_prune_layer_roundtrip_shapes():
    net = GrowableNet(in_dim=4, out_dim=2, hidden=(4,), seed=2)
    n0 = len(net.layers)
    assert net.add_layer()
    assert len(net.layers) == n0 + 1
    assert net.prune_layer()
    assert len(net.layers) == n0
    X = np.random.default_rng(2).standard_normal((5, 4))
    out, _ = net.forward(X)
    assert out.data.shape == (5, 2)


def test_hierarchical_toggle_roundtrip():
    net = GrowableNet(in_dim=8, out_dim=2, hidden=(4,), num_chunks=4, seed=3)
    X = np.random.default_rng(3).standard_normal((5, 8))
    assert net.toggle_hierarchical()
    assert net.use_patch
    out, _ = net.forward(X)
    assert out.data.shape == (5, 2)
    assert net.toggle_hierarchical()
    assert not net.use_patch
    out2, _ = net.forward(X)
    assert out2.data.shape == (5, 2)


def test_clone_is_a_true_independent_copy():
    net = GrowableNet(in_dim=4, out_dim=2, hidden=(3,), seed=4)
    clone = net.clone()
    clone.grow_neuron(0)
    assert net.hidden_sizes() != clone.hidden_sizes()
    clone.layers[0]["W"][0, 0] = 999.0
    assert net.layers[0]["W"][0, 0] != 999.0


def test_optimizer_step_reduces_loss_on_a_toy_problem():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((40, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    net = GrowableNet(in_dim=4, out_dim=2, hidden=(6,), seed=5)
    opt = LearnedOptimizer(seed=5)

    def loss_of(net):
        logits, _ = net.forward(X)
        loss, _ = softmax_cross_entropy(logits, y)
        return float(loss.data)

    before = loss_of(net)
    for _ in range(150):
        logits, params = net.forward(X)
        loss, _ = softmax_cross_entropy(logits, y)
        loss.backward()
        opt.step(net, params)
    after = loss_of(net)
    assert after < before * 0.5, f"expected substantial loss decrease, {before} -> {after}"


def test_optimizer_survives_a_shape_change_mid_training():
    """Regression test for the momentum/shape-mismatch bug: pruning a layer
    between steps used to crash the next optimizer.step() call."""
    rng = np.random.default_rng(6)
    X = rng.standard_normal((20, 4))
    y = (X[:, 0] > 0).astype(int)
    net = GrowableNet(in_dim=4, out_dim=2, hidden=(5,), seed=6)
    opt = LearnedOptimizer(seed=6)
    logits, params = net.forward(X)
    loss, _ = softmax_cross_entropy(logits, y)
    loss.backward()
    opt.step(net, params)
    net.prune_neuron(0)  # shapes change here
    logits, params = net.forward(X)  # must not raise
    loss, _ = softmax_cross_entropy(logits, y)
    loss.backward()
    opt.step(net, params)  # must not raise


def test_curiosity_module_prediction_error_shrinks_on_repeated_signal():
    icm = IntrinsicCuriosityModule(state_dim=5, seed=7)
    state = np.array([2.0, 1.0, 0.05, 1.1, 3.0])
    errs = [icm.update(state, 0.42) for _ in range(60)]
    assert errs[-1] < errs[0], "forward model should get better at a repeated, learnable signal"


def test_self_model_policy_drifts_from_uniform_with_lopsided_evidence():
    sm = SelfModel()
    for _ in range(30):
        sm.observe_outcome("grow", 1.0)
        sm.observe_outcome("prune", -1.0)
    assert sm.action_weights["grow"] > sm.action_weights["prune"]


def test_zero_white_end_to_end_runs_and_improves():
    rng = np.random.default_rng(8)
    N = 120
    X = rng.standard_normal((N, 8))
    y = ((X[:, 0] * X[:, 1] + X[:, 2]) > 0).astype(int) + (X[:, 3] > 0.2).astype(int)
    y = np.clip(y, 0, 2)
    Xtr, ytr, Xval, yval = X[:90], y[:90], X[90:], y[90:]

    zw = ZeroWhite(in_dim=8, out_dim=3, hidden=(4,), num_chunks=4, seed=9)
    start_loss, _ = zw.evaluate(zw.net, Xval, yval)
    zw.run(Xtr, ytr, Xval, yval, generations=12, probation_steps=15, verbose=False)
    end_loss, _ = zw.evaluate(zw.net, Xval, yval)
    assert len(zw.self_model.history) == 12
    assert end_loss < start_loss, f"expected val_loss to improve, {start_loss} -> {end_loss}"
    assert zw.net.num_params() > 0
    for l in zw.net.layers:
        assert l["W"].shape[0] > 0 and l["W"].shape[1] > 0  # never mutated into a broken shape


# ---------------- v2 tests ----------------
from zerowhite.tasks import Curriculum, make_shapes, fgsm  # noqa: E402
from zerowhite.zero_white import run_curriculum  # noqa: E402
from zerowhite import checkpoint  # noqa: E402
import tempfile  # noqa: E402


def test_mask_removes_noop_moves():
    zw = ZeroWhite(in_dim=8, out_dim=3, hidden=(4,), num_chunks=4, seed=1)
    f = zw.feasible_actions()
    assert f["patch_on"] and not f["patch_off"] and not f["prune_layer"]
    zw.net.toggle_hierarchical()
    f = zw.feasible_actions()
    assert f["patch_off"] and not f["patch_on"]
    zw.self_model.max_layers = len(zw.net.layers)
    assert not zw.feasible_actions()["add_layer"]


def test_selection_always_includes_noop_and_respects_mask_and_bench():
    sm = SelfModel(patience=2, cooldown=3)
    rng = np.random.default_rng(0)
    feas = {a: True for a in sm.ACTIONS}
    feas["grow"] = False
    for _ in range(20):
        ch = sm.select_actions(feas, rng, k=4)
        assert "noop" in ch and "grow" not in ch and len(set(ch)) == len(ch)
    sm.generation = 1
    sm.record_result(["perturb"], "noop"); sm.record_result(["perturb"], "noop")
    assert "perturb" not in sm.eligible(feas)
    sm.generation = 10
    assert "perturb" in sm.eligible(feas)


def test_decay_lr_halves_and_respects_floor():
    opt = LearnedOptimizer(base_lr=0.08)
    assert abs(opt.halve_lr().base_lr - 0.04) < 1e-12
    zw = ZeroWhite(in_dim=8, out_dim=3, hidden=(4,), num_chunks=4, seed=1, lr_floor=0.01)
    zw.opt.base_lr = 0.015
    assert not zw.feasible_actions()["decay_lr"]


def test_meta_mutate_lr_is_capped():
    from zerowhite.meta_optimizer import MAX_LR
    opt = LearnedOptimizer(base_lr=MAX_LR)
    for _ in range(50):
        opt = opt.meta_mutate()
        assert opt.base_lr <= MAX_LR


def test_grow_neurons_preserves_function_and_respects_cap():
    net = GrowableNet(in_dim=5, out_dim=2, hidden=(8,), seed=0)
    X = np.random.default_rng(0).standard_normal((6, 5))
    before = net.forward(X)[0].data.copy()
    assert net.grow_neurons(max_units=64) and net.hidden_sizes() == [10]
    assert np.allclose(before, net.forward(X)[0].data)
    assert net.grow_neurons(k=100, max_units=12) and net.hidden_sizes() == [12]
    assert not net.grow_neurons(max_units=12)


def test_fgsm_raises_loss():
    X, y = make_shapes(0, 40, seed=3)
    net = GrowableNet(64, 6, hidden=(16,), num_chunks=8, seed=0)
    opt = LearnedOptimizer(seed=0)
    for _ in range(80):
        lg, p = net.forward(X); l, _ = softmax_cross_entropy(lg, y); l.backward(); opt.step(net, p)
    l0 = float(softmax_cross_entropy(net.forward(X)[0], y)[0].data)
    l1 = float(softmax_cross_entropy(net.forward(fgsm(net, X, y, 0.3))[0], y)[0].data)
    assert l1 > l0


def test_curriculum_promotes_only_after_patience():
    net = GrowableNet(64, 6, hidden=(4,), num_chunks=8, seed=0)
    c = Curriculum(n_per_class=20, patience=2)
    assert not c.step(0.99, net) and c.level == 0
    assert not c.step(0.10, net) and c.level == 0      # streak reset
    c.step(0.99, net); assert c.step(0.99, net) and c.level == 1
    c.level = 3; c._build(); c.refresh_adversarial(net)
    assert len(c.Xtr) == 2 * len(c._clean[0])           # adversarial half added


def test_checkpoint_roundtrip_and_resume():
    X, y = make_shapes(0, 30, seed=1)
    zw = ZeroWhite(64, 6, hidden=(5,), num_chunks=8, seed=2)
    cur = Curriculum(n_per_class=30)
    run_curriculum(zw, cur, max_generations=3, probation_steps=5, verbose=False)
    d = tempfile.mkdtemp(); checkpoint.save(zw, d, cur)
    z2, c2 = checkpoint.load(d)
    assert (zw.predict(X) == z2.predict(X)).all()
    assert z2.self_model.generation == 3 and c2.level == cur.level
    run_curriculum(z2, c2, max_generations=2, probation_steps=5, verbose=False)
    assert z2.self_model.generation == 5


def main():
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {e}")
            failed.append(t.__name__)
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print("Failed:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
