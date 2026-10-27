"""Tests for 6.5 Mixture-of-Agents routing (forge/moa.py)."""
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ZERO_WHITE_DATA", tempfile.mkdtemp(prefix="zw_test_data_"))

from zerowhite.engine import Tensor, mse_loss                          # noqa: E402
from zerowhite.forge import moa                                         # noqa: E402
from zerowhite.forge import task as T                                   # noqa: E402
from zerowhite.forge.families import FAMILIES, families_for            # noqa: E402
from zerowhite.nn import Params, init_mlp                               # noqa: E402

_DIRS = []


def tmp():
    d = tempfile.mkdtemp(prefix="zw_t65m_")
    _DIRS.append(d)
    return d


import atexit                                                          # noqa: E402
atexit.register(lambda: [shutil.rmtree(d, ignore_errors=True) for d in _DIRS])

G = dict(n_experts=3, hidden=16, top_k=2, balance=0.02, gate_noise=0.1, mix=0.0, use_program=0, lr=0.02, wd=1e-5, batch=64)


def regimes(n=900, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 4))
    y = np.where(X[:, 0] > 0, np.sin(3 * X[:, 1]) + X[:, 2], -X[:, 1] * X[:, 3] + 0.5 * X[:, 2])
    return T.from_arrays("regimes", X, y, "regression", seed=seed)


def regimes_clf(n=900, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 4))
    y = np.where(X[:, 0] > 0, X[:, 1] > 0, X[:, 2] * X[:, 3] > 0).astype(int)
    return T.from_arrays("regimes_clf", X, y, "classification", seed=seed)


# ======================================================================================================================================= the router
def test_topk_mask_picks_k_per_row_deterministically():
    lg = np.array([[0.1, 0.9, 0.5], [2.0, 2.0, 1.0], [-1.0, -2.0, -3.0]])
    m1, m2 = moa.topk_mask(lg, 1), moa.topk_mask(lg, 2)
    assert m1.sum(axis=1).tolist() == [1, 1, 1] and m2.sum(axis=1).tolist() == [2, 2, 2]
    assert m1[0].argmax() == 1 and m1[1].argmax() == 0 and m1[2].argmax() == 0     # a tie goes to the lower index, every time
    assert (moa.topk_mask(lg, 9) == 1).all()                                      # k larger than K means every expert


def test_the_balance_loss_is_one_when_even_and_K_when_one_expert_takes_everything():
    K, B = 4, 64
    even = Tensor(np.full((B, K), 1.0 / K), requires_grad=True)
    even.data[np.arange(B), np.arange(B) % K] += 1e-6                              # first choices spread evenly
    mask = np.zeros((B, K))
    mask[np.arange(B), np.arange(B) % K] = 1.0
    assert abs(float(moa.balance_loss(even, mask).data) - 1.0) < 1e-3
    collapsed = np.full((B, K), 0.01)
    collapsed[:, 0] = 0.97
    mask2 = np.zeros((B, K))
    mask2[:, 0] = 1.0
    assert abs(float(moa.balance_loss(Tensor(collapsed, requires_grad=True), mask2).data) - K * 0.97) < 1e-6
    t = Tensor(collapsed.copy(), requires_grad=True)                               # and it is differentiable in the router probabilities
    moa.balance_loss(t, mask2).backward()
    assert t.grad is not None and np.abs(t.grad).sum() > 0


def test_usage_stats_report_what_the_router_does():
    even = np.eye(4)[np.arange(100) % 4]
    st = moa.usage_stats(even)
    assert st["entropy"] > 0.99 and st["active"] == 4
    one = moa.usage_stats(np.eye(4)[np.zeros(100, int)])
    assert one["entropy"] == 0.0 and one["active"] == 1 and one["share"][0] == 1.0


def _gate_grad(top_k):
    rng = np.random.default_rng(0)
    p = Params()
    init_mlp(p, "g", [3, 3], rng, out_scale=1.0)
    X = rng.normal(size=(8, 3))
    Y = rng.normal(size=(3, 8, 1))

    def loss_of(arrays):
        pp = Params({k: v for k, v in arrays.items()})
        leaves = pp.leaves()
        Gt, _, _ = moa.gate_forward(leaves, Tensor(X), top_k)
        out = None
        for j in range(3):
            oh = np.zeros(3)
            oh[j] = 1.0
            part = (Gt * Tensor(oh)).sum(axis=1, keepdims=True) * Tensor(Y[j])
            out = part if out is None else out + part
        loss = mse_loss(out, np.ones((8, 1)))
        return loss, leaves
    loss, leaves = loss_of(p.a)
    loss.backward()
    return float(np.abs(leaves["g.W0"].grad).sum())


def test_the_router_receives_gradient_from_the_task_loss_with_top1_as_well_as_top2():
    """With top-1 and a renormalised gate the expert's weight is a constant 1 and the router learns nothing from the task: that was the first version's bug. Switch-style (un-renormalised) gating fixes it."""
    assert _gate_grad(1) > 1e-6 and _gate_grad(2) > 1e-6


def test_the_whole_mixture_loss_matches_finite_differences_on_the_gate_and_an_expert():
    t = regimes(200)
    a = FAMILIES["moa"].build(t, dict(G, gate_noise=0.0, mix=0.5), 0)
    X, y = t.Xtr[:32], (t.ytr[:32] - a.ymu) / a.ysd

    def loss(arrays):
        a.p.a = {k: v.copy() for k, v in arrays.items()}
        l = a.p.leaves()
        out, full, mask, recons = a._mix(l, Tensor(X), False, None)
        total = mse_loss(out, y[:, None]) + moa.balance_loss(full, mask) * 0.05
        for r in recons:
            total = total + mse_loss(r, X) * 0.2 / max(1, len(recons))
        return total, l
    base = {k: v.copy() for k, v in a.p.a.items()}
    total, leaves = loss(base)
    total.backward()
    worst, rng = 0.0, np.random.default_rng(1)
    ae, ml = a.types.index("ae"), a.types.index("mlp")
    for name in ("g.W0", f"e{ml}.W0", f"e{ae}.enc.W0", f"e{ae}.head.W1"):
        for _ in range(4):
            idx = tuple(int(rng.integers(0, s)) for s in base[name].shape)
            hi = {k: v.copy() for k, v in base.items()}
            lo = {k: v.copy() for k, v in base.items()}
            hi[name][idx] += 1e-6
            lo[name][idx] -= 1e-6
            num = (float(loss(hi)[0].data) - float(loss(lo)[0].data)) / 2e-6
            an = float(leaves[name].grad[idx])
            worst = max(worst, abs(num - an) / (abs(num) + abs(an) + 1e-8))
    assert worst < 1e-4, worst


# ======================================================================================================================================= the family
def test_moa_is_offered_for_tables_and_policy_mixtures_for_environments_and_the_blend_is_never_proposed():
    reg, env = regimes(100), T.from_env_name("cartpole")
    assert "moa" in {f.name for f in families_for(reg)} and "moa_policy" not in {f.name for f in families_for(reg)}
    assert "moa_policy" in {f.name for f in families_for(env)} and "moa" not in {f.name for f in families_for(env)}
    assert "moa_blend" in FAMILIES and "moa_blend" not in {f.name for f in families_for(reg)} | {f.name for f in families_for(env)}


def test_the_mixture_learns_a_task_with_two_regimes_and_uses_more_than_one_expert():
    t = regimes(1200)
    a = FAMILIES["moa"].build(t, dict(G, top_k=2, hidden=24), 0)
    for r in range(4):
        a.train(t, 1.0, np.random.default_rng(r))
    skill, m = a.val_score(t)
    assert skill > 0.6 and m["experts_active"] >= 2, (skill, m)
    X1 = np.c_[t.Xtr, np.ones(len(t.Xtr))]
    w = np.linalg.lstsq(X1, t.ytr, rcond=None)[0]                                   # the best single linear model, for scale
    lin = 1 - np.mean((np.c_[t.Xval, np.ones(len(t.Xval))] @ w - t.yval) ** 2) / np.var(t.yval)
    assert skill > lin + 0.3, (skill, lin)
    assert 0.0 <= m["router_entropy"] <= 1.0


def test_a_classification_mixture_beats_chance_and_returns_probabilities():
    t = regimes_clf(1200)
    a = FAMILIES["moa"].build(t, dict(G, hidden=24, mix=0.6), 1)
    for r in range(4):
        a.train(t, 1.0, np.random.default_rng(r))
    skill, m = a.val_score(t)
    assert skill > 0.4 and m["acc"] > 0.7, (skill, m)
    P = a.outputs(t.Xval)
    assert P.shape == (len(t.Xval), 2) and np.allclose(P.sum(axis=1), 1.0)


def test_only_the_chosen_experts_run_on_a_row():
    t = regimes(300)
    for k in (1, 2, 3):
        a = FAMILIES["moa"].build(t, dict(G, top_k=k), 0)
        calls = []
        real = a._expert_raw
        a._expert_raw = lambda j, X, real=real: (calls.append(len(X)), real(j, X))[1]
        a.predict(t.Xval)
        assert sum(calls) == k * len(t.Xval), (k, sum(calls))                       # k experts per row, not all K


def test_the_expert_types_are_mixed_and_the_balance_weight_keeps_the_router_from_collapsing():
    assert set(moa.expert_types(6, 0.0)) == {"mlp", "ae", "linear"} and moa.expert_types(3, 0.9)[0] != moa.expert_types(3, 0.0)[0]
    t = regimes(1200)
    g = dict(G, top_k=1, hidden=16)
    loose = FAMILIES["moa"].build(t, dict(g, balance=1e-3), 0)
    tight = FAMILIES["moa"].build(t, dict(g, balance=0.3), 0)
    for r in range(4):
        loose.train(t, 1.0, np.random.default_rng(r))
        tight.train(t, 1.0, np.random.default_rng(r))
    assert tight.route_stats(t.Xval)["entropy"] >= loose.route_stats(t.Xval)["entropy"] - 0.02
    assert tight.route_stats(t.Xval)["entropy"] > 0.9


def test_an_agent_survives_save_and_load_with_identical_predictions():
    t = regimes(400)
    a = FAMILIES["moa"].build(t, dict(G, mix=0.5), 0)
    a.train(t, 0.5, np.random.default_rng(0))
    d = tmp()
    a.save(d)
    b = FAMILIES["moa"].load(d, t, a.genome)
    assert np.allclose(a.predict(t.Xval), b.predict(t.Xval))
    assert b.describe() == a.describe()


def test_a_frozen_program_expert_is_blended_by_the_router_and_survives_a_reload():
    t = regimes(500)
    a = FAMILIES["moa"].build(t, dict(G, use_program=1, top_k=2), 0)
    assert a.use_prog and a.n_gate == a.K + 1
    a.train(t, 0.5, np.random.default_rng(0))
    assert a.prog and "def predict" in a.prog["src"] and "frozen evolved program" in a.describe()
    before = {k: v.copy() for k, v in a.p.a.items()}
    a.train(t, 0.2, np.random.default_rng(1))
    assert a.prog["src"] and not any(k.startswith("prog") for k in a.p.a)           # the program is source text, not parameters: training cannot change it
    assert any(not np.allclose(before[k], a.p.a[k]) for k in a.p.a if k.startswith("g."))
    d = tmp()
    a.save(d)
    b = FAMILIES["moa"].load(d, t, a.genome)
    assert np.allclose(a.predict(t.Xval), b.predict(t.Xval)) and b.prog["src"] == a.prog["src"]
    c = FAMILIES["moa"].build(t, dict(G, use_program=1), 0)                          # and a classification task ignores the gene: there is no program expert for it
    cls = FAMILIES["moa"].build(regimes_clf(300), dict(G, use_program=1), 0)
    assert c.use_prog and not cls.use_prog


# ======================================================================================================================================= control
def test_a_mixture_of_policies_trains_acts_and_keeps_the_default_agent_unchanged():
    from zerowhite.agents import PGAgent, make_env
    env = make_env("cartpole")
    plain = PGAgent(4, 2, seed=3)
    assert plain._aux_loss(plain.p.leaves(), Tensor(np.zeros((2, 4)))) is None
    assert np.allclose(plain._logits_raw(np.ones((1, 4))), __import__("zerowhite.nn", fromlist=["x"]).mlp_raw(plain.p.a, "pi", np.ones((1, 4))))
    task = T.from_env_name("cartpole")
    a = FAMILIES["moa_policy"].build(task, dict(hidden=16, depth=1, lr=0.02, gamma=0.97, entropy=0.01, curiosity=0.0, n_experts=3, balance=0.05, top_k=2), 0)
    g_before = a.agent.p.a["g.W0"].copy()
    a.train(task, 0.4, np.random.default_rng(0))
    assert not np.allclose(g_before, a.agent.p.a["g.W0"]) and not any(k.startswith("pi.") for k in a.agent.p.a)
    skill, m = a.val_score(task)
    assert -1.0 <= skill <= 1.0 and "experts_active" in m
    d = tmp()
    a.save(d)
    b = FAMILIES["moa_policy"].load(d, task, a.genome)
    o = env.reset(np.random.default_rng(0))
    assert np.allclose(a.agent.policy(o[None]), b.agent.policy(o[None]))
    assert "mixture-of-policies" in a.describe()


# ======================================================================================================================================= the Autonomy layer
class _Region:
    """A fake finished agent that is right where `mask(x)` is true and confidently wrong elsewhere."""
    def __init__(self, fam, truth, mask):
        self.family, self.genome, self.spent = fam, dict(), 0.0
        self.truth, self.mask = truth, mask

    def predict(self, X):
        right = self.truth(X)
        return np.where(self.mask(X), right, 1 - right)

    def n_params(self):
        return 1

    def describe(self):
        return f"{self.family} fake"

    def save(self, d):
        raise NotImplementedError


class _Cand:
    def __init__(self, agent, skill):
        self.agent, self.skill = agent, skill


def test_a_router_over_finished_agents_beats_each_member_on_rows_it_never_saw():
    t = regimes_clf(1200)
    truth = lambda X: (np.where(X[:, 0] > 0, X[:, 1] > 0, X[:, 2] * X[:, 3] > 0)).astype(int)      # noqa: E731
    left = _Cand(_Region("mlp", truth, lambda X: X[:, 0] > 0), 0.5)
    right = _Cand(_Region("selfmod_net", truth, lambda X: X[:, 0] <= 0), 0.5)
    b = moa.build_blend(t, [left, right], seed=0)
    assert b is not None and len(b.members) == 2
    fit_rows, held = moa.BlendAgent.split(len(t.Xval))
    assert set(fit_rows).isdisjoint(held) and len(fit_rows) + len(held) == len(t.Xval)
    blend_skill = moa.heldout_skill(b, t)
    assert blend_skill > moa.heldout_skill(left.agent, t) + 0.3 and blend_skill > moa.heldout_skill(right.agent, t) + 0.3, blend_skill
    assert blend_skill > 0.8
    s, m = b.val_score(t)
    assert m["scored_on"].startswith("held-out half") and abs(s - blend_skill) < 1e-9


def test_the_gate_is_fitted_on_the_even_validation_rows_only():
    t = regimes_clf(400)
    truth = lambda X: (np.where(X[:, 0] > 0, X[:, 1] > 0, X[:, 2] * X[:, 3] > 0)).astype(int)      # noqa: E731
    members = [_Cand(_Region("a", truth, lambda X: X[:, 0] > 0), 0.5), _Cand(_Region("b", truth, lambda X: X[:, 0] <= 0), 0.5)]
    seen = []
    real = moa.BlendAgent._stack
    moa.BlendAgent._stack = lambda self, X, task: (seen.append(len(X)), real(self, X, task))[1]
    try:
        b = moa.build_blend(t, members, steps=5)
    finally:
        moa.BlendAgent._stack = real
    assert b is not None and seen == [len(t.Xval[0::2])]                           # one stack, over the even rows, during fitting
    assert moa.build_blend(t, [members[0]]) is None                                # fewer than two families: no router
    assert moa.build_blend(t, [members[0], _Cand(_Region("a", truth, lambda X: X[:, 0] > 0), 0.4)]) is None   # two members of the SAME family are one family


def test_a_blend_of_real_agents_regression_is_saved_loaded_and_scored_identically():
    from zerowhite.forge.zoo import Zoo
    t = regimes(800)
    cands = []
    for fam, g in (("mlp", dict(hidden=24, depth=1, lr=0.02, wd=1e-5, batch=64)), ("moa", dict(G, hidden=16))):
        a = FAMILIES[fam].build(t, g, 0)
        a.train(t, 1.0, np.random.default_rng(0))
        cands.append(_Cand(a, a.val_score(t)[0]))
    b = moa.build_blend(t, sorted(cands, key=lambda c: -c.skill), seed=1)
    assert b is not None and len(b.members) == 2
    z = Zoo(tmp())
    aid = z.add("regimes", b, dict(skill=0.5, kind="regression"), make_champion=True)
    spec = z.spec("regimes", aid)
    loaded = FAMILIES[spec["family"]].load(f"{z.task_dir('regimes')}/{aid}", t, spec["genome"])
    assert spec["family"] == "moa_blend"
    assert np.allclose(b.predict(t.Xval), loaded.predict(t.Xval))
    assert abs(b.val_score(t)[0] - loaded.val_score(t)[0]) < 1e-9


def test_the_generalist_keeps_a_blend_only_when_it_gains_on_unseen_rows_and_never_fails_a_solve_for_it():
    from zerowhite.forge.generalist import Generalist
    t = regimes_clf(700)
    g = Generalist(tmp(), seed=0, verbose=False, episodic=False)
    rec = g.solve(t, budget=0.3, n_new=6, rungs=2)
    assert rec["family"] in FAMILIES and rec["skill"] > 0.2
    off = Generalist(tmp(), seed=0, verbose=False, episodic=False, blend=False)
    assert off.solve(t, budget=0.3, n_new=6, rungs=2)["family"] != "moa_blend"
    bad = Generalist(tmp(), seed=0, verbose=False, episodic=False)
    real = moa.build_blend
    moa.build_blend = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("blend exploded"))
    try:
        assert bad.solve(t, budget=0.2, n_new=4, rungs=2)["skill"] > -1.0
    finally:
        moa.build_blend = real
