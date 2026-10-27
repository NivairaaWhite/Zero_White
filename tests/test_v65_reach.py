"""Tests for 6.5: the source patcher's widened reach (the benchmark now runs every patcher.MUTABLE module) and topology invention (forge/topology.py)."""
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ZERO_WHITE_DATA", tempfile.mkdtemp(prefix="zw_test_data_"))

from zerowhite.forge import families as F                               # noqa: E402
from zerowhite.forge import task as T                                   # noqa: E402
from zerowhite.forge import topology as TP                              # noqa: E402
from zerowhite.selfmod import bench, patcher                            # noqa: E402

_DIRS = []


def tmp():
    d = tempfile.mkdtemp(prefix="zw_t65r_")
    _DIRS.append(d)
    return d


import atexit                                                           # noqa: E402
atexit.register(lambda: [shutil.rmtree(d, ignore_errors=True) for d in _DIRS])

PKG = os.path.join(ROOT, "zerowhite")


def _xor(n=360, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 2))
    return T.from_arrays("xor", X, ((X[:, 0] * X[:, 1]) > 0).astype(int), "classification", seed=seed)


# --------------------------------------------------------------------------- the patcher's reach
def test_the_benchmark_runs_code_from_every_module_the_patcher_may_change():
    seen = set()
    mutable = {os.path.join(PKG, f) for f in patcher.MUTABLE}

    def prof(frame, event, arg):
        if event == "call":
            fn = frame.f_code.co_filename
            if fn in mutable:
                seen.add(fn)
    sys.setprofile(prof)
    try:
        bench.run_seed(3)
    finally:
        sys.setprofile(None)
    assert mutable <= seen, f"patcher.MUTABLE modules the benchmark never runs: {sorted(os.path.basename(m) for m in mutable - seen)}"


def test_the_benchmark_reports_each_part_is_deterministic_and_is_the_mean_of_its_parts():
    a, b = bench.run_seed(5), bench.run_seed(5)
    assert a == b
    parts = ("digits_val_acc", "cartpole", "landscape", "vae", "charlm", "curiosity")
    assert all(0.0 <= a[k] <= 1.0 for k in parts)
    assert abs(a["score"] - np.mean([a[k] for k in parts])) < 1e-12
    assert bench.run_seed(6) != a


def test_a_change_to_a_value_in_each_newly_covered_module_changes_the_benchmark_so_the_patcher_can_see_it():
    from zerowhite import curiosity, generative
    base = bench.run_seed(2)

    def with_defaults(fn, new):
        old = fn.__defaults__
        fn.__defaults__ = new
        try:
            return bench.run_seed(2)
        finally:
            fn.__defaults__ = old
    d = generative.CharLM.__init__.__defaults__                           # (ctx, emb, hidden, depth, lr, seed)
    lm = with_defaults(generative.CharLM.__init__, d[:4] + (0.05,) + d[5:])
    assert lm["charlm"] != base["charlm"] and lm["vae"] == base["vae"] and lm["cartpole"] == base["cartpole"]
    d = generative.CVAE.__init__.__defaults__                             # (latent, hidden, depth, beta, lr, seed)
    vae = with_defaults(generative.CVAE.__init__, d[:4] + (0.02,) + d[5:])
    assert vae["vae"] != base["vae"] and vae["charlm"] == base["charlm"]
    d = curiosity.ForwardDynamicsCuriosity.__init__.__defaults__          # (hidden, lr, seed)
    cur = with_defaults(curiosity.ForwardDynamicsCuriosity.__init__, (8,) + d[1:])
    assert cur["curiosity"] != base["curiosity"] and cur["vae"] == base["vae"]
    assert base == bench.run_seed(2)                                       # restored


def test_the_patcher_finds_editable_values_in_the_generative_and_curiosity_modules_and_never_in_the_measurement():
    sites = patcher.find_sites(PKG, patcher.MUTABLE)
    files = {s.file for s in sites}
    for f in ("generative.py", "curiosity.py", "architecture.py"):
        assert f in files, f
    assert not any(s.name in patcher.SKIP_NAMES for s in sites)
    src = open(os.path.join(PKG, "selfmod", "bench.py")).read()
    assert "bench.py" not in "".join(patcher.MUTABLE) and "selfmod/bench.py" in patcher.CONSTITUTION                # not patchable (only MUTABLE is) and sealed: it is the judge
    assert "mlp_raw(lm.p.a" in src and "mlp_raw(icm.params.a" in src                                     # measured here, not by the module under patch


# --------------------------------------------------------------------------- topology invention
def _check_valid(net):
    for i in list(net.order) + [TP.OUTPUT]:
        assert len(set(net.src[i])) == len(net.src[i]), "duplicate edge"
        for j in net.src[i]:
            assert net.pos(j) < net.pos(i), f"edge {j}->{i} points backwards"
            W = net.p.a[f"W.{j}.{i}"]
            assert W.shape == (net.width(j), net.width(i)), (j, i, W.shape)
        assert net.p.a[f"b.{i}"].shape == (net.width(i),)
    assert set(k for k in net.p.a if k.startswith("W.")) == {f"W.{j}.{i}" for j, i in net.edges()}


def test_every_topology_mutation_leaves_the_networks_outputs_exactly_unchanged_and_the_graph_valid():
    r = np.random.default_rng(0)
    X = r.normal(size=(25, 3))
    for seed, outd in ((0, 2), (1, 1), (2, 4)):
        net = TP.TopologyNet(3, outd, hidden=(5, 4), seed=seed)
        for k in range(40):
            before = net.forward(X)
            note = net.mutate(np.random.default_rng(100 * seed + k))
            _check_valid(net)
            after = net.forward(X)
            assert after.shape == before.shape
            assert np.abs(after - before).max() < 1e-12, (note, np.abs(after - before).max())
        assert len(net.order) <= 12 and max(net.nodes[i]["w"] for i in net.order) <= 96
    kinds = set()
    net = TP.TopologyNet(3, 2, hidden=(4,), seed=0)
    for k in range(60):
        note = net.mutate(np.random.default_rng(k))
        if note:
            kinds.add(note.split()[0])
    assert {"widen", "add"} <= kinds and any(j for j, i in net.edges() if net.pos(i) - net.pos(j) > 1)     # skip connections exist


def test_the_differentiable_forward_equals_the_numpy_forward_and_its_gradients_match_finite_differences():
    from zerowhite.engine import mse_loss
    net = TP.TopologyNet(3, 1, hidden=(4, 3), acts=("tanh", "sigmoid"), seed=0)
    for k in range(6):
        net.mutate(np.random.default_rng(k))
    r = np.random.default_rng(1)
    X, y = r.normal(size=(9, 3)), r.normal(size=(9, 1))
    leaves = net.p.leaves()
    out = net.forward(X, leaves)
    assert np.allclose(out.data, net.forward(X), atol=1e-12)
    loss = mse_loss(out, y)
    loss.backward()
    name = next(k for k in sorted(net.p.a) if k.startswith("W.") and net.p.a[k].size > 1)
    g = leaves[name].grad
    base = {k: v.copy() for k, v in net.p.a.items()}
    for (a, b) in [(0, 0), (min(1, g.shape[0] - 1), min(1, g.shape[1] - 1))]:
        h = 1e-6
        net.p.a[name][a, b] += h
        up = float(np.mean((net.forward(X) - y) ** 2))
        net.p.a[name][a, b] -= 2 * h
        dn = float(np.mean((net.forward(X) - y) ** 2))
        net.p.a[name] = base[name].copy()
        assert abs((up - dn) / (2 * h) - g[a, b]) < 1e-5 * max(1.0, abs(g[a, b])), (name, a, b)


def test_a_mutation_that_cannot_keep_the_graph_valid_is_refused_not_half_applied():
    net = TP.TopologyNet(2, 2, hidden=(4,), seed=0)
    n_edges = len(net.edges())
    assert not net.add_edge(TP.OUTPUT, 1) and not net.add_edge(1, 1) and not net.add_edge(TP.INPUT, TP.INPUT)
    assert not net.add_node(5, "tanh", [TP.OUTPUT], [1]) and not net.add_node(5, "tanh", [42], [1])
    assert not net.add_node(5, "sine", [TP.INPUT], [TP.OUTPUT])                                     # an activation the engine does not have
    assert not net.add_node(5, "tanh", [1], [1])                                                    # would feed itself / go backwards
    assert not net.widen(99, 3) and not net.widen(1, 500)
    assert len(net.edges()) == n_edges and net.order == [1]
    _check_valid(net)


def test_the_topology_agent_grows_when_it_helps_never_makes_the_held_out_loss_worse_and_beats_a_too_small_mlp():
    task = _xor()
    g = dict(init_width=2, lr=0.03, wd=1e-5, rounds=5, cands=3, ft_steps=30)
    a = F.FAMILIES["topology_net"].build(task, g, 0)
    a.train(task, 1.0, np.random.default_rng(0))
    assert a.rounds_done == 5 and 1 <= a.rounds_changed <= 5 and len(a.history) == a.rounds_changed
    assert len(a.loss_trail) == a.rounds_changed and all(after < before for before, after in a.loss_trail)       # a change is kept only when it lowered the held-out loss
    assert len(a.net.order) > 1 or len(a.net.edges()) > 2                                          # the wiring really changed
    small = F.FAMILIES["mlp"].build(task, dict(hidden=2, depth=1, lr=0.03, wd=1e-5, batch=64), 0)
    small.train(task, 1.0, np.random.default_rng(0))
    assert a.val_score(task)[0] > small.val_score(task)[0] + 0.1, (a.val_score(task), small.val_score(task))
    assert "architecture rounds changed it" in a.describe()


def test_a_topology_agent_survives_save_and_load_with_its_graph_and_predictions_and_works_for_regression():
    task = _xor()
    g = dict(init_width=3, lr=0.03, wd=1e-5, rounds=3, cands=2, ft_steps=20)
    a = F.FAMILIES["topology_net"].build(task, g, 1)
    a.train(task, 0.5, np.random.default_rng(1))
    d = tmp()
    a.save(d)
    b = F.FAMILIES["topology_net"].load(d, task, g)
    assert np.array_equal(a.predict(task.Xte), b.predict(task.Xte)) and a.net.spec() == b.net.spec()
    assert abs(a.val_score(task)[0] - b.val_score(task)[0]) < 1e-12 and a.history == b.history
    assert np.allclose(a.outputs(task.Xte), b.outputs(task.Xte))
    r = np.random.default_rng(2)
    X = r.normal(size=(300, 3))
    y = np.sin(2 * X[:, 0]) * X[:, 1] + 0.05 * r.normal(size=300)
    rt = T.from_arrays("sinprod", X, y, "regression", seed=0)
    ar = F.FAMILIES["topology_net"].build(rt, dict(g, init_width=6), 0)
    ar.train(rt, 1.0, np.random.default_rng(0))
    assert ar.val_score(rt)[0] > 0.2 and "r2" in ar.test_metrics(rt)


def test_the_family_is_registered_for_supervised_tasks_only_and_the_generalist_can_race_it():
    from zerowhite.forge.families import families_for
    assert "topology_net" in F.FAMILIES
    assert "topology_net" in {f.name for f in families_for(_xor())}
    env = T.from_env_name("gridworld5")
    assert "topology_net" not in {f.name for f in families_for(env)}
    from zerowhite.forge.generalist import Generalist
    gen = Generalist(tmp(), seed=0, verbose=False)
    task = _xor(200)
    rec = gen.solve(task, budget=0.3, n_new=10, rungs=2)
    assert rec["skill"] > 0.2 and rec["family"] in F.FAMILIES
    assert gen.prior is not None
