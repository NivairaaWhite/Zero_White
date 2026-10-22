"""Tests for v4: safety gate, AST mutation, program evolution, history, sandbox, evolvable hooks, source patcher,
the general model (forge), POET-lite, doctor, and the 'no LLM / no default model' guarantees."""
import ast
import glob
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("ZERO_WHITE_DATA", tempfile.mkdtemp(prefix="zw_test_data_"))

from zerowhite import doctor, evolved, fileio                         # noqa: E402
from zerowhite.evolve import Evolver, Gene                            # noqa: E402
from zerowhite.selfmod import astmut as A                             # noqa: E402
from zerowhite.selfmod import hooks, patcher, sandbox                 # noqa: E402
from zerowhite.selfmod.history import HistoryStore                    # noqa: E402
from zerowhite.selfmod.islands import ProgramEvolver                  # noqa: E402
from zerowhite.selfmod.safety import is_safe, violations              # noqa: E402


def tmp():
    return tempfile.mkdtemp(prefix="zw_t4_")


# --------------------------------------------------------------------------- safety gate
def test_safety_accepts_plain_math_and_rejects_dangerous_code():
    assert is_safe("def f(x):\n    return np.tanh(x) * 2.0 if x > 0 else math.sqrt(abs(x))\n")
    for bad in ["import os\n", "def f(x):\n    return x.__class__\n", "def f(x):\n    return eval('1')\n",
                "def f(x):\n    for i in range(9):\n        pass\n    return x\n", "def f(x):\n    return np.load('a')\n",
                "def f(x):\n    return open('a')\n", "def f(x):\n    return getattr(x, 'a')\n",
                "def f(x):\n    return [i for i in x]\n", "x = 1\nos.system('ls')\n", "class C:\n    pass\n"]:
        assert not is_safe(bad), bad
    assert violations("def f(:\n")[0].startswith("syntax error")
    assert not is_safe("def f(x):\n    return " + "x + " * 2000 + "x\n")                  # size cap


# --------------------------------------------------------------------------- AST mutation
def test_mutation_always_yields_safe_runnable_programs():
    rng = np.random.default_rng(0)
    v = A.Vocab("predict", ("x",), col_var="x", n_cols=3)
    cur, X = A.seed_program(v, "0.0"), np.random.default_rng(1).standard_normal((6, 3))
    made = 0
    for _ in range(200):
        m = A.mutate(cur, v, rng, n_mut=2)
        if m is None:
            continue
        assert not violations(m, **v.safety), m
        out = np.broadcast_to(A.compile_program(m, "predict", **v.safety)(X), (6,))
        assert np.all(np.isfinite(out)) or True            # non-finite is allowed (scored invalid), crashing is not
        made += 1
        cur = m if rng.random() < 0.5 else cur
    assert made > 150


def test_constant_tuning_recovers_a_linear_formula():
    X = np.random.default_rng(2).standard_normal((200, 3))
    y = 3.1 * X[:, 0] - 0.7
    score = lambda s: -np.mean((A.compile_program(s, "predict")(X) - y) ** 2)       # noqa: E731
    src, sc, _ = A.tune_constants("def predict(x):\n    return x[:, 0] * 1.0 + 0.0\n", score, np.random.default_rng(0), steps=300)
    assert -sc < 1e-3, (src, sc)


def test_negative_literals_are_tunable_constants():
    assert A.constants_of("def f(x):\n    return x * -0.5 + (-2)\n") == [-0.5, -2]


def test_simplify_folds_constants_drops_dead_code_and_preserves_behaviour():
    s = "def select_parent(fitnesses, rng):\n    t0 = len(fitnesses)\n    t1 = 0.5\n    return -1.506 - np.abs(3.0) - t0\n"
    out = A.simplify(s, probe=hooks._probe("select_parent"))
    assert "t1" not in out and "np.abs" not in out and "-4.506" in out
    assert A.simplify("def f(x):\n    return x + 1.0\n").strip().endswith("x + 1.0")


# --------------------------------------------------------------------------- program evolution
def test_program_evolver_discovers_a_formula_and_the_fec_cache_saves_work():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 2))
    y = 2.0 * X[:, 0] + 1.0
    v = A.Vocab("predict", ("x",), col_var="x", n_cols=2, max_nodes=60)
    score = lambda fn, src: -float(np.mean((np.broadcast_to(fn(X), y.shape) - y) ** 2)) - 1e-3 * A.count_nodes(src) / 10   # noqa: E731
    px = X[:12]
    ev = ProgramEvolver(v, score, [A.seed_program(v, "0.0")], np.random.default_rng(1),
                        probe_fn=lambda fn: np.broadcast_to(fn(px), (12,)), tune_steps=10)
    best = ev.run(700)
    pred = np.broadcast_to(A.compile_program(best.src, "predict")(X), y.shape)
    assert 1 - np.mean((pred - y) ** 2) / np.var(y) > 0.98, best.src
    assert ev.fec_hits > 0


def test_program_evolver_scores_crashing_programs_as_invalid_not_fatal():
    v = A.Vocab("f", ("x",), col_var="x", n_cols=1)
    ev = ProgramEvolver(v, lambda fn, src: 1 / 0, ["def f(x):\n    return 0.0\n"], np.random.default_rng(0))
    ev.run(5, max_steps=20)
    assert ev.invalid > 0 and ev.best is None


# --------------------------------------------------------------------------- history / rollback
def test_history_put_rollback_lineage_and_tamper_detection():
    h = HistoryStore(tmp())
    for t in ("v1", "v2", "v3"):
        h.put("a", t, {"note": t})
    assert h.text("a") == "v3" and [e["version"] for e in h.lineage("a")] == [1, 2, 3]
    assert h.rollback("a")["version"] == 2 and h.text("a") == "v2"
    assert h.rollback("a")["version"] == 1
    assert [e["event"] for e in h.log("a")].count("rollback") == 2        # rollbacks are logged, nothing is erased
    assert h.verify() == []
    sha = h.head("a")["sha"]
    with open(h._obj_path(sha), "w", encoding="utf-8") as f:
        f.write("tampered")
    assert h.verify() and "hash" in h.verify()[0]
    try:
        h.get(sha)
        raise AssertionError("tampered object must not load")
    except ValueError:
        pass


# --------------------------------------------------------------------------- sandbox
def test_sandbox_returns_output_and_kills_runaway_code_including_children():
    r = sandbox.run_code("print(6*7)")
    assert r.ok and r.stdout.strip() == "42"
    t = sandbox.run_code("while True:\n    pass\n", timeout=2)
    assert t.timed_out and t.seconds < 10 and not t.ok
    spawn = ("import subprocess, sys, time\n"
             "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
             "time.sleep(60)\n")
    t2 = sandbox.run_code(spawn, timeout=2)
    assert t2.timed_out and t2.seconds < 15


def test_sandbox_environment_is_scrubbed():
    os.environ["ZW_SECRET_TEST"] = "hunter2"
    try:
        r = sandbox.run_code("import os; print(os.environ.get('ZW_SECRET_TEST'))")
    finally:
        del os.environ["ZW_SECRET_TEST"]
    assert r.ok and r.stdout.strip() == "None"


def test_sandbox_memory_limit_where_the_os_supports_it():
    if sandbox.resource is None:
        return                                                  # Windows: documented as not enforced
    r = sandbox.run_code("x = bytearray(10**11)", timeout=20)
    assert not r.ok


# --------------------------------------------------------------------------- evolvable hooks
def test_hooks_compile_rejects_unsafe_source_and_policy_survives_faulty_hooks():
    try:
        evolved.compile_hook("sigma_update", "import os\ndef sigma_update(sigma, improved, generation):\n    return sigma\n")
        raise AssertionError("unsafe hook accepted")
    except ValueError:
        pass
    boom = evolved.Policy(sigma_update=lambda s, i, g: 1 / 0, select_parent=lambda f, r: float("nan"))
    space = {"a": Gene("float", 0, 1)}
    ev = Evolver(space, lambda g, p: (-(g["a"] - .3) ** 2, None, {}), population=4, mu=2, seed=0, policy=boom)
    for _ in range(5):
        ev.step()
    assert ev.best.fitness > -0.5 and boom.faults["sigma_update"] > 0 and boom.faults["select_parent"] > 0


def test_genesis_hooks_are_trivial_so_nothing_smart_is_hard_coded():
    assert "1.3" not in evolved.genesis_source("sigma_update") and "0.85" not in evolved.genesis_source("sigma_update")
    p = hooks.genesis_policy()
    assert p.update_sigma(0.37, True, 5) == 0.37 and p.update_sigma(0.37, False, 5) == 0.37


def test_suite_score_is_deterministic():
    a = hooks.suite_score(hooks.genesis_policy(), ("sphere",), (0,), 6)
    b = hooks.suite_score(hooks.genesis_policy(), ("sphere",), (0,), 6)
    assert a == b and 0.0 < a <= 1.0


def test_hook_install_rollback_and_live_loading():
    ws = tmp()
    mgr = hooks.HookManager(ws).activate()
    try:
        mgr.install("sigma_update", "def sigma_update(sigma, improved, generation):\n    return 0.7\n", dict(note="t"))
        assert mgr.policy("live").update_sigma(0.2, True, 1) == 0.7
        mgr.install("sigma_update", "def sigma_update(sigma, improved, generation):\n    return 0.9\n", dict(note="t2"))
        assert mgr.policy("live").update_sigma(0.2, True, 1) == 0.9
        mgr.rollback("sigma_update")
        assert mgr.policy("live").update_sigma(0.2, True, 1) == 0.7
        mgr.rollback("sigma_update")                              # nothing older: back to genesis
        assert mgr.policy("live").update_sigma(0.2, True, 1) == 0.2 and mgr.status()["sigma_update"]["live"] is False
        try:
            mgr.install("sigma_update", "import os\n", {})
            raise AssertionError("unsafe install accepted")
        except ValueError:
            pass
    finally:
        evolved.use(None)


def test_evolve_hook_runs_end_to_end_and_never_installs_without_beating_the_margin():
    ws = tmp()
    mgr = hooks.HookManager(ws)
    rep = hooks.evolve_hook(mgr, "sigma_update", evals=40, seed=0, train_seeds=(0,), heldout_seeds=(100, 101), verbose=False,
                            margin=5.0)                            # impossible margin: must reject
    assert rep["accepted"] is False and mgr.live_source("sigma_update") is None
    assert rep["evals"] > 0 and "reference" in rep


# --------------------------------------------------------------------------- source patcher
def test_patcher_finds_hyperparameter_sites_and_never_measurement_or_budget_sites():
    sites = patcher.find_sites(os.path.join(ROOT, "zerowhite"))
    names = {s.name for s in sites}
    assert len(sites) > 30 and {"lr", "gamma", "entropy", "score_weight"} <= names
    assert not (names & {"episodes", "per_class", "val_cap", "step_cost", "seed", "feat_dim", "generations", "updates"})
    assert all(s.file in patcher.MUTABLE for s in sites)
    assert not set(patcher.MUTABLE) & {pathlib.PurePosixPath(c).name for c in patcher.CONSTITUTION}


def test_patch_is_exactly_one_literal_and_structure_changes_are_refused():
    src = "def f(lr=0.01, n=3):\n    return lr * n\n"
    tree = ast.parse(src)
    const = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == 0.01][0]
    site = patcher.Site("x.py", const.lineno, const.col_offset, const.end_col_offset, 0.01, "lr")
    new = patcher.apply_to_text(src, site, 0.02)
    assert new == "def f(lr=0.02, n=3):\n    return lr * n\n" and patcher.single_literal_change(src, new)
    assert not patcher.single_literal_change(src, new.replace("lr * n", "lr + n"))
    assert not patcher.single_literal_change(src, new.replace("n=3", "n=3, m=4"))
    crlf = src.replace("\n", "\r\n")
    assert patcher.apply_to_text(crlf, site, 0.02).count("\r\n") == 2          # Windows line endings survive


def test_patch_search_control_flow_with_stubbed_benchmark_apply_and_rollback():
    repo = os.path.join(tmp(), "repo")
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns("__pycache__", ".git", "runs", "*.egg-info"))
    store = HistoryStore(tmp())
    import re

    def fake_bench(tree_root, seeds, timeout=0):          # score rises as evolve.py's sigma default approaches 0.9
        txt = fileio.read_text(os.path.join(tree_root, "zerowhite", "evolve.py"))
        s = float(re.search(r"sigma=([0-9.e-]+)[,)]", txt.split("class Evolver")[1].split("def step")[0]).group(1))
        return 1.0 - abs(s - 0.9)

    real_b, real_t = patcher.bench_in, patcher.tests_pass
    patcher.bench_in, patcher.tests_pass = fake_bench, (lambda root, timeout=0: (True, "stub ok"))
    try:
        rep = patcher.search(repo, store, evals=60, seed=1, apply=True, verbose=False, files=("evolve.py",), margin=0.0)
    finally:
        patcher.bench_in, patcher.tests_pass = real_b, real_t
    assert rep.applied and rep.accepted and rep.diff.startswith("--- a/zerowhite/evolve.py")
    new_txt = fileio.read_text(os.path.join(repo, "zerowhite", "evolve.py"))
    old_txt = fileio.read_text(os.path.join(ROOT, "zerowhite", "evolve.py"))
    assert new_txt != old_txt and ast.parse(new_txt)                       # real tree changed, still valid Python
    assert [e["version"] for e in store.lineage("src:evolve.py")] == [1, 2]
    patcher.rollback_file(repo, store, "evolve.py")
    assert fileio.read_text(os.path.join(repo, "zerowhite", "evolve.py")) == old_txt
    try:
        patcher.rollback_file(repo, store, "fileio.py")
        raise AssertionError("constitution file must not be rollback-able via the patcher")
    except PermissionError:
        pass


def test_patch_search_rejects_everything_when_tests_fail():
    repo = os.path.join(tmp(), "repo")
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns("__pycache__", ".git", "runs", "*.egg-info"))
    before = fileio.read_text(os.path.join(repo, "zerowhite", "evolve.py"))
    real_b, real_t = patcher.bench_in, patcher.tests_pass
    counter = {"n": 0}

    def rising(tree_root, seeds, timeout=0):
        counter["n"] += 1
        return counter["n"] * 0.01                         # every candidate looks better than the last
    patcher.bench_in, patcher.tests_pass = rising, (lambda root, timeout=0: (False, "1 failed"))
    try:
        rep = patcher.search(repo, HistoryStore(tmp()), evals=4, seed=0, apply=True, verbose=False, files=("evolve.py",), margin=-1)
    finally:
        patcher.bench_in, patcher.tests_pass = real_b, real_t
    assert not rep.applied and not rep.accepted and "test suite failed" in rep.note
    assert fileio.read_text(os.path.join(repo, "zerowhite", "evolve.py")) == before


# --------------------------------------------------------------------------- the general model
def _reg_task(seed=0, n=300):
    from zerowhite.forge import task as T
    r = np.random.default_rng(seed)
    X = r.uniform(-2, 2, size=(n, 3))
    return T.from_arrays("t_reg", X, 2.0 * X[:, 0] - X[:, 1] + 0.3 * r.normal(size=n), "regression")


def test_there_is_no_default_agent_or_model_anywhere_in_the_source():
    bad = []
    for path in glob.glob(os.path.join(ROOT, "zerowhite", "**", "*.py"), recursive=True):
        for node in ast.walk(ast.parse(fileio.read_text(path))):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    # QUICKDRAW_DEFAULT is the default list of drawing categories to download: data, not a model
                    if isinstance(t, ast.Name) and t.id.endswith("_DEFAULT") and t.id != "QUICKDRAW_DEFAULT":
                        bad.append(f"{os.path.basename(path)}:{t.id}")
    assert not bad, bad
    from zerowhite.forge.prior import AgentPrior
    from zerowhite.forge.families import FAMILIES
    pr = AgentPrior()
    assert np.allclose(pr.family_probs("classification", ["mlp", "selfmod_net"]), [0.5, 0.5])      # day-one prior is uniform
    rng = np.random.default_rng(0)
    g1 = pr.sample_genome("regression", "mlp", FAMILIES["mlp"].space, rng)
    g2 = pr.sample_genome("regression", "mlp", FAMILIES["mlp"].space, rng)
    assert g1 != g2                                                                                # random, not a fixed default


def test_prior_learns_which_family_works_and_where_good_genomes_live():
    from zerowhite.forge.prior import AgentPrior
    from zerowhite.forge.families import FAMILIES
    pr, rng = AgentPrior(), np.random.default_rng(0)
    for _ in range(12):
        pr.observe("env", "program_policy", {"n_islands": 3}, 0.9)
        pr.observe("env", "actor_critic", {"hidden": 8}, 0.1)
    p = pr.family_probs("env", ["actor_critic", "program_policy"])
    assert p[1] > 0.8 and p.min() >= 0.07
    space = FAMILIES["mlp"].space
    for _ in range(12):
        pr.observe("regression", "mlp", {"hidden": 100, "depth": 2, "lr": 0.01, "wd": 1e-4, "batch": 64}, 0.9)
    hs = [pr.sample_genome("regression", "mlp", space, rng)["hidden"] for _ in range(60)]
    assert np.mean(hs) > 60                                   # concentrates near the elites (uniform would average ~66 over 4..128)
    path = os.path.join(tmp(), "p.json")
    pr.path = path
    pr.save()
    assert AgentPrior(path).observations == pr.observations
    fileio.write_text(path, "{not json")
    assert AgentPrior(path).observations == 0                 # a corrupt prior is survivable


def test_csv_loader_detects_kind_and_splits_without_overlap():
    from zerowhite.forge import task as T
    d = tmp()
    p = os.path.join(d, "t.csv")
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write("a,b,y\n" + "\n".join(f"{i},{(i * 7) % 5},{i % 3}" for i in range(120)))
    t = T.from_csv(p)
    assert t.kind == "classification" and t.n_classes == 3 and len(t.Xtr) + len(t.Xval) + len(t.Xte) == 120
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write("a,y\n" + "\n".join(f"{i},{i * 0.37 + 0.011}" for i in range(120)))
    assert T.from_csv(p).kind == "regression"


def test_generalist_builds_an_agent_for_a_new_task_persists_it_and_never_regresses():
    from zerowhite.forge.generalist import Generalist
    ws, task = tmp(), _reg_task()
    g = Generalist(ws, seed=0, verbose=False)
    r1 = g.solve(task, budget=0.4, n_new=4, rungs=2)
    assert r1["improved"] and r1["test"]["r2"] > 0.5 and r1["skill"] > 0.5
    assert "t_reg" in g.zoo.tasks() and g.prior.observations >= 3
    g2 = Generalist(ws, seed=1, verbose=False)               # fresh process, same workspace: it remembers
    r2 = g2.solve(task, budget=0.3, n_new=3, rungs=2)
    assert r2["skill"] >= r1["skill"] - 1e-12                # the champion can only get better
    assert abs(r2["_agent"].val_score(task)[0] - r2["skill"]) < 1e-9     # and what it hands back IS the champion
    best = g2.zoo.best("t_reg")
    spec = g2.zoo.spec("t_reg", best["id"])
    assert spec["family"] in ("mlp", "program_formula") and spec["description"]


def test_generalist_transfers_a_champion_to_a_task_with_the_same_signature():
    from zerowhite.forge.generalist import Generalist
    from zerowhite.forge import task as T
    ws, a = tmp(), _reg_task(0)
    g = Generalist(ws, seed=0, verbose=False)
    g.solve(a, budget=0.4, n_new=3, rungs=2)
    b = _reg_task(0)
    b.name = "t_reg_sibling"
    seen = []
    g.log = lambda *x: seen.append(" ".join(map(str, x)))
    g.verbose = True
    g.solve(b, budget=0.3, n_new=2, rungs=2)
    assert any("transfer from 't_reg'" in s for s in seen), seen


def test_generalist_solves_an_environment_with_a_readable_evolved_program():
    from zerowhite.forge.generalist import Generalist
    from zerowhite.forge import task as T
    g = Generalist(tmp(), seed=0, verbose=False)
    r = g.solve(T.from_env_name("mountaincar"), budget=0.6, n_new=5, rungs=2)
    assert r["skill"] > 0.3, r["description"]
    assert r["family"] in ("program_policy", "actor_critic")


def test_custom_env_file_is_loaded_and_scored_against_a_random_baseline():
    from zerowhite.forge import task as T
    t = T.load_env_file(os.path.join(ROOT, "examples", "catch_env.py"))
    assert t.kind == "env" and (t.obs_dim, t.n_actions) == (3, 3) and t.max_return == 10.0
    rnd, _ = t.random_baseline()
    perfect = lambda o: 0 if o[2] > o[0] else (2 if o[2] < o[0] else 1)                 # noqa: E731
    ret, _ = t.evaluate_fn(perfect, 5)
    assert ret > rnd + 3 and t.skill(ret) > t.skill(rnd)


def test_text_and_image_generation_tasks_are_built_by_the_general_model():
    from zerowhite.forge.generalist import Generalist
    from zerowhite.forge import task as T
    from zerowhite import datasets
    g = Generalist(tmp(), seed=0, verbose=False)
    text = ("the quick brown fox jumps over the lazy dog. " * 40)
    r = g.solve(T.TextTask("fox", text), budget=0.3, n_new=2, rungs=2)
    assert r["family"] == "char_lm" and r["test"]["bits_per_char"] < r["test"]["uniform"]
    assert isinstance(r["_agent"].sample(T.TextTask("fox", text), "the", 30), str)
    digits = T.from_dataset_task(datasets.load_digits(allow_bundled=True, download=False))
    img = T.ImageGenTask(digits)
    r2 = g.solve(img, budget=0.2, n_new=2, rungs=2)
    assert r2["family"] == "cvae" and 0.0 <= r2["test"]["fidelity"] <= 1.0
    assert "digits" in g.zoo.tasks()                          # the classifier it had to build first is kept too


# --------------------------------------------------------------------------- environments / POET
def test_param_cartpole_default_matches_cartpole_and_mountaincar_is_solvable():
    from zerowhite.agents import make_env
    from zerowhite.forge.envs import ParamCartPole, MountainCar
    a, b = make_env("cartpole"), ParamCartPole()
    oa, ob = a.reset(np.random.default_rng(0)), b.reset(np.random.default_rng(0))
    for k in range(30):
        oa, ra, da, _ = a.step(k % 2)
        ob, rb, db, _ = b.step(k % 2)
        assert np.allclose(oa, ob, atol=1e-9) and da == db
    m, done, steps = MountainCar(), False, 0
    o = m.reset(np.random.default_rng(0))
    while not done:
        o, _, done, won = m.step(2 if o[1] >= 0 else 0)
        steps += 1
    assert won and steps < 200


def test_poet_mutations_stay_inside_bounds_and_a_short_run_produces_an_archive():
    from zerowhite.forge import poet as P
    from zerowhite.forge.envs import ParamCartPole
    rng = np.random.default_rng(0)
    p = dict(ParamCartPole.DEFAULTS)
    for _ in range(100):
        p = P.mutate_params(p, rng, 0.8)
        assert all(ParamCartPole.BOUNDS[k][0] <= p[k] <= ParamCartPole.BOUNDS[k][1] for k in ParamCartPole.BOUNDS)
    r = P.poet(tmp(), rounds=2, seed=0, units=0.3, verbose=False, tries=4)
    assert len(r["environments"]) >= 1 and all("program" in e for e in r["environments"])


# --------------------------------------------------------------------------- doctor
def test_doctor_is_clean_on_a_fresh_workspace_and_detects_then_repairs_a_faulty_hook():
    ws = tmp()
    fs = doctor.run(ws, fix=False, quick=True)
    assert not [f for f in fs if f.level == "fail"], [f for f in fs if f.level == "fail"]
    mgr = hooks.HookManager(ws)
    mgr.install("sigma_update", "def sigma_update(sigma, improved, generation):\n    return sigma / 0.0\n", {})    # passes the gate, fails at runtime
    bad = [f for f in doctor.run(ws, quick=True) if f.area == "hooks"]
    assert bad and bad[0].level == "fail"
    fixed = [f for f in doctor.run(ws, fix=True, quick=True) if f.area == "hooks"]
    assert fixed and fixed[0].level == "fixed"
    assert mgr.live_source("sigma_update") is None                                  # rolled back to genesis
    assert [f for f in doctor.run(ws, quick=True) if f.area == "hooks"][0].level == "ok"


def test_doctor_quarantines_corrupt_zoo_pointers_and_removes_stale_temp_files():
    ws = tmp()
    zd = os.path.join(ws, "forge", "zoo", "t")
    os.makedirs(zd)
    fileio.write_json(os.path.join(zd, "best.json"), dict(id="ghost"))
    stale = os.path.join(ws, ".tmp_leftover.npz")
    with open(stale, "wb") as f:
        f.write(b"x")
    os.utime(stale, (0, 0))
    assert [f for f in doctor.run(ws, quick=True) if f.level == "fail" and f.area == "zoo"]
    doctor.run(ws, fix=True, quick=True)
    assert not os.path.exists(stale) and not os.path.exists(os.path.join(zd, "best.json"))
    assert glob.glob(os.path.join(zd, "best.json.corrupt-*"))                       # evidence kept, not deleted


def test_constitution_seal_detects_edits_and_is_crlf_insensitive():
    d = tmp()
    p = os.path.join(d, "f.py")
    with open(p, "wb") as f:
        f.write(b"a = 1\nb = 2\n")
    h_lf = doctor._sha(p)
    with open(p, "wb") as f:
        f.write(b"a = 1\r\nb = 2\r\n")
    assert doctor._sha(p) == h_lf                                                   # git autocrlf on Windows must not break the seal
    with open(p, "wb") as f:
        f.write(b"a = 1\nb = 3\n")
    assert doctor._sha(p) != h_lf


def test_shipped_seal_matches_the_constitution_files():
    assert os.path.exists(doctor.SEAL), "run: python -m zerowhite doctor --seal"
    out = []
    doctor.check_constitution(out)
    assert out[0].level == "ok", out[0].msg


# --------------------------------------------------------------------------- guarantees
NETWORK_MODULES_ALLOWED_IN = {                      # the only files that may import network code, and why
    "datasets.py",                                  # public dataset downloader (v3)
    "chat/corpus/fetch.py",                         # public dataset downloader for the chat model (explicit `chat fetch`)
    "chat/service/server.py",                       # the local HTTP service (binds 127.0.0.1 unless told otherwise)
    "chat/service/client.py",                       # talks to that local service
    "forage/fetch.py",                              # 6.3: the opt-in forager (`zerowhite forage --allow-host ...`): allow-listed, robots-aware, sealed
}


def test_no_language_model_or_llm_client_anywhere_and_network_code_is_confined_to_named_modules():
    banned = {"openai", "anthropic", "transformers", "torch", "tensorflow", "jax", "langchain", "litellm", "cohere",
              "google", "huggingface_hub", "requests", "httpx", "aiohttp", "llama_cpp", "ollama", "vllm"}
    netmods = {"urllib", "socket", "http", "ftplib", "smtplib", "ssl", "xmlrpc"}
    pkg = os.path.join(ROOT, "zerowhite")
    seen_allowed = set()
    for path in glob.glob(os.path.join(pkg, "**", "*.py"), recursive=True):
        mods = set()
        for node in ast.walk(ast.parse(fileio.read_text(path))):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods.add(node.module.split(".")[0])
        rel = os.path.relpath(path, pkg).replace(os.sep, "/")
        assert not (mods & banned), (rel, mods & banned)
        if rel in NETWORK_MODULES_ALLOWED_IN:
            seen_allowed.add(rel)
        else:
            assert not (mods & netmods), (rel, mods & netmods)
    assert seen_allowed == NETWORK_MODULES_ALLOWED_IN, "the allow-list names a file that no longer exists"


def test_cli_solve_and_doctor_work_from_a_cp1252_console():
    ws = tmp()
    env = dict(os.environ, PYTHONIOENCODING="cp1252", ZERO_WHITE_DATA=os.environ["ZERO_WHITE_DATA"])
    env.pop("PYTHONUTF8", None)
    rng = np.random.default_rng(0)                                                       # a table written by the test itself
    pts = rng.normal(size=(300, 3))
    csv_path = os.path.join(ws, "demo_classes.csv")
    os.makedirs(ws, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("x,y,noise,label\n")
        for x, y, nz in pts:
            f.write(f"{x:.5f},{y:.5f},{nz:.5f},{int(x * y > 0)}\n")
    r = subprocess.run([sys.executable, "-m", "zerowhite", "solve", "--workspace", ws, "--csv", csv_path,
                        "--budget", "0.3", "--candidates", "3", "--rungs", "2"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stderr[-800:]
    assert "winner" in r.stdout
    z = subprocess.run([sys.executable, "-m", "zerowhite", "zoo", "--workspace", ws, "--show"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=60)
    assert z.returncode == 0 and "demo_classes" in z.stdout
    d = subprocess.run([sys.executable, "-m", "zerowhite", "doctor", "--workspace", ws, "--quick"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=120)
    assert d.returncode == 0, d.stdout[-800:]
