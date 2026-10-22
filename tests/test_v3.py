"""test_v3.py - tests for everything added in v3 (plain asserts, no pytest needed).
Run all tests with:  python3 tests/run_all.py"""

import ast
import glob
import gzip
import io
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from zerowhite import agents  # noqa: E402
from zerowhite import checkpoint  # noqa: E402
from zerowhite import datasets  # noqa: E402
from zerowhite import factory  # noqa: E402
from zerowhite import fileio  # noqa: E402
from zerowhite import report  # noqa: E402
from zerowhite.architecture import GrowableNet  # noqa: E402
from zerowhite.curiosity import ForwardDynamicsCuriosity  # noqa: E402
from zerowhite.engine import Tensor, bce_with_logits, embedding, policy_gradient_loss  # noqa: E402
from zerowhite.evolve import Evolver, Gene  # noqa: E402
from zerowhite.generative import CVAE, CharLM, CharVocab, split_text_tokens  # noqa: E402
from zerowhite.meta_optimizer import LearnedOptimizer  # noqa: E402
from zerowhite.nn import Adam, Params  # noqa: E402
from zerowhite.self_model import SelfModel  # noqa: E402
from zerowhite.tasks import make_shapes, split  # noqa: E402
from zerowhite.zero_white import ZeroWhite  # noqa: E402


def _tmp():
    return tempfile.mkdtemp()


def _num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(*x.shape):
        o = x[i]; x[i] = o + h; fp = f(x); x[i] = o - h; fm = f(x); x[i] = o
        g[i] = (fp - fm) / (2 * h)
    return g


def _check_grad(build, x, tol=1e-5):
    t = Tensor(x.copy(), True)
    build(t).backward()
    ng = _num_grad(lambda a: float(build(Tensor(a, True)).data), x.copy())
    assert np.abs(t.grad - ng).max() < tol, np.abs(t.grad - ng).max()


# ------------------------------------------------------------------ engine
def test_new_ops_match_numerical_gradients():
    rng = np.random.default_rng(0)
    z, t01 = rng.standard_normal((5, 7)), rng.random((5, 7))
    _check_grad(lambda t: bce_with_logits(t, t01), z)
    _check_grad(lambda t: bce_with_logits(t, t01, sum_features=False), z)
    acts = rng.integers(0, 7, 5)
    _check_grad(lambda t: policy_gradient_loss(t, acts, np.linspace(-1, 1, 5), 0.07), z)
    tab, idx = rng.standard_normal((9, 4)), rng.integers(0, 9, (5, 3))
    _check_grad(lambda t: (embedding(t, idx).reshape(5, 12) ** 2).sum(), tab)
    _check_grad(lambda t: (t.clip(-0.5, 0.5).exp() * t.sigmoid()).sum(), z)


def test_sigmoid_is_stable_for_extreme_inputs():
    with np.errstate(over="raise", invalid="raise", divide="raise"):   # underflow to 0 is fine; overflow/nan is not
        s = Tensor(np.array([-800.0, 0.0, 800.0])).sigmoid().data
    assert np.allclose(s, [0.0, 0.5, 1.0])


def test_backward_survives_a_very_deep_graph():
    x = Tensor(np.ones(3), True)
    y = x
    for _ in range(4000):                       # the old recursive backward() died near depth ~1000
        y = y + 1e-3
    y.sum().backward()
    assert np.allclose(x.grad, 1.0)


def test_constants_do_not_receive_gradients():
    X, w = Tensor(np.ones((4, 3))), Tensor(np.ones((3, 2)), True)
    X.matmul(w).sum().backward()
    assert X.grad is None and w.grad is not None


def test_input_gradient_still_works_for_fgsm():
    net = GrowableNet(6, 3, hidden=(5,), seed=0)
    g = net.input_gradient(np.random.default_rng(0).standard_normal((4, 6)), np.array([0, 1, 2, 0]))
    assert g.shape == (4, 6) and np.abs(g).sum() > 0


# ------------------------------------------------------------------ architecture
def test_predict_raw_matches_autograd_forward_with_and_without_patch():
    net = GrowableNet(8, 3, hidden=(5, 4), num_chunks=4, seed=1)
    X = np.random.default_rng(1).standard_normal((6, 8))
    assert np.allclose(net.predict_raw(X), net.forward(X)[0].data)
    net.toggle_hierarchical()
    assert np.allclose(net.predict_raw(X), net.forward(X)[0].data)


def test_add_layer_preserves_function_closely():
    for seed in range(5):
        net = GrowableNet(6, 3, hidden=(7,), seed=seed)
        X = np.random.default_rng(seed).standard_normal((10, 6))
        before = net.predict_raw(X)
        assert net.add_layer()
        assert np.abs(net.predict_raw(X) - before).max() < 0.3, "inserted layer must be near-identity"


def test_add_layer_never_inserts_in_front_of_raw_input():
    for seed in range(20):
        net = GrowableNet(6, 3, hidden=(7,), seed=seed)
        net.add_layer()
        assert net.layers[0]["W"].shape[0] == 6      # first layer still reads the real input


# ------------------------------------------------------------------ optimizers
def test_learned_optimizer_skips_nonfinite_gradients():
    net = GrowableNet(4, 2, hidden=(3,), seed=0)
    X = np.random.default_rng(0).standard_normal((5, 4))
    logits, params = net.forward(X)
    logits.sum().backward()
    for t in params.values():
        t.grad = np.full_like(t.data, np.nan)
    before = [l["W"].copy() for l in net.layers]
    assert LearnedOptimizer(seed=0).step(net, params) is False
    assert all(np.array_equal(b, l["W"]) for b, l in zip(before, net.layers))


def test_adam_clips_and_skips_bad_steps():
    p = Params({"w": np.zeros(3)})
    opt = Adam(lr=0.1, clip=1.0)
    leaf = {"w": Tensor(np.zeros(3), True)}
    leaf["w"].grad = np.array([np.inf, 0, 0])
    assert opt.step(p, leaf) is False and (p.a["w"] == 0).all()
    leaf["w"].grad = np.array([1e6, 0, 0])
    assert opt.step(p, leaf) is True and abs(p.a["w"][0]) <= 0.1 + 1e-9    # huge gradient, bounded step


def test_forward_dynamics_curiosity_bonus_drops_as_it_learns():
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((64, 4)); act = rng.integers(0, 2, 64)
    nxt = obs * 0.5 + act[:, None]
    icm = ForwardDynamicsCuriosity(4, 2, hidden=16, seed=0)
    b0 = icm.bonus(obs, act, nxt).mean()
    for _ in range(300):
        icm.update(obs, act, nxt)
    assert icm.bonus(obs, act, nxt).mean() < 0.2 * b0


# ------------------------------------------------------------------ self model
def test_self_model_history_is_bounded_but_lifetime_tally_is_not():
    sm = SelfModel(max_history=5)
    for i in range(20):
        sm.generation = i
        sm.log({"action": "noop"})
        sm.record_result(["noop", "grow"], "grow")
    assert len(sm.history) == 5 and sm.tally["grow"] == {"tried": 20, "won": 20}
    sm2 = SelfModel.from_dict(json.loads(json.dumps(sm.to_dict())))
    assert sm2.tally["grow"]["won"] == 20 and sm2.max_history == 5


# ------------------------------------------------------------------ file I/O + Windows encoding regression
def test_fileio_roundtrips_unicode_and_is_utf8_on_disk():
    p = os.path.join(_tmp(), "x.txt")
    fileio.write_text(p, "arrow \u2192 caf\u00e9 \u65e5\u672c")
    assert fileio.read_text(p) == "arrow \u2192 caf\u00e9 \u65e5\u672c"
    assert b"\xe2\x86\x92" in open(p, "rb").read()


def test_failed_write_leaves_old_file_intact_and_no_temp_files():
    d = _tmp(); p = os.path.join(d, "a.json")
    fileio.write_json(p, {"ok": 1})
    try:
        fileio.write_json(p, {"bad": object()})
    except TypeError:
        pass
    assert fileio.read_json(p) == {"ok": 1}
    assert os.listdir(d) == ["a.json"]


def test_no_text_file_is_opened_without_an_explicit_encoding():
    """The cp1252 UnicodeEncodeError came from open(p, 'w') with no encoding. Lint every source file."""
    bad = []
    for path in glob.glob(os.path.join(ROOT, "zerowhite", "**", "*.py"), recursive=True):
        for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            is_open = (isinstance(f, ast.Name) and f.id == "open") or \
                      (isinstance(f, ast.Attribute) and f.attr == "open" and isinstance(f.value, ast.Name) and f.value.id == "gzip")
            if not is_open:
                continue
            mode = None
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            if "b" in (mode or "") and not (mode or "").endswith("t"):
                continue
            if not any(kw.arg == "encoding" for kw in node.keywords):
                bad.append(f"{os.path.basename(path)}:{node.lineno}")
    assert not bad, f"text open() without encoding at {bad}"


def test_report_is_written_as_utf8_with_the_arrow_that_crashed_windows():
    d = _tmp()
    zw = ZeroWhite(64, 4, hidden=(5,), num_chunks=8, seed=0)
    X, y = make_shapes(0, 20, seed=0)
    Xtr, ytr, Xv, yv = split(X, y)
    zw.run(Xtr, ytr, Xv, yv, generations=3, probation_steps=4, verbose=False)
    zw.self_model.save(os.path.join(d, "run_journal.json"))
    p = report.write(d)
    raw = open(p, "rb").read()
    assert b"\xe2\x86\x92" in raw and raw.decode("utf-8")


def test_cli_works_under_a_cp1252_console_and_writes_the_report():
    d = _tmp()
    env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONUTF8="0", ZERO_WHITE_DATA=_tmp())
    app = ["-m", "zerowhite"]
    for cmd in (["train", "--out", d, "--task", "curriculum", "--generations", "2", "--probation", "3"], ["report", "--run", d]):
        r = subprocess.run([sys.executable] + app + cmd, env=env, cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-400:]
    assert os.path.exists(os.path.join(d, "report.html"))


# ------------------------------------------------------------------ checkpoints
def test_checkpoint_v3_preserves_momentum_and_random_streams():
    X, y = make_shapes(0, 30, seed=1)
    zw = ZeroWhite(64, 4, hidden=(5,), num_chunks=8, seed=2)
    zw.run(X, y, X, y, generations=3, probation_steps=5, verbose=False)
    d = _tmp(); checkpoint.save(zw, d)
    z2, _ = checkpoint.load(d)
    assert (zw.predict(X) == z2.predict(X)).all()
    assert set(z2.opt.momentum) == set(zw.opt.momentum) and z2.opt.momentum
    assert zw.rng.integers(0, 10**9) == z2.rng.integers(0, 10**9)         # resumes the stream, not seed 0
    assert z2.steps_trained == zw.steps_trained


def test_checkpoint_loads_the_older_v2_layout():
    X, y = make_shapes(0, 20, seed=1)
    zw = ZeroWhite(64, 4, hidden=(5,), num_chunks=8, seed=2)
    zw.run(X, y, X, y, generations=2, probation_steps=3, verbose=False)
    d = _tmp(); checkpoint.save(zw, d)
    w = np.load(os.path.join(d, checkpoint.FILE))
    meta = json.loads(bytes(w["__state__"]).decode("utf-8"))
    meta["base_lr"] = meta["opt"]["base_lr"]
    for k in ("opt", "rng", "patch_dims"):
        meta.pop(k)
    arrays = {k: w[k] for k in w.files if k != "__state__" and not k.startswith("mom.")}
    d2 = _tmp()
    np.savez(os.path.join(d2, "weights.npz"), **arrays)
    fileio.write_json(os.path.join(d2, "state.json"), meta)
    z2, _ = checkpoint.load(d2)
    assert (zw.predict(X) == z2.predict(X)).all()


def test_classifier_consolidation_and_minibatching_learn_real_digits():
    os.environ["ZERO_WHITE_DATA"] = _tmp()
    t = datasets.load_digits(allow_bundled=True, download=False)
    zw = factory.make_classifier(t, seed=0)
    zw.run(t.Xtr, t.ytr, t.Xval, t.yval, generations=12, probation_steps=20, verbose=False)
    assert zw.evaluate(zw.net, t.Xte, t.yte)[1] > 0.8        # chance is 0.10


# ------------------------------------------------------------------ datasets
def _with_sources(patch, fn):
    old, old_env = dict(datasets.SOURCES), os.environ.get("ZERO_WHITE_DATA")
    datasets.SOURCES.update(patch)
    os.environ["ZERO_WHITE_DATA"] = _tmp()
    try:
        return fn()
    finally:
        datasets.SOURCES.clear(); datasets.SOURCES.update(old)
        if old_env is None:
            os.environ.pop("ZERO_WHITE_DATA", None)
        else:
            os.environ["ZERO_WHITE_DATA"] = old_env


def _file_url(path):
    # Path.as_uri() yields file:///C:/... on Windows; the old "file://" + "C:/..." made C: the URL *host*
    return pathlib.Path(path).resolve().as_uri()


def test_parse_idx_roundtrip_and_rejects_corruption():
    arr = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    raw = bytes([0, 0, 8, 3]) + struct.pack(">III", 2, 3, 3) + arr.tobytes()
    assert (datasets.parse_idx(raw) == arr).all()
    for bad in (raw[:-1], b"\x01\x00\x08\x01" + raw[4:], bytes([0, 0, 9, 3]) + raw[4:]):
        try:
            datasets.parse_idx(bad)
        except ValueError:
            continue
        raise AssertionError("corrupt IDX accepted")


def test_fashion_loader_with_local_mirror_and_official_split():
    d = _tmp(); rng = np.random.default_rng(0)
    for split_, n in (("train", 200), ("t10k", 50)):
        imgs = rng.integers(0, 256, (n, 28, 28), dtype=np.uint8); labs = (np.arange(n) % 10).astype(np.uint8)
        with gzip.open(os.path.join(d, f"{split_}-images-idx3-ubyte.gz"), "wb") as f:
            f.write(bytes([0, 0, 8, 3]) + struct.pack(">III", n, 28, 28) + imgs.tobytes())
        with gzip.open(os.path.join(d, f"{split_}-labels-idx1-ubyte.gz"), "wb") as f:
            f.write(bytes([0, 0, 8, 1]) + struct.pack(">I", n) + labs.tobytes())
    t = _with_sources({"fashion": ["file:///definitely/not/here/", _file_url(d) + "/"]}, datasets.load_fashion)
    assert len(t.Xtr) + len(t.Xval) == 200 and len(t.Xte) == 50 and t.in_dim == 784 and t.num_chunks == 28
    assert 0 <= t.Xtr.min() and t.Xtr.max() <= 1.0 and t.image_shape == (28, 28)


def test_corrupt_cached_download_is_deleted_with_a_clear_error():
    d = _tmp()
    for fn in datasets._FASHION_FILES:
        with gzip.open(os.path.join(d, fn), "wb") as f:
            f.write(b"not an idx file at all")
    def go():
        try:
            datasets.load_fashion()
        except datasets.DatasetUnavailable as e:
            assert "corrupt" in str(e)
            return True
        return False
    assert _with_sources({"fashion": [_file_url(d) + "/"]}, go)


def test_sonar_loader_maps_labels_and_standardizes_on_train_only():
    d = _tmp(); rng = np.random.default_rng(0)
    lines = [",".join(f"{v:.4f}" for v in rng.random(60)) + ("," + ("R" if i % 2 else "M")) for i in range(60)]
    open(os.path.join(d, "sonar.csv"), "w", encoding="utf-8").write("\n".join(lines))
    t = _with_sources({"sonar": ["file:///nope/sonar.all-data", _file_url(os.path.join(d, "sonar.csv"))]}, datasets.load_sonar)
    assert t.in_dim == 60 and t.n_classes == 2 and abs(t.Xtr.mean()) < 1e-6
    assert len(t.Xtr) + len(t.Xval) + len(t.Xte) == 60 and t.class_names == ("rock", "mine")


def test_digits_loader_uses_official_split_from_the_uci_zip():
    d = _tmp(); rng = np.random.default_rng(0)
    def rows(n):
        return "\n".join(",".join(map(str, list(rng.integers(0, 17, 64)) + [i % 10])) for i in range(n))
    with zipfile.ZipFile(os.path.join(d, "o.zip"), "w") as z:
        z.writestr("optdigits.tra", rows(100)); z.writestr("optdigits.tes", rows(30)); z.writestr("optdigits.names", "x")
    t = _with_sources({"digits_zip": [_file_url(os.path.join(d, "o.zip"))]}, datasets.load_digits)
    assert len(t.Xte) == 30 and len(t.Xtr) + len(t.Xval) == 100 and "official" in t.source and t.Xtr.max() <= 1.0


def test_digits_falls_back_to_the_bundled_real_subset_offline():
    t = _with_sources({"digits_zip": ["file:///nope/o.zip"]}, datasets.load_digits)
    assert "bundled" in t.source and len(t.Xtr) + len(t.Xval) + len(t.Xte) == 1797


def test_all_sources_failing_names_the_file_to_place_by_hand():
    def go():
        try:
            datasets.load_sonar()
        except datasets.DatasetUnavailable as e:
            return "sonar.all-data" in str(e) and "place it at" in str(e)
        return False
    assert _with_sources({"sonar": ["file:///a/b", "file:///c/d"]}, go)


def test_quickdraw_prefix_reader_and_loader_with_spaces_in_class_names():
    arr = np.random.default_rng(0).integers(0, 256, (10, 784), dtype=np.uint8)
    buf = io.BytesIO(); np.save(buf, arr)
    assert (datasets._npy_prefix(io.BytesIO(buf.getvalue()), 4) == arr[:4]).all()
    d = _tmp()
    for name in ("circle", "ice cream"):
        np.save(os.path.join(d, name + ".npy"), arr)
    t = _with_sources({"quickdraw": [_file_url(d) + "/"]},
                      lambda: datasets.load_quickdraw(("circle", "ice cream"), per_class=6))
    assert t.n_classes == 2 and len(t.Xtr) + len(t.Xval) + len(t.Xte) == 12 and t.class_names[1] == "ice cream"


def test_splits_are_disjoint_stratified_and_chunks_divide_the_input():
    X = np.arange(300 * 4, dtype=float).reshape(300, 4); y = np.repeat(np.arange(3), 100)
    t = datasets.make_task("t", X, y, seed=1)
    ids = [set(map(tuple, a)) for a in (t.Xtr, t.Xval, t.Xte)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert all(abs(np.mean(a == 0) - 1 / 3) < 0.05 for a in (t.ytr, t.yval, t.yte))
    assert [datasets.best_chunks(d) for d in (64, 784, 60)] == [8, 28, 6]


# ------------------------------------------------------------------ evolver
def test_evolver_optimises_and_children_inherit_only_when_shape_genes_match():
    space = {"x": Gene("float", -10, 10, shape=True), "lr": Gene("log", 1e-4, 1.0)}
    seen = []
    def ev_fn(g, parent):
        seen.append((parent is not None, g["x"]))
        return -(g["x"] - 3) ** 2 - (np.log10(g["lr"]) + 2) ** 2, dict(g), {}
    ev = Evolver(space, ev_fn, population=6, mu=2, seed=0)
    assert ev.run(25, verbose=False).fitness > -0.5
    assert any(inherited for inherited, _ in seen)


def test_evolver_never_regresses_below_its_incumbent():
    space = {"x": Gene("float", 0, 1)}
    ev = Evolver(space, lambda g, p: (-1.0, g, {}), population=3, mu=2, seed=0)
    ev.inject({"x": 0.5}, 5.0, "incumbent")
    ev.run(4, verbose=False)
    assert ev.best.fitness == 5.0 and ev.best.artifact == "incumbent" and ev.improved is False


def test_evolver_shape_gene_change_trains_from_scratch():
    space = {"w": Gene("int", 1, 50, shape=True)}
    flags = []
    ev = Evolver(space, lambda g, p: (flags.append(p is None) or float(g["w"]), g, {}), population=4, mu=1, seed=1)
    ev.inject({"w": 25}, 0.0, "parent")
    ev.step()
    assert all(flags), "an int shape-gene always moves, so no child may inherit the parent's weights"


# ------------------------------------------------------------------ generative models
def test_cvae_learns_samples_valid_images_and_roundtrips():
    X, y = make_shapes(0, 60, seed=0)
    X = np.clip(X, 0, 1)
    g = CVAE(64, 4, latent=4, hidden=48, beta=0.3, seed=0)
    e0 = g.elbo(X, y)["elbo"]
    g.fit(X, y, 400)
    assert g.elbo(X, y)["elbo"] < 0.7 * e0
    s = g.sample(np.arange(8) % 4, np.random.default_rng(0))
    assert s.shape == (8, 64) and s.min() >= 0 and s.max() <= 1
    p = os.path.join(_tmp(), "g.npz"); g.save(p)
    assert np.allclose(CVAE.load(p).sample([0, 1], np.random.default_rng(3)), g.sample([0, 1], np.random.default_rng(3)))


def test_charlm_learns_a_pattern_generates_it_and_roundtrips():
    text = "abcabcabc " * 200
    vocab = CharVocab.from_text(text); tok = vocab.encode(text)
    lm = CharLM(len(vocab), ctx=6, emb=8, hidden=48, seed=0)
    b0 = lm.bits_per_char(tok)
    lm.fit(tok, 400)
    assert lm.bits_per_char(tok) < 0.3 * b0
    out = lm.generate(vocab, "abcabc", 20, 0.3, np.random.default_rng(0))
    assert out.startswith("abcabc") and "?" not in out
    p = os.path.join(_tmp(), "lm.npz"); lm.save(p, vocab)
    lm2, v2 = CharLM.load(p)
    assert v2.chars == vocab.chars and lm2.bits_per_char(tok) == lm.bits_per_char(tok)


def test_text_split_is_stable_when_text_is_appended():
    rng = np.random.default_rng(0)
    tok = rng.integers(1, 30, 12000)
    tr1, va1 = split_text_tokens(tok)
    tr2, va2 = split_text_tokens(np.concatenate([tok, rng.integers(1, 30, 4000)]))
    assert len(va1) > 0 and np.array_equal(va2[:len(va1)], va1), "old validation text must stay validation text"


# ------------------------------------------------------------------ agents
def test_gridworld_layouts_are_always_solvable():
    env = agents.GridWorld(5); rng = np.random.default_rng(0)
    for _ in range(100):
        env.reset(rng)
        assert env._reachable(env.walls, env.pos, env.goal) and env.pos != env.goal


def test_cartpole_falls_without_control():
    env = agents.CartPole(); env.reset(np.random.default_rng(0))
    for t in range(200):
        if env.step(0)[2]:
            break
    assert t < 60


def test_gridworld_pays_for_the_goal_and_blocks_walls_and_edges():
    g = agents.GridWorld(5)
    g.pos, g.goal, g.walls, g.t = (2, 2), (2, 3), {(1, 2)}, 0
    assert g.step(0)[1] == -g.step_cost and g.pos == (2, 2)            # wall above: stays put
    g.pos = (0, 0); g.step(0); g.step(2)
    assert g.pos == (0, 0)                                              # edges block too
    g.pos = (2, 2)
    obs, r, done, won = g.step(3)
    assert r == 1.0 and done and won and obs.shape == (75,)


def test_actor_critic_learns_cartpole_from_scratch():
    env = agents.make_env("cartpole")
    a = agents.PGAgent(4, 2, hidden=32, seed=0)
    before = a.evaluate(env, 10)[0]
    a.train(env, updates=60, episodes=8, rng=np.random.default_rng(0))
    assert before < 40 and a.evaluate(env, 10)[0] > 100


def test_agent_roundtrip_keeps_the_policy():
    a = agents.PGAgent(4, 2, hidden=16, seed=0)
    a.train(agents.make_env("cartpole"), updates=3, episodes=4, rng=np.random.default_rng(0))
    p = os.path.join(_tmp(), "a.npz"); a.save(p)
    b = agents.PGAgent.load(p)
    obs = np.random.default_rng(0).standard_normal((5, 4))
    assert np.allclose(a.policy(obs), b.policy(obs)) and b.updates_trained == 3


# ------------------------------------------------------------------ autopilot (end to end, offline)
def test_autopilot_builds_everything_then_resumes_instead_of_restarting():
    ws = factory.Workspace(_tmp())
    corpus = os.path.join(_tmp(), "corpus.txt")
    open(corpus, "w", encoding="utf-8").write("the quick brown fox jumps over the lazy dog. " * 120)
    kw = dict(datasets_=["shapes"], texts=[corpus], envs=["cartpole"], seed=0, verbose=False, class_generations=3,
              gen_generations=1, lm_generations=1, agent_generations=1, scale=0.15,
              dataset_kwargs={"shapes": dict(n_per_class=60)})
    r1 = factory.autopilot(ws, rounds=1, **kw)
    assert [r["kind"] for r in r1] == ["classifier", "generator", "lm", "agent"]
    for sub in ("classifier/shapes0/weights.npz", "generators/shapes0/cvae.npz", "generators/shapes0/samples.png",
                "lm/corpus/lm.npz", "agents/cartpole/agent.npz", "journal.jsonl"):
        assert os.path.exists(os.path.join(ws.root, sub)), sub
    r2 = factory.autopilot(ws, rounds=1, **kw)
    assert r2[0]["test_acc_before"] is not None and r2[0]["generations"] == 6        # continued, not restarted
    assert r2[2]["steps"] > r1[2]["steps"] and r2[3]["updates"] > r1[3]["updates"]    # models kept training
    assert len(ws.journal()) == 8
    html_path = report.write(ws.root)
    assert "shapes0" in open(html_path, encoding="utf-8").read()


def test_autopilot_respects_a_time_budget_and_survives_interruption():
    ws = factory.Workspace(_tmp())
    assert factory.autopilot(ws, envs=["cartpole"], rounds=5, seconds=0, verbose=False) == []
    from zerowhite import factory as f
    orig = f.improve_agent
    f.improve_agent = lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
    try:
        f.autopilot(ws, envs=["cartpole"], verbose=False)
    finally:
        f.improve_agent = orig
    assert ws.journal()[-1]["kind"] == "interrupted"
