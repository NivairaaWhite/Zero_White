"""Tests for 6.5 episodic memory (forge/episodic.py): the contrastive encoder, recall, splicing, and its use by the Generalist."""
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ZERO_WHITE_DATA", tempfile.mkdtemp(prefix="zw_test_data_"))

from zerowhite.forge import episodic as E                                # noqa: E402
from zerowhite.forge import task as T                                    # noqa: E402
from zerowhite.selfmod import astmut as A                                # noqa: E402

_DIRS = []


def tmp():
    d = tempfile.mkdtemp(prefix="zw_t65e_")
    _DIRS.append(d)
    return d


import atexit                                                          # noqa: E402
atexit.register(lambda: [shutil.rmtree(d, ignore_errors=True) for d in _DIRS])


def clf(name, d, seed, n=300, kind="classification"):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d))
    if kind == "classification":
        return T.from_arrays(name, X, (X[:, 0] + X[:, 1] > 0).astype(int), "classification", seed=seed)
    return T.from_arrays(name, X, X[:, 0] * 2.0 + 0.1 * r.normal(size=n), "regression", seed=seed)


# ======================================================================================================================================= the encoder
def test_the_contrastive_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    enc = E.Encoder(f_in=20, hidden=8, out=5, seed=1)
    XA = rng.normal(size=(6, 20))
    XB = XA + 0.1 * rng.normal(size=(6, 20))
    pos = np.eye(6)
    pos[0, 1] = pos[1, 0] = 1.0                                           # a multi-positive row: the SupCon case, not just plain InfoNCE
    _, g = enc.loss_grads(XA, XB, pos, 0.2)
    worst = 0.0
    for k in enc.p:
        for _ in range(8):
            idx = tuple(int(rng.integers(0, s)) for s in enc.p[k].shape)
            old = enc.p[k][idx]
            enc.p[k][idx] = old + 1e-6
            lp, _ = enc.loss_grads(XA, XB, pos, 0.2)
            enc.p[k][idx] = old - 1e-6
            lm, _ = enc.loss_grads(XA, XB, pos, 0.2)
            enc.p[k][idx] = old
            num = (lp - lm) / 2e-6
            worst = max(worst, abs(num - g[k][idx]) / (abs(num) + abs(g[k][idx]) + 1e-9))
    assert worst < 1e-5, worst


def test_training_pulls_records_of_one_task_together_and_generalises_to_unseen_descriptions():
    """12 synthetic tasks; each is described 5 ways (a shared core of words and numbers plus noise). The encoder is trained on 4 descriptions per task. The 5th, which it never saw, must find
    a record of ITS OWN task as nearest neighbour more often than the untrained encoder does."""
    rng = np.random.default_rng(3)
    mem = E.EpisodicMemory(tmp(), seed=0)
    words = [f"w{i}" for i in range(60)]
    held = []
    for t in range(12):
        core = list(rng.choice(words, 2, replace=False))
        base = rng.random(E.NUM)
        for r in range(5):
            text = " ".join(core + list(rng.choice(words, 8, replace=False)))
            num = np.clip(base + 0.5 * rng.normal(size=E.NUM), 0, 1)
            if r == 4:
                held.append((t, E.featurize(text, num)))
            else:
                mem.add("champion", text, num, task=f"task{t}", task_kind="classification", family=f"fam{t}", dim=10, skill=0.9)

    def heldout_accuracy():
        V = mem.vectors()
        hits = 0
        for t, f in held:
            q = mem.enc.embed(f)[0]
            hits += mem.records[int(np.argmax(V @ q))]["task"] == f"task{t}"
        return hits / len(held)
    before = heldout_accuracy()
    rep = mem.fit(steps=300)
    after = heldout_accuracy()
    assert rep["trained"] and rep["loss_last"] < 0.5 * rep["loss_first"], rep
    assert after >= before + 0.2 and after >= 0.7, (before, after)
    assert rep["retrieval_after"] > rep["retrieval_before"]


def test_too_few_records_are_not_trained_on_and_say_so():
    mem = E.EpisodicMemory(tmp())
    mem.add("champion", "a b c", np.zeros(E.NUM), task="a")
    r = mem.fit()
    assert r["trained"] is False and "nothing to contrast" in r["reason"]


# ======================================================================================================================================= recall
def test_a_new_task_recalls_the_champion_of_the_most_similar_earlier_task():
    mem = E.EpisodicMemory(tmp())
    for i in range(3):
        mem.remember_task(clf(f"small_binary_{i}", 6, i), family="mlp", genome=dict(hidden=16, depth=1), skill=0.9)
        mem.remember_task(clf(f"wide_regression_{i}", 40, 10 + i, kind="regression"), family="program_formula", genome=dict(n_islands=2), skill=0.8)
    mem.fit(steps=150)
    near = mem.recall(clf("fresh_binary", 6, 99), k=3)
    assert near and all(r["family"] == "mlp" for r in near), [(r["task"], r["family"]) for r in near]
    far = mem.recall(clf("fresh_wide", 40, 98, kind="regression"), k=3)
    assert far and far[0]["family"] == "program_formula" and far[0]["score"] > 0.5
    assert [r["score"] for r in near] == sorted((r["score"] for r in near), reverse=True)


def test_recall_skips_failures_low_skill_champions_and_optionally_the_task_itself():
    mem = E.EpisodicMemory(tmp())
    t = clf("same", 5, 1)
    mem.remember_task(t, family="mlp", genome=dict(hidden=8), skill=0.95)
    mem.remember_task(clf("other", 5, 2), family="mlp", genome=dict(hidden=9), skill=0.02)
    mem.remember_task(clf("bad", 5, 3), family="selfmod_net", genome=dict(hidden0=8), skill=-0.4, outcome="failure")
    got = mem.recall(t, k=5, min_skill=0.1)
    assert [r["task"] for r in got] == ["same"]
    assert mem.recall(t, k=5, min_skill=0.1, exclude_task=True) == []
    assert {r["task"] for r in mem.recall(t, k=5, kinds=("failure",))} == {"bad"}


def test_states_and_code_are_embedded_and_similar_code_is_found():
    mem = E.EpisodicMemory(tmp())
    mem.remember_code("cart", "def policy(obs):\n    return sel(obs[2] + 0.3 * obs[3], 1.0, 0.0)\n", 0.9, "program_policy")
    mem.remember_code("sort", "def predict(x):\n    return np.tanh(x[:, 0] * 2.0 + psqrt(x[:, 1]))\n", 0.8, "program_formula")
    got = mem.similar_code("def policy(obs):\n    return sel(obs[2] + 0.5 * obs[3], 1.0, 0.0)\n", k=2)
    assert got[0]["task"] == "cart" and got[0]["score"] > got[1]["score"]
    env = T.from_env_name("cartpole")
    obs = E.probe_obs(env, 80)
    assert obs.shape == (80, 4) and mem.remember_states(env, obs) == 4
    assert sum(r["kind"] == "state" for r in mem.records) == 4


def test_the_memory_survives_a_restart_and_an_unreadable_one_starts_empty():
    ws = tmp()
    mem = E.EpisodicMemory(ws)
    for i in range(4):
        mem.remember_task(clf(f"t{i}", 6, i), family="mlp", genome=dict(hidden=8 + i), skill=0.9)
    mem.fit(steps=40)
    mem.save()
    again = E.EpisodicMemory(ws)
    assert len(again) == 4 and np.allclose(again.vectors(), mem.vectors(), atol=1e-5) and again.enc.trained_steps == mem.enc.trained_steps
    with open(os.path.join(ws, "forge", "episodic", "records.json"), "w", encoding="utf-8") as f:
        f.write("{broken")
    assert len(E.EpisodicMemory(ws)) == 0


def test_the_oldest_least_valuable_records_are_forgotten_first_when_the_memory_is_full():
    old = E.MAX_RECORDS
    E.MAX_RECORDS = 10
    try:
        mem = E.EpisodicMemory(tmp())
        mem.add("champion", "keep me", np.zeros(E.NUM), task="champ", family="mlp", skill=0.9)
        for i in range(14):
            mem.add("state", f"state {i}", np.zeros(E.NUM), task=f"s{i}")
        assert len(mem) == 10 and len(mem.feats) == 10
        assert any(r["kind"] == "champion" for r in mem.records)
    finally:
        E.MAX_RECORDS = old


# ======================================================================================================================================= splicing
def _vocab(n_cols, name="policy", args=("obs",), col="obs", vector=False):
    return A.Vocab(name, args, col_var=col, n_cols=n_cols, vector=vector, max_nodes=80, leaves=())


def test_a_recalled_program_is_rewritten_to_the_new_width_and_a_bad_one_is_refused():
    v = _vocab(3)
    src = "def policy(obs):\n    return obs[7] + 0.5 * obs[2] - obs[0]\n"
    out = E.adapt_program(src, v)
    assert out is not None and "obs[1]" in out and "obs[7]" not in out                         # 7 wraps to 7 % 3
    assert A.compile_program(out, "policy", **v.safety)(np.array([1.0, 2.0, 3.0])) == 2.0 + 1.5 - 1.0
    assert E.adapt_program("def policy(obs):\n    import os\n    return 1\n", v) is None          # the safety gate applies to recalled code too
    assert E.adapt_program("def other(a, b):\n    return a\n", v) is None                       # wrong signature
    assert E.adapt_program("not python (", v) is None
    assert E.adapt_program("def policy(obs):\n    return " + " + ".join(["obs[0]"] * 200) + "\n", v) is None   # too big for this vocabulary
    vv = _vocab(4, "predict", ("x",), "x", vector=True)
    out2 = E.adapt_program("def predict(x):\n    return x[:, 9] * 2.0\n", vv)
    assert out2 is not None and "x[:, 1]" in out2


def test_seeds_for_returns_adapted_programs_and_their_crossovers_that_all_compile():
    mem = E.EpisodicMemory(tmp())
    progs = ["def policy(obs):\n    return sel(obs[2] + 0.3 * obs[3], 1.0, 0.0)\n", "def policy(obs):\n    return np.tanh(obs[0] - 2.0 * obs[1]) + obs[3]\n"]
    for i, s in enumerate(progs):
        mem.remember_task(T.from_env_name("cartpole"), family="program_policy", genome=dict(max_nodes=60), skill=0.8, src=s, fn_name="policy")
    v = _vocab(4)
    seeds = mem.seeds_for(T.from_env_name("cartpole:g=12"), v, np.random.default_rng(0), k=3)
    assert len(seeds) >= 2 and len(set(seeds)) == len(seeds)
    for s in seeds:
        A.compile_program(s, "policy", **v.safety)
    assert mem.seeds_for(T.from_env_name("cartpole:g=12"), _vocab(4, "predict", ("x",), "x", True), np.random.default_rng(0)) == []     # another kind of program: nothing recalled


def test_recalled_genomes_are_clipped_into_the_gene_ranges_and_blended():
    from zerowhite.forge.families import MLPFamily
    mem = E.EpisodicMemory(tmp())
    mem.remember_task(clf("a", 6, 1), family="mlp", genome=dict(hidden=9999, depth=1, lr=0.01, wd=1e-4, batch=32), skill=0.9)
    mem.remember_task(clf("b", 6, 2), family="mlp", genome=dict(hidden=20, depth=2, lr=0.02, wd=1e-3, batch=64), skill=0.8)
    got = mem.genomes_for(clf("c", 6, 3), "mlp", MLPFamily.space, k=2, rng=np.random.default_rng(0))
    assert len(got) == 3 and got[2][1].count("+") == 1                      # two recalled + one blend
    for g, _ in got:
        for k, gene in MLPFamily.space.items():
            assert gene.lo <= g[k] <= gene.hi, (k, g[k])
    clipped = [g for g, lab in got if lab == "a"][0]
    assert clipped["hidden"] == MLPFamily.space["hidden"].hi                  # the 9999 was clipped to the range, not trusted


# ======================================================================================================================================= in the Generalist
def test_the_generalist_recalls_earlier_champions_for_a_similar_task_and_records_what_it_learns():
    from zerowhite.forge.generalist import Generalist
    ws = tmp()
    g = Generalist(ws, seed=0, verbose=False)
    r1 = g.solve(clf("alpha", 6, 1), budget=0.25, n_new=3, rungs=2)
    assert r1["recalled"] == 0 and len(g.memory) >= 1
    r2 = g.solve(clf("beta", 6, 2), budget=0.25, n_new=6, rungs=2)
    assert r2["recalled"] >= 1
    assert any(r["kind"] == "champion" and r["task"] == "alpha" for r in g.memory.records)
    g2 = Generalist(ws, seed=1, verbose=False)                                                   # a new process reads the same memory
    assert len(g2.memory) == len(g.memory)
    off = Generalist(tmp(), seed=0, verbose=False, episodic=False)
    assert off.memory is None and off.solve(clf("gamma", 6, 3), budget=0.2, n_new=3, rungs=2)["recalled"] == 0


def test_a_broken_memory_never_stops_a_solve():
    from zerowhite.forge.generalist import Generalist
    ws = tmp()
    g = Generalist(ws, seed=0, verbose=False)
    g.solve(clf("alpha", 6, 1), budget=0.2, n_new=3, rungs=2)

    def boom(*a, **k):
        raise RuntimeError("memory exploded")
    g.memory.recall = boom
    g.memory.remember_task = boom
    r = g.solve(clf("beta", 6, 2), budget=0.2, n_new=3, rungs=2)
    assert r["skill"] > -1 and r["recalled"] == 0


def test_a_program_agent_starts_from_the_recalled_programs_not_only_the_trivial_one():
    from zerowhite.forge.families import ProgramPolicyFamily
    mem = E.EpisodicMemory(tmp())
    env = T.from_env_name("cartpole")
    good = "def policy(obs):\n    return sel(obs[2] + 0.3 * obs[3], 1.0, 0.0)\n"
    mem.remember_task(env, family="program_policy", genome=dict(max_nodes=60), skill=0.9, src=good, fn_name="policy")
    env2 = T.from_env_name("cartpole:g=12")
    env2._episodic, env2._recalled = mem, mem.recall(env2, k=4, exclude_task=True, min_skill=0.1)
    fam = ProgramPolicyFamily()
    genome = dict(n_islands=2, island_size=4, tune_steps=0, max_nodes=60, n_mut=1)
    a = fam.build(env2, genome, 0)
    assert a._memory_seeds and A.normalize(a._memory_seeds[0]) in a.pe.seen
    assert a.pe.best is not None and a.pe.best.score > -1.0                 # the recalled program was evaluated on the new task and already scores well
    plain = fam.build(T.from_env_name("cartpole:g=12"), genome, 0)
    assert plain._memory_seeds == []
