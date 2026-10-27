"""Tests for 6.4: the items 6.3 listed as 'not done'. Same rule as test_v63: each test must fail when the property it names breaks."""
import ast
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("ZERO_WHITE_DATA", tempfile.mkdtemp(prefix="zw_test_data_"))

from zerowhite import fileio                                           # noqa: E402


def tmp():
    return tempfile.mkdtemp(prefix="zw_t64_")


# ================================================================================================================ e-graph
def test_the_egraph_finds_simplifications_the_old_simplifier_cannot():
    from zerowhite.selfmod import egraph as E
    cases = {
        "np.minimum(np.minimum(obs[0], 3.0), 2.0)": "np.minimum(obs[0], 2.0)",
        "np.maximum(obs[0], np.minimum(obs[0], obs[1]))": "obs[0]",
        "((obs[0] - obs[0]) + obs[1]) * 1.0 + 0.0": "obs[1]",
        "np.maximum(obs[0], 0.0)": "relu(obs[0])",
        "-(-obs[0])": "obs[0]",
        "np.abs(np.abs(obs[0]))": "np.abs(obs[0])",
        "obs[0] * obs[0]": "np.square(obs[0])",
        "(obs[0] * obs[1]) + (obs[0] * obs[2])": None,                      # a factored form of size 5 (operand order is not asserted)
    }
    for src, want in cases.items():
        new, before, after = E.simplify_expr(src)
        assert after < before, (src, new)
        if want is not None:
            assert new == want, (src, new)
    prog = "def policy(obs):\n    t = np.minimum(obs[0], 3.0)\n    u = np.minimum(t, 2.0)\n    return (u * 1.0) + (obs[1] - obs[1])\n"
    out = E.simplify(prog)
    assert "np.minimum(obs[0], 2.0)" in out and "obs[1]" not in out and out.count("\n") <= 3


def test_simplified_programs_compute_the_same_numbers_on_hundreds_of_random_evolved_programs():
    from zerowhite.selfmod import astmut as A, egraph as E
    voc = A.Vocab("policy", ("obs",), col_var="obs", n_cols=6, vector=False, max_nodes=60, leaves=())
    rng = np.random.default_rng(7)
    smaller = same = 0
    for i in range(300):
        body = A.rand_expr_src(rng, voc, scope=(), depth=int(rng.integers(2, 5)))
        src = f"def policy(obs):\n    return {body}\n"
        if rng.random() < 0.3:                                              # sometimes with temporaries, as evolution writes them
            src = f"def policy(obs):\n    t = {A.rand_expr_src(rng, voc, scope=(), depth=2)}\n    return {body} + t\n"
        out = E.simplify(src)
        assert A.count_nodes(out) <= A.count_nodes(src)
        fa, fb = A.compile_program(src, "policy"), A.compile_program(out, "policy")
        with np.errstate(all="ignore"):
            for _ in range(20):                                             # inputs the simplifier never saw, wider range than its own probes
                o = rng.normal(0, 3, size=6)
                a, b = float(np.asarray(fa(o), dtype=float)), float(np.asarray(fb(o), dtype=float))
                assert (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-7 * max(1.0, abs(a)), (src, out, a, b)
        smaller += A.count_nodes(out) < A.count_nodes(src)
        same += out == src
    assert smaller >= 20, smaller                                           # it does find savings in a meaningful share of random programs


def test_the_egraph_leaves_what_it_does_not_understand_alone_and_its_safety_net_catches_a_wrong_rule():
    from zerowhite.selfmod import egraph as E
    loop = "def policy(obs):\n    s = 0.0\n    for i in range(3):\n        s = s + obs[i]\n    return s\n"
    branch = "def policy(obs):\n    if obs[0] > 0:\n        return obs[1]\n    return obs[2]\n"
    twice = "def policy(obs):\n    t = obs[0]\n    t = t + 1.0\n    return t - 1.0\n"
    for src in (loop, branch, twice, "def policy(obs):\n    return unknown_fn(obs[0]) + 0.0\n"):
        assert E.simplify(src) == src
    real = E.EGraph._proposals

    def wrong(self, cid, node):                                              # a deliberately unsound rule: a + b = a
        out = real(self, cid, node)
        if node[0] == "add":
            out.append(node[1])
        return out
    E.EGraph._proposals = wrong
    try:
        src = "def policy(obs):\n    return (obs[0] + obs[1]) + 0.0\n"
        assert E.simplify(src) == src                                       # the numeric equivalence check refuses the unsound result
    finally:
        E.EGraph._proposals = real
    assert E.simplify("def policy(obs):\n    return (obs[0] + obs[1]) + 0.0\n").count("obs[1]") == 1


def test_a_saved_program_is_the_cleaned_one_and_its_cleaning_never_changes_what_it_does():
    import test_v63 as T63
    from zerowhite.forge import families as F, task as T
    from zerowhite.selfmod import astmut as A
    task = T.EnvTask(T63.HiDimBandit())
    genome = dict(n_islands=2, island_size=6, tune_steps=4, max_nodes=60, n_mut=2)
    a = F.ProgramPolicyAgent(task, genome, 0)
    a.train(task, 1.0, np.random.default_rng(0))
    d = tmp()
    a.save(d)
    shown = fileio.read_text(os.path.join(d, "program.py"))
    assert shown == shown.split("def policy")[0] + a._clean(a.best_src())                   # exactly what describe() / the zoo show
    live, clean = A.compile_program(a.best_src(), "policy"), A.compile_program(a._clean(a.best_src()), "policy")
    rng = np.random.default_rng(3)
    for _ in range(50):
        o = a.proj(rng.normal(0, 2, size=40))
        assert float(np.asarray(live(o), dtype=float)) == float(np.asarray(clean(o), dtype=float)) or abs(float(live(o)) - float(clean(o))) < 1e-9


def test_the_simplifier_output_does_not_depend_on_the_hash_seed():
    code = ("import sys; sys.path.insert(0, %r)\nfrom zerowhite.selfmod import egraph as E\n"
            "for e in ['np.minimum(np.minimum(obs[0], 3.0), 2.0)', '(obs[0] * obs[1]) + (obs[0] * obs[2])', 'np.minimum(np.minimum(np.minimum(obs[0], obs[1]), obs[0]), np.maximum(obs[1], obs[0]))']:\n"
            "    print(E.simplify_expr(e)[0])\n" % ROOT)
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, PYTHONHASHSEED=str(h)), timeout=60).stdout for h in (0, 1, 2, 3, 99)}
    assert len(outs) == 1 and "np.minimum(obs[0], 2.0)" in outs.pop()


# ============================================================================================================ bits metric
def test_bits_conversions_match_the_information_theory_they_claim():
    from zerowhite import metric as M
    assert M.symmetric_channel_bits(1.0, 2) == 1.0 and M.symmetric_channel_bits(0.5, 2) == 0.0 and M.symmetric_channel_bits(0.3, 2) == 0.0
    assert abs(M.symmetric_channel_bits(1.0, 8) - 3.0) < 1e-12                           # a perfect 8-way classifier reveals 3 bits per example
    assert abs(M.symmetric_channel_bits(0.9, 2) - (1 - 0.4689955935892812)) < 1e-9        # 1 - H(0.1)
    assert abs(M.classification_bits(1.0, 2, 0.5, 100) - 100.0) < 1e-9 and M.classification_bits(0.0, 2, 0.5, 100) == 0.0
    imb = M.classification_bits(1.0, 2, 0.9, 100)                                          # a 90/10 task: the majority guess is worth nothing, perfect is worth H(0.1) bits each
    assert abs(imb - 100 * 0.4689955935892812) < 1e-6
    assert abs(M.regression_bits(0.75, 100) - 100.0) < 1e-9 and M.regression_bits(0.0, 100) == 0.0
    assert abs(M.lm_bits(1.0, 1024, 10) - 100.0) < 1e-9                                    # log2(1024) = 10 bits per token
    ks = [M.classification_bits(s, 5, 0.3, 50) for s in np.linspace(0, 1, 21)]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(ks, ks[1:]))                             # monotone in skill: ordering on one task never changes
    bits, basis = M.skill_to_bits(dict(kind="other"), 0.4, 50.0)
    assert (bits, basis) == (20.0, "assumed") and M.skill_to_bits(dict(kind="regression", n=10), 0.5)[1] == "measured"
    assert M.skill_to_bits(None, 0.5)[1] == "assumed" and M.skill_to_bits(dict(kind="classification", k=3, chance=0.4, n=100), float("nan") if False else 0.0)[0] == 0.0


class _Scripted(__import__("zerowhite.organism", fromlist=["x"]).Organism):
    meta = {}
    skills = {}

    def _act_solve(self, dec, t0):
        t = dec["task"]
        sk = self.skills[t].pop(0) if len(self.skills[t]) > 1 else self.skills[t][0]
        prev = self.state["tasks"][t]["best_skill"]
        return dict(ok=True, skill=sk, improved=prev is None or sk > prev + 1e-9, family="mlp", seconds=1.0, meta=self.meta.get(t))


def _two_task_organism(metric):
    import test_v6 as T6
    from zerowhite import organism as O
    inbox, ws = tmp(), tmp()
    for i, name in enumerate(("a_big", "z_small")):
        T6.put(inbox, f"{name}.csv", T6.csv_bytes(100, seed=200 + i))
    cls = type("S64", (_Scripted,), dict(meta={"a_big": dict(kind="classification", k=2, chance=0.5, n=2000), "z_small": dict(kind="classification", k=2, chance=0.5, n=20)},
                                         skills={"a_big": [0.3, 0.4, 0.5], "z_small": [0.3, 0.4, 0.5]}))
    return cls(ws, O.Settings(inbox=[inbox], quiet=True, probe_budget=0.25, standard_budget=1.0, patience=50, min_gain=0.001, doctor_every=0, metric=metric))


def test_bits_scheduling_prefers_the_task_with_more_to_learn_where_skill_cannot_tell_them_apart():
    picks = {}
    for metric in ("skill", "bits"):
        org = _two_task_organism(metric)
        for _ in range(2):                                                                 # one baseline attempt each ...
            org.step()
        assert {t["attempts"] for t in org.state["tasks"].values()} == {1}
        dec = org.decide(dict(free_gb=8.0, new_tasks=[], counts={}))                       # ... then who is next?
        picks[metric] = dec["task"]
        if metric == "bits":
            assert "bits" in dec["reason"] and org.expected_bits(org.state["tasks"]["a_big"]) > 50 * org.expected_bits(org.state["tasks"]["z_small"])
    assert picks == {"skill": "z_small", "bits": "a_big"}, picks                          # equal skill gains: skill breaks the tie by name, bits by what is at stake


def test_the_default_metric_is_unchanged_and_the_cli_exposes_the_choice():
    from zerowhite import organism as O
    assert O.Settings().metric == "skill"
    r = subprocess.run([sys.executable, "-m", "zerowhite", "organism", "--help"], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert "--metric" in r.stdout and "--bits-per-skill" in r.stdout


# ========================================================================================================= learned sniffer
def test_a_short_row_in_an_all_text_table_no_longer_crashes_the_parser():
    """6.2 bug: padded cells are None and the header check called .strip() on them, so any all-text table with one short row raised AttributeError."""
    from zerowhite.omnivore import tables
    words = ["red", "green", "blue", "north", "south"]
    rows = ["colour,place,size"] + [f"{words[i % 5]},{words[(i + 1) % 5]},{words[(i + 2) % 5]}" for i in range(30)]
    rows[7] = "red,north"
    t = tables.parse_delimited("\n".join(rows) + "\n")
    assert t is not None and t.columns == ["colour", "place", "size"] and any("padded" in r for r in t.repairs)


def test_the_shipped_ranker_picks_the_delimiter_on_files_it_never_saw_better_than_the_rule():
    from zerowhite.omnivore import learned, synth, tables
    R = learned.get_ranker()
    assert os.path.getsize(learned.RANK_WEIGHTS) < 20_000 and learned.delim_features("a,b\n1,2\n3,4\n").shape == (len(learned.DELIMS), learned.N_RANK)
    for hard in (0.0, 1.0):
        ds = synth.dataset(500, 7777, hard)                                                    # a seed that neither training nor my tuning used
        got = sum(R.rank(l["text"])[0] == l["delim"] for _, l in ds) / len(ds)
        rule = sum(tables.find_delimiter(l["text"])[0] == l["delim"] for _, l in ds) / len(ds)
        assert got > 0.97 and got > rule + 0.1, (hard, got, rule)


def _tables_by_mode(ds, mode):
    from zerowhite.omnivore import learned, sniff
    out = []
    for data, lab in ds:
        d = sniff.decode(data)
        out.append(learned.parse_with_hint(d.text, data, mode) if d else None)
    return out


def test_rescue_mode_only_ever_adds_tables_and_arbitrate_fixes_what_the_rules_get_wrong():
    from zerowhite.omnivore import synth
    ds = synth.dataset(500, 4242, 0.5)
    off, rescue, arb = (_tables_by_mode(ds, m) for m in ("off", "rescue", "arbitrate"))
    rescued = 0
    for (data, lab), a, b in zip(ds, off, rescue):
        if a is not None:
            assert b is not None and b.delimiter == a.delimiter and b.columns == a.columns       # a rule that succeeded is never overruled in rescue mode
        elif b is not None:
            rescued += 1
            assert b.delimiter == lab["delim"] and any("learned delimiter ranker" in r for r in b.repairs)
    assert rescued >= 3, rescued                                                                # it does rescue files the rules gave up on
    right = lambda tabs: sum(t is not None and t.delimiter == lab["delim"] for t, (_, lab) in zip(tabs, ds))     # noqa: E731
    assert right(off) < 0.85 * len(ds) and right(arb) > 0.93 * len(ds) and right(arb) > right(off) + 0.1 * len(ds), (right(off), right(rescue), right(arb))


def test_arbitrate_reads_a_space_delimited_european_table_that_the_rules_split_on_the_decimal_commas():
    from zerowhite.omnivore import learned, tables
    rows = ["id temp city"] + [f"{i} {20 + i * 0.37:.2f}".replace(".", ",") + f" {['Köln', 'Zürich', 'Genève'][i % 3]}" for i in range(40)]
    data = ("\n".join(rows) + "\n").encode("utf-8")
    text = data.decode("utf-8")
    rule = tables.parse_delimited(text)
    assert rule is not None and rule.delimiter == ","                                           # the rules: "consistent" comma counts win over whitespace
    assert learned.parse_with_hint(text, data, "rescue").delimiter == ","                       # rescue never overrules a table the rules made
    t = learned.parse_with_hint(text, data, "arbitrate")
    assert t.delimiter == " " and t.columns == ["id", "temp", "city"] and any("learned delimiter ranker" in r for r in t.repairs)
    assert tables.table_quality(t) == 2 > tables.table_quality(rule)
    assert tables._decimal_mode(t) == ","                                                       # the decimal mark then follows from the (unchanged) rule


def test_prose_is_never_turned_into_a_table_by_the_ranker_in_any_mode():
    from zerowhite.omnivore import learned
    sentences = ("The committee met on Tuesday to discuss the quarterly report, the budget for next year, and who would write the summary.\n"
                 "Several members, including the chair, raised concerns; others, however, were content. Minutes were not taken.\n") * 8
    words = "\n".join("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu" for _ in range(20)) + "\n"
    for text in (sentences, words):
        base = learned.parse_with_hint(text, text.encode("utf-8"), "off")
        for mode in ("rescue", "arbitrate"):
            got = learned.parse_with_hint(text, text.encode("utf-8"), mode)
            assert (got is None) == (base is None) and (got is None or got.delimiter == base.delimiter), (mode, got and got.delimiter, base and base.delimiter)


def test_without_the_weights_or_with_the_switch_off_devour_behaves_exactly_as_before():
    from zerowhite.omnivore import learned, devour
    inbox = tmp()
    path = os.path.join(inbox, "a.csv")
    fileio.write_text(path, "a,b,label\n" + "\n".join(f"{i},{i % 7},{i % 2}" for i in range(40)) + "\n")
    old = time.time() - 60.0                                                                    # devour leaves a file younger than `settle_seconds` (1s) alone as "deferred"; without this the
    os.utime(path, (old, old))                                                                  # test compared 'deferred' with 'deferred' on a fast machine and flipped to 'task' on a slow one
    base = {k: v for k, v in devour(inbox, tmp(), learned="off", seed=0).counts().items()}
    assert base == {"task": 1}, base                                                            # a real digest, so the runs below reach the ranker instead of all being deferred
    real_r = dict(learned._rcache)
    learned._rcache["r"] = None                                                                 # the weights file is missing
    try:
        assert learned.hint(b"a,b\n1,2\n3,4\n") is None
        assert devour(inbox, tmp(), seed=0).counts() == base
    finally:
        learned._rcache.clear()
        learned._rcache.update(real_r)
    assert devour(inbox, tmp(), learned="rescue", seed=0).counts() == base and devour(inbox, tmp(), learned="arbitrate", seed=0).counts() == base


def _ood_files():
    """Real-world-shaped files written independently of synth.py: (text, delim, decimal or None, header or None)."""
    import random
    R = random.Random(5)
    out = []
    cities = ["Berlin", "Lyon", "Porto", "Leeds", "Oslo", "Graz", "Turin", "Gent", "Bern", "Riga"]
    def dec(x, d, n=2): return f"{x:.{n}f}".replace(".", d)
    n = 60
    out.append(("date,temp_c,humidity,wind_kph,condition\n" + "\n".join(f"2024-03-{1 + i % 28:02d},{dec(R.gauss(12, 6), '.', 1)},{R.randint(30, 99)},{dec(R.uniform(0, 40), '.', 1)},{R.choice(['rain', 'sun', 'fog'])}" for i in range(n)) + "\n", ",", ".", True))
    out.append(("Produkt;Menge;Preis;Umsatz\n" + "\n".join(f"{R.choice(cities)};{R.randint(1, 500)};{dec(R.uniform(1, 90), ',')};{R.randint(1, 9)}.{R.randint(100, 999)},{R.randint(10, 99)}" for i in range(n)) + "\n", ";", ",", True))
    out.append(("\n".join("\t".join(dec(R.gauss(0, 1), '.', 4) for _ in range(6)) for _ in range(n)) + "\n", "\t", ".", False))
    out.append(("\n".join(f"2024-03-01T10:{i % 60:02d}:00|{R.choice(['INFO', 'WARN', 'ERROR'])}|{R.choice(['auth', 'db', 'api'])}|{R.choice(['login ok', 'timeout', 'retry'])}|{R.randint(1, 900)}ms" for i in range(n)) + "\n", "|", None, False))
    out.append(("\n".join(" ".join(dec(R.uniform(0, 8), '.', 1) for _ in range(4)) for _ in range(n)) + "\n", " ", ".", False))
    out.append(("\n".join(" ".join(dec(R.uniform(0, 8), ',', 1) for _ in range(4)) for _ in range(n)) + "\n", " ", ",", False))
    out.append(("name,role,salary\n" + "\n".join(f'"{R.choice(["Smith", "Lee", "Khan"])}, {R.choice(["Ann", "Bo", "Cy"])}",{R.choice(["dev", "ops"])},{R.randint(30, 90)}000' for i in range(n)) + "\n", ",", ".", True))
    out.append(("city;country;continent\n" + "\n".join(f"{R.choice(cities)};{R.choice(['DE', 'FR', 'PT', 'UK'])};{R.choice(['Europe', 'Asia'])}" for i in range(n)) + "\n", ";", None, True))
    out.append(("\n".join(",".join(dec(R.uniform(0, 8), '.', 1) for _ in range(4)) + f",{R.randint(0, 2)}" for _ in range(n)) + "\n", ",", ".", False))
    out.append(("when\tuser\tamount\n" + "\n".join(f"2023-0{R.randint(1, 9)}-1{R.randint(0, 9)}\t{R.choice(['u1', 'u2', 'u3'])}\t{dec(R.uniform(1, 500), '.')}" for i in range(n)) + "\n", "\t", ".", True))
    out.append(("a, b, c, d\n" + "\n".join(", ".join(str(R.randint(0, 99)) for _ in range(4)) for _ in range(n)) + "\n", ",", ".", True))
    out.append(("id;region;share;count\n" + "\n".join(f"{i};{R.choice(cities)};{dec(R.uniform(0, 100), ',', 1)}%;{R.randint(1, 90)}" for i in range(n)) + "\n", ";", ",", True))
    out.append(("id|flag|a|b|c|d|e|f\n" + "\n".join(f"{i}|{R.choice(['true', 'false'])}|" + "|".join(str(R.randint(0, 50)) for _ in range(6)) for i in range(n)) + "\n", "|", None, True))
    out.append(("x,y,z\n" + "\n".join(f"{R.gauss(0, 1):.3e},{R.gauss(-5, 2):.3f},{R.randint(-9, 9)}" for i in range(n)) + "\n", ",", ".", True))
    out.append(("\n".join("\t".join(dec(R.uniform(0, 8), ',', 2) for _ in range(5)) for _ in range(n)) + "\n", "\t", ",", False))
    out.append(("k,v,w\n" + "\n".join(f"{i},{'' if R.random() < 0.15 else R.randint(1, 99)},{'' if R.random() < 0.15 else dec(R.random(), '.')}" for i in range(n)) + "\n", ",", ".", True))
    return out


def test_the_ranker_beats_the_rules_on_files_written_independently_of_the_generator_and_never_loses_a_file_the_rules_had_right():
    from zerowhite.omnivore import learned, tables
    files = _ood_files()
    R = learned.get_ranker()
    rules = sum(tables.find_delimiter(t)[0] == d for t, d, _, _ in files)
    ranker = sum(R.rank(t)[0] == d for t, d, _, _ in files)
    assert ranker >= 15 and ranker > rules, (ranker, rules)
    for text, delim, _, _ in files:                                                          # arbitrate end to end: the delimiter the table ends up with
        h = learned.hint(text.encode("utf-8"), min_p=0.9)
        tab = tables.parse_delimited(text, hint=dict(h, arbitrate=True)) if h else tables.parse_delimited(text)
        ruled = tables.find_delimiter(text)[0]
        assert tab is not None and (tab.delimiter == delim or ruled != delim), (delim, tab and tab.delimiter, ruled)


# ============================================================================================================ lazy graph + kernels
def _random_graph(rng, G, depth=4):
    """A random expression over a few leaves with broadcasting, reductions, reshapes and a matmul; domains are kept safe (log / sqrt / div of positive things)."""
    base = (int(rng.integers(2, 6)), int(rng.integers(2, 7)))
    leaves = [G.lazy(rng.normal(size=base)), G.lazy(rng.normal(size=base)), G.lazy(rng.normal(size=(base[1],))), G.lazy(rng.normal(size=(base[0], 1))), G.lazy(rng.normal(size=()))]

    def build(d):
        if d == 0 or rng.random() < 0.15:
            return leaves[int(rng.integers(len(leaves)))]
        k = int(rng.integers(0, 11))
        a, b = build(d - 1), build(d - 1)
        if k == 0:
            return a + b
        if k == 1:
            return a - b
        if k == 2:
            return a * b
        if k == 3:
            return a / (b.abs() + 1.0)
        if k == 4:
            return G.maximum(a, b)
        if k == 5:
            return G.minimum(a, b)
        if k == 6:
            return a.tanh() + b.relu()
        if k == 7:
            return (a.abs() + 1.0).log() * b.exp().sqrt().__neg__().__neg__() if False else (a.abs() + 1.0).log() - (b * b + 1.0).sqrt()
        if k == 8:
            return a.exp().sqrt() if False else (-(a * a)).exp() + b
        if k == 9:
            return a * 0.5 + b * (-2.0)
        return -a + b.abs()
    g = build(depth)
    r = int(rng.integers(0, 4))
    if r == 1:
        g = g.sum(axis=int(rng.integers(0, max(1, g.ndim)))) if g.ndim else g.sum()
    elif r == 2 and g.ndim:
        g = g.max(axis=-1, keepdims=bool(rng.integers(0, 2)))
    elif r == 3:
        g = g.reshape(-1) if hasattr(g, "reshape") else g
        g = g * 2.0 + 1.0
    return g


def test_c_and_emulated_cuda_backends_agree_with_numpy_on_random_graphs_and_actually_fuse():
    from zerowhite.kernels import cjit, cuda, graph as G
    if not cjit.available():
        return                                                                               # no C compiler on this machine: nothing to verify (the CLI says so)

    class Emu:
        def run(self, src, arrs, out):
            return cuda.emulate_cuda(src, arrs, out, grid=3, block=8)
    rng = np.random.default_rng(11)
    fused_total = kernels_total = 0
    for _ in range(60):
        g = _random_graph(rng, G)
        ref = G.evaluate(g, "numpy")
        st = {}
        for be, kw in (("c", dict(stats=st)), ("cuda", dict(cuda=Emu()))):
            got = G.evaluate(g, be, **kw)
            assert got.shape == ref.shape and np.allclose(got, ref, rtol=1e-9, atol=1e-9, equal_nan=True), (be, np.abs(got - ref).max())
        fused_total += st["fused_ops"]
        kernels_total += st["kernels"]
    assert fused_total > 1.5 * kernels_total, (fused_total, kernels_total)                    # several operations per generated loop, on average


def test_fusion_makes_one_loop_for_a_chain_and_keeps_a_shared_value_materialised():
    from zerowhite.kernels import cjit, graph as G
    if not cjit.available():
        return
    r = np.random.default_rng(0)
    x, y = G.lazy(r.normal(size=(8, 5))), G.lazy(r.normal(size=(8, 5)))
    chain = ((x * y + 1.0).relu() - 0.5).abs() * 2.0
    st = {}
    assert np.allclose(G.evaluate(chain, "c", stats=st), G.evaluate(chain, "numpy")) and st["kernels"] == 1 and st["fused_ops"] == 6
    shared = x * y
    two = (shared + 1.0) * (shared - 1.0)                                                        # `shared` has two consumers: it is computed once, not twice
    st = {}
    assert np.allclose(G.evaluate(two, "c", stats=st), G.evaluate(two, "numpy")) and st["kernels"] == 2 and "x" not in st["sources"][0]
    assert sum(src.count("L0 * L1") for src in st["sources"]) == 1
    red = chain.sum(axis=1)                                                                      # a reduction ends the region; the loop before it still fuses
    st = {}
    assert np.allclose(G.evaluate(red, "c", stats=st), G.evaluate(red, "numpy")) and st["kernels"] == 1


def test_the_jit_caches_by_source_and_reports_compile_errors_with_the_compilers_message():
    from zerowhite.kernels import cjit
    if not cjit.available():
        return
    src = "double f(double x) { return x * 2.0 + 1.0; }\n"
    a, b = cjit.build(src), cjit.build(src)
    assert a is b
    try:
        cjit.build("this is not C;\n")
        raise AssertionError("a compile error must raise")
    except RuntimeError as ex:
        assert "C compilation failed" in str(ex) and "error" in str(ex)


class _FakeLib:
    """A stand-in for libcuda / libnvrtc: records every call (name + arguments) and fills output parameters, like the real C API."""
    def __init__(self, fail=None):
        self.calls, self.fail, self.mem = [], fail, {}
        self.next_ptr = 0x1000

    def _rec(self, name, *a):
        self.calls.append((name,) + a)
        return 7 if self.fail == name else 0

    def cuInit(self, flags): return self._rec("cuInit", flags)                                  # noqa: E704
    def cuDeviceGet(self, ref, i): ref._obj.value = 5; return self._rec("cuDeviceGet", i)       # noqa: E704
    def cuCtxCreate_v2(self, ref, flags, dev): ref._obj.value = 99; return self._rec("cuCtxCreate_v2", flags, dev.value)   # noqa: E704
    def cuModuleLoadData(self, ref, image): ref._obj.value = 42; return self._rec("cuModuleLoadData", bytes(image))   # noqa: E704
    def cuModuleGetFunction(self, ref, mod, name): ref._obj.value = 43; return self._rec("cuModuleGetFunction", mod.value, name)   # noqa: E704

    def cuMemAlloc_v2(self, ref, n):
        ref._obj.value = self.next_ptr
        self.next_ptr += 0x1000
        return self._rec("cuMemAlloc_v2", n.value)

    def cuMemFree_v2(self, p): return self._rec("cuMemFree_v2", p.value)                         # noqa: E704
    def cuMemcpyHtoD_v2(self, d, h, n): return self._rec("cuMemcpyHtoD_v2", d.value, n.value)     # noqa: E704
    def cuMemcpyDtoH_v2(self, h, d, n): return self._rec("cuMemcpyDtoH_v2", d.value, n.value)     # noqa: E704

    def cuLaunchKernel(self, fn, gx, gy, gz, bx, by, bz, sh, stream, params, extra):
        import ctypes
        vals = [ctypes.cast(params[i], ctypes.POINTER(ctypes.c_uint64)).contents.value for i in range(3)]
        return self._rec("cuLaunchKernel", fn.value, gx, gy, gz, bx, by, bz, sh, tuple(vals))

    def cuCtxSynchronize(self): return self._rec("cuCtxSynchronize")                              # noqa: E704


def test_the_ctypes_driver_layer_passes_the_right_names_arguments_and_turns_error_codes_into_exceptions():
    from zerowhite.kernels import cuda
    lib = _FakeLib()
    d = cuda.RawDriver(lib)
    d.init(0)
    mod = d.module_load(b"PTX")
    fn = d.get_function(mod, "k")
    p = d.mem_alloc(800)
    a = np.arange(10.0)
    d.htod(p, a)
    d.launch(fn, 4, 256, [p, p, p])
    d.sync()
    d.dtoh(a, p)
    d.mem_free(p)
    names = [c[0] for c in lib.calls]
    assert names == ["cuInit", "cuDeviceGet", "cuCtxCreate_v2", "cuModuleLoadData", "cuModuleGetFunction", "cuMemAlloc_v2", "cuMemcpyHtoD_v2", "cuLaunchKernel",
                     "cuCtxSynchronize", "cuMemcpyDtoH_v2", "cuMemFree_v2"], names
    by = {c[0]: c for c in lib.calls}
    assert by["cuModuleGetFunction"][2] == b"k" and by["cuMemAlloc_v2"][1] == 800 and by["cuMemcpyHtoD_v2"][2] == 80
    assert by["cuLaunchKernel"][2:9] == (4, 1, 1, 256, 1, 1, 0) and by["cuLaunchKernel"][9] == (0x1000, 0x1000, 0x1000)   # grid, block, packed device pointers
    try:
        cuda.RawDriver(_FakeLib(fail="cuInit")).init(0)
        raise AssertionError("a nonzero return code must raise")
    except cuda.CudaError as ex:
        assert "cuInit failed with code 7" in str(ex)
    assert cuda.driver_names("win32") == ["nvcuda.dll"] and cuda.driver_names("linux")[0] == "libcuda.so.1" and cuda.driver_names("darwin") == []
    assert all(n.startswith("nvrtc64_") for n in cuda.nvrtc_names("win32")) and "libnvrtc.so" in cuda.nvrtc_names("linux")
    real = cuda.load_first
    cuda.load_first = lambda names, loader=None: (None, None)
    try:
        try:
            cuda.RawDriver()
            raise AssertionError("no driver: must say so")
        except cuda.CudaError as ex:
            assert "driver library was not found" in str(ex)
    finally:
        cuda.load_first = real


class _FakeDriver:
    """The CudaBackend's view of a driver, executing 'kernels' by emulating the generated CUDA source on the CPU; records the order of operations."""
    def __init__(self, fail_launch=False):
        self.log, self.bufs, self.fail_launch, self.srcs = [], {}, fail_launch, {}

    def init(self, device=0): self.log.append("init")                                              # noqa: E704
    def module_load(self, image): self.log.append("module_load"); self.srcs[1] = image.decode(); return 1   # noqa: E704
    def get_function(self, mod, name): self.log.append("get_function"); return ("fn", mod, name)    # noqa: E704

    def mem_alloc(self, n):
        k = len(self.bufs) + 1
        self.bufs[k] = np.zeros(max(1, n // 8))
        self.log.append(("alloc", n))
        return k

    def mem_free(self, p): self.log.append(("free", p)); self.bufs.pop(p, None)                      # noqa: E704
    def htod(self, p, a): self.bufs[p][:a.size] = a.ravel(); self.log.append(("htod", p))             # noqa: E704

    def launch(self, fn, grid, block, ptrs, n=None):
        from zerowhite.kernels import cuda
        self.log.append(("launch", grid, block))
        if self.fail_launch:
            raise cuda.CudaError("cuLaunchKernel failed with code 700")
        ins = [self.bufs[p][:n] if False else self.bufs[p] for p in ptrs[:-1]]
        out = self.bufs[ptrs[-1]]
        cuda.emulate_cuda(self.srcs[1], ins, out, grid=grid, block=block)

    def sync(self): self.log.append("sync")                                                            # noqa: E704
    def dtoh(self, a, p): a.ravel()[:] = self.bufs[p][:a.size]; self.log.append(("dtoh", p))           # noqa: E704


class _FakeNvrtc:
    def compile(self, src, name="k.cu", options=(), cubin=False):
        assert '__global__' in src and "gridDim.x" in src
        return src.encode()


def test_the_cuda_backend_runs_the_sequence_in_order_gets_numpy_numbers_and_frees_device_memory_even_on_failure():
    from zerowhite.kernels import cjit, cuda, graph as G
    if not cjit.available():
        return
    r = np.random.default_rng(2)
    a, b = r.normal(size=(5, 13)), r.normal(size=(13,))
    g = ((G.lazy(a) * G.lazy(b) + 1.0).relu() - 0.5).abs()
    drv = _FakeDriver()
    be = cuda.CudaBackend(driver=drv, nvrtc=_FakeNvrtc())
    got = G.evaluate(g, "cuda", cuda=be)
    assert np.allclose(got, G.evaluate(g, "numpy"), atol=1e-12)
    kinds = [x if isinstance(x, str) else x[0] for x in drv.log]
    assert kinds[:3] == ["init", "module_load", "get_function"] and kinds.index("launch") < kinds.index("sync") < kinds.index("dtoh")
    assert kinds.count("alloc") == 5 and kinds.count("free") == 5 and not drv.bufs               # four leaves (a, b and the two scalar constants) + the output, all released
    launch = [x for x in drv.log if isinstance(x, tuple) and x[0] == "launch"][0]
    assert launch[1] == 1 and launch[2] == 256                                                   # 65 elements: one block of 256
    G.evaluate(g, "cuda", cuda=be)
    assert [x for x in drv.log if x == "module_load"] == ["module_load"]                         # the same source is compiled once
    bad = _FakeDriver(fail_launch=True)
    try:
        G.evaluate(g, "cuda", cuda=cuda.CudaBackend(driver=bad, nvrtc=_FakeNvrtc()))
        raise AssertionError("a failed launch must raise")
    except cuda.CudaError:
        pass
    assert not bad.bufs                                                                           # ... and still frees what it allocated


def test_the_emulation_covers_sizes_that_are_not_a_multiple_of_the_launch_and_smaller_than_it():
    from zerowhite.kernels import cjit, cuda, graph as G
    if not cjit.available():
        return
    for n in (1, 7, 24, 25, 100):
        x = np.linspace(-2, 2, n)
        st = {}
        g = (G.lazy(x) * 3.0 + 1.0).tanh()
        class Emu:
            def run(self, src, arrs, out):
                return cuda.emulate_cuda(src, arrs, out, grid=3, block=8)                       # 24 threads, grid-stride for the rest
        assert np.allclose(G.evaluate(g, "cuda", cuda=Emu(), stats=st), np.tanh(x * 3.0 + 1.0), atol=1e-14), n


def test_the_kernels_command_runs_and_says_plainly_what_this_machine_can_do():
    r = subprocess.run([sys.executable, "-m", "zerowhite", "kernels", "--size", "200000"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "numpy" in r.stdout and "CUDA:" in r.stdout, r.stdout + r.stderr


# ================================================================================================================ audio codec
_CODEC_CACHE = {}


def _speech(texts_voices):
    from zerowhite.voice.formant import synthesize
    return [synthesize(t, f0_hz=float(f0), rate=float(r)) for t, f0, r in texts_voices]


def _small_codec():
    if "c" not in _CODEC_CACHE:
        from zerowhite.voice import codec as C
        S = ["the quick brown fox jumps over the lazy dog", "hello world how are you today", "she sells sea shells by the sea shore", "please call me back tomorrow morning"]
        train = _speech([(s, f0, r) for s in S for f0, r in [(105, 1.0), (125, 0.9), (150, 1.1), (180, 1.0)]])
        _CODEC_CACHE["c"] = (C.train(train, stages=4, codewords=64, hidden=128, latent=32, epochs=60, finetune_epochs=10, seed=0), train)      # ~10 s
    return _CODEC_CACHE["c"]


def test_the_codec_frames_reconstruct_exactly_and_tokens_have_the_documented_shape_and_range():
    from zerowhite.voice import codec as C
    x = np.random.default_rng(0).normal(size=5000)
    fr, n = C.frames_of(x)
    assert np.allclose(C.overlap_add(fr, n), x, atol=1e-12)                       # analysis + synthesis windows give perfect reconstruction without any coding
    assert np.allclose(C.WIN[:C.HOP] ** 2 + C.WIN[C.HOP:] ** 2, 1.0)
    cd, train = _small_codec()
    w = _speech([("good morning everyone", 120, 1.0)])[0]
    toks = cd.encode(w)
    assert toks.dtype == np.int32 and toks.shape == (1 + len(w) // C.HOP + 1, 4) or toks.shape[1] == 4
    assert toks.min() >= 0 and toks.max() < cd.codewords and abs(cd.bitrate() - 4 * 6 * 100) < 1e-9
    y = cd.decode(toks, len(w))
    assert y.shape == w.shape and np.isfinite(y).all()
    assert np.array_equal(cd.encode(w), toks)                                       # deterministic
    assert np.array_equal(C.text_to_tokens(C.tokens_to_text(toks)), toks) and C.tokens_to_text(toks[:2]).startswith("a")


def test_more_stages_always_mean_a_better_signal_and_any_prefix_of_the_code_is_decodable():
    from zerowhite.voice import codec as C
    cd, train = _small_codec()
    held = _speech([("the weather is nice", 130, 1.0), ("thank you very much", 160, 1.0)])
    snr = {s: np.mean([C.snr_db(w, cd.roundtrip(w, s)) for w in held]) for s in (1, 2, 3, 4)}
    assert snr[1] < snr[2] < snr[3] < snr[4], snr                                    # residual quantisation: each stage removes some of what was left
    assert snr[4] > 1.0 and cd.bitrate(2) == cd.bitrate() / 2
    toks = cd.encode(held[0])
    full, coarse = cd.decode(toks, len(held[0])), cd.decode(toks[:, :2], len(held[0]))        # decode a 2-stage prefix of a 4-stage code
    assert C.snr_db(held[0], coarse) < C.snr_db(held[0], full) and np.isfinite(coarse).all()
    assert np.array_equal(cd.encode(held[0], stages=2), toks[:, :2])                  # ... which is exactly what encoding with 2 stages gives


def test_a_codec_survives_save_and_load_and_the_cli_trains_encodes_and_decodes():
    from zerowhite.voice import codec as C, audio
    cd, _ = _small_codec()
    d = tmp()
    cd.save(os.path.join(d, "c.npz"))
    cd2 = C.Codec.load(os.path.join(d, "c.npz"))
    w = _speech([("hello world how are you today", 150, 1.0)])[0]
    assert np.abs(cd.roundtrip(w) - cd2.roundtrip(w)).max() < 1e-3                  # float32 on disk: tiny differences, same codes in practice
    assert (cd.encode(w) == cd2.encode(w)).mean() > 0.99
    audio.save_audio(os.path.join(d, "in.wav"), w[:16000], 16000)
    env = dict(os.environ, ZERO_WHITE_DATA=os.environ["ZERO_WHITE_DATA"])
    run = lambda *args: subprocess.run([sys.executable, "-m", "zerowhite", "codec", *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)    # noqa: E731
    r = run("encode", os.path.join(d, "in.wav"), "--model", os.path.join(d, "c.npz"), "--out", os.path.join(d, "t.txt"))
    assert r.returncode == 0 and "kbps" in r.stdout, r.stdout + r.stderr
    r = run("decode", os.path.join(d, "t.txt"), "--model", os.path.join(d, "c.npz"), "--out", os.path.join(d, "out.wav"))
    y, sr = audio.load_audio(os.path.join(d, "out.wav"), 16000)
    assert r.returncode == 0 and sr == 16000 and abs(len(y) - 16000) <= C.HOP * 2


def test_a_properly_trained_codec_beats_pca_at_the_same_bitrate():                       # slow (about a minute): ZW_SLOW_TESTS=1; measured numbers are in docs/VOICE.md
    if os.environ.get("ZW_SLOW_TESTS") != "1":
        return
    from zerowhite.voice import codec as C
    S = ["the quick brown fox jumps over the lazy dog", "hello world how are you today", "she sells sea shells by the sea shore", "please call me back tomorrow morning",
         "a stitch in time saves nine", "we hold these truths to be self evident", "open the door and turn on the light", "numbers one two three four five six seven"]
    train = _speech([(s, f0, r) for s in S for f0, r in [(105, 1.0), (125, 0.9), (150, 1.1), (180, 1.0)]])
    held = _speech([("good morning everyone", 112, 1.0), ("the weather is nice", 140, 1.05), ("thank you very much", 170, 0.95)])
    cd = C.train(train, stages=6, codewords=128, hidden=256, latent=48, epochs=100, finetune_epochs=15)
    pca = C.pca_baseline(train, 6, 128, 48)
    for s, margin in ((1, 0.3), (3, 0.7), (6, 0.7)):                                  # measured: +0.5, +1.3, +1.3 dB (docs/VOICE.md)
        assert np.mean([C.snr_db(w, cd.roundtrip(w, s)) for w in held]) > np.mean([C.snr_db(w, pca.roundtrip(w, s)) for w in held]) + margin, s


# ====================================================================================================== mixture of experts
def _moe_model(experts=4, top_k=2, seed=0, aux=0.01, dtype="float64"):
    from zerowhite.chat.model import GPT, GPTConfig
    return GPT(GPTConfig(vocab=23, d_model=16, n_layer=2, n_head=2, ffn=24, ctx=16, dtype=dtype, n_experts=experts, top_k=top_k, moe_aux=aux), seed=seed, init_std=0.3)


def test_moe_gradients_match_finite_differences_for_router_experts_and_attention_including_the_balancing_loss():
    import test_v63 as T63
    for top_k, aux in ((2, 0.0), (2, 0.5), (1, 0.5), (4, 0.5)):                          # top-1, top-2 and 'all experts' routing, with and without the aux loss
        m = _moe_model(experts=4, top_k=top_k, aux=aux)
        b = T63._gbatch()
        loss, grads, _ = m.loss_and_grads(b)
        rng = np.random.default_rng(3)
        checked = 0
        for name in ["b0.rt", "b1.rt", "b0.e0.wg", "b0.e2.wd", "b1.e3.wu", "b1.e1.wd", "b0.wq", "b1.wo", "tok_emb"]:
            P = m.p[name]
            for _ in range(5):
                ix = tuple(int(rng.integers(s)) for s in P.shape)
                old = P[ix]
                eps = 1e-6
                P[ix] = old + eps
                lp = m.loss_and_grads(b)[0]
                P[ix] = old - eps
                lm = m.loss_and_grads(b)[0]
                P[ix] = old
                num = (lp - lm) / (2 * eps)
                if abs(num) > 1e-7 or abs(grads[name][ix]) > 1e-7:
                    checked += 1
                    assert abs(num - grads[name][ix]) <= 1e-5 * max(1.0, abs(num)), (top_k, aux, name, ix, num, grads[name][ix])
        assert checked >= 20, checked                                                    # the check had real, non-zero gradients to look at


def test_a_moe_model_generates_with_the_kv_cache_exactly_as_the_training_forward_computes():
    from zerowhite.chat.generate import KVCache, forward_cached
    m = _moe_model(experts=4, top_k=2, aux=0.0)
    ids = np.random.default_rng(5).integers(1, 23, 12)
    full = m.token_logprobs(dict(ids=ids[None, :-1], tgt=ids[None, 1:], w=np.ones((1, 11), np.float32), pos=np.arange(11)[None], seg=np.ones((1, 11), np.int64)))[0]
    cache = KVCache(m.cfg)
    got = []
    for t in range(11):
        lg = forward_cached(m, cache, ids[t:t + 1])
        lg = lg - lg.max()
        got.append(lg[ids[t + 1]] - np.log(np.exp(lg).sum()))
    assert np.allclose(got, full, atol=1e-9)


def test_to_moe_preserves_the_function_exactly_then_experts_diverge_and_the_router_balances():
    import test_v63 as T63
    from zerowhite.chat.grow import grow, to_moe
    from zerowhite.chat.optim import AdamW
    m = T63._gpt(dtype="float64")
    b = T63._gbatch()
    base = T63._logits(m, b)
    for experts, k in ((2, 1), (4, 2), (3, 3)):
        mo = to_moe(m, experts, top_k=k, seed=1)
        assert np.max(np.abs(T63._logits(mo, b) - base)) < 1e-10, (experts, k)          # identical experts: any gates summing to one give the dense output
        assert mo.cfg.n_experts == experts and mo.num_params() > m.num_params()
    mo = to_moe(m, 4, top_k=2, seed=1)
    assert abs(mo.eval_nll(b)[0] - m.eval_nll(b)[0]) < 1e-9 if isinstance(mo.eval_nll(b), tuple) else True
    for bad in (dict(experts=1), dict(experts=4, top_k=5), dict(experts=4, top_k=0)):
        try:
            to_moe(m, **bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    try:
        to_moe(mo, 4)
        raise AssertionError("already MoE")
    except ValueError:
        pass
    try:
        grow(mo, layers=1)
        raise AssertionError("dense-only growth must refuse a MoE model")
    except ValueError as ex:
        assert "dense" in str(ex)
    # training: the experts move apart (they were identical) and the model learns
    m32 = T63._gpt(dtype="float32")
    mo = to_moe(m32, 4, top_k=2, seed=1)
    opt = AdamW(mo.p, 0.9, 0.99, weight_decay=0.0)
    loss0 = mo.loss_and_grads(b)[0]
    for _ in range(60):
        loss, grads, info = mo.loss_and_grads(b)
        opt.step(mo.p, {k: v / max(1, info["n"]) for k, v in grads.items()}, 0.01, 1.0)
    assert mo.loss_and_grads(b)[0] < 0.6 * loss0
    e0, e1 = mo.p["b0.e0.wd"], mo.p["b0.e1.wd"]
    assert np.abs(e0 - e1).max() > 1e-3                                                  # experts that started identical are no longer


def test_the_balancing_loss_spreads_tokens_over_experts_instead_of_leaving_one_expert_busiest():
    """Measured (4 seeds, top-1 of 4 experts, router started biased to expert 0): busiest expert's share 0.57 without the loss, 0.28 with it (even would be 0.25)."""
    import test_v63 as T63
    from zerowhite.chat.grow import to_moe
    from zerowhite.chat.optim import AdamW
    share = {}
    for aux in (0.0, 0.5):
        vals = []
        for seed in range(3):
            m = to_moe(T63._gpt(dtype="float32"), 4, top_k=1, aux=aux, seed=seed)
            m.p["b0.rt"][:] = 0.0
            m.p["b0.rt"][:, 0] = 0.3                                                     # a router that starts out preferring expert 0
            b = T63._gbatch(B=6, T=12, seed=seed + 1)
            opt = AdamW(m.p, 0.9, 0.99, weight_decay=0.0)
            for _ in range(80):
                loss, grads, info = m.loss_and_grads(b)
                opt.step(m.p, {k: v / max(1, info["n"]) for k, v in grads.items()}, 0.02, 1.0)
            idx = m._forward(b, keep=True)[1]["layers"][0]["moe"]["idx"]
            vals.append(np.bincount(idx.ravel(), minlength=4).max() / idx.size)
        share[aux] = float(np.mean(vals))
    assert share[0.0] > 0.5 and share[0.5] < 0.4 and share[0.0] - share[0.5] > 0.15, share


def test_moe_checkpoints_round_trip_older_dense_checkpoints_still_load_and_export_refuses_moe():
    import test_v63 as T63
    from zerowhite.chat.grow import to_moe
    from zerowhite.chat.model import GPT, read_meta
    from zerowhite.chat import export
    d = tmp()
    mo = to_moe(T63._gpt(dtype="float32"), 3, top_k=2, seed=0)
    path = os.path.join(d, "m.npz")
    mo.save(path, meta={})
    m2, _ = GPT.load(path)
    assert m2.cfg.n_experts == 3 and m2.cfg.top_k == 2 and all(np.array_equal(mo.p[k], m2.p[k]) for k in mo.p)
    dense = T63._gpt(dtype="float32")
    dense.save(os.path.join(d, "d.npz"), meta={})
    cfg = read_meta(os.path.join(d, "d.npz"))["config"]
    for k in ("n_experts", "top_k", "moe_aux"):
        cfg.pop(k, None)                                                                 # a 6.3 checkpoint has none of these fields
    from zerowhite.chat.model import GPTConfig
    assert GPTConfig.from_dict(cfg).n_experts == 1
    try:
        export.to_llama_tensors(mo)
        raise AssertionError("must refuse")
    except ValueError as ex:
        assert "mixture-of-experts" in str(ex)
    assert export.to_llama_tensors(dense)                                                # the dense path is untouched


def test_a_workspace_becomes_a_moe_model_and_the_real_trainer_continues_training_it():
    import test_chat as TC
    from zerowhite.chat import grow as G, prepare as PR, train as TR
    from zerowhite.chat.model import GPT, read_meta
    cfg, lay = TC.copy_workspace()
    old = read_meta(lay.current_weights())["config"]
    out = G.grow_current(lay, experts=3, top_k=2)
    new = read_meta(lay.current_weights())["config"]
    assert new["n_experts"] == 3 and new["top_k"] == 2 and "experts" in out["summary"] or "->" in out["summary"]
    assert old.get("n_experts", 1) == 1 and lay.current()["grown_ops"]["experts"] == 3
    P = PR.prepare(cfg, lay, do_bench=False)
    assert P.model_cfg is not None and P.model_cfg.n_experts == 3
    res = TR.Trainer(P).run()
    model, _ = GPT.load(lay.current_weights())
    assert model.cfg.n_experts == 3 and res["steps"] > 0 and all(np.isfinite(v).all() for v in model.p.values())


# ================================================================================================= attention encoder (Perceiver-style)
def _blobs(n, seed, size=12, sigma=1.3):
    r = np.random.default_rng(seed)
    xy = r.uniform(1.5, size - 2.5, size=(n, 2))
    g = np.arange(size)
    img = np.exp(-(((g[None, :, None] - xy[:, 0, None, None]) ** 2 + (g[None, None, :] - xy[:, 1, None, None]) ** 2) / (2 * sigma ** 2)))
    return img.reshape(n, -1) + 0.01 * r.normal(size=(n, size * size)), xy


def test_the_encoders_hand_written_gradients_match_finite_differences():
    from zerowhite.forge.latent import LatentEncoder as LE
    rng = np.random.default_rng(0)
    p = LE.init_params(10, 4, 2, 3, 8, rng)
    X = rng.normal(size=(5, 10))
    loss, g = LE.loss_grads(p, X, 4)
    checked = 0
    for n in p:
        for _ in range(6):
            ix = tuple(int(rng.integers(s)) for s in p[n].shape)
            old, e = p[n][ix], 1e-6
            p[n][ix] = old + e
            lp = LE.loss_grads(p, X, 4)[0]
            p[n][ix] = old - e
            lm = LE.loss_grads(p, X, 4)[0]
            p[n][ix] = old
            num = (lp - lm) / (2 * e)
            if abs(num) > 1e-9 or abs(g[n][ix]) > 1e-9:
                checked += 1
                assert abs(num - g[n][ix]) <= 1e-5 * max(1e-3, abs(num), abs(g[n][ix])), (n, ix, num, g[n][ix])
    assert checked >= 40                                                                 # every parameter group was actually exercised (token embeddings, attention, code, decoder)


def test_the_encoder_describes_a_curved_set_far_better_than_pca_with_the_same_number_of_codes():
    """Measured with 1500 training pictures: test MSE 0.067 vs 0.70 (k=2), 0.027 vs 0.25 (k=8). Here a smaller fit (800 pictures, 70 epochs, ~10 s): 0.086 vs 0.58 at k=3."""
    from zerowhite.forge.latent import LatentEncoder, pca_recon_mse
    Xtr, _ = _blobs(800, 0)
    Xte, _ = _blobs(300, 9)
    enc = LatentEncoder.fit(Xtr, k=3, epochs=70, seed=0)
    assert enc.recon_mse(Xte) < 0.3 * pca_recon_mse(np.vstack([Xtr, Xte]), 3), (enc.recon_mse(Xte), pca_recon_mse(np.vstack([Xtr, Xte]), 3))
    z = np.array([enc(x) for x in Xte[:100]])
    assert z.shape == (100, 3) and np.isfinite(z).all() and np.abs(z.mean(0)).max() < 0.5 and 0.5 < z.std(0).mean() < 1.5       # standardised codes for the programs
    enc2 = LatentEncoder.from_dict(json.loads(json.dumps(enc.to_dict())))
    assert np.allclose(enc2(Xte[0]), enc(Xte[0]))
    try:
        LatentEncoder.fit(Xtr[:5], k=2)
        raise AssertionError("too few observations must be refused")
    except ValueError:
        pass


def test_a_program_agent_can_use_the_encoder_and_a_reloaded_agent_sees_the_same_code_and_pca_stays_the_default():
    from zerowhite.forge import families as F, task as T
    from zerowhite.forge.latent import LatentEncoder
    from zerowhite.forge.reduce import Projector

    class Blob:
        obs_dim, n_actions, max_steps, max_return, name = 144, 2, 10, 10.0, "blob"

        def reset(self, rng):
            self.rng, self.t = rng, 0
            return self._n()

        def _n(self):
            self.x = self.rng.uniform(1.5, 9.5)
            g = np.arange(12)
            return (np.exp(-(((g[:, None] - self.x) ** 2 + (g[None, :] - 5.5) ** 2) / (2 * 1.3 ** 2))) + 0.01 * self.rng.normal(size=(12, 12))).ravel()

        def step(self, a):
            self.t += 1
            return self._n(), float(int(a) == int(self.x > 5.5)), self.t >= 10, False
    genome = dict(n_islands=2, island_size=6, tune_steps=4, max_nodes=50, n_mut=2)
    assert F.ProgramPolicyAgent.encoder == "pca" and isinstance(F.ProgramPolicyAgent(T.EnvTask(Blob()), genome, 0).proj, Projector)
    F.ProgramPolicyAgent.encoder = "latent"
    real_fit = LatentEncoder.fit
    LatentEncoder.fit = classmethod(lambda cls, obs, **kw: real_fit.__func__(cls, obs, **dict(kw, epochs=6)))         # this test is about plumbing, not quality
    try:
        task = T.EnvTask(Blob())
        a = F.ProgramPolicyAgent(task, genome, 0)
        assert isinstance(a.proj, LatentEncoder) and a.vocab.n_cols == 8
        a.train(task, 0.5, np.random.default_rng(0))
        d = tmp()
        a.save(d)
        st = json.loads(fileio.read_text(os.path.join(d, "state.json")))
        assert "latent" in st and "projector" not in st and "attention-encoder" in fileio.read_text(os.path.join(d, "program.py"))
        b = F.ProgramPolicyFamily().load(d, task, genome)
        xs = [Blob().reset(np.random.default_rng(i)) for i in range(10)]
        assert np.allclose(b.proj(xs[0]), a.proj(xs[0])) and [a._act_fn()(x) for x in xs] == [b._act_fn()(x) for x in xs]
    finally:
        F.ProgramPolicyAgent.encoder = "pca"
        LatentEncoder.fit = real_fit


# ================================================================================================================ sleep
def _sleep_setup(**over):
    import test_chat as TC
    from zerowhite.chat import prepare as PR
    from zerowhite.chat.generate import Engine
    from zerowhite.chat.session import ChatService
    cfg, lay = TC.copy_workspace()
    P = PR.prepare(cfg.replace(**over), lay, do_bench=False)
    eng = Engine(P.cfg, lay)
    assert eng.load()
    return cfg, P, lay, eng, ChatService(P.cfg, lay, eng)


def _teach(svc, text="purple orange", q="Say purple then orange."):
    for _ in range(3):
        r = svc.reply(None, q, temperature=0.0)
        svc.feedback(r["reply_id"], rating=-1, correction=text)


def test_the_consolidation_gate_watches_the_worst_age_band_where_the_mean_would_have_let_forgetting_through():
    from zerowhite.chat.online import OnlineLearner
    from zerowhite.chat.sleep import BANDS, ConsolidationLearner
    cfg, P, lay, eng, svc = _sleep_setup(**{"online.min_events": 2, "online.steps_per_update": 4, "online.lr": 0.003, "online.max_regression": 0.05})
    _teach(svc)
    cl = ConsolidationLearner(P.cfg, lay, eng, P.corpus)
    bands = cl._probe_batches(eng.model.cfg.ctx, eng.tpl.pad_id)
    assert len(bands) == BANDS and all(len(b["ids"]) >= 1 for b in bands)                      # one batch per age band of the corpus
    sched = {"calls": 0}
    real = ConsolidationLearner.band_losses

    def fake(model, batches):
        if len(batches) == 1:
            return [1.0]                                                                      # the fit batch
        sched["calls"] += 1
        return [1.0, 1.0, 1.0, 1.0] if sched["calls"] == 1 else [0.4, 0.4, 0.4, 1.3]          # after: three bands much better, the oldest 30% worse
    ConsolidationLearner.band_losses = staticmethod(fake)
    try:
        assert cl._nll(eng.model, bands) == 1.0 and abs(np.mean([0.4, 0.4, 0.4, 1.3]) - 0.625) < 1e-9    # worst band, not mean
        before_file = lay.current()["file"]
        sched["calls"] = 0                                                                    # the direct call above must not count: update() makes its own two
        out = cl.update(force=True)
    finally:
        ConsolidationLearner.band_losses = real
    assert out["updated"] is False and "regression gate" in out["reason"] and lay.current()["file"] == before_file, out
    assert out["probe_after"] == 1.3 and out["probe_before"] == 1.0                           # the numbers the gate compared: worst bands
    assert 0.625 < 1.0 * 1.05 and 1.3 > 1.0 * 1.05                                            # ... a mean-based gate (0.625 <= 1.05) would have accepted this model


def test_a_consolidation_pass_learns_the_feedback_and_reports_loss_per_band_and_a_too_aggressive_one_is_refused():
    from zerowhite.chat import sleep
    over = {"online.min_events": 2, "online.steps_per_update": 15, "online.lr": 0.003, "online.max_regression": 5.0}
    cfg, P, lay, eng, svc = _sleep_setup(**over)
    _teach(svc)
    before_file = lay.current()["file"]
    out = sleep.consolidate(P.cfg, lay, steps_mult=2.0, lr_scale=1.0, replay_mult=1.0, regression_scale=1.0)
    assert out["updated"] is True and lay.current()["file"] != before_file, out
    assert len(out["bands_before"]) == len(out["bands_after"]) == sleep.BANDS and out["fit_after"] < out["fit_before"]
    assert all(np.isfinite(out["bands_after"])) and out["forgotten"] is False
    nothing = sleep.consolidate(P.cfg, lay)                                                   # the same feedback is not learned twice
    assert nothing["updated"] is False and "nothing to learn" in nothing["reason"]
    cfg2, P2, lay2, eng2, svc2 = _sleep_setup(**{**over, "online.max_regression": 0.02})
    _teach(svc2, "totally different answer")
    keep = lay2.current()["file"]
    with np.errstate(all="ignore"):
        refused = sleep.consolidate(P2.cfg, lay2, steps_mult=3.0, lr_scale=400.0, regression_scale=1.0)   # a learning rate of ~1.2: the model is wrecked, the gate must see it
    assert refused["updated"] is False and lay2.current()["file"] == keep and "regression gate" in refused["reason"], refused
    assert refused["forgotten"] is False                                                               # the live model was left untouched, so the bands afterwards are the old ones
    assert refused["bands_after"] == refused["bands_before"]


def test_the_organism_sleeps_only_when_asked_when_everything_has_plateaued_and_not_more_often_than_its_cooldown():
    import test_v63 as T63
    from zerowhite import organism as O
    chat_dir = tmp()
    def run(sleep_on):
        org, cls = T63._scripted_organism({"t": [(0.5, 0.2)] * 40}, sleep=sleep_on, sleep_cooldown_s=3600.0, chat=chat_dir, patience=2, min_gain=0.2, max_budget=2.0,
                                          chat_set=[])
        calls = []
        cls._act_sleep = lambda self, dec, t0: (calls.append(dec) or self.state.__setitem__("last_sleep", time.time()) or dict(ok=True, seconds=0.0))
        acts = []
        for _ in range(14):
            acts.append(org.step()["decision"]["action"])
        return acts, calls
    acts, calls = run(True)
    assert "sleep" in acts and len(calls) == 1 and "plateaued" in calls[0]["reason"], (acts, calls)     # once: the cooldown stops a second pass
    assert acts.index("sleep") > acts.index("solve")                                                    # after real work, not instead of it
    acts, calls = run(False)
    assert "sleep" not in acts and not calls                                                            # off by default


# ================================================================================================================ curiosity foraging
def test_novelty_is_zero_for_text_already_known_high_for_new_text_and_falls_as_the_system_reads_it():
    from zerowhite.forage import curiosity as C
    known = "the committee met on tuesday to discuss the quarterly report and the budget for next year"
    nov = C.Novelty()
    nov.add(known)
    assert nov.score(known) == 0.0 and nov.score("") == 0.0
    new = "mitochondria produce adenosine triphosphate through oxidative phosphorylation across the inner membrane"
    assert nov.score(new) == 1.0
    mixed = known + " " + new
    assert 0.3 < nov.score(mixed) < 0.7
    nov.add(new)
    assert nov.score(new) == 0.0 and nov.score(mixed) < 0.15                                      # once read, it is no longer a gap (only the join between the two texts is new)


def test_a_curious_pass_keeps_new_pages_drops_known_ones_and_never_reads_outside_the_allow_list():
    import test_v63 as T63
    from zerowhite.forage import curiosity as C
    new_page = "<html><body><p>" + ("Mitochondria produce adenosine triphosphate through oxidative phosphorylation across the inner membrane of the organelle. " * 3) + "</p></body></html>"
    known = "The committee met on Tuesday to discuss the quarterly report and the budget for next year, and nobody objected to the plan. " * 3
    pages = {"https://docs.example.org/new": (200, "text/html", new_page, None), "https://docs.example.org/old": (200, "text/plain", known, None),
             "https://docs.example.org/old2": (200, "text/plain", known + " Minutes were not taken.", None)}
    f, seen, _, d = T63._forager(pages, respect_robots=False, delay=0.0)
    nov = C.Novelty()
    nov.add(known)
    final = os.path.join(d, "kept")
    rows = C.curious_pass(f, list(pages) + ["https://evil.example.com/x"], nov, final, threshold=0.4)
    by = {r["url"].rsplit("/", 1)[-1]: r for r in rows}
    assert by["new"]["kept"] and by["new"]["novelty"] > 0.9 and os.path.exists(by["new"]["path"]) and by["new"]["path"].startswith(final)
    assert not by["old"]["kept"] and by["old"]["novelty"] == 0.0 and "dropped" in by["old"]["reason"]
    assert not by["old2"]["kept"] and by["old2"]["novelty"] < 0.4
    assert not by["x"]["kept"] and "allow-list" in by["x"]["reason"] and not any("evil" in u for u in seen)
    assert sorted(os.listdir(final)) == [os.path.basename(by["new"]["path"])] and not os.listdir(os.path.join(d, "inbox"))     # nothing left in staging
    again = C.curious_pass(f, ["https://docs.example.org/new"], nov, final, threshold=0.4)       # the page just kept is now known: not kept twice
    assert not again[0]["kept"]


def test_the_wishlist_is_read_in_order_and_resumes_where_the_last_pass_stopped():
    from zerowhite.forage import curiosity as C
    d = tmp()
    wl = os.path.join(d, "wish.txt")
    fileio.write_text(wl, "# my reading list\nhttps://a.example/1\thistory\nhttps://a.example/2\n\nhttps://a.example/3\nhttps://a.example/4\n")
    st = os.path.join(d, "state.json")
    first, pos = C.read_wishlist(wl, st, 2)
    assert first == ["https://a.example/1", "https://a.example/2"] and pos == 0
    fileio.write_json(st, dict(pos=pos + len(first)))
    second, pos = C.read_wishlist(wl, st, 5)
    assert second == ["https://a.example/3", "https://a.example/4"] and pos == 2
    assert C.read_wishlist(os.path.join(d, "missing.txt"), st, 3) == ([], 0)


def test_the_organism_reads_only_when_opted_in_with_hosts_and_a_wishlist_and_every_task_has_plateaued():
    import test_v63 as T63
    d = tmp()
    wl = os.path.join(d, "wish.txt")
    fileio.write_text(wl, "https://docs.example.org/a\n")

    def run(**kw):
        org, cls = T63._scripted_organism({"t": [(0.5, 0.2)] * 40}, patience=2, min_gain=0.2, max_budget=2.0, forage_cooldown_s=3600.0, **kw)
        calls = []
        cls._act_forage = lambda self, dec, t0: (calls.append(dec) or self.state.__setitem__("last_forage", time.time()) or dict(ok=True, seconds=0.0))
        acts = [org.step()["decision"]["action"] for _ in range(14)]
        return acts, calls
    acts, calls = run(forage_hosts=("docs.example.org",), forage_wishlist=wl)
    assert "forage" in acts and len(calls) == 1 and acts.index("forage") > acts.index("solve") and "plateaued" in calls[0]["reason"], (acts, calls)
    for kw in (dict(), dict(forage_hosts=("docs.example.org",)), dict(forage_wishlist=wl), dict(forage_hosts=("docs.example.org",), forage_wishlist=os.path.join(d, "nope.txt"))):
        acts, calls = run(**kw)
        assert "forage" not in acts and not calls, kw                                              # any one of the opt-ins missing: never


def test_the_organisms_real_forage_action_reads_a_local_server_keeps_the_novel_page_and_resumes_the_wishlist():
    from zerowhite import organism as O
    import test_v63 as T63
    novel = "<html><body><p>" + ("Tardigrades survive desiccation by producing disordered proteins that vitrify the cytoplasm and protect their membranes. " * 4) + "</p></body></html>"
    known = ("The committee met on Tuesday to discuss the quarterly report and the budget for next year, and nobody objected to the plan. " * 4)
    srv, H = T63._serve({"/novel": (200, "text/html; charset=utf-8", novel, None), "/known": (200, "text/plain", known, None), "/third": (200, "text/plain", "x" * 300, None)})
    try:
        port = srv.server_address[1]
        ws, d = tmp(), tmp()
        fileio.write_text(os.path.join(ws, "forage", "inbox", "seen.txt"), known)               # what the system already holds
        wl = os.path.join(d, "wish.txt")
        fileio.write_text(wl, f"http://127.0.0.1:{port}/novel\nhttp://127.0.0.1:{port}/known\nhttp://127.0.0.1:{port}/third\n")
        org = O.Organism(ws, O.Settings(inbox=[], quiet=True, forage_hosts=("127.0.0.1",), forage_wishlist=wl, forage_max_urls=2, forage_allow_local=True, doctor_every=0))
        res = org._act_forage(dict(action="forage"), time.time())
        assert res["ok"] and res["forage"]["urls"] == 2 and res["forage"]["kept"] == 1, res
        kept = os.listdir(os.path.join(ws, "forage", "inbox"))
        assert len(kept) == 2 and any(open(os.path.join(ws, "forage", "inbox", f), encoding="utf-8").read().startswith("Tardigrades") for f in kept)
        assert "/robots.txt" in H.hits and "/third" not in H.hits                               # robots first; the wishlist was read only as far as max_urls
        res2 = org._act_forage(dict(action="forage"), time.time())
        assert res2["forage"]["urls"] == 1 and "/third" in H.hits                               # the next pass resumes with the remaining URL
        assert org._act_forage(dict(action="forage"), time.time())["forage"]["note"] == "the wishlist is exhausted"
        ledger = [json.loads(x) for x in open(os.path.join(ws, "forage", "ledger.jsonl"), encoding="utf-8")]
        assert len(ledger) >= 3 and all("url" in x for x in ledger)
    finally:
        srv.shutdown()


# ============================================================================================ one stream, shared memory, system 2
def test_tables_audio_and_episodes_round_trip_through_plain_text_and_each_line_says_what_it_is():
    from zerowhite.chat import modal as M
    cols, rows = ["region", "units", "price"], [["north", 120, 9.5], ["south | east", 7, 1234.5678912], ["west", -3, 1e-7]]
    text = M.table_to_text("sales 2024", cols, rows)
    name, c2, r2 = M.text_to_table(text)
    assert name == "sales_2024" and c2 == cols and r2[0] == ["north", 120, 9.5] and r2[1][0] == "south / east" and abs(r2[1][2] - 1234.57) < 1e-2 and abs(r2[2][2] - 1e-7) < 1e-12
    assert [M.modality_of(ln) for ln in text.splitlines()] == ["table", "row", "row", "row"] and M.modality_of("hello world") == "text"
    for bad in ("hello", "[row] 1 | 2"):
        try:
            M.text_to_table(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    rng = np.random.default_rng(0)
    O, A_, R = rng.normal(size=(6, 4)), rng.integers(0, 3, 6), rng.random(6)
    o2, a2, r2 = M.text_to_episode(M.episode_to_text(O, A_, R))
    assert np.allclose(o2, O, atol=1e-5) and np.array_equal(a2, A_) and np.allclose(r2, R, atol=1e-5)
    toks = rng.integers(0, 64, size=(9, 4)).astype(np.int32)
    t = M.audio_to_text(toks)
    assert t.startswith("[audio] a") and M.modality_of(t) == "audio" and np.array_equal(M.text_to_audio(t), toks)


def test_the_zoo_and_the_systems_own_source_become_text_documents_the_chat_pipeline_can_ingest():
    import test_v63 as T63
    from zerowhite.chat import shared
    from zerowhite.forge import families as F, task as T
    from zerowhite.forge.zoo import Zoo
    ws = tmp()
    task = T.EnvTask(T63.HiDimBandit())
    genome = dict(n_islands=2, island_size=6, tune_steps=4, max_nodes=50, n_mut=2)
    a = F.ProgramPolicyAgent(task, genome, 0)
    a.train(task, 0.5, np.random.default_rng(0))
    aid = Zoo(ws).add("hidim_bandit", a, dict(skill=0.42), True)
    out = os.path.join(tmp(), "data")
    r = shared.export_zoo(ws, os.path.join(out, "_zoo"))
    doc = fileio.read_text(os.path.join(out, "_zoo", f"zoo-hidim_bandit-{aid}.txt"))
    assert r == dict(documents=1, tasks=1) and "[zoo] task: hidim_bandit" in doc and "skill: 0.42" in doc and "def policy" in doc
    s = shared.ingest_self_source(os.path.join(out, "_self"))
    names = os.listdir(os.path.join(out, "_self"))
    assert s["documents"] == len(names) > 40 and "organism.txt" in names and "chat__grow.txt" in names
    org = fileio.read_text(os.path.join(out, "_self", "organism.txt"))
    assert org.startswith("[source zerowhite/organism.py]") and "class Organism" in org and "class Settings" in org
    again = tmp()
    shared.ingest_self_source(again)
    assert fileio.read_text(os.path.join(again, "organism.txt")) == org                                  # deterministic
    # the real ingestion path reads them: a chat workspace with these files in its data folder sees more documents
    import test_chat as TC
    from zerowhite.chat import prepare as PR
    cfg, lay = TC.copy_workspace()
    shared.ingest_self_source(os.path.join(lay.data, "_self"), os.path.join(ROOT, "zerowhite"))
    for f in os.listdir(os.path.join(lay.data, "_self")):                                  # ingestion skips files younger than its settle time: age them
        os.utime(os.path.join(lay.data, "_self", f), (time.time() - 3600, time.time() - 3600))
    rep = PR.prepare(cfg, lay, do_bench=False).ingest
    assert rep.files_new >= 40 and rep.records_out >= 40 and not rep.errors, rep.as_dict()   # the real ingestion path took them in as documents


def test_the_system2_race_rejects_unsafe_and_broken_proposals_and_ranks_the_programs_that_actually_work():
    from zerowhite.forge import system2 as S2, task as T

    class Bandit:
        obs_dim, n_actions, max_steps, max_return, name = 3, 2, 12, 12.0, "sign_bandit"

        def reset(self, rng):
            self.rng, self.t = rng, 0
            return self._n()

        def _n(self):
            self.o = self.rng.normal(size=3)
            return self.o.copy()

        def step(self, a):
            r = float(int(a) == int(self.o[0] > 0))                                            # the reward is for THIS observation, computed before the next one is drawn
            self.t += 1
            return self._n(), r, self.t >= 12, False
    task = T.EnvTask(Bandit())
    good = "def policy(obs):\n    return obs[0]\n"
    props = ["def policy(obs):\n    return obs[1]\n", good, "import os\ndef policy(obs):\n    return os.getpid()\n", "def policy(obs):\n    return (\n",
             "def policy(obs):\n    return -obs[0]\n", "def policy(obs):\n    return obs[0] + 0.0 * obs[2]\n", "def policy(obs):\n    return 1.0\n"]
    rows = S2.race(task, props)
    assert rows[0]["src"] in (good, props[5]) and rows[0]["skill"] > 0.8 and rows[1]["src"] in (good, props[5])       # the programs that read obs[0] win
    bad = [r for r in rows if not r["valid"]]
    assert len(bad) == 2 and all(r["skill"] is None for r in bad) and "unsafe" in "".join(r["reason"] for r in bad) and rows[-1] in bad and rows[-2] in bad
    ranked = [r["skill"] for r in rows if r["valid"]]
    assert ranked == sorted(ranked, reverse=True)
    assert max(r["episodes"] for r in rows) == 8 and min(r["episodes"] for r in rows if r["valid"]) < 8                # halving: weak programs stopped early
    never = [r for r in rows if r["src"] == props[2]][0]
    assert never["episodes"] == 0                                                                                     # the unsafe program was never run


def test_a_chat_model_can_be_the_proposer_and_the_winners_seed_the_agents_search():
    from zerowhite.forge import families as F, system2 as S2, task as T
    outputs = iter(["Sure! Here is one:\n```python\ndef policy(obs):\n    return obs[0]\n```\nand another\n```\ndef policy(obs):\n    return obs[1]\n```", "no code at all, sorry",
                    "def policy(obs):\n    return obs[0] * 2.0\n"])
    prop = S2.chat_proposer(lambda prompt: next(outputs))
    srcs = prop("a contextual bandit", n=3)
    assert len(srcs) == 3 and srcs[0].startswith("def policy") and "obs[1]" in srcs[1] and "2.0" in srcs[2]

    class Bandit:
        obs_dim, n_actions, max_steps, max_return, name = 3, 2, 12, 12.0, "sign_bandit"

        def reset(self, rng):
            self.rng, self.t = rng, 0
            return self._n()

        def _n(self):
            self.o = self.rng.normal(size=3)
            return self.o.copy()

        def step(self, a):
            r = float(int(a) == int(self.o[0] > 0))                                            # the reward is for THIS observation, computed before the next one is drawn
            self.t += 1
            return self._n(), r, self.t >= 12, False
    task = T.EnvTask(Bandit())
    genome = dict(n_islands=2, island_size=6, tune_steps=4, max_nodes=50, n_mut=2)
    cold = F.ProgramPolicyAgent(task, genome, 0)
    warm = F.ProgramPolicyAgent(task, genome, 0)
    n = S2.seed_agent(warm, ["def policy(obs):\n    return obs[0]\n", "import os\n", "def policy(obs):\n    return (\n"])
    assert n == 1                                                                                   # only the valid one is admitted
    assert warm.pe.best.score > cold.pe.best.score + 0.3 and "obs[0]" in warm.pe.best.src          # it starts from the good program instead of a trivial one


# ================================================================================================================ engine tier
def _repo_copy():
    repo = os.path.join(tmp(), "repo")
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns("__pycache__", ".git", "runs", "*.egg-info", ".zw_promotion"))
    return repo


def test_the_engine_gate_accepts_new_code_inside_functions_and_refuses_new_imports_dangerous_calls_and_missing_functions():
    from zerowhite.selfmod import engine_tier as E
    old = "import numpy as np\n\n\ndef f(x):\n    return x + 1\n\n\nclass K:\n    def g(self, y):\n        return y * 2\n"
    assert E.engine_safe(old, old) == (True, "")
    new_code = old.replace("    return x + 1\n", "    s = 0.0\n    for i in range(3):\n        s += np.tanh(x) * i\n    return s\n")
    assert E.engine_safe(old, new_code)[0]                                                  # loops, new tensor expressions: allowed
    for bad, why in ((old.replace("import numpy as np", "import numpy as np\nimport os"), "imports"), (old.replace("x + 1", "open('f').read()"), "open"),
                     (old.replace("x + 1", "np.sum(os.listdir('.'))"), "os"), (old.replace("def f(x)", "def renamed(x)"), "renames"), (old.replace("    return x + 1\n", "    return (\n"), "parse"),
                     (old.replace("y * 2", "__import__('os').getcwd()"), "__import__"), (old.replace("class K:", "class K2:"), "renames")):
        ok, reason = E.engine_safe(old, bad)
        assert not ok and why in reason, (why, reason)
    assert E.engine_safe("def f():\n    import os\n    return 1\n", "def f():\n    return 1\n") == (False, "the set of imports changed")      # removing an import changes the set too: refused
    assert E.engine_safe("def f(x):\n    return os.sep\n", "def f(x):\n    return os.sep\n")[0]                              # danger already present in the original is not 'new'


def test_single_edit_mutations_of_the_real_numerical_core_are_exactly_one_edit_safe_and_formatting_preserving():
    from zerowhite.selfmod import engine_tier as E
    rng = np.random.default_rng(0)
    for f in E.ENGINE:
        src = fileio.read_text(os.path.join(ROOT, "zerowhite", f))
        muts = E.mutations(src, rng, limit=30)
        assert len(muts) >= 15, (f, len(muts))
        for new, desc in muts:
            assert new != src and E.one_edit(src, new) and E.engine_safe(src, new)[0], (f, desc)
            a, b = src.splitlines(), new.splitlines()
            assert len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1, (f, desc)           # one line changed, the rest byte-for-byte as written
        two = muts[0][0]
        two = E._replace(two, 1, 0, 0, "")                                                           # a no-op replace keeps it one edit ...
        assert E.one_edit(src, two)
        both = E.mutations(muts[0][0], rng, limit=1)[0][0]
        assert not E.one_edit(src, both)                                                             # ... but two edits are not a single edit


def test_the_tier_never_reaches_the_judge_and_the_constitution_still_lists_it():
    from zerowhite import doctor
    from zerowhite.selfmod import engine_tier as E, patcher
    assert set(E.ENGINE).isdisjoint(patcher.MUTABLE)
    sealed = {s.split("/")[-1] for s in doctor.CONSTITUTION}
    for judge in ("safety.py", "sandbox.py", "patcher.py", "history.py", "stats.py", "watchdog.py", "winjob.py", "fileio.py", "doctor.py", "engine_tier.py", "astmut.py"):
        assert judge in sealed, judge
    assert not [f for f in E.ENGINE if f.split("/")[-1] in sealed] and "astmut.py" not in E.ENGINE
    for f in ("astmut.py", "selfmod/safety.py", "doctor.py", "evolve.py"):
        try:
            E.evaluate_candidate(ROOT, None, {f: "x = 1\n"})
            raise AssertionError(f)
        except PermissionError:
            pass


def _stubs(repo, tests_ok=True, real_ok=True, bench=None):
    from zerowhite.selfmod import patcher
    calls = {"tests": 0}
    orig = fileio.read_text(os.path.join(repo, "zerowhite", "nn.py"))

    def tests(root, timeout=0):
        calls["tests"] += 1
        return (real_ok, "real tree disagrees") if os.path.abspath(root) == os.path.abspath(repo) else (tests_ok, "sandbox says no")

    def default_bench(root, seeds, timeout=0):
        changed = fileio.read_text(os.path.join(root, "zerowhite", "nn.py")) != orig
        return 0.5 + (0.03 if changed else 0.0) + 0.001 * (seeds[0] % 3)
    return calls, tests, bench or default_bench, orig


def _with_stubs(tests, bench, fn):
    from zerowhite.selfmod import patcher
    rb, rt = patcher.bench_in, patcher.tests_pass
    patcher.bench_in, patcher.tests_pass = bench, tests
    try:
        return fn()
    finally:
        patcher.bench_in, patcher.tests_pass = rb, rt


def test_an_engine_candidate_goes_through_every_gate_and_only_a_full_pass_changes_the_repository():
    from zerowhite.selfmod import engine_tier as E
    from zerowhite.selfmod.history import HistoryStore
    repo = _repo_copy()
    nn_path = os.path.join(repo, "zerowhite", "nn.py")
    calls, tests, bench, orig = _stubs(repo)
    good = orig.replace("0.999", "0.99", 1)
    assert good != orig
    # (a) accepted but not applied: the repository is untouched
    r = _with_stubs(tests, bench, lambda: E.evaluate_candidate(repo, HistoryStore(tmp()), {"nn.py": good}, apply=False))
    assert r["accepted"] and not r["applied"] and r["gate"]["p"] <= 1 / 64 + 1e-12 and fileio.read_text(nn_path) == orig
    # (b) applied through the journal
    r = _with_stubs(tests, bench, lambda: E.evaluate_candidate(repo, HistoryStore(tmp()), {"nn.py": good}, apply=True))
    assert r["accepted"] and r["applied"] and fileio.read_text(nn_path) == good
    states = [json.loads(open(os.path.join(repo, ".zw_promotion", n, "journal.json")).read())["state"] for n in os.listdir(os.path.join(repo, ".zw_promotion"))]
    assert states == ["committed"]
    fileio.write_text(nn_path, orig)
    # (c) an unsafe candidate is stopped before any test or benchmark runs
    calls["tests"] = 0
    r = _with_stubs(tests, bench, lambda: E.evaluate_candidate(repo, HistoryStore(tmp()), {"nn.py": "import os\n" + good}, apply=True))
    assert r["stage"] == "structure" and not r["accepted"] and calls["tests"] == 0 and fileio.read_text(nn_path) == orig
    # (d) tests fail in the sandbox copy
    calls2, tests2, bench2, _ = _stubs(repo, tests_ok=False)
    r = _with_stubs(tests2, bench2, lambda: E.evaluate_candidate(repo, HistoryStore(tmp()), {"nn.py": good}, apply=True))
    assert r["stage"] == "tests" and "sandbox says no" in r["reason"] and fileio.read_text(nn_path) == orig
    # (e) sandbox passes, the real tree disagrees: rolled back by the journal
    calls3, tests3, bench3, _ = _stubs(repo, real_ok=False)
    r = _with_stubs(tests3, bench3, lambda: E.evaluate_candidate(repo, HistoryStore(tmp()), {"nn.py": good}, apply=True))
    assert r["stage"] == "promotion" and "real tree disagrees" in r["reason"] and fileio.read_text(nn_path) == orig
    # (f) a gain that comes from one lucky seed is refused by the significance gate, although its mean clears the margin
    def lucky(root, seeds, timeout=0):
        changed = fileio.read_text(os.path.join(root, "zerowhite", "nn.py")) != orig
        return 0.5 + (0.25 if (changed and seeds[0] == 100) else (-0.002 if changed else 0.0))
    calls4, tests4, _, _ = _stubs(repo)
    r = _with_stubs(tests4, lucky, lambda: E.evaluate_candidate(repo, HistoryStore(tmp()), {"nn.py": good}, apply=True))
    assert r["final_patched"] > r["final_base"] + 0.004 and r["stage"] == "gate" and "not significant" in r["reason"] and fileio.read_text(nn_path) == orig


def test_the_engine_search_hill_climbs_on_single_edits_and_hands_the_survivors_to_the_judge():
    from zerowhite.selfmod import engine_tier as E
    from zerowhite.selfmod.history import HistoryStore
    repo = _repo_copy()
    calls, tests, bench, orig = _stubs(repo)
    r = _with_stubs(tests, bench, lambda: E.search(repo, HistoryStore(tmp()), evals=6, seed=3, apply=True, verbose=False, files=("nn.py",)))
    assert r["accepted"] and r["applied"] and r["tried"] >= 1 and len(r["kept"]) >= 1, r
    now = fileio.read_text(os.path.join(repo, "zerowhite", "nn.py"))
    assert now != orig and E.engine_safe(orig, now)[0]
    repo2 = _repo_copy()
    flat = lambda root, seeds, timeout=0: 0.5                                                      # nothing ever helps: the judge's answer is 'no change'     # noqa: E731
    r = _with_stubs(tests, flat, lambda: E.search(repo2, HistoryStore(tmp()), evals=5, seed=3, apply=True, verbose=False, files=("nn.py",)))
    assert not r["accepted"] and r["stage"] == "search" and fileio.read_text(os.path.join(repo2, "zerowhite", "nn.py")) == orig
    cli = subprocess.run([sys.executable, "-m", "zerowhite", "engine-tier", "--help"], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert cli.returncode == 0 and "--apply" in cli.stdout


def test_quantiser_dropout_is_what_makes_a_prefix_of_the_code_decode_well():
    """Measured (seeds 0, 1; stages 1/2/3): with dropout 0.55/1.05/1.32 and 0.70/1.10/1.34 dB, without 0.41/0.80/1.13 and 0.57/0.96/1.28 - modest but consistent."""
    from zerowhite.voice import codec as C
    cd, train = _small_codec()
    S = ["the quick brown fox jumps over the lazy dog", "hello world how are you today", "she sells sea shells by the sea shore", "please call me back tomorrow morning"]
    held = _speech([("the weather is nice", 130, 1.0), ("thank you very much", 160, 1.0)])
    plain = C.train(train, stages=4, codewords=64, hidden=128, latent=32, epochs=60, finetune_epochs=10, seed=0, quantiser_dropout=False)
    prefix = lambda c: np.mean([np.mean([C.snr_db(w, c.roundtrip(w, s)) for w in held]) for s in (1, 2, 3)])        # noqa: E731
    assert prefix(cd) > prefix(plain) + 0.08, (prefix(cd), prefix(plain))


def test_the_ranker_alone_would_turn_the_example_essay_into_an_eleven_column_table_and_the_table_check_is_what_stops_it():
    from zerowhite.omnivore import learned, tables
    d = tmp()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "examples", "messy_inbox.py"), d], capture_output=True, text=True, timeout=120)
    essay = fileio.read_text(os.path.join(d, "essay.md"))
    assert tables.parse_delimited(essay) is None                                                  # the rules: prose
    h = learned.hint(essay.encode("utf-8"), 0.97)
    assert h == {"delim": " "}                                                                    # the ranker is sure it is space-delimited ...
    junk = tables.parse_delimited(essay, hint=dict(h, arbitrate=True))
    assert junk is not None and len(junk.columns) >= 8 and tables.table_quality(junk) == 0        # ... and the parse it licenses is a many-column table with no numeric column
    for mode in ("rescue", "arbitrate"):
        assert learned.parse_with_hint(essay, essay.encode("utf-8"), mode) is None                # so devour does not accept it


def test_the_cuda_launch_covers_the_whole_array_with_the_right_grid_for_big_sizes():
    from zerowhite.kernels import cjit, cuda, graph as G
    if not cjit.available():
        return
    x = np.linspace(-3, 3, 1000)
    g = (G.lazy(x) * 2.0 + 1.0).tanh()
    drv = _FakeDriver()
    be = cuda.CudaBackend(driver=drv, nvrtc=_FakeNvrtc())
    got = G.evaluate(g, "cuda", cuda=be)
    assert np.allclose(got, np.tanh(x * 2.0 + 1.0), atol=1e-14)
    launch = [e for e in drv.log if isinstance(e, tuple) and e[0] == "launch"][0]
    assert launch[1:] == (4, 256)                                                                 # ceil(1000 / 256) blocks of 256 threads
