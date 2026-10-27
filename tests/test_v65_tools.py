"""Tests for 6.5 tool forging (forge/tools.py): synthesis from examples, the script gate, the sandbox, the toolbox, chaining, requests.
Each test must fail when the property it names breaks."""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ZERO_WHITE_DATA", tempfile.mkdtemp(prefix="zw_test_data_"))

from zerowhite.forge import tools as T                                  # noqa: E402

_DIRS = []


def tmp():
    d = tempfile.mkdtemp(prefix="zw_t65t_")
    _DIRS.append(d)
    return d


import atexit                                                          # noqa: E402
atexit.register(lambda: [shutil.rmtree(d, ignore_errors=True) for d in _DIRS])

CSV_JSON = [("a,b,c", '["a","b","c"]'), ("x,y", '["x","y"]'), ("1,2,3,4", '["1","2","3","4"]'), ("hello,world", '["hello","world"]'), ("p,q,r", '["p","q","r"]'), ("m,n", '["m","n"]')]


def test_a_tool_is_forged_from_examples_stored_with_a_hash_and_reproduces_them_in_the_sandbox():
    tb = T.Toolbox(tmp())
    r = tb.forge("csv_to_json", CSV_JSON, "turn one CSV line into a JSON list")
    assert r.ok, r.reason
    assert r.sandbox_ok and r.validated and r.holdout_n >= 1 and r.train_n + r.holdout_n == len(CSV_JSON)
    spec = tb.spec("csv_to_json")
    assert spec["origin"] == "forged" and spec["in_type"] == "str" and spec["out_type"] == "str" and spec["validated"] is True
    assert tb.intact("csv_to_json") and spec["sha256"]
    src = tb.source("csv_to_json")
    assert T.check_source(src) == (True, "") and "def run(x):" in src and "import" in src
    ok, res = T.run_in_sandbox(src, ["u,v,w", "k"])                      # a fresh input, run by a subprocess, never seen during the search
    assert ok and [r_["v"] for r_ in res] == ['["u","v","w"]', '["k"]']


def test_the_script_is_a_standalone_program_that_reads_json_on_stdin():
    import subprocess
    tb = T.Toolbox(tmp())
    assert tb.forge("sum_in_text", [("I have 3 apples and 4 pears", 7), ("12 and 30", 42), ("no numbers", 0), ("5", 5), ("1 2 3 4", 10), ("a7b8", 15)]).ok
    p = subprocess.run([sys.executable, "-B", os.path.join(tb.dir("sum_in_text"), "tool.py")], input=json.dumps("room 101 and 9"), capture_output=True, text=True, timeout=30)
    assert p.returncode == 0 and json.loads(p.stdout) == 110, (p.stdout, p.stderr)


def test_a_program_that_only_fits_the_training_examples_is_rejected_by_the_held_out_ones():
    train = [("aa", "a"), ("bb", "b"), ("cc", "c")]
    hold = [("ab", "a"), ("zy", "z")]
    expr, info = T.synthesize(train, hold, max_size=4)
    assert expr is not None and expr.show() == "head1(x)", expr.show() if expr else info
    assert info["overfit"] >= 1                                           # at least one program matched the training examples and failed the held-out ones
    expr2, info2 = T.synthesize(train, [], max_size=4)                    # without held-out examples the first match wins, whatever it is
    assert expr2 is not None and not T._equal(T._eval_expr(expr2, "ab"), "a"), expr2.show()    # ... which is exactly the program the held-out examples exist to catch


def test_with_fewer_than_five_examples_nothing_is_held_out_and_the_result_says_so():
    tb = T.Toolbox(tmp())
    r = tb.forge("upper_bang", [("hello", "HELLO!"), ("abc def", "ABC DEF!"), ("x", "X!"), ("good night", "GOOD NIGHT!")])
    assert r.ok and r.holdout_n == 0 and r.validated is False and tb.spec("upper_bang")["validated"] is False
    assert not tb.forge("two", [("a", "A"), ("b", "B")]).ok                # too few examples is refused, not guessed at


def test_a_tool_is_built_from_stored_tools_and_its_script_carries_their_source():
    tb = T.Toolbox(tmp())
    assert tb.forge("csv_nums", [("1,2,3", [1, 2, 3]), ("10,20", [10, 20]), ("5", [5]), ("7,8,9,10", [7, 8, 9, 10]), ("4,4", [4, 4])]).ok
    r = tb.forge("csv_spread", [("1,5,3", 4), ("10,20", 10), ("7", 0), ("3,9,6", 6), ("100,1", 99), ("2,2,2", 0)])
    assert r.ok, r.reason
    assert r.uses == ["csv_nums"] and tb.spec("csv_spread")["parts"] == ["csv_nums"], (r.expr, r.uses)
    src = tb.source("csv_spread")
    assert "_t_csv_nums" in src and T.check_source(src)[0]                 # one file, no import of the other tool
    ok, res = T.run_in_sandbox(src, ["2,50,7"])
    assert ok and res[0]["v"] == 48
    assert tb.run("csv_spread", "9,1,4") == 8


def test_chaining_runs_tools_one_after_the_other_and_counts_uses_and_failures():
    tb = T.Toolbox(tmp())
    assert tb.forge("csv_nums", [("1,2,3", [1, 2, 3]), ("10,20", [10, 20]), ("5", [5]), ("7,8,9,10", [7, 8, 9, 10]), ("4,4", [4, 4])]).ok
    assert tb.forge("total", [([1, 2, 3], 6), ([10, 20], 30), ([5], 5), ([7, 8], 15), ([], 0)]).ok
    assert tb.chain(["csv_nums", "total"], "4,5,6") == 15
    assert tb.spec("csv_nums")["uses"] == 1 and tb.spec("total")["uses"] == 1
    try:
        tb.run("csv_nums", "not,a,number!")
        failed = False
    except Exception:                                                      # noqa: BLE001
        failed = True
    s = tb.spec("csv_nums")
    assert s["uses"] == 2 and (failed is False or s["failures"] == 1)
    assert [t["name"] for t in tb.list()] == ["csv_nums", "total"]


def test_the_script_gate_refuses_what_a_tool_may_not_do():
    bad = {
        "os import": "import os\ndef run(x):\n    return os.getcwd()\n",
        "open": "def run(x):\n    return open('/etc/passwd').read()\n",
        "exec": "def run(x):\n    exec('1')\n    return x\n",
        "eval": "def run(x):\n    return eval(x)\n",
        "__import__": "def run(x):\n    return __import__('os')\n",
        "dunder attribute": "def run(x):\n    return x.__class__\n",
        "from import": "from os import path\ndef run(x):\n    return x\n",
        "getattr": "def run(x):\n    return getattr(x, 'upper')()\n",
        "global": "def run(x):\n    global y\n    return x\n",
        "class": "class A:\n    pass\ndef run(x):\n    return x\n",
        "no run": "def other(x):\n    return x\n",
        "syntax": "def run(x:\n    return x\n",
        "sys use": "import sys\ndef run(x):\n    return sys.modules\n",
        "subprocess": "import subprocess\ndef run(x):\n    return 1\n",
        "empty": "",
    }
    for what, src in bad.items():
        ok, why = T.check_source(src)
        assert not ok and why, f"{what} was accepted"
    assert T.check_source("import re, json\n\n\ndef run(x):\n    return re.findall(r'\\d+', x)\n")[0]


def test_a_script_that_passes_the_gate_but_never_ends_is_killed_by_the_sandbox():
    src = "def run(x):\n    while True:\n        pass\n"
    assert T.check_source(src)[0]
    ok, res = T.run_in_sandbox(src, [1], timeout=3)
    assert not ok and "timeout" in res


def test_a_script_cannot_reach_files_or_modules_even_if_it_slipped_past_the_gate_in_process():
    src = "def run(x):\n    return len(x)\n"
    f = T.load_function(src)
    assert f("abc") == 3
    for evil in ("def run(x):\n    return open('x')\n", "import os\ndef run(x):\n    return 1\n"):
        try:
            T.load_function(evil)
            raise AssertionError("accepted")
        except ValueError:
            pass
    ns_src = "import json\ndef run(x):\n    return json.dumps(x)\n"
    assert T.load_function(ns_src)([1]) == "[1]"
    try:                                                                   # the restricted import table itself refuses anything outside the allow-list
        T._safe_import("os")
        raise AssertionError("os imported")
    except ImportError:
        pass


def test_a_supplied_script_must_reproduce_its_examples_in_the_sandbox_and_never_runs_in_process():
    tb = T.Toolbox(tmp())
    good = "def run(x):\n    return x.upper()\n"
    rec = tb.register("shout", good, [{"input": "a", "output": "A"}, {"input": "bc", "output": "BC"}])
    assert rec["origin"] == "supplied"
    assert tb.run("shout", "q") == "Q"
    for src, ex, why in (("def run(x):\n    return x.lower()\n", [("A", "A")], "did not reproduce"), ("import os\ndef run(x):\n    return x\n", [("a", "a")], "refused")):
        try:
            tb.register("nope", src, ex)
            raise AssertionError("accepted")
        except ValueError as e:
            assert why in str(e)
    assert "nope" not in tb.names()
    assert not [p for p in tb.tool_prims() if p.name == "tool:shout"]      # supplied code is never used as a building block of an in-process search


def test_a_tampered_tool_is_not_run_in_process_and_a_dangerous_edit_is_refused_by_the_sandbox_path():
    tb = T.Toolbox(tmp())
    assert tb.forge("upper_bang", [("hello", "HELLO!"), ("abc def", "ABC DEF!"), ("x", "X!"), ("good night", "GOOD NIGHT!")]).ok
    p = os.path.join(tb.dir("upper_bang"), "tool.py")
    with open(p, "a", encoding="utf-8") as f:
        f.write("\nimport os\n")
    assert not tb.intact("upper_bang")
    try:
        tb.run("upper_bang", "q")
        raise AssertionError("ran")
    except RuntimeError as e:
        assert "refused" in str(e)
    assert not [p_ for p_ in tb.tool_prims() if p_.name == "tool:upper_bang"]


def test_every_primitive_returns_what_it_declares_and_its_emitted_script_agrees_with_it():
    samples = {"str": "Hello, World 42\nline two", "strs": ["b", "a", "b", " c "], "nums": [3, 1, 2, 3], "rows": [["h1", "h2"], ["a", "b"]], "dict": {"k": "v", "n": 2},
               "dicts": [{"k": "v", "n": 1}, {"k": "w", "n": 2}], "num": 7, "bool": True}
    c = T.Consts.from_examples(["a,b;c=d", "x"], ["A", "B"])
    prims = T.build_prims(c)
    assert len(prims) > 200
    checked = 0
    for p in prims:
        if p.arity != 1 or p.args[0] not in samples:
            continue
        try:
            want = p.fn(samples[p.args[0]])
        except Exception:                                                  # noqa: BLE001 - a primitive may legitimately not apply to this sample
            continue
        assert T._types_ok((want,), p.ret), (p.name, want, p.ret)
        e_in = T.Expr(None, (), (samples[p.args[0]],), 1, p.args[0], is_var=True)
        e = T.Expr(p, (e_in,), (want,), 2, p.ret)
        src = T.emit_script(e, "t", "d", [])
        ok, why = T.check_source(src)
        assert ok, (p.name, why)
        got = T.load_function(src)(samples[p.args[0]])
        assert T._equal(got, want), (p.name, got, want)
        checked += 1
    assert checked > 150


def test_a_search_that_cannot_succeed_says_why_and_stays_inside_its_budget():
    import time
    tb = T.Toolbox(tmp())
    nonsense = [("a", "q"), ("b", "z"), ("c", "m"), ("d", "x"), ("e", "k"), ("f", "w")]
    t0 = time.time()
    r = tb.forge("random_map", nonsense, seconds=3, max_size=5)
    assert not r.ok and r.reason and time.time() - t0 < 30 and "random_map" not in tb.names()
    r2 = tb.forge("mixed", [("a", 1), ("b", "x"), ("c", 2)])
    assert not r2.ok and "same shape" in r2.reason


def test_names_are_validated_and_an_existing_tool_is_not_overwritten_unless_asked():
    tb = T.Toolbox(tmp())
    for bad in ("../x", "Upper", "a b", "", "9x", "x" * 40):
        assert not tb.forge(bad, CSV_JSON).ok
    assert tb.forge("one", CSV_JSON).ok
    again = tb.forge("one", CSV_JSON)
    assert not again.ok and "already exists" in again.reason
    assert tb.forge("one", CSV_JSON, replace=True).ok


def test_requests_dropped_in_a_folder_are_answered_once_with_a_done_or_a_failed_record():
    ws = tmp()
    d = T.requests_dir(ws)
    os.makedirs(d)
    with open(os.path.join(d, "csv_to_json.json"), "w", encoding="utf-8") as f:
        json.dump(dict(name="csv_to_json", description="d", examples=[dict(input=i, output=o) for i, o in CSV_JSON]), f)
    with open(os.path.join(d, "impossible.json"), "w", encoding="utf-8") as f:
        json.dump(dict(name="impossible", seconds=2, examples=[dict(input=c, output=o) for c, o in [("a", "q"), ("b", "z"), ("c", "m"), ("d", "x"), ("e", "k")]]), f)
    with open(os.path.join(d, "broken.json"), "w", encoding="utf-8") as f:
        f.write("{not json")
    assert len(T.pending_requests(ws)) == 3
    res = T.process_requests(ws, limit=5, seconds=3)
    assert {r.name: r.ok for r in res} == {"broken": False, "csv_to_json": True, "impossible": False}
    assert T.pending_requests(ws) == []
    left = sorted(os.listdir(d))
    assert left == ["broken.failed.json", "csv_to_json.done.json", "impossible.failed.json"], left
    assert T.process_requests(ws) == []                                    # answered once, never retried
    assert T.Toolbox(ws).names() == ["csv_to_json"]


def test_forged_tools_cover_the_kinds_of_job_the_wishlist_names():
    """A parser, a converter and a small solver, each from examples, each executed in the sandbox."""
    jobs = {
        "html_links": [('<a href="http://a.com/x">A</a>', ["http://a.com/x"]), ('<p>hi</p><a href="/b">B</a><a href="/c">C</a>', ["/b", "/c"]), ("none", []), ('<a href="z">Z</a>', ["z"]),
                       ('<a href="q">1</a> <a href="r">2</a>', ["q", "r"])],
        "kv_to_json": [("a=1;b=2", '{"a":"1","b":"2"}'), ("x=y", '{"x":"y"}'), ("k=v;l=w;m=n", '{"k":"v","l":"w","m":"n"}'), ("name=bob;age=7", '{"name":"bob","age":"7"}'), ("u=1", '{"u":"1"}')],
        "md_to_csv": [("| a | b |\n| --- | --- |\n| 1 | 2 |", "a,b\n1,2"), ("| x | y |\n|---|---|\n| p | q |\n| r | s |", "x,y\np,q\nr,s"), ("| n |\n| - |\n| 5 |", "n\n5"),
                      ("| u | v | w |\n| - | - | - |\n| 1 | 2 | 3 |", "u,v,w\n1,2,3")],
        "parity": [(2, "even"), (3, "odd"), (10, "even"), (7, "odd"), (0, "even"), (13, "odd")],
        "mean_of": [([1, 2, 3], 2), ([10, 20], 15), ([5], 5), ([2, 4, 6, 8], 5), ([1, 1, 4], 2)],
    }
    tb = T.Toolbox(tmp())
    for name, ex in jobs.items():
        r = tb.forge(name, ex, seconds=20)
        assert r.ok and r.sandbox_ok, (name, r.reason)
    assert tb.run("parity", 21) == "odd" and tb.run("mean_of", [4, 6]) == 5
    assert tb.run("kv_to_json", "p=1;q=2;r=3") == '{"p":"1","q":"2","r":"3"}'
