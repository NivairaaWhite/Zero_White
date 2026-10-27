"""Tests for 6.5 'the chat model as architect' (forge/architect.py), the structural program mutators it can ask for (selfmod/astmut.py) and the pinned-program support in the program agents."""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ZERO_WHITE_DATA", tempfile.mkdtemp(prefix="zw_test_data_"))

from zerowhite.forge import architect as R                              # noqa: E402
from zerowhite.forge import families as F                               # noqa: E402
from zerowhite.forge import task as T                                   # noqa: E402
from zerowhite.selfmod import astmut as A                               # noqa: E402

_DIRS = []


def tmp():
    d = tempfile.mkdtemp(prefix="zw_t65a_")
    _DIRS.append(d)
    return d


import atexit                                                           # noqa: E402
atexit.register(lambda: [shutil.rmtree(d, ignore_errors=True) for d in _DIRS])

SPACES = {n: f.space for n, f in F.FAMILIES.items()}
PROGRAM_GENOME = dict(n_islands=2, island_size=6, tune_steps=6, max_nodes=60, n_mut=1)


class FakeEngine:
    """Stands in for chat.generate.Engine: answers every question with a fixed reply."""
    ready = True

    def __init__(self, reply):
        self.reply_text, self.asked = reply, []

    def reply(self, gen, system, turns, **over):
        self.asked.append((system, turns[-1]["content"], over))
        return self.reply_text


def _regression_task(n=240, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 3))
    y = 3.0 * X[:, 0] + 0.05 * r.normal(size=n)
    return T.from_arrays("lin3", X, y, "regression", seed=seed)


def _formula_agent(task, src, seed=0):
    a = F.FAMILIES["program_formula"].build(task, PROGRAM_GENOME, seed)
    a.adopt(src)
    return a


# --------------------------------------------------------------------------- structural mutators
def test_structural_mutators_make_safe_programs_that_run_and_that_differ_from_their_parent():
    v = A.Vocab("predict", ("x",), col_var="x", n_cols=3, vector=True, max_nodes=80)
    src = "def predict(x):\n    return x[:, 0] * 2.0\n"
    X = np.random.default_rng(0).normal(size=(7, 3))
    base = A.compile_program(src, "predict", **v.safety)(X)
    for kind in ("gate", "neuron", "mgate"):
        got = 0
        for s in range(12):
            out = A.apply_mutator(src, v, kind, np.random.default_rng(s))
            if out is None:
                continue
            got += 1
            assert not A.violations(out, **v.safety), out
            assert A.count_nodes(out) <= v.max_nodes
            y = A.compile_program(out, "predict", **v.safety)(X)
            assert np.all(np.isfinite(y)) and np.shape(y) == (7,)
            assert A.normalize(out) != A.normalize(src)
            if kind == "neuron":
                assert "t0" in out                                               # a new hidden unit, wired into the return value
        assert got >= 6, f"the '{kind}' mutator found something to do in only {got} of 12 tries"
    gated = A.apply_mutator(src, v, "gate", np.random.default_rng(1))
    assert gated is not None and not np.allclose(base, A.compile_program(gated, "predict", **v.safety)(X))


def test_a_scalar_program_can_get_a_branch_and_apply_mutator_rejects_names_it_does_not_have():
    v = A.Vocab("policy", ("obs",), col_var="obs", n_cols=4, vector=False, max_nodes=80, leaves=())
    out = A.apply_mutator("def policy(obs):\n    return obs[1] + 0.5 * obs[2]\n", v, "gate", np.random.default_rng(0))
    assert out is not None and (" if " in out or "sel(" in out)
    fn = A.compile_program(out, "policy", **v.safety)
    assert np.isfinite(float(fn(np.array([0.1, 0.2, -0.3, 0.4]))))
    try:
        A.apply_mutator("def policy(obs):\n    return 0.0\n", v, "format_disk", np.random.default_rng(0))
        raise AssertionError("an unknown mutator name was accepted")
    except KeyError:
        pass


def test_structural_moves_are_off_by_default_and_on_when_the_vocabulary_asks_for_them():
    assert A.Vocab("p", ("x",)).structural == 0.0
    v0 = A.Vocab("policy", ("obs",), col_var="obs", n_cols=4, vector=False, max_nodes=80, leaves=())
    vs = A.Vocab("policy", ("obs",), col_var="obs", n_cols=4, vector=False, max_nodes=80, leaves=(), structural=1.0)
    src = "def policy(obs):\n    return obs[1]\n"
    plain, structural = 0, 0
    for s in range(25):
        a = A.mutate(src, v0, np.random.default_rng(s))
        b = A.mutate(src, vs, np.random.default_rng(s))
        plain += int(a is not None and ("t0" in a or "sig(" in a and "* sig(" in a))
        structural += int(b is not None and ("t0" in b or "sel(" in b or " if " in b or "sig(" in b))
        for out in (a, b):
            if out is not None:
                assert not A.violations(out, **v0.safety)
    assert structural >= 12 and structural > plain


# --------------------------------------------------------------------------- the grammar
def test_the_grammar_accepts_exactly_what_it_describes_and_gives_a_reason_for_everything_else():
    text = "\n".join([
        "Sure! Here are my ideas:",
        'HYPOTHESIS {"kind": "program", "mutator": "neuron", "n": 2}',
        '{"kind": "genome", "family": "mlp", "changes": {"hidden": {"x": 2}, "lr": 0.01}}',
        'HYPOTHESIS {"kind": "hook", "hook": "sigma_update"}',
        'HYPOTHESIS {"kind": "program", "mutator": "neuron", "n": 2}',                         # a duplicate
        'HYPOTHESIS {"kind": "program", "mutator": "format_disk"}',
        'HYPOTHESIS {"kind": "program", "mutator": "gate", "n": 0}',
        'HYPOTHESIS {"kind": "program", "mutator": "gate", "n": true}',
        'HYPOTHESIS {"kind": "program", "mutator": "gate", "cmd": "rm -rf /"}',
        'HYPOTHESIS {"kind": "genome", "family": "mlp", "changes": {"nope": 1}}',
        'HYPOTHESIS {"kind": "genome", "family": "no_such_family", "changes": {"hidden": 8}}',
        'HYPOTHESIS {"kind": "genome", "family": "mlp", "changes": {"hidden": {"x": 100}}}',
        'HYPOTHESIS {"kind": "genome", "family": "mlp", "changes": {"hidden": "__import__(\'os\')"}}',
        'HYPOTHESIS {"kind": "hook", "hook": "../../etc/passwd"}',
        'HYPOTHESIS {"kind": "patch", "file": "safety.py"}',
        'HYPOTHESIS {"kind": "teleport"}',
        "HYPOTHESIS not json at all",
        "HYPOTHESIS {" + '"kind": "program", "mutator": "gate",' + " " * 700 + '"n": 1}',
        "{broken json",
    ])
    good, bad = R.parse(text, spaces=SPACES, patch_files=("evolve.py", "agents.py"))
    assert [h.kind for h in good] == ["program", "genome", "hook"]
    assert good[0].params == dict(mutator="neuron", n=2)
    assert good[1].params == dict(family="mlp", changes={"hidden": {"x": 2.0}, "lr": 0.01})
    assert len(bad) == 14 and all(isinstance(why, str) and why for _, why in bad)
    whys = " | ".join(w for _, w in bad)
    for frag in ("mutator must be", "n must be", "unknown keys", "no gene", "no agent family", "factor between", "names one of", "not one of the files", "kind must be", "no JSON", "too long"):
        assert frag in whys, frag


def test_a_hypothesis_survives_being_written_and_read_back_and_a_patch_may_only_name_a_patchable_file():
    for h in (R.Hypothesis("program", dict(mutator="mgate", n=3)), R.Hypothesis("genome", dict(family="mlp", changes={"hidden": 12, "lr": {"x": 0.5}})),
              R.Hypothesis("hook", dict(hook="select_parent")), R.Hypothesis("patch", dict(file="evolve.py"))):
        good, bad = R.parse(h.line(), spaces=SPACES, patch_files=("evolve.py",))
        assert not bad and len(good) == 1 and (good[0].kind, good[0].params) == (h.kind, h.params), h.line()
    from zerowhite.selfmod import patcher
    assert "evolve.py" in patcher.MUTABLE and not any("safety" in f for f in patcher.MUTABLE)


# --------------------------------------------------------------------------- proposers and the back-off
def test_the_heuristic_proposer_only_writes_grammatical_hypotheses_and_follows_what_has_worked():
    arms = R.Arms()
    state = dict(task="t", kind="regression", family="mlp", skill=0.2, src="def predict(x):\n    return x[:, 0]\n", genome=dict(hidden=16, depth=1, lr=0.01, wd=1e-4, batch=32),
                 space=SPACES["mlp"], hooks=False)
    hs = R.HeuristicProposer().propose(state, arms, np.random.default_rng(0), 6)
    assert len(hs) == 6 and len({h.text for h in hs}) == 6
    good, bad = R.parse("\n".join(h.text for h in hs), spaces=SPACES)
    assert len(good) == 6 and not bad
    for m in A.MUTATOR_KINDS:
        arms.d[f"program/{m}"] = [0, 40]
    for g in SPACES["mlp"]:
        arms.d[f"genome/mlp/{g}"] = [0, 40]
    arms.d["program/neuron"] = [40, 40]
    top = [R.HeuristicProposer().propose(state, arms, np.random.default_rng(s), 1)[0].key() for s in range(20)]
    assert top.count("program/neuron") >= 14


def test_the_chat_proposer_asks_in_the_documented_format_keeps_valid_lines_and_counts_the_rest():
    eng = FakeEngine('Try this:\nHYPOTHESIS {"kind": "program", "mutator": "gate", "n": 1}\nHYPOTHESIS {"kind": "program", "mutator": "lunch"}\nHYPOTHESIS {"kind": "hook", "hook": "sigma_update"}')
    cp = R.ChatProposer(eng, seed=3)
    state = dict(task="lin3", kind="regression", family="program_formula", skill=0.31, plateau=True, src="def predict(x):\n    return x[:, 0] * 0.5\n", genome=None, space=None)
    hs = cp.propose(state, R.Arms(), np.random.default_rng(0), 3, spaces=SPACES, recent=[dict(key="program/const", accepted=True)])
    assert [h.key() for h in hs] == ["program/gate", "hook/sigma_update"] and all(h.source == "chat" for h in hs)
    assert cp.invalid == 1 and cp.asked == 1
    prompt = eng.asked[0][1]
    assert prompt == R.build_prompt(state, [dict(key="program/const", accepted=True)], 3)
    assert "[architect]" in prompt and "plateau" in prompt and "program/const -> kept" in prompt and "HYPOTHESIS {json}" in prompt
    eng.ready = False
    assert cp.propose(state, R.Arms(), np.random.default_rng(0), 3, spaces=SPACES) == []


def test_a_chat_model_that_fails_or_rambles_costs_nothing_and_a_chat_model_that_keeps_missing_is_asked_less():
    class Broken(FakeEngine):
        def reply(self, *a, **k):
            raise RuntimeError("model exploded")
    cp = R.ChatProposer(Broken(""), seed=1)
    assert cp.propose(dict(task="t", kind="env", family="f", skill=0.0, src="def policy(obs):\n    return 0.0\n"), R.Arms(), np.random.default_rng(0), 2, spaces=SPACES) == []
    arch = R.Architect(tmp(), seed=0, chat=R.ChatProposer(FakeEngine("nothing useful")))
    assert arch.chat_share() == R.np.clip(0.5, 0.1, 0.9) or abs(arch.chat_share() - 0.5) < 1e-9          # no evidence: equal footing
    arch.arms.d["source/chat"], arch.arms.d["source/heuristic"] = [0, 60], [20, 60]
    assert arch.chat_share() == 0.1                                                                       # it is still probed
    arch.arms.d["source/chat"], arch.arms.d["source/heuristic"] = [50, 60], [1, 60]
    assert arch.chat_share() == 0.9
    assert R.Architect(tmp(), seed=0).chat_share() == 0.0


def test_a_task_whose_architect_rounds_keep_failing_waits_exponentially_longer_and_a_success_resets_it():
    arch = R.Architect(tmp(), seed=0)
    waits = []
    for _ in range(5):
        arch._after_call("t", 0)
        n = 0
        while not arch.due("t"):
            n += 1
        waits.append(n)
    assert waits == [0, 1, 3, 7, 15]
    arch._after_call("t", 2)
    assert arch.due("t") and arch.cool["t"] == dict(misses=0, skip=0)
    assert R.Architect(arch.ws_root, seed=0).cool["t"]["misses"] == 0            # persisted


# --------------------------------------------------------------------------- the verdict
class StubSubject:
    """Just enough of a subject to drive Architect._test: what 'better' and 'worse' mean is decided by the scores the test hands in."""
    kind = "program"
    n_pairs = 12

    def __init__(self, base, new):
        self.task = type("t", (), dict(name="stub"))()
        self.agent = type("a", (), dict(family="stub"))()
        self._base, self._new, self.adopted = np.asarray(base, float), np.asarray(new, float), []

    def src(self):
        return "def policy(obs):\n    return 0.0\n"

    def realise(self, h, rng):
        return "def policy(obs):\n    return 1.0\n"

    def scores(self, src):
        return self._new if "1.0" in src else self._base

    def adopt(self, src):
        self.adopted.append(src)


def _verdict(base, new, alpha_each=0.05):
    arch = R.Architect(tmp(), seed=0)
    sub = StubSubject(base, new)
    rep = dict(tried=0, accepted=0, hypotheses=[])
    arch._test(sub, R.Hypothesis("program", dict(mutator="const", n=1)), alpha_each, rep)
    return rep, sub, arch


def test_a_hypothesis_is_kept_only_when_it_wins_the_paired_test_a_loss_or_a_lucky_pair_is_not_kept():
    base = np.linspace(0.2, 0.5, 12)
    rep, sub, arch = _verdict(base, base + 0.15)
    assert rep["accepted"] == 1 and sub.adopted and rep["hypotheses"][0]["p"] < 0.01
    rep, sub, _ = _verdict(base, base - 0.15)
    assert rep["accepted"] == 0 and not sub.adopted and rep["tried"] == 1
    lucky = base.copy()
    lucky[3] += 3.0                                                              # one pair went very well, the rest did not move or got worse
    lucky[[0, 1, 2, 4, 5]] -= 0.02
    rep, sub, _ = _verdict(base, lucky)
    assert rep["accepted"] == 0 and not sub.adopted
    rep, sub, _ = _verdict(base, base + 0.15, alpha_each=0.0001)                 # the shared significance level is what is tested against
    assert rep["accepted"] == 0 and "not significant" in rep["hypotheses"][0]["reason"]


def test_every_test_is_written_to_the_ledger_with_the_proposers_name_and_the_arms_learn_from_it():
    base = np.linspace(0.2, 0.5, 12)
    rep, sub, arch = _verdict(base, base + 0.15)
    led = arch.ledger()
    assert len(led) == 1 and led[0]["accepted"] and led[0]["source"] == "heuristic" and led[0]["key"] == "program/const" and led[0]["counted"]
    assert arch.arms.counts("program/const") == (1, 1) and arch.arms.counts("source/heuristic") == (1, 1)
    again = R.Architect(arch.ws_root, seed=0)
    assert again.arms.counts("program/const") == (1, 1) and len(again.ledger()) == 1
    assert "1 hypotheses tested, 1 kept" in again.report()


def test_too_few_pairs_to_ever_reach_the_significance_level_means_nothing_is_tested_instead_of_a_guess():
    arch = R.Architect(tmp(), seed=0)
    assert arch._max_tests(4) == 0 and arch._max_tests(8) == 6 and arch._max_tests(16) > 1000 and arch._max_tests(40) > 100
    task = _regression_task()
    ag = _formula_agent(task, "def predict(x):\n    return x[:, 0] * 0.5\n")
    rep = arch.improve(task, ag, rounds=2, per_round=2, n_pairs=4)
    assert rep["tried"] == 0 or rep["pairs"] >= 8
    assert arch.improve(task, ag, rounds=0, per_round=2)["tried"] == 0


# --------------------------------------------------------------------------- end to end
def test_the_architect_improves_a_program_agent_with_a_validated_edit_and_the_agent_keeps_it_through_save_and_load():
    task = _regression_task()
    ag = _formula_agent(task, "def predict(x):\n    return x[:, 0] * 0.5\n")
    before = ag.val_score(task)[0]
    ws = tmp()
    chat = R.ChatProposer(FakeEngine('HYPOTHESIS {"kind": "program", "mutator": "const", "n": 1}'))
    arch = R.Architect(ws, seed=1, chat=chat)
    arch.arms.d["program/const"] = [30, 31]                                      # the heuristic agrees: constants first
    rep = arch.improve(task, ag, rounds=2, per_round=2, tune_steps=24)
    assert rep["accepted"] >= 1, rep
    assert rep["agent"] is ag and rep["skill_after"] > before + 0.3
    acc = [h for h in rep["hypotheses"] if h["accepted"]]
    assert all(h["p"] < rep["alpha_each"] and h["gain"] > 0 for h in acc)
    assert ag.val_score(task)[0] > before + 0.3
    d = tmp()
    ag.save(d)
    text = open(os.path.join(d, "program.py")).read()
    back = F.FAMILIES["program_formula"].load(d, task, ag.genome)
    assert A.normalize(back.best_src()) == A.normalize(ag.best_src()) and abs(back.val_score(task)[0] - ag.val_score(task)[0]) < 1e-9
    assert "def predict" in text


def test_a_validated_program_stays_the_answer_until_the_search_itself_finds_something_better():
    task = _regression_task()
    ag = _formula_agent(task, "def predict(x):\n    return x[:, 0] * 0.0\n")
    pinned = ag.best_src()
    ag.train(task, 0.02, np.random.default_rng(0))                               # 40 evaluations of the island search
    assert ag.pe.best.score > ag.pe.seen[A.normalize(pinned)] + 1e-9
    assert A.normalize(ag.best_src()) != A.normalize(pinned)                     # beaten on the selection score: the pin is released
    ag2 = _formula_agent(task, "def predict(x):\n    return x[:, 0] * 3.0\n")
    ag2.train(task, 0.02, np.random.default_rng(0))
    assert A.normalize(ag2.best_src()) == A.normalize("def predict(x):\n    return x[:, 0] * 3.0\n")


def test_the_architect_never_leaves_an_agent_worse_on_the_data_it_was_judged_on():
    task = _regression_task()
    ag = _formula_agent(task, "def predict(x):\n    return x[:, 0] * 3.0\n")
    before = ag.val_score(task)[0]
    arch = R.Architect(tmp(), seed=2)
    rep = arch.improve(task, ag, rounds=2, per_round=3, tune_steps=8)
    assert rep["skill_after"] >= before - 1e-9, rep
    for h in rep["hypotheses"]:
        assert not h["accepted"] or (h["p"] < rep["alpha_each"] and h["gain"] > 0)


def test_the_architect_improves_an_mlp_by_a_genome_change_judged_on_paired_seeds_against_a_control():
    r = np.random.default_rng(0)
    X = r.normal(size=(300, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)                                    # XOR: a 2-unit network cannot do it well, an 8-unit one can
    task = T.from_arrays("xor", X, y, "classification", seed=0)
    genome = dict(hidden=4, depth=1, lr=0.03, wd=1e-5, batch=64)
    ag = F.FAMILIES["mlp"].build(task, genome, 0)
    ag.train(task, 2.0, np.random.default_rng(0))
    chat = R.ChatProposer(FakeEngine('HYPOTHESIS {"kind": "genome", "family": "mlp", "changes": {"hidden": 32, "depth": 2}}'))
    arch = R.Architect(tmp(), seed=0, chat=chat)
    arch.chat_share = lambda: 1.0
    rep = arch.improve(task, ag, rounds=1, per_round=1, n_pairs=8, units=2.0)
    assert rep["tried"] == 1 and rep["pairs"] == 8
    h = rep["hypotheses"][0]
    assert h["source"] == "chat" and h["key"] == "genome/mlp/depth,hidden"
    if h["accepted"]:
        assert rep["agent"] is not ag and rep["agent"].genome["hidden"] == 32 and h["p"] < 0.05
        assert rep["skill_after"] >= rep["skill_before"] - 0.05
    else:
        assert rep["agent"] is ag
    led = arch.ledger()[-1]
    assert led["pairs"] == 8 and led["counted"] and led["state"]["genome"] is not None


def test_a_proposal_for_a_family_or_agent_the_architect_cannot_change_says_so():
    task = _regression_task()
    arch = R.Architect(tmp(), seed=0)

    class Odd:
        family = "unheard_of"
    rep = arch.improve(task, Odd(), rounds=1, per_round=1)
    assert rep["tried"] == 0 and "no hypothesis applies" in rep["note"]


# --------------------------------------------------------------------------- the loop back into the chat model
def test_validated_hypotheses_are_exported_as_training_text_in_the_prompt_answer_format_and_failures_only_as_lessons():
    ws = tmp()
    arch = R.Architect(ws, seed=0)
    base = np.linspace(0.2, 0.5, 12)
    sub = StubSubject(base, base + 0.15)
    rep = dict(tried=0, accepted=0, hypotheses=[])
    arch._test(sub, R.Hypothesis("program", dict(mutator="neuron", n=1), source="chat"), 0.05, rep)
    sub2 = StubSubject(base, base - 0.2)
    arch._test(sub2, R.Hypothesis("program", dict(mutator="gate", n=1)), 0.05, rep)
    out = tmp()
    info = arch.export_for_chat(out)
    assert info == dict(documents=1, lessons=1)
    docs = sorted(os.listdir(out))
    assert docs == ["architect-0000.txt", "architect-lessons.txt"]
    doc = open(os.path.join(out, docs[0])).read()
    assert doc.startswith("[architect]") and '"mutator": "neuron"' in doc and "[validated]" in doc and "gate" not in doc
    good, bad = R.parse(doc, spaces=SPACES)
    assert len(good) == 1 and good[0].params["mutator"] == "neuron" and not bad
    lessons = open(os.path.join(out, "architect-lessons.txt")).read()
    assert "program/gate" in lessons and "HYPOTHESIS" not in lessons                   # never in the answer position
    assert R.parse(lessons, spaces=SPACES)[0] == []


def test_the_documents_land_where_the_chat_trainer_reads_and_the_prompt_in_them_is_the_prompt_the_chat_proposer_asks():
    ws = tmp()
    arch = R.Architect(ws, seed=0)
    base = np.linspace(0.2, 0.5, 12)
    arch._test(StubSubject(base, base + 0.15), R.Hypothesis("program", dict(mutator="mgate", n=1)), 0.05, dict(tried=0, accepted=0, hypotheses=[]))
    out = os.path.join(tmp(), "_architect")
    arch.export_for_chat(out)
    doc = open(os.path.join(out, "architect-0000.txt")).read()
    state = dict(task="stub", kind="?", family="stub", skill=float("nan"), src="def policy(obs):\n    return 0.0\n", genome=None, space=None)
    assert doc.startswith(R.build_prompt(state, [], 1))


def test_a_chat_model_that_never_returns_a_good_line_is_still_a_working_system_the_heuristic_carries_the_load():
    task = _regression_task()
    ag = _formula_agent(task, "def predict(x):\n    return x[:, 0] * 0.5\n")
    chat = R.ChatProposer(FakeEngine("I think you should try turning it off and on again."))
    arch = R.Architect(tmp(), seed=3, chat=chat)
    arch.arms.d["program/const"] = [30, 31]
    rep = arch.improve(task, ag, rounds=1, per_round=3, tune_steps=24)
    assert rep["tried"] >= 1 and all(h["source"] == "heuristic" for h in rep["hypotheses"])
    assert chat.invalid == 0 and chat.asked >= 0
