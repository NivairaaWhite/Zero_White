"""Tests for 6.1: the chat model is one of the organism's tasks, and a lone failing task is retried after its back-off."""
import argparse
import json
import math
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import test_v6 as T                                              # noqa: E402  (helpers only: importing names with test_ would re-run them)
from zerowhite import chat_task as CT                            # noqa: E402
from zerowhite import cli_v6                                     # noqa: E402
from zerowhite import organism as O                              # noqa: E402

TINY = ["model.d_model=32", "model.n_layer=2", "model.n_head=2", "model.ctx=64", "tokenizer.vocab_size=320",
        "train.max_steps=120", "train.batch_tokens=512", "train.effective_batch_tokens=512", "train.lr=0.004",
        "train.eval_every=40", "train.eval_tokens=2000", "train.warmup_steps=2", "train.threads=1", "data.workers=1",
        "data.settle_s=0", "data.val_min_docs=10", "data.val_fraction=0.1"]


# ------------------------------------------------------------------------------------------------- a stand-in chat workspace
class FakeChat:
    """What the organism needs from a chat workspace, scripted: no tokenizer, no training."""
    path = "/fake/chat"

    def __init__(self, snapshot=None, has_data=True):
        self.snap = snapshot or dict(has_model=False)
        self.has_data, self.changed, self.pid, self.fed = has_data, False, None, []

    def busy(self):
        return self.pid

    def feed_prose(self, library, pool):
        self.fed.append(pool)
        return False

    def sync(self):
        return dict(changed=self.changed, has_data=self.has_data, summary="", unsettled=0, errors=[])

    def snapshot(self):
        return self.snap

    def service_running(self):
        return None

    def layout(self):
        return types.SimpleNamespace(data="/fake/chat/data")


class ScriptedChat(T.Scripted):
    """The scripted organism, with a (fake) chat task whose attempts follow the same kind of script."""
    fake = None

    def chat_path(self):
        return self.fake.path if self.fake is not None else None

    def chat_task(self):
        return self.fake

    def _act_chat(self, dec, t0):
        seq = self.script[dec["task"]]
        res = seq.pop(0) if len(seq) > 1 else seq[0]
        self.calls.append((dec["task"], round(dec["budget"], 3)))
        if isinstance(res, dict):
            return dict(ok=True, seconds=60.0, **res)
        skill, secs = res
        prev = self.state["tasks"][dec["task"]]["best_skill"]
        return dict(ok=True, skill=skill, skill_before=prev, improved=prev is None or skill > prev + 1e-9, family="chat-transformer",
                    seconds=secs, overhead_s=5.0)


def with_chat(script, fake=None, **kw):
    inbox, ws = T.tmp(), T.tmp()
    for i, name in enumerate(n for n in script if n != "chat"):
        T.put(inbox, f"{name}.csv", T.csv_bytes(100, seed=100 + i))
    settings = dict(inbox=[inbox], quiet=True, probe_budget=0.25, standard_budget=1.0, patience=2, min_gain=0.01, doctor_every=0,
                    chat_first_minutes=10.0, chat_probe_minutes=2.0, chat_standard_minutes=10.0, chat_max_minutes=120.0, poll_seconds=0.0)
    settings.update(kw)
    cls = type("SC", (ScriptedChat,), dict(script={k: list(v) for k, v in script.items()}, calls=[], fake=fake or FakeChat()))
    return cls(ws, O.Settings(**settings)), cls


# ------------------------------------------------------------------------------------------------- skill and result mapping
def test_chat_skill_is_the_share_of_a_uniform_guess_that_the_model_removed():
    v = 500
    assert abs(CT.skill_from_nll(math.log(v), v)) < 1e-9                         # no better than guessing: 0
    assert CT.skill_from_nll(0.0, v) == 1.0                                      # perfect: 1
    assert abs(CT.skill_from_nll(math.log(v) / 2, v) - 0.5) < 1e-9
    assert CT.skill_from_nll(99.0, v) == 0.0 and CT.skill_from_nll(-1.0, v) == 1.0   # clamped, never outside [0, 1]
    assert CT.skill_from_nll(None, v) is None and CT.skill_from_nll(float("nan"), v) is None and CT.skill_from_nll(1.0, 1) is None


def test_chat_result_scores_the_live_model_on_one_validation_set():
    v = 400
    better = CT.summarise(dict(promoted=True, val_nll=3.0, baseline_val=4.0, params=1000, reason="finished"), v, had_model=True)
    assert better["improved"] and better["skill"] > better["skill_before"]
    assert abs(better["skill"] - (1 - 3.0 / math.log(v))) < 1e-9 and abs(better["skill_before"] - (1 - 4.0 / math.log(v))) < 1e-9
    worse = CT.summarise(dict(promoted=False, val_nll=4.5, baseline_val=4.0), v, had_model=True)       # the trainer kept the live model
    assert not worse["improved"] and worse["skill"] == worse["skill_before"]                         # so the skill is the LIVE model's, not the rejected run's
    first = CT.summarise(dict(promoted=True, val_nll=5.0, baseline_val=None), v, had_model=False)
    assert first["improved"] and first["skill_before"] is None and first["skill"] > 0                # a first model is always progress
    stopped = CT.summarise(dict(promoted=False, reason="stopped", val_nll=None, baseline_val=4.0), v, had_model=True)
    assert stopped["interrupted"] and not stopped["improved"]
    blind = CT.summarise(dict(promoted=True, val_nll=None, baseline_val=None), v, had_model=False)     # too little data for a validation split
    assert blind["measured"] is False and blind["skill"] == 0.0 and "no validation data" in blind["description"]


def test_cli_chat_defaults_match_the_settings_they_feed():
    p = argparse.ArgumentParser()
    cli_v6.register(p.add_subparsers(), lambda sp: sp.add_argument("--seed", type=int, default=0))
    a = p.parse_args(["organism"])
    s = O.Settings()
    assert (a.chat, a.chat_minutes, a.chat_first_minutes, a.chat_probe_minutes, a.chat_max_minutes) == \
           (s.chat, s.chat_standard_minutes, s.chat_first_minutes, s.chat_probe_minutes, s.chat_max_minutes)      # the two must not drift apart
    assert a.no_chat_feed is False and s.chat_feed is True and a.chat_set is None


# ------------------------------------------------------------------------------------------------- scheduling
def test_chat_is_adopted_as_a_task_and_a_new_model_gets_a_real_first_budget_not_a_probe():
    org, cls = with_chat({"aaa": [(0.5, 1.0)], "chat": [(0.40, 600.0), (0.45, 600.0)]})
    org.step()
    org.step()
    assert [c[0] for c in cls.calls] == ["aaa", "chat"]                          # the cheap fresh task first, the language model last
    assert cls.calls[0][1] == 0.25 and cls.calls[1][1] == 10.0                  # a table gets a cheap probe; a NEW model gets minutes (it fixes its size)
    ts = org.state["tasks"]["chat"]
    assert ts["kind"] == "chat" and ts["attempts"] == 1 and ts["best_skill"] == 0.40 and ts["gain_ewma"] is None   # first attempt = baseline, not progress
    assert ts["overhead_s"] == 5.0 and org.report()["chat"]["kind"] == "chat"
    assert org.state["tasks"]["aaa"]["kind"] == "general"


def test_an_already_trained_chat_model_is_known_before_the_first_attempt_and_probed_briefly():
    fake = FakeChat(snapshot=dict(has_model=True, skill=0.35, ppl=90.0))
    org, cls = with_chat({"chat": [(0.36, 120.0)]}, fake=fake)
    org.observe()
    ts = org.state["tasks"]["chat"]
    assert ts["best_skill"] == 0.35 and ts["attempts"] == 0 and ts["budget"] == 2.0         # standing known, first attempt short: the model's size is already fixed
    org.step()
    assert cls.calls == [("chat", 2.0)] and org.state["tasks"]["chat"]["gain_ewma"] is None   # still a baseline, even though a skill was already recorded


def test_a_chat_attempt_is_fitted_to_the_time_left_and_a_short_task_is_not_blocked_by_a_long_one():
    org, cls = with_chat({"aaa": [(0.5, 1.0)], "chat": [(0.4, 600.0)]})
    org.step()                                                                     # aaa baseline (cost now known)
    org.deadline = time.time() + 180
    d = org.decide(org.observe())
    assert d["action"] == "solve" and d["task"] == "chat" and d["unit"] == "minutes"
    assert d["budget"] < 10.0 and "cut" in d["reason"] and "minutes" in d["reason"]
    ts = org.state["tasks"]["chat"]
    assert org.cost(ts, 2.0) == 120.0 + org.s.chat_overhead_s                      # known before any attempt: minutes honoured by the trainer + assumed overhead
    ts["overhead_s"] = 40.0
    assert org.cost(ts, 2.0) == 160.0                                              # and the measured overhead replaces the assumption
    ts.pop("overhead_s")
    assert d["predicted_s"] <= 180 and d["predicted_s"] == org.cost(ts, d["budget"])   # the decision uses that one formula
    org.deadline = time.time() + 20                                                # too little for any chat attempt, plenty for a small table
    d = org.decide(org.observe())
    assert d["action"] == "solve" and d["task"] == "aaa" and "not 'chat'" in d["reason"]   # fell through to the best attempt that FITS
    org.state["tasks"]["aaa"]["plateaued"] = True
    d = org.decide(org.observe())
    assert d["action"] == "idle" and "chat" in d["reason"] and "remain" in d["reason"]


def test_a_chat_attempt_that_gains_nothing_doubles_its_minutes_up_to_its_own_maximum():
    org, cls = with_chat({"chat": [(0.50, 60.0), (0.50, 60.0), (0.50, 60.0), (0.50, 60.0)]}, patience=3, chat_max_minutes=30.0)
    for _ in range(4):
        org.step()
    assert [c[1] for c in cls.calls] == [10.0, 10.0, 20.0, 30.0]                 # baseline, then 10 -> 20 -> 40 capped at 30
    ts = org.state["tasks"]["chat"]
    assert ts["plateaued"] and ts["stale"] == 3 and ts["budget"] == 30.0         # the stored budget never exceeds the maximum either
    assert org.decide(org.observe())["action"] in ("idle", "self_modify")      # a plateaued language model is left alone


def test_gain_is_measured_against_the_same_validation_set_not_an_older_best():
    org, _ = with_chat({"chat": [(0.5, 60.0)]})
    org.observe()
    ts = org.state["tasks"]["chat"]
    ts.update(attempts=1, best_skill=0.60, first_skill=0.60)                       # recorded on the old validation set
    dec = dict(action="solve", task="chat", budget=10.0, reason="t")
    org.learn(dec, dict(ok=True, skill=0.35, skill_before=0.30, improved=True, family="chat-transformer", seconds=600.0, overhead_s=9.0))
    assert abs(ts["last_gain"] - 0.05) < 1e-9 and ts["stale"] == 0 and ts["best_skill"] == 0.35   # new data lowered the bar; the model still gained 0.05 on it
    assert ts["overhead_s"] == 9.0 and ts["attempts"] == 2


def test_deferred_and_interrupted_chat_attempts_are_not_results():
    org, _ = with_chat({"chat": [(0.5, 60.0)]})
    org.observe()
    ts = org.state["tasks"]["chat"]
    ts.update(attempts=2, best_skill=0.5, stale=1, budget=10.0)
    dec = dict(action="solve", task="chat", budget=10.0, reason="t")
    org.learn(dec, dict(ok=True, deferred=True, seconds=0.1))
    assert ts["attempts"] == 2 and ts["stale"] == 1 and ts["budget"] == 10.0 and ts["until"] == org.state["cycle"] + 2   # try again shortly, no penalty
    org.learn(dec, dict(ok=True, interrupted=True, skill=0.0, improved=False, seconds=30.0))
    assert ts["attempts"] == 2 and ts["stale"] == 1 and ts["budget"] == 10.0                                          # cut short on request: nothing to learn


def test_an_interrupted_chat_attempt_ends_the_run_cleanly():
    org, cls = with_chat({"chat": [dict(skill=0.0, skill_before=None, improved=False, interrupted=True, family="chat-transformer")]},
                         max_cycles=5)
    out = org.run()
    assert "interrupted" in out["stopped"] and len(cls.calls) == 1               # Ctrl-C inside the trainer must not be swallowed by the loop


def test_a_chat_model_being_trained_elsewhere_is_waited_for_not_trained_twice():
    fake = FakeChat()
    fake.pid = 4242
    org, cls = with_chat({"aaa": [(0.5, 1.0), (0.6, 1.0)], "chat": [(0.4, 600.0)]}, fake=fake)
    org.step()
    org.step()
    assert {c[0] for c in cls.calls} == {"aaa"}                                    # the table carries on; the language model is not touched
    org.state["tasks"]["aaa"]["plateaued"] = True
    d = org.decide(org.observe())
    assert d["action"] == "idle" and d["waiting"] and "4242" in d["reason"]
    c0 = org.state["cycle"]
    org.step()
    assert org.state["cycle"] == c0 + 1                                            # waiting advances time, so a back-off can expire
    fake.pid = None
    d = org.decide(org.observe())
    assert d["action"] == "solve" and d["task"] == "chat"                         # free now: adopted and scheduled
    org.step()                                                                     # (one real attempt, so the task is known and open)
    fake.pid = 4242                                                                # busy again, now that it is a known task
    org.state["tasks"]["chat"]["plateaued"] = False
    d = org.decide(org.observe())
    assert d["action"] == "idle" and d["waiting"]
    org.state["tasks"]["chat"]["plateaued"] = True                                 # but a plateaued chat task being busy changes nothing
    assert not org.decide(org.observe()).get("waiting")


def test_new_chat_data_reopens_a_plateaued_chat_task_and_a_hook_win_does_not():
    fake = FakeChat()
    org, _ = with_chat({"chat": [(0.5, 60.0)]}, fake=fake)
    org.observe()
    ts = org.state["tasks"]["chat"]
    ts.update(attempts=4, stale=3, plateaued=True, why_plateaued="3 attempt(s) in a row gained less than 0.01", fails=2, until=99)
    org.learn(dict(action="self_modify", arm="hooks:select_parent"), dict(ok=True, won=True, seconds=1.0))
    assert ts["plateaued"]                                                         # hooks steer the general model's search; they say nothing about the chat trainer
    fake.changed = True
    obs = org.observe()
    assert not ts["plateaued"] and ts["stale"] == 0 and ts["until"] == 0 and ts["fails"] == 0
    assert any("data changed" in n for n in obs["new_tasks"])


def test_chat_task_is_forgotten_when_it_has_no_data_or_is_switched_off():
    fake = FakeChat(has_data=False)
    org, _ = with_chat({"chat": [(0.5, 60.0)]}, fake=fake)
    obs = org.observe()
    assert "chat" not in org.state["tasks"] and "no usable data" in obs["chat"]
    fake.has_data = True
    org.observe()
    assert "chat" in org.state["tasks"]
    org2 = O.Organism(T.tmp(), O.Settings(chat="off", quiet=True))
    org2.state["tasks"]["chat"] = org2._new_task_state(kind="chat")
    org2.observe()
    assert "chat" not in org2.state["tasks"] and org2.chat_task() is None
    org3 = O.Organism(T.tmp(), O.Settings(chat="auto", quiet=True))               # auto never invents a chat workspace
    assert org3.chat_path() is None
    assert O.Organism(T.tmp(), O.Settings(chat="/some/where", quiet=True)).chat_path() == os.path.abspath("/some/where")


def test_a_lone_failing_task_is_retried_after_its_back_off_instead_of_idling_forever():
    org, _ = T.scripted({"only": [(0.5, 1.0)]}, forever=True, poll_seconds=0.0)
    tries = []

    def boom(dec, t0):
        tries.append(org.state["cycle"])
        raise RuntimeError("boom")
    org._act_solve = boom
    outs = [org.step()["decision"] for _ in range(12)]
    assert len(tries) >= 3 and tries[1] - tries[0] >= 2 and tries[2] - tries[1] > tries[1] - tries[0]   # retried, with the gaps growing (2, 4, ...)
    assert any(d["action"] == "idle" and d.get("waiting") and "backing off" in d["reason"] for d in outs)  # and the wait says what it is
    t_idle = [d for d in outs if d["action"] == "idle"]
    assert all("no tasks yet" not in d["reason"] for d in t_idle)                  # the old message ("nothing to work on") was simply wrong


# ------------------------------------------------------------------------------------------------- the real thing (tiny models, a few seconds each)
def _prose(n, seed):
    import random
    r = random.Random(seed)
    subj, verb = ["the cat", "a farmer", "my sister", "the old teacher"], ["watches", "carries", "remembers", "paints"]
    obj = ["a quiet garden", "the red lantern", "an empty road", "three wooden boats"]
    return "\n\n".join(" ".join(f"{r.choice(subj).capitalize()} {r.choice(verb)} {r.choice(obj)}." for _ in range(8)) for _ in range(n))


def _chat_ws(root, docs=100):
    path = os.path.join(root, "chat")
    os.makedirs(os.path.join(path, "data"), exist_ok=True)
    with open(os.path.join(path, "data", "book.txt"), "w", encoding="utf-8") as f:
        f.write(_prose(docs, 1))
    return path


def test_real_a_file_in_the_inbox_teaches_both_models_and_the_scheduler_trains_the_language_model():
    ws, inbox = T.tmp(), T.tmp()
    chat = _chat_ws(ws)
    T.put(inbox, "diary.txt", _prose(40, 2).encode())
    s = O.Settings(inbox=[inbox], max_cycles=3, quiet=True, chat="auto", chat_set=TINY, chat_first_minutes=0.4, chat_standard_minutes=0.4,
                   chat_probe_minutes=0.2, poll_seconds=0.05, doctor_every=0)
    org = O.Organism(ws, s)
    out = org.run()
    ts = org.state["tasks"]
    assert "chat" in ts and ts["chat"]["kind"] == "chat" and ts["chat"]["attempts"] >= 1 and ts["chat"]["best_skill"] > 0.1
    assert "corpus-notes" in ts                                                    # the same inbox file also reached the general model
    mirror = os.path.join(chat, "data", "_omnivore", "corpus-notes.txt")
    assert os.path.exists(mirror)
    assert "farmer" in open(mirror, encoding="utf-8").read()                       # the diary's own words, mirrored byte for byte
    snap = CT.ChatTask(chat, TINY).snapshot()
    assert snap["has_model"] and snap["params"] > 0 and 0.0 < snap["skill"] <= 1.0
    from zerowhite.chat.corpus.store import Ingestor
    ct = CT.ChatTask(chat, TINY)
    names = [getattr(x, "name", getattr(x, "shard", str(x))) for x in Ingestor(ct.config(), ct.layout()).shards()]
    assert any("omnivore" in n for n in names) and any("book" in n for n in names)        # the prose became chat training data
    m0 = os.path.getmtime(mirror)
    time.sleep(0.05)
    assert ct.feed_prose(org.lib, "notes") is False and os.path.getmtime(mirror) == m0       # unchanged text is never rewritten
    txt = O.status(ws)
    assert "chat" in txt and "language model: perplexity" in txt and "min" in txt


def test_real_a_stop_request_ends_a_chat_attempt_at_a_checkpoint_and_the_next_one_resumes():
    ws = T.tmp()
    chat = _chat_ws(ws)
    ct = CT.ChatTask(chat, TINY)
    r1 = ct.attempt(0.4, should_stop=lambda: True)
    assert r1["ok"] and r1["interrupted"] and not r1["promoted"] and not ct.snapshot()["has_model"]      # nothing half-trained went live
    r2 = ct.attempt(0.4)
    assert r2["ok"] and not r2["interrupted"] and r2["promoted"] and ct.snapshot()["has_model"]          # and the run carried on to a real model
    assert r2["overhead_s"] >= 0 and r2["skill"] > 0.1


def test_real_a_chat_model_trained_by_another_process_is_deferred_not_corrupted():
    ws = T.tmp()
    chat = _chat_ws(ws)
    ct = CT.ChatTask(chat, TINY)
    lock = ct.layout().train_lock
    with open(lock, "w", encoding="utf-8") as f:
        f.write(f"{os.getppid()} {time.time():.0f}\n")                                # a live process that is not us holds the training lock
    try:
        assert ct.busy() == os.getppid()
        r = ct.attempt(0.3)
        assert r["ok"] and r["deferred"] and not ct.snapshot()["has_model"]
    finally:
        os.remove(lock)
    assert ct.busy() is None
