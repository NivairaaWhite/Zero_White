"""organism.py - the heartbeat: observe -> decide -> act -> learn, with nobody at the keyboard.

Until now every capability waited for a command. The organism wires them into one loop that decides *for itself* what to do next,
from what it measures, and writes down why:

  OBSERVE   look at the inbox (everything is digested by `omnivore`: any file, however damaged, becomes a task or a recorded reason),
            at the machine (free memory, the cached `substrate` probe) and at its own history.
  DECIDE    one explainable rule, in this order:
              1. a task that has never been attempted is attempted (curiosity: the cheapest way to learn what is learnable)
              2. otherwise the task whose *learning progress per second* is highest (an exponentially weighted gain in validation skill
                 divided by what an attempt costs on THIS machine), with a budget that doubles after every attempt that gained nothing
              3. a task that gained nothing `patience` times in a row has plateaued and is left alone
              4. when every task has plateaued the system is stuck, which is the moment to change how it searches: it picks one
                 self-improvement (rewrite a search rule under the test-gated hook machinery; invent new environments with POET) by
                 Thompson sampling over what has worked before. A successful rewrite re-opens every plateaued task.
  ACT       run it, inside try/except: a failure is logged, makes that task / arm back off exponentially, and never stops the loop.
  LEARN     update the per-task gain, the per-unit cost model and the per-arm success counts; persist (atomically) after every step.

Since 6.1 the chat model is one of the tasks (see chat_task.py): one scheduler weighs a table, an environment and a language model by the
same measure - learning progress per second - and an inbox file of prose can teach both models.

What it does NOT do: edit its own source files (that stays behind `selfmod patch` and its five gates; opt in explicitly), run your
code (`allow_code=False` unless you say so), touch your files (digestion only reads), or run two at once on one workspace (a lock).

Everything it decides is in `<workspace>/organism/log.jsonl`; its memory is `<workspace>/organism/state.json`.
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from . import chat_task, fileio, forecast, metric, substrate
from .chat import hardware as hw

ARMS = ("hooks:select_parent", "hooks:sigma_update", "poet")


@dataclass
class Settings:
    inbox: list = field(default_factory=list)       # folders / files to watch and digest every cycle
    minutes: float | None = None                    # stop after this long
    max_cycles: int | None = None                   # ... or after this many decisions
    forever: bool = False                           # keep watching when there is nothing to do
    self_modify: str = "off"                        # off | hooks   (source patching is a separate, explicit command)
    allow_code: bool = False                        # run environment files found in the inbox (your code)
    pool: str = "notes"                             # name of the shared text corpus
    probe_budget: float = 0.25                      # the budget of a task's first attempt: small, to learn what an attempt costs
    standard_budget: float = 1.0                    # the budget of its second attempt (what `solve` uses by default), if it is affordable
    candidates: int = 6                             # new agents proposed per attempt
    rungs: int = 3
    min_gain: float = 0.005                         # a validation-skill gain below this counts as "nothing happened"
    patience: int = 3                               # attempts without gain before a task counts as plateaued
    max_budget: float = 32.0
    max_action_minutes: float = 10.0                # no single attempt is planned to take longer than this
    time_margin: float = 1.5                        # predicted cost is multiplied by this before deciding whether an attempt fits (the
                                                    # cost model is extrapolated from a small probe and is optimistic: see docs/ORGANISM.md)
    forage_hosts: tuple = ()                        # 6.4: hosts the organism may read from when every task has plateaued (empty = it never touches the network)
    forage_wishlist: str = ""                       # ... a file of URLs, one per line (a person's list: the organism chooses WHEN, not what)
    forage_max_urls: int = 5
    forage_cooldown_s: float = 3600.0
    forage_min_novelty: float = 0.4                 # keep a page only if this share of its word 3-grams is new to the system (forage/curiosity.py)
    forage_allow_local: bool = False                # for tests against a local server
    sleep: bool = False                             # 6.4: when every task has plateaued, consolidate the chat model's pending feedback in a long, gentle, gated pass (chat/sleep.py)
    sleep_cooldown_s: float = 900.0                 # ... at most this often
    learned_sniffer: str = "rescue"                 # 6.4: 'off' | 'rescue' (the byte model may only rescue files the rules gave up on) | 'arbitrate' (may overrule them)
    metric: str = "skill"                           # 6.4: 'skill' (per-task, the default) or 'bits': schedule by bits of information gained per second (metric.py)
    bits_per_skill: float = 50.0                    # ... the exchange rate for tasks with no information model (control tasks)
    calibrate: bool = True                          # 6.3: replace the fixed margin by one measured from this task's own past overruns (forecast.py)
    coverage: float = 0.9                           # ... the share of attempts that should fit inside the calibrated prediction
    min_free_gb: float = 0.25                       # below this much free memory the organism rests instead of working
    doctor_every: int = 10                          # run the self-check every N decisions (and after every self-modification)
    hook_evals: int = 200
    poll_seconds: float = 2.0
    seed: int = 0
    quiet: bool = False
    # ---- the chat model as a task (budgets are MINUTES of training, not the unit-less multiplier of the general model)
    chat: str = "auto"                              # off | auto (use <workspace>/chat if it exists) | on (create it) | PATH to a chat workspace
    chat_set: list = field(default_factory=list)    # KEY=VALUE overrides for the chat config, e.g. ["train.lr=0.002"]
    chat_feed: bool = True                          # mirror prose digested from the inbox into the chat data folder
    chat_first_minutes: float = 10.0                # a first model's budget: it also decides the model's size, so it is not a cheap probe
    chat_probe_minutes: float = 2.0                 # first attempt on an existing model
    chat_standard_minutes: float = 10.0             # second attempt, if affordable
    chat_max_minutes: float = 120.0                 # no chat attempt is planned to run longer than this (max_action_minutes is for the general model)
    chat_min_minutes: float = 0.25                  # an attempt shorter than this is not worth its overhead
    chat_overhead_s: float = 15.0                   # assumed ingest / tokenise / save time until the first attempt measures it


def _ewma(old, new, a=0.5):
    return new if old is None else a * new + (1 - a) * old


class Organism:
    def __init__(self, workspace: str, settings: Settings | None = None, log=print):
        self.ws = os.path.abspath(workspace)
        self.s = settings or Settings()
        self.say = (lambda *a: None) if self.s.quiet else log
        self.dir = os.path.join(self.ws, "organism")
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, "state.json")
        self.log_path = os.path.join(self.dir, "log.jsonl")
        self.stop_path = os.path.join(self.dir, "STOP")
        self.lock_path = os.path.join(self.dir, "LOCK")
        self.rng = np.random.default_rng(self.s.seed)
        self.state = self._load()
        self.t_start = time.time()
        self.deadline = self.t_start + self.s.minutes * 60 if self.s.minutes else None
        self._gen = None
        self._lib = None
        self._limits = None
        self._machine = None
        self._chat = None
        self._halt = ""
        self._noted_service = False
        self._chat_notes = {}

    # ------------------------------------------------------------------------------------------ persistence
    def _load(self) -> dict:
        fresh = dict(version=1, cycle=0, seconds_alive=0.0, tasks={}, arms={a: dict(tries=0, wins=0, fails=0, until=0) for a in ARMS})
        if not os.path.exists(self.state_path):
            return fresh
        try:
            st = fileio.read_json(self.state_path)
            for k, v in fresh.items():
                st.setdefault(k, v)
            for a in ARMS:
                st["arms"].setdefault(a, dict(tries=0, wins=0, fails=0, until=0))
            return st
        except (ValueError, OSError):
            fileio.replace(self.state_path, f"{self.state_path}.corrupt-{int(time.time())}")      # keep the evidence, start over
            return fresh

    def _save(self):
        fileio.write_json(self.state_path, self.state, indent=1)

    def _event(self, **ev):
        ev["t"] = time.time()
        ev["cycle"] = self.state["cycle"]
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, default=fileio._json_default) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------------------------------ lock
    def _acquire(self):
        if os.path.exists(self.lock_path):
            try:
                pid = int(open(self.lock_path, encoding="utf-8").read().strip() or 0)
            except (ValueError, OSError):
                pid = 0
            if pid and pid != os.getpid() and hw.pid_alive(pid):
                raise RuntimeError(f"another organism (pid {pid}) is already running on this workspace; stop it first "
                                   f"(create {self.stop_path}) or use another --workspace")
        fileio.write_text(self.lock_path, str(os.getpid()))

    def _release(self):
        try:
            if open(self.lock_path, encoding="utf-8").read().strip() == str(os.getpid()):
                os.remove(self.lock_path)
        except OSError:
            pass

    # ------------------------------------------------------------------------------------------ collaborators (lazy)
    @property
    def lib(self):
        if self._lib is None:
            from .omnivore import Library
            self._lib = Library(self.ws)
        return self._lib

    @property
    def limits(self):
        if self._limits is None:
            from .omnivore import default_limits
            self._limits = default_limits(self.ws)
        return self._limits

    def chat_path(self):
        """Where the chat workspace is, or None when the chat model is not part of this organism."""
        c = (self.s.chat or "auto").strip()
        if c.lower() == "off":
            return None
        path = os.path.join(self.ws, "chat") if c.lower() in ("auto", "on") else os.path.abspath(c)
        if c.lower() == "auto" and not os.path.isdir(path):
            return None                                     # auto adopts a chat workspace that exists; it never creates one
        return path

    def chat_task(self):
        path = self.chat_path()
        if path is None:
            return None
        if self._chat is None or self._chat.path != os.path.abspath(path):
            self._chat = chat_task.ChatTask(path, overrides=self.s.chat_set)
        return self._chat

    @staticmethod
    def is_chat(ts: dict) -> bool:
        return ts.get("kind") == chat_task.NAME

    def generalist(self):
        if self._gen is None:
            from .forge.generalist import Generalist
            self._gen = Generalist(self.ws, seed=self.s.seed + self.state["cycle"], verbose=False)
        return self._gen

    # ------------------------------------------------------------------------------------------ OBSERVE
    def observe(self) -> dict:
        obs = dict(new_tasks=[], counts={}, free_gb=round(hw.ram_available_bytes() / 2 ** 30, 2))
        if self.s.inbox:
            from .omnivore import devour
            d = devour(self.s.inbox, self.ws, limits=self.limits, allow_code=self.s.allow_code, pool=self.s.pool, seed=0, learned=self.s.learned_sniffer)
            obs["counts"] = {k: v for k, v in d.counts().items() if k not in ("unchanged", "skipped")}
            if d.healed:
                obs["healed"] = d.healed
            for name in d.tasks:
                ts = self.state["tasks"].get(name)
                if ts is None:
                    self.state["tasks"][name] = self._new_task_state()
                    obs["new_tasks"].append(name)
                else:                                              # the data changed: whatever had plateaued may not have any more
                    ts.update(stale=0, plateaued=False, fails=0, until=0)
                    obs["new_tasks"].append(name + " (updated)")
        for name in self.lib.names():                              # tasks digested by an earlier `devour` are adopted too
            self.state["tasks"].setdefault(name, self._new_task_state())
        for name in [n for n, t in self.state["tasks"].items() if not self.is_chat(t) and self.lib.spec(n) is None]:
            del self.state["tasks"][name]                          # its data was removed from the library
        self._observe_chat(obs)
        return obs

    def _observe_chat(self, obs: dict):
        """Adopt the chat model as a task, feed it what the inbox digested, and re-open it when its data changed."""
        tasks = self.state["tasks"]
        ct = self.chat_task()
        if ct is None:
            tasks.pop(chat_task.NAME, None)                        # switched off: forget it (its workspace is untouched)
            return
        try:
            pid = ct.busy()
            if pid:                                                # someone else is training it: never touch shards or tokens meanwhile
                obs["chat_busy"] = pid
                return
            if self.s.chat_feed and self.s.inbox and ct.feed_prose(self.lib, self.s.pool):
                obs["chat_fed"] = True
            info = ct.sync()
        except Exception as ex:                                    # noqa: BLE001 - a chat workspace problem must not stop the other tasks
            obs["chat_error"] = f"{type(ex).__name__}: {ex}"[:200]
            return
        if info["errors"]:
            obs["chat_errors"] = info["errors"]
        if not info["has_data"]:
            if chat_task.NAME in tasks:
                del tasks[chat_task.NAME]
            obs["chat"] = f"no usable data yet in {ct.layout().data}"
            return
        ts = tasks.get(chat_task.NAME)
        if ts is None:
            snap = ct.snapshot()
            ts = tasks[chat_task.NAME] = self._new_task_state(kind=chat_task.NAME, has_model=snap.get("has_model", False))
            if snap.get("skill") is not None:
                ts["best_skill"] = snap["skill"]                   # an existing model's standing is known before the first attempt
            obs["new_tasks"].append(chat_task.NAME)
        elif info["changed"]:                                      # new or changed data: a plateau of the old data says nothing about it
            ts.update(stale=0, plateaued=False, fails=0, until=0)
            obs["new_tasks"].append(chat_task.NAME + " (data changed)")
        if not self._noted_service and ct.service_running():
            self._noted_service = True
            obs["chat_service"] = ("the chat service is running: its own auto-training queues behind this organism's runs (one training lock); "
                                   "set daemon.autotrain=false if the organism should be the only one scheduling training")

    def _new_task_state(self, kind: str = "general", has_model: bool = False) -> dict:
        budget = self.s.probe_budget
        if kind == chat_task.NAME:                                 # minutes: a first model is a real run (it fixes the model's size)
            budget = self.s.chat_probe_minutes if has_model else self.s.chat_first_minutes
        return dict(kind=kind, attempts=0, stale=0, budget=budget, gain_ewma=None, sec_per_unit=None, best_skill=None, last_gain=None,
                    last_seconds=None, plateaued=False, fails=0, until=0, history=[])

    # ------------------------------------------------------------------------------------------ DECIDE
    def cost(self, ts: dict, budget: float):
        """Predicted seconds for an attempt at `budget` (None until one attempt has been timed). For the chat model the budget is minutes
        the trainer honours, so the prediction exists before the first attempt: budget plus the overhead around it."""
        if self.is_chat(ts):
            return budget * 60.0 + self.chat_overhead(ts)
        if ts.get("sec_per_unit") is None:
            return None
        if self.s.calibrate:                                   # 6.3: a fitted curve once the history can support one (else the linear probe below)
            fitted = forecast.predict(ts.get("cost_obs"), budget * (self.s.candidates + 1))
            if fitted is not None:
                return fitted
        return ts["sec_per_unit"] * budget * (self.s.candidates + 1)

    def pred_kind(self, ts: dict) -> str:
        """'fitted' when cost() answers from the fitted curve, else 'linear' (the first-attempt probe). Only fitted predictions feed the measured pad:
        the linear probe's early, much larger overruns come from a different predictor and would otherwise inflate the pad long after it was replaced."""
        return "fitted" if (self.s.calibrate and forecast.predict(ts.get("cost_obs"), 1.0) is not None) else "linear"

    def pad(self, ts: dict, key: str = "cost_ratios"):
        """(multiplier, basis, n) for a prediction: measured from this task's past overruns when there are enough of them, else the fixed time_margin."""
        fixed = max(1.0, self.s.time_margin)
        if not self.s.calibrate:
            return fixed, "fixed", 0
        return forecast.overrun_margin(ts.get(key), q=min(0.99, max(0.5, self.s.coverage)), fallback=fixed)

    @staticmethod
    def _pad_note(pad: float, basis: str, n: int, q: float) -> str:
        if basis == "calibrated":
            return f"cost estimate x{pad:.2f}: the {q:.0%} overrun measured over {n} attempts"
        return f"cost estimate x{pad:g} for safety"

    def _remaining(self) -> float:
        return float("inf") if self.deadline is None else self.deadline - time.time()

    @staticmethod
    def headroom(ts: dict) -> float:
        """How much better the task could still get. Skill is normalised everywhere so that 1.0 is perfect."""
        return max(0.0, 1.0 - (ts["best_skill"] if ts["best_skill"] is not None else 0.0))

    def expected_gain(self, ts: dict) -> float:
        """What another attempt is expected to add: the measured average gain, or (before anything has been measured) half the headroom;
        never more than the headroom."""
        head = self.headroom(ts)
        eg = ts["gain_ewma"] if ts["gain_ewma"] is not None else 0.5 * head
        return min(eg, head)

    def bits_of(self, ts: dict, skill) -> float:
        return metric.skill_to_bits(ts.get("meta"), 0.0 if skill is None else skill, self.s.bits_per_skill)[0]

    def expected_bits(self, ts: dict) -> float:
        """The bits-denominated twin of expected_gain: measured average bits gained per attempt, or half the bits still to gain."""
        head = max(0.0, self.bits_of(ts, 1.0) - self.bits_of(ts, ts["best_skill"]))
        eg = ts.get("bits_gain_ewma")
        return min(eg if eg is not None else 0.5 * head, head)

    def priority(self, ts: dict) -> float:
        gain = self.expected_bits(ts) if self.s.metric == "bits" else self.expected_gain(ts)
        return gain / max(ts["last_seconds"] or 1.0, 0.5)

    def decide(self, obs: dict) -> dict:
        cyc = self.state["cycle"]
        if obs["free_gb"] < self.s.min_free_gb:
            return dict(action="rest", reason=f"only {obs['free_gb']} GiB of memory is free (below {self.s.min_free_gb}): waiting")
        tasks = self.state["tasks"]
        busy_pid = obs.get("chat_busy")
        usable = {n: t for n, t in tasks.items() if not (busy_pid and self.is_chat(t))}
        live = {n: t for n, t in usable.items() if t["until"] <= cyc}
        fresh = sorted((n for n, t in live.items() if t["attempts"] == 0), key=lambda n: (self.is_chat(live[n]), n))   # the cheap ones first
        open_ = {n: t for n, t in live.items() if t["attempts"] > 0 and not t["plateaued"]}
        cands = [(n, live[n], f"'{n}' has never been attempted (curiosity first)") for n in fresh]
        for n in sorted(open_, key=lambda k: (self.priority(open_[k]), k), reverse=True):   # ties: the greater name, as before
            ts = open_[n]
            measured = "measured" if ts["gain_ewma"] is not None else "assumed: half the headroom"
            budget = f"{ts['budget']:g} min" if self.is_chat(ts) else f"x{ts['budget']:g}"
            gain_txt = (f"{self.expected_bits(ts):.1f} bits ({metric.skill_to_bits(ts.get('meta'), 1.0, self.s.bits_per_skill)[1]})" if self.s.metric == "bits"
                        else f"{self.expected_gain(ts):.4f} skill ({measured})")
            cands.append((n, ts, f"'{n}' has the highest learning progress per second: expected gain {gain_txt} over "
                                 f"{ts['last_seconds'] or 0:.1f}s = {self.priority(ts):.5f}/s (budget {budget})"))
        first_idle = None
        for n, ts, why in cands:                       # the best candidate that FITS the time left: a long attempt must not block a short one
            d = self._solve_decision(n, ts, why)
            if d["action"] == "solve":
                if first_idle is not None:
                    d["reason"] += f"  (not '{first_idle[0]}': {first_idle[1]})"
                return d
            first_idle = first_idle or (n, d["reason"])
        if first_idle is not None:
            return dict(action="idle", reason=first_idle[1])
        n_plateau = sum(t["plateaued"] for t in tasks.values())
        waiting = [n for n, t in usable.items() if t["until"] > cyc and not t["plateaued"]]
        known = [t for t in tasks.values() if self.is_chat(t)]
        if busy_pid and not (known and all(t["plateaued"] for t in known)):
            # a language model that merely cannot be touched right now (known or not yet adopted) is not a plateau: do not start rewriting
            # search rules, or declare there is nothing to do, because of it
            return dict(action="idle", waiting=True, reason=f"the chat model is being trained by process {busy_pid}; waiting for it to finish")
        if waiting:                                                    # failures back off; waiting is not the same as being stuck
            return dict(action="idle", waiting=True,
                        reason=f"{len(waiting)} task(s) are backing off after a failure ({', '.join(sorted(waiting))[:80]}): waiting to retry")
        if self._forage_due():
            return dict(action="forage", reason=f"{'every one of ' + str(n_plateau) + ' task(s) has plateaued' if n_plateau else 'no task to work on'}: reading up to "
                                                f"{self.s.forage_max_urls} page(s) from the wishlist on {len(self.s.forage_hosts)} allowed host(s), keeping only what is new to me")
        if self.s.sleep and self.chat_path() and time.time() - float(self.state.get("last_sleep") or 0.0) > self.s.sleep_cooldown_s:
            return dict(action="sleep", reason=f"{'every one of ' + str(n_plateau) + ' task(s) has plateaued' if n_plateau else 'no task to work on'}: sleeping, i.e. "
                                               f"consolidating the chat model's pending feedback (replay-heavy, gated on the worst age band of its corpus)")
        if self.s.self_modify == "hooks":
            arm = self._pick_arm()
            if arm:
                return dict(action="self_modify", arm=arm,
                            reason=f"{'every one of ' + str(n_plateau) + ' task(s) has plateaued' if n_plateau else 'no task to work on'}: "
                                   f"changing how it searches ('{arm}' chosen by Thompson sampling over past results)")
        why = (f"all {n_plateau} task(s) have plateaued and self-modification is off" if n_plateau
               else "there is nothing to work on: no tasks yet")
        return dict(action="idle", reason=why)

    def _solve_decision(self, name: str, ts: dict, why: str) -> dict:
        if self.is_chat(ts):
            return self._chat_decision(name, ts, why)
        budget = float(min(ts["budget"], self.s.max_budget))
        cap = min(self.s.max_action_minutes * 60, self._remaining())
        c = self.cost(ts, budget)
        margin, basis, n_pad = self.pad(ts)
        note = ""
        if c is not None and c * margin > cap:
            if self.s.calibrate and forecast.predict(ts.get("cost_obs"), 1.0) is not None:       # a fitted curve: solve cost(b) * pad <= cap
                affordable = forecast.affordable_units(lambda b: self.cost(ts, b), cap, margin, budget)
            else:
                per_unit = ts["sec_per_unit"] * (self.s.candidates + 1) * margin
                affordable = cap / max(per_unit, 1e-9)
            if affordable < 0.05:
                return dict(action="idle", reason=f"'{name}' needs about {c * margin:.0f}s at its smallest budget (with a x{margin:.3g} margin) "
                                                  f"but only {cap:.0f}s remain")
            note = (f"; budget cut from x{budget:g} to x{affordable:.2f} to fit the {cap:.0f}s available "
                    f"({self._pad_note(margin, basis, n_pad, self.s.coverage)})")
            budget = affordable
        return dict(action="solve", task=name, budget=budget, reason=why + note, predicted_s=self.cost(ts, budget),
                    pred_kind=self.pred_kind(ts))

    def chat_overhead(self, ts: dict) -> float:
        measured = ts.get("overhead_s")
        return float(measured if measured is not None else self.s.chat_overhead_s)

    def _chat_decision(self, name: str, ts: dict, why: str) -> dict:
        """A chat attempt's budget is minutes of training, which the trainer honours; only the overhead around it (ingest, tokenise,
        benchmark, save) has to be estimated, and that is the part the safety margin pads."""
        minutes = float(min(ts["budget"], self.s.chat_max_minutes))
        over = self.chat_overhead(ts)
        cap = self._remaining()
        pad, basis, n_pad = self.pad(ts, "over_ratios")
        fit = (cap - over * pad) / 60.0
        note = ""
        if minutes > fit:
            if fit < self.s.chat_min_minutes:
                return dict(action="idle", reason=f"'{name}' needs at least {self.s.chat_min_minutes:g} min of training plus about "
                                                  f"{over * pad:.0f}s of overhead but only {cap:.0f}s remain")
            why_pad = (f"overhead x{pad:.2f}: the {self.s.coverage:.0%} overrun measured over {n_pad} attempts" if basis == "calibrated"
                       else f"overhead x{pad:g} for safety")
            note = f"; budget cut from {minutes:g} to {fit:.1f} minutes to fit the {cap:.0f}s left ({why_pad})"
            minutes = fit
        return dict(action="solve", task=name, budget=minutes, unit="minutes", reason=why + note, predicted_s=self.cost(ts, minutes))

    def _pick_arm(self):
        """Thompson sampling over the self-improvement arms: Beta(wins+1, losses+1) per arm; POET only when an environment task exists."""
        cyc = self.state["cycle"]
        has_env = any((self.lib.spec(n) or {}).get("kind") == "env" for n in self.state["tasks"])
        best, best_v = None, -1.0
        for a in ARMS:
            st = self.state["arms"][a]
            if st["until"] > cyc or (a == "poet" and not has_env):
                continue
            v = float(self.rng.beta(st["wins"] + 1, max(0, st["tries"] - st["wins"]) + 1))
            if v > best_v:
                best, best_v = a, v
        return best

    # ------------------------------------------------------------------------------------------ ACT
    def act(self, dec: dict) -> dict:
        t0 = time.time()
        a = dec["action"]
        if a == "solve":
            if self.is_chat(self.state["tasks"][dec["task"]]):
                return self._act_chat(dec, t0)
            return self._act_solve(dec, t0)
        if a == "self_modify":
            return self._act_self_modify(dec, t0)
        if a == "sleep":
            return self._act_sleep(dec, t0)
        if a == "forage":
            return self._act_forage(dec, t0)
        if a in ("rest", "idle"):
            return dict(ok=True, seconds=0.0)
        raise ValueError(f"unknown action {a}")

    def _solve(self, task_name: str, budget: float) -> dict:
        task = self.lib.load_task(task_name, limits=self.limits, allow_code=self.s.allow_code)
        rec = self.generalist().solve(task, budget=budget, n_new=self.s.candidates, rungs=self.s.rungs)
        return dict(skill=float(rec["skill"]), improved=bool(rec["improved"]), family=rec["family"], test=rec.get("test"),
                    description=str(rec.get("description", ""))[:300], meta=metric.task_meta(task))

    def _act_solve(self, dec: dict, t0: float) -> dict:
        r = self._solve(dec["task"], dec["budget"])
        r.update(ok=True, seconds=time.time() - t0)
        return r

    def _act_chat(self, dec: dict, t0: float) -> dict:
        ct = self.chat_task()
        if ct is None:
            raise RuntimeError("the chat model is switched off (--chat off)")
        r = ct.attempt(dec["budget"], should_stop=lambda: os.path.exists(self.stop_path))     # a STOP file ends the run at a checkpoint
        r.setdefault("seconds", time.time() - t0)
        return r

    def forage_dir(self) -> str:
        return os.path.join(self.ws, "forage")

    def _forage_due(self) -> bool:
        s = self.s
        return bool(s.forage_hosts and s.forage_wishlist and os.path.exists(s.forage_wishlist)
                    and time.time() - float(self.state.get("last_forage") or 0.0) > s.forage_cooldown_s)

    def _act_forage(self, dec: dict, t0: float) -> dict:
        """Opt-in, allow-listed reading (forage/): the wishlist is a person's; the organism decides only WHEN, and keeps only what is new to it."""
        from .forage import Forager, Policy                              # imported here, never at module level: with no forage_hosts this code is unreachable
        from .forage import curiosity as cur
        self.state["last_forage"] = time.time()
        base = self.forage_dir()
        urls, pos = cur.read_wishlist(self.s.forage_wishlist, os.path.join(base, "wishlist.state.json"), self.s.forage_max_urls)
        if not urls:
            return dict(ok=True, seconds=time.time() - t0, forage=dict(urls=0, kept=0, note="the wishlist is exhausted"))
        ref_dirs = [os.path.join(base, "inbox")] + ([os.path.join(self.chat_path(), "data")] if self.chat_path() else [])
        nov = cur.load_reference(ref_dirs)
        pol = Policy(allow_hosts=tuple(self.s.forage_hosts), allow_local=self.s.forage_allow_local, delay=2.0, max_urls=self.s.forage_max_urls)
        f = Forager(pol, os.path.join(base, "staging"), os.path.join(base, "ledger.jsonl"))
        rows = cur.curious_pass(f, urls, nov, os.path.join(base, "inbox"), self.s.forage_min_novelty)
        fileio.write_json(os.path.join(base, "wishlist.state.json"), dict(pos=pos + len(urls)))
        kept = sum(r["kept"] for r in rows)
        return dict(ok=True, seconds=time.time() - t0, forage=dict(urls=len(rows), kept=kept, dropped=[r["url"] for r in rows if not r["kept"]][:5],
                                                                   inbox=os.path.join(base, "inbox")))

    def _act_sleep(self, dec: dict, t0: float) -> dict:
        from .chat import sleep
        self.state["last_sleep"] = time.time()                       # recorded first: a pass that finds nothing to learn must not be retried at once
        ct = self.chat_task()
        cfg = ct.config()
        res = sleep.consolidate(cfg, ct.layout(cfg))
        return dict(ok=True, seconds=time.time() - t0, sleep={k: v for k, v in res.items() if k != "skipped"})

    def _act_self_modify(self, dec: dict, t0: float) -> dict:
        arm = dec["arm"]
        if arm.startswith("hooks:"):
            from .selfmod import hooks
            mgr = hooks.HookManager(self.ws).activate()
            rep = hooks.evolve_hook(mgr, arm.split(":", 1)[1], evals=self.s.hook_evals, seed=self.s.seed + self.state["cycle"], verbose=False)
            won = bool(rep.get("accepted"))
            ref = rep.get("reference", {})
            return dict(ok=True, won=won, seconds=time.time() - t0, live=ref.get("live_heldout"), best=(rep.get("best") or {}).get("heldout"),
                        version=rep.get("version"))
        from .forge.poet import poet
        r = poet(self.ws, rounds=2, seed=self.s.seed + self.state["cycle"], units=0.3, verbose=False)
        n = len(r["environments"])
        return dict(ok=True, won=n >= 2, environments=n, seconds=time.time() - t0)

    # ------------------------------------------------------------------------------------------ LEARN
    def learn(self, dec: dict, res: dict):
        a = dec["action"]
        if a == "solve":
            ts = self.state["tasks"][dec["task"]]
            chat = self.is_chat(ts)
            if not res.get("ok"):
                ts["fails"] += 1
                ts["until"] = self.state["cycle"] + 2 ** min(ts["fails"], 6)
                return
            if res.get("deferred"):                                              # another process holds the chat training lock: not a result
                ts["until"] = self.state["cycle"] + 2
                return
            if res.get("interrupted"):                                           # stopped on request: the checkpoint resumes it; nothing to learn
                return
            ts["fails"] = 0
            skill = res["skill"]
            # the gain is measured against the live model scored on the SAME validation set when the task says so (the chat model's data
            # changes between attempts, and with it the exam); otherwise against the best skill recorded so far
            prev = res["skill_before"] if res.get("skill_before") is not None else ts["best_skill"]
            baseline = ts["attempts"] == 0
            gain = 0.0 if (baseline or prev is None) else (max(0.0, skill - prev) if res["improved"] else 0.0)
            if res.get("meta"):
                ts["meta"] = res["meta"]
            elif chat and res.get("vocab"):
                ts["meta"] = dict(kind="lm", vocab=res["vocab"], n=res.get("tokens") or 0)
            bits_gain = 0.0 if (baseline or prev is None or not res["improved"]) else max(0.0, self.bits_of(ts, skill) - self.bits_of(ts, prev))
            if not baseline:
                ts["bits_gain_ewma"] = _ewma(ts.get("bits_gain_ewma"), bits_gain)
            ts["attempts"] += 1
            ts["last_gain"], ts["last_seconds"] = gain, res["seconds"]
            if chat:
                if res.get("overhead_s") is not None:
                    if ts.get("overhead_s") is not None and res["overhead_s"] > 0:         # how wrong the overhead estimate was: the pad's raw material
                        ts["over_ratios"] = forecast.push(ts.get("over_ratios"), res["overhead_s"] / max(self.chat_overhead(ts), 1e-9))
                    ts["overhead_s"] = _ewma(ts.get("overhead_s"), res["overhead_s"])
                ts["info"] = {k: res.get(k) for k in ("val_nll", "ppl", "params", "tokens", "promoted", "promote_reason", "measured", "reason", "compute")}
            else:
                unit = max(1e-9, dec["budget"] * (self.s.candidates + 1))
                if dec.get("predicted_s") and res["seconds"] > 0 and dec.get("pred_kind") == "fitted":   # what was said vs what happened (6.3)
                    ts["cost_ratios"] = forecast.push(ts.get("cost_ratios"), res["seconds"] / dec["predicted_s"])
                ts["cost_obs"] = forecast.push(ts.get("cost_obs"), [unit, res["seconds"]])
                ts["sec_per_unit"] = _ewma(ts["sec_per_unit"], res["seconds"] / unit)
            ts["best_skill"] = skill
            if baseline:
                # the first attempt is a measurement, not progress: it tells what an attempt costs and where the task stands.
                # The next one uses the standard budget (the cost model trims it if it would not fit the time available).
                ts["first_skill"] = skill
                ts["budget"] = max(ts["budget"], self.s.chat_standard_minutes if chat else self.s.standard_budget)
            else:
                ts["gain_ewma"] = _ewma(ts["gain_ewma"], gain)
                if gain >= self.s.min_gain:
                    ts["stale"] = 0
                else:
                    ts["stale"] += 1
                    ts["budget"] = min(self.s.chat_max_minutes if chat else self.s.max_budget, ts["budget"] * 2)      # nothing happened: spend more next time
            at_ceiling = self.headroom(ts) < self.s.min_gain
            ts["plateaued"] = ts["stale"] >= self.s.patience or at_ceiling
            ts["why_plateaued"] = ("at its ceiling (skill within min_gain of 1.0)" if at_ceiling else
                                   f"{ts['stale']} attempt(s) in a row gained less than {self.s.min_gain}") if ts["plateaued"] else ""
            ts["history"] = (ts["history"] + [dict(cycle=self.state["cycle"], budget=round(dec["budget"], 3), gain=round(gain, 4),
                                                   skill=round(skill, 4), seconds=round(res["seconds"], 2), family=res.get("family"))])[-20:]
        elif a == "self_modify":
            st = self.state["arms"][dec["arm"]]
            if not res.get("ok"):
                st["fails"] += 1
                st["until"] = self.state["cycle"] + 2 ** min(st["fails"], 5)
                return
            st["tries"] += 1
            st["wins"] += int(bool(res.get("won")))
            st["until"] = self.state["cycle"] + 2                               # skip the next decision: never the same arm back to back
            if res.get("won") and dec["arm"].startswith("hooks:"):
                self._gen = None                                                # pick up the new live hook
                for t in self.state["tasks"].values():                          # the search rules changed: plateaus may not hold
                    if t["plateaued"] and not self.is_chat(t):                  # (the hooks steer the general model's search, not the chat trainer)
                        t.update(plateaued=False, stale=max(0, self.s.patience - 1))
                self._doctor("after a self-modification")

    def _doctor(self, why: str):
        try:
            from . import doctor
            found = doctor.run(self.ws, fix=True, quick=True)
            bad = [f for f in found if f.level in ("fail", "warn")]
            fixed = [f for f in found if f.level == "fixed"]
            self._event(kind="doctor", why=why, problems=[f"{f.area}: {f.msg}"[:160] for f in bad[:6]], repaired=[f"{f.area}: {f.msg}"[:160] for f in fixed[:6]])
            if bad or fixed:
                self.say(f"[organism] self-check ({why}): {len(bad)} problem(s), {len(fixed)} repaired")
        except Exception as ex:                                  # noqa: BLE001 - a broken self-check must not stop the loop
            self._event(kind="doctor", why=why, error=f"{type(ex).__name__}: {ex}")

    # ------------------------------------------------------------------------------------------ the loop
    def step(self) -> dict:
        """One heartbeat. Returns {'decision': ..., 'result': ...}; never raises on a failed action."""
        t0 = time.time()
        obs = self.observe()
        notes = {k: v for k, v in obs.items() if k.startswith("chat") and v}
        fresh_notes = bool(notes) and notes != self._chat_notes                  # say a chat note once, not on every idle poll
        self._chat_notes = notes
        if obs["new_tasks"] or obs["counts"] or obs.get("healed") or fresh_notes:
            self._event(kind="observe", **{k: v for k, v in obs.items() if v})
            if obs["new_tasks"]:
                self.say(f"[organism] new / updated task(s): {', '.join(obs['new_tasks'])}")
            for k in ("chat_error", "chat_service"):
                if obs.get(k):
                    self.say(f"[organism] {k.replace('_', ' ')}: {obs[k]}")
        dec = self.decide(obs)
        try:
            res = self.act(dec)
        except Exception as ex:                                  # noqa: BLE001 - a failing action is data, not a crash
            res = dict(ok=False, seconds=time.time() - t0, error=f"{type(ex).__name__}: {ex}",
                       trace=traceback.format_exc(limit=4).strip().splitlines()[-3:])
            self.say(f"[organism] {dec.get('task') or dec.get('arm')} failed: {res['error']}")
        self.learn(dec, res)
        if res.get("interrupted") and not os.path.exists(self.stop_path):
            self._halt = "interrupted by a signal (the chat model's checkpoint resumes next time)"
        if dec["action"] == "idle" and dec.get("waiting"):
            self.state["cycle"] += 1                              # waiting out a back-off must let it expire, or one failing task idles forever
        if dec["action"] not in ("idle", "rest"):
            self.state["cycle"] += 1
            self.state["seconds_alive"] += time.time() - t0
            if self.s.doctor_every and self.state["cycle"] % self.s.doctor_every == 0:
                self._doctor("periodic")
        self._event(kind="decision", decision=dec, result={k: v for k, v in res.items() if k != "description"})
        self._save()
        if dec["action"] == "solve" and res.get("ok") and res.get("deferred"):
            self.say(f"[organism] {dec['task']}: another process is training it; retrying shortly")
        elif dec["action"] == "solve" and res.get("ok") and res.get("interrupted"):
            self.say(f"[organism] {dec['task']}: stopped on request after {res['seconds']:.0f}s (its checkpoint resumes next time)")
        elif dec["action"] == "solve" and res.get("ok"):
            self.say(f"[organism] {dec['task']}: skill {res['skill']:.3f} ({res['family']}) "
                     f"{'improved' if res['improved'] else 'unchanged'} in {res['seconds']:.0f}s  | why: {dec['reason']}")
        elif dec["action"] == "self_modify" and res.get("ok"):
            self.say(f"[organism] {dec['arm']}: {'WON' if res.get('won') else 'no gain'} in {res['seconds']:.0f}s  | why: {dec['reason']}")
        elif dec["action"] in ("idle", "rest"):
            self.say(f"[organism] {dec['action']}: {dec['reason']}")
        return dict(decision=dec, result=res)

    def run(self) -> dict:
        self._acquire()
        try:
            return self._run()
        finally:
            self._release()
            self._save()

    def _run(self) -> dict:
        self._event(kind="start", settings=dict(self.s.__dict__, inbox=list(self.s.inbox)), machine=self._machine_summary())
        self.say(f"[organism] workspace {self.ws}; {self._machine_summary()}")
        self._doctor("at start")
        steps = 0
        idle_rounds = 0
        stop_reason = ""
        while True:
            if self._halt:
                stop_reason = self._halt
                break
            if os.path.exists(self.stop_path):
                stop_reason = f"stop file found ({self.stop_path})"
                try:
                    os.remove(self.stop_path)
                except OSError:
                    pass
                break
            if self.deadline is not None and time.time() >= self.deadline:
                stop_reason = "time budget used"
                break
            if self.s.max_cycles is not None and steps >= self.s.max_cycles:
                stop_reason = "cycle budget used"
                break
            out = self.step()
            kind = out["decision"]["action"]
            if kind in ("idle", "rest"):
                idle_rounds += 1
                bounded = self.s.minutes is not None or self.s.max_cycles is not None or not self.s.forever
                waiting = bool(out["decision"].get("waiting")) and self.deadline is not None      # a back-off ends; the time budget bounds the wait
                if kind == "idle" and bounded and not self.s.forever and idle_rounds >= 1 and not waiting:
                    stop_reason = "nothing worth doing: " + out["decision"]["reason"]
                    break
                time.sleep(min(30.0, self.s.poll_seconds * (1 + idle_rounds // 5)))
            else:
                idle_rounds = 0
                steps += 1
        self._event(kind="stop", reason=stop_reason)
        self.say(f"[organism] stopped: {stop_reason}")
        return dict(stopped=stop_reason, cycles=self.state["cycle"], tasks=self.report())

    def _machine_summary(self) -> str:
        if self._machine is None:
            try:
                p = substrate.cached(self.ws)
                self._machine = f"{p['cpus']} cpu(s), {p['ram_available_gb']} GiB free, {p['gflops']} GFLOP/s"
            except Exception:                                    # noqa: BLE001
                self._machine = "machine not measured"
        return self._machine

    # ------------------------------------------------------------------------------------------ reporting
    def report(self) -> dict:
        return {n: dict(kind=t.get("kind", "general"), attempts=t["attempts"], best_skill=t["best_skill"], plateaued=t["plateaued"], budget=t["budget"],
                        last_gain=t["last_gain"]) for n, t in sorted(self.state["tasks"].items())}


def status(workspace: str, last: int = 8) -> str:
    """A readable account of what the organism knows and did, from its files alone (no workspace is modified)."""
    d = os.path.join(os.path.abspath(workspace), "organism")
    sp = os.path.join(d, "state.json")
    if not os.path.exists(sp):
        return "no organism has run in this workspace yet (`zerowhite organism --workspace W --inbox DIR`)"
    st = fileio.read_json(sp)
    lines = [f"cycles: {st['cycle']}   awake: {st['seconds_alive']:.0f}s", "tasks:"]
    for n, t in sorted(st["tasks"].items()):
        flag = ("plateaued: " + t.get("why_plateaued", "")) if t["plateaued"] else ("backing off" if t["until"] > st["cycle"] else "open")
        skill = f"{t['best_skill']:.3f}" if t["best_skill"] is not None else "  -  "
        is_chat = t.get("kind") == chat_task.NAME
        budget = f"{t['budget']:g} min" if is_chat else f"x{t['budget']:g}"
        lines.append(f"  {n:<26} skill {skill}  attempts {t['attempts']:>2}  next budget {budget:<8} {flag}")
        inf = t.get("info") or {}
        if is_chat and inf.get("ppl"):
            lines.append(f"  {'':<26} language model: perplexity {inf['ppl']:.1f}" + (f", {inf['params']:,} parameters" if inf.get("params") else "")
                         + (f", trained on {inf['compute']}" if inf.get("compute") else "")
                         + ("" if inf.get("measured", True) else " (no validation data: skill is not measurable)"))
    lines.append("self-improvement arms (wins / tries):")
    for a, s in st["arms"].items():
        lines.append(f"  {a:<22} {s['wins']} / {s['tries']}" + (f"   ({s['fails']} failure(s))" if s["fails"] else ""))
    lp = os.path.join(d, "log.jsonl")
    if os.path.exists(lp):
        lines.append("recent decisions:")
        evs = []
        for line in open(lp, "r", encoding="utf-8", errors="replace"):
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("kind") == "decision":
                evs.append(ev)
        for ev in evs[-last:]:
            dec, res = ev["decision"], ev.get("result", {})
            lines.append(f"  #{ev['cycle']:<3} {dec['action']:<11} {dec.get('task') or dec.get('arm') or '':<22} {dec['reason'][:100]}"
                         + ("" if res.get("ok", True) else f"   FAILED: {res.get('error', '')[:60]}"))
    return "\n".join(lines)
