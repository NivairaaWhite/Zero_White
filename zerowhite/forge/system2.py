"""system2.py - a language model proposes programs, the forge checks them: 'System 2 writes, System 1 verifies'.

The chat model is good at producing plausible text, bad at guaranteeing it is correct; the forge is the opposite: it can score any candidate program on the real task, cheaply
and exactly. This module is the loop between them:

  proposals   any source of candidate policy functions: the chat model through `chat_proposer(generate)` (it parses code out of free text), a person, a file;
  validate    each candidate must pass the sandbox's AST safety rules (selfmod/safety.py) AND compile to a function: nothing else is ever executed;
  race        successive halving on the real task: every survivor plays a few episodes, the better half plays more, and so on; the ranking uses the task's own skill measure;
  seed        the winners are admitted into a program agent's islands (`seed_agent`) as starting points of its evolutionary search.

HONEST STATUS: the loop is built and tested with scripted proposers. Nothing here shows that THIS project's chat models, which are small and trained on little text, propose
useful programs; a random or poor proposer costs a few episodes and is simply out-raced (the tests include one). What it adds is a safe place for a better proposer to plug in.
"""
from __future__ import annotations

import math
import re
from typing import Callable, List, Sequence

import numpy as np

from ..selfmod import astmut as A, safety


def validate(src: str, name: str = "policy"):
    """The compiled function for `src`, or (None, reason). Nothing is executed beyond compilation, and only after the safety rules accept the source."""
    try:
        bad = safety.violations(src)
        if bad:
            return None, "unsafe: " + "; ".join(bad[:3])
        fn = A.compile_program(src, name)
    except Exception as ex:                                             # noqa: BLE001 - any failure to build the function is just a rejected proposal
        return None, f"{type(ex).__name__}: {ex}"
    return fn, ""


def race(task, proposals: Sequence[str], episodes: Sequence[int] = (2, 4, 8), keep: float = 0.5, name: str = "policy", seed0: int = 40000) -> List[dict]:
    """Successive halving over the valid proposals. Returns [{src, skill, episodes, valid, reason}] best first (invalid ones last, with their reason)."""
    from .families import _to_action
    rows = [dict(src=s, valid=False, skill=None, episodes=0, reason="") for s in proposals]
    live = []
    for r in rows:
        fn, why = validate(r["src"], name)
        r["valid"], r["reason"] = fn is not None, why
        if fn is not None:
            live.append((r, fn, []))
    for rnd, n_ep in enumerate(episodes):
        for r, fn, rets in live:
            act = lambda ob, fn=fn: _to_action(fn(ob), task.n_actions)              # noqa: E731
            have = len(rets)
            for e in range(have, n_ep):
                try:
                    rets.append(task.rollout(act, seed0 + e)[0])
                except Exception:                                               # noqa: BLE001 - a program that crashes on the task scores as the worst thing seen
                    rets.append(-1e9)
            r["skill"], r["episodes"] = float(task.skill(float(np.mean(rets)))), len(rets)
        live.sort(key=lambda t: -t[0]["skill"])
        if rnd < len(episodes) - 1:
            live = live[:max(1, int(math.ceil(len(live) * keep)))]
    rows.sort(key=lambda r: (not r["valid"], -(r["skill"] if r["skill"] is not None else -1e9)))
    return rows


def seed_agent(agent, sources: Sequence[str]) -> int:
    """Admit validated sources into every island of a ProgramPolicyAgent's evolver. Returns how many were admitted."""
    n = 0
    for s in sources:
        if validate(s)[0] is not None:
            for isl in agent.pe.islands:
                agent.pe._admit(isl, s)
            n += 1
    return n


_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.S)
_DEF = re.compile(r"(def \w+\(.*?\):\n(?:[ \t]+.*\n?)+)")


def chat_proposer(generate: Callable[[str], str], prompt_template: str = "Write a Python function `def policy(obs):` that returns a number; its sign picks the action.\n{task}\n") -> Callable:
    """Wrap any text generator (`generate(prompt) -> str`, e.g. the chat Engine's reply) as a proposer: propose(task_text, n) -> [source, ...] parsed from fenced blocks or bare defs."""
    def propose(task_text: str, n: int = 4) -> List[str]:
        out = []
        for i in range(n):
            txt = generate(prompt_template.format(task=task_text, i=i))
            blocks = _BLOCK.findall(txt) or _DEF.findall(txt)
            out.extend(b.strip() + "\n" for b in blocks)
        return out
    return propose
