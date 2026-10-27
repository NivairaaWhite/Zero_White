"""architect.py - the chat model (or a built-in stand-in) as the system's ARCHITECT: it proposes changes, the machinery tests them, only proven ones are kept.

After a plateau (the Generalist's race stopped improving) the architect is asked for HYPOTHESES about how to do better. A hypothesis is one line of a deliberately tiny grammar, so a small
model can write it and nothing it writes can do more than the grammar allows:

    HYPOTHESIS {"kind": "program", "mutator": "neuron", "n": 1}                     one named structural/numeric edit of the champion program (astmut.MUTATOR_KINDS)
    HYPOTHESIS {"kind": "genome", "family": "mlp", "changes": {"hidden": 64, "lr": {"x": 0.5}}}      other values for named genes of the champion's family
    HYPOTHESIS {"kind": "hook", "hook": "sigma_update"}                              ask the self-modifier to search for a better search rule (selfmod/hooks.py, its own gate)
    HYPOTHESIS {"kind": "patch", "file": "evolve.py"}                                a DRY RUN of the source patcher on one MUTABLE file: reports, never applies

Nothing here is free text that gets executed: a program edit is made by astmut (the same safe grammar every evolved program lives in), a genome change is clipped to the Gene ranges, a hook
or patch is a name from a short list. A line that does not parse is counted and dropped.

WHAT MAKES IT AN ARCHITECT AND NOT A SUGGESTION BOX is the verdict, which the proposer never sees:

  * PAIRED CONTROL. A program edit is scored on the same held-out episodes (or the same held-out validation blocks) as the incumbent; a genome change trains the incumbent's genome and the new
    genome from the SAME seeds for the same budget. The difference per pair goes through `stats.gate` (a sign-flip test) - the same test that judges self-written hooks and source patches.
  * MULTIPLE TESTING. Every hypothesis tested in one `improve` call shares the significance level (alpha / number tested), and a call that has too few pairs to ever reach it tests fewer.
  * HIT-RATE BACK-OFF. Every proposer is a bandit arm with its own measured acceptance rate; a chat model that keeps proposing nonsense (or things that do not help) falls to a small probing
    share, and a task whose architect rounds keep failing waits exponentially longer before being asked again.
  * EXPORT. Accepted hypotheses are written as training text for the chat model in the very prompt/answer format the ChatProposer uses, so training the chat model on its own validated
    successes is a loop the person can run (`zerowhite architect export`); rejected ones go to a short 'lessons' file, never in the answer position.

Honest limits. The built-in HeuristicProposer is a Thompson-sampling bandit over the grammar, NOT a language model: it is what runs when no chat model is trained, and it is the benchmark the chat
proposer has to beat (`report()` prints both acceptance rates). A chat model this small is not shown to propose better than the bandit; the loop is the mechanism, whether it pays is measured, not
assumed. Program hypotheses apply to program agents (env policies, formulas); genome hypotheses to every other family. The held-out data is the validation split / the gate episodes, which many
hypotheses share, so the correction above is applied and the reported test-set number stays the only unbiased one.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .. import fileio
from ..selfmod import astmut as A
from ..selfmod import stats

KINDS = ("program", "genome", "hook", "patch")
HOOKS = ("sigma_update", "select_parent")
MAX_LINE = 600
GATE_SEED0 = 40000                      # held-out episodes: disjoint from the selection (20000+), the validation (12345+) and the test (777000+) seeds of families.py
ALPHA = 0.05
SMALLEST_P = 1.0 / 20001                # the finest p a Monte Carlo sign-flip test can report (n > 16 pairs); exact tests reach 2**-n


# =========================================================================== the grammar
@dataclass
class Hypothesis:
    kind: str
    params: dict
    source: str = "heuristic"           # which proposer wrote it: "heuristic" | "chat"
    text: str = ""

    def key(self) -> str:
        p = self.params
        if self.kind == "program":
            return f"program/{p['mutator']}"
        if self.kind == "genome":
            return "genome/" + p["family"] + "/" + ",".join(sorted(p["changes"]))
        return f"{self.kind}/{p.get('hook') or p.get('file')}"

    def line(self) -> str:
        return "HYPOTHESIS " + json.dumps(dict(kind=self.kind, **self.params), sort_keys=False, separators=(", ", ": "))


class GrammarError(ValueError):
    pass


def validate(obj, spaces: Optional[Dict[str, dict]] = None, patch_files=None) -> Hypothesis:
    """Check one decoded hypothesis against the grammar. `spaces`: family name -> {gene: Gene}, so genome hypotheses can be checked against real genes; None skips that part."""
    if not isinstance(obj, dict):
        raise GrammarError("a hypothesis is a JSON object")
    kind = obj.get("kind")
    if kind not in KINDS:
        raise GrammarError(f"kind must be one of {KINDS}")
    if kind == "program":
        extra = set(obj) - {"kind", "mutator", "n"}
        if extra:
            raise GrammarError(f"unknown keys {sorted(extra)}")
        m = obj.get("mutator")
        if m not in A.MUTATOR_KINDS:
            raise GrammarError(f"mutator must be one of {sorted(A.MUTATOR_KINDS)}")
        n = obj.get("n", 1)
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 3:
            raise GrammarError("n must be an integer 1..3")
        return Hypothesis("program", dict(mutator=m, n=n))
    if kind == "genome":
        extra = set(obj) - {"kind", "family", "changes"}
        if extra:
            raise GrammarError(f"unknown keys {sorted(extra)}")
        fam, ch = obj.get("family"), obj.get("changes")
        if not isinstance(fam, str) or not fam or len(fam) > 40:
            raise GrammarError("family must be a name")
        if not isinstance(ch, dict) or not 1 <= len(ch) <= 4:
            raise GrammarError("changes must name 1..4 genes")
        space = (spaces or {}).get(fam)
        if spaces is not None and space is None:
            raise GrammarError(f"no agent family '{fam}'")
        clean = {}
        for g, v in ch.items():
            if space is not None and g not in space:
                raise GrammarError(f"family '{fam}' has no gene '{g}' (it has {sorted(space)})")
            if isinstance(v, bool):
                raise GrammarError("a gene value is a number or {\"x\": factor}")
            if isinstance(v, (int, float)):
                if not math.isfinite(v):
                    raise GrammarError("a gene value must be finite")
                clean[g] = float(v) if isinstance(v, float) else int(v)
            elif isinstance(v, dict) and set(v) == {"x"} and isinstance(v["x"], (int, float)) and not isinstance(v["x"], bool) and 0.25 <= v["x"] <= 4.0:
                clean[g] = {"x": float(v["x"])}
            else:
                raise GrammarError("a gene value is a number or {\"x\": factor between 0.25 and 4}")
        return Hypothesis("genome", dict(family=fam, changes=clean))
    if kind == "hook":
        if set(obj) - {"kind", "hook"} or obj.get("hook") not in HOOKS:
            raise GrammarError(f"a hook hypothesis names one of {HOOKS}")
        return Hypothesis("hook", dict(hook=obj["hook"]))
    if set(obj) - {"kind", "file"} or not isinstance(obj.get("file"), str):
        raise GrammarError("a patch hypothesis names a file")
    if patch_files is not None and obj["file"] not in patch_files:
        raise GrammarError(f"'{obj['file']}' is not one of the files the patcher may change")
    return Hypothesis("patch", dict(file=obj["file"]))


def parse(text: str, spaces=None, patch_files=None, source="chat"):
    """Hypotheses out of a model's reply: every line that starts with HYPOTHESIS (or is a bare JSON object) and passes the grammar. -> (hypotheses, [(line, why rejected)])."""
    good, bad, seen = [], [], set()
    for raw in str(text or "").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        body = ln[len("HYPOTHESIS"):].strip() if ln.startswith("HYPOTHESIS") else ln
        if not body.startswith("{"):
            if ln.startswith("HYPOTHESIS"):
                bad.append((ln[:80], "no JSON object after HYPOTHESIS"))
            continue
        if len(body) > MAX_LINE:
            bad.append((ln[:80], "line too long"))
            continue
        try:
            h = validate(json.loads(body), spaces, patch_files)
        except (ValueError, TypeError) as e:
            bad.append((ln[:80], str(e)))
            continue
        h.source, h.text = source, h.line()
        if h.text not in seen:
            seen.add(h.text)
            good.append(h)
    return good, bad


# =========================================================================== bandit statistics (persisted)
class Arms:
    """Beta(1 + hits, 1 + misses) per arm. `sample` is Thompson sampling; the means are what reports and the back-off use."""

    def __init__(self, d=None):
        self.d: Dict[str, List[int]] = {k: [int(v[0]), int(v[1])] for k, v in (d or {}).items()}

    def record(self, key: str, hit: bool):
        h = self.d.setdefault(key, [0, 0])
        h[1] += 1
        h[0] += int(bool(hit))

    def mean(self, key: str) -> float:
        h, n = self.d.get(key, (0, 0))
        return (1.0 + h) / (2.0 + n)

    def counts(self, key: str):
        h, n = self.d.get(key, (0, 0))
        return int(h), int(n)

    def sample(self, key: str, rng) -> float:
        h, n = self.d.get(key, (0, 0))
        return float(rng.beta(1 + h, 1 + n - h))


# =========================================================================== proposers
class HeuristicProposer:
    """Thompson sampling over the grammar's moves. NOT a language model: the stand-in that runs when no chat model exists, and the benchmark a chat model has to beat."""
    name = "heuristic"

    def propose(self, state: dict, arms: Arms, rng, n: int) -> List[Hypothesis]:
        cand = []
        if state.get("src"):
            for m in sorted(A.MUTATOR_KINDS):
                cand.append(("program", Hypothesis("program", dict(mutator=m, n=1)), f"program/{m}"))
        genes = state.get("space") or {}
        if state.get("genome") and genes:
            for g, gene in genes.items():
                for fac in (0.5, 2.0):
                    cand.append(("genome", Hypothesis("genome", dict(family=state["family"], changes={g: {"x": fac}})), f"genome/{state['family']}/{g}"))
        if state.get("hooks"):
            for h in HOOKS:
                cand.append(("hook", Hypothesis("hook", dict(hook=h)), f"hook/{h}"))
        if not cand:
            return []
        draws = np.array([arms.sample(k, rng) for _, _, k in cand])
        order = np.argsort(-draws, kind="stable")
        out, used = [], set()
        for i in order:
            h = cand[int(i)][1]
            h.source, h.text = self.name, h.line()
            if h.text in used:
                continue
            used.add(h.text)
            out.append(h)
            if len(out) >= n:
                break
        return out


def build_prompt(state: dict, recent: list, n: int) -> str:
    """The question put to the chat model - and, word for word, the prompt of the training documents `Architect.export_for_chat` writes."""
    lines = ["[architect]", f"task: {state.get('task', '?')} kind={state.get('kind', '?')} family={state.get('family', '?')}"]
    sk = state.get("skill")
    if isinstance(sk, (int, float)) and math.isfinite(sk):
        lines.append(f"skill: {sk:.3f}" + (" (plateau)" if state.get("plateau") else ""))
    elif state.get("plateau"):
        lines.append("skill: plateau")
    if state.get("src"):
        lines.append("program:")
        lines += ["    " + ln for ln in str(state["src"]).strip().splitlines()[:24]]
    if state.get("genome"):
        lines.append("genes: " + ", ".join(f"{k}={_fmt(v)}" + (f" ({state['space'][k].kind} {_fmt(state['space'][k].lo)}..{_fmt(state['space'][k].hi)})" if state.get("space") and k in state["space"] else "")
                                           for k, v in sorted(state["genome"].items())))
    if recent:
        lines.append("recent: " + "; ".join(f"{r['key']} -> {'kept' if r['accepted'] else 'rejected'}" for r in recent[-4:]))
    lines.append(f"propose up to {n} hypotheses, one per line: HYPOTHESIS {{json}}")
    return "\n".join(lines)


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


class ChatProposer:
    """Asks the chat model (chat.generate.Engine) for hypotheses. Anything that does not pass the grammar is dropped and counted; an engine that is not ready proposes nothing."""
    name = "chat"

    def __init__(self, engine, gen=None, temperature: float = 0.8, max_new_tokens: int = 160, seed: Optional[int] = None):
        self.engine, self.temperature, self.max_new_tokens, self.seed = engine, temperature, max_new_tokens, seed
        if gen is None:
            from ..chat.generate import Generator
            gen = Generator()
        self.gen = gen
        self.invalid = 0
        self.asked = 0
        self.last_reply = ""

    def ready(self) -> bool:
        return bool(getattr(self.engine, "ready", True))

    def propose(self, state: dict, arms: Arms, rng, n: int, spaces=None, patch_files=None, recent=None) -> List[Hypothesis]:
        if not self.ready():
            return []
        prompt = build_prompt(state, recent or [], n)
        try:
            reply = self.engine.reply(self.gen, "You are the architect of a self-improving learning system. Answer only with HYPOTHESIS lines.",
                                      [{"role": "user", "content": prompt}], temperature=self.temperature, max_new_tokens=self.max_new_tokens,
                                      seed=self.seed if self.seed is not None else int(rng.integers(1 << 30)))
        except Exception:                                  # noqa: BLE001 - a chat model that fails proposes nothing; the heuristic carries on
            return []
        self.asked += 1
        self.last_reply = reply
        good, bad = parse(reply, spaces, patch_files, source=self.name)
        self.invalid += len(bad)
        return good[:n]


# =========================================================================== subjects: what a hypothesis is tested ON
def _chunks(n_rows: int, k: int, seed: int = 4242):
    idx = np.random.default_rng(seed).permutation(n_rows)
    return [c for c in np.array_split(idx, k) if len(c)]


class ProgramSubject:
    """A program agent (env policy or regression formula): hypotheses edit its source; the verdict is paired per held-out episode / per held-out validation block."""
    kind = "program"

    def __init__(self, task, agent, n_pairs: int = 16, tune_steps: int = 12):
        self.task, self.agent, self.n_pairs, self.tune_steps = task, agent, int(n_pairs), int(tune_steps)
        self.vocab = agent.vocab
        self._cache: Dict[str, np.ndarray] = {}
        self._blocks = _chunks(len(task.Xval), min(self.n_pairs, 10)) if task.kind == "regression" else None
        if self._blocks is not None:
            self.n_pairs = len(self._blocks)

    # ---- state
    def src(self) -> str:
        return self.agent.best_src()

    def skill(self) -> float:
        return float(self.agent.val_score(self.task)[0])

    def state(self) -> dict:
        return dict(task=self.task.name, kind=self.task.kind, family=self.agent.family, skill=self.skill(), src=self.src(), genome=None, space=None)

    # ---- paired scores
    def scores(self, src: str) -> np.ndarray:
        key = A.normalize(src)
        if key in self._cache:
            return self._cache[key]
        if self.task.kind == "env":
            fn = A.compile_program(src, "policy", **self.vocab.safety)
            from .families import _to_action
            act = lambda o: _to_action(fn(self.agent._view(o)), self.agent.n_actions)            # noqa: E731
            v = np.array([self.task.skill(self.task.rollout(act, GATE_SEED0 + i)[0]) for i in range(self.n_pairs)])
        else:
            y = self.task.yval
            with np.errstate(all="ignore"):
                p = self.agent.predict(self.task.Xval, src)
            tot = float(np.var(y) + 1e-12)
            v = np.array([1.0 - float(np.mean((p[b] - y[b]) ** 2)) / tot for b in self._blocks])
            v = np.nan_to_num(np.clip(v, -1.0, 1.0), nan=-1.0)
        self._cache[key] = v
        return v

    # ---- a hypothesis -> a candidate program
    def realise(self, h: Hypothesis, rng) -> Optional[str]:
        src = self.src()
        for _ in range(h.params["n"]):
            nxt = A.apply_mutator(src, self.vocab, h.params["mutator"], rng)
            if nxt is None:
                break
            src = nxt
        if A.normalize(src) == A.normalize(self.src()):
            return None
        if self.tune_steps and A.constants_of(src):
            src, _, _ = A.tune_constants(src, self.agent.pe._evaluate, rng, steps=self.tune_steps)       # tuned on the SELECTION episodes / training rows, never on the gate's
        return src

    def adopt(self, src: str):
        self.agent.adopt(src)
        return self.agent


class GenomeSubject:
    """Any other family: a genome hypothesis trains the incumbent's genome and the proposed genome from the same seeds for the same budget; the paired difference is per seed."""
    kind = "genome"

    def __init__(self, task, agent, families=None, n_pairs: int = 8, units: float = 0.5):
        from .families import FAMILIES
        self.task, self.agent, self.n_pairs, self.units = task, agent, int(n_pairs), float(units)
        self.fam = (families or FAMILIES)[agent.family]
        self.genome = dict(agent.genome)
        self._control: Optional[tuple] = None
        self._made: Dict[str, tuple] = {}

    def state(self) -> dict:
        return dict(task=self.task.name, kind=self.task.kind, family=self.agent.family, skill=float(self.agent.val_score(self.task)[0]), src=None,
                    genome=dict(self.genome), space=self.fam.space)

    def realise(self, h: Hypothesis, rng) -> Optional[dict]:
        if h.params["family"] != self.agent.family:
            return None
        g = dict(self.genome)
        for name, v in h.params["changes"].items():
            gene = self.fam.space[name]
            nv = gene.clip(g[name] * v["x"]) if isinstance(v, dict) else gene.clip(v)
            if gene.kind == "int" and isinstance(v, dict) and nv == g[name]:
                nv = gene.clip(g[name] + (1 if v["x"] >= 1.0 else -1))
            g[name] = nv
        return None if g == self.genome else g

    def _train(self, genome: dict):
        scores, agents = [], []
        for s in range(self.n_pairs):
            a = self.fam.build(self.task, genome, 9000 + s)
            a.train(self.task, self.units, np.random.default_rng(9100 + s))
            scores.append(float(a.val_score(self.task)[0]))
            agents.append(a)
        return np.array(scores), agents

    def control(self) -> np.ndarray:
        if self._control is None:
            self._control = self._train(self.genome)
        return self._control[0]

    def scores_for(self, genome: dict) -> np.ndarray:
        sc, ag = self._train(genome)
        self._made[json.dumps(genome, sort_keys=True)] = (sc, ag)
        return sc

    def adopt(self, genome: dict):
        sc, ag = self._made[json.dumps(genome, sort_keys=True)]
        self.genome = dict(genome)
        self._control = (sc, ag)
        self.agent = ag[int(np.argmax(sc))]
        return self.agent


# =========================================================================== the architect
class Architect:
    def __init__(self, ws_root: str, seed: int = 0, chat=None, alpha: float = ALPHA, margin: float = 0.0, allow_hooks: bool = False, allow_patch: bool = False,
                 repo_root: Optional[str] = None, hook_evals: int = 60, verbose: bool = False):
        self.root = os.path.join(os.path.abspath(ws_root), "architect")
        os.makedirs(self.root, exist_ok=True)
        self.ws_root, self.seed, self.alpha, self.margin = ws_root, int(seed), float(alpha), float(margin)
        self.allow_hooks, self.allow_patch, self.repo_root, self.hook_evals, self.verbose = allow_hooks, allow_patch, repo_root, int(hook_evals), verbose
        self.chat = chat                                          # a ChatProposer or None
        self.heuristic = HeuristicProposer()
        self.rng = np.random.default_rng(seed)
        self._path = os.path.join(self.root, "state.json")
        self._ledger = os.path.join(self.root, "ledger.jsonl")
        st = fileio.read_json(self._path) if os.path.exists(self._path) else {}
        self.arms = Arms(st.get("arms"))
        self.cool = {k: dict(v) for k, v in (st.get("cooldown") or {}).items()}      # task -> {"misses": consecutive failed calls, "skip": calls still to skip}
        self.calls = int(st.get("calls", 0))

    # ---- bookkeeping ------------------------------------------------------------------------------
    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def save(self):
        fileio.write_json(self._path, dict(arms=self.arms.d, cooldown=self.cool, calls=self.calls))

    def _record(self, rec: dict):
        with open(self._ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, default=_jd) + "\n")

    def ledger(self) -> List[dict]:
        if not os.path.exists(self._ledger):
            return []
        out = []
        with open(self._ledger, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    pass
        return out

    # ---- back-off ---------------------------------------------------------------------------------
    def due(self, task_name: str) -> bool:
        """False while a task is serving its back-off (every failed call doubles the wait, up to 16 solves); asking consumes one wait."""
        c = self.cool.get(task_name)
        if not c or c.get("skip", 0) <= 0:
            return True
        c["skip"] -= 1
        self.save()
        return False

    def _after_call(self, task_name: str, accepted: int):
        c = self.cool.setdefault(task_name, dict(misses=0, skip=0))
        if accepted:
            c.update(misses=0, skip=0)
        else:
            c["misses"] = int(c.get("misses", 0)) + 1
            c["skip"] = min(16, 2 ** (c["misses"] - 1) - 1) if c["misses"] > 1 else 0      # 1st miss: ask again next time; then wait 1, 3, 7, 15 ...
        self.save()

    def chat_share(self) -> float:
        """The probability that a proposal slot goes to the chat model: its measured acceptance rate relative to the heuristic's, never below 10% (it must keep being probed) nor above 90%."""
        if self.chat is None:
            return 0.0
        c, h = self.arms.mean("source/chat"), self.arms.mean("source/heuristic")
        return float(np.clip(c / (c + h), 0.1, 0.9))

    # ---- proposing --------------------------------------------------------------------------------
    def _spaces(self):
        from .families import FAMILIES
        return {n: f.space for n, f in FAMILIES.items()}

    def _patch_files(self):
        from ..selfmod import patcher
        return patcher.MUTABLE

    def propose(self, state: dict, n: int, recent: list) -> List[Hypothesis]:
        state = dict(state, hooks=self.allow_hooks)
        out: List[Hypothesis] = []
        k_chat = int(sum(self.rng.random() < self.chat_share() for _ in range(n))) if self.chat is not None else 0
        if k_chat:
            out += self.chat.propose(state, self.arms, self.rng, k_chat, spaces=self._spaces(), patch_files=self._patch_files(), recent=recent)
        have = {h.text for h in out}
        for h in self.heuristic.propose(state, self.arms, self.rng, n + len(out)):
            if len(out) >= n:
                break
            if h.text not in have:
                out.append(h)
                have.add(h.text)
        return out[:n]

    # ---- the verdict ------------------------------------------------------------------------------
    def _max_tests(self, n_pairs: int) -> int:
        floor_p = 2.0 ** -n_pairs if n_pairs <= stats.EXACT_MAX else SMALLEST_P
        return int(self.alpha / (2.0 * floor_p))                  # alpha_each must clear the smallest p the test can report with room to spare

    def improve(self, task, agent, rounds: int = 2, per_round: int = 3, n_pairs: Optional[int] = None, units: float = 0.5, tune_steps: int = 12, plateau: bool = True) -> dict:
        """Run `rounds` of propose -> test -> adopt on one finished agent. Returns a report; `report['agent']` is the (possibly new) agent. Never makes the agent worse on the
        held-out pairs it is judged on, and never touches anything it was not given."""
        t0 = time.time()
        self.calls += 1
        subject = self._subject(task, agent, n_pairs, units, tune_steps)
        if subject is None:
            return dict(task=task.name, tried=0, accepted=0, note=f"no hypothesis applies to a '{getattr(agent, 'family', '?')}' agent", agent=agent, hypotheses=[])
        n = subject.n_pairs
        budget = max(0, min(rounds * per_round, self._max_tests(n)))
        alpha_each = self.alpha / max(1, budget)
        rep = dict(task=task.name, family=agent.family, pairs=n, tried=0, accepted=0, alpha_each=alpha_each, hypotheses=[], note="", agent=agent,
                   skill_before=float(subject.state()["skill"]))
        if budget == 0:
            rep["note"] = f"{n} paired scores can never reach p < {self.alpha:g}; nothing tested (raise n_pairs)"
            self._after_call(task.name, 0)
            return rep
        recent = self.ledger()[-6:]
        for r in range(rounds):
            if rep["tried"] >= budget:
                break
            st = dict(subject.state(), plateau=plateau)
            for h in self.propose(st, min(per_round, budget - rep["tried"]), recent):
                if rep["tried"] >= budget:
                    break
                res = self._test(subject, h, alpha_each, rep)
                if res is None:
                    continue
                recent = recent + [res]
        rep["agent"] = subject.agent
        rep["skill_after"] = float(subject.state()["skill"])
        rep["seconds"] = time.time() - t0
        if not rep["note"]:
            rep["note"] = f"{rep['accepted']} of {rep['tried']} hypotheses survived a paired test over {n} pairs at alpha {alpha_each:.4f} each"
        self._after_call(task.name, rep["accepted"])
        self.log(f"[architect] {task.name}: {rep['note']}")
        return rep

    def _subject(self, task, agent, n_pairs, units, tune_steps):
        if hasattr(agent, "pe") and hasattr(agent, "vocab") and hasattr(agent, "adopt"):
            if task.kind not in ("env", "regression"):
                return None
            return ProgramSubject(task, agent, n_pairs=n_pairs or 16, tune_steps=tune_steps)
        if hasattr(agent, "genome") and getattr(agent, "family", None) and task.kind in ("classification", "regression", "env", "text", "image_gen"):
            from .families import FAMILIES
            if agent.family in FAMILIES and FAMILIES[agent.family].space:
                return GenomeSubject(task, agent, n_pairs=n_pairs or 8, units=units)
        return None

    def _test(self, subject, h: Hypothesis, alpha_each: float, rep: dict) -> Optional[dict]:
        t0 = time.time()
        base_rec = dict(task=subject.task.name, task_kind=getattr(subject.task, "kind", "?"), key=h.key(), kind=h.kind, source=h.source, hypothesis=h.text or h.line(),
                        family=getattr(subject.agent, "family", None), pairs=subject.n_pairs, skill=rep.get("skill_before"))
        if h.kind in ("hook", "patch"):
            return self._test_system(subject, h, base_rec, rep)
        cand = subject.realise(h, self.rng)
        if cand is None:
            self._record(dict(base_rec, accepted=False, reason="the edit changed nothing", counted=False, t=time.time()))
            return None
        if h.kind == "program":
            base = subject.scores(subject.src())
            new = subject.scores(cand)
        else:
            base = subject.control()
            new = subject.scores_for(cand)
        v = stats.gate(base, new, margin=self.margin, alpha=alpha_each, min_n=1)
        rep["tried"] += 1
        self.arms.record(h.key(), v["accept"])
        self.arms.record("source/" + h.source, v["accept"])
        rec = dict(base_rec, accepted=bool(v["accept"]), gain=float(v["mean_gain"]), p=float(v["p"]), positive=int(v["positive"]), n=int(v["n"]), alpha=float(alpha_each), reason=v["reason"],
                   counted=True, seconds=time.time() - t0, t=time.time(), result=cand,
                   state=dict(src=subject.src() if h.kind == "program" else None, genome=dict(subject.genome) if h.kind == "genome" else None))
        if v["accept"]:
            subject.adopt(cand)
            rep["accepted"] += 1
            self.log(f"[architect]   KEPT {h.key()} ({h.source}): gain {v['mean_gain']:+.3f}, p={v['p']:.4f}, {v['positive']}/{v['n']} pairs better")
        else:
            self.log(f"[architect]   no   {h.key()} ({h.source}): {v['reason']}")
        rep["hypotheses"].append(dict(key=h.key(), source=h.source, accepted=bool(v["accept"]), gain=float(v["mean_gain"]), p=float(v["p"]), reason=v["reason"]))
        self._record(rec)
        self.save()
        return dict(key=h.key(), accepted=bool(v["accept"]))

    def _test_system(self, subject, h: Hypothesis, rec: dict, rep: dict) -> Optional[dict]:
        """Hook and patch hypotheses are judged by the self-modifier's own gates (selfmod/hooks.py, selfmod/patcher.py), which are their own paired tests. A patch is only ever a DRY RUN."""
        t0 = time.time()
        verdict, note = False, ""
        try:
            if h.kind == "hook":
                if not self.allow_hooks:
                    return None
                from ..selfmod.hooks import HookManager, evolve_hook
                r = evolve_hook(HookManager(self.ws_root), h.params["hook"], evals=self.hook_evals, seed=self.seed + self.calls, verbose=False)
                verdict, note = bool(r.get("accepted")), (r.get("gate") or {}).get("reason", "")
            else:
                if not (self.allow_patch and self.repo_root and os.path.isdir(os.path.join(self.repo_root, "tests"))):
                    return None
                from ..selfmod import patcher
                from ..selfmod.history import HistoryStore
                r = patcher.search(self.repo_root, HistoryStore(os.path.join(self.ws_root, "selfmod", "history")), evals=6, seed=self.seed + self.calls, apply=False,
                                   files=(h.params["file"],), verbose=False)
                verdict, note = bool(getattr(r, "accepted", None)), str(getattr(r, "note", ""))[:200]
        except Exception as e:                                # noqa: BLE001 - a failed system search is a missed opportunity, recorded as such
            note = f"{type(e).__name__}: {e}"
        rep["tried"] += 1
        self.arms.record(h.key(), verdict)
        self.arms.record("source/" + h.source, verdict)
        rep["accepted"] += int(verdict)
        rep["hypotheses"].append(dict(key=h.key(), source=h.source, accepted=verdict, reason=note))
        self._record(dict(rec, accepted=verdict, reason=note, counted=True, seconds=time.time() - t0, t=time.time(), dry_run=h.kind == "patch"))
        self.save()
        return dict(key=h.key(), accepted=verdict)

    # ---- reporting and export ---------------------------------------------------------------------
    def report(self) -> str:
        led = [r for r in self.ledger() if r.get("counted")]
        lines = [f"architect: {len(led)} hypotheses tested, {sum(1 for r in led if r['accepted'])} kept (alpha {self.alpha:g}, shared across each call)"]
        for src in ("heuristic", "chat"):
            h, n = self.arms.counts("source/" + src)
            if n:
                lines.append(f"  {src:<10} {h}/{n} kept ({100.0 * h / n:.0f}%)")
        if self.chat is not None:
            lines.append(f"  chat share of proposals now {100 * self.chat_share():.0f}% ({self.chat.invalid} lines failed the grammar)")
        best = sorted(((k, v) for k, v in self.arms.d.items() if "/" in k and not k.startswith("source/") and v[1] > 0), key=lambda kv: -(kv[1][0] + 1) / (kv[1][1] + 2))[:5]
        for k, (h, n) in best:
            lines.append(f"  {k:<34} {h}/{n}")
        return "\n".join(lines)

    def export_for_chat(self, out_dir: str, max_docs: int = 200) -> dict:
        """Accepted hypotheses -> plain-text training documents in the exact prompt/answer format of ChatProposer; the rest -> one capped 'lessons' file."""
        os.makedirs(out_dir, exist_ok=True)
        led = [r for r in self.ledger() if r.get("counted")]
        n_doc = 0
        for i, r in enumerate(r for r in led if r["accepted"]):
            if n_doc >= max_docs:
                break
            st = r.get("state") or {}
            state = dict(task=r["task"], kind=r.get("task_kind", "?"), family=r.get("family") or "?", skill=r.get("skill"), src=st.get("src"), genome=st.get("genome"), space=None)
            doc = build_prompt(state, [], 1) + "\n" + r["hypothesis"] + f"\n[validated] gain {r['gain']:+.3f}, p={r['p']:.4f}, {r['positive']} of {r['n']} paired scores better\n"
            fileio.write_text(os.path.join(out_dir, f"architect-{i:04d}.txt"), doc[:6000])
            n_doc += 1
        lessons = ["[architect lessons] hypotheses that were tested and did NOT survive a paired test (they are not suggestions)"]
        for r in [r for r in led if not r["accepted"]][-120:]:
            lessons.append(f"{r['task']}: {r['key']} ({r['source']}) - {str(r.get('reason', ''))[:140]}")
        fileio.write_text(os.path.join(out_dir, "architect-lessons.txt"), "\n".join(lessons)[:6000] + "\n")
        return dict(documents=n_doc, lessons=len(lessons) - 1)


def _jd(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
