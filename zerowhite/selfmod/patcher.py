"""patcher.py - level-2 self-modification: Zero White edits its OWN SOURCE FILES.

What it may change: numeric hyperparameter *sites* in a short list of modules (MUTABLE):
signature defaults, keyword arguments, class/module constants, `self.x = const` inside `__init__`,
and the weights in the self-modifier's candidate-scoring formula. It will not touch literals that
are part of the maths (a `0.5` inside a loss, an epsilon, an index), and it can never change
anything outside MUTABLE: safety.py, history.py, sandbox.py, this file, fileio.py, doctor.py and the
tests are the CONSTITUTION, which the machinery that proposes patches is not allowed to edit.

Every patch is judged in a throw-away COPY of the tree, in a sandboxed subprocess:

  1. structural check   exactly one literal differs between old and new source (AST comparison);
  2. SELECT seeds       the deterministic benchmark must improve (greedy hill-climb);
  3. CONFIRM seeds      different seeds; the patch must not make things worse there;
  4. FINAL seeds        used once, on the accumulated set of patches, with a margin;
  5. TEST SUITE         the repository's whole test suite must pass in the patched copy.

Only if all of that holds is anything applied to the real tree (and only when apply=True). Applying
snapshots the original into the history store, re-runs the tests on the real tree, and automatically
restores the originals if they fail. `selfmod rollback --file X` restores any earlier version.

HONEST LIMIT: the benchmark is small. A patch that passes is "better on this benchmark, on seeds it
never trained on, with the whole test suite green" - useful evidence, not proof of general improvement.
"""

import ast
import difflib
import json
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .. import fileio
from . import stats, watchdog
from .history import HistoryStore
from .sandbox import SandboxCopy, run_python

MUTABLE = ("zero_white.py", "self_model.py", "meta_optimizer.py", "evolve.py", "agents.py",
           "generative.py", "curiosity.py", "architecture.py")
CONSTITUTION = ("selfmod/safety.py", "selfmod/history.py", "selfmod/sandbox.py", "selfmod/patcher.py", "selfmod/winjob.py",
                "selfmod/stats.py", "selfmod/watchdog.py", "selfmod/engine_tier.py", "forage/fetch.py",
                "selfmod/astmut.py", "fileio.py", "doctor.py", "evolved/__init__.py", "selfmod/memwatch.py",
                "forage/act.py", "selfmod/bench.py", "forge/architect.py")      # 6.5: the actuator's policy, the judge of patches, the judge of architect proposals
# Never patched: structural dimensions, anything that defines HOW A MODEL IS SCORED (a self-modifying system must
# not be able to raise its score by changing the measurement), the environments' reward definitions, and plain
# compute budgets (more compute is not a better algorithm).
SKIP_NAMES = frozenset({
    "seed", "axis", "size", "ndigits", "indent", "timeout", "eps", "epsilon", "decimals", "version", "ddof", "dtype",
    "dim", "in_dim", "out_dim", "n_classes", "num_chunks", "obs_dim", "n_actions", "feat_dim", "state_dim",
    "max_steps", "step_cost",                                                   # structure / reward definition
    "episodes", "eval_episodes", "per_class", "val_cap", "n", "n_per_class", "val_frac", "block",  # measurement
    "generations", "max_generations", "probation_steps", "updates", "steps", "population", "max_history",
    "max_vocab", "consolidate_steps",                                           # budgets
})


@dataclass
class Site:
    file: str
    lineno: int
    col: int
    end_col: int
    old: float
    name: str
    is_int: bool = False

    def key(self):
        return (self.file, self.lineno, self.col)

    def label(self):
        return f"{self.file}:{self.lineno} {self.name}={self.old}"


# --------------------------------------------------------------------------- finding sites
def _read_raw(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8")


def _write_raw(path, text):
    fileio.write_bytes(path, text.encode("utf-8"))


def _line_parts(src):
    return re.split(r"(?<=\n)", src)


def _numeric(node):
    return isinstance(node, ast.Constant) and type(node.value) in (int, float) and not isinstance(node.value, bool)


def find_sites(pkg_dir, files=MUTABLE):
    sites, seen = [], set()
    for fn in files:
        path = os.path.join(pkg_dir, fn)
        if not os.path.exists(path):
            continue
        tree = ast.parse(_read_raw(path))

        def add(name, node):
            if not _numeric(node) or name in SKIP_NAMES:
                return
            v = node.value
            if v in (0, 1, -1) or abs(v) < 1e-6 or node.end_lineno != node.lineno:
                return
            s = Site(fn, node.lineno, node.col_offset, node.end_col_offset, v, name, isinstance(v, int))
            if s.key() not in seen:
                seen.add(s.key())
                sites.append(s)

        def visit(node, scope):
            if isinstance(node, ast.FunctionDef):
                a = node.args
                pos = a.posonlyargs + a.args
                for arg, d in zip(pos[len(pos) - len(a.defaults):], a.defaults):
                    add(arg.arg, d)
                for arg, d in zip(a.kwonlyargs, a.kw_defaults):
                    if d is not None:
                        add(arg.arg, d)
                for ch in node.body:
                    visit(ch, ("func", node.name))
                return
            if isinstance(node, ast.ClassDef):
                for ch in node.body:
                    visit(ch, ("class", node.name))
                return
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg:
                        add(kw.arg, kw.value)
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                tgt, val = node.targets[0], node.value
                if isinstance(tgt, ast.Name) and scope[0] in ("module", "class"):
                    add(tgt.id, val)
                    if isinstance(val, ast.Tuple) and isinstance(tgt, ast.Name):
                        for el in val.elts:
                            add(tgt.id, el)
                elif isinstance(tgt, ast.Tuple) and scope[0] in ("module", "class") and isinstance(val, ast.Tuple):
                    for t, el in zip(tgt.elts, val.elts):
                        if isinstance(t, ast.Name):
                            add(t.id, el)
                elif isinstance(tgt, ast.Attribute) and scope == ("func", "__init__"):
                    add(tgt.attr, val)
                elif isinstance(tgt, ast.Name) and tgt.id == "score" and scope == ("func", "self_improve_generation"):
                    for sub in ast.walk(val):                        # the candidate-scoring weights
                        add("score_weight", sub)
            for ch in ast.iter_child_nodes(node):
                visit(ch, scope)

        for top in tree.body:
            visit(top, ("module", ""))
    return sites


def propose_value(site, rng):
    for _ in range(20):
        f = float(np.exp(rng.normal(0.0, 0.45)))
        new = site.old * f
        if site.is_int:
            new = int(max(1, round(new)))
            if new == site.old:
                new = site.old + int(rng.choice([-1, 1]))
            new = max(1, new)
        else:
            new = float(f"{new:.3g}")
        if new != site.old and (site.is_int or new != 0.0):
            return new
    return None


def apply_to_text(src, site, new):
    parts = _line_parts(src)
    line = parts[site.lineno - 1]
    raw = line.encode("utf-8")
    old_tok = raw[site.col:site.end_col].decode("utf-8")
    new_tok = repr(new) if not site.is_int else str(int(new))
    parts[site.lineno - 1] = (raw[:site.col] + new_tok.encode("utf-8") + raw[site.end_col:]).decode("utf-8")
    assert old_tok.strip(), "empty token at site"
    return "".join(parts)


def single_literal_change(old_src, new_src):
    """True iff the two sources parse to identical trees except for exactly one numeric literal."""
    try:
        a, b = list(ast.walk(ast.parse(old_src))), list(ast.walk(ast.parse(new_src)))
    except SyntaxError:
        return False
    if len(a) != len(b):
        return False
    diffs = 0
    for x, y in zip(a, b):
        if type(x) is not type(y):
            return False
        if isinstance(x, ast.Constant):
            if x.value != y.value or type(x.value) is not type(y.value):
                if not (_numeric(x) and _numeric(y)):
                    return False
                diffs += 1
        else:
            for f in x._fields:
                vx, vy = getattr(x, f, None), getattr(y, f, None)
                if isinstance(vx, (str, int, float, bool, type(None))) and vx != vy:
                    return False
    return diffs == 1


# --------------------------------------------------------------------------- measuring
def bench_in(tree_root, seeds, timeout=240):
    r = run_python(["-m", "zerowhite", "selfmod", "bench", "--seeds"] + [str(s) for s in seeds],
                   cwd=tree_root, timeout=timeout)
    if not r.ok:
        return None
    for line in reversed(r.stdout.strip().splitlines()):
        try:
            return float(json.loads(line)["score"])
        except (ValueError, KeyError):
            continue
    return None


def bench_each(tree_root, seeds, timeout=240):
    """One score per seed (one benchmark process per seed), or None if any seed failed. The significance gate needs the individual
    scores, not their mean; going through `bench_in` keeps a single place that knows how to run and parse the benchmark."""
    out = []
    for s in seeds:
        v = bench_in(tree_root, [s], timeout)
        if v is None:
            return None
        out.append(float(v))
    return out


def tests_pass(tree_root, timeout=420):
    r = run_python([os.path.join("tests", "run_all.py")], cwd=tree_root, timeout=timeout)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    return r.ok, tail


@dataclass
class PatchReport:
    accepted: list = field(default_factory=list)
    tried: int = 0
    rejected: dict = field(default_factory=dict)
    base_select: float = 0.0
    final_base: float = 0.0
    final_patched: float = 0.0
    tests_ok: bool = None
    tests_tail: str = ""
    applied: bool = False
    diff: str = ""
    note: str = ""
    gate: dict = field(default_factory=dict)             # 6.3: the paired significance test on the final seeds (stats.gate)

    def to_dict(self):
        return dict(self.__dict__)


# --------------------------------------------------------------------------- the search
def search(repo_root, store, evals=24, seed=0, select_seeds=(0, 1, 2), confirm_seeds=(10, 11, 12),
           final_seeds=tuple(range(100, 106)), margin=0.004, apply=False, verbose=True, files=MUTABLE, alpha=0.05):
    """Hill-climb on source hyperparameters in a sandboxed copy of `repo_root`; optionally apply the result.

    6.3: the FINAL decision is the margin AND a paired sign-flip test on the per-seed scores (p < alpha, `alpha=None` restores the
    margin-only rule). Six final seeds can reach p = 1/64; fewer than five could never be significant at 0.05."""
    rng = np.random.default_rng(seed)
    rep = PatchReport()
    pkg = os.path.join(repo_root, "zerowhite")
    watchdog.recover(repo_root)                  # 6.3: a change that an earlier crash left half-applied is rolled back before anything else
    originals = {f: _read_raw(os.path.join(pkg, f)) for f in files if os.path.exists(os.path.join(pkg, f))}
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    with SandboxCopy(repo_root) as sc:
        cpkg = os.path.join(sc.root, "zerowhite")
        cur_sel = rep.base_select = bench_in(sc.root, select_seeds)
        cur_con = bench_in(sc.root, confirm_seeds)
        if cur_sel is None or cur_con is None:
            rep.note = "baseline benchmark failed to run in the sandbox copy; nothing attempted"
            return rep
        log(f"[patch] baseline  select={cur_sel:.4f}  confirm={cur_con:.4f}")
        current = dict(originals)
        for i in range(evals):
            sites = find_sites(cpkg, files)
            if not sites:
                break
            site = sites[int(rng.integers(len(sites)))]
            new = propose_value(site, rng)
            if new is None:
                continue
            rep.tried += 1
            text = apply_to_text(current[site.file], site, new)
            if not single_literal_change(current[site.file], text):
                rep.rejected["structure"] = rep.rejected.get("structure", 0) + 1
                continue
            path = os.path.join(cpkg, site.file)
            _write_raw(path, text)
            sel = bench_in(sc.root, select_seeds)
            verdict = None
            if sel is None:
                verdict = "crash"
            elif sel <= cur_sel + 1e-9:
                verdict = "no-gain"
            else:
                con = bench_in(sc.root, confirm_seeds)
                if con is None or con < cur_con - 1e-9:
                    verdict = "confirm-worse"
            if verdict:
                _write_raw(path, current[site.file])
                rep.rejected[verdict] = rep.rejected.get(verdict, 0) + 1
                log(f"[patch {i + 1:02d}] {site.label()} -> {new}: rejected ({verdict})")
                continue
            current[site.file] = text
            cur_sel, cur_con = sel, con
            rep.accepted.append(dict(file=site.file, line=site.lineno, name=site.name, old=site.old, new=new,
                                     select=sel, confirm=con))
            log(f"[patch {i + 1:02d}] {site.label()} -> {new}: kept (select {sel:.4f}, confirm {con:.4f})")

        changed = [f for f in current if current[f] != originals[f]]
        if not changed:
            rep.note = "no patch survived selection and confirmation; the source stays as it is"
            return rep

        # final, one-shot decision on seeds nothing has been tuned on (compare against the ORIGINAL tree)
        with SandboxCopy(repo_root) as base:
            base_rows = bench_each(base.root, final_seeds)
        patched_rows = bench_each(sc.root, final_seeds)
        if base_rows is None or patched_rows is None:
            rep.note = "the benchmark failed on the final seeds; rejected"
            rep.accepted = []
            return rep
        rep.final_base, rep.final_patched = float(np.mean(base_rows)), float(np.mean(patched_rows))
        rep.gate = stats.gate(base_rows, patched_rows, margin=margin, alpha=alpha)
        log(f"[patch] FINAL seeds: original {rep.final_base:.4f}  patched {rep.final_patched:.4f}  "
            f"(p = {rep.gate['p']:.3f}, {rep.gate['positive']}/{rep.gate['n']} seeds improved)")
        if not rep.gate["accept"]:
            rep.note = ("patched tree did not beat the original by the margin on unseen seeds; rejected" if rep.final_patched <= rep.final_base + margin
                        else f"rejected by the significance gate: {rep.gate['reason']}")
            rep.accepted = []
            return rep
        ok, tail = tests_pass(sc.root)
        rep.tests_ok, rep.tests_tail = ok, tail
        log(f"[patch] test suite in patched copy: {'PASS' if ok else 'FAIL'}  ({tail})")
        if not ok:
            rep.note = "test suite failed in the patched copy; rejected"
            rep.accepted = []
            return rep

        diffs = []
        for f in changed:
            diffs += list(difflib.unified_diff(originals[f].splitlines(True), current[f].splitlines(True),
                                               f"a/zerowhite/{f}", f"b/zerowhite/{f}"))
        rep.diff = "".join(diffs)

    if apply:
        for f in changed:                                                    # snapshot first (the audit trail shows every attempt) ...
            if f not in MUTABLE:
                raise PermissionError(f"{f} is not mutable")
            store.put("src:" + f, originals[f], dict(kind="original"))
            store.put("src:" + f, current[f], dict(kind="self-patch", patches=[a for a in rep.accepted if a["file"] == f]))
        # ... then ONE journaled transaction (watchdog.py): atomic per-file swaps, the test suite on the real tree and a check that the sealed
        # constitution is untouched; any failure, exception or crash restores every original (a crash is repaired by watchdog.recover at the
        # next start). Before 6.3 the files were written one by one with nothing to repair a half-finished write.
        res = watchdog.run(repo_root, {f"zerowhite/{f}": current[f] for f in changed}, [f"zerowhite/{f}" for f in MUTABLE],
                           verify=lambda: tests_pass(repo_root))
        if not res["ok"]:
            for f in changed:
                store.put("src:" + f, originals[f], dict(kind="auto-rollback", reason=res["reason"][:200]))
            rep.note = f"{res['reason']}; originals restored"
            rep.accepted = []
        else:
            rep.applied = True
            rep.note = f"applied {len(rep.accepted)} patch(es) to {', '.join(changed)}"
    else:
        rep.note = "accepted in the sandbox; NOT applied (run again with --apply to modify the real source)"
    return rep


def rollback_file(repo_root, store, file, steps=1):
    """Restore an earlier version of a mutable source file into the real tree."""
    if file not in MUTABLE:
        raise PermissionError(f"{file} is not under self-modification control")
    rec = store.rollback("src:" + file, steps)
    _write_raw(os.path.join(repo_root, "zerowhite", file), store.get(rec["sha"]))
    return rec


def save_report(ws_root, rep):
    d = os.path.join(ws_root, "selfmod", "patches")
    os.makedirs(d, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    fileio.write_json(os.path.join(d, f"{stamp}.json"), rep.to_dict(), indent=2)
    if rep.diff:
        fileio.write_text(os.path.join(d, f"{stamp}.diff"), rep.diff)
    return os.path.join(d, stamp)
