"""engine_tier.py - an opt-in second tier of self-modification: the numerical core (`nn.py`, `engine.py`) may be edited, under the SAME judge as everything else.

Level 2 (patcher.py) changes numeric literals in eight tuning modules. This tier lets the machinery, or a person, or a model through `forge/system2`-style proposals, change CODE in
the files `ENGINE` lists: swap an operator, flip a comparison, change a constant, or replace a whole function body with new code (loops, new tensor expressions, anything that is
valid Python). The point of the tier is that this is allowed WITHOUT loosening a single check:

  structure   `engine_safe`: the edited file must parse; it may not add an import, may not call exec / eval / compile / open / __import__ or touch os / sys / subprocess / socket /
              ctypes / shutil / importlib / pickle where the original did not, and may not remove or rename any function or class (so every caller still finds what it calls);
  tests       the full test suite must pass in a sandboxed copy;
  benchmark   the final-seed gain must clear the margin AND the paired significance test (selfmod/stats.py) against the ORIGINAL tree;
  promotion   `watchdog.run`: a journaled transaction, the tests again on the real tree, and a byte-for-byte check that the sealed constitution is untouched.

WHAT STAYS SEALED, AND WHY. The judge (safety.py, sandbox.py, patcher.py, history.py, stats.py, watchdog.py, winjob.py, fileio.py, doctor.py) and this file are not editable by the machinery:
a tier that could rewrite the judge would not be judged. `astmut.py` is also still sealed although the review asked to free it: it holds `compile_program`, the boundary between evolved
text and executed code, so it belongs with the judge rather than with the engine. The tier is OFF unless a person enables it (`selfmod --engine-tier`), and `ENGINE` can only be widened by
editing this sealed file.

HONEST STATUS. `evaluate_candidate` accepts arbitrary valid code for the ENGINE files, but the only built-in PROPOSER is single-edit mutation (`mutations`); the System-2 loop's proposers have never been
shown to write useful numerical code. The benchmark may not exercise much of nn.py / engine.py, in which case almost every candidate ends as 'no gain': that is the judge working, not a bug.
"""
from __future__ import annotations

import ast
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import patcher, stats, watchdog
from .sandbox import SandboxCopy

ENGINE = ("nn.py", "engine.py")
DANGEROUS_CALLS = {"exec", "eval", "compile", "open", "__import__", "input", "breakpoint"}
DANGEROUS_ROOTS = {"os", "sys", "subprocess", "socket", "ctypes", "shutil", "importlib", "builtins", "pickle", "marshal", "urllib", "http", "multiprocessing", "threading"}


# ------------------------------------------------------------------------------------------------ the structure gate
def _imports(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {("import", a.name) for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            out |= {(f"from {'.' * n.level}{n.module or ''}", a.name) for a in n.names}
    return out


def _defs(tree):
    out = set()

    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(prefix + ch.name)
                walk(ch, prefix + ch.name + ".")
            else:
                walk(ch, prefix)
    walk(tree, "")
    return out


def _danger(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in DANGEROUS_CALLS:
            out.add(n.func.id)
        if isinstance(n, ast.Name) and n.id in DANGEROUS_ROOTS:
            out.add(n.id)
        if isinstance(n, ast.Attribute):
            root = n
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in DANGEROUS_ROOTS:
                out.add(root.id)
    return out


def engine_safe(old: str, new: str) -> Tuple[bool, str]:
    try:
        a, b = ast.parse(old), ast.parse(new)
    except SyntaxError as ex:
        return False, f"does not parse: {ex.msg} (line {ex.lineno})"
    if _imports(b) != _imports(a):
        return False, "the set of imports changed"
    added = _danger(b) - _danger(a)
    if added:
        return False, "uses " + ", ".join(sorted(added)) + " where the original did not"
    gone = _defs(a) - _defs(b)
    if gone:
        return False, "removes or renames: " + ", ".join(sorted(gone))
    return True, ""


# ------------------------------------------------------------------------------------------------ single-edit mutation
_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}
_SWAP = {"+": "-", "-": "+", "*": "/"}
_CMP = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
_CSWAP = {"<": "<=", "<=": "<", ">": ">=", ">=": ">"}


def one_edit(old: str, new: str) -> bool:
    """True iff the two sources parse to trees identical except for EXACTLY one operator / comparison / numeric constant."""
    try:
        a, b = list(ast.walk(ast.parse(old))), list(ast.walk(ast.parse(new)))
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
                if not (isinstance(x.value, (int, float)) and isinstance(y.value, (int, float)) and not isinstance(x.value, bool)):
                    return False
                diffs += 1
        elif isinstance(x, ast.BinOp) and type(x.op) is not type(y.op):
            diffs += 1
        elif isinstance(x, ast.Compare) and [type(o) for o in x.ops] != [type(o) for o in y.ops]:
            diffs += 1
        else:
            for f in x._fields:
                vx, vy = getattr(x, f, None), getattr(y, f, None)
                if isinstance(vx, (str, bool, type(None))) and vx != vy:
                    return False
    return diffs == 1


def _replace(src: str, lineno: int, col: int, end_col: int, text: str) -> str:
    lines = src.splitlines(keepends=True)
    raw = lines[lineno - 1].encode("utf-8")
    lines[lineno - 1] = (raw[:col] + text.encode("utf-8") + raw[end_col:]).decode("utf-8")
    return "".join(lines)


def mutations(src: str, rng: np.random.Generator, limit: int = 40) -> List[Tuple[str, str]]:
    """Up to `limit` single-edit candidates [(new_source, description)], in random order, each verified by `one_edit`. Sites are inside function bodies only."""
    tree = ast.parse(src)
    lines = src.encode("utf-8").splitlines(keepends=True)
    sites = []
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    doc_nodes = {id(f.body[0].value) for f in funcs if f.body and isinstance(f.body[0], ast.Expr) and isinstance(f.body[0].value, ast.Constant)}
    seen = set()
    for f in funcs:
        for n in ast.walk(f):
            if id(n) in seen:
                continue
            seen.add(id(n))
            if isinstance(n, ast.BinOp) and type(n.op) in _OPS and n.left.end_lineno == n.right.lineno:
                seg = lines[n.left.end_lineno - 1][n.left.end_col_offset:n.right.col_offset].decode("utf-8")
                op = seg.strip()
                if op in _SWAP and (len(seg) - len(seg.lstrip()) + len(op) + len(seg) - len(seg.rstrip())) >= 0:
                    off = n.left.end_col_offset + len(seg) - len(seg.lstrip())
                    sites.append((n.left.end_lineno, off, off + len(op), _SWAP[op], f"{op} -> {_SWAP[op]}", f.name))
            elif isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _CMP and n.left.end_lineno == n.comparators[0].lineno:
                seg = lines[n.left.end_lineno - 1][n.left.end_col_offset:n.comparators[0].col_offset].decode("utf-8")
                op = seg.strip()
                if op in _CSWAP:
                    off = n.left.end_col_offset + len(seg) - len(seg.lstrip())
                    sites.append((n.left.end_lineno, off, off + len(op), _CSWAP[op], f"{op} -> {_CSWAP[op]}", f.name))
            elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool) and id(n) not in doc_nodes and n.lineno == n.end_lineno:
                v = n.value
                for new in ([v + 1, v - 1] if isinstance(v, int) else [v * 1.1, v * 0.9]):
                    if new != v and (isinstance(v, float) or new >= 0):
                        txt = repr(int(new)) if isinstance(v, int) else f"{new:.4g}"
                        sites.append((n.lineno, n.col_offset, n.end_col_offset, txt, f"{v!r} -> {txt}", f.name))
    order = rng.permutation(len(sites))
    out = []
    for i in order:
        ln, c0, c1, text, desc, fn = sites[int(i)]
        new = _replace(src, ln, c0, c1, text)
        if one_edit(src, new):
            out.append((new, f"{fn}() line {ln}: {desc}"))
            if len(out) >= limit:
                break
    return out


# ------------------------------------------------------------------------------------------------ the judge, reused
def evaluate_candidate(repo_root: str, store, changes: Dict[str, str], apply: bool = False, margin: float = 0.004, alpha: float = 0.05,
                       final_seeds=tuple(range(100, 106)), verbose: bool = False) -> dict:
    """Judge a candidate change to ENGINE files (a dict {file relative to zerowhite/: new full text}). Returns dict(accepted, applied, stage, reason, gate, ...)."""
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    pkg = os.path.join(repo_root, "zerowhite")
    res = dict(accepted=False, applied=False, stage="", reason="", gate={}, files=sorted(changes))
    for f in changes:
        if f not in ENGINE:
            raise PermissionError(f"{f} is not an engine-tier file {ENGINE}")
    originals = {f: patcher._read_raw(os.path.join(pkg, f)) for f in changes}
    for f, new in changes.items():
        ok, why = engine_safe(originals[f], new)
        if not ok:
            return dict(res, stage="structure", reason=f"{f}: {why}")
    if all(changes[f] == originals[f] for f in changes):
        return dict(res, stage="structure", reason="nothing differs")
    watchdog.recover(repo_root)
    with SandboxCopy(repo_root) as sc:
        for f, new in changes.items():
            patcher._write_raw(os.path.join(sc.root, "zerowhite", f), new)
        ok, tail = patcher.tests_pass(sc.root)
        if not ok:
            return dict(res, stage="tests", reason=f"the test suite failed in the sandboxed copy ({tail})")
        patched = patcher.bench_each(sc.root, final_seeds)
    with SandboxCopy(repo_root) as base:
        base_rows = patcher.bench_each(base.root, final_seeds)
    if base_rows is None or patched is None:
        return dict(res, stage="benchmark", reason="the benchmark failed on the final seeds")
    gate = stats.gate(base_rows, patched, margin=margin, alpha=alpha)
    res.update(gate=gate, final_base=float(np.mean(base_rows)), final_patched=float(np.mean(patched)))
    log(f"[engine] final seeds: original {res['final_base']:.4f} patched {res['final_patched']:.4f} (p = {gate['p']:.3f})")
    if not gate["accept"]:
        return dict(res, stage="gate", reason=gate["reason"])
    res["accepted"] = True
    if not apply:
        return dict(res, stage="done", reason="accepted by the judge; NOT applied (pass apply=True)")
    for f in changes:
        store.put("src:" + f, originals[f], dict(kind="original"))
        store.put("src:" + f, changes[f], dict(kind="engine-patch"))
    tx = watchdog.run(repo_root, {f"zerowhite/{f}": changes[f] for f in changes}, [f"zerowhite/{f}" for f in ENGINE], verify=lambda: patcher.tests_pass(repo_root))
    if not tx["ok"]:
        for f in changes:
            store.put("src:" + f, originals[f], dict(kind="auto-rollback", reason=tx["reason"][:200]))
        return dict(res, accepted=False, stage="promotion", reason=f"{tx['reason']}; originals restored")
    return dict(res, applied=True, stage="done", reason="applied through the journaled transaction")


def search(repo_root: str, store, evals: int = 12, seed: int = 0, select_seeds=(0, 1, 2), confirm_seeds=(10, 11, 12), apply: bool = False, verbose: bool = True,
           files=ENGINE, **judge) -> dict:
    """Hill-climb on single-edit mutations of the ENGINE files (cheap select / confirm benchmarks, like patcher.search), then put the survivors through `evaluate_candidate`."""
    rng = np.random.default_rng(seed)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    pkg = os.path.join(repo_root, "zerowhite")
    originals = {f: patcher._read_raw(os.path.join(pkg, f)) for f in files}
    current = dict(originals)
    kept, tried = [], 0
    with SandboxCopy(repo_root) as sc:
        cur_sel, cur_con = patcher.bench_in(sc.root, select_seeds), patcher.bench_in(sc.root, confirm_seeds)
        if cur_sel is None or cur_con is None:
            return dict(accepted=False, applied=False, stage="baseline", reason="the baseline benchmark failed in the sandbox copy", tried=0, kept=[])
        for i in range(evals):
            f = files[int(rng.integers(len(files)))]
            cands = mutations(current[f], rng, limit=1)
            if not cands:
                continue
            new, desc = cands[0]
            ok, why = engine_safe(current[f], new)
            if not ok:
                continue
            tried += 1
            path = os.path.join(sc.root, "zerowhite", f)
            patcher._write_raw(path, new)
            sel = patcher.bench_in(sc.root, select_seeds)
            con = patcher.bench_in(sc.root, confirm_seeds) if sel is not None and sel > cur_sel + 1e-9 else None
            if sel is None or sel <= cur_sel + 1e-9 or con is None or con < cur_con - 1e-9:
                patcher._write_raw(path, current[f])
                log(f"[engine {i + 1:02d}] {f} {desc}: rejected")
                continue
            current[f], cur_sel, cur_con = new, sel, con
            kept.append(dict(file=f, edit=desc, select=sel, confirm=con))
            log(f"[engine {i + 1:02d}] {f} {desc}: kept (select {sel:.4f}, confirm {con:.4f})")
    changes = {f: current[f] for f in files if current[f] != originals[f]}
    if not changes:
        return dict(accepted=False, applied=False, stage="search", reason="no single-edit mutation survived selection and confirmation", tried=tried, kept=[])
    out = evaluate_candidate(repo_root, store, changes, apply=apply, verbose=verbose, **judge)
    return dict(out, tried=tried, kept=kept)
