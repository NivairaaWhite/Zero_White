"""shared.py - make what the forge has learned, and what the program is, readable by the chat model, as ordinary training text.

Two one-way bridges into the chat model's data folder (the folder `zerowhite chat prepare` ingests; files there are plain .txt):

  export_zoo         the 'zoo' of agents the forge built (the programs and formulas it discovered, with their skill on each task) -> one text document per agent
                     (`_zoo/`). The chat model can then read, and be trained on, what the other half of the system found.
  ingest_self_source the system's own source -> one text document per module (`_self/`): the module docstring and the signature and first docstring line of every
                     top-level function and class. A model trained on it can answer questions about how Zero White is put together.

What this is not: the chat model does not CALL the zoo's programs as tools (it only reads about them), feedback given in chat (/good, /bad, /fix) does not update the forge's priors
(`forge/prior.py` learns from forge results, not from conversations), and a model this small is not shown to benefit from reading source code. The bridges are the plumbing; whether it
helps is for a training run to say. Output is deterministic (same inputs, same files) and capped in size.
"""
from __future__ import annotations

import ast
import os
from typing import Dict

from .. import fileio

MAX_DOC = 6000


def export_zoo(ws: str, out_dir: str, per_task: int = 5) -> Dict[str, int]:
    from ..forge.zoo import Zoo
    zoo = Zoo(ws)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for task in zoo.tasks():
        ents = sorted(zoo.entries(task), key=lambda e: -(e.get("skill") if isinstance(e.get("skill"), (int, float)) else -1e9))[:per_task]
        for e in ents:
            prog = os.path.join(zoo.task_dir(task), e["id"], "program.py")
            body = fileio.read_text(prog)[:3000] if os.path.exists(prog) else ""
            doc = (f"[zoo] task: {task}\nagent: {e['id']}\nfamily: {e.get('family')}\nskill: {e.get('skill')}\nwhat it is: {str(e.get('description', ''))[:600]}\n"
                   + (f"program:\n{body}\n" if body else ""))
            fileio.write_text(os.path.join(out_dir, f"zoo-{task}-{e['id']}.txt"), doc[:MAX_DOC])
            n += 1
    return dict(documents=n, tasks=len(zoo.tasks()))


def _summary(path: str, rel: str) -> str:
    try:
        tree = ast.parse(fileio.read_text(path))
    except (SyntaxError, OSError):
        return ""
    lines = [f"[source zerowhite/{rel}]"]
    doc = ast.get_docstring(tree)
    if doc:
        lines.append(doc[:900])
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            args = ", ".join(a.arg for a in node.args.args) if isinstance(node, ast.FunctionDef) else ""
            first = (ast.get_docstring(node) or "").strip().splitlines()[:1]
            lines.append(f"{'def' if isinstance(node, ast.FunctionDef) else 'class'} {node.name}({args})" + (f": {first[0][:160]}" if first else ""))
    return "\n".join(lines)[:MAX_DOC]


def ingest_self_source(out_dir: str, pkg_root: str = None) -> Dict[str, int]:
    pkg = pkg_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for root, dirs, files in os.walk(pkg):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", "evolved") and not d.startswith("."))
        for f in sorted(files):
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(root, f), pkg).replace(os.sep, "/")
                s = _summary(os.path.join(root, f), rel)
                if s:
                    fileio.write_text(os.path.join(out_dir, rel.replace("/", "__")[:-3] + ".txt"), s + "\n")
                    n += 1
    return dict(documents=n)
