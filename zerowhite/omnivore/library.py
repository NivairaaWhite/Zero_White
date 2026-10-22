"""library.py - what has been digested, what was set aside and why, and the tasks made from it.

  <workspace>/omnivore/ledger.json        content hash -> what happened to it (and path -> (size, mtime, hash) so unchanged files cost nothing)
  <workspace>/omnivore/quarantine.jsonl   append-only: every file that was set aside, with the reason and a fingerprint
  <workspace>/omnivore/tasks/<name>/      spec.json + data.npz (table) | corpus.txt (text) | env.py (environment)

Source files are NEVER moved, edited or deleted: "quarantine" is a record, not an action on the file. A quarantined file is looked at
again automatically when its content changes (a half-written file that is finished later), or on request.

Self-healing: a ledger that cannot be parsed is renamed `ledger.json.corrupt-<time>` and rebuilt from scratch (every file is then simply
digested again); the original is kept.
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np

from .. import fileio

FINAL = {"task", "pooled", "quarantined", "expanded", "duplicate"}


def _safe(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._-")
    return (s or "task")[:60]


class Library:
    def __init__(self, workspace: str):
        self.root = os.path.join(os.path.abspath(workspace), "omnivore")
        self.tasks_dir = os.path.join(self.root, "tasks")
        os.makedirs(self.tasks_dir, exist_ok=True)
        self.ledger_path = os.path.join(self.root, "ledger.json")
        self.quarantine_path = os.path.join(self.root, "quarantine.jsonl")
        self.healed = None
        self.ledger = self._load()

    # ------------------------------------------------------------------------------------------- ledger
    def _load(self) -> dict:
        empty = dict(version=1, files={}, paths={})
        if not os.path.exists(self.ledger_path):
            return empty
        try:
            led = fileio.read_json(self.ledger_path)
            if not isinstance(led, dict) or not isinstance(led.get("files"), dict) or not isinstance(led.get("paths"), dict):
                raise ValueError("unexpected ledger structure")
            return led
        except (ValueError, OSError, UnicodeDecodeError) as e:
            dest = f"{self.ledger_path}.corrupt-{int(time.time())}"
            try:
                fileio.replace(self.ledger_path, dest)
                self.healed = f"the ledger was unreadable ({type(e).__name__}); kept as {os.path.basename(dest)} and started a new one"
            except OSError:
                self.healed = f"the ledger was unreadable ({type(e).__name__}); a new one is used"
            return empty

    def save(self):
        fileio.write_json(self.ledger_path, self.ledger)

    def entry(self, sha: str):
        return self.ledger["files"].get(sha)

    def record(self, sha: str, **fields):
        cur = self.ledger["files"].get(sha, {})
        cur.update(fields, t=time.time())
        self.ledger["files"][sha] = cur

    def path_state(self, path: str):
        return self.ledger["paths"].get(os.path.abspath(path))

    def note_path(self, path: str, size: int, mtime_ns: int, sha: str):
        self.ledger["paths"][os.path.abspath(path)] = dict(fp=[int(size), int(mtime_ns)], sha=sha)

    def quarantine(self, **event):
        try:
            with open(self.quarantine_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(event, t=time.time()), default=str) + "\n")
        except OSError:
            pass

    def quarantined(self) -> list:
        out = []
        if os.path.exists(self.quarantine_path):
            for line in open(self.quarantine_path, "r", encoding="utf-8", errors="replace"):
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass                                          # a torn last line from a hard kill
        return out

    # ------------------------------------------------------------------------------------------- tasks
    def task_dir(self, name: str) -> str:
        return os.path.join(self.tasks_dir, _safe(name))

    def spec(self, name: str):
        try:
            return fileio.read_json(os.path.join(self.task_dir(name), "spec.json"))
        except (OSError, ValueError):
            return None

    def names(self) -> list:
        return sorted(d for d in os.listdir(self.tasks_dir) if os.path.exists(os.path.join(self.tasks_dir, d, "spec.json")))

    def unique_name(self, base: str, source_key: str) -> str:
        """A task name for `source_key`: the same source keeps its name when it is re-digested; a different source never takes it."""
        base = _safe(base)
        for cand in (base, f"{base}-{abs(hash_str(source_key)) % 10 ** 6:06d}"):
            sp = self.spec(cand)
            if sp is None or sp.get("source_key") == source_key:
                return cand
        n = 2
        while self.spec(f"{base}-{n}") is not None:
            n += 1
        return f"{base}-{n}"

    def add_table(self, name: str, built, source_key: str, sha: str) -> str:
        d = self.task_dir(name)
        os.makedirs(d, exist_ok=True)
        fileio.save_npz(os.path.join(d, "data.npz"), dict(X=built.X, y=built.y))
        fileio.write_json(os.path.join(d, "spec.json"), dict(name=name, kind="table", source_key=source_key, sha=sha, updated=time.time(),
                                                             notes=built.notes, repairs=built.repairs, **built.spec), indent=2)
        return name

    def add_env(self, name: str, source_text: str, source_key: str, sha: str, info: dict) -> str:
        d = self.task_dir(name)
        os.makedirs(d, exist_ok=True)
        fileio.write_text(os.path.join(d, "env.py"), source_text)
        fileio.write_json(os.path.join(d, "spec.json"), dict(name=name, kind="env", task_kind="env", source_key=source_key, sha=sha,
                                                             updated=time.time(), **info), indent=2)
        return name

    def add_text(self, pool: str, text: str, label: str, sha: str, min_chars: int):
        """Append `text` to the corpus called `pool`. -> (task name or None, total characters)."""
        name = _safe(f"corpus-{pool}")
        d = self.task_dir(name)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "corpus.txt")
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(text if text.endswith("\n") else text + "\n")
            f.write("\n")
        total = os.path.getsize(path)
        pend = os.path.join(d, ".pending.json")
        sp = self.spec(name)
        if sp is None and os.path.exists(pend):
            try:
                sp = fileio.read_json(pend)
            except (OSError, ValueError):
                sp = None
        sp = sp or dict(name=name, kind="text", task_kind="text", sources=[])
        sp["sources"] = (sp.get("sources") or [])[-200:] + [dict(label=label, sha=sha[:12], chars=len(text))]
        sp.update(updated=time.time(), bytes=total, source_key=f"pool:{pool}")
        if total >= min_chars:
            fileio.write_json(os.path.join(d, "spec.json"), sp, indent=2)
            if os.path.exists(pend):
                os.remove(pend)
            return name, total
        fileio.write_json(pend, sp, indent=2)                                  # not a task yet: too little text
        return None, total

    def load_task(self, name: str, limits=None, allow_code: bool = False, seed: int = 0):
        """Rebuild the task object for `name` (deterministic: the same split every time, so an incumbent is judged on the same data)."""
        from ..forge import task as T
        sp = self.spec(name)
        if sp is None:
            raise KeyError(f"no task called '{name}' in {self.tasks_dir}")
        d = self.task_dir(name)
        kind = sp["kind"]
        if kind == "table":
            z = fileio.load_npz(os.path.join(d, "data.npz"))
            t = T.from_arrays(name, z["X"], z["y"], sp["task_kind"], seed=seed)
            if sp["task_kind"] == "classification":
                t.class_names = tuple(sp.get("classes") or ())
            t.meta = dict(spec=sp)
            return t
        if kind == "text":
            with open(os.path.join(d, "corpus.txt"), "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            cap = getattr(limits, "max_text_chars", 2_000_000)
            if len(text) > cap:
                text = text[-cap:]                                 # the most recent text: it is what the user added last
            return T.TextTask(name, text)
        if kind == "env":
            if not allow_code:
                raise PermissionError(f"task '{name}' is an environment written by you: loading it runs your code, which needs allow_code=True "
                                      f"(`--allow-code`)")
            t = T.load_env_file(os.path.join(d, "env.py"))
            t.name = name                                          # the zoo keys agents by task name: two env files must not both be "env"
            return t
        raise ValueError(f"task '{name}' has an unknown kind '{kind}'")


def hash_str(s: str) -> int:
    """A stable (process-independent) hash of a string; Python's own str hash is randomised per process."""
    h = 1469598103934665603
    for b in s.encode("utf-8", "replace"):
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h
