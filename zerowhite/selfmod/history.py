"""history.py - content-addressed, append-only version history for everything Zero White
rewrites about itself (hook functions, source patches, evolved programs).

  <root>/objects/ab/abcdef...   the exact text of one version (name = sha256 of the text)
  <root>/log.jsonl              append-only audit trail: every put and every rollback
  <root>/heads.json             name -> current version (replaced atomically)

Nothing is ever deleted: a rollback just moves the head back (and is itself logged),
so the full lineage of every self-modification stays inspectable. `verify()` re-hashes
every object, which is how `doctor` detects a corrupted or tampered store.
"""

import hashlib
import json
import os
import time

from .. import fileio


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class HistoryStore:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(os.path.join(self.root, "objects"), exist_ok=True)
        self._heads_path = os.path.join(self.root, "heads.json")
        self._log_path = os.path.join(self.root, "log.jsonl")

    # ---- plumbing ------------------------------------------------------------------
    def _heads(self):
        if not os.path.exists(self._heads_path):
            return {}
        try:
            return fileio.read_json(self._heads_path)
        except ValueError:
            return {}

    def _obj_path(self, sha):
        return os.path.join(self.root, "objects", sha[:2], sha)

    def _event(self, ev):
        ev = dict(ev, t=time.time())
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, default=fileio._json_default) + "\n")

    # ---- API -----------------------------------------------------------------------
    def put(self, name, text, meta=None):
        """Record `text` as the new head of `name`. Returns the version record."""
        sha = _sha(text)
        if not os.path.exists(self._obj_path(sha)):
            fileio.write_text(self._obj_path(sha), text)
        heads = self._heads()
        prev = heads.get(name)
        rec = dict(name=name, sha=sha, version=(prev["version"] + 1) if prev else 1,
                   parent=prev["sha"] if prev else None, meta=meta or {})
        heads[name] = dict(sha=sha, version=rec["version"])
        self._event(dict(rec, event="put"))
        fileio.write_json(self._heads_path, heads, indent=2)
        return rec

    def head(self, name):
        h = self._heads().get(name)
        return dict(h, name=name) if h else None

    def names(self):
        return sorted(self._heads())

    def get(self, sha):
        p = self._obj_path(sha)
        if not os.path.exists(p):
            raise KeyError(f"unknown version {sha[:12]}")
        text = fileio.read_text(p)
        if _sha(text) != sha:
            raise ValueError(f"object {sha[:12]} is corrupt (hash mismatch)")
        return text

    def text(self, name):
        h = self.head(name)
        return None if h is None else self.get(h["sha"])

    def log(self, name=None):
        if not os.path.exists(self._log_path):
            return []
        out = []
        for line in fileio.read_text(self._log_path).splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue                                          # a torn last line after a hard kill
            if name is None or ev.get("name") == name:
                out.append(ev)
        return out

    def lineage(self, name):
        """Versions of `name` from first to current, following parent links (rollbacks excluded)."""
        # 6.3: keyed by (version, content), not by content alone. Recording the SAME text twice (original -> patch -> original again, which is what
        # the patcher's auto-rollback does) made the old walk, keyed by content, go round in a circle forever: any later `lineage()` call hung
        # and its list grew until the process was killed. A version number plus the parent's hash identifies one step uniquely, and `seen` makes a
        # damaged log end the walk instead of looping.
        puts = [e for e in self.log(name) if e["event"] == "put"]
        by_step = {(e["version"], e["sha"]): e for e in puts}                    # a later put of the same (version, content) replaces an earlier one
        h = self.head(name)
        chain, seen = [], set()
        step = (h["version"], h["sha"]) if h else None
        while step is not None and step in by_step and step not in seen:
            seen.add(step)
            e = by_step[step]
            chain.append(e)
            step = (e["version"] - 1, e["parent"]) if e.get("parent") else None
        return list(reversed(chain))

    def rollback(self, name, steps=1, to_sha=None):
        """Move `name` back `steps` versions (or to an exact sha). Returns the new head record."""
        chain = self.lineage(name)
        if to_sha is not None:
            hit = [e for e in chain if e["sha"].startswith(to_sha)]
            if not hit:
                raise KeyError(f"{to_sha} is not in the lineage of {name}")
            target = hit[-1]
        else:
            if len(chain) <= steps:
                raise ValueError(f"cannot roll '{name}' back {steps} step(s): only {len(chain)} version(s) recorded")
            target = chain[-1 - steps]
        heads = self._heads()
        cur = heads.get(name)
        heads[name] = dict(sha=target["sha"], version=target["version"])
        self._event(dict(event="rollback", name=name, sha=target["sha"], version=target["version"],
                         from_sha=cur["sha"] if cur else None))
        fileio.write_json(self._heads_path, heads, indent=2)
        return dict(heads[name], name=name)

    def verify(self):
        """Re-hash every object and check every head resolves. Returns a list of problem strings."""
        problems = []
        base = os.path.join(self.root, "objects")
        for d in sorted(os.listdir(base)):
            for fn in os.listdir(os.path.join(base, d)):
                try:
                    if _sha(fileio.read_text(os.path.join(base, d, fn))) != fn:
                        problems.append(f"object {fn[:12]} does not match its hash")
                except (OSError, UnicodeDecodeError) as e:
                    problems.append(f"object {fn[:12]} unreadable: {e}")
        for name, h in self._heads().items():
            if not os.path.exists(self._obj_path(h["sha"])):
                problems.append(f"head of '{name}' points at a missing object")
        return problems
