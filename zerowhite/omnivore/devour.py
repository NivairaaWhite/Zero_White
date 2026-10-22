"""devour.py - hand it anything; it gives back tasks and an honest report. It does not raise on what it is given.

  devour(paths, workspace)  ->  Digest
      walks files and folders (nothing outside what you name; symlinks and the workspace itself are not followed),
      identifies each file by content, opens archives safely (and salvages damaged ones), repairs what is repairable,
      turns tables into supervised tasks, text into a growing corpus, your environment files into environments,
      and sets the rest aside with a reason. Re-running costs nothing for files that have not changed.

Each file has exactly one outcome:  task | pooled | expanded | duplicate | unchanged | deferred | quarantined | skipped | missing
`deferred` means "not yet" (still being written; needs --allow-code); everything else is final until the file's content changes.
"""
from __future__ import annotations

import io
import os
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import archives, sniff, tables
from .library import FINAL, Library, _safe
from .limits import Limits

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".idea", ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints"}
TEMP_SUFFIXES = (".part", ".crdownload", ".tmp", ".temp", ".partial", ".download", "~", ".swp", ".lock")

_UNSUPPORTED = {
    "image": "image: pixels need labels to become a task (image tasks come from labelled datasets: `solve --dataset NAME`); not read",
    "pdf": "PDF: text cannot be extracted without a PDF library; export it to text and drop that in instead",
    "audio": "audio: it belongs in the chat model's data/audio folder together with transcripts (docs/VOICE.md); not read here",
    "parquet": "parquet needs the optional pyarrow package (`pip install pyarrow`), or export the table to CSV",
    "executable": "executable code is never run, and is not training data",
}

_SMOKE = '''
import numpy as np
_SRC = {src!r}
ns = {{"__name__": "zw_user_env"}}
exec(compile(_SRC, "env_file", "exec"), ns)
env = ns["Env"]()
rng = np.random.default_rng(0)
o = env.reset(rng)
assert np.asarray(o, dtype=float).ravel().shape[0] == int(env.obs_dim), "reset() must return obs_dim numbers"
for _ in range(8):
    o, r, d, w = env.step(0)
    float(r)
    if d:
        o = env.reset(rng)
print("ENV-OK", int(env.obs_dim), int(env.n_actions))
'''


@dataclass
class Entry:
    label: str
    status: str = "quarantined"
    kind: str = ""
    task: str = ""
    reason: str = ""
    repairs: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    sha: str = ""

    def to_dict(self) -> dict:
        return dict(label=self.label, status=self.status, kind=self.kind, task=self.task, reason=self.reason,
                    repairs=list(self.repairs), notes=list(self.notes), sha=self.sha[:16])


@dataclass
class Digest:
    entries: list = field(default_factory=list)
    tasks: list = field(default_factory=list)                 # names of tasks created or updated by this run
    healed: str = ""
    seconds: float = 0.0
    limits: dict = field(default_factory=dict)

    def counts(self) -> dict:
        return dict(Counter(e.status for e in self.entries))

    def to_dict(self) -> dict:
        return dict(counts=self.counts(), tasks=self.tasks, healed=self.healed, seconds=round(self.seconds, 2),
                    entries=[e.to_dict() for e in self.entries], limits=self.limits)

    def summary(self, max_lines: int = 60) -> str:
        c = self.counts()
        order = ("task", "pooled", "expanded", "duplicate", "unchanged", "deferred", "skipped", "quarantined", "missing")
        head = ", ".join(f"{c[k]} {k}" for k in order if k in c) or "nothing to digest"
        lines = [f"digested {len(self.entries)} item(s) in {self.seconds:.1f}s: {head}"]
        if self.healed:
            lines.append(f"  self-healed: {self.healed}")
        shown = 0
        for e in self.entries:
            if e.status in ("unchanged", "skipped") and len(self.entries) > 12:
                continue
            if shown >= max_lines:
                lines.append(f"  ... {len(self.entries) - shown} more (see the report with --json)")
                break
            shown += 1
            what = e.task and f"-> {e.task}" or e.reason
            lines.append(f"  {e.status:<11} {e.label[-58:]:<58} {what[:110]}")
            for r in e.repairs[:4]:
                lines.append(f"  {'':<11} {'':<58} repaired: {r[:120]}")
        return "\n".join(lines)


class _Devourer:
    def __init__(self, workspace, limits, allow_code, target, pool, retry, log, seed, learned="rescue"):
        self.learned = learned
        self.ws = os.path.abspath(workspace)
        self._ws_key = os.path.normcase(os.path.realpath(workspace))
        self.lib = Library(workspace)
        self.limits, self.allow_code, self.target = limits, allow_code, target
        self.pool, self.retry, self.log, self.seed = pool, retry, log, seed
        self.entries: list = []
        self.changed: set = set()
        self._since_save = 0

    # ------------------------------------------------------------------------------------------- recording
    def _new(self, label, **kw) -> Entry:
        e = Entry(label=label, **kw)
        self.entries.append(e)
        return e

    def _seal(self, e: Entry, sha: str, retry_when: str = ""):
        """Write the outcome into the ledger (and the quarantine log). Deferred outcomes are not remembered as final."""
        e.sha = sha
        if e.status == "deferred":
            return
        self.lib.record(sha, label=e.label, status=e.status, kind=e.kind, task=e.task, reason=e.reason, retry_when=retry_when,
                        notes=e.notes[:8], repairs=e.repairs[:8])
        if e.status == "quarantined":
            self.lib.quarantine(label=e.label, sha=sha, kind=e.kind, reason=e.reason, retry_when=retry_when or "content changes")
        self._since_save += 1
        if self._since_save >= 25:
            self.lib.save()
            self._since_save = 0
        if self.log:
            self.log(f"  {e.status:<11} {e.label[-60:]}  {e.task and '-> ' + e.task or e.reason[:80]}")

    def _quarantine(self, e: Entry, reason: str, sha: str, kind: str = "", retry_when: str = "") -> Entry:
        e.status, e.reason = "quarantined", reason
        e.kind = kind or e.kind
        self._seal(e, sha, retry_when)
        return e

    # ------------------------------------------------------------------------------------------- walking
    def _in_workspace(self, path: str) -> bool:
        """True for the workspace folder itself and everything under it (compared the way this file system compares names)."""
        key = os.path.normcase(os.path.realpath(path))
        return key == self._ws_key or key.startswith(self._ws_key + os.sep)

    def run(self, paths) -> None:
        count = 0
        for p in paths:
            p = os.fspath(p)
            if not os.path.exists(p):
                self._new(p, status="missing", reason="this path does not exist")
                continue
            if os.path.isdir(p):
                for root, dirs, files in os.walk(p, followlinks=False):
                    dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not self._in_workspace(os.path.join(root, d)))
                    for f in sorted(files):
                        count += 1
                        if count > self.limits.max_files:
                            self._new(root, status="skipped", reason=f"more than {self.limits.max_files} files: the rest were not visited")
                            return
                        self._file_safely(os.path.join(root, f))
            else:
                self._file_safely(p)

    def _fail(self, n_before: int, label: str, sha: str, ex: BaseException):
        """Turn an unexpected exception into one clean quarantine entry (and drop the half-built ones it left behind)."""
        tail = traceback.format_exc(limit=3).strip().splitlines()[-1]
        reason = f"internal error while digesting ({type(ex).__name__}: {ex}); {tail}"[:300]
        mine = [e for e in self.entries[n_before:] if e.label == label]
        self.entries[n_before:] = [e for e in self.entries[n_before:] if not (e.status == "quarantined" and not e.reason)]
        e = mine[0] if mine and mine[0] in self.entries else self._new(label)
        self._quarantine(e, reason, sha=sha)

    def _file_safely(self, path: str):
        n_before = len(self.entries)
        try:
            self._file(path)
        except Exception as ex:                                  # noqa: BLE001 - nothing a file contains may stop the run
            self._fail(n_before, path, f"err:{os.path.abspath(path)}", ex)
        if len(self.entries) == n_before:
            self._new(path, status="skipped", reason="nothing to do")

    def _file(self, path: str):
        ap = os.path.abspath(path)
        name = os.path.basename(path)
        if self._in_workspace(path) and not os.path.islink(path):
            return
        if os.path.islink(path):
            self._new(path, status="skipped", reason="symbolic link: not followed")
            return
        if name.startswith(".") or name.lower().endswith(TEMP_SUFFIXES):
            self._new(path, status="skipped", reason="hidden or temporary file")
            return
        try:
            st = os.stat(path)
        except OSError as ex:
            self._quarantine(self._new(path), f"cannot be read ({type(ex).__name__}: {ex})", sha=f"stat:{ap}")
            return
        fp = [int(st.st_size), int(st.st_mtime_ns)]
        prev = self.lib.path_state(path)
        if prev and prev["fp"] == fp and not self.retry:
            old = self.lib.entry(prev["sha"])
            if old and old.get("status") in FINAL and not (old.get("retry_when") == "allow_code" and self.allow_code):
                self._new(path, status="unchanged", kind=old.get("kind", ""), task=old.get("task", ""), reason=old.get("reason", ""), sha=prev["sha"])
                return
        age = time.time() - st.st_mtime
        if -60.0 < age < self.limits.settle_seconds:        # a slightly *future* mtime (coarse timestamps, FAT rounds up) also means "just written"
            self._new(path, status="deferred", reason=f"modified {max(age, 0.0):.1f}s ago: it may still be being written; it will be looked at again")
            return
        notes = []
        try:
            with open(path, "rb") as f:
                data = f.read(self.limits.max_file_bytes + 1)
        except OSError as ex:
            self._quarantine(self._new(path), f"cannot be read ({type(ex).__name__}: {ex})", sha=f"read:{ap}:{fp}")
            return
        if len(data) > self.limits.max_file_bytes:
            data = data[:self.limits.max_file_bytes]
            notes.append(f"larger than {self.limits.max_file_bytes // (1 << 20)} MiB: only the first part was read")
        sha = sniff.sha256(data + (str(st.st_size).encode() if notes else b""))
        ent = self._digest(path, data, sha, 0, notes)
        if ent is not None and ent.status != "deferred":
            self.lib.note_path(path, fp[0], fp[1], sha)

    # ------------------------------------------------------------------------------------------- one blob of bytes
    def _digest(self, label: str, data: bytes, sha: str, depth: int, notes: list, repairs: list = ()):
        prior = self.lib.entry(sha)
        if prior and not self.retry and prior.get("status") in FINAL:
            if prior.get("retry_when") == "allow_code" and not self.allow_code:
                return self._new(label, status="deferred", kind=prior.get("kind", ""), reason=prior.get("reason", ""), sha=sha)
            reopen = prior.get("retry_when") == "allow_code" and self.allow_code
            missing = prior.get("status") == "task" and prior.get("task") and self.lib.spec(prior["task"]) is None
            if not reopen and not missing:
                same = prior.get("label") == label
                e = self._new(label, status="unchanged" if same else "duplicate", kind=prior.get("kind", ""), task=prior.get("task", ""),
                              reason=prior.get("reason", "") if same else f"same content as {prior.get('label')}", sha=sha)
                if not same and prior.get("status") == "quarantined":
                    e.notes.append(f"set aside earlier: {prior.get('reason', '')}")
                return e
        e = self._new(label, notes=list(notes), repairs=list(repairs), sha=sha)
        kind = sniff.identify(data)
        e.kind = kind
        if kind == "empty":
            return self._quarantine(e, "empty file", sha)
        if kind in ("gzip", "bz2", "xz", "zip", "tar"):
            return self._archive(e, label, data, sha, kind, depth)
        if kind == "npy":
            return self._npy(e, label, data, sha)
        if kind == "sqlite":
            return self._sqlite(e, label, data, sha)
        if kind in _UNSUPPORTED:
            return self._quarantine(e, _UNSUPPORTED[kind], sha)
        return self._text(e, label, data, sha)

    # ------------------------------------------------------------------------------------------- containers
    def _archive(self, e: Entry, label: str, data: bytes, sha: str, kind: str, depth: int) -> Entry:
        if depth >= self.limits.max_depth:
            return self._quarantine(e, f"archives nested more than {self.limits.max_depth} levels deep: not opened", sha)
        n_ok = n_bad = 0
        base = os.path.basename(label.split("!/")[-1])
        for m in archives.expand(data, base, kind, self.limits):
            mlabel = f"{label}!/{m.name}"
            if m.error:
                n_bad += 1
                self._quarantine(self._new(mlabel, notes=list(m.notes)), m.error, sniff.sha256(sha.encode() + m.name.encode()))
                continue
            if m.name in ("(rest of tar)", "(tar)"):
                e.notes.extend(m.notes)
                continue
            n_ok += 1
            self._digest_member(mlabel, m, depth + 1)
        e.kind = kind
        e.notes.append(f"{n_ok} file(s) opened" + (f", {n_bad} unreadable" if n_bad else ""))
        if n_ok == 0 and n_bad == 0:
            return self._quarantine(e, "the archive holds no files", sha)
        e.status = "expanded"
        self._seal(e, sha)
        return e

    def _digest_member(self, label: str, m: archives.Member, depth: int):
        n_before = len(self.entries)
        try:
            self._digest(label, m.data, sniff.sha256(m.data), depth, [], list(m.notes))
        except Exception as ex:                                  # noqa: BLE001
            self._fail(n_before, label, sniff.sha256(m.data) + "!err", ex)

    def _npy(self, e: Entry, label: str, data: bytes, sha: str) -> Entry:
        try:
            arr = np.load(io.BytesIO(data), allow_pickle=False)
        except Exception as ex:                                  # noqa: BLE001 - truncated, pickled objects, anything
            return self._quarantine(e, f"numpy file could not be read safely ({type(ex).__name__}: {str(ex)[:80]})", sha)
        tab = tables.table_from_array(arr)
        if tab is None:
            return self._quarantine(e, f"numpy array of shape {tuple(arr.shape)}: only a 2-D numeric array (rows x columns) can become a task", sha)
        return self._table_or_quarantine(e, label, sha, tab, base=os.path.splitext(os.path.basename(label.split("!/")[-1]))[0])

    def _sqlite(self, e: Entry, label: str, data: bytes, sha: str) -> Entry:
        fd, tmp = tempfile.mkstemp(suffix=".db", prefix="zw_sqlite_")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            try:
                found = tables.read_sqlite(tmp, self.limits.max_rows)
            except tables.Unusable as u:
                return self._quarantine(e, u.reason, sha)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        if not found:
            return self._quarantine(e, "SQLite database with no table of at least 3 rows and 2 columns", sha)
        stem = os.path.splitext(os.path.basename(label.split("!/")[-1]))[0]
        for tname, tab in found:
            child = self._new(f"{label}!/{tname}", kind="sqlite-table", notes=list(tab.notes))
            self._table_or_quarantine(child, child.label, sniff.sha256(sha.encode() + tname.encode()), tab, base=f"{stem}_{tname}")
        e.status, e.kind = "expanded", "sqlite"
        e.notes.append(f"{len(found)} table(s)")
        self._seal(e, sha)
        return e

    # ------------------------------------------------------------------------------------------- text-like content
    def _text(self, e: Entry, label: str, data: bytes, sha: str) -> Entry:
        dec = sniff.decode(data)
        if dec is None:
            strings = sniff.printable_strings(data)
            return self._quarantine(e, f"binary data in no known format ({len(data):,} bytes, entropy {sniff.entropy_bits(data):.1f} bits/byte, "
                                       f"{len(strings)} readable string(s))", sha, kind="binary")
        e.kind = "text"
        e.repairs.extend(dec.repairs)
        text = dec.text
        base = os.path.splitext(os.path.basename(label.split("!/")[-1]))[0] or "data"
        sub, info = sniff.classify_text(text)
        if sub == "empty":
            return self._quarantine(e, "only whitespace", sha)
        if sub in ("json", "jsonl"):
            e.kind = sub
            e.repairs.extend(info["repairs"])
            tab = tables.table_from_records(info["records"], self.limits.max_rows)
            if tab is None:
                return self._pool(e, label, sha, text, "JSON without a table in it: kept as text")
            return self._table_or_text(e, label, sha, tab, base, text)
        if sub == "python":
            e.kind = "python"
            if info.get("has_env"):
                return self._env(e, label, sha, text, base)
            return self._pool(e, label, sha, text, "Python source without an Env class: kept as text")
        from . import learned                                                 # 6.4: the rules run first and are unchanged; see learned.py for what the ranker may do
        tab = learned.parse_with_hint(text, data, self.learned, self.limits.max_rows)
        if tab is not None:
            e.kind = "table"
            return self._table_or_text(e, label, sha, tab, base, text)
        return self._pool(e, label, sha, text, "")

    def _table_or_text(self, e, label, sha, tab, base, text):
        try:
            return self._table(e, label, sha, tab, base)
        except tables.Unusable as u:
            if u.textlike:
                return self._pool(e, label, sha, text, f"looked like a table but has no usable columns ({u.reason}); kept as text")
            return self._quarantine(e, u.reason, sha)

    def _table_or_quarantine(self, e, label, sha, tab, base):
        try:
            return self._table(e, label, sha, tab, base)
        except tables.Unusable as u:
            return self._quarantine(e, u.reason, sha)

    def _table(self, e: Entry, label: str, sha: str, tab: tables.Table, base: str) -> Entry:
        built = tables.build(tab, self.limits, target=self.target, seed=self.seed)
        name = self.lib.unique_name(base, label)
        self.lib.add_table(name, built, label, sha)
        e.status, e.task = "task", name
        e.kind = e.kind if e.kind not in ("text", "") else "table"
        e.repairs.extend(built.repairs)
        e.notes.extend(built.notes)
        self.changed.add(name)
        self._seal(e, sha)
        return e

    def _pool(self, e: Entry, label: str, sha: str, text: str, why: str) -> Entry:
        cleaned, reps = sniff.clean_text(text)
        e.repairs.extend(reps)
        if len(cleaned.strip()) < 20:
            return self._quarantine(e, f"almost no text ({len(cleaned.strip())} characters)", sha)
        name, total = self.lib.add_text(self.pool, cleaned, label, sha, self.limits.min_text_chars)
        e.status, e.task = "pooled", name or ""
        if why:
            e.notes.append(why)
        e.notes.append(f"corpus '{self.pool}' now holds {total:,} characters" + ("" if name else f" (a text task needs {self.limits.min_text_chars:,})"))
        if name:
            self.changed.add(name)
        self._seal(e, sha)
        return e

    # ------------------------------------------------------------------------------------------- your environments
    def _env(self, e: Entry, label: str, sha: str, text: str, base: str) -> Entry:
        if not self.allow_code:
            e.status = "deferred"
            e.reason = "defines an environment (class Env); running it executes your code, so it waits for --allow-code"
            self.lib.quarantine(label=label, sha=sha, kind="env", reason=e.reason, retry_when="allow_code")
            self.lib.record(sha, label=label, status="quarantined", kind="env", reason=e.reason, retry_when="allow_code")
            return e
        from ..selfmod import sandbox
        res = sandbox.run_code(_SMOKE.format(src=text), timeout=30)
        if not res.ok or "ENV-OK" not in res.stdout:
            tail = (res.stderr or res.stdout).strip().splitlines()[-1:] or ["no output"]
            return self._quarantine(e, f"the environment failed a smoke test in the sandbox ({'timed out' if res.timed_out else tail[0][:160]})", sha)
        parts = res.stdout.split("ENV-OK")[-1].split()
        name = self.lib.unique_name(base, label)
        self.lib.add_env(name, text, label, sha, dict(obs_dim=int(parts[0]), n_actions=int(parts[1])))
        e.status, e.task, e.kind = "task", name, "env"
        e.notes.append(f"environment: {parts[0]} observation value(s), {parts[1]} action(s); smoke-tested in the sandbox")
        self.changed.add(name)
        self._seal(e, sha)
        return e


def default_limits(workspace: str) -> Limits:
    """Caps for the machine this runs on (measured once per machine and cached in the workspace)."""
    try:
        from .. import substrate
        return Limits.for_machine(substrate.plan(substrate.cached(workspace)))
    except Exception:                                            # noqa: BLE001 - a failed probe means conservative built-in caps
        return Limits()


def devour(paths, workspace: str, limits: Limits | None = None, allow_code: bool = False, target=None, pool: str = "notes",
           retry: bool = False, log=None, seed: int = 0, learned: str = "rescue") -> Digest:
    """Digest every file under `paths`. Never raises on content; returns what happened to each file."""
    t0 = time.time()
    os.makedirs(workspace, exist_ok=True)
    limits = limits or default_limits(workspace)
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    d = _Devourer(workspace, limits, allow_code, target, _safe(pool), retry, log, seed, learned)
    try:
        d.run(paths)
    finally:
        try:
            d.lib.save()
        except OSError:
            pass
    return Digest(entries=d.entries, tasks=sorted(d.changed), healed=d.lib.healed or "", seconds=time.time() - t0, limits=limits.as_dict())


def load_tasks(workspace: str, names=None, allow_code: bool = False, limits: Limits | None = None, seed: int = 0, quiet: bool = True) -> list:
    """Task objects (ready for `Generalist.solve`) for the given library task names, or all of them. Tasks that cannot be built are skipped."""
    lib = Library(workspace)
    limits = limits or default_limits(workspace)
    out = []
    for n in (names if names is not None else lib.names()):
        try:
            out.append(lib.load_task(n, limits=limits, allow_code=allow_code, seed=seed))
        except (PermissionError, KeyError, ValueError, OSError) as ex:
            if not quiet:
                print(f"[omnivore] task '{n}' skipped: {ex}")
    return out
