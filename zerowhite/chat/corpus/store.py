"""store.py - the incremental ingestor behind the data folder.

    data/ (anything you drop in)  --Ingestor.ingest()-->  state/corpus/<name>.jsonl.gz   one shard per source file
                                                          state/manifest.json            what was ingested, hashes, drop counts
                                                          state/dedupe.sqlite            record hashes across all files and runs

Properties the daemon relies on:
* a file is only read once it has been unchanged for `data.settle_s` seconds (no half-copied files). 6.5: a timestamp far in the FUTURE (a zip extracted
  with future dates, a share whose clock runs fast) no longer means "still being copied" forever: such a file settles when its size and timestamp have been
  seen unchanged for `data.settle_s` seconds of this process's own clock;
* new / changed / deleted files are handled individually, the manifest is updated atomically after EACH file, so a crash
  or Ctrl-C never loses finished work and never leaves a partial shard;
* one broken file is reported and skipped, never fatal; it is retried only after it changes;
* a folder's dataset.json overrides adapter, column mapping, weight, filters ... for everything beneath it.
"""
from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional

from ... import fileio
from ..config import Config, _strip_json_comments
from ..paths import Layout, safe_name
from ..util import FileLock
from . import adapters as A
from . import readers as RD
from . import records as R
from .filters import RecordFilter

MANIFEST_VERSION = 1
FP_BYTES = 65536
FUTURE_SLACK_S = 60.0          # a modification time up to this far ahead of our clock is ordinary skew; beyond it the wall clock says nothing about the file


@dataclass
class FileEntry:
    relpath: str
    abspath: str
    size: int
    mtime_ns: int
    status: str                      # new | changed | unchanged | unsettled | ignored | audio_skipped
    kind: Optional[str] = None
    wait_s: float = 0.0              # unsettled files: seconds until they would count as settled if nothing touches them again


@dataclass
class ShardInfo:
    name: str
    path: str
    source: str
    kinds: Dict[str, int]
    records: int
    created: float


@dataclass
class IngestReport:
    files_new: int = 0
    files_changed: int = 0
    files_removed: int = 0
    files_unchanged: int = 0
    files_unsettled: int = 0
    unsettled_wait_s: float = 0.0    # the longest remaining wait among the unsettled files (what a caller should sleep before trying again)
    audio_skipped: int = 0
    records_in: int = 0
    records_out: int = 0
    dropped: Dict[str, int] = field(default_factory=dict)
    shards_written: List[str] = field(default_factory=list)
    errors: List[tuple] = field(default_factory=list)
    seconds: float = 0.0

    def add_drop(self, reason: str, n: int = 1):
        self.dropped[reason] = self.dropped.get(reason, 0) + n

    @property
    def changed_anything(self) -> bool:
        return bool(self.files_new or self.files_changed or self.files_removed)

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def summary(self) -> str:
        parts = [f"{self.files_new} new, {self.files_changed} changed, {self.files_removed} removed, {self.files_unchanged} unchanged"]
        if self.files_unsettled:
            parts.append(f"{self.files_unsettled} still being copied (will retry)")
        if self.audio_skipped:
            parts.append(f"{self.audio_skipped} audio files left for the voice trainer")
        parts.append(f"records {self.records_in:,} read -> {self.records_out:,} kept in {self.seconds:.1f}s")
        if self.dropped:
            parts.append("dropped: " + ", ".join(f"{k}={v:,}" for k, v in sorted(self.dropped.items(), key=lambda kv: -kv[1])))
        if self.errors:
            parts.append(f"{len(self.errors)} file(s) with errors")
        return "; ".join(parts)


def _fingerprint(path: str, size: int) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(FP_BYTES))
        if size > FP_BYTES * 2:
            f.seek(max(0, size - FP_BYTES))
            h.update(f.read(FP_BYTES))
    return h.hexdigest()


def _flatten(prefix: str, d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and f"{prefix}.{k}" not in ("data.schema_map", "data.weights"):
            out.update(_flatten(f"{prefix}.{k}", v))
        else:
            out[f"{prefix}.{k}"] = v
    return out


class Ingestor:
    def __init__(self, cfg: Config, layout: Layout):
        self.cfg, self.lay = cfg, layout
        os.makedirs(layout.corpus_dir, exist_ok=True)
        self._sidecar_cache: Dict[str, tuple] = {}
        self._seen: Dict[str, tuple] = {}            # far-future-dated files: path -> ((size, mtime_ns), first time we saw exactly that), on time.monotonic()

    # ---- manifest -----------------------------------------------------------------------------------------------
    def _load_manifest(self) -> dict:
        p = self.lay.manifest_path
        if os.path.exists(p):
            try:
                m = fileio.read_json(p)
                if m.get("version") == MANIFEST_VERSION:
                    return m
            except (OSError, ValueError):
                fileio.replace(p, p + f".corrupt-{int(time.time())}")
        return {"version": MANIFEST_VERSION, "files": {}, "last_ingest": None}

    def _save_manifest(self, m: dict):
        fileio.write_json(self.lay.manifest_path, m, indent=1)

    # ---- sidecar (dataset.json) --------------------------------------------------------------------------------------
    def sidecar_for(self, relpath: str) -> tuple:
        """(dict, hash) of the nearest dataset.json at or above the file's folder, within the data folder."""
        folder = os.path.dirname(relpath)
        while True:
            cand = os.path.join(self.lay.data, *([folder] if folder else []), "dataset.json")
            if os.path.isfile(cand):
                try:
                    raw = fileio.read_text(cand)
                    d = json.loads(_strip_json_comments(raw) or "{}")
                    if not isinstance(d, dict):
                        raise ValueError("dataset.json must hold a JSON object")
                    return d, hashlib.blake2b(raw.encode("utf-8"), digest_size=8).hexdigest()
                except ValueError as e:
                    raise A.AdapterError(f"{os.path.relpath(cand, self.lay.data)}: {e}") from e
            if not folder:
                return {}, ""
            folder = os.path.dirname(folder)

    def file_config(self, relpath: str, sidecar: dict) -> Config:
        cfg = self.cfg
        flat = {}
        flat.update(_flatten("filter", sidecar.get("filter", {})))
        flat.update(_flatten("data", sidecar.get("data", {})))
        if flat:
            cfg = cfg.merge(flat, f"dataset.json:{relpath}")
        return cfg

    # ---- discovery ----------------------------------------------------------------------------------------------
    def _settle_wait(self, path: str, st, now: float, settle: float) -> float:
        """Seconds until this file counts as settled (0 = settled now). Normally the wall-clock age of its modification time; a timestamp more than
        FUTURE_SLACK_S ahead of our clock carries no information, so then the file must simply look the same on every visit for `settle` seconds."""
        age = now - st.st_mtime
        if age >= -FUTURE_SLACK_S:
            return max(0.0, settle - age)
        key, sig, mono = os.path.normcase(os.path.abspath(path)), (st.st_size, st.st_mtime_ns), time.monotonic()
        prev = self._seen.get(key)
        if prev is None or prev[0] != sig:
            self._seen[key] = (sig, mono)
            return settle
        return max(0.0, settle - (mono - prev[1]))

    def discover(self, include_unsettled: bool = False, manifest: Optional[dict] = None) -> List[FileEntry]:
        cfg = self.cfg
        manifest = manifest or self._load_manifest()
        ignore = list(cfg.get("data.ignore"))
        exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in cfg.get("data.extensions")}
        settle = float(cfg.get("data.settle_s"))
        now = time.time()
        out: List[FileEntry] = []
        root = self.lay.data
        if not os.path.isdir(root):
            return out
        audio_root = os.path.normcase(os.path.abspath(self.lay.audio_data))          # speech datasets belong to `voice train-stt`, not to the text pipeline
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and os.path.normcase(os.path.abspath(os.path.join(dirpath, d))) != audio_root)
            if not cfg.get("data.recursive") and dirpath != root:
                dirnames[:] = []
                continue
            for fn in sorted(filenames):
                ap = os.path.join(dirpath, fn)
                rel = os.path.relpath(ap, root).replace(os.sep, "/")
                if os.path.islink(ap) and not os.path.realpath(ap).startswith(os.path.realpath(root)):
                    continue
                if any(fnmatch.fnmatch(fn, pat) for pat in ignore):
                    continue
                try:
                    st = os.stat(ap)
                except OSError:
                    continue
                kind = RD.file_kind(fn)
                if kind == "audio":
                    out.append(FileEntry(rel, ap, st.st_size, st.st_mtime_ns, "audio_skipped", kind))
                    continue
                _, ext, _ = RD.split_ext(fn)
                if kind is None or (exts and ext not in exts):
                    continue
                old = manifest["files"].get(rel)
                wait = 0.0 if include_unsettled else self._settle_wait(ap, st, now, settle)
                if wait > 0.0:
                    out.append(FileEntry(rel, ap, st.st_size, st.st_mtime_ns, "unsettled", kind, wait_s=wait))
                    continue
                if old is None:
                    status = "new"
                elif old.get("size") == st.st_size and old.get("mtime_ns") == st.st_mtime_ns:
                    status = "unchanged"
                else:
                    status = "changed"
                out.append(FileEntry(rel, ap, st.st_size, st.st_mtime_ns, status, kind))
        return out

    # ---- dedupe db ----------------------------------------------------------------------------------------------------
    def _db(self):
        con = sqlite3.connect(self.lay.dedupe_db, timeout=30)
        con.execute("CREATE TABLE IF NOT EXISTS seen (h BLOB PRIMARY KEY, shard TEXT) WITHOUT ROWID")
        con.execute("CREATE INDEX IF NOT EXISTS seen_shard ON seen(shard)")
        return con

    # ---- ingest -------------------------------------------------------------------------------------------------------
    def shard_name(self, relpath: str) -> str:
        return f"{safe_name(relpath.replace('/', '__'), 60)}-{hashlib.blake2b(relpath.encode('utf-8'), digest_size=4).hexdigest()}"

    def ingest(self, progress: Optional[Callable[[dict], None]] = None, force: bool = False,
               only: Optional[List[str]] = None) -> IngestReport:
        t0 = time.time()
        rep = IngestReport()
        with FileLock(os.path.join(self.lay.run, "ingest.lock")).acquire(wait_s=120):
            manifest = self._load_manifest()
            entries = self.discover(manifest=manifest)
            present = {e.relpath for e in entries}
            con = self._db()
            try:
                # removed files
                for rel in [r for r in list(manifest["files"]) if r not in present and not r.startswith("~")]:
                    if only and rel not in only:
                        continue
                    self._drop_source(manifest, con, rel)
                    rep.files_removed += 1
                    self._save_manifest(manifest)
                    if progress:
                        progress(dict(event="removed", path=rel))
                for e in entries:
                    if e.status == "audio_skipped":
                        rep.audio_skipped += 1
                        continue
                    if e.status == "unsettled":
                        rep.files_unsettled += 1
                        rep.unsettled_wait_s = max(rep.unsettled_wait_s, e.wait_s)
                        continue
                    if only and e.relpath not in only:
                        continue
                    old = manifest["files"].get(e.relpath)
                    try:
                        sidecar, side_hash = self.sidecar_for(e.relpath)
                    except A.AdapterError as err:
                        rep.errors.append((e.relpath, str(err)))
                        continue
                    if e.status == "unchanged" and not force:
                        if old and old.get("side", "") == side_hash:
                            rep.files_unchanged += 1
                            continue
                        e.status = "changed"                      # the sidecar changed
                    fp = _fingerprint(e.abspath, e.size)
                    if old and not force and e.status == "changed" and old.get("fp") == fp and old.get("side", "") == side_hash:
                        old.update(size=e.size, mtime_ns=e.mtime_ns)          # touched but identical
                        self._save_manifest(manifest)
                        rep.files_unchanged += 1
                        continue
                    entry = self._ingest_file(e, sidecar, side_hash, fp, con, rep)
                    if entry.get("status") == "error":
                        rep.errors.append((e.relpath, entry["error"]))
                    else:
                        if old:
                            rep.files_changed += 1
                        else:
                            rep.files_new += 1
                    manifest["files"][e.relpath] = entry
                    manifest["last_ingest"] = time.time()
                    self._save_manifest(manifest)
                    if progress:
                        progress(dict(event="file", path=e.relpath, status=entry.get("status"), records=entry.get("records_out", 0),
                                      dropped=entry.get("dropped", {}), error=entry.get("error")))
            finally:
                con.close()
        rep.seconds = time.time() - t0
        return rep

    def _drop_source(self, manifest: dict, con, rel: str):
        old = manifest["files"].pop(rel, None)
        if old and old.get("shard"):
            con.execute("DELETE FROM seen WHERE shard=?", (old["shard"],))
            con.commit()
            p = os.path.join(self.lay.corpus_dir, old["shard"] + ".jsonl.gz")
            if os.path.exists(p):
                os.remove(p)

    def forget(self, relpath: str):
        with FileLock(os.path.join(self.lay.run, "ingest.lock")).acquire(wait_s=60):
            m = self._load_manifest()
            con = self._db()
            try:
                self._drop_source(m, con, relpath)
            finally:
                con.close()
            self._save_manifest(m)

    def _ingest_file(self, e: FileEntry, sidecar: dict, side_hash: str, fp: str, con, rep: IngestReport) -> dict:
        shard = self.shard_name(e.relpath)
        entry = dict(size=e.size, mtime_ns=e.mtime_ns, fp=fp, side=side_hash, ingested=time.time())
        tmp = os.path.join(self.lay.corpus_dir, f".tmp_{shard}.jsonl.gz")
        final = os.path.join(self.lay.corpus_dir, shard + ".jsonl.gz")
        dropped: Dict[str, int] = {}
        kinds: Dict[str, int] = {}
        n_in = n_out = 0
        try:
            cfg = self.file_config(e.relpath, sidecar)
            filt = RecordFilter(cfg)
            top = e.relpath.split("/")[0] if "/" in e.relpath else ""
            weight = float(cfg.get("data.weights").get(top, 1.0)) * float(sidecar.get("weight", 1.0))
            mapping = dict(cfg.get("data.schema_map") or {})
            mapping.update(sidecar.get("mapping") or {})
            ctx = A.Ctx(cfg=cfg, relpath=e.relpath, mapping=mapping, system=sidecar.get("system"),
                        user_name=sidecar.get("user_name"), weight=weight, tags=list(sidecar.get("tags") or ([top] if top else [])),
                        force_kind=sidecar.get("kind"))
            sniffed = RD.sniff(e.abspath, int(cfg.get("data.sniff_rows")), cfg)
            forced = sidecar.get("adapter") or (cfg.get("data.adapter") if cfg.get("data.adapter") != "auto" else None)
            adapter, conf = A.detect_adapter(sniffed, ctx, forced)
            entry["adapter"] = adapter.name
            limit = sidecar.get("limit") or cfg.get("data.max_records_per_file")
            con.execute("DELETE FROM seen WHERE shard=?", (shard,))
            with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=3) as out:
                for idx, row in enumerate(RD.iter_rows(e.abspath, cfg)):
                    if limit and idx >= limit:
                        break
                    ctx.index = idx
                    for rec in A.convert_row(adapter, row, ctx):
                        n_in += 1
                        rec, why = filt.apply(rec)
                        if rec is None:
                            dropped[why] = dropped.get(why, 0) + 1
                            continue
                        if cfg.get("data.dedupe"):
                            cur = con.execute("INSERT OR IGNORE INTO seen VALUES (?,?)", (R.record_hash(rec), shard))
                            if cur.rowcount == 0:
                                dropped["duplicate"] = dropped.get("duplicate", 0) + 1
                                continue
                        out.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                        kinds[rec["kind"]] = kinds.get(rec["kind"], 0) + 1
                        n_out += 1
            con.commit()
            fileio.replace(tmp, final)
            entry.update(status="ok", shard=shard, records_in=n_in, records_out=n_out, kinds=kinds, dropped=dropped)
            rep.shards_written.append(shard)
        except Exception as ex:  # noqa: BLE001 - isolate per-file failures
            con.rollback()
            for p in (tmp,):
                if os.path.exists(p):
                    os.remove(p)
            msg = f"{type(ex).__name__}: {ex}"
            entry.update(status="error", error=msg, shard=None, records_out=0, dropped={})
            n_in = n_out = 0
            dropped = {}
        rep.records_in += n_in
        rep.records_out += n_out
        for k, v in dropped.items():
            rep.add_drop(k, v)
        return entry

    # ---- reading back ---------------------------------------------------------------------------------------------------
    def shards(self) -> List[ShardInfo]:
        out = []
        for rel, f in sorted(self._load_manifest()["files"].items()):
            if f.get("status") == "ok" and f.get("shard"):
                p = os.path.join(self.lay.corpus_dir, f["shard"] + ".jsonl.gz")
                if os.path.exists(p):
                    out.append(ShardInfo(f["shard"], p, rel, f.get("kinds", {}), f.get("records_out", 0), f.get("ingested", 0)))
        return out

    def iter_shard(self, shard: ShardInfo) -> Iterator[dict]:
        with gzip.open(shard.path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def iter_records(self, kinds: Optional[List[str]] = None, shard: Optional[str] = None) -> Iterator[dict]:
        for s in self.shards():
            if shard and s.name != shard:
                continue
            if kinds and not any(s.kinds.get(k) for k in kinds):
                continue
            for rec in self.iter_shard(s):
                if not kinds or rec["kind"] in kinds:
                    yield rec

    def stats(self) -> dict:
        m = self._load_manifest()
        kinds: Dict[str, int] = {}
        dropped: Dict[str, int] = {}
        by_source = {}
        errors = {}
        for rel, f in m["files"].items():
            if f.get("status") == "ok":
                by_source[rel] = dict(records=f.get("records_out", 0), adapter=f.get("adapter"), read=f.get("records_in", 0))
                for k, v in f.get("kinds", {}).items():
                    kinds[k] = kinds.get(k, 0) + v
                for k, v in f.get("dropped", {}).items():
                    dropped[k] = dropped.get(k, 0) + v
            else:
                errors[rel] = f.get("error")
        return dict(files=len(m["files"]), records=sum(kinds.values()), kinds=kinds, dropped=dropped, by_source=by_source,
                    errors=errors, last_ingest=m.get("last_ingest"))
