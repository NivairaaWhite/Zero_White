"""readers.py - stream rows out of any supported file without loading it whole.

Every reader yields dicts. Text-like files (code, markdown, plain text) yield {"text": chunk, "_ext": ".py", "_path": name}.
Compressed variants (.gz .bz2 .xz) are transparent; .zip/.tar archives are walked member by member.
Rows never contain keys starting with "_" except the reader's own metadata.
"""
from __future__ import annotations

import bz2
import csv
import gzip
import io
import itertools
import json
import lzma
import os
import sys
import tarfile
import zipfile
from typing import Dict, Iterator, List, Optional

JSON_EXT = {".jsonl", ".ndjson", ".json"}
TABLE_EXT = {".csv", ".tsv"}
PARQUET_EXT = {".parquet"}
ARCHIVE_EXT = {".zip", ".tar", ".tgz"}
TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".text", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cc", ".cpp", ".h", ".hpp",
            ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".bash", ".ps1", ".sql", ".html", ".htm", ".css", ".scss", ".kt", ".swift",
            ".lua", ".r", ".tex", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml"}
CODE_EXT = TEXT_EXT - {".txt", ".md", ".markdown", ".rst", ".text", ".tex"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".aif", ".webm"}
COMPRESSION = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}
MAX_MEMBER_BYTES = 2 << 30          # one archive member larger than this is skipped (zip-bomb guard)
MAX_JSON_WHOLE_BYTES = 512 << 20


class ReaderError(RuntimeError):
    pass


def split_ext(name: str):
    """('base', '.jsonl', '.gz') - the real extension and the compression extension (or '')."""
    base, ext = os.path.splitext(name)
    comp = ""
    if ext.lower() in COMPRESSION:
        comp, (base, ext) = ext.lower(), os.path.splitext(base)
    if comp == "" and ext.lower() == ".tgz":
        return base, ".tar", ".gz"
    return base, ext.lower(), comp


def supported_extensions() -> set:
    return JSON_EXT | TABLE_EXT | PARQUET_EXT | ARCHIVE_EXT | TEXT_EXT


def file_kind(name: str) -> Optional[str]:
    _, ext, _ = split_ext(name)
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in JSON_EXT:
        return "json"
    if ext in TABLE_EXT:
        return "table"
    if ext in PARQUET_EXT:
        return "parquet"
    if ext in ARCHIVE_EXT:
        return "archive"
    if ext in TEXT_EXT:
        return "text"
    return None


def _open_binary(path: str, comp: str):
    return COMPRESSION[comp](path, "rb") if comp else open(path, "rb")


def _text_stream(binary) -> io.TextIOBase:
    return io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace", newline="")


# ---- JSON ---------------------------------------------------------------------------------------------------------
def iter_json_array(f: io.TextIOBase, chunk: int = 1 << 20) -> Iterator[object]:
    """Incrementally yield the elements of a top-level JSON array without holding the whole file."""
    dec = json.JSONDecoder()
    buf = f.read(chunk)
    i = 0
    n = len(buf)

    def skip_ws():
        nonlocal i, buf, n
        while True:
            while i < n and buf[i] in " \t\r\n,":
                i += 1
            if i < n:
                return True
            more = f.read(chunk)
            if not more:
                return False
            buf, i = buf[i:] + more, 0
            n = len(buf)

    if not skip_ws() or buf[i] != "[":
        raise ReaderError("expected a JSON array")
    i += 1
    while True:
        if not skip_ws():
            return
        if buf[i] == "]":
            return
        while True:
            try:
                obj, end = dec.raw_decode(buf, i)
                i = end
                yield obj
                break
            except json.JSONDecodeError:
                more = f.read(chunk)
                if not more:
                    raise ReaderError("truncated JSON array")
                buf, i = buf[i:] + more, 0
                n = len(buf)
        if i > (4 << 20):                       # keep the buffer small
            buf, i = buf[i:], 0
            n = len(buf)


def _rows_from_json_obj(obj) -> Iterator[dict]:
    if isinstance(obj, list):
        for r in obj:
            yield r if isinstance(r, dict) else {"text": r} if isinstance(r, str) else {"value": r}
    elif isinstance(obj, dict):
        lists = {k: v for k, v in obj.items() if isinstance(v, list) and v}
        if lists and all(isinstance(v, list) for v in obj.values()) and len({len(v) for v in obj.values()}) == 1:
            keys = list(obj)                                           # columnar {"a": [...], "b": [...]}
            for vals in zip(*[obj[k] for k in keys]):
                yield dict(zip(keys, vals))
        elif lists:
            best = max(lists, key=lambda k: len(lists[k]))              # {"data": [...], "version": 1}
            yield from _rows_from_json_obj(lists[best])
        else:
            yield obj


class _Prefixed:
    """A reader that first returns `prefix`, then continues with `f` (used after peeking at a file's first characters)."""

    def __init__(self, prefix: str, f):
        self.prefix, self.f = prefix, f

    def read(self, n: int = -1) -> str:
        if self.prefix:
            out, self.prefix = self.prefix, ""
            return out
        return self.f.read(n if n and n > 0 else 1 << 20)


def _as_row(obj) -> dict:
    return obj if isinstance(obj, dict) else {"text": obj} if isinstance(obj, str) else {"value": obj}


def _iter_json(binary, ext: str) -> Iterator[dict]:
    f = _text_stream(binary)
    first = f.read(1 << 20)
    stripped = first.lstrip()
    if not stripped:
        return
    if stripped[0] == "[":
        for obj in iter_json_array(_Prefixed(first, f)):
            yield _as_row(obj)
        return
    # JSON lines (also what most '.json' files from dataset hubs really are) or one whole object
    lines_mode = ext in (".jsonl", ".ndjson")
    if not lines_mode:
        head_line = stripped.split("\n", 1)[0]
        try:
            json.loads(head_line)
            lines_mode = "\n" in stripped.strip()       # parsable single line followed by more lines -> jsonl
        except ValueError:
            lines_mode = False
    if lines_mode:
        tail = ""
        chunk = first
        while chunk:
            chunk = tail + chunk
            *lines, tail = chunk.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue                           # a torn / junk line is skipped, never fatal
                if isinstance(obj, (dict, str)):
                    yield _as_row(obj)
            chunk = f.read(1 << 20)
        if tail.strip():
            try:
                obj = json.loads(tail)
                if isinstance(obj, (dict, str)):
                    yield _as_row(obj)
            except ValueError:
                pass
        return
    body = first + f.read()
    if len(body) > MAX_JSON_WHOLE_BYTES:
        raise ReaderError("a single JSON object larger than 512 MB cannot be streamed; convert it to .jsonl first")
    yield from _rows_from_json_obj(json.loads(body))


# ---- tables ---------------------------------------------------------------------------------------------------------
def _looks_like_header(fields: List[str]) -> bool:
    return all(len(h) <= 40 and len(h.split()) <= 3 for h in fields) and not any(h.strip().isdigit() for h in fields)


def _iter_table(binary, ext: str) -> Iterator[dict]:
    try:
        csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    except OverflowError:
        csv.field_size_limit(2 ** 30)
    f = _text_stream(binary)
    first_line = f.readline()
    if not first_line.strip():
        return
    delim = "\t" if ext == ".tsv" else ","
    if ext == ".csv":
        try:
            delim = csv.Sniffer().sniff(first_line, delimiters=",;|\t").delimiter
        except csv.Error:
            delim = ","
    elif first_line.count("|") > first_line.count("\t"):
        delim = "|"
    lines = itertools.chain([first_line], f)
    reader = csv.reader(lines, delimiter=delim)
    header = next(reader, None)
    if header is None:
        return
    if _looks_like_header(header):
        names = [h.strip() or f"col{i}" for i, h in enumerate(header)]
    else:
        names = [f"col{i}" for i in range(len(header))]
        yield dict(zip(names, header))
    for row in reader:
        if not row or all(not c.strip() for c in row):
            continue
        yield dict(zip(names, row))


def _iter_parquet(path: str, binary) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq                                  # optional
    except ImportError as e:
        raise ReaderError("reading .parquet needs `pip install pyarrow`. Alternatives: `zerowhite chat fetch <name>` downloads "
                          "datasets as JSON lines through the Hugging Face rows API, or convert the file to .jsonl yourself.") from e
    pf = pq.ParquetFile(binary if not isinstance(path, str) or not os.path.exists(path) else path)
    for batch in pf.iter_batches(batch_size=1024):
        cols = batch.to_pydict()
        keys = list(cols)
        for vals in zip(*[cols[k] for k in keys]):
            yield dict(zip(keys, vals))


# ---- plain text / code ------------------------------------------------------------------------------------------------
def iter_text_chunks(text: str, chunk_chars: int, code: bool) -> Iterator[str]:
    """Cut a document into pieces of about chunk_chars at paragraph (prose) or line (code) boundaries."""
    if len(text) <= chunk_chars:
        if text.strip():
            yield text
        return
    sep = "\n" if code else "\n\n"
    cur: List[str] = []
    size = 0
    for part in text.split(sep):
        plen = len(part) + len(sep)
        if plen > chunk_chars:                                          # one gigantic paragraph/line
            if cur:
                yield sep.join(cur)
                cur, size = [], 0
            for s in range(0, len(part), chunk_chars):
                yield part[s:s + chunk_chars]
            continue
        if size + plen > chunk_chars and cur:
            yield sep.join(cur)
            cur, size = [], 0
        cur.append(part)
        size += plen
    if cur and "".join(cur).strip():
        yield sep.join(cur)


def _iter_text(binary, name: str, ext: str, chunk_chars: int) -> Iterator[dict]:
    text = _text_stream(binary).read()
    code = ext in CODE_EXT
    for i, piece in enumerate(iter_text_chunks(text, chunk_chars, code)):
        yield {"text": piece, "_ext": ext, "_path": name, "_part": i}


# ---- dispatch ----------------------------------------------------------------------------------------------------------
def _iter_binary(binary, name: str, ext: str, chunk_chars: int, path_for_parquet: str = "") -> Iterator[dict]:
    kind = file_kind(name)
    if kind == "json":
        yield from _iter_json(binary, ext)
    elif kind == "table":
        yield from _iter_table(binary, ext)
    elif kind == "parquet":
        yield from _iter_parquet(path_for_parquet, binary)
    elif kind == "text":
        yield from _iter_text(binary, name, ext, chunk_chars)


def _iter_archive(path: str, chunk_chars: int, limit_members: Optional[int] = None) -> Iterator[dict]:
    _, ext, comp = split_ext(os.path.basename(path))
    if ext == ".zip":
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir() or info.file_size > MAX_MEMBER_BYTES:
                    continue
                name = info.filename.replace("\\", "/")
                if name.startswith("/") or ".." in name.split("/") or "__MACOSX" in name or os.path.basename(name).startswith("."):
                    continue
                _, e2, c2 = split_ext(name)
                if file_kind(name) in (None, "audio", "archive"):
                    continue
                with z.open(info) as raw:
                    binary = COMPRESSION[c2](raw, "rb") if c2 else raw
                    for row in _iter_binary(binary, name, e2, chunk_chars):
                        yield row
    else:
        with tarfile.open(path, "r:*") as t:
            for m in t:
                if not m.isfile() or m.size > MAX_MEMBER_BYTES:
                    continue
                name = m.name.replace("\\", "/")
                if name.startswith("/") or ".." in name.split("/") or os.path.basename(name).startswith("."):
                    continue
                _, e2, c2 = split_ext(name)
                if file_kind(name) in (None, "audio", "archive"):
                    continue
                raw = t.extractfile(m)
                if raw is None:
                    continue
                binary = COMPRESSION[c2](raw, "rb") if c2 else raw
                for row in _iter_binary(binary, name, e2, chunk_chars):
                    yield row


def iter_rows(path: str, cfg=None, limit: Optional[int] = None) -> Iterator[Dict]:
    """Stream the rows of one file. `limit` stops early (used for sniffing)."""
    name = os.path.basename(path)
    _, ext, comp = split_ext(name)
    kind = file_kind(name)
    if kind is None:
        raise ReaderError(f"unsupported file type '{ext or name}'")
    chunk_chars = int(cfg.data.text_chunk_chars) if cfg is not None else 4000
    n = 0
    try:
        if kind == "archive":
            gen = _iter_archive(path, chunk_chars)
        elif kind == "parquet":
            gen = _iter_parquet(path, None)
        else:
            binary = _open_binary(path, comp)
            gen = _iter_binary(binary, name, ext, chunk_chars)
        for row in gen:
            if kind == "text":
                row["_path"] = name
            yield row
            n += 1
            if limit is not None and n >= limit:
                break
    except (OSError, EOFError, zipfile.BadZipFile, tarfile.TarError, gzip.BadGzipFile, lzma.LZMAError) as e:
        raise ReaderError(f"cannot read {name}: {e}") from e


def sniff(path: str, n: int = 50, cfg=None) -> List[Dict]:
    return list(iter_rows(path, cfg, limit=n))
