"""sniff.py - decide what a file IS from its bytes, not from its name; decode text without ever raising.

Three jobs:

  identify(data)       container / binary formats by magic number (gzip, zip, tar, numpy, sqlite, png, wav ...)
  decode(data)         bytes -> text. BOMs, UTF-16 without a BOM, legacy single-byte encodings, trailing NUL padding (what a crashed
                       write leaves behind) and stray bad bytes are all handled; what was repaired is reported, never hidden
  classify_text(text)  json | jsonl | table | python | text      (a *content* decision: `data.txt` may hold JSON)
  json_records(text)   list of records out of whole, line-delimited, columnar, nested or TRUNCATED JSON

Nothing in here raises on hostile input; every function has a defined answer for garbage.
"""
from __future__ import annotations

import ast
import codecs
import hashlib
import json
import math
import re
from dataclasses import dataclass, field

# (prefix, kind)
_MAGIC = [
    (b"\x1f\x8b", "gzip"), (b"BZh", "bz2"), (b"\xfd7zXZ\x00", "xz"),
    (b"PK\x03\x04", "zip"), (b"PK\x05\x06", "zip"), (b"PK\x07\x08", "zip"),
    (b"\x93NUMPY", "npy"), (b"SQLite format 3\x00", "sqlite"), (b"PAR1", "parquet"),
    (b"\x89PNG\r\n\x1a\n", "image"), (b"\xff\xd8\xff", "image"), (b"GIF87a", "image"), (b"GIF89a", "image"),
    (b"%PDF", "pdf"), (b"\x7fELF", "executable"), (b"OggS", "audio"), (b"fLaC", "audio"),
    (b"ID3", "audio"), (b"\xca\xfe\xba\xbe", "executable"), (b"\xcf\xfa\xed\xfe", "executable"),
]


@dataclass
class Decoded:
    text: str
    encoding: str
    damage: float = 0.0                       # fraction of characters that had to be replaced
    repairs: list = field(default_factory=list)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def entropy_bits(data: bytes, limit: int = 1 << 20) -> float:
    """Shannon entropy in bits per byte of the first `limit` bytes (8.0 = indistinguishable from random / compressed)."""
    data = data[:limit]
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = float(len(data))
    return -sum(c / n * math.log2(c / n) for c in counts if c)


def _magic_ok(kind: str, prefix: bytes, data: bytes) -> bool:
    """Short magic numbers collide with ordinary text ("ID3 tags are...", "BZh"); confirm them with one more fact."""
    if kind == "bz2":
        return data[3:4] in (b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8", b"9") and data[4:10] == b"1AY&SY"
    if prefix == b"ID3":
        return len(data) > 3 and data[3] < 0x10
    if kind == "parquet":
        return b"\x00" in data[:512]
    if kind in ("pdf", "image") and prefix.startswith(b"GIF"):
        return True
    return True


def identify(data: bytes) -> str:
    """'empty' | 'gzip' | 'bz2' | 'xz' | 'zip' | 'tar' | 'npy' | 'sqlite' | 'parquet' | 'image' | 'pdf' | 'audio' | 'executable'
    | 'text' | 'binary'. Text is decided by `decode`, so this only says 'text' when nothing binary matched and there are no NULs."""
    if not data:
        return "empty"
    head = data[:512]
    for prefix, kind in _MAGIC:
        if kind and head.startswith(prefix) and _magic_ok(kind, prefix, data):
            return kind
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio"
    if len(data) > 262 and data[257:262] == b"ustar":
        return "tar"
    if head[:4] in (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"):
        return "text"                                    # UTF-32 BOM
    if head.startswith((b"\xff\xfe", b"\xfe\xff", b"\xef\xbb\xbf")):
        return "text"
    return "text" if b"\x00" not in data[:8192] else "binary"


def _printable_fraction(s: str) -> float:
    if not s:
        return 1.0
    bad = sum(1 for ch in s if (ord(ch) < 32 and ch not in "\n\r\t\f") or ch == "\ufffd" or 0x7F <= ord(ch) < 0xA0)
    return 1.0 - bad / len(s)


def decode(data: bytes) -> Decoded | None:
    """Best-effort bytes -> text. Returns None when the bytes are not text at all (too many control bytes after every attempt)."""
    repairs: list = []
    if not data:
        return Decoded("", "utf-8")
    # BOMs first: they are certain
    for bom, enc in ((codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"), (codecs.BOM_UTF8, "utf-8-sig"),
                     (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16")):
        if data.startswith(bom):
            txt = data.decode(enc, errors="replace")
            bad = txt.count("\ufffd")
            if bad:
                repairs.append(f"{bad} undecodable character(s) replaced")
            return Decoded(txt.replace("\ufffd", "?") if bad else txt, enc, bad / max(1, len(txt)), repairs)
    # NUL padding at the end: what an interrupted write leaves on some file systems
    stripped = data.rstrip(b"\x00")
    if len(stripped) < len(data) and len(stripped) >= 1:
        repairs.append(f"{len(data) - len(stripped)} trailing NUL byte(s) removed (an interrupted write?)")
        data = stripped
    sample = data[:65536]
    # UTF-16 without a BOM: NULs on every other byte
    if len(sample) >= 8 and b"\x00" in sample:
        even, odd = sample[0::2].count(0), sample[1::2].count(0)
        half = len(sample) / 2
        if odd > 0.4 * half and even < 0.05 * half:
            return Decoded(data.decode("utf-16-le", errors="replace"), "utf-16-le", 0.0, repairs + ["UTF-16 without a BOM"])
        if even > 0.4 * half and odd < 0.05 * half:
            return Decoded(data.decode("utf-16-be", errors="replace"), "utf-16-be", 0.0, repairs + ["UTF-16 without a BOM"])
        return None
    try:
        txt = data.decode("utf-8")
        return Decoded(txt, "utf-8", 0.0, repairs)
    except UnicodeDecodeError:
        pass
    txt = data.decode("utf-8", errors="replace")
    bad = txt.count("\ufffd")
    high = sum(1 for b in data[:200000] if b >= 0x80)
    if bad / max(1, high) > 0.5 or bad / max(1, len(txt)) > 0.02:
        # mostly invalid as UTF-8: a legacy single-byte file (Windows-1252 if it can be, else Latin-1)
        for enc in ("cp1252", "latin-1"):
            try:
                t2 = data.decode(enc)
            except UnicodeDecodeError:
                continue
            if _printable_fraction(t2) >= 0.97:
                return Decoded(t2, enc, 0.0, repairs + [f"not UTF-8; decoded as {enc}"])
        return None
    if _printable_fraction(txt) < 0.97:
        return None
    repairs.append(f"{bad} undecodable byte(s) replaced")
    return Decoded(txt.replace("\ufffd", "?"), "utf-8", bad / max(1, len(txt)), repairs)


def printable_strings(data: bytes, min_len: int = 8, cap: int = 200000) -> list:
    """Runs of printable ASCII (like `strings`): what can be read from a binary of unknown format."""
    out, total = [], 0
    for m in re.finditer(rb"[\x20-\x7e]{%d,}" % min_len, data[:8 * 1024 * 1024]):
        s = m.group().decode("ascii")
        out.append(s)
        total += len(s)
        if total >= cap:
            break
    return out


# ------------------------------------------------------------------------------------------------ text classification
def _python_info(text: str) -> dict | None:
    """If `text` is Python source: whether it defines the environment contract (class Env with the four members)."""
    if not re.search(r"^\s*(def |class |import |from \w+ import )", text, re.M):
        return None
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    info = dict(has_env=False, n_defs=sum(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in tree.body))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Env":
            methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            attrs = set(methods)                                   # obs_dim may be a property
            def names(t):
                if isinstance(t, ast.Name):
                    yield t.id
                elif isinstance(t, ast.Attribute):
                    yield t.attr
                elif isinstance(t, (ast.Tuple, ast.List)):         # obs_dim, n_actions = 3, 3
                    for el in t.elts:
                        yield from names(el)
            for n in ast.walk(node):
                if isinstance(n, ast.Assign):
                    for t in n.targets:
                        attrs.update(names(t))
                elif isinstance(n, ast.AnnAssign):
                    attrs.update(names(n.target))
            info["has_env"] = {"reset", "step"} <= methods and {"obs_dim", "n_actions"} <= attrs
    return info


def classify_text(text: str) -> tuple:
    """-> (subkind, info) with subkind in 'empty' | 'json' | 'jsonl' | 'python' | 'table' | 'text'.
    'table' is decided by the caller (tables.parse), because it needs the delimiter search; here a table candidate is
    simply returned as 'text' unless JSON or Python matched first."""
    s = text.strip()
    if not s:
        return "empty", {}
    recs, repairs = json_records(s)
    if recs:
        is_lines = any(r.startswith("line:") for r in repairs)
        shown = [r[len("line:"):].strip() if r.startswith("line:") else r for r in repairs]      # "line:" is an internal marker, not user text
        return ("jsonl" if is_lines else "json"), dict(records=recs, repairs=shown)
    py = _python_info(text)
    if py is not None:
        return "python", py
    return "text", {}


def _whole_json(s: str):
    try:
        return json.loads(s)
    except (ValueError, RecursionError):
        return None


def _flatten(obj, prefix="", depth=0, out=None):
    """Nested dict -> flat {'a.b': value}; lists of scalars become their length and their first element's value is dropped."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict) and depth < 3:
                _flatten(v, key, depth + 1, out)
            elif isinstance(v, (list, dict)):
                out[key + ".len"] = len(v)
            else:
                out[key] = v
    else:
        out[prefix or "value"] = obj
    return out


def _as_records(items) -> list:
    recs = []
    for it in items:
        if isinstance(it, dict):
            recs.append(_flatten(it))
        elif isinstance(it, (list, tuple)) and it and all(not isinstance(x, (dict, list)) for x in it):
            recs.append({f"c{i}": x for i, x in enumerate(it)})
    return recs


def _records_in(obj) -> tuple:
    """Find a table inside a parsed JSON value. -> (records, where)"""
    if isinstance(obj, list):
        recs = _as_records(obj)
        return (recs, "top-level array") if len(recs) >= 2 else ([], "")
    if isinstance(obj, dict):
        # columnar: {"a": [..], "b": [..]} with equal lengths
        cols = {k: v for k, v in obj.items() if isinstance(v, list) and v and all(not isinstance(x, (dict, list)) for x in v)}
        if len(cols) >= 2 and len({len(v) for v in cols.values()}) == 1 and len(next(iter(cols.values()))) >= 2:
            n = len(next(iter(cols.values())))
            return [{k: v[i] for k, v in cols.items()} for i in range(n)], "columnar object"
        for k, v in obj.items():                          # {"data": [ {...}, ... ]}
            if isinstance(v, list):
                recs = _as_records(v)
                if len(recs) >= 2:
                    return recs, f"records under key '{k}'"
        for k, v in obj.items():
            if isinstance(v, dict):
                recs, where = _records_in(v)
                if recs:
                    return recs, f"{where} under key '{k}'"
        # a dict of dicts: {"id1": {...}, "id2": {...}}
        vals = [v for v in obj.values() if isinstance(v, dict)]
        if len(vals) >= 2 and len(vals) == len(obj):
            return _as_records(vals), "object of objects"
    return [], ""


def json_records(text: str, cap: int = 2_000_000) -> tuple:
    """-> (records, repairs). Records are flat dicts. `repairs` says what had to be done to read them: skipping unparsable lines
    (JSONL), keeping the complete records of a truncated array. ([], []) when the text is not JSON-shaped at all."""
    s = text.strip()
    if not s or s[0] not in "[{":
        return [], []
    whole = _whole_json(s)
    if whole is not None:
        recs, where = _records_in(whole)
        return recs[:cap], ([f"JSON {where}"] if recs else [])
    repairs: list = []
    # JSON Lines: one value per line, some lines possibly damaged
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) >= 2:
        good, bad = [], 0
        for ln in lines[:cap]:
            try:
                v = json.loads(ln)
            except (ValueError, RecursionError):
                bad += 1
                continue
            good.append(v)
        recs = _as_records(good)
        if len(recs) >= 2 and len(recs) >= 0.6 * len(lines):
            if bad:
                repairs.append(f"line: {bad} unparsable line(s) skipped (JSON Lines)")
            else:
                repairs.append("line: JSON Lines")
            return recs, repairs
    # truncated / half-written array: take every complete element from the first '['
    start = s.find("[")
    if 0 <= start < 400:
        dec, i, items = json.JSONDecoder(), start + 1, []
        n = len(s)
        while i < n and len(items) < cap:
            while i < n and s[i] in " \t\r\n,":
                i += 1
            if i >= n or s[i] == "]":
                break
            try:
                obj, i = dec.raw_decode(s, i)
            except (ValueError, RecursionError):
                break
            items.append(obj)
        recs = _as_records(items)
        if len(recs) >= 2:
            repairs.append(f"truncated JSON: kept the {len(recs)} complete record(s) before the damage")
            return recs, repairs
    return [], []


# ------------------------------------------------------------------------------------------------ text cleanup
_TAG = re.compile(r"<[^>]{1,500}>")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f\u200b\u200c\u200d\u2060\ufeff]")


def clean_text(text: str) -> tuple:
    """Normalise a text for learning: newlines, control characters, HTML/XML markup. -> (text, repairs)"""
    repairs = []
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    n_tags = len(_TAG.findall(t[:200000]))
    if n_tags >= 20 and n_tags * 12 > min(len(t), 200000) * 0.02:
        import html
        t = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", t)
        t = html.unescape(_TAG.sub(" ", t))
        repairs.append("markup tags removed")
    n_before = len(t)
    t = _CONTROL.sub("", t)
    if len(t) != n_before:
        repairs.append(f"{n_before - len(t)} control / zero-width character(s) removed")
    t2 = re.sub(r"\n{4,}", "\n\n\n", t)
    t2 = re.sub(r"[ \t]{3,}", "  ", t2)
    return t2.strip() + "\n", repairs
