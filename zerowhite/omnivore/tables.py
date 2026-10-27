"""tables.py - turn anything table-shaped into a supervised task, repairing what is broken and saying what was done.

  parse_delimited(text)      CSV / TSV / semicolon / pipe / whitespace tables: the delimiter is *found* (the one that makes the rows the same
                             width), a preamble before the header is skipped, ragged rows are padded / trimmed, an unbalanced quote does not
                             swallow the file, a half-written last row is dropped
  table_from_records / _array / read_sqlite     the same Table from JSON records, numpy arrays and SQLite databases
  build(table, ...)          columns -> numbers: numeric (thousands separators, decimal commas, %, currency), booleans, dates, categories
                             (one-hot), missing values (median + indicator), identifiers and constants dropped; a target column is chosen
                             (named, or by name hint, or the last one); classification vs regression is decided from the target

A table that cannot be used raises `Unusable(reason)`: a sentence that says why and what to do. Nothing else escapes.
"""
from __future__ import annotations

import csv
import io
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

NA_TOKENS = {"", "na", "n/a", "nan", "null", "none", "nil", "?", "-", "--", "missing", "#n/a", "#na", "undefined", "n.a."}
TARGET_NAMES = ("target", "label", "class", "y", "outcome", "response")           # a column with one of these names is the target
_BOOL = {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "t": 1.0, "f": 0.0, "y": 1.0, "n": 0.0}
_DATE_FORMATS = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M", "%Y/%m/%d",
                 "%m/%d/%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%b-%Y", "%b %d, %Y")
_CURRENCY = "$€£¥₹"
_NUM = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")
_THOUSANDS_DOT = re.compile(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?")
_THOUSANDS_COMMA = re.compile(r"[+-]?\d{1,3}(\.\d{3})+(,\d+)?")
_THOUSANDS_SPACE = re.compile(r"[+-]?\d{1,3}([ \u00a0]\d{3})+([.,]\d+)?")
_DEC_COMMA = re.compile(r"[+-]?\d+,\d+")
_DEC_DOT = re.compile(r"[+-]?\d+\.\d+")

try:
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
except (OverflowError, ValueError):
    pass


class Unusable(Exception):
    """The input is understood but cannot become a task. `textlike` says it is mostly prose (so the caller may keep it as text)."""

    def __init__(self, reason: str, textlike: bool = False):
        super().__init__(reason)
        self.reason, self.textlike = reason, textlike


@dataclass
class Table:
    columns: list
    rows: list                                   # each row: a list of str | None, exactly len(columns) long
    repairs: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    delimiter: str = ""


@dataclass
class Built:
    X: np.ndarray
    y: np.ndarray
    kind: str                                    # classification | regression
    feature_names: list
    target_name: str
    class_names: list
    repairs: list
    notes: list
    spec: dict


# ------------------------------------------------------------------------------------------------ cell parsing
def parse_number(s, decimal: str = "."):
    """A float from a cell, or None. Handles 1,234.5  1.234,5  1 234,5  $12  (7.5)  45%  1e-3; nan / inf are missing, not numbers."""
    if s is None:
        return None
    if decimal == ".":
        try:
            v = float(s)
            return v if math.isfinite(v) else None
        except ValueError:
            pass
    t = s.strip()
    if not t:
        return None
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg, t = True, t[1:-1].strip()
    for ch in _CURRENCY:
        t = t.replace(ch, "")
    if t.endswith("%"):
        t = t[:-1]
    t = t.strip()
    if _THOUSANDS_SPACE.fullmatch(t):
        t = re.sub(r"[ \u00a0]", "", t)
    if decimal == ".":
        if _THOUSANDS_DOT.fullmatch(t):
            t = t.replace(",", "")
    else:
        if _THOUSANDS_COMMA.fullmatch(t):
            t = t.replace(".", "")
        t = t.replace(",", ".")
    if _NUM.fullmatch(t):
        try:
            v = float(t)
        except ValueError:
            return None
        if math.isfinite(v) and abs(v) < 1e15:
            return -v if neg else v
    return None


def _clean_cell(c):
    if c is None:
        return None
    c = c.strip().strip("\u200b\ufeff")
    return None if c.lower() in NA_TOKENS else c


def _date_format(cells: list):
    """The strptime format that parses (nearly) all of `cells`, else None."""
    sample = [c for c in cells[:200] if c]
    if len(sample) < 3 or not any(ch.isdigit() for ch in sample[0]):
        return None
    best, best_n = None, 0
    for fmt in _DATE_FORMATS:
        n = 0
        for c in sample:
            try:
                datetime.strptime(c, fmt)
                n += 1
            except ValueError:
                pass
        if n > best_n:
            best, best_n = fmt, n
    return best if best_n >= 0.9 * len(sample) else None


def _to_days(c, fmt):
    try:
        return (datetime.strptime(c, fmt) - datetime(1970, 1, 1)).total_seconds() / 86400.0
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------------------------------------ delimited text
def _widths(lines: list, delim: str) -> list:
    if delim == " ":
        return [len(ln.split()) for ln in lines]
    try:
        return [len(r) for r in csv.reader(lines, delimiter=delim)]
    except csv.Error:
        return []


def find_delimiter(text: str):
    """The delimiter that makes the rows the same width, with its consistency (0..1); (None, 0) when nothing does."""
    lines = []
    for ln in text.split("\n"):
        if ln.strip():
            lines.append(ln)
            if len(lines) >= 300:
                break
    if len(lines) < 3:
        return None, 0.0, 0
    best = (None, 0.0, 0, 0.0)
    for d in (",", ";", "\t", "|", " "):
        w = _widths(lines, d)
        if len(w) < 3:
            continue
        width, n = Counter(w).most_common(1)[0]
        if width < 2:
            continue
        cons = n / len(w)
        if cons < 0.6:
            continue
        if d == " " and best[0] is not None:
            continue                                      # whitespace is the last resort
        # a wrong delimiter that is merely consistent (the decimal commas of a ';' file) splits into FEWER columns than the right one,
        # so width counts too: score = consistency^2 * log2(width)
        score = cons * cons * math.log2(width)
        if best[0] is None or score > best[3]:
            best = (d, cons, width, score)
    return best[:3]


def _read_rows(text: str, delim: str, quoting, cap: int) -> tuple:
    rows, bad = [], 0
    if delim == " ":
        for ln in text.split("\n"):
            if ln.strip():
                rows.append(ln.split())
                if len(rows) >= cap:
                    break
        return rows, 0
    reader = csv.reader(io.StringIO(text), delimiter=delim, quoting=quoting, skipinitialspace=True)
    guard = 0
    while len(rows) < cap and guard < cap * 4 + 1000:
        guard += 1
        try:
            r = next(reader)
        except StopIteration:
            break
        except csv.Error:
            bad += 1
            continue
        if r and any(c.strip() for c in r):
            rows.append(r)
    return rows, bad


def _looks_like_header(first: list, body: list) -> bool:
    sig = 0
    for j in range(len(first)):
        col = [r[j] for r in body[:200] if j < len(r) and _clean_cell(r[j]) is not None]
        if len(col) < 3:
            continue
        numfrac = sum(parse_number(c) is not None for c in col) / len(col)
        if numfrac >= 0.8:
            sig += 1 if (first[j].strip() and parse_number(first[j]) is None) else -1
    if sig:
        return sig > 0
    cells = [c.strip() for c in first]
    if all(cells) and len(set(cells)) == len(cells) and all(parse_number(c) is None for c in cells):
        below = [{(r[j].strip() if j < len(r) and r[j] is not None else None) for r in body[:200]} for j in range(len(first))]     # 6.4: padded cells are None
        return not any(cells[j] in below[j] for j in range(len(first)))
    return False


def _unique_names(names: list) -> list:
    seen, out = Counter(), []
    for j, n in enumerate(names):
        n = (n or "").strip() or f"col{j}"
        seen[n] += 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def parse_delimited(text: str, max_rows: int = 50_000, hint: dict = None):
    """Text -> Table, or None when it is not a table. Every repair is listed in `.repairs`.

    `hint` (6.4, from omnivore/learned.py): {'delim': ..., 'arbitrate': bool}. The delimiter is taken from it when the rules found none, or (arbitrate) when it
    differs from the rules'; the caller decides whether the result is better (see table_quality). Without a hint nothing changes."""
    delim, cons, width = find_delimiter(text)
    repairs, notes = [], []
    if hint and hint.get("delim") and (delim is None or (hint.get("arbitrate") and hint["delim"] != delim)):
        repairs.append(f"delimiter {hint['delim']!r} taken from the learned delimiter ranker ("
                       + ("the rules found none" if delim is None else f"the rules said {delim!r}") + ")")
        delim = hint["delim"]
    if delim is None:
        return None
    cap = max_rows * 2 + 2
    rows, bad = _read_rows(text, delim, csv.QUOTE_MINIMAL, cap)
    n_lines = sum(1 for ln in text.split("\n", cap * 2)[:cap * 2] if ln.strip())
    if delim != " " and len(rows) < 0.5 * min(n_lines, cap):
        alt, bad2 = _read_rows(text, delim, csv.QUOTE_NONE, cap)
        if len(alt) > len(rows):
            rows, bad = alt, bad2
            repairs.append("quote characters ignored (an unbalanced quote was swallowing rows)")
    if bad:
        repairs.append(f"{bad} unreadable line(s) skipped")
    if len(rows) < 3:
        return None
    width = Counter(len(r) for r in rows).most_common(1)[0][0]
    if width < 2:
        return None
    first_ok = next((i for i, r in enumerate(rows) if len(r) == width), None)
    if first_ok is None:
        return None
    if first_ok:
        repairs.append(f"{first_ok} preamble line(s) before the table skipped")
        rows = rows[first_ok:]
    if not text.endswith(("\n", "\r")) and len(rows[-1]) < width and len(rows) > 3:
        rows = rows[:-1]
        repairs.append("the last row was cut off (file ends mid-line) and was dropped")
    ragged = sum(len(r) != width for r in rows)
    if ragged > 0.4 * len(rows):
        return None
    if ragged:
        short = sum(len(r) < width for r in rows)
        repairs.append(f"{ragged} ragged row(s) fixed ({short} padded with missing values, {ragged - short} trimmed)")
        rows = [(r + [None] * (width - len(r)))[:width] for r in rows]
    sample = [c for r in rows[:200] for c in r if c and c.strip()]
    if sample:
        wordy = sum(len(c.split()) >= 3 for c in sample) / len(sample)
        numeric = sum(parse_number(c) is not None for c in sample) / len(sample)
        if numeric < 0.1 and (wordy >= 0.5 or sum(len(c) for c in sample) / len(sample) > 40):
            return None                                      # sentences in the cells and no numbers: prose that happens to contain commas
    header = _looks_like_header(rows[0], rows[1:])
    if header:
        cols, body = _unique_names(rows[0]), rows[1:]
    else:
        cols, body = [f"c{j}" for j in range(width)], rows
        notes.append("no header row found: columns are named c0, c1, ...")
    if len(rows) >= cap:
        notes.append(f"only the first {len(body):,} rows were read")
    return Table(cols, body, repairs, notes, delim)


def table_quality(tab) -> int:
    """How table-like a parse is: the number of columns in which at least 80% of the non-empty cells (and at least 3 of them) are numbers. Prose split on spaces
    scores 0; a real table of measurements scores its number of numeric columns. Used to decide whether a delimiter suggested by the learned ranker is believable."""
    if tab is None:
        return 0
    n = 0
    for j in range(len(tab.columns)):
        cells = [r[j] for r in tab.rows[:300] if r[j] is not None and str(r[j]).strip()]
        if len(cells) >= 3 and sum(parse_number(str(c), ".") is not None or parse_number(str(c), ",") is not None for c in cells) >= 0.8 * len(cells):
            n += 1
    return n


# ------------------------------------------------------------------------------------------------ other sources
def _cell(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return repr(float(v)) if isinstance(v, (float, np.floating)) else str(int(v))
    if isinstance(v, (bytes, bytearray)):
        return None
    if isinstance(v, str):
        return v
    return str(v)


def table_from_records(records: list, max_rows: int = 50_000):
    if len(records) < 3:
        return None
    cols: list = []
    seen = set()
    for r in records[:1000]:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    notes = []
    if len(cols) > 500:
        cols = cols[:500]
        notes.append("only the first 500 fields were kept")
    rows = [[_cell(r.get(c)) for c in cols] for r in records[:max_rows * 2 + 2]]
    return Table([str(c) for c in cols], rows, [], notes, "")


def table_from_array(arr):
    """A 2-D numpy array (or a structured array) -> Table; anything else -> None."""
    try:
        if arr.dtype.names:
            cols = list(arr.dtype.names)
            rows = [[_cell(v.item() if hasattr(v, "item") else v) for v in rec] for rec in arr.tolist()]
            return Table(cols, rows, [], ["structured numpy array"], "")
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2 or arr.dtype.kind not in "biuf":
            return None
        return Table([f"c{j}" for j in range(arr.shape[1])], [[_cell(v) for v in row] for row in arr.tolist()], [],
                     ["numpy array: columns are named c0, c1, ..."], "")
    except Exception:                                    # noqa: BLE001
        return None


def read_sqlite(path: str, max_rows: int = 50_000, max_tables: int = 20) -> list:
    """[(table_name, Table)] from a SQLite file, opened read-only. Raises Unusable when it cannot be opened."""
    try:
        import sqlite3
        from pathlib import Path
    except ImportError as e:
        raise Unusable(f"this Python has no sqlite3 module ({e})")
    out = []
    try:
        con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
    except Exception as e:                               # noqa: BLE001
        raise Unusable(f"SQLite file could not be opened read-only: {type(e).__name__}: {e}")
    try:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for name in names[:max_tables]:
            try:
                cur = con.execute('SELECT * FROM "%s" LIMIT %d' % (name.replace('"', '""'), int(max_rows * 2 + 2)))
                cols = [d[0] for d in cur.description]
                rows = [[_cell(v) for v in r] for r in cur.fetchall()]
            except Exception:                            # noqa: BLE001 - a damaged table must not hide the others
                continue
            if len(rows) >= 3 and len(cols) >= 2:
                out.append((name, Table(_unique_names(cols), rows, [], [f"SQLite table '{name}'"], "")))
    except Exception as e:                               # noqa: BLE001 - not a database after all / corrupt
        raise Unusable(f"SQLite file could not be read: {type(e).__name__}: {e}")
    finally:
        con.close()
    return out


# ------------------------------------------------------------------------------------------------ tables -> numbers
def _pick_target(table: Table, target, notes: list) -> int:
    cols = table.columns
    if target is not None:
        if isinstance(target, int) and not isinstance(target, bool) and 0 <= target < len(cols):
            return target
        if str(target) in cols:
            return cols.index(str(target))
        low = [c.lower() for c in cols]
        if str(target).lower() in low:
            return low.index(str(target).lower())
        raise Unusable(f"the requested target column '{target}' does not exist (columns: {', '.join(cols[:12])}"
                       f"{' ...' if len(cols) > 12 else ''})")
    hinted = [i for i, c in enumerate(cols) if c.strip().lower() in TARGET_NAMES]
    if len(hinted) == 1:
        notes.append(f"target = '{cols[hinted[0]]}' (chosen by its name)")
        return hinted[0]
    notes.append(f"target = '{cols[-1]}' (the last column; pass --target to choose another)")
    return len(cols) - 1


def _decimal_mode(table: Table) -> str:
    if table.delimiter == ",":
        return "."
    comma = dot = 0
    for r in table.rows[:500]:
        for c in r:
            if c:
                t = c.strip()
                if _DEC_COMMA.fullmatch(t):
                    comma += 1
                elif _DEC_DOT.fullmatch(t):
                    dot += 1
    return "," if comma > 0 and dot == 0 else "."


def build(table: Table, limits, target=None, seed: int = 0) -> Built:
    """Table -> design matrix + target. Raises Unusable with a reason a person can act on."""
    repairs, notes = list(table.repairs), list(table.notes)
    n_raw = len(table.rows)
    if n_raw < limits.min_rows:
        raise Unusable(f"only {n_raw} usable row(s); at least {limits.min_rows} are needed to split train / validation / test")
    if len(table.columns) < 2:
        raise Unusable("a single column: there is nothing to predict it from")
    dec = _decimal_mode(table)
    if dec == ",":
        repairs.append("decimal commas read as decimal points")
    ti = _pick_target(table, target, notes)
    tname = table.columns[ti]
    rng = np.random.default_rng(seed)

    # ---- target
    tcells = [_clean_cell(r[ti]) for r in table.rows]
    keep = [i for i, c in enumerate(tcells) if c is not None]
    if len(keep) < limits.min_rows:
        raise Unusable(f"target '{tname}' is missing in {n_raw - len(keep)} of {n_raw} rows: only {len(keep)} are left")
    if len(keep) < n_raw:
        repairs.append(f"{n_raw - len(keep)} row(s) without a target value dropped")
    if len(keep) > limits.max_rows:
        keep = sorted(rng.choice(keep, size=limits.max_rows, replace=False).tolist())
        notes.append(f"{limits.max_rows:,} of {len(tcells):,} rows kept (a random sample; the cap comes from this machine's speed)")
    tvals = [tcells[i] for i in keep]
    nums = [parse_number(c, dec) for c in tvals]
    n_num = sum(v is not None for v in nums)
    bools = [_BOOL.get(c.lower()) for c in tvals]
    if n_num >= 0.9 * len(tvals):
        ok = [i for i, v in enumerate(nums) if v is not None]
        if len(ok) < len(tvals):
            repairs.append(f"{len(tvals) - len(ok)} row(s) whose target is not a number dropped")
        keep, yv = [keep[i] for i in ok], np.array([nums[i] for i in ok], dtype=np.float64)
        distinct = np.unique(yv)
        is_int = bool(np.allclose(yv, np.round(yv)))
        if is_int and len(distinct) <= 20:
            kind, (class_vals, y) = "classification", np.unique(yv, return_inverse=True)
            class_names = [str(int(v)) for v in class_vals]
        else:
            kind, y, class_names = "regression", yv, []
    elif sum(b is not None for b in bools) == len(tvals) and len(set(bools)) == 2:
        kind, y, class_names = "classification", np.array(bools, dtype=np.int64), ["false/no", "true/yes"]
    else:
        counts = Counter(tvals)
        if len(counts) > limits.max_classes:
            raise Unusable(f"target '{tname}' has {len(counts)} distinct text values: it looks like an identifier or free text, "
                           f"not a label. Choose the column to predict with --target", textlike=len(counts) > 0.5 * len(tvals))
        kind = "classification"
        names = sorted(counts, key=lambda k: (-counts[k], k))
        index = {k: i for i, k in enumerate(names)}
        y, class_names = np.array([index[c] for c in tvals], dtype=np.int64), list(names)
    if kind == "classification":
        counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=len(class_names))
        rare = [i for i, n in enumerate(counts) if 0 < n < limits.min_class_count]
        if rare:
            drop = np.isin(y, rare)
            repairs.append(f"{int(drop.sum())} row(s) of {len(rare)} class(es) with fewer than {limits.min_class_count} examples set aside")
            keep = [k for k, d in zip(keep, drop) if not d]
            y = y[~drop]
            live = sorted(set(int(v) for v in y))
            remap = {v: i for i, v in enumerate(live)}
            y = np.array([remap[int(v)] for v in y], dtype=np.int64)
            class_names = [class_names[v] for v in live]
        if len(class_names) < 2:
            raise Unusable(f"target '{tname}' has fewer than 2 classes with at least {limits.min_class_count} examples: nothing to tell apart")
    if len(keep) < limits.min_rows:
        raise Unusable(f"only {len(keep)} row(s) are usable after cleaning; at least {limits.min_rows} are needed")

    # ---- features
    names: list = []
    blocks: list = []
    dropped: list = []
    text_cols = 0
    n_feat = 0
    for j, cname in enumerate(table.columns):
        if j == ti:
            continue
        cells = [_clean_cell(table.rows[i][j]) for i in keep]
        present = [c for c in cells if c is not None]
        if not present:
            dropped.append((cname, "empty"))
            continue
        nums_j = [parse_number(c, dec) for c in cells]
        n_ok = sum(v is not None for v in nums_j)
        col = None
        if n_ok >= 0.9 * len(present):                                       # numeric
            arr = np.array([np.nan if v is None else v for v in nums_j], dtype=np.float64)
            vals = arr[~np.isnan(arr)]
            if len(vals) and len(np.unique(vals)) == 1:
                dropped.append((cname, "constant"))
                continue
            if (len(vals) == len(arr) and np.allclose(vals, np.round(vals)) and len(np.unique(vals)) == len(vals)
                    and vals.max() - vals.min() + 1 == len(vals)):
                dropped.append((cname, "row index / identifier (unique consecutive integers)"))
                continue
            n_junk = len(present) - n_ok
            if n_junk:
                repairs.append(f"column '{cname}': {n_junk} non-numeric cell(s) treated as missing")
            col = ([cname], [arr])
        else:
            fmt = _date_format(cells)
            if fmt:                                                          # dates -> days since 1970
                days = [None if c is None else _to_days(c, fmt) for c in cells]
                arr = np.array([np.nan if d is None else d for d in days])
                if np.isfinite(arr).sum() >= 0.9 * len(present) and len(np.unique(arr[np.isfinite(arr)])) > 1:
                    col = ([cname + " (days)"], [arr])
            if col is None and all(c.lower() in _BOOL for c in present) and len({c.lower() for c in present}) == 2:
                col = ([cname], [np.array([np.nan if c is None else _BOOL[c.lower()] for c in cells])])
            if col is None:
                counts = Counter(present)
                if len(counts) == 1 and len(present) == len(cells):
                    dropped.append((cname, "constant"))
                    continue
                if len(counts) > limits.max_levels and len(counts) > 0.5 * len(present):
                    dropped.append((cname, f"{len(counts)} distinct text values: an identifier or free text"))
                    text_cols += 1
                    continue
                levels = [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:limits.max_levels]
                has_other = len(counts) > len(levels)
                if len(levels) == 2 and not has_other and len(present) == len(cells):
                    lv = levels[0]
                    col = ([f"{cname}={lv}"], [np.array([1.0 if c == lv else 0.0 for c in cells])])
                else:
                    cnames, carrs = [], []
                    for lv in levels:
                        cnames.append(f"{cname}={lv}")
                        carrs.append(np.array([1.0 if c == lv else 0.0 for c in cells]))
                    if has_other:
                        cnames.append(f"{cname}=(other)")
                        carrs.append(np.array([1.0 if (c is not None and c not in levels) else 0.0 for c in cells]))
                    if any(c is None for c in cells):
                        cnames.append(f"{cname}=(missing)")
                        carrs.append(np.array([1.0 if c is None else 0.0 for c in cells]))
                    col = (cnames, carrs)
        cnames, carrs = col
        if n_feat + len(cnames) > limits.max_features:
            dropped.append((cname, f"feature cap of {limits.max_features} reached"))
            continue
        for nm, a in zip(cnames, carrs):
            nan = np.isnan(a)
            if nan.any():                                                    # missing numbers: median + an indicator when it matters
                med = float(np.median(a[~nan])) if (~nan).any() else 0.0
                a = np.where(nan, med, a)
                repairs.append(f"column '{nm}': {int(nan.sum())} missing value(s) filled with the median ({med:.4g})")
                if nan.sum() >= max(3, 0.01 * len(a)) and n_feat + len(cnames) + 1 <= limits.max_features:
                    names.append(nm + " (was missing)")
                    blocks.append(nan.astype(np.float64))
                    n_feat += 1
            names.append(nm)
            blocks.append(a)
            n_feat += 1
    if not blocks:
        why = "; ".join(f"{c}: {r}" for c, r in dropped[:5]) or "no feature columns"
        raise Unusable(f"no usable feature column ({why})", textlike=text_cols > 0 and text_cols == len(dropped))
    X = np.column_stack(blocks)
    X = np.where(np.isfinite(X), X, 0.0)
    if dropped:
        notes.append("columns left out: " + "; ".join(f"{c} ({r})" for c, r in dropped[:8]) + (" ..." if len(dropped) > 8 else ""))
    notes.append(f"{kind}: {X.shape[0]:,} rows x {X.shape[1]} features" + (f", {len(class_names)} classes" if class_names else ""))
    spec = dict(task_kind=kind, target=tname, features=names, classes=class_names, rows=int(X.shape[0]), raw_rows=n_raw,
                dropped=[dict(column=c, reason=r) for c, r in dropped])
    return Built(X, np.asarray(y), kind, names, tname, class_names, repairs, notes, spec)
