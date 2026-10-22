"""datasets.py - find (audio, transcript) pairs in a speech dataset folder, whatever its layout.

Understood layouts (nothing is keyed to a dataset name; the rules are about file structure):
  * a metadata table (.csv / .tsv / .jsonl / .json) with a column naming the audio file and a column holding the text
        header-less `id|text|normalised text`  (LJ Speech style)        header `path<TAB>sentence`  (Common Voice style)
  * a sidecar transcript with the same name as the audio: clip.wav + clip.txt / .lab / .trn  (also when the texts live in a
    parallel folder, e.g. wav48/p225/p225_001.flac  <->  txt/p225/p225_001.txt  (VCTK style))
A `dataset.json` next to a metadata file can state the answer outright:
    {"audio_column": "file", "text_column": "words", "delimiter": "|", "audio_dir": "wavs", "text_column_index": 2}
"""
from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

from .audio import AUDIO_EXTENSIONS

AUDIO_COLS = ("path", "file", "filename", "file_name", "audio", "audio_path", "audio_file", "wav", "wav_path", "clip", "clip_path", "utt_id", "id", "uid")
TEXT_COLS = ("sentence", "text", "transcript", "transcription", "normalized_text", "normalised_text", "normalized", "utterance", "words", "label", "raw_text")
TEXT_SIDECAR = (".txt", ".lab", ".trn", ".text")
TABLE_EXT = (".csv", ".tsv", ".txt")           # .txt only counts as a table when it has delimiters on every line and is not a sidecar
JSON_EXT = (".jsonl", ".ndjson", ".json")


def normalize_text(s: str) -> str:
    """The transcript form the recogniser is trained on and decodes to: lower case, letters/digits/apostrophes/spaces only."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
    out = []
    for ch in s:
        if ch.isalnum() or ch == "'":
            out.append(ch)
        elif ch.isspace() or unicodedata.category(ch).startswith(("P", "S", "Z")):
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _stem_key(name: str) -> str:
    """Join key for 'p225_001_mic1' <-> 'p225_001' and similar microphone/rate suffixes."""
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    return re.sub(r"[_\-.](mic\d+|16k|24k|22k|44k|48k|norm|trim|clean)$", "", stem)


class AudioIndex:
    def __init__(self, root: str):
        self.root = root
        self.by_rel: Dict[str, str] = {}
        self.by_stem: Dict[str, List[str]] = defaultdict(list)
        self.texts: Dict[str, List[str]] = defaultdict(list)
        self.tables: List[str] = []
        self.sidecars: List[str] = []
        for dp, dn, fn in os.walk(root):
            dn[:] = sorted(d for d in dn if not d.startswith("."))
            for f in sorted(fn):
                if f.startswith("."):
                    continue
                ap = os.path.join(dp, f)
                ext = os.path.splitext(f)[1].lower()
                rel = os.path.relpath(ap, root).replace(os.sep, "/")
                if ext in AUDIO_EXTENSIONS:
                    self.by_rel[rel.lower()] = ap
                    self.by_stem[_stem_key(f)].append(ap)
                elif ext in (".csv", ".tsv") or ext in JSON_EXT or f.lower() == "dataset.json":
                    if f.lower() != "dataset.json":
                        self.tables.append(ap)
                elif ext in TEXT_SIDECAR:
                    self.texts[_stem_key(f)].append(ap)
                    if f.lower().startswith(("metadata", "transcript", "train", "validated", "test", "dev", "labels")):
                        self.tables.append(ap)

    def find_audio(self, ref: str, base_dirs: List[str]) -> Optional[str]:
        ref = ref.strip().strip('"')
        if not ref:
            return None
        cands = [ref]
        if not os.path.splitext(ref)[1].lower() in AUDIO_EXTENSIONS:
            cands += [ref + e for e in (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")]
        for c in cands:
            if os.path.isabs(c) and os.path.isfile(c):
                return c
            for b in base_dirs:
                p = os.path.normpath(os.path.join(b, c))
                if os.path.isfile(p):
                    return p
                for sub in ("wavs", "wav", "clips", "audio", "flac"):
                    p2 = os.path.normpath(os.path.join(b, sub, c))
                    if os.path.isfile(p2):
                        return p2
            hit = self.by_rel.get(c.replace("\\", "/").lower())
            if hit:
                return hit
        stems = self.by_stem.get(_stem_key(ref))
        return stems[0] if stems else None


def _sniff_delim(sample: str) -> str:
    counts = {d: sample.count(d) for d in ("|", "\t", ",", ";")}
    return max(counts, key=counts.get) if max(counts.values()) else "\t"


def _read_table(path: str, side: dict) -> Iterator[dict]:
    """Rows as {column: value} for csv/tsv (with or without header) and jsonl/json."""
    ext = os.path.splitext(path)[1].lower()
    if ext in JSON_EXT:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            head = f.read(1)
            f.seek(0)
            if head == "[" or ext == ".json":
                try:
                    data = json.load(f)
                except ValueError:
                    return
                rows = data if isinstance(data, list) else data.get("rows", []) if isinstance(data, dict) else []
                for r in rows:
                    if isinstance(r, dict):
                        yield r.get("row", r)
            else:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(r, dict):
                        yield r
        return
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        if not sample.strip():
            return
        delim = side.get("delimiter") or (_sniff_delim(sample) if ext != ".tsv" else "\t")
        first = sample.splitlines()[0].split(delim)
        low = [c.strip().lower() for c in first]
        has_header = any(c in AUDIO_COLS + TEXT_COLS for c in low) and not side.get("no_header")
        if has_header:
            for r in csv.DictReader(f, delimiter=delim, quoting=csv.QUOTE_NONE if delim == "|" else csv.QUOTE_MINIMAL):
                yield {(k or "").strip(): v for k, v in r.items() if k is not None}
        else:
            for r in csv.reader(f, delimiter=delim, quoting=csv.QUOTE_NONE if delim == "|" else csv.QUOTE_MINIMAL):
                if r:
                    yield {f"#{i}": v for i, v in enumerate(r)}


def _pick(row: dict, names, side_key: Optional[str]) -> Optional[str]:
    if side_key and side_key in row:
        return str(row[side_key])
    low = {k.lower(): k for k in row}
    for n in names:
        if n in low and row[low[n]] not in (None, ""):
            v = row[low[n]]
            if isinstance(v, dict):                                        # HF style {"path": ..., "bytes": ...}
                v = v.get("path") or v.get("src") or ""
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                v = v[0].get("src") or v[0].get("path") or ""
            return str(v)
    return None


def discover(root: str, limit: Optional[int] = None) -> Tuple[List[Tuple[str, str]], dict]:
    """[(audio_path, normalised transcript)], and a report {tables, sidecars, missing_audio, empty_text}."""
    idx = AudioIndex(root)
    pairs: Dict[str, str] = {}
    rep = dict(tables=[], sidecars=0, missing_audio=0, empty_text=0, audio_files=sum(len(v) for v in idx.by_stem.values()))
    for table in idx.tables:
        folder = os.path.dirname(table)
        side: dict = {}
        for d in (folder, root):
            sp = os.path.join(d, "dataset.json")
            if os.path.isfile(sp):
                try:
                    side = json.load(open(sp, encoding="utf-8"))
                except (OSError, ValueError):
                    side = {}
                break
        bases = [folder, root] + ([os.path.join(folder, side["audio_dir"]), os.path.join(root, side["audio_dir"])] if side.get("audio_dir") else [])
        n_before = len(pairs)
        for row in _read_table(table, side):
            ref = txt = None
            if any(k.startswith("#") for k in row):                          # header-less: first column id, text = column index or the last text-like one
                ref = row.get("#0")
                ti = side.get("text_column_index")
                if ti is None:
                    ti = 2 if "#2" in row and len((row.get("#2") or "").strip()) else 1
                txt = row.get(f"#{ti}")
            else:
                ref = _pick(row, AUDIO_COLS, side.get("audio_column"))
                txt = _pick(row, TEXT_COLS, side.get("text_column"))
            if not ref or txt is None:
                continue
            ap = idx.find_audio(ref, bases)
            if ap is None:
                rep["missing_audio"] += 1
                continue
            t = normalize_text(txt)
            if not t:
                rep["empty_text"] += 1
                continue
            pairs.setdefault(ap, t)
            if limit and len(pairs) >= limit:
                break
        if len(pairs) > n_before:
            rep["tables"].append(os.path.relpath(table, root))
        if limit and len(pairs) >= limit:
            break
    # sidecar transcripts for audio not yet covered
    for stem, files in idx.by_stem.items():
        for ap in files:
            if ap in pairs:
                continue
            tx = idx.texts.get(stem)
            if not tx:
                continue
            same_dir = [t for t in tx if os.path.dirname(t) == os.path.dirname(ap)]
            try:
                with open((same_dir or tx)[0], "r", encoding="utf-8", errors="replace") as f:
                    t = normalize_text(f.read())
            except OSError:
                continue
            if t:
                pairs[ap] = t
                rep["sidecars"] += 1
        if limit and len(pairs) >= limit:
            break
    return list(pairs.items()), rep
