"""convert.py - convert any supported dataset into another format (or into Zero White's own normalised records).

    zerowhite chat convert openhermes.json -o hermes_openai.jsonl --to openai
    zerowhite chat convert pippa.jsonl.gz -o sister.jsonl --to normalized --filter

Targets: normalized | openai | sharegpt | alpaca | chatml | plain | csv | preference | text.
Output container is chosen by the extension of -o (.jsonl, .json, .csv, .txt, each optionally + .gz).
"""
from __future__ import annotations

import csv
import gzip
import json
import os
from typing import Callable, Iterator, List, Optional

from ..config import Config
from . import adapters as A
from . import readers as RD
from .filters import RecordFilter
from ... import fileio

TARGETS = ("normalized", "openai", "sharegpt", "alpaca", "chatml", "plain", "csv", "preference", "text")


def render_plain(rec: dict) -> str:
    out = []
    if rec.get("system"):
        out.append(f"System: {rec['system']}")
    if rec["kind"] == "text":
        return rec["text"]
    for t in rec["turns"]:
        out.append(f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}")
    if rec["kind"] == "preference":
        out.append(f"Assistant: {rec['chosen']}")
    return "\n".join(out)


def _chatml(rec: dict) -> str:
    parts = []
    if rec.get("system"):
        parts.append(f"<|im_start|>system\n{rec['system']}<|im_end|>")
    for t in rec["turns"]:
        parts.append(f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>")
    return "\n".join(parts)


def format_record(rec: dict, to: str):
    """One normalised record -> the target item (dict or str), or None when this target cannot represent the record."""
    k = rec["kind"]
    if to == "normalized":
        return rec
    if to == "text":
        return {"text": rec["text"] if k == "text" else render_plain(rec)}
    if to == "plain":
        return render_plain(rec)
    if to == "preference":
        if k != "preference":
            return None
        prompt = render_plain(dict(rec, kind="chat"))
        return {"prompt": prompt, "chosen": rec["chosen"], "rejected": rec["rejected"]}
    if k != "chat":
        return None
    if to == "openai":
        msgs = ([{"role": "system", "content": rec["system"]}] if rec.get("system") else []) + rec["turns"]
        return {"messages": msgs}
    if to == "sharegpt":
        conv = ([{"from": "system", "value": rec["system"]}] if rec.get("system") else [])
        conv += [{"from": "human" if t["role"] == "user" else "gpt", "value": t["content"]} for t in rec["turns"]]
        return {"conversations": conv}
    if to == "chatml":
        return {"text": _chatml(rec)}
    turns = rec["turns"]
    last_a = max((i for i, t in enumerate(turns) if t["role"] == "assistant"), default=None)
    if last_a is None:
        return None
    last_u = max((i for i, t in enumerate(turns[:last_a]) if t["role"] == "user"), default=None)
    if last_u is None:
        return None
    hist = render_plain(dict(rec, turns=turns[:last_u], kind="chat", system=rec.get("system"))) if (turns[:last_u] or rec.get("system")) else ""
    if to == "alpaca":
        return {"instruction": turns[last_u]["content"], "input": hist, "output": turns[last_a]["content"]}
    if to == "csv":
        return {"system": rec.get("system") or "", "prompt": turns[last_u]["content"], "response": turns[last_a]["content"]}
    raise ValueError(f"unknown target '{to}'")


def _files(src: str) -> List[tuple]:
    if os.path.isdir(src):
        out = []
        for d, dirs, files in os.walk(src):
            dirs[:] = sorted(x for x in dirs if not x.startswith("."))
            for fn in sorted(files):
                if RD.file_kind(fn) not in (None, "audio") and not fn.startswith(".") and fn != "dataset.json":
                    p = os.path.join(d, fn)
                    out.append((p, os.path.relpath(p, src).replace(os.sep, "/")))
        return out
    return [(src, os.path.basename(src))]


def iter_records(src: str, cfg: Config, adapter: Optional[str] = None, limit: Optional[int] = None,
                 apply_filters: bool = True, stats: Optional[dict] = None) -> Iterator[dict]:
    stats = stats if stats is not None else {}
    filt = RecordFilter(cfg) if apply_filters else None
    n = 0
    for path, rel in _files(src):
        try:
            ctx = A.Ctx(cfg=cfg, relpath=rel, mapping=dict(cfg.get("data.schema_map") or {}))
            ad, _ = A.detect_adapter(RD.sniff(path, int(cfg.get("data.sniff_rows")), cfg), ctx,
                                     adapter or (cfg.get("data.adapter") if cfg.get("data.adapter") != "auto" else None))
            stats.setdefault("adapters", {})[rel] = ad.name
            for idx, row in enumerate(RD.iter_rows(path, cfg)):
                ctx.index = idx
                for rec in A.convert_row(ad, row, ctx):
                    stats["read"] = stats.get("read", 0) + 1
                    if filt:
                        rec, why = filt.apply(rec)
                        if rec is None:
                            d = stats.setdefault("dropped", {})
                            d[why] = d.get(why, 0) + 1
                            continue
                    yield rec
                    n += 1
                    if limit and n >= limit:
                        return
        except (A.AdapterError, RD.ReaderError) as e:
            stats.setdefault("errors", []).append((rel, str(e)))


def convert(src: str, dst: str, to: Optional[str] = None, cfg: Optional[Config] = None, adapter: Optional[str] = None,
            limit: Optional[int] = None, apply_filters: bool = True, progress: Optional[Callable[[int], None]] = None,
            overwrite: bool = False) -> dict:
    cfg = cfg or Config.load()
    base = dst[:-3] if dst.endswith(".gz") else dst
    ext = os.path.splitext(base)[1].lower() or ".jsonl"
    if to is None:
        to = "csv" if ext == ".csv" else "plain" if ext == ".txt" else "normalized"
    if to not in TARGETS:
        raise ValueError(f"--to must be one of {TARGETS}")
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError("the output file is the input file; choose another name")
    if os.path.exists(dst) and not overwrite:
        raise FileExistsError(f"{dst} already exists; choose another name or pass overwrite (--overwrite on the command line)")
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    tmp = os.path.join(os.path.dirname(os.path.abspath(dst)), f".tmp_{os.path.basename(dst)}")
    stats: dict = {}
    written = skipped = 0
    opener = (lambda p: gzip.open(p, "wt", encoding="utf-8", newline="")) if dst.endswith(".gz") else (lambda p: open(p, "w", encoding="utf-8", newline=""))
    writer = None
    first = True
    try:
        with opener(tmp) as out:
            if ext == ".json":
                out.write("[")
            for rec in iter_records(src, cfg, adapter, limit, apply_filters, stats):
                item = format_record(rec, to)
                if item is None:
                    skipped += 1
                    continue
                if ext == ".csv":
                    if not isinstance(item, dict):
                        item = {"text": item}
                    if writer is None:
                        writer = csv.DictWriter(out, fieldnames=list(item), extrasaction="ignore", lineterminator="\n")
                        writer.writeheader()
                    writer.writerow(item)
                elif ext == ".json":
                    out.write(("" if first else ",\n") + json.dumps(item, ensure_ascii=False))
                elif ext == ".txt":
                    out.write(("" if first else "\n\n") + (item if isinstance(item, str) else item.get("text") or json.dumps(item, ensure_ascii=False)))
                else:
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                first = False
                written += 1
                if progress and written % 1000 == 0:
                    progress(written)
            if ext == ".json":
                out.write("]\n")
        fileio.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dict(written=written, skipped_wrong_kind=skipped, read=stats.get("read", 0), dropped=stats.get("dropped", {}),
                errors=stats.get("errors", []), adapters=stats.get("adapters", {}), output=dst, target=to)
