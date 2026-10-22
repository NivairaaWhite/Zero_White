"""records.py - the one normalised record format every dataset is converted into.

    {"kind": "chat",       "id", "system", "turns": [{"role","content"}...], "weight", "meta"}
    {"kind": "preference", "id", "system", "turns": [context...], "chosen", "rejected", "weight", "meta"}
    {"kind": "text",       "id", "text", "weight", "meta"}

`id` is a stable hash of where the record came from (file + row), so re-ingesting a file reproduces the same ids and the
train/validation split (which hashes the id) never reshuffles.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, List, Optional

# role names found in public datasets -> our three roles ("tool"/"function" turns are kept as assistant-side text)
ROLE_MAP = {
    "user": "user", "human": "user", "prompter": "user", "customer": "user", "person": "user", "questioner": "user",
    "input": "user", "instruction": "user", "client": "user", "speaker1": "user", "usr": "user", "you": "user",
    "assistant": "assistant", "gpt": "assistant", "bot": "assistant", "model": "assistant", "ai": "assistant",
    "chatgpt": "assistant", "claude": "assistant", "response": "assistant", "output": "assistant", "agent": "assistant",
    "answer": "assistant", "speaker2": "assistant", "character": "assistant", "gpt4": "assistant", "bing": "assistant",
    "system": "system", "persona": "system", "context": "system",
    "tool": "tool", "function": "tool", "observation": "tool", "function_call": "assistant", "function_response": "tool",
}


def norm_role(r: Any) -> Optional[str]:
    if r is None:
        return None
    return ROLE_MAP.get(str(r).strip().lower().replace("-", "").replace(" ", ""))


def clean_text(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False) if isinstance(s, (dict, list)) else str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return s.strip()


def stable_id(*parts: Any) -> str:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()


def merge_turns(turns: List[dict]) -> List[dict]:
    """Drop empty turns, force known roles, merge consecutive turns of the same role."""
    out: List[dict] = []
    for t in turns:
        role, text = t.get("role"), clean_text(t.get("content"))
        if role not in ("user", "assistant") or not text:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + text
        else:
            out.append({"role": role, "content": text})
    return out


def make_chat(rid: str, system: Optional[str], turns: List[dict], weight: float = 1.0, meta: Optional[dict] = None):
    turns = merge_turns(turns)
    if not turns:
        return None
    system = clean_text(system) or None
    return {"kind": "chat", "id": rid, "system": system, "turns": turns, "weight": float(weight), "meta": meta or {}}


def make_text(rid: str, text: str, weight: float = 1.0, meta: Optional[dict] = None):
    text = clean_text(text)
    if not text:
        return None
    return {"kind": "text", "id": rid, "text": text, "weight": float(weight), "meta": meta or {}}


def make_pref(rid: str, system: Optional[str], turns: List[dict], chosen: str, rejected: str, weight: float = 1.0,
              meta: Optional[dict] = None):
    chosen, rejected = clean_text(chosen), clean_text(rejected)
    turns = merge_turns(turns)
    if not chosen or not rejected or chosen == rejected or not turns:
        return None
    return {"kind": "preference", "id": rid, "system": clean_text(system) or None, "turns": turns, "chosen": chosen,
            "rejected": rejected, "weight": float(weight), "meta": meta or {}}


def texts_of(rec: dict) -> List[str]:
    """Every human-readable string of a record (for filtering)."""
    k = rec["kind"]
    out: List[str] = []
    if rec.get("system"):
        out.append(rec["system"])
    if k == "text":
        out.append(rec["text"])
    else:
        out.extend(t["content"] for t in rec["turns"])
    if k == "preference":
        out.extend([rec["chosen"], rec["rejected"]])
    return out


def canonical_text(rec: dict) -> str:
    return "\x1e".join(texts_of(rec)) + "\x1d" + rec["kind"]


def record_hash(rec: dict) -> bytes:
    return hashlib.blake2b(canonical_text(rec).encode("utf-8", "replace"), digest_size=8).digest()


def map_strings(rec: dict, fn) -> dict:
    """Apply fn to every text field, returning a new record."""
    r = dict(rec)
    if r.get("system"):
        r["system"] = fn(r["system"])
    if r["kind"] == "text":
        r["text"] = fn(r["text"])
    else:
        r["turns"] = [dict(t, content=fn(t["content"])) for t in r["turns"]]
    if r["kind"] == "preference":
        r["chosen"], r["rejected"] = fn(r["chosen"]), fn(r["rejected"])
    return r
