"""filters.py - quality, persona and safety filtering of normalised records (all behaviour comes from `filter.*` config)."""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Tuple

from . import records as R
from .safety import SafetyFilter


def keyword_regex(keywords: List[str], mode: str) -> Optional["re.Pattern"]:
    pats = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        if mode == "regex":
            pats.append(kw)
            continue
        tokens = [t for t in re.split(r"[\s\-]+", kw) if t]
        body = r"[\s\-]*".join(re.escape(t) for t in tokens)
        pats.append(body if mode == "substring" else rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])")
    if not pats:
        return None
    try:
        return re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE)
    except re.error as e:
        raise ValueError(f"filter.persona.keywords: invalid pattern ({e})") from e


class PersonaFilter:
    def __init__(self, cfg):
        p = cfg.filter.persona
        self.enabled = bool(p.enabled)
        self.where = set(p.where)
        self.require_all = p.require == "all"
        self.keywords = [k for k in p.keywords if k.strip()]
        self.rx = keyword_regex(self.keywords, p.match) if self.enabled else None
        self.each = [keyword_regex([k], p.match) for k in self.keywords] if (self.enabled and self.require_all) else []

    def haystack(self, rec: dict) -> str:
        parts: List[str] = []
        if "system" in self.where and rec.get("system"):
            parts.append(rec["system"])
        if "meta" in self.where:
            m = rec.get("meta", {})
            parts += [str(m.get(k, "")) for k in ("persona_name", "persona_text")]
            parts += [str(t) for t in m.get("tags", [])]
            parts += [str(c) for c in (m.get("categories") or [])]
        if "turns" in self.where:
            parts += [t["content"] for t in rec.get("turns", [])] + ([rec["text"]] if rec["kind"] == "text" else [])
        return "\n".join(parts)

    def matches(self, rec: dict) -> bool:
        if not self.enabled:
            return True
        h = self.haystack(rec)
        if self.require_all:
            return all(rx.search(h) for rx in self.each)
        return bool(self.rx and self.rx.search(h))


class RecordFilter:
    """quality -> persona -> safety -> PII redaction. apply() returns (record_or_None, reason_or_'')."""

    def __init__(self, cfg):
        self.cfg = cfg
        q = cfg.filter.quality
        self.min_chars = int(cfg.get("data.min_chars"))
        self.max_chars = cfg.get("data.max_chars")
        self.require_assistant = bool(q.require_assistant)
        self.min_asst = int(q.min_assistant_chars)
        self.max_repeat = float(q.max_repeat_ratio)
        self.persona = PersonaFilter(cfg)
        self.safety = SafetyFilter(cfg)

    def _quality(self, rec: dict) -> str:
        k = rec["kind"]
        strings = R.texts_of(rec)
        longest = max((len(s) for s in strings), default=0)
        if longest < self.min_chars:
            return "quality_short"
        if self.max_chars and sum(len(s) for s in strings) > self.max_chars:
            return "quality_long"
        if k == "chat":
            asst = [t["content"] for t in rec["turns"] if t["role"] == "assistant"]
            if self.require_assistant and not asst:
                return "quality_no_assistant"
            if asst and sum(len(a) for a in asst) < self.min_asst:
                return "quality_short"
            body = " ".join(asst)
        elif k == "text":
            body = rec["text"]
        else:
            body = rec["chosen"]
        words = body.lower().split()
        if len(words) >= 20 and self.max_repeat < 1.0:
            top = Counter(words).most_common(1)[0][1]
            if top / len(words) > self.max_repeat:
                return "quality_repetitive"
        return ""

    def apply(self, rec: dict) -> Tuple[Optional[dict], str]:
        why = self._quality(rec)
        if why:
            return None, why
        if not self.persona.matches(rec):
            return None, "persona_mismatch"
        why = self.safety.verdict("\n".join(R.texts_of(rec)))
        if why:
            return None, why
        if self.safety.redact_kinds:
            rec = R.map_strings(rec, self.safety.redact_text)
        return rec, ""


def describe_filters(cfg) -> str:
    f = RecordFilter(cfg)
    lines = [f"quality : min_chars={f.min_chars} max_chars={f.max_chars} require_assistant={f.require_assistant} "
             f"max_repeat_ratio={f.max_repeat}",
             f"persona : {'ON keywords=' + str(f.persona.keywords) + ' match=' + cfg.filter.persona.match if f.persona.enabled else 'off'}",
             f"safety  : drop_sexual={f.safety.drop_sexual} (sexual content with a minor or family member is ALWAYS dropped) "
             f"redact={f.safety.redact_kinds} extra_terms={len(f.safety.terms)} extra_regex={len(f.safety.extra_re)}"]
    return "\n".join(lines)
