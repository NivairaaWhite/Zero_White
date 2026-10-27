"""adapters.py - turn rows of any known dataset shape into normalised records.

Detection is by columns, never by file name: the first `data.sniff_rows` rows are offered to every adapter and the one that
can convert the largest share (>= 50%) wins; ties go to the more specific (higher-priority) adapter. Unknown shapes raise
AdapterError listing the columns found and an example dataset.json mapping that fixes it.

All column-name vocabularies are in the tables right below, so supporting a new dataset's naming is a one-line change.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import records as R

# ---- column vocabularies (lower-case) --------------------------------------------------------------------------------
LIST_KEYS = ("messages", "conversation", "conversations", "dialog", "dialogue", "chat", "turns", "history", "utterances")
ROLE_KEYS = ("role", "from", "speaker", "author", "sender", "name")
CONTENT_KEYS = ("content", "value", "text", "message", "utterance", "msg")
QA_USER = ("instruction", "prompt", "question", "query", "problem", "inputs", "input_text", "human", "user", "orig_instruction")
QA_ASSISTANT = ("output", "response", "answer", "solution", "targets", "completion", "target", "assistant", "gpt", "reply")
QA_SYSTEM = ("system_prompt", "system", "persona")
QA_CONTEXT = ("input", "context")
TEXT_KEYS = ("text", "content", "body", "document", "article", "code", "story", "passage", "raw_content")
TOOL_KEYS = ("tools", "functions", "function", "apis", "api_list", "available_tools")
PIPPA_USER_PLACEHOLDERS = ("{{user}}", "<USER>", "{{User}}", "<user>")
PIPPA_CHAR_PLACEHOLDERS = ("{{char}}", "<BOT>", "{{Char}}", "<bot>")


class AdapterError(ValueError):
    pass


@dataclass
class Ctx:
    cfg: Any
    relpath: str = ""
    index: int = 0
    mapping: Dict[str, str] = field(default_factory=dict)
    system: Optional[str] = None            # dataset.json "system": attached to records without one
    user_name: Optional[str] = None
    weight: float = 1.0
    tags: List[str] = field(default_factory=list)
    force_kind: Optional[str] = None        # "chat" / "text" / "preference"

    def rid(self, *extra) -> str:
        return R.stable_id(self.relpath, self.index, *extra)

    def meta(self, adapter: str, **kw) -> dict:
        m = dict(source=self.relpath, adapter=adapter)
        if self.tags:
            m["tags"] = list(self.tags)
        m.update({k: v for k, v in kw.items() if v})
        return m

    @property
    def user_alias(self) -> str:
        return self.user_name or self.cfg.get("persona.user_name") or self.cfg.get("data.user_alias")


# ---- small helpers ----------------------------------------------------------------------------------------------------
def _ci(row: dict) -> Dict[str, Any]:
    return {str(k).lower(): v for k, v in row.items()}


def _first(row_ci: dict, names: Sequence[str]) -> Optional[str]:
    for n in names:
        if n in row_ci and row_ci[n] not in (None, "", [], {}):
            return n
    return None


def _text(v: Any) -> str:
    return R.clean_text(v)


def _maybe_json(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "[{" and s[-1:] in "]}":
            try:
                return json.loads(s)
            except ValueError:
                return v
    return v


def _truncate_system(ctx: Ctx, s: Optional[str]) -> Optional[str]:
    lim = ctx.cfg.get("data.max_system_chars")
    if s and lim and len(s) > lim:
        return s[:lim].rstrip()
    return s


def _render_tools(row_ci: dict) -> str:
    k = _first(row_ci, TOOL_KEYS)
    if not k:
        return ""
    v = _maybe_json(row_ci[k])
    body = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return "Available tools:\n" + body


def _compose_system(ctx: Ctx, *parts: Optional[str]) -> Optional[str]:
    txt = "\n\n".join(p for p in (R.clean_text(x) for x in parts) if p)
    return _truncate_system(ctx, txt or ctx.system)


def _turns_from_list(items: list, user_first: bool = True) -> Tuple[List[dict], List[str]]:
    """list of dicts/strings -> (turns, system texts)."""
    turns: List[dict] = []
    systems: List[str] = []
    for i, it in enumerate(items):
        if isinstance(it, str):
            turns.append({"role": "user" if (i % 2 == 0) == user_first else "assistant", "content": it})
            continue
        if not isinstance(it, dict):
            continue
        low = _ci(it)
        content = None
        for ck in CONTENT_KEYS:
            if ck in low and low[ck] not in (None, ""):
                content = low[ck]
                break
        if content is None:
            continue
        if "is_human" in low:
            role = "user" if low["is_human"] else "assistant"
        elif "is_user" in low:
            role = "user" if low["is_user"] else "assistant"
        else:
            role = None
            for rk in ROLE_KEYS:
                if rk in low:
                    role = R.norm_role(low[rk])
                    if role:
                        break
            if role is None:
                role = "user" if (i % 2 == 0) == user_first else "assistant"
        if role == "system":
            systems.append(_text(content))
        elif role == "tool":
            turns.append({"role": "assistant", "content": "[tool result] " + _text(content)})
        else:
            turns.append({"role": role, "content": content})
    return turns, systems


def _replace_placeholders(s: str, char: str, user: str) -> str:
    for ph in PIPPA_CHAR_PLACEHOLDERS:
        s = s.replace(ph, char)
    for ph in PIPPA_USER_PLACEHOLDERS:
        s = s.replace(ph, user)
    return s


# ---- adapters ---------------------------------------------------------------------------------------------------------
class Adapter:
    name = "base"
    priority = 0
    kinds = ("chat",)

    def detect_row(self, row: dict, ctx: Ctx) -> bool:
        raise NotImplementedError

    def convert(self, row: dict, ctx: Ctx) -> List[dict]:
        raise NotImplementedError


class MappedAdapter(Adapter):
    """data.schema_map / dataset.json "mapping": keys user, assistant, system, text, chosen, rejected, turns, prompt."""
    name, priority = "mapped", 100

    def detect_row(self, row, ctx):
        return bool(ctx.mapping) and any(str(v).lower() in _ci(row) for v in ctx.mapping.values())

    def convert(self, row, ctx):
        m, low = {k: str(v).lower() for k, v in ctx.mapping.items()}, _ci(row)
        get = lambda k: low.get(m[k]) if k in m else None                           # noqa: E731
        sysm = _compose_system(ctx, get("system"))
        meta = ctx.meta(self.name)
        if "turns" in m and isinstance(_maybe_json(get("turns")), list):
            turns, systems = _turns_from_list(_maybe_json(get("turns")))
            rec = R.make_chat(ctx.rid(), _compose_system(ctx, sysm, *systems), turns, ctx.weight, meta)
            return [rec] if rec else []
        if "chosen" in m and "rejected" in m:
            ctxturns = [{"role": "user", "content": _text(get("prompt") or get("user"))}]
            rec = R.make_pref(ctx.rid(), sysm, ctxturns, _text(get("chosen")), _text(get("rejected")), ctx.weight, meta)
            out = [rec] if rec else []
            if out and ctx.cfg.get("data.preference_as_sft"):
                c = R.make_chat(ctx.rid("sft"), sysm, ctxturns + [{"role": "assistant", "content": _text(get("chosen"))}], ctx.weight, meta)
                out += [c] if c else []
            return out
        if "user" in m and "assistant" in m:
            u = _text(get("user"))
            rec = R.make_chat(ctx.rid(), sysm, [{"role": "user", "content": u}, {"role": "assistant", "content": _text(get("assistant"))}],
                              ctx.weight, meta)
            return [rec] if rec else []
        if "text" in m:
            rec = R.make_text(ctx.rid(), _text(get("text")), ctx.weight, meta)
            return [rec] if rec else []
        raise AdapterError(f"schema_map needs one of: turns | user+assistant | chosen+rejected | text (got {sorted(m)})")


class PippaAdapter(Adapter):
    name, priority = "pippa", 90

    def detect_row(self, row, ctx):
        low = _ci(row)
        return "conversation" in low and ("bot_name" in low or "bot_description" in low)

    def convert(self, row, ctx):
        low = _ci(row)
        char = _text(low.get("bot_name")) or "Assistant"
        user = ctx.user_alias
        desc = _replace_placeholders(_text(low.get("bot_description")), char, user)
        defs = _replace_placeholders(_text(low.get("bot_definitions")), char, user)
        persona_text = "\n".join(x for x in (desc, defs) if x)
        sysm = f"You are {char}."
        if desc:
            sysm += " " + desc
        if defs:
            sysm += "\n\n" + defs
        sysm = _truncate_system(ctx, _compose_system(ctx, sysm, ctx.system))
        conv = _maybe_json(low.get("conversation"))
        if not isinstance(conv, list):
            return []
        turns, _ = _turns_from_list([{"message": _replace_placeholders(_text(it.get("message") or it.get("content")), char, user),
                                      "is_human": bool(it.get("is_human"))} for it in conv if isinstance(it, dict)])
        rec = R.make_chat(ctx.rid(), sysm, turns, ctx.weight,
                          ctx.meta(self.name, persona_name=char, persona_text=persona_text,
                                   categories=low.get("categories") if isinstance(low.get("categories"), list) else None))
        return [rec] if rec else []


class ChatListAdapter(Adapter):
    name, priority = "chat_list", 60

    def _list(self, low):
        k = _first(low, LIST_KEYS)
        if not k:
            return None
        v = _maybe_json(low[k])
        return v if isinstance(v, list) and v else None

    def detect_row(self, row, ctx):
        v = self._list(_ci(row))
        return bool(v) and all(isinstance(x, (dict, str)) for x in v[:4])

    def convert(self, row, ctx):
        low = _ci(row)
        turns, systems = _turns_from_list(self._list(low))
        sk = _first(low, QA_SYSTEM)
        sysm = _compose_system(ctx, low[sk] if sk else None, *systems, _render_tools(low), ctx.system if not (sk or systems) else None)
        rec = R.make_chat(ctx.rid(), sysm, turns, ctx.weight, ctx.meta(self.name))
        return [rec] if rec else []


class QAAdapter(Adapter):
    name, priority = "qa", 50

    def _cols(self, low):
        u, a = _first(low, QA_USER), _first(low, QA_ASSISTANT)
        if u is None and a is not None and "input" in low and low["input"]:
            u = "input"
        return u, a

    def detect_row(self, row, ctx):
        low = _ci(row)
        u, a = self._cols(low)
        return u is not None and a is not None and isinstance(low[u], str) and not isinstance(low[a], (list, dict))

    def convert(self, row, ctx):
        low = _ci(row)
        u, a = self._cols(low)
        user = _text(low[u])
        extra = _first(low, QA_CONTEXT)
        if extra and extra != u and _text(low[extra]):
            user = f"{user}\n\n{_text(low[extra])}"
        sk = _first(low, QA_SYSTEM)
        sysm = _compose_system(ctx, low[sk] if sk else None, _render_tools(low))
        lang = low.get("language") or low.get("lang")
        rec = R.make_chat(ctx.rid(), sysm, [{"role": "user", "content": user}, {"role": "assistant", "content": _text(low[a])}],
                          ctx.weight, ctx.meta(self.name, language=lang if isinstance(lang, str) else None))
        return [rec] if rec else []


class MbppAdapter(Adapter):
    name, priority = "mbpp", 80

    def detect_row(self, row, ctx):
        low = _ci(row)
        return "text" in low and "code" in low and ("test_list" in low or "test_setup_code" in low)

    def convert(self, row, ctx):
        low = _ci(row)
        tests = low.get("test_list")
        tests = _maybe_json(tests)
        tests_txt = "\n".join(map(str, tests)) if isinstance(tests, list) else _text(tests)
        prompt = f"Write a Python function. {_text(low.get('text') or low.get('prompt'))}"
        if tests_txt:
            prompt += f"\nYour code should pass these tests:\n{tests_txt}"
        rec = R.make_chat(ctx.rid(), ctx.system, [{"role": "user", "content": prompt}, {"role": "assistant", "content": _text(low.get("code"))}],
                          ctx.weight, ctx.meta(self.name, language="python"))
        return [rec] if rec else []


_HH_SPLIT = re.compile(r"\n\n(Human|Assistant):", re.IGNORECASE)


def parse_hh(text: str) -> List[dict]:
    """'\\n\\nHuman: hi\\n\\nAssistant: hello' -> turns."""
    text = "\n\n" + text.strip() if not text.startswith("\n\n") else text
    parts = _HH_SPLIT.split(text)
    turns = []
    for i in range(1, len(parts) - 1, 2):
        turns.append({"role": "user" if parts[i].lower() == "human" else "assistant", "content": parts[i + 1].strip()})
    return turns


class PreferenceAdapter(Adapter):
    name, priority, kinds = "preference", 85, ("preference", "chat")

    def detect_row(self, row, ctx):
        low = _ci(row)
        if "chosen" in low and "rejected" in low:
            return True
        return "answers" in low and isinstance(_maybe_json(low["answers"]), list) and "prompt" in low

    def _pair(self, low) -> Optional[Tuple[Optional[str], List[dict], str, str]]:
        sysm = low.get("system") if isinstance(low.get("system"), str) else None
        if "chosen" in low and "rejected" in low:
            ch, rj = _maybe_json(low["chosen"]), _maybe_json(low["rejected"])
            if isinstance(ch, str) and isinstance(rj, str):
                if "Human:" in ch or "Assistant:" in ch:
                    ct, rt = parse_hh(ch), parse_hh(rj)
                    if not ct or not rt or ct[-1]["role"] != "assistant" or rt[-1]["role"] != "assistant":
                        return None
                    return sysm, ct[:-1], ct[-1]["content"], rt[-1]["content"]
                prompt = low.get("prompt") or low.get("question") or low.get("instruction")
                if not prompt:
                    return None
                return sysm, [{"role": "user", "content": _text(prompt)}], ch, rj
            if isinstance(ch, list) and isinstance(rj, list):
                ct, cs = _turns_from_list(ch)
                rt, _ = _turns_from_list(rj)
                if not ct or not rt or ct[-1]["role"] != "assistant" or rt[-1]["role"] != "assistant":
                    return None
                ctxt = ct[:-1]
                if not ctxt and low.get("prompt"):
                    ctxt = [{"role": "user", "content": _text(low["prompt"])}]
                return (cs[0] if cs else sysm), ctxt, ct[-1]["content"], rt[-1]["content"]
            return None
        answers = _maybe_json(low["answers"])
        scored = [a for a in answers if isinstance(a, dict) and (a.get("answer") or a.get("text"))]
        if len(scored) < 2:
            return None
        key = (lambda a: a.get("rank", 99))
        scored.sort(key=key)
        prompt = _text(low["prompt"])
        ctxt = parse_hh(prompt) if "Human:" in prompt else [{"role": "user", "content": prompt}]
        if ctxt and ctxt[-1]["role"] == "assistant" and not ctxt[-1]["content"]:
            ctxt = ctxt[:-1]
        best, worst = scored[0], scored[-1]
        return sysm, ctxt, _text(best.get("answer") or best.get("text")), _text(worst.get("answer") or worst.get("text"))

    def convert(self, row, ctx):
        pr = self._pair(_ci(row))
        if not pr:
            return []
        sysm, context, chosen, rejected = pr
        sysm = _compose_system(ctx, sysm)
        meta = ctx.meta(self.name)
        out = []
        rec = R.make_pref(ctx.rid(), sysm, context, chosen, rejected, ctx.weight, meta)
        if rec:
            out.append(rec)
            if ctx.cfg.get("data.preference_as_sft"):
                c = R.make_chat(ctx.rid("sft"), sysm, list(context) + [{"role": "assistant", "content": chosen}], ctx.weight, meta)
                if c:
                    out.append(c)
        return out


class CritiqueAdapter(Adapter):
    """Prometheus-style judge data from the original columns (when the pre-rendered instruction/output columns are absent)."""
    name, priority = "critique", 70

    def detect_row(self, row, ctx):
        low = _ci(row)
        return "orig_instruction" in low and "orig_response" in low and "orig_feedback" in low and "instruction" not in low

    def convert(self, row, ctx):
        low = _ci(row)
        rubric = [f"Score {i}: {_text(low.get(f'orig_score{i}_description'))}" for i in range(1, 6) if low.get(f"orig_score{i}_description")]
        prompt = ("###Task Description:\nAn instruction, a response to evaluate and a score rubric are given. Write detailed feedback that "
                  "assesses the response strictly by the rubric, then end with '[RESULT] <score>'.\n\n"
                  f"###The instruction to evaluate:\n{_text(low['orig_instruction'])}\n\n###Response to evaluate:\n{_text(low['orig_response'])}\n\n")
        if low.get("orig_reference_answer"):
            prompt += f"###Reference Answer (Score 5):\n{_text(low['orig_reference_answer'])}\n\n"
        prompt += f"###Score Rubrics:\n[{_text(low.get('orig_criteria'))}]\n" + "\n".join(rubric)
        ans = f"{_text(low['orig_feedback'])} [RESULT] {_text(low.get('orig_score'))}".strip()
        rec = R.make_chat(ctx.rid(), ctx.system, [{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}],
                          ctx.weight, ctx.meta(self.name))
        return [rec] if rec else []


class TextAdapter(Adapter):
    name, priority, kinds = "text", 10, ("text",)

    def detect_row(self, row, ctx):
        low = _ci(row)
        k = _first(low, TEXT_KEYS)
        return k is not None and isinstance(low[k], str) and len(low[k]) > 0

    def convert(self, row, ctx):
        low = _ci(row)
        k = _first(low, TEXT_KEYS)
        ext = low.get("_ext")
        meta = ctx.meta(self.name, language=ext.lstrip(".") if isinstance(ext, str) and ext else None, path=low.get("_path"))
        rec = R.make_text(ctx.rid(low.get("_part", "")), _text(low[k]), ctx.weight, meta)
        return [rec] if rec else []


ADAPTERS: List[Adapter] = sorted([MappedAdapter(), PippaAdapter(), PreferenceAdapter(), MbppAdapter(), CritiqueAdapter(), ChatListAdapter(),
                                  QAAdapter(), TextAdapter()], key=lambda a: -a.priority)
BY_NAME: Dict[str, Adapter] = {a.name: a for a in ADAPTERS}


def columns_of(rows: List[dict]) -> List[str]:
    seen: Dict[str, int] = {}
    for r in rows:
        for k in r:
            if not str(k).startswith("_"):
                seen[k] = seen.get(k, 0) + 1
    return sorted(seen, key=lambda k: -seen[k])


def detect_adapter(rows: List[dict], ctx: Ctx, forced: Optional[str] = None) -> Tuple[Adapter, float]:
    """Choose the adapter for a file from sniffed rows. Raises AdapterError with advice when nothing fits."""
    if ctx.force_kind == "text" and not forced:
        forced = "text"
    if forced and forced != "auto":
        if forced not in BY_NAME:
            raise AdapterError(f"unknown adapter '{forced}'; available: {', '.join(sorted(BY_NAME))}")
        return BY_NAME[forced], 1.0
    if not rows:
        raise AdapterError("file has no readable rows")
    best, best_frac = None, 0.0
    for a in ADAPTERS:
        ok = 0
        for r in rows:
            try:
                ok += 1 if a.detect_row(r, ctx) else 0
            except Exception:  # noqa: BLE001 - a detector must never crash ingestion
                pass
        frac = ok / len(rows)
        if frac >= 0.5 and (best is None or frac > best_frac + 1e-9):
            best, best_frac = a, frac
    if best is None:
        cols = columns_of(rows)
        raise AdapterError(
            f"could not recognise this dataset's layout. Columns found: {cols[:20]}. Put a dataset.json next to the file, e.g. "
            '{"mapping": {"user": "' + (cols[0] if cols else "question") + '", "assistant": "' + (cols[1] if len(cols) > 1 else "answer") + '"}} '
            'or {"adapter": "text"}')
    return best, best_frac


def convert_row(adapter: Adapter, row: dict, ctx: Ctx) -> List[dict]:
    try:
        return adapter.convert(row, ctx)
    except AdapterError:
        raise
    except Exception:  # noqa: BLE001 - one malformed row must not kill a 2-million-row file
        return []
