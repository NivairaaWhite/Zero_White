"""session.py - multi-turn conversations on top of the Engine.

One ChatService is shared by every front end (terminal chat, HTTP/web UI, voice loop). A Session holds the turns of one conversation
and its KV cache, so each new user message only costs the new tokens. Every reply is logged to the experience log with an id, which is
what thumbs-up / thumbs-down / corrections / "regenerate" refer to (see online.py). Transcripts are kept in state/sessions/<id>.jsonl.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Dict, Iterator, List, Optional

from .config import Config
from .generate import Engine, Generator
from .online import ExperienceLog
from .paths import Layout, safe_name
from .util import append_jsonl


class Session:
    def __init__(self, sid: str, system: Optional[str]):
        self.id, self.system = sid, system
        self.turns: List[dict] = []                 # {"role", "content", "id"?}
        self.gen = Generator()
        self.lock = threading.Lock()
        self.last_used = time.time()
        self.last_regen_of: Optional[str] = None

    def public_turns(self) -> List[dict]:
        return [dict(role=t["role"], content=t["content"], id=t.get("id")) for t in self.turns]


class ChatService:
    def __init__(self, cfg: Config, lay: Layout, engine: Engine):
        self.cfg, self.lay, self.engine = cfg, lay, engine
        self.explog = ExperienceLog(lay)
        self.sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    # ---- sessions -------------------------------------------------------------------------------------------------
    @staticmethod
    def new_id() -> str:
        return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]

    def _path(self, sid: str) -> str:
        return os.path.join(self.lay.sessions_dir, safe_name(sid) + ".jsonl")

    def get(self, sid: Optional[str] = None, system: Optional[str] = None) -> Session:
        sid = safe_name(sid) if sid else self.new_id()
        with self._lock:
            self.evict_idle()
            s = self.sessions.get(sid)
            if s is None:
                s = Session(sid, system if system is not None else self.cfg.system_prompt())
                self._load(s)
                self.sessions[sid] = s
            s.last_used = time.time()
            return s

    def _load(self, s: Session):
        p = self._path(s.id)
        if not os.path.exists(p):
            return
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("event") == "discard":
                    for i in range(len(s.turns) - 1, -1, -1):
                        if s.turns[i].get("id") == e.get("reply_id"):
                            del s.turns[i]
                            break
                elif e.get("event") == "system":
                    s.system = e.get("content")
                elif e.get("role") in ("user", "assistant"):
                    s.turns.append(dict(role=e["role"], content=e["content"], id=e.get("reply_id")))

    def _persist(self, s: Session, **ev):
        append_jsonl(self._path(s.id), dict(t=round(time.time(), 2), **ev))

    def evict_idle(self):
        ttl = float(self.cfg.get("server.session_ttl_s"))
        now = time.time()
        for k in [k for k, v in self.sessions.items() if now - v.last_used > ttl and not v.lock.locked()]:
            del self.sessions[k]

    def reset(self, sid: str) -> Session:
        with self._lock:
            self.sessions.pop(safe_name(sid), None)
        return self.get(None)

    def history(self, sid: str) -> List[dict]:
        return self.get(sid).public_turns()

    # ---- replies --------------------------------------------------------------------------------------------------
    def send(self, sid: Optional[str], text: str, **gen) -> Iterator[dict]:
        """Answer one user message. Yields {"delta": text} pieces, then {"done": True, "reply_id", "text", "session"}."""
        s = self.get(sid)
        text = (text or "").strip()
        if not text:
            raise ValueError("empty message")
        with s.lock:
            s.turns.append(dict(role="user", content=text))
            self._persist(s, role="user", content=text)
            yield from self._answer(s, regen_of=None, gen=gen)

    def regenerate(self, sid: str, **gen) -> Iterator[dict]:
        """Throw away the last reply and answer again (the old reply becomes the 'rejected' side of a preference if the new one is liked)."""
        s = self.get(sid)
        with s.lock:
            if not s.turns or s.turns[-1]["role"] != "assistant":
                raise ValueError("there is no reply to regenerate")
            old = s.turns.pop()
            self._persist(s, event="discard", reply_id=old.get("id"))
            yield from self._answer(s, regen_of=old.get("id"), gen=gen)

    def _answer(self, s: Session, regen_of: Optional[str], gen: dict) -> Iterator[dict]:
        context = [dict(role=t["role"], content=t["content"]) for t in s.turns]
        parts: List[str] = []
        for d in self.engine.stream(s.gen, s.system, context, **gen):
            parts.append(d)
            yield dict(delta=d)
        reply = "".join(parts).strip()
        rid = self.explog.log_reply(s.id, s.system, context, reply, self.engine.version, regen_of=regen_of)
        s.turns.append(dict(role="assistant", content=reply, id=rid))
        self._persist(s, role="assistant", content=reply, reply_id=rid, version=self.engine.version)
        yield dict(done=True, reply_id=rid, text=reply, session=s.id, version=self.engine.version)

    def reply(self, sid: Optional[str], text: str, **gen) -> dict:
        last: dict = {}
        for ev in self.send(sid, text, **gen):
            last = ev
        return last

    # ---- feedback -------------------------------------------------------------------------------------------------
    def feedback(self, reply_id: str, rating: int = 0, correction: Optional[str] = None) -> dict:
        if rating not in (-1, 0, 1):
            raise ValueError("rating must be -1, 0 or +1")
        if not rating and not (correction or "").strip():
            raise ValueError("give a rating (+1 / -1) or a correction")
        return self.explog.log_feedback(reply_id, rating, correction)
