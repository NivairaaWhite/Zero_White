"""client.py - talk to a running daemon from the terminal or from code (standard library only).

    c = Client.discover(cfg, lay)          # None when no daemon is running
    for ev in c.chat("hello", session="s1"): print(ev.get("delta", ""), end="")
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterator, Optional

from ... import fileio
from ..config import Config
from ..hardware import pid_alive
from ..paths import Layout


class ClientError(RuntimeError):
    pass


class Client:
    def __init__(self, url: str, token: str, timeout: float = 600.0):
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout

    @classmethod
    def discover(cls, cfg: Config, lay: Layout) -> Optional["Client"]:
        """A client for the daemon of this workspace, or None when it is not running."""
        if not os.path.exists(lay.server_file):
            return None
        try:
            info = fileio.read_json(lay.server_file)
            if not pid_alive(int(info.get("pid", 0))):
                return None
            token = fileio.read_text(lay.token_file).strip() if os.path.exists(lay.token_file) else ""
            c = cls(info["url"], str(cfg.get("server.token", "") if cfg.get("server.token", "") not in ("", "auto") else token))
            c.get("/api/health", timeout=3)
            return c
        except (OSError, ValueError, KeyError, ClientError):
            return None

    # ---- transport ------------------------------------------------------------------------------------------------
    def _req(self, path: str, data: Optional[bytes] = None, ctype: str = "application/json", timeout: Optional[float] = None):
        req = urllib.request.Request(self.url + path, data=data, headers={"X-ZW-Token": self.token}, method="POST" if data is not None else "GET")
        if data is not None:
            req.add_header("Content-Type", ctype)
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode("utf-8")).get("error", "")
            except Exception:                                  # noqa: BLE001
                msg = ""
            raise ClientError(msg or f"HTTP {e.code} from {path}") from None
        except (urllib.error.URLError, OSError) as e:
            raise ClientError(f"cannot reach the server at {self.url}: {e}") from None

    def get(self, path: str, timeout: Optional[float] = None) -> dict:
        with self._req(path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def post(self, path: str, body: Optional[dict] = None, timeout: Optional[float] = None) -> dict:
        with self._req(path, json.dumps(body or {}).encode("utf-8"), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    # ---- API ------------------------------------------------------------------------------------------------------
    def status(self) -> dict:
        return self.get("/api/status")

    def _stream(self, path: str, body: dict) -> Iterator[dict]:
        with self._req(path, json.dumps(dict(body, stream=True)).encode("utf-8")) as r:
            buf = b""
            while True:
                chunk = r.read1(4096) if hasattr(r, "read1") else r.read(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n\n" in buf:
                    block, buf = buf.split(b"\n\n", 1)
                    if block.startswith(b"data:"):
                        yield json.loads(block[5:].decode("utf-8"))

    def chat(self, message: str, session: Optional[str] = None, **gen) -> Iterator[dict]:
        return self._stream("/api/chat", dict(gen, message=message, session=session))

    def regenerate(self, session: str, **gen) -> Iterator[dict]:
        return self._stream("/api/regenerate", dict(gen, session=session))

    def reply(self, message: str, session: Optional[str] = None, **gen) -> dict:
        last: dict = {}
        for ev in self.chat(message, session, **gen):
            last = ev
        return last

    def feedback(self, reply_id: str, rating: int = 0, correction: Optional[str] = None) -> dict:
        return self.post("/api/feedback", dict(reply_id=reply_id, rating=rating, correction=correction))

    def learn(self) -> dict:
        return self.post("/api/learn")

    def train(self) -> dict:
        return self.post("/api/train")

    def rollback(self, to: Optional[str] = None) -> dict:
        return self.post("/api/rollback", dict(to=to))

    def stt(self, wav: bytes) -> str:
        with self._req("/api/stt", wav, "audio/wav") as r:
            return json.loads(r.read().decode("utf-8"))["text"]

    def tts(self, text: str) -> bytes:
        with self._req("/api/tts", json.dumps(dict(text=text)).encode("utf-8")) as r:
            return r.read()
