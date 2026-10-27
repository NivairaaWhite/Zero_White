"""server.py - the local HTTP API and web chat, standard library only.

Security model (this is a service that edits a model on your machine, so it is strict by default):
  * binds to 127.0.0.1; any other bind address makes the token mandatory for every request
  * `Host` header must be a local name (defeats DNS-rebinding); `Origin` of a POST must be this server or in server.cors_origins
  * every state-changing request needs the access token (header X-ZW-Token or "Authorization: Bearer"); the web UI served from
    this same origin gets it embedded, other web pages cannot read it
  * request bodies are size-limited; responses are never cached
"""
from __future__ import annotations

import hmac
import json
import os
import socket
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Optional

from ... import fileio
from ..config import Config
from ..generate import Engine
from ..online import OnlineLearner, rollback
from ..paths import Layout
from ..session import ChatService

LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}
UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def load_or_create_token(cfg: Config, lay: Layout) -> str:
    t = cfg.get("server.token", None)
    if t and t != "auto":
        return str(t)
    if os.path.exists(lay.token_file):
        tok = fileio.read_text(lay.token_file).strip()
        if tok:
            return tok
    import secrets
    tok = secrets.token_urlsafe(24)
    os.makedirs(os.path.dirname(lay.token_file), exist_ok=True)
    fileio.write_text(lay.token_file, tok + "\n")
    try:
        os.chmod(lay.token_file, 0o600)
    except OSError:
        pass
    return tok


class App:
    """Everything the request handlers share."""

    def __init__(self, cfg: Config, lay: Layout, engine: Engine, chat: ChatService, learner: Optional[OnlineLearner],
                 hooks: Optional[dict] = None):
        self.cfg, self.lay, self.engine, self.chat, self.learner = cfg, lay, engine, chat, learner
        self.hooks = hooks or {}                                  # request_train(), status() supplied by the daemon
        self.token = load_or_create_token(cfg, lay)
        self.host = str(cfg.get("server.host"))
        self.local_only = self.host in LOOPBACK
        self.allowed_hosts = set(LOOPBACK) | {self.host}
        self.started = time.time()
        self._voice = None
        self._voice_err: Optional[str] = None
        self._vlock = threading.Lock()

    # ---- voice (optional) -----------------------------------------------------------------------------------------
    def voice(self):
        with self._vlock:
            if self._voice is None and self._voice_err is None:
                try:
                    from ...voice import VoiceService
                    self._voice = VoiceService(self.cfg, self.lay)
                except Exception as e:                     # noqa: BLE001
                    self._voice_err = f"{type(e).__name__}: {e}"
            if self._voice is None:
                raise ApiError(501, f"voice is not available: {self._voice_err}")
            return self._voice

    # ---- status ---------------------------------------------------------------------------------------------------
    def status(self) -> dict:
        from ..corpus.store import Ingestor
        from ..online import ExperienceLog
        eng, lay = self.engine, self.lay
        cur = lay.current() or {}
        model = dict(ready=eng.ready, version=eng.version, params=eng.model.num_params() if eng.ready else None,
                     ctx=eng.model.cfg.ctx if eng.ready else None, step=cur.get("step"), val_nll=cur.get("val_nll"),
                     trained=cur.get("trained"), online_updates=cur.get("online_updates", 0), mode=cur.get("mode"))
        training = None
        try:
            training = fileio.read_json(lay.train_status_path)
        except (OSError, ValueError):
            pass
        try:
            data = Ingestor(self.cfg, lay).stats()
            data = dict(files=data["files"], records=data["records"], kinds=data["kinds"], errors=len(data["errors"]))
        except Exception:                                  # noqa: BLE001
            data = dict(files=0, records=0, kinds={}, errors=0)
        online = dict(ExperienceLog(lay).stats(), enabled=bool(self.cfg.get("online.enabled")))
        if self.learner:
            online.update(updates=self.learner.state.get("updates", 0), last_update=self.learner.state.get("last_update"),
                          last_result=self.learner.state.get("last_result"))
        out = dict(name=self.cfg.get("persona.name") or "Zero White", greeting=self.cfg.get("persona.greeting"), model=model,
                   training=training, data=data, online=online, data_folder=lay.data, uptime_s=time.time() - self.started,
                   require_token=not self.local_only)
        if "status" in self.hooks:
            out["daemon"] = self.hooks["status"]()
        return out


# ---------------------------------------------------------------------------------------------------------------------
def make_handler(app: App):
    max_body = int(float(app.cfg.get("server.max_body_mb")) * 1024 * 1024)
    cors = set(app.cfg.get("server.cors_origins") or [])

    class Handler(BaseHTTPRequestHandler):
        server_version = "ZeroWhite"
        sys_version = ""
        protocol_version = "HTTP/1.0"                       # one request per connection: simple, and streaming needs no chunking

        def log_message(self, fmt, *args):                  # quiet: the daemon logs what matters
            pass

        # ---- plumbing ---------------------------------------------------------------------------------------------
        def _host_ok(self) -> bool:
            h = (self.headers.get("Host") or "").strip().lower()
            name = h.rsplit(":", 1)[0] if not h.startswith("[") else h.split("]")[0] + "]"
            return name in app.allowed_hosts

        def _origin_ok(self) -> bool:
            o = self.headers.get("Origin")
            if not o:
                return True
            if o in cors:
                return True
            host = urllib.parse.urlparse(o).hostname or ""
            return host in {x.strip("[]") for x in app.allowed_hosts}

        def _token_ok(self) -> bool:
            t = self.headers.get("X-ZW-Token") or ""
            a = self.headers.get("Authorization") or ""
            if a.lower().startswith("bearer "):
                t = a[7:].strip()
            return bool(t) and hmac.compare_digest(t.encode(), app.token.encode())

        def _send(self, status: int, body: bytes, ctype: str, extra: Optional[Dict[str, str]] = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self._security_headers()
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _security_headers(self):
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                                                       "media-src 'self' blob:; connect-src 'self'; img-src 'self' data:")
            o = self.headers.get("Origin")
            if o and o in cors:
                self.send_header("Access-Control-Allow-Origin", o)
                self.send_header("Vary", "Origin")

        def _json(self, obj, status: int = 200):
            self._send(status, json.dumps(obj, ensure_ascii=False, default=fileio._json_default).encode("utf-8"), "application/json; charset=utf-8")

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            if n > max_body:
                raise ApiError(413, f"request body too large ({n} bytes; limit server.max_body_mb)")
            return self.rfile.read(n) if n else b""

        def _json_body(self) -> dict:
            raw = self._body()
            if not raw:
                return {}
            try:
                d = json.loads(raw.decode("utf-8"))
            except ValueError:
                raise ApiError(400, "the request body is not valid JSON") from None
            if not isinstance(d, dict):
                raise ApiError(400, "the JSON body must be an object")
            return d

        def _gate(self, post: bool):
            if not self._host_ok():
                raise ApiError(403, "unexpected Host header (this server only answers to local names)")
            if post and not self._origin_ok():
                raise ApiError(403, "cross-origin request refused")
            if (post or not app.local_only) and not self._token_ok():
                raise ApiError(401, "missing or wrong access token (X-ZW-Token header). The token is in state/token.")

        # ---- verbs ------------------------------------------------------------------------------------------------
        def do_OPTIONS(self):
            self.send_response(204)
            self._security_headers()
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-ZW-Token, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            self._dispatch(False)

        def do_POST(self):
            self._dispatch(True)

        def _dispatch(self, post: bool):
            url = urllib.parse.urlparse(self.path)
            path, q = url.path, dict(urllib.parse.parse_qsl(url.query))
            try:
                if not self._host_ok():
                    raise ApiError(403, "unexpected Host header")
                if path == "/" and not post:
                    return self._ui()
                if path == "/favicon.ico":
                    return self._send(204, b"", "image/x-icon")
                self._gate(post)
                fn = (ROUTES_POST if post else ROUTES_GET).get(path)
                if fn is None:
                    raise ApiError(404, f"no such endpoint: {path}")
                fn(self, q)
            except ApiError as e:
                self._error(e.status, e.message)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except ValueError as e:
                self._error(400, str(e))
            except Exception as e:                                    # noqa: BLE001
                app.hooks.get("log", lambda *_: None)("error", f"{path}: {traceback.format_exc()}")
                self._error(500, f"{type(e).__name__}: {e}")

        def _error(self, status: int, msg: str):
            try:
                self._json(dict(error=msg), status)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def _ui(self):
            try:
                html = fileio.read_text(UI_PATH)
            except OSError:
                raise ApiError(500, "the web UI file is missing from this installation")
            embed = app.token if app.local_only else ""
            html = html.replace("__ZW_TOKEN__", embed).replace("__ZW_REQUIRE_TOKEN__", "0" if app.local_only else "1")
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

        # ---- helpers for routes -----------------------------------------------------------------------------------
        def stream_events(self, events):
            """Server-sent events over a plain response; ends when the generator ends or the client leaves."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self._security_headers()
            self.end_headers()
            try:
                for ev in events:
                    self.wfile.write(b"data: " + json.dumps(ev, ensure_ascii=False).encode("utf-8") + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                events.close()

    return Handler


# ---------------------------------------------------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------------------------------------------------
ROUTES_GET: Dict[str, Callable] = {}
ROUTES_POST: Dict[str, Callable] = {}
_APP: Optional[App] = None


def route(method: str, path: str):
    def deco(fn):
        (ROUTES_POST if method == "POST" else ROUTES_GET)[path] = fn
        return fn
    return deco


def _gen_opts(d: dict) -> dict:
    out = {}
    for k in ("temperature", "top_k", "top_p", "max_new_tokens", "repetition_penalty", "seed"):
        if d.get(k) is not None:
            out[k] = d[k]
    return out


def _need_model():
    if not _APP.engine.ready:
        raise ApiError(409, "there is no trained model yet. Put dataset files in the data folder (training starts by itself), "
                            "or run `zerowhite chat train`.")


@route("GET", "/api/health")
def r_health(h, q):
    h._json(dict(ok=True))


@route("GET", "/api/status")
def r_status(h, q):
    h._json(_APP.status())


@route("GET", "/api/session")
def r_session_get(h, q):
    sid = q.get("id")
    if not sid:
        raise ApiError(400, "pass ?id=SESSION")
    h._json(dict(session=sid, turns=_APP.chat.history(sid)))


@route("POST", "/api/session")
def r_session_new(h, q):
    d = h._json_body()
    s = _APP.chat.get(None, system=d.get("system"))
    h._json(dict(session=s.id, greeting=_APP.cfg.get("persona.greeting")))


@route("POST", "/api/chat")
def r_chat(h, q):
    d = h._json_body()
    _need_model()
    msg = str(d.get("message") or "").strip()
    if not msg:
        raise ApiError(400, "message is empty")
    events = _APP.chat.send(d.get("session"), msg, **_gen_opts(d))
    if d.get("stream"):
        return h.stream_events(events)
    h._json(list(events)[-1])


@route("POST", "/api/regenerate")
def r_regen(h, q):
    d = h._json_body()
    _need_model()
    events = _APP.chat.regenerate(str(d.get("session") or ""), **_gen_opts(d))
    if d.get("stream"):
        return h.stream_events(events)
    h._json(list(events)[-1])


@route("POST", "/api/feedback")
def r_feedback(h, q):
    d = h._json_body()
    ev = _APP.chat.feedback(str(d.get("reply_id") or ""), int(d.get("rating") or 0), d.get("correction"))
    if _APP.learner:
        _APP.learner.poke.set()
    h._json(dict(ok=True, saved=ev))


@route("POST", "/api/learn")
def r_learn(h, q):
    if not _APP.learner:
        raise ApiError(409, "online learning is not running in this process")
    h._json(_APP.learner.update(force=True))


@route("POST", "/api/train")
def r_train(h, q):
    f = _APP.hooks.get("request_train")
    if not f:
        raise ApiError(409, "this server was started without the training scheduler; use `zerowhite chat train`")
    h._json(f())


@route("POST", "/api/rollback")
def r_rollback(h, q):
    d = h._json_body()
    now = rollback(_APP.lay, d.get("to"))
    if not now:
        raise ApiError(409, "there is no earlier version to go back to")
    _APP.engine.load()
    h._json(dict(ok=True, version=now))


@route("GET", "/api/versions")
def r_versions(h, q):
    cur = (_APP.lay.current() or {}).get("file")
    h._json(dict(current=cur, versions=[dict(file=os.path.basename(p), kind=k, modified=m, live=os.path.basename(p) == cur)
                                         for p, k, m in _APP.lay.versions()]))


@route("POST", "/api/reload")
def r_reload(h, q):
    h._json(dict(reloaded=_APP.engine.reload_if_changed(), version=_APP.engine.version))


# ---- voice ------------------------------------------------------------------------------------------------------------
@route("GET", "/api/voice")
def r_voice_info(h, q):
    try:
        h._json(_APP.voice().describe())
    except ApiError as e:
        h._json(dict(available=False, reason=e.message))


@route("POST", "/api/stt")
def r_stt(h, q):
    audio = h._body()
    if not audio:
        raise ApiError(400, "send the recording as the request body (audio/wav)")
    h._json(dict(text=_APP.voice().transcribe(audio)))


@route("POST", "/api/tts")
def r_tts(h, q):
    d = h._json_body()
    text = str(d.get("text") or "").strip()
    if not text:
        raise ApiError(400, "text is empty")
    wav = _APP.voice().synthesize(text)
    h._send(200, wav, "audio/wav")


@route("POST", "/api/voice")
def r_voice_chat(h, q):
    """audio in -> transcript -> reply (JSON). Fetch the spoken reply with /api/tts."""
    _need_model()
    audio = h._body()
    if not audio:
        raise ApiError(400, "send the recording as the request body (audio/wav)")
    heard = _APP.voice().transcribe(audio).strip()
    if not heard:
        return h._json(dict(heard="", reply=None))
    last = list(_APP.chat.send(q.get("session"), heard))[-1]
    h._json(dict(heard=heard, reply=last["text"], reply_id=last["reply_id"], session=last["session"]))


# ---------------------------------------------------------------------------------------------------------------------
class Server:
    """ThreadingHTTPServer wrapper that finds a free port and records its address in run/server.json."""

    def __init__(self, app: App):
        global _APP
        _APP = app
        self.app = app
        host, port = app.host, int(app.cfg.get("server.port"))
        handler = make_handler(app)
        last: Optional[Exception] = None
        for p in range(port, port + (50 if app.cfg.get("server.port_fallback") else 1)):
            try:
                ThreadingHTTPServer.daemon_threads = True
                ThreadingHTTPServer.allow_reuse_address = os.name != "nt"
                if ":" in host:
                    ThreadingHTTPServer.address_family = socket.AF_INET6
                self.httpd = ThreadingHTTPServer((host, p), handler)
                self.port = int(self.httpd.server_address[1])      # differs from p when p == 0 (the OS picks a free port)
                break
            except OSError as e:
                last = e
        else:
            raise OSError(f"cannot bind {host}:{port}: {last}")
        self.thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        host = "localhost" if self.app.host in ("0.0.0.0", "::") else self.app.host
        return f"http://{host}:{self.port}/"

    def start(self):
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs=dict(poll_interval=0.25), daemon=True, name="http")
        self.thread.start()
        fileio.write_json(self.app.lay.server_file, dict(host=self.app.host, port=self.port, url=self.url, pid=os.getpid(),
                                                         started=time.time()), indent=2)

    def stop(self):
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        finally:
            try:
                os.remove(self.app.lay.server_file)
            except OSError:
                pass
