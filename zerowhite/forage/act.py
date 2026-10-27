"""forage/act.py - the ONLY code in Zero White that sends data to a web server (POST / PUT / PATCH), under a policy that is enforced in code and that nothing the learner does can widen.

`forage/fetch.py` reads pages. This module lets a program that is LEARNING (an evolved policy, an actor-critic) use a web API by trial and error: log in, take a token, send JSON, follow a link,
wait out a rate limit. That is the riskiest capability in the project, so the design is "deny by default, and make the denial the code, not the documentation":

  * `ActPolicy` is written by a PERSON (CLI flags / code). It is a frozen dataclass; the learner never receives it, and the actor has no method that changes it.
  * `write_hosts` is empty by default: with no host named, EVERY request is refused (and ledgered). A host named without `allow_remote` must be loopback (localhost / 127.0.0.1 / ::1) AND
    must really resolve to loopback. Anything else needs `allow_remote=True`, https, the port the entry names (443 unless written `host:port`), a public address, and a file ledger.
  * Methods: GET / POST / PUT / PATCH. DELETE only with `allow_delete=True`. CONNECT / TRACE never. Redirects are followed by hand for GET only, stay on the same origin, and are never replayed
    for a write (a 3xx to a POST is returned, not obeyed).
  * What the learner controls is DATA, validated here: a path (relative, no scheme / authority / backslash / control character / `..` even percent-encoded, strict character set), a query, a JSON body
    (depth and size capped), and headers from a short allow-list whose values cannot contain CR/LF. It can NOT choose the host, the scheme, the port, `Cookie`, `Host`, `Content-Length`...
    Values the server sent earlier (tokens, ids, `next` links) are used only through `Ref("slot", name)` and are re-validated (header-safe charset, path-safe charset) at the point of use.
  * Secrets are given by the person (`secrets={"password": ...}`) and referenced by NAME with `Ref("secret", name)`; they are substituted here, never returned, never logged: the ledger and the
    in-memory tail show `***` for them and for any JSON field whose name looks secret (token, password, auth, cookie, csrf, nonce, ...) and for any response text that echoes a secret value.
    A secret is never placed in a URL (path or query).
  * Budgets: `max_requests` per Actor (a run), `max_requests_per_episode` (the caller passes the episode's count), a per-host minimum delay for non-loopback hosts, request / response size caps,
    a kill-switch file checked before every request, `dry_run` (everything is checked and ledgered, nothing is sent), and no proxy unless `use_env_proxy=True` (a proxy would see the writes).
  * Every attempt, refusals included, is one line of the ledger (`ledger.jsonl`) with the request method / path / redacted body / status / sizes / SHA-256 of the response.

HONEST STATUS. Verified here, against a real local HTTP server (`mockapi.py`) and fakes: the policy decisions above, the injection attempts listed in the tests (CRLF in header values and names, `@` and userinfo,
absolute URLs and `//host` paths, `..` and `%2e%2e` and double-encoded variants, other schemes, secrets in URLs, body / header size), cookie and slot handling, redirect handling, ledger redaction, budgets, the kill
switch, dry run. NOT verified: any real remote service, TLS behaviour, a hostile DNS server (the address check and the connection are separate steps: DNS rebinding can still defeat the loopback / public-address
test; the host allow-list is the real control), HTTP/2, large uploads, proxies, Windows / macOS (the CI legs run the same tests). The actor is not thread-safe. Like `fetch.py`, this file is part of the sealed
constitution: the self-modifier cannot edit the policy it is held to; and it is NEVER called by the organism or by self-modification (CLI only).
"""
from __future__ import annotations

import collections
import hashlib
import ipaddress
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

USER_AGENT = "zero-white-actor/6.5 (+explicit write allow-list; loopback by default)"
LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})
ALWAYS_REFUSED = frozenset({"CONNECT", "TRACE"})
BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})
_SECRETISH = re.compile(r"pass|secret|token|auth|key|cookie|session|credential|bearer|csrf|nonce|otp|signature", re.I)
_HEADER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9-]{0,39}$")
_BAD_VALUE = re.compile(r"[\x00-\x1f\x7f]")
_PATH_OK = re.compile(r"^/[A-Za-z0-9._~!$&'()*+,;=:@%/\-]*$")
_QUERY_OK = re.compile(r"^[A-Za-z0-9._~!$&'()*+,;=:@%/?\-]*$")
_FMT_OK = re.compile(r"^[^{}]*\{\}[^{}]*$")
_FORBIDDEN_HEADERS = frozenset(h.lower() for h in ("Host", "Cookie", "Content-Length", "Content-Type", "Transfer-Encoding", "Connection", "Upgrade", "Proxy-Authorization",
                                                  "Proxy-Connection", "TE", "Trailer", "Expect", "User-Agent", "Set-Cookie"))


# ----------------------------------------------------------------------------------------------------------------------------- references
@dataclass(frozen=True)
class Ref:
    """A named placeholder inside a request template. kind='slot': a value some earlier response contained (the Session fills it in). kind='secret': a value the PERSON supplied (the Actor fills it in; nobody else can read it).
    `fmt` wraps the value (e.g. 'Bearer {}' for an Authorization header)."""
    kind: str
    name: str
    fmt: str = "{}"

    def __post_init__(self):
        if self.kind not in ("slot", "secret") or not isinstance(self.name, str) or not self.name or len(self.name) > 64:
            raise ValueError("a Ref is a slot or a secret with a short name")
        if not isinstance(self.fmt, str) or len(self.fmt) > 64 or not _FMT_OK.match(self.fmt):
            raise ValueError("a Ref format is plain text with exactly one '{}' (no other braces, so no attribute or index tricks)")

    def to_json(self) -> dict:
        d = {"$" + self.kind: self.name}
        if self.fmt != "{}":
            d["$fmt"] = self.fmt
        return d


def ref_from_json(obj):
    """{'$slot': 'token', '$fmt': 'Bearer {}'} -> Ref; anything else is returned unchanged (recursively inside dicts and lists)."""
    if isinstance(obj, dict):
        for kind in ("slot", "secret"):
            if "$" + kind in obj and isinstance(obj["$" + kind], str):
                return Ref(kind, obj["$" + kind], str(obj.get("$fmt", "{}")))
        return {k: ref_from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [ref_from_json(v) for v in obj]
    return obj


def ref_to_json(obj):
    if isinstance(obj, Ref):
        return obj.to_json()
    if isinstance(obj, dict):
        return {k: ref_to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ref_to_json(v) for v in obj]
    return obj


# ----------------------------------------------------------------------------------------------------------------------------- responses
@dataclass
class Response:
    status: int = 0                    # 0 = nothing came back (refused, dry run, transport error)
    headers: dict = field(default_factory=dict)       # lower-case names; repeated headers joined with ', ' (Set-Cookie is in set_cookies)
    set_cookies: list = field(default_factory=list)   # [(name, value, deleted)]
    json: object = None                # the parsed body when it is JSON and within bounds, else None
    text: str = ""                     # the body as text, at most 4000 characters
    elapsed: float = 0.0
    nbytes: int = 0
    sent: bool = False                 # False for a refused request or a dry run
    refused: bool = False
    reason: str = ""                   # why it was refused / what went wrong; '' when all is well
    truncated: bool = False
    redirected: int = 0

    @property
    def ok(self) -> bool:
        return self.sent and 200 <= self.status < 300


def parse_json_bounded(text: str, max_depth: int = 16, max_nodes: int = 5000):
    """The JSON value of `text`, or None when it is not JSON or is deeper / larger than the bounds (never raises)."""
    try:
        val = json.loads(text, parse_constant=lambda c: None)
    except (ValueError, RecursionError):
        return None
    nodes, stack = 0, [(val, 1)]
    while stack:
        v, d = stack.pop()
        nodes += 1
        if nodes > max_nodes or d > max_depth:
            return None
        if isinstance(v, dict):
            stack.extend((x, d + 1) for x in v.values())
        elif isinstance(v, list):
            stack.extend((x, d + 1) for x in v)
    return val


def _strip_refs(obj, _depth: int = 0):
    """The same structure with every Ref replaced by a plain string, so the shape checks see only data."""
    if isinstance(obj, Ref):
        return "x"
    if _depth > 64:
        return obj
    if isinstance(obj, dict):
        return {k: _strip_refs(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strip_refs(v, _depth + 1) for v in obj]
    return obj


def _depth_ok(obj, max_depth: int, max_nodes: int = 5000) -> bool:
    nodes, stack = 0, [(obj, 1)]
    while stack:
        v, d = stack.pop()
        nodes += 1
        if nodes > max_nodes or d > max_depth:
            return False
        if isinstance(v, dict):
            if any(not isinstance(k, str) for k in v):
                return False
            stack.extend((x, d + 1) for x in v.values())
        elif isinstance(v, (list, tuple)):
            stack.extend((x, d + 1) for x in v)
        elif isinstance(v, float):
            if v != v or v in (float("inf"), float("-inf")):                  # NaN / Infinity are not JSON
                return False
        elif not (v is None or isinstance(v, (str, int, bool))):
            return False
    return True


def redact(obj, secret_values=(), _key: str = "", _depth: int = 0):
    """A copy of a JSON-like value that is safe to write to a ledger: values under secret-looking keys, and any string containing a known secret value, become '***'; long strings are cut."""
    if _depth > 12:
        return "..."
    if isinstance(obj, dict):
        return {str(k): redact(v, secret_values, str(k), _depth + 1) for k, v in list(obj.items())[:40]}
    if isinstance(obj, (list, tuple)):
        return [redact(v, secret_values, _key, _depth + 1) for v in list(obj)[:20]]
    if _key and _SECRETISH.search(_key) and obj is not None:
        return "***"
    if isinstance(obj, str):
        if any(s and s in obj for s in secret_values):
            return "***"
        return obj if len(obj) <= 80 else obj[:77] + "..."
    return obj


def scrub(obj, secret_values=(), _depth: int = 0):
    """A copy of a parsed response with every occurrence of a known secret value replaced by '***' (a server that echoes what it was sent must not hand the secret back to the learner's session)."""
    if _depth > 40:
        return obj
    if isinstance(obj, dict):
        return {k: scrub(v, secret_values, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v, secret_values, _depth + 1) for v in obj]
    if isinstance(obj, str):
        for sv in secret_values:
            if sv and sv in obj:
                obj = obj.replace(sv, "***")
    return obj


# ----------------------------------------------------------------------------------------------------------------------------- policy
@dataclass(frozen=True)
class ActPolicy:
    """What a PERSON allows. Every field is a limit, not a default behaviour: with no `write_hosts` nothing is sent."""
    write_hosts: tuple = ()                    # 'host' or 'host:port'; empty = every request is refused
    allow_remote: bool = False                 # hosts other than localhost / 127.0.0.1 / ::1 (then https, a public address and a file ledger are required)
    allow_delete: bool = False
    methods: tuple = ("GET", "POST", "PUT", "PATCH")
    allowed_headers: tuple = ("Accept", "Authorization", "X-CSRF-Token", "X-Api-Key", "Idempotency-Key")
    max_requests: int = 500                    # per Actor (one run)
    max_requests_per_episode: int = 50         # the caller passes how many the current episode has used
    max_body_bytes: int = 16 * 1024
    max_response_bytes: int = 256 * 1024
    max_json_depth: int = 16
    max_path_chars: int = 300
    max_header_chars: int = 512
    delay: float = 1.0                         # minimum seconds between requests to one NON-loopback host
    timeout: float = 10.0
    redirects: int = 2                         # GET only, same origin only
    kill_file: str = ""                        # if this file exists no request is sent
    use_env_proxy: bool = False                # a proxy would see every write: off unless the person says otherwise

    def hosts(self) -> set:
        return {h.strip().lower() for h in self.write_hosts if h and h.strip()}


def _addr_is_local(ip: str) -> bool:
    a = ipaddress.ip_address(ip.split("%")[0])
    return a.is_private or a.is_loopback or a.is_link_local or a.is_multicast or a.is_reserved or a.is_unspecified


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):                       # every hop is checked by hand
        return None


_OPENERS: dict = {}


def _opener(use_env_proxy: bool):
    """One opener per proxy setting for the whole process: building one loads the system certificate store (about 20 ms), which would otherwise be paid on every request."""
    op = _OPENERS.get(bool(use_env_proxy))
    if op is None:
        handlers = [_NoRedirect()] + ([] if use_env_proxy else [urllib.request.ProxyHandler({})])
        op = _OPENERS[bool(use_env_proxy)] = urllib.request.build_opener(*handlers)
    return op


def _default_open(method: str, url: str, headers: dict, body, timeout: float, max_bytes: int, use_env_proxy: bool = False):
    """(status, headers, body, location). Reads at most max_bytes + 1 so 'too big' can be told from 'exactly the cap'."""
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    op = _opener(use_env_proxy)
    try:
        with op.open(req, timeout=timeout) as r:
            return r.status, r.headers, r.read(max_bytes + 1), None
    except urllib.error.HTTPError as e:
        data = b""
        try:
            data = e.read(max_bytes + 1)
        except Exception:                                      # noqa: BLE001
            pass
        return e.code, e.headers, data, e.headers.get("Location")


def _header_items(headers):
    """[(lower-case name, value)] from an email.Message, a dict or a list of pairs."""
    if headers is None:
        return []
    if hasattr(headers, "items") and not isinstance(headers, dict):
        return [(k.lower(), v) for k, v in headers.items()]
    if isinstance(headers, dict):
        return [(str(k).lower(), v) for k, v in headers.items()]
    return [(str(k).lower(), v) for k, v in headers]


def _parse_set_cookie(value: str):
    parts = [p.strip() for p in value.split(";")]
    if not parts or "=" not in parts[0]:
        return None
    name, val = parts[0].split("=", 1)
    name, val = name.strip(), val.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,64}", name) or _BAD_VALUE.search(val) or len(val) > 512 or any(c in val for c in ";, "):
        return None
    deleted = any(p.lower() in ("max-age=0",) or p.lower().startswith("max-age=-") for p in parts[1:]) or val == ""
    return name, val, deleted


# ----------------------------------------------------------------------------------------------------------------------------- the actor
class Actor:
    """Sends requests to ONE origin under a policy. `request()` never raises for a refused or failed request: it returns a Response with `refused` / `reason` set and writes a ledger line."""

    def __init__(self, base_url: str, policy: ActPolicy, ledger_path: str | None = None, secrets: dict | None = None, dry_run: bool = False,
                 opener=None, resolver=None, sleep=time.sleep, clock=time.monotonic):
        u = urllib.parse.urlsplit(base_url)
        if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password or u.query or u.fragment:
            raise ValueError("base_url must look like http(s)://host[:port][/prefix] with no credentials, query or fragment")
        self.scheme, self.host = u.scheme, u.hostname.lower()
        self.port = u.port or (443 if u.scheme == "https" else 80)
        self.prefix = u.path.rstrip("/")
        if self.prefix and not _PATH_OK.match(self.prefix):
            raise ValueError("base_url path contains characters that are not allowed")
        self.base_url = f"{self.scheme}://{self._netloc()}{self.prefix}"
        self.policy, self.ledger_path, self.dry_run = policy, ledger_path, bool(dry_run)
        self._secrets = {str(k): str(v) for k, v in (secrets or {}).items()}
        self.opener, self.resolver, self.sleep, self.clock = opener, resolver, sleep, clock
        self.n_requests = self.n_refused = self.n_errors = 0
        self.tail = collections.deque(maxlen=200)               # the last ledger records, in memory (also when there is no file)
        self._last = {}
        self._addr_cache = {}

    # ---- helpers -------------------------------------------------------------------------------------------------
    def _netloc(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        default = 443 if self.scheme == "https" else 80
        return host if self.port == default else f"{host}:{self.port}"

    def secret_names(self) -> list:
        """The NAMES of the secrets the person supplied (the values are never exposed)."""
        return sorted(self._secrets)

    def stats(self) -> dict:
        return dict(requests=self.n_requests, refused=self.n_refused, errors=self.n_errors, remaining=max(0, self.policy.max_requests - self.n_requests))

    def _log(self, rec: dict):
        rec = dict(rec, t=time.time())
        self.tail.append(rec)
        if self.ledger_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.ledger_path)), exist_ok=True)
                with open(self.ledger_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
            except OSError:
                pass

    def _secret_values(self):
        return [v for v in self._secrets.values() if len(v) >= 3]

    # ---- the origin policy -----------------------------------------------------------------------------------------
    def _addresses(self, host: str, port: int):
        if host in self._addr_cache:
            return self._addr_cache[host]
        try:
            ipaddress.ip_address(host)
            ips = [host]
        except ValueError:
            ips = self.resolver(host) if self.resolver else [ai[4][0] for ai in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]
        self._addr_cache[host] = list(ips)
        return self._addr_cache[host]

    def _check_origin(self):
        """(ok, reason). Pure policy; resolves the host's address (once per Actor) to check it is what the entry claims."""
        p, host, port = self.policy, self.host, self.port
        hosts = p.hosts()
        if not hosts:
            return False, "no write host is allowed: name at least one in the policy (`act request --allow-host HOST`; nothing is sent by default)"
        bracket = f"[{host}]" if ":" in host else host
        if host not in hosts and f"{bracket}:{port}" not in hosts and f"{host}:{port}" not in hosts:
            return False, f"host '{host}' is not on the write allow-list"
        try:
            ips = self._addresses(host, port)
        except (OSError, ValueError) as ex:
            return False, f"cannot resolve '{host}': {ex}"
        if not ips:
            return False, f"'{host}' has no address"
        if host in LOOPBACK_NAMES:
            if not all(ipaddress.ip_address(ip.split('%')[0]).is_loopback for ip in ips):
                return False, f"'{host}' does not resolve to a loopback address"
            return True, ""
        if not p.allow_remote:
            return False, f"'{host}' is not a loopback host: writing to a remote host needs allow_remote (--allow-remote)"
        if self.scheme != "https":
            return False, "a remote host must be reached over https"
        if port != 443 and f"{bracket}:{port}" not in hosts and f"{host}:{port}" not in hosts:
            return False, f"port {port} is refused for '{host}' unless the allow-list entry names it ('{host}:{port}')"
        if any(_addr_is_local(ip) for ip in ips):
            return False, f"'{host}' resolves to a private / loopback / link-local address"
        if not self.ledger_path:
            return False, "a remote host needs a file ledger (ledger_path) so every attempt is on record"
        return True, ""

    # ---- validation of what the learner chose ---------------------------------------------------------------------------
    def _clean_path(self, path, query):
        """(path_with_prefix_and_query, reason). `path` must be a relative absolute-path string; `query` a flat dict of scalars."""
        if not isinstance(path, str) or not path:
            return None, "the path must be a non-empty string"
        if len(path) > self.policy.max_path_chars:
            return None, "the path is too long"
        qs = ""
        if "?" in path:
            path, qs = path.split("?", 1)
            if "?" in qs or not _QUERY_OK.match(qs) or len(qs) > self.policy.max_path_chars:
                return None, "the query in the path has characters that are not allowed"
        if not path.startswith("/") or path.startswith("//") or "\\" in path or "#" in path:
            return None, "the path must be a plain absolute path on the base host (no scheme, host, '//', backslash or fragment)"
        if not _PATH_OK.match(path):
            return None, "the path has characters that are not allowed"
        decoded = path
        for _ in range(3):                                      # percent-encoded / double-encoded tricks: look at what the server would see after each decoding
            decoded = urllib.parse.unquote(decoded)
            if "\\" in decoded or "//" in decoded or _BAD_VALUE.search(decoded) or " " in decoded:
                return None, "the path decodes to something that is not allowed"
            if any(seg.split(";")[0] in (".", "..") for seg in decoded.split("/")):          # '..;x' is a dot segment to some servers
                return None, "dot segments ('..') are not allowed in a path"
        if query:
            if not isinstance(query, dict) or len(query) > 20:
                return None, "the query must be a small dict"
            try:
                pairs = []
                for k, v in query.items():
                    if isinstance(v, Ref):
                        return None, "a query value must be resolved before it reaches the actor"
                    if not isinstance(v, (str, int, float, bool)) or _BAD_VALUE.search(str(k) + str(v)):
                        return None, "query values must be plain scalars"
                    pairs.append((str(k), str(v)))
                extra = urllib.parse.urlencode(pairs, quote_via=urllib.parse.quote)
            except (TypeError, ValueError):
                return None, "the query could not be encoded"
            qs = (qs + "&" if qs else "") + extra
        full = self.prefix + path + ("?" + qs if qs else "")
        return full, ""

    def _materialize(self, obj, depth: int = 0):
        """(value, redacted_view) with Ref('secret') replaced; Ref('slot') must already be gone."""
        if isinstance(obj, Ref):
            if obj.kind == "slot":
                raise ValueError(f"slot '{obj.name}' was not resolved before sending")
            if obj.name not in self._secrets:
                raise ValueError(f"unknown secret '{obj.name}'")
            return obj.fmt.format(self._secrets[obj.name]), "***"
        if isinstance(obj, dict):
            out, red = {}, {}
            for k, v in obj.items():
                out[k], red[k] = self._materialize(v, depth + 1)
            return out, red
        if isinstance(obj, (list, tuple)):
            pairs = [self._materialize(v, depth + 1) for v in obj]
            return [a for a, _ in pairs], [b for _, b in pairs]
        return obj, obj

    def _clean_headers(self, headers):
        """({name: value}, redacted_view, reason)."""
        out, red = {}, {}
        allowed = {h.lower() for h in self.policy.allowed_headers} - _FORBIDDEN_HEADERS
        for name, val in (headers or {}).items():
            if not isinstance(name, str) or not _HEADER_NAME.match(name):
                return None, None, "a header name is not valid"
            if name.lower() not in allowed:
                return None, None, f"header '{name}' is not on the allow-list"
            try:
                if isinstance(val, Ref):
                    v, _ = self._materialize(val)
                else:
                    v = val
            except ValueError as ex:
                return None, None, str(ex)
            if not isinstance(v, str) or _BAD_VALUE.search(v) or len(v) > self.policy.max_header_chars:
                return None, None, f"the value of header '{name}' is not acceptable (control characters or too long)"
            out[name] = v
            red[name] = "***" if (isinstance(val, Ref) and val.kind == "secret") or _SECRETISH.search(name) else v
        return out, red, ""

    def _pace(self):
        if self.host in LOOPBACK_NAMES:
            return
        last = self._last.get(self.host)
        if last is not None:
            wait = self.policy.delay - (self.clock() - last)
            if wait > 0:
                self.sleep(wait)
        self._last[self.host] = self.clock()

    # ---- one request ----------------------------------------------------------------------------------------------------
    def request(self, method: str, path, json_body=None, headers=None, query=None, cookie: dict | None = None, episode_used: int = 0) -> Response:
        t0 = time.time()
        method = str(method).upper()
        rec = dict(method=method, host=self.host, path=str(path)[:200], verdict="refused", reason="", status=None, req_bytes=0, resp_bytes=0, sha256=None, elapsed=0.0)

        def refuse(why: str) -> Response:
            self.n_refused += 1
            rec.update(verdict="refused", reason=why, elapsed=round(time.time() - t0, 4))
            self._log(rec)
            return Response(refused=True, reason=why)

        p = self.policy
        if p.kill_file and os.path.exists(p.kill_file):
            return refuse("the kill-switch file exists")
        if self.n_requests >= p.max_requests:
            return refuse(f"the run's budget of {p.max_requests} requests is used up")
        if episode_used >= p.max_requests_per_episode:
            return refuse(f"the episode's budget of {p.max_requests_per_episode} requests is used up")
        if method in ALWAYS_REFUSED or method not in {m.upper() for m in p.methods} or (method == "DELETE" and not p.allow_delete):
            return refuse(f"method {method} is not allowed")
        ok, why = self._check_origin()
        if not ok:
            return refuse(why)
        full, why = self._clean_path(path, query)
        if full is None:
            return refuse(why)
        rec["path"] = full.split("?")[0][:200]
        url = f"{self.scheme}://{self._netloc()}{full}"
        u = urllib.parse.urlsplit(url)                          # defence in depth: whatever was built must still point at the origin we checked
        if (u.scheme, (u.hostname or "").lower(), u.port or (443 if u.scheme == "https" else 80)) != (self.scheme, self.host, self.port) or u.username or u.password:
            return refuse("the built URL does not point at the base origin")
        hdrs, red_headers, why = self._clean_headers(headers)
        if hdrs is None:
            return refuse(why)
        body, red_body = None, None
        if json_body is not None:
            if method not in BODY_METHODS:
                return refuse(f"a {method} request cannot carry a body")
            if not _depth_ok(_strip_refs(json_body), p.max_json_depth):
                return refuse("the body is not plain JSON data, or is too deep or too large")
            try:
                value, red_body = self._materialize(json_body)
                body = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("ascii")
            except (ValueError, TypeError) as ex:
                return refuse(f"the body could not be built: {ex}")
            if len(body) > p.max_body_bytes:
                return refuse(f"the body is larger than {p.max_body_bytes} bytes")
            hdrs["Content-Type"] = "application/json"
        hdrs.setdefault("Accept", "application/json")
        hdrs["User-Agent"] = USER_AGENT
        if cookie:
            items = [(k, v) for k, v in cookie.items() if re.fullmatch(r"[A-Za-z0-9_.\-]{1,64}", str(k)) and not _BAD_VALUE.search(str(v)) and not any(c in str(v) for c in ";, ")]
            if items:
                hdrs["Cookie"] = "; ".join(f"{k}={v}" for k, v in items)
        rec.update(req_bytes=len(body or b""), headers=red_headers, body=redact(red_body, self._secret_values()) if red_body is not None else None,
                   query=sorted((query or {}).keys()) if isinstance(query, dict) else None)
        self.n_requests += 1
        if self.dry_run:
            rec.update(verdict="dry_run", reason="dry run: checked, nothing sent", elapsed=round(time.time() - t0, 4))
            self._log(rec)
            return Response(reason="dry run: checked, nothing sent")
        return self._send(method, url, hdrs, body, rec, t0)

    def _send(self, method, url, hdrs, body, rec, t0) -> Response:
        p = self.policy
        hops = 0
        while True:
            self._pace()
            opener = self.opener or (lambda m, u, h, b, to, mx: _default_open(m, u, h, b, to, mx, p.use_env_proxy))
            try:
                status, rheaders, data, loc = opener(method, url, hdrs, body, p.timeout, p.max_response_bytes)
            except (urllib.error.URLError, OSError, ValueError) as ex:
                self.n_errors += 1
                rec.update(verdict="error", reason=f"request failed: {type(ex).__name__}: {str(ex)[:120]}", elapsed=round(time.time() - t0, 4))
                self._log(rec)
                return Response(reason=rec["reason"], sent=False)
            if method == "GET" and status in (301, 302, 303, 307, 308) and loc and hops < p.redirects:
                nxt = urllib.parse.urljoin(url, loc)
                nu = urllib.parse.urlsplit(nxt)
                same = (nu.scheme, (nu.hostname or "").lower(), nu.port or (443 if nu.scheme == "https" else 80)) == (self.scheme, self.host, self.port) and not nu.username
                if not same:
                    rec.update(verdict="sent", status=status, reason="redirect to another origin refused (not followed)")
                    break
                path_q = nu.path + ("?" + nu.query if nu.query else "")
                full, why = self._clean_path(path_q[len(self.prefix):] if self.prefix and path_q.startswith(self.prefix) else path_q, None)
                if full is None:
                    rec.update(verdict="sent", status=status, reason=f"redirect refused: {why}")
                    break
                hops += 1
                self._log(dict(rec, verdict="redirect", status=status, reason=f"following hop {hops}", path=nu.path[:200]))
                url = nxt
                continue
            rec.update(verdict="sent", status=status)
            if status in (301, 302, 303, 307, 308) and loc and not rec.get("reason"):
                rec["reason"] = "redirect not followed (only GET redirects on the same origin are)" if method != "GET" or hops >= p.redirects else ""
            break
        items = _header_items(rheaders)
        resp_headers, cookies = {}, []
        for k, v in items:
            if k == "set-cookie":
                c = _parse_set_cookie(str(v))
                if c:
                    cookies.append(c)
            else:
                resp_headers[k] = f"{resp_headers[k]}, {v}" if k in resp_headers else str(v)
        truncated = len(data) > p.max_response_bytes
        data = data[: p.max_response_bytes]
        ctype = resp_headers.get("content-type", "")
        charset = re.search(r"charset=([\w-]+)", ctype, re.I)
        try:
            text = data.decode(charset.group(1) if charset else "utf-8", "replace")
        except LookupError:
            text = data.decode("utf-8", "replace")
        parsed = None if truncated else parse_json_bounded(text, p.max_json_depth)
        secret_values = self._secret_values()
        if secret_values:
            text = scrub(text, secret_values)
            parsed = scrub(parsed, secret_values) if parsed is not None else None
        snippet = redact(parsed, secret_values) if parsed is not None else (None if not text else ("***" if any(s in text for s in secret_values) else text[:120]))
        rec.update(resp_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), elapsed=round(time.time() - t0, 4), response=snippet)
        self._log(rec)
        return Response(status=status, headers=resp_headers, set_cookies=cookies, json=parsed, text=text[:4000], elapsed=time.time() - t0, nbytes=len(data), sent=True,
                        reason=rec.get("reason", ""), truncated=truncated, redirected=hops)


# ----------------------------------------------------------------------------------------------------------------------------- the session
_SLOT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")
_HEADER_SAFE = re.compile(r"^[A-Za-z0-9._~+/=:\-]{1,256}$")
_SEGMENT_SAFE = re.compile(r"^[A-Za-z0-9._~\-]{1,64}$")


class Session:
    """The state a client keeps between requests: cookies, the values earlier responses contained (`memory`, referenced by Ref('slot', name)) and the last few responses.
    The learner reads and writes NOTHING here directly; it picks an action and the environment calls `request`."""

    MEMORY_SLOTS = 32

    def __init__(self, actor: Actor, history: int = 8):
        self.actor = actor
        self.memory: dict = {}
        self.cookies: dict = {}
        self.history = collections.deque(maxlen=history)
        self.n_requests = 0

    def reset(self):
        self.memory.clear()
        self.cookies.clear()
        self.history.clear()
        self.n_requests = 0

    # ---- resolving slots ----------------------------------------------------------------------------------------------
    def _slot(self, ref: Ref, where: str):
        if ref.name not in self.memory:
            raise LookupError(f"slot '{ref.name}' has no value yet")
        v = self.memory[ref.name]
        text = str(v)
        if where == "header":
            if not _HEADER_SAFE.match(text):
                raise LookupError(f"slot '{ref.name}' is not safe to place in a header")
        elif where == "segment":
            if not _SEGMENT_SAFE.match(text) or text in (".", ".."):
                raise LookupError(f"slot '{ref.name}' is not safe to place in a path segment")
        elif where == "path":
            if not _PATH_OK.match(text.split("?")[0]) or not text.startswith("/"):
                raise LookupError(f"slot '{ref.name}' is not a path")
        return ref.fmt.format(text) if where != "body" else (v if ref.fmt == "{}" else ref.fmt.format(text))

    def _resolve_body(self, obj):
        if isinstance(obj, Ref):
            return self._slot(obj, "body") if obj.kind == "slot" else obj
        if isinstance(obj, dict):
            return {k: self._resolve_body(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._resolve_body(v) for v in obj]
        return obj

    def build_path(self, path) -> str:
        """A path given as a string, a Ref (a whole link from an earlier response) or a list of segments (strings and Refs) -> a path string."""
        if isinstance(path, Ref):
            if path.kind != "slot":
                raise LookupError("secrets are never placed in a URL")
            return self._slot(path, "path")
        if isinstance(path, (list, tuple)):
            segs = []
            for s in path:
                if isinstance(s, Ref):
                    if s.kind != "slot":
                        raise LookupError("secrets are never placed in a URL")
                    segs.append(self._slot(s, "segment"))
                else:
                    segs.append(str(s).strip("/"))
            return "/" + "/".join(segs)
        return str(path)

    # ---- one request ------------------------------------------------------------------------------------------------------
    def request(self, method: str, path, json=None, headers=None, query=None) -> Response:
        try:
            p = self.build_path(path)
            hdr = {}
            for k, v in (headers or {}).items():
                hdr[k] = self._slot(v, "header") if isinstance(v, Ref) and v.kind == "slot" else v
            q = {k: (self._slot(v, "segment") if isinstance(v, Ref) else v) for k, v in (query or {}).items()}
            body = self._resolve_body(json) if json is not None else None
        except LookupError as ex:
            resp = Response(refused=True, reason=str(ex))                     # the template cannot be filled in yet: nothing is sent, and nothing is counted
            self.actor._log(dict(method=str(method).upper(), host=self.actor.host, path="?", verdict="refused", reason=str(ex), status=None))
            self.actor.n_refused += 1
            self.history.append(resp)
            return resp
        resp = self.actor.request(method, p, json_body=body, headers=hdr, query=q or None, cookie=self.cookies, episode_used=self.n_requests)
        self.n_requests += 1
        self._learn(resp)
        self.history.append(resp)
        return resp

    def _learn(self, resp: Response):
        for name, val, deleted in resp.set_cookies:
            if deleted:
                self.cookies.pop(name, None)
            else:
                self.cookies[name] = val
        if not resp.sent or resp.json is None:
            return
        found = {}

        def walk(o, depth):
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(k, str) and _SLOT_NAME.match(k):
                        if isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 256):
                            found.setdefault(k, v)
                        elif isinstance(v, dict) and depth < 2:
                            walk(v, depth + 1)
        walk(resp.json, 0)
        for k, v in found.items():
            if k in self.memory:
                del self.memory[k]
            self.memory[k] = v
        while len(self.memory) > self.MEMORY_SLOTS:
            self.memory.pop(next(iter(self.memory)))
