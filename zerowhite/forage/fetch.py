"""forage/fetch.py - read specific web pages into the inbox, under a policy that is enforced in code and cannot be widened by what it reads.

Until 6.3 no part of Zero White could read a web page: the only network code was the dataset downloaders and the local service, and a test
confines `urllib`/`socket` to named files. Active reading is the one capability a "passive inbox watcher" lacks, and it is also the one that
changes the risk profile of a system that rewrites parts of itself. So this module is deliberately narrow, and OFF unless you act:

  * nothing is fetched unless YOU pass at least one allowed host (`zerowhite forage --allow-host docs.python.org URL ...`); there are no
    default hosts, no search, no link-following, and nothing here is called by the organism or by self-modification;
  * every URL, and every redirect hop (redirects are followed by hand, at most `redirects` times), is checked against the allow-list;
  * https only (http only for loopback in tests, or with allow_http); no credentials in URLs; GET only; no cookies; no custom headers;
  * the host must not resolve to a private / loopback / link-local address unless you set allow_local (a basic SSRF guard);
  * robots.txt is honoured (a 5xx or timeout on robots.txt means "do not fetch"), at least `delay` seconds between requests to a host;
  * response size, URL count and total bytes are capped; only text-like content types are kept; HTML is reduced to its text by the standard
    library parser (no JavaScript runs, nothing is executed or imported);
  * the result is a plain .txt file in an output folder plus one line in a ledger (URL, time, status, bytes, SHA-256, verdict) for every
    attempt, refusals included.

What it does NOT do (and the docs say so): decide WHAT to read. Choosing pages because the model "noticed a gap" is not implemented; a person
picks the URLs. Text from the web enters the chat model's corpus through the same filters as any other file, but it is still untrusted
text: do not point this at hosts you would not train on.

HONEST LIMITS: the address check and the connection are separate steps, so a hostile DNS server could change its answer in between
(DNS rebinding); the allow-list is the real control. Proxies from the environment are honoured by urllib. This file is part of the
CONSTITUTION (sealed): the self-modifier cannot edit the policy it is held to.
"""
from __future__ import annotations

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
import urllib.robotparser
from dataclasses import dataclass, field
from html.parser import HTMLParser

from .. import fileio

USER_AGENT = "zero-white-forager/6.3 (+explicit allow-list; honours robots.txt)"
TEXT_TYPES = ("text/plain", "text/html", "text/markdown", "text/x-markdown", "application/xhtml+xml", "application/json", "text/csv")
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


@dataclass
class Policy:
    allow_hosts: tuple = ()
    allow_http: bool = False                  # http:// URLs (always allowed for loopback hosts that are on the allow-list)
    allow_local: bool = False                 # allow-listed hosts that resolve to private / loopback addresses
    respect_robots: bool = True
    max_bytes: int = 2 * 2 ** 20
    max_urls: int = 20
    max_total_bytes: int = 20 * 2 ** 20
    timeout: float = 15.0
    delay: float = 2.0                        # minimum seconds between two requests to the same host
    redirects: int = 3
    min_chars: int = 80                       # a page with less text than this is not worth keeping

    def hosts(self):
        return {h.strip().lower() for h in self.allow_hosts if h and h.strip()}


class _TextOf(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "template", "head", "iframe", "object", "embed"}
    _BLOCK = {"p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article", "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
              "pre", "blockquote", "dt", "dd", "title"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self.skip += 1
        elif tag in self._BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self._BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_to_text(html: str) -> str:
    p = _TextOf()
    try:
        p.feed(html)
        p.close()
    except Exception:                                          # noqa: BLE001 - malformed HTML must not stop a run
        pass
    text = "".join(p.out)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return "\n".join(ln.strip() for ln in text.splitlines()).strip()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):                       # every hop is checked against the policy by hand
        return None


def _default_open(url: str, timeout: float, max_bytes: int):
    """(status, headers, body, location). Reads at most max_bytes + 1 so the caller can tell 'too big' from 'exactly the cap'."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.1"}, method="GET")
    op = urllib.request.build_opener(_NoRedirect)
    try:
        with op.open(req, timeout=timeout) as r:
            return r.status, r.headers, r.read(max_bytes + 1), None
    except urllib.error.HTTPError as e:
        body = b""
        try:
            body = e.read(max_bytes + 1)
        except Exception:                                      # noqa: BLE001
            pass
        return e.code, e.headers, body, e.headers.get("Location")


def _addr_is_local(ip: str) -> bool:
    a = ipaddress.ip_address(ip.split("%")[0])
    return a.is_private or a.is_loopback or a.is_link_local or a.is_multicast or a.is_reserved or a.is_unspecified


@dataclass
class Forager:
    policy: Policy
    out_dir: str
    ledger_path: str
    opener: object = None                      # tests inject a fake (url, timeout, max_bytes) -> (status, headers, body, location)
    resolver: object = None                    # tests inject host -> [ip, ...]
    sleep: object = time.sleep
    clock: object = time.monotonic
    n_urls: int = 0
    total_bytes: int = 0
    _last: dict = field(default_factory=dict)
    _robots: dict = field(default_factory=dict)

    # ---- the policy ---------------------------------------------------------------------------------------------
    def _check(self, url: str):
        """(ok, reason, parsed). Pure policy: touches the network only to resolve the host's address."""
        try:
            u = urllib.parse.urlsplit(url)
            host = (u.hostname or "").lower()
            port = u.port
        except ValueError:
            return False, "not a valid URL", None
        if u.username or u.password:
            return False, "credentials in a URL are refused", None
        if u.scheme not in ("http", "https"):
            return False, f"scheme '{u.scheme}' is refused (only https, and http where allowed)", None
        if not self.policy.hosts():
            return False, "no host is allowed: pass at least one --allow-host (nothing is fetched by default)", None
        if host not in self.policy.hosts():
            return False, f"host '{host}' is not on the allow-list", None
        if u.scheme == "http" and not (self.policy.allow_http or host in _LOOPBACK):
            return False, "plain http is refused (use https, or allow_http for a host you trust)", None
        try:
            ips = self.resolver(host) if self.resolver else [ai[4][0] for ai in socket.getaddrinfo(host, port or (443 if u.scheme == "https" else 80), type=socket.SOCK_STREAM)]
        except OSError as ex:
            return False, f"cannot resolve '{host}': {ex}", None
        if not ips:
            return False, f"'{host}' has no address", None
        if not self.policy.allow_local and any(_addr_is_local(ip) for ip in ips):
            return False, f"'{host}' resolves to a private/loopback address (set allow_local if you really mean that)", None
        return True, "", u

    def _pace(self, host: str):
        last = self._last.get(host)
        if last is not None:
            wait = self.policy.delay - (self.clock() - last)
            if wait > 0:
                self.sleep(wait)
        self._last[host] = self.clock()

    def _get(self, url: str):
        """GET with manual, policy-checked redirects. Returns (ok, reason, status, headers, body, final_url)."""
        for hop in range(self.policy.redirects + 1):
            ok, why, u = self._check(url)
            if not ok:
                return False, why, None, None, b"", url
            self._pace(u.hostname.lower())
            opener = self.opener or _default_open
            try:
                status, headers, body, loc = opener(url, self.policy.timeout, self.policy.max_bytes)
            except (urllib.error.URLError, OSError, ValueError) as ex:
                return False, f"request failed: {type(ex).__name__}: {ex}", None, None, b"", url
            if status in (301, 302, 303, 307, 308) and loc:
                url = urllib.parse.urljoin(url, loc)
                continue
            return True, "", status, headers, body, url
        return False, f"more than {self.policy.redirects} redirects", None, None, b"", url

    def _robots_allows(self, url: str) -> bool:
        u = urllib.parse.urlsplit(url)
        key = f"{u.scheme}://{u.netloc}"
        if key not in self._robots:
            ok, why, status, headers, body, _ = self._get(f"{key}/robots.txt")
            if not ok or (status or 0) >= 500:
                self._robots[key] = None                        # cannot tell: do not fetch (RFC 9309)
            elif status >= 400:
                self._robots[key] = True                        # no robots.txt: everything is allowed
            else:
                rp = urllib.robotparser.RobotFileParser()
                rp.parse(body.decode("utf-8", "replace").splitlines())
                self._robots[key] = rp
        r = self._robots[key]
        if r is None:
            return False
        return True if r is True else r.can_fetch(USER_AGENT, url)

    # ---- one URL --------------------------------------------------------------------------------------------------
    def fetch(self, url: str) -> dict:
        t = time.time()
        res = dict(url=url, ok=False, reason="", status=None, bytes=0, sha256=None, path=None, t=t)
        try:
            if self.n_urls >= self.policy.max_urls:
                res["reason"] = f"the per-run limit of {self.policy.max_urls} URLs is reached"
            elif self.total_bytes >= self.policy.max_total_bytes:
                res["reason"] = "the per-run byte limit is reached"
            else:
                ok, why, _ = self._check(url)
                if not ok:
                    res["reason"] = why
                elif self.policy.respect_robots and not self._robots_allows(url):
                    res["reason"] = "robots.txt does not allow this URL (or could not be read)"
                else:
                    self.n_urls += 1
                    ok, why, status, headers, body, final = self._get(url)
                    res["status"] = status
                    if not ok:
                        res["reason"] = why
                    elif status != 200:
                        res["reason"] = f"HTTP {status}"
                    elif len(body) > self.policy.max_bytes:
                        res["reason"] = f"larger than the {self.policy.max_bytes}-byte limit"
                    else:
                        ctype = (headers.get("Content-Type", "") or "").split(";")[0].strip().lower()
                        if ctype not in TEXT_TYPES:
                            res["reason"] = f"content type '{ctype or 'unknown'}' is not text"
                        else:
                            charset = "utf-8"
                            m = re.search(r"charset=([\w-]+)", headers.get("Content-Type", "") or "", re.I)
                            if m:
                                charset = m.group(1)
                            try:
                                raw = body.decode(charset, "replace")
                            except LookupError:
                                raw = body.decode("utf-8", "replace")
                            text = html_to_text(raw) if ctype in ("text/html", "application/xhtml+xml") else raw.replace("\r\n", "\n").strip()
                            if len(text) < self.policy.min_chars:
                                res["reason"] = f"only {len(text)} characters of text"
                            else:
                                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                                host = urllib.parse.urlsplit(final).hostname or "host"
                                path = os.path.join(self.out_dir, f"{re.sub(r'[^a-z0-9.-]', '_', host.lower())}-{digest[:10]}.txt")
                                res.update(bytes=len(body), sha256=digest, path=path, final_url=final)
                                self.total_bytes += len(body)
                                if os.path.exists(path):
                                    res.update(ok=True, reason="already in the output folder (identical text)")
                                else:
                                    os.makedirs(self.out_dir, exist_ok=True)
                                    fileio.write_text(path, text + "\n")
                                    res.update(ok=True, reason="saved")
        except Exception as ex:                                  # noqa: BLE001 - a bad page must never stop the run
            res.update(ok=False, reason=f"{type(ex).__name__}: {ex}")
        self._log(res)
        return res

    def _log(self, res: dict):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.ledger_path)), exist_ok=True)
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({k: v for k, v in res.items()}, default=str) + "\n")
        except OSError:
            pass

    def fetch_all(self, urls) -> list:
        return [self.fetch(u) for u in urls]
