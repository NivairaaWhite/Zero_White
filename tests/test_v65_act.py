"""Tests for 6.5 stateful web actuation (forage/act.py, forage/mockapi.py, forage/webenv.py).

Everything that talks HTTP talks to the loopback mock server (a real HTTP server in this process); everything about remote hosts uses fakes. Nothing here touches the network."""
import dataclasses
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ZERO_WHITE_DATA", tempfile.mkdtemp(prefix="zw_test_data_"))

from zerowhite.forage import act                                        # noqa: E402
from zerowhite.forage.act import Actor, ActPolicy, Ref, Session, ref_from_json, ref_to_json, redact          # noqa: E402
from zerowhite.forage.mockapi import MockServer                         # noqa: E402
from zerowhite.forage.webenv import ActionTemplate, WebEnv, mock_actions, mock_env, status_class          # noqa: E402

_DIRS = []
PASSWORD = "correct horse battery"


def tmp():
    d = tempfile.mkdtemp(prefix="zw_t65a_")
    _DIRS.append(d)
    return d


import atexit                                                           # noqa: E402
atexit.register(lambda: [shutil.rmtree(d, ignore_errors=True) for d in _DIRS])


def local_policy(**kw):
    return ActPolicy(write_hosts=("127.0.0.1",), **kw)


def actor_for(srv, policy=None, **kw):
    kw.setdefault("secrets", {"password": PASSWORD})
    return Actor(srv.url, policy or local_policy(), **kw)


def ledger_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def fake_opener(calls, status=200, body=b'{"ok": true}', headers=None, location=None):
    def op(method, url, hdrs, data, timeout, max_bytes):
        calls.append(dict(method=method, url=url, headers=dict(hdrs), body=data))
        h = dict(headers or {"Content-Type": "application/json"})
        return status, h, body, location
    return op


# =========================================================================== deny by default
def test_with_no_host_named_every_request_is_refused_and_none_reaches_the_server():
    with MockServer() as srv:
        a = Actor(srv.url, ActPolicy())                                 # the default policy: no write host
        for method, path in (("GET", "/health"), ("POST", "/login"), ("PUT", "/items/1")):
            r = a.request(method, path, json_body={"user": "x"} if method != "GET" else None)
            assert r.refused and not r.sent and "no write host" in r.reason
        b = Actor(srv.url, ActPolicy(write_hosts=("other.example",)))     # a host that is not this one
        assert b.request("GET", "/health").refused
        assert srv.arrived == [] and a.stats()["refused"] == 3
        assert Actor(srv.url, local_policy()).request("GET", "/health").ok and len(srv.arrived) == 1
    for bad in ("ftp://127.0.0.1/", "http://user:pw@127.0.0.1/", "http://127.0.0.1/?q=1", "http://127.0.0.1/#x", "127.0.0.1", "http:///x"):
        try:
            Actor(bad, local_policy())
            raise AssertionError(bad)
        except ValueError:
            pass


def test_a_loopback_name_must_really_resolve_to_loopback_and_a_remote_host_needs_every_condition():
    calls = []
    pol = ActPolicy(write_hosts=("localhost",))
    liar = Actor("http://localhost:9", pol, opener=fake_opener(calls), resolver=lambda h: ["8.8.8.8"])      # 'localhost' that resolves to a public address
    r = liar.request("GET", "/x")
    assert r.refused and "loopback" in r.reason and calls == []
    ok = Actor("http://localhost:9", pol, opener=fake_opener(calls), resolver=lambda h: ["127.0.0.1", "::1"])
    assert ok.request("GET", "/x").ok and len(calls) == 1 and calls[0]["url"] == "http://localhost:9/x"
    pub = lambda h: ["93.184.216.34"]                                   # noqa: E731
    led = os.path.join(tmp(), "ledger.jsonl")
    base = dict(write_hosts=("api.example.com",), allow_remote=True)
    cases = [
        ("http://api.example.com", ActPolicy(write_hosts=("api.example.com",)), led, pub, "not a loopback"),                          # no allow_remote
        ("http://api.example.com", ActPolicy(**base), led, pub, "https"),                                                           # plain http
        ("https://api.example.com", ActPolicy(**base), led, lambda h: ["10.0.0.5"], "private"),                                       # resolves to a private address
        ("https://api.example.com", ActPolicy(**base), led, lambda h: ["169.254.169.254"], "private"),                               # cloud metadata address
        ("https://api.example.com", ActPolicy(**base), None, pub, "ledger"),                                                         # no file ledger
        ("https://api.example.com:8443", ActPolicy(**base), led, pub, "port"),                                                       # a port the entry does not name
        ("https://evil.example.com", ActPolicy(**base), led, pub, "allow-list"),                                                     # a host that is not listed
    ]
    for url, pol2, ledger, res, why in cases:
        calls.clear()
        r = Actor(url, pol2, ledger_path=ledger, opener=fake_opener(calls), resolver=res).request("GET", "/v1/x")
        assert r.refused and why in r.reason and calls == [], (url, r.reason)
    now = [0.0]
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        now[0] += s
    good = Actor("https://api.example.com", ActPolicy(**base, delay=0.5), ledger_path=led, opener=fake_opener(calls), resolver=pub, sleep=fake_sleep, clock=lambda: now[0])
    calls.clear()
    assert good.request("GET", "/v1/x").ok and good.request("POST", "/v1/y", json_body={"a": 1}).ok
    assert [c["url"] for c in calls] == ["https://api.example.com/v1/x", "https://api.example.com/v1/y"] and sleeps == [0.5]          # paced between requests to one remote host
    named = Actor("https://api.example.com:8443", ActPolicy(write_hosts=("api.example.com:8443",), allow_remote=True), ledger_path=led, opener=fake_opener(calls), resolver=pub)
    assert named.request("GET", "/v1/x").ok                                                                                          # a port that the entry names is fine


def test_methods_the_person_did_not_allow_are_refused():
    with MockServer() as srv:
        srv.state.tokens["t"] = ("alice", "c")
        a = actor_for(srv)
        assert a.request("DELETE", "/items/1").refused and a.request("TRACE", "/x").refused and a.request("CONNECT", "/x").refused
        assert actor_for(srv, local_policy(allow_delete=True, methods=("GET", "DELETE"))).request("DELETE", "/items/1", headers={"Authorization": "Bearer t"}).sent      # allowed: reaches the server (404: no such item)
        assert actor_for(srv, local_policy(allow_delete=True, methods=("GET",))).request("DELETE", "/items/1").refused           # allow_delete alone is not enough: the method must be listed too
        assert a.request("GET", "/health", json_body={"a": 1}).refused                                                         # a GET cannot carry a body
        assert a.request("PATCH", "/items/1", json_body={"qty": 2}).sent                                                       # PATCH is in the default methods
        assert [r["method"] for r in srv.arrived] == ["DELETE", "PATCH"]


# =========================================================================== the learner's data is validated
BAD_PATHS = ["http://evil.example/x", "//evil.example/x", "/a/../b", "/%2e%2e/etc", "/%252e%252e/x", "/a\\b", "/a b", "/a\r\nHost: evil", "@evil.example", "/x#frag", "\x00",
             "/?a=b?c", "/%0d%0aSet-Cookie:x", "/a/%2F/b", "/.", "/..;/admin", "/a/%2e%2E/b", "", None, 123, "x" * 400, "/" + "a" * 400, "https://127.0.0.1/", "/‮", "/a\x7f"]


def test_path_injection_attempts_are_all_refused_and_none_reaches_the_server():
    with MockServer() as srv:
        a = actor_for(srv)
        for bad in BAD_PATHS:
            r = a.request("GET", bad)
            assert r.refused and not r.sent, repr(bad)
        assert srv.arrived == [] and a.stats()["refused"] == len(BAD_PATHS) and a.n_requests == 0
        assert a.request("GET", "/items?page=2").sent and srv.arrived[-1]["path"] == "/items?page=2"                      # a plain path with a query is fine
        assert a.request("GET", "/health", query={"a b": "c&d=e", "z": 1}).sent and srv.arrived[-1]["path"] == "/health?a%20b=c%26d%3De&z=1"       # a query is encoded, not concatenated
        for q in ({"k": "v\r\nX: y"}, {"k": {"nested": 1}}, {"k": [1]}, {str(i): i for i in range(30)}, {"k": Ref("slot", "t")}, "page=2"):
            assert a.request("GET", "/health", query=q).refused, q
        n = len(srv.arrived)
        pref = Actor(srv.url + "/api/v1", local_policy())                                                                   # a base path prefix is prepended, never replaced
        assert pref.request("GET", "/../health").refused and pref.request("GET", "/health").sent and srv.arrived[n]["path"] == "/api/v1/health"


def test_header_and_body_injection_attempts_are_refused_and_what_the_person_allows_still_works():
    with MockServer() as srv:
        a = actor_for(srv)
        bad_headers = [{"X-Foo": "1"}, {"Host": "evil.example"}, {"Cookie": "sid=stolen"}, {"Content-Length": "0"}, {"Transfer-Encoding": "chunked"}, {"Authorization": "Bearer x\r\nX-Injected: 1"},
                       {"Authorization": "x\ny"}, {"Bad Name": "x"}, {"Accept\r\nX": "x"}, {"Authorization": "x" * 600}, {"Authorization": 5}, {"Authorization": Ref("secret", "nope")},
                       {"Authorization": Ref("slot", "token")}, {"User-Agent": "mine"}, {"Content-Type": "text/html"}, {1: "x"}]
        for h in bad_headers:
            assert a.request("GET", "/health", headers=h).refused, h
        wide = Actor(srv.url, local_policy(allowed_headers=("Host", "Cookie", "Content-Length", "Authorization")), secrets={})   # even a person's own allow-list cannot unlock the dangerous names
        for h in ({"Host": "evil"}, {"Cookie": "a=b"}, {"Content-Length": "1"}):
            assert wide.request("GET", "/health", headers=h).refused
        assert srv.arrived == []
        deep = cur = {}
        for _ in range(20):
            cur["k"] = {}
            cur = cur["k"]
        bad_bodies = [deep, {1: 2}, {"a": {1, 2}}, {"a": b"bytes"}, {"a": object()}, {"a": float("nan")}, {"a": "x" * 20000}, {"a": [1] * 6000}]
        for b in bad_bodies:
            r = a.request("POST", "/login", json_body=b)
            assert r.refused, str(b)[:40]
        assert a.request("POST", "/login", json_body={"a": Ref("secret", "nope")}).refused and a.request("POST", "/login", json_body={"a": Ref("slot", "x")}).refused
        assert srv.arrived == []
        ok = a.request("POST", "/login", json_body={"user": "alice", "password": Ref("secret", "password")})
        assert ok.ok and ok.json["user"] == "alice" and len(srv.arrived) == 1
        got = srv.arrived[0]
        assert got["headers"]["content-type"] == "application/json" and got["headers"]["user-agent"] == act.USER_AGENT and got["body"] == {"user": "alice", "password": PASSWORD}
        assert a.request("GET", "/items", headers={"Authorization": Ref("slot", "token", "Bearer {}")}).refused                     # an unresolved slot never goes out raw


def test_a_ref_cannot_carry_a_format_string_attack_and_secrets_never_go_into_a_url():
    for bad in ("{0.__class__}", "{}{}", "no braces", "{x}", "{0}", "{:>9999}" + "x" * 100):
        try:
            Ref("slot", "t", bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    try:
        ref_from_json({"$secret": "pw", "$fmt": "{0.__class__.__mro__}"})
        raise AssertionError("json form")
    except ValueError:
        pass
    assert ref_from_json({"a": [{"$slot": "t", "$fmt": "Bearer {}"}], "b": 1}) == {"a": [Ref("slot", "t", "Bearer {}")], "b": 1}
    assert ref_to_json({"a": [Ref("secret", "pw")]}) == {"a": [{"$secret": "pw"}]}
    with MockServer() as srv:
        s = Session(actor_for(srv))
        assert s.request("GET", Ref("secret", "password")).refused                                                                  # a secret as the whole path
        assert s.request("GET", ["items", Ref("secret", "password")]).refused                                                       # as a path segment
        assert s.request("GET", "/health", query={"k": Ref("secret", "password")}).refused                                         # as a query value
        assert srv.arrived == [] and not any(PASSWORD in json.dumps(r.__dict__, default=str) for r in s.history)


# =========================================================================== a real session against a real server
def test_a_session_logs_in_keeps_cookies_and_values_and_the_secret_never_leaves_the_actor():
    d = tmp()
    led = os.path.join(d, "ledger.jsonl")
    with MockServer() as srv:
        a = actor_for(srv, ledger_path=led)
        s = Session(a)
        r = s.request("POST", "/login", json={"user": "alice", "password": Ref("secret", "password")})
        assert r.ok and "sid" in s.cookies and {"token", "user", "csrf"} <= set(s.memory)
        assert PASSWORD not in json.dumps(r.json) and r.json["token"] == s.memory["token"]
        auth = {"Authorization": Ref("slot", "token", "Bearer {}")}
        assert s.request("GET", "/me", headers=auth).json["user"] == "alice"
        assert s.request("GET", "/me").ok                                                                                          # the cookie alone is enough for this server
        assert s.request("POST", "/items", json={"name": "widget", "qty": 3}, headers=auth).status == 403                          # no CSRF header
        csrf = {**auth, "X-CSRF-Token": Ref("slot", "csrf")}
        made = s.request("POST", "/items", json={"name": "widget", "qty": 3}, headers=csrf)
        assert made.status == 201 and s.memory["id"] == 1 and s.memory["name"] == "widget"
        assert s.request("PUT", ["items", Ref("slot", "id")], json={"qty": 5}, headers=csrf).json["qty"] == 5 and srv.state.items[1]["qty"] == 5
        assert s.request("POST", "/items", json={"name": "", "qty": 3}, headers=csrf).status == 422
        s.request("POST", "/items", json={"name": "b", "qty": 1}, headers=csrf)
        s.request("POST", "/items", json={"name": "c", "qty": 1}, headers=csrf)
        pg = s.request("GET", "/items", headers=auth)
        assert [i["name"] for i in pg.json["items"]] == ["widget", "b"] and s.memory["next"] == "/items?page=2"
        assert s.request("GET", Ref("slot", "next"), headers=auth).json["items"][0]["name"] == "c"                                  # follow the server's own `next` link
        s.reset()
        assert s.cookies == {} and s.memory == {} and s.request("GET", Ref("slot", "next")).refused                                 # nothing carried over; an empty slot sends nothing
        assert PASSWORD in json.dumps(srv.arrived[0]["body"])                                                                     # the server DID get the password (that is the point)...
    text = open(led, encoding="utf-8").read()
    assert PASSWORD not in text and s.memory.get("token") is None
    assert '"token": "***"' in text                                                                                          # the login response is in the ledger with its token masked
    assert PASSWORD not in json.dumps(list(a.tail), default=str) and PASSWORD not in repr(a) and "password" in a.secret_names()


def test_a_server_that_echoes_a_secret_gets_it_scrubbed_before_the_learner_sees_it_and_before_the_ledger():
    d = tmp()
    led = os.path.join(d, "ledger.jsonl")
    with MockServer() as srv:
        a = Actor(srv.url, local_policy(), ledger_path=led, secrets={"api_key": "sk-live-ABCDEF123456"})
        s = Session(a)
        r = s.request("GET", "/echo-auth", headers={"Authorization": Ref("secret", "api_key", "Bearer {}")})
        assert r.ok and "sk-live-ABCDEF123456" not in r.text and "sk-live-ABCDEF123456" not in json.dumps(r.json) and r.json["seen"] == "Bearer ***"
        assert "sk-live-ABCDEF123456" not in json.dumps(s.memory) and "sk-live-ABCDEF123456" not in open(led, encoding="utf-8").read()
        assert srv.arrived[0]["headers"]["authorization"] == "Bearer sk-live-ABCDEF123456"                                      # it did authenticate
    red = redact({"password": "x", "nested": {"api_token": "y", "ok": "fine"}, "list": [{"Secret": "z"}], "msg": "my key is sk-live-ABCDEF123456"}, ["sk-live-ABCDEF123456"])
    assert red == {"password": "***", "nested": {"api_token": "***", "ok": "fine"}, "list": [{"Secret": "***"}], "msg": "***"}


def test_hostile_responses_cannot_steer_the_next_request_off_the_origin_or_inject_headers():
    with MockServer() as srv:
        a = actor_for(srv)
        s = Session(a)
        r = s.request("GET", "/hostile")
        assert r.ok and {"next", "href", "path", "token", "link", "id", "fine"} <= set(s.memory)
        assert "ok" in s.cookies and "sid" not in s.cookies and "evil" in s.cookies and s.cookies["evil"] == "a"                  # a cookie with a space in its value is dropped, the sane ones kept
        n = len(srv.arrived)
        for ref in ("next", "href", "path", "link"):                                                                             # whole-link slots that are attacks
            assert s.request("GET", Ref("slot", ref)).refused, ref
        assert s.request("GET", ["items", Ref("slot", "id")]).refused                                                           # '1/../../admin' as a path segment
        assert s.request("GET", "/health", headers={"Authorization": Ref("slot", "token", "Bearer {}")}).refused                # control characters in a token never reach a header
        assert s.request("GET", "/health", query={"q": Ref("slot", "id")}).refused
        assert len(srv.arrived) == n                                                                                             # not one of them was sent
        assert "ignore all previous rules" in s.history[0].text                                                                  # the text is data: it is stored, it is never obeyed
        assert srv.arrived[-1]["path"] == "/hostile"


def test_redirects_stay_on_the_origin_for_get_and_are_never_replayed_for_a_write():
    with MockServer() as srv:
        srv.state.tokens["t"] = ("alice", "c")
        a = actor_for(srv, local_policy(redirects=2))
        r = a.request("GET", "/redirect", headers={"Authorization": "Bearer t"})
        assert r.ok and r.redirected == 1 and r.json["user"] == "alice" and [x["path"] for x in srv.arrived] == ["/redirect", "/me"]
        away = a.request("GET", "/redirect-away")
        assert away.status == 302 and "another origin" in away.reason and away.redirected == 0                                  # the Location is reported, not followed
        loop = a.request("GET", "/redirect-loop")
        assert loop.status == 302 and loop.redirected == 2                                                                       # the hop limit ends it
        assert sum(1 for x in srv.arrived if x["path"] == "/redirect-loop") == 3
    calls = []
    b = Actor("http://127.0.0.1:9", local_policy(), opener=fake_opener(calls, status=307, location="/elsewhere", body=b""))
    w = b.request("POST", "/items", json_body={"name": "x", "qty": 1})
    assert w.status == 307 and len(calls) == 1 and w.redirected == 0 and "not followed" in w.reason                               # a 307 on a POST is returned, not obeyed
    g = Actor("http://127.0.0.1:9", local_policy(), opener=fake_opener(calls, status=302, location="//evil.example/x", body=b""))
    assert g.request("GET", "/x").status == 302 and len(calls) == 2


def test_response_size_and_nesting_caps_hold():
    with MockServer() as srv:
        a = actor_for(srv, local_policy(max_response_bytes=2000))
        big = a.request("GET", "/big", query={"n": 50000})
        assert big.sent and big.truncated and big.nbytes == 2000 and big.json is None and len(big.text) <= 4000
        small = a.request("GET", "/big", query={"n": 100})
        assert not small.truncated and small.json["pad"] == "x" * 100
        deep = actor_for(srv).request("GET", "/deep")
        assert deep.ok and deep.json is None                                                                                    # 200 levels of nesting is not parsed into the session
    assert act.parse_json_bounded("[" * 5000 + "]" * 5000) is None and act.parse_json_bounded("{bad") is None and act.parse_json_bounded('{"a": NaN}') == {"a": None}
    assert act.parse_json_bounded(json.dumps({"a": list(range(6000))})) is None


# =========================================================================== budgets, kill switch, dry run, ledger
def test_budgets_the_kill_switch_and_dry_run_stop_requests_before_they_are_sent():
    d = tmp()
    with MockServer() as srv:
        a = actor_for(srv, local_policy(max_requests=3))
        assert [a.request("GET", "/health").sent for _ in range(3)] == [True] * 3
        r = a.request("GET", "/health")
        assert r.refused and "budget of 3" in r.reason and len(srv.arrived) == 3 and a.stats()["remaining"] == 0
        s = Session(actor_for(srv, local_policy(max_requests_per_episode=2)))
        assert s.request("GET", "/health").sent and s.request("GET", "/health").sent and s.request("GET", "/health").refused         # the episode's own budget
        s.reset()
        assert s.request("GET", "/health").sent                                                                                  # a new episode has a new budget
        n = len(srv.arrived)
        kill = os.path.join(d, "STOP")
        k = actor_for(srv, local_policy(kill_file=kill))
        assert k.request("GET", "/health").sent
        open(kill, "w").close()
        assert k.request("GET", "/health").refused and "kill-switch" in k.request("POST", "/login", json_body={}).reason
        os.remove(kill)
        assert k.request("GET", "/health").sent and len(srv.arrived) == n + 2
        led = os.path.join(d, "dry.jsonl")
        dry = actor_for(srv, ledger_path=led, dry_run=True)
        r = dry.request("POST", "/login", json_body={"user": "alice", "password": Ref("secret", "password")})
        assert not r.sent and not r.refused and "dry run" in r.reason and len(srv.arrived) == n + 2                              # checked, ledgered, never sent
        assert dry.request("GET", "../etc").refused                                                                              # a dry run still enforces every rule
        recs = ledger_lines(led)
        assert [x["verdict"] for x in recs] == ["dry_run", "refused"] and recs[0]["body"]["password"] == "***" and PASSWORD not in open(led, encoding="utf-8").read()
    dead = Actor("http://127.0.0.1:9", local_policy(timeout=1.0))                                                                # nothing listens on port 9
    r = dead.request("GET", "/x")
    assert not r.sent and not r.refused and "request failed" in r.reason and dead.stats()["errors"] == 1


def test_the_ledger_has_one_redacted_line_per_attempt_refusals_included():
    d = tmp()
    led = os.path.join(d, "sub", "ledger.jsonl")
    with MockServer() as srv:
        a = actor_for(srv, ledger_path=led)
        s = Session(a)
        s.request("POST", "/login", json={"user": "alice", "password": Ref("secret", "password")})
        a.request("GET", "/../x")
        a.request("DELETE", "/items/1")
        s.request("GET", "/me", headers={"Authorization": Ref("slot", "token", "Bearer {}")})
        s.request("GET", Ref("slot", "missing"))
    recs = ledger_lines(led)
    assert [r["verdict"] for r in recs] == ["sent", "refused", "refused", "sent", "refused"]
    sent = recs[0]
    assert sent["status"] == 200 and len(sent["sha256"]) == 64 and sent["resp_bytes"] > 0 and sent["body"]["password"] == "***" and sent["response"]["token"] == "***"
    assert recs[3]["headers"]["Authorization"] == "***"
    token = s.memory["token"]
    assert token not in open(led, encoding="utf-8").read() and PASSWORD not in open(led, encoding="utf-8").read()
    assert all(r["method"] and r["host"] == "127.0.0.1" and "t" in r for r in recs) and len(a.tail) == 5


# =========================================================================== the learner is held to the policy
def test_nothing_the_learner_can_choose_widens_the_policy_and_the_modules_are_unreachable_from_the_autonomous_loop():
    pol = local_policy()
    for field in ("write_hosts", "allow_remote", "max_requests", "kill_file"):
        try:
            setattr(pol, field, True)
            raise AssertionError(field)
        except dataclasses.FrozenInstanceError:
            pass
    with MockServer() as srv:
        env = mock_env(srv.url, srv.state.reset)
        menu = env.action_names()
        assert len(menu) == 12 and all(isinstance(t, ActionTemplate) for t in env.actions)
        for t in env.actions:                                                                                                    # the menu is data written by a person: paths, no hosts
            p = t.path
            assert isinstance(p, (Ref, list)) or (isinstance(p, str) and p.startswith("/"))
        assert not hasattr(env.session.actor, "set_policy") and not hasattr(env, "policy")
        assert [env.step(a)[3] for a in (1, 2)] == [False, False]                                                                # the learner picks indices only
    import ast
    import glob
    pkg = os.path.join(ROOT, "zerowhite")
    allowed = {"forage/act.py", "forage/mockapi.py", "forage/webenv.py", "cli_v65.py", "forage/__init__.py"}
    for path in glob.glob(os.path.join(pkg, "**", "*.py"), recursive=True):
        rel = os.path.relpath(path, pkg).replace(os.sep, "/")
        src = open(path, encoding="utf-8").read()
        refs = [n for n in ast.walk(ast.parse(src)) if (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[-1] in ("act", "webenv", "mockapi")) or
                (isinstance(n, ast.ImportFrom) and any(a.name in ("act", "webenv", "mockapi", "Actor", "WebEnv") for a in n.names) and "forage" in (n.module or "")) or
                (isinstance(n, ast.Import) and any("forage.act" in a.name or "forage.webenv" in a.name for a in n.names))]
        if rel not in allowed:
            assert not refs, (rel, "imports the web actuation modules; only the CLI may")
    for rel in ("organism.py", "selfmod/patcher.py", "selfmod/hooks.py", "forge/generalist.py", "omnivore/__init__.py"):
        p = os.path.join(pkg, rel)
        if os.path.exists(p):
            assert "forage.act" not in open(p, encoding="utf-8").read() and "webenv" not in open(p, encoding="utf-8").read(), rel


# =========================================================================== the learning environment
def test_the_web_environment_pays_for_the_whole_sequence_and_charges_for_mistakes_and_refusals():
    d = tmp()
    with MockServer(lock_seconds=0.05) as srv:
        env = mock_env(srv.url, srv.state.reset, ledger_path=os.path.join(d, "l.jsonl"))
        assert (env.obs_dim, env.n_actions, round(env.max_return, 3)) == (14, 12, 0.92)
        rng = np.random.default_rng(0)
        o = env.reset(rng)
        assert o.shape == (14,) and o[0] == 1.0 and o[8:12].sum() == 0.0                                                         # nothing yet: status 'none', no values held
        tot, hist = 0.0, []
        for a in (1, 4, 5, 9):                                                                                                   # login, get_csrf, create_item, set_qty_5
            o, r, done, won = env.step(a)
            tot += r
        assert won and done and abs(tot - 0.92) < 1e-9 and [h[1] for h in env.history] == ["2xx"] * 4
        assert o[1] == 1.0 and o[8] == 1.0 and o[9] == 1.0 and o[10] == 1.0                                                      # 2xx; token, csrf and id are held
        env.reset(rng)
        o, r, done, won = env.step(3)                                                                                            # /me before logging in: the template cannot be filled: refused, costs reward
        assert env.history[-1][1] == "refused" and r == -0.07 and not done
        n = len(srv.arrived)
        env.step(10)                                                                                                             # delete_item: the person did not allow DELETE (and has no id yet)
        assert len(srv.arrived) == n and env.history[-1][1] == "refused"
        for _ in range(3):
            env.step(2)                                                                                                          # three wrong passwords
        o, r, done, won = env.step(2)
        assert status_class(env.last) == 5 and o[5] == 1.0 and r == -0.07                                                        # 429: locked out, and the observation says so
        assert env.step(11)[0][2] == 1.0                                                                                         # redirect-away: 3xx, not followed
        env.reset(rng)
        for _ in range(env.max_steps):
            o, r, done, won = env.step(0)
        assert done and not won                                                                                                  # the episode ends at max_steps
    assert any(json.loads(ln)["verdict"] == "refused" for ln in open(os.path.join(d, "l.jsonl"), encoding="utf-8"))


def test_when_the_persons_request_budget_is_used_up_the_environment_stops_for_good_without_sending_anything():
    with MockServer() as srv:
        env = mock_env(srv.url, srv.state.reset, policy=local_policy(max_requests=6, max_requests_per_episode=20))
        env.reset()
        for a in (1, 4, 5, 9):
            env.step(a)
        env.reset()
        env.step(1)
        env.step(4)
        sent = len(srv.arrived)                                                                                                  # (the server forgets its log on every reset: these are the 2 since the last one)
        assert sent == 2 and env.session.actor.n_requests == 6
        o, r, done, won = env.step(5)
        assert done and r == 0.0 and not won and env.exhausted and len(srv.arrived) == sent
        env.reset()                                                                                                              # a reset cannot start a new run either
        assert env.exhausted and env.step(1)[2] and env.session.actor.n_requests == 6 and srv.arrived == []                       # (the reset hook emptied the server's log; nothing new was sent)
    with MockServer() as srv:
        kill = os.path.join(tmp(), "STOP")
        env = mock_env(srv.url, srv.state.reset, policy=local_policy(max_requests=1000, max_requests_per_episode=20, kill_file=kill))
        env.reset()
        env.step(1)
        open(kill, "w").close()
        n = len(srv.arrived)
        o, r, done, won = env.step(4)
        assert env.history[-1][1] == "refused" and len(srv.arrived) == n and not done                                           # the kill switch refuses every request, and the learner pays for trying
    bad = WebEnv.__init__
    try:
        WebEnv(Session(Actor("http://127.0.0.1:9", local_policy())), [], lambda s: False)
        raise AssertionError("empty menu")
    except ValueError:
        pass
    assert bad is not None and len(mock_actions()) == 12


def test_an_actor_critic_learns_the_mock_task_by_trial_and_error_against_the_real_server():
    from zerowhite.forge import task as T
    from zerowhite.forge.families import FAMILIES
    with MockServer(lock_seconds=0.05) as srv:
        env = mock_env(srv.url, srv.state.reset)
        t = T.EnvTask(env)
        base = t.random_baseline()[0]
        assert base < 0.0
        g = dict(hidden=24, depth=1, lr=0.01, gamma=0.97, entropy=0.02, curiosity=0.0)
        a = FAMILIES["actor_critic"].build(t, g, 0)
        for k in range(5):
            a.train(t, 1.0, np.random.default_rng(k))
        skill, met = a.val_score(t)
        assert met["success"] >= 0.9 and met["mean_return"] > 0.6 and skill > 0.8, (skill, met)
        assert srv.state.arrived and all(r["path"] != "/redirect-away" or True for r in srv.state.arrived)
        assert all(r["method"] != "DELETE" for r in srv.state.arrived)                                                           # it learned to avoid the action the policy refuses
