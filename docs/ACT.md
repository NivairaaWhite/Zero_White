# Stateful web actuation (6.5)

`forage/act.py`, `forage/webenv.py`, `forage/mockapi.py`, CLI `zerowhite act ...`. `forage/fetch.py` *reads* pages. This lets a program that is **learning** (an evolved policy, an actor-critic) use a web API by trial and error: log in, hold a token, send JSON, follow a `next` link, wait out a rate limit.

It is the riskiest capability in the project, so the design is *deny by default, and make the denial the code*, not the documentation.

```
zerowhite act demo                         # learn the practice task against a loopback server (mockapi.py), then show the ledger
zerowhite act serve-mock                   # run the practice server
zerowhite act request POST /items --base-url http://127.0.0.1:8080 --allow-host 127.0.0.1 \
        --json '{"name": "x", "qty": 1}' --header X-CSRF-Token=abc --dry-run
zerowhite act ledger | kill | unkill
```

## The rules (all enforced in `act.py`, all tests)

* **A person writes the policy** (`ActPolicy`, a frozen dataclass). The learner never receives it; the actor has no method that changes it.
* **No host named, nothing is sent.** Every request is refused and ledgered. A named host without `--allow-remote` must be loopback **and** must really resolve to loopback. Anything else needs `--allow-remote`, https, the named port (443 unless written `host:port`), a public address, and a file ledger.
* **Methods:** GET / POST / PUT / PATCH. DELETE only with `--allow-delete`. CONNECT and TRACE never. Redirects are followed by hand, GET only, same origin, never replayed for a write.
* **The learner controls data, never destination.** It can choose from a menu a person wrote (`ActionTemplate`), and the actor validates what is chosen: a relative path with a strict character set (no scheme, authority, backslash, control character, or `..` even percent-encoded), a capped JSON body, headers from a short allow-list with no CR/LF. It cannot choose the host, scheme, port, `Cookie`, `Host` or `Content-Length`.
* **Server-supplied values** (a token, an id, a `next` link) are used only through `Ref("slot", name)` and are re-validated where they are used. A hostile server that returns `next: "https://evil/"` or `//host/x` gets a refused request, not a request to that host.
* **Secrets** are given by name from the environment (`--secret-env NAME=ENVVAR`, never on the command line), substituted inside the actor, never returned, never in a URL, and shown as `***` in the ledger, the in-memory tail, and any response text that echoes them.
* **Budgets:** requests per run, requests per episode, a minimum delay per non-loopback host, request / response size caps, a **kill file** checked before every request, and `--dry-run` (checked and ledgered, nothing sent). No proxy unless asked (a proxy would see the writes).
* **Every attempt is a ledger line**, refusals included: method, path, redacted body, status, sizes, SHA-256 of the response.

## The learning side

`WebEnv` wraps a `Session` and a menu of actions as a reinforcement-learning environment (the project's usual protocol), so any family that works on environments can learn it. It sees a small vector from the session (class of the last status, which named values it holds, whether it has a cookie, progress through the episode). The reward comes from `goal(session)` plus a per-request cost and an extra cost for refused or failed requests. A refused action costs the learner reward and costs the server nothing: it never arrives.

The practice task (`mock_env`): log in with a password the learner can only *name*, fetch the CSRF value, create an item with both, update its quantity. The mock server also exercises lock-out after bad passwords (429 + `Retry-After`), paging, 422 validation, redirects to itself and away, huge and deeply nested responses, and a `/hostile` endpoint whose values are attacks on the client.

## Where it can and cannot run

* **CLI only.** Neither the organism, the generalist, the patcher nor any hook imports `act.py` (a test checks every one). A person starts it, with a policy, on purpose.
* `act.py` is in the **sealed constitution**: the self-modifier cannot edit the policy it is held to.

## Honest status

Verified against a **real local HTTP server** and fakes: the policy decisions above, the injection attempts in the tests (CRLF in header names and values, `@` and userinfo, absolute URLs and `//host` paths, `..` and `%2e%2e` and double-encoded variants, other schemes, secrets in URLs, body and header sizes), cookies and slots, redirects, ledger redaction, budgets, the kill switch, dry run, and that a refused request never reaches the server.

**Not verified:** any real remote service; TLS behaviour; a hostile DNS server (the address check and the connection are separate steps, so DNS rebinding can defeat the loopback / public test; **the host allow-list is the real control**); HTTP/2; large uploads; proxies. The actor is not thread-safe. The reward is only as honest as the server: for a real service, write `goal` so it checks something the server cannot cheaply fake, or verify with a separate read.
