# Forage (6.3): reading a few web pages, on your terms

```bash
python -m zerowhite forage --allow-host docs.python.org --allow-host numpy.org \
    https://docs.python.org/3/library/statistics.html https://numpy.org/doc/stable/user/basics.broadcasting.html
python -m zerowhite forage --allow-host example.org --urls-file reading-list.txt     # one URL per line, # comments
python -m zerowhite organism --inbox zw/forage/inbox ...                               # let the organism digest what was saved
```

Until 6.3 nothing in Zero White could read a web page: the only network code was the dataset downloaders and the local service, and a test confines `urllib` / `socket` to named files. `forage` is one more named file
(`forage/fetch.py`, listed in that test with its reason) and it is deliberately narrow, because a system that rewrites parts of itself should not also decide for itself where to read from.

**Off unless you act.** With no `--allow-host` it exits with an explanation and does nothing. There are no default hosts, no search, no link-following. The chat service and the self-modifier never call it. Since 6.4 the organism can, **only if you pass both `--forage-host H` (repeatable) and `--forage-wishlist FILE`**, and only when every task has plateaued (below);
otherwise you run it yourself and the organism reads the result the way it reads any inbox. A test fails if any module other than `cli_v6.py` and `organism.py` imports it, and `organism.py` may import it only inside `_act_forage`.

**What the code enforces** (each point has a test, several against a real local HTTP server):

* every URL and every redirect hop must be on your allow-list (redirects are followed by hand, at most 3, so a redirect to another host is refused *before* it is requested);
* https only (plain http for loopback hosts you listed, or with `--allow-http`); no credentials in URLs; GET only; no cookies; one fixed User-Agent;
* an allow-listed host that resolves to a private / loopback / link-local address is refused unless you pass `--allow-local`;
* `robots.txt` is read first and honoured (a 5xx or a timeout on it means "do not fetch"; `--no-robots` exists and is discouraged), at least `--delay` seconds (2) between requests to a host;
* limits per page (`--max-mb`, 2), per run (`--max-urls`, 20; 20 MB in total);
* only text-like content types are kept; HTML is reduced to its words by the standard-library parser (no JavaScript runs, nothing is executed or imported), pages with under 80 characters of text are dropped;
* output: one `.txt` per page in `zw/forage/inbox/` (identical text is stored once) and **one ledger line per attempt, refusals included**, in `zw/forage/ledger.jsonl` (URL, time, status, bytes, SHA-256, verdict).

## Curiosity (6.4): the organism decides WHEN, and what to KEEP

```bash
python -m zerowhite organism --inbox inbox --forage-host docs.python.org --forage-host numpy.org --forage-wishlist reading-list.txt
```
When every task has plateaued (or there is none), no more than once per `--forage-cooldown` (3600 s), the organism reads the next `--forage-max-urls` (5) URLs of YOUR wishlist through exactly the forager above (same
allow-list, robots, caps, ledger) into a staging folder. A page is then **kept in `zw/forage/inbox/` only if at least `--forage-min-novelty` (40%) of its word 3-grams have never been seen** in what the system already holds (the
folder itself plus the chat data folder); near-duplicates are deleted and the ledger says why. A kept page counts as seen, so a second page about the same new thing is less novel. A pointer in
`zw/forage/wishlist.state.json` makes the next pass continue where this one stopped; an exhausted list does nothing.

What this is: a *curiosity filter*, not a gap detector. Novelty is lexical (shared phrasing), so a paraphrase of something known looks new and a short page can look new because it is short; it ignores the model's loss. The organism
chooses WHEN and what to KEEP; a person still wrote the list and the allow-list. Tested end to end against a local HTTP server (a novel page is kept, a known one dropped, the wishlist resumes, robots.txt is asked first); never against the public internet.

**What it does not do.** It does not choose URLs or search: a person picks them. Text from the web is still untrusted text that will enter
a training corpus through the usual filters; do not allow hosts you would not train on.

**Honest limits.** The address check and the connection are separate steps, so a hostile DNS server could answer differently in between (DNS rebinding); the allow-list is the real control. Proxies configured in the environment are
honoured by `urllib`. The real-network path (TLS to a public host) is exercised only against a local plain-http server in the tests, because the build environment had no network. `forage/fetch.py` is part of the sealed constitution.

## Writing to the web: `forage/act.py` (6.5)

`fetch.py` only reads. Sending data (POST / PUT / PATCH, logging in, holding a token) lives in a separate sealed module, `forage/act.py`, with its own deny-by-default policy, ledger, budgets and kill file. It is **CLI only**: neither the organism nor the self-modifier imports it (the guard test for this file's rule, `test_only_the_forager_and_its_cli_command_know_about_the_forager_and_it_is_sealed`, allows the 6.5 CLI module that hosts the `act` commands and nothing else). See [ACT.md](ACT.md).
