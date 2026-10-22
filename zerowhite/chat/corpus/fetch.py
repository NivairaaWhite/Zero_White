"""fetch.py - download Hugging Face datasets into the data folder with nothing but the standard library.

This is the ONLY module of the chat stack allowed to talk to the network (a test enforces that). It uses the public
datasets-server JSON API:
    GET {fetch.endpoint}/splits?dataset=ID
    GET {fetch.endpoint}/rows?dataset=ID&config=C&split=S&offset=O&length=L
and appends the rows as JSON lines to  <data>/<folder>/<name>.jsonl , resuming where an interrupted download stopped.

Honesty notes: the dev sandbox that produced this code cannot reach huggingface.co, so the HTTP layer is tested against a
local mock of those two endpoints, not against the live service. Datasets that are script-based, gated, audio-only or too
large for the rows API may fail; the error message says what happened and what to do instead.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, Optional

from ... import fileio
from ..config import Config
from ..paths import Layout, safe_name

USER_AGENT = "zerowhite-fetch/1 (+local dataset downloader)"


class FetchError(RuntimeError):
    pass


def load_catalog(cfg: Config) -> Dict[str, dict]:
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog.json")
    cat = {k: v for k, v in fileio.read_json(here).items() if not k.startswith("_")}
    user = cfg.get("fetch.catalog")
    if user:
        p = cfg.resolve_path(user)
        if not os.path.isfile(p):
            raise FetchError(f"fetch.catalog file not found: {p}")
        cat.update({k: v for k, v in fileio.read_json(p).items() if not k.startswith("_")})
    return cat


def _get_json(cfg: Config, url: str, token: Optional[str]) -> dict:
    retries = int(cfg.get("fetch.retries"))
    timeout = float(cfg.get("fetch.timeout_s"))
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            if e.code in (401, 403):
                raise FetchError(
                    f"HTTP {e.code}: this dataset is gated or private. Accept its licence on its Hugging Face page, create an access "
                    f"token, and set it in the environment variable {cfg.get('fetch.token_env')} (then run the command again). {body}") from None
            if e.code == 404:
                raise FetchError(f"HTTP 404 for {url.split('?')[0]}: the dataset id/config/split was not found, or the dataset is not "
                                 f"supported by the rows API (script-based, or too large). Check the id, pass --config/--split, or "
                                 f"download the files by hand into the data folder. {body}") from None
            if e.code == 429 or e.code >= 500:
                last = f"HTTP {e.code}"
                ra = e.headers.get("Retry-After") if e.headers else None
                delay = float(ra) if ra and ra.replace(".", "", 1).isdigit() else min(60.0, 1.5 * 2 ** attempt)
            else:
                raise FetchError(f"HTTP {e.code} from {url.split('?')[0]}: {body}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as e:
            last = f"{type(e).__name__}: {e}"
            delay = min(60.0, 1.5 * 2 ** attempt)
        if attempt < retries:
            time.sleep(delay * (0.75 + 0.5 * random.random()))
    raise FetchError(f"giving up after {retries + 1} attempts ({last}). Check your connection / proxy settings, or retry later "
                     f"(partial downloads are resumed).")


def _strip_audio(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "src" in v[0]:
            out[k + "_url"] = v[0]["src"]
        elif isinstance(v, dict) and "src" in v:
            out[k + "_url"] = v["src"]
        else:
            out[k] = v
    return out


def _count_complete_lines(path: str) -> int:
    """Number of complete JSON lines already in `path`; a torn last line is cut off so appending stays valid."""
    if not os.path.exists(path):
        return 0
    n, good_end = 0, 0
    with open(path, "rb") as f:
        pos = 0
        for line in f:
            pos += len(line)
            if line.endswith(b"\n"):
                n += 1
                good_end = pos
    if good_end != os.path.getsize(path):
        with open(path, "r+b") as f:
            f.truncate(good_end)
    return n


def resolve(cfg: Config, name: str, repo: Optional[str], config: Optional[str], split: Optional[str]) -> dict:
    cat = load_catalog(cfg)
    entry = dict(cat.get(name, {}))
    if repo:
        entry["repo"] = repo
    elif "repo" not in entry:
        if "/" in name:
            entry["repo"] = name
        else:
            raise FetchError(f"unknown dataset '{name}'. Known: {', '.join(sorted(cat))}. Or pass owner/name, or --repo owner/name.")
    if config:
        entry["config"] = config
    if split:
        entry["split"] = split
    entry.setdefault("folder", "fetched")
    entry.setdefault("kind", "chat")
    return entry


def fetch_dataset(cfg: Config, layout: Layout, name: str, limit: Optional[int] = None, config: Optional[str] = None,
                  split: Optional[str] = None, repo: Optional[str] = None,
                  progress: Optional[Callable[[dict], None]] = None) -> str:
    """Download up to `limit` rows (default fetch.default_limit) of a dataset. Returns the path of the .jsonl file."""
    entry = resolve(cfg, name, repo, config, split)
    if entry.get("usable") is False:
        raise FetchError(f"'{name}' cannot be fetched through the rows API: {entry.get('note', 'see docs/DATA.md')}")
    ds = entry["repo"]
    endpoint = cfg.get("fetch.endpoint").rstrip("/")
    token = os.environ.get(cfg.get("fetch.token_env")) or None
    limit = int(limit or cfg.get("fetch.default_limit"))
    page = max(1, min(100, int(cfg.get("fetch.page_size"))))

    cfg_name, split_name = entry.get("config"), entry.get("split")
    if not cfg_name or not split_name:
        sp = _get_json(cfg, f"{endpoint}/splits?dataset={urllib.parse.quote(ds, safe='/')}", token).get("splits", [])
        if not sp:
            raise FetchError(f"{ds}: the rows API reports no splits (dataset not viewable through it).")
        pick = [s for s in sp if (not cfg_name or s.get("config") == cfg_name) and (not split_name or s.get("split") == split_name)]
        if not pick:
            raise FetchError(f"{ds}: no split matches config={cfg_name} split={split_name}. Available: "
                             + ", ".join(f"{s.get('config')}/{s.get('split')}" for s in sp[:20]))
        pref = [s for s in pick if s.get("split") == "train"] or pick
        cfg_name, split_name = pref[0]["config"], pref[0]["split"]
        if progress and len({s.get("config") for s in sp}) > 1:
            progress(dict(event="note", message=f"{ds} has several configs ({sorted({s.get('config') for s in sp})}); using '{cfg_name}'. "
                                                f"Pass --subset to choose another."))

    folder = os.path.join(layout.data, entry["folder"])
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, safe_name(name if "/" not in name else name.replace("/", "__")) + ".jsonl")
    offset = _count_complete_lines(out_path)
    total = None
    if progress and offset:
        progress(dict(event="resume", offset=offset))
    truncated = 0
    with open(out_path, "a", encoding="utf-8", newline="\n") as out:
        while offset < limit:
            length = min(page, limit - offset)
            q = urllib.parse.urlencode(dict(dataset=ds, config=cfg_name, split=split_name, offset=offset, length=length))
            data = _get_json(cfg, f"{endpoint}/rows?{q}", token)
            rows = data.get("rows", [])
            total = data.get("num_rows_total", total)
            if not rows:
                break
            buf = []
            for r in rows:
                if r.get("truncated_cells"):
                    truncated += 1
                buf.append(json.dumps(_strip_audio(r.get("row", {})), ensure_ascii=False))
            out.write("\n".join(buf) + "\n")
            out.flush()
            offset += len(rows)
            if progress:
                progress(dict(event="page", offset=offset, limit=min(limit, total or limit), total=total))
            if total is not None and offset >= total:
                break
    if truncated and progress:
        progress(dict(event="note", message=f"{truncated} rows had cells truncated by the API (very long texts)."))
    side = os.path.join(folder, "dataset.json")
    hint = {k: entry[k] for k in ("adapter",) if k in entry}
    if entry.get("kind") in ("text",):
        hint["kind"] = "text"
    if not os.path.exists(side) and hint:
        fileio.write_json(side, hint, indent=2)
    if entry.get("note") and progress:
        progress(dict(event="note", message=entry["note"]))
    if progress and entry.get("verified") is False:
        progress(dict(event="note", message=f"'{name}' ({ds}) is not verified against the live hub; if this failed, check the id."))
    return out_path
