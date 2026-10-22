"""curiosity.py - decide WHAT of the allow-listed reading is worth keeping: pages whose content is new to the system.

The forager (fetch.py) reads URLs a person listed from hosts a person allowed. Reading everything on a wishlist would mostly re-teach what the system already holds, so this
module adds the curiosity rule: a fetched page is kept only if enough of its word 3-grams have never been seen in the reference text (the corpus the system already has plus
what it kept earlier). A near-duplicate scores near 0 and is dropped; unfamiliar material scores near 1 and is kept; kept pages count as seen, so a second page about the same
new thing is less novel than the first.

WHAT THIS IS, AND IS NOT: novelty is lexical (shared phrasing), not semantic: a paraphrase of something known looks new, and a short page can be 'new' because it is short. It
measures 'have I read these words in this order', which is a crude proxy for 'is there a knowledge gap here'. It does not look at the model's loss, does not choose URLs (a person
wrote the wishlist) and cannot reach any host outside the allow-list; and the organism only uses it when every task has plateaued and the opt-in flags are set.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import zlib

from .. import fileio

_WORD = re.compile(r"[a-z0-9']+")


def grams(text: str, n: int = 3):
    ws = _WORD.findall(text.lower())
    return {zlib.crc32(" ".join(ws[i:i + n]).encode("utf-8")) for i in range(max(0, len(ws) - n + 1))}


class Novelty:
    def __init__(self, n: int = 3):
        self.n, self.seen = n, set()

    def add(self, text: str):
        self.seen |= grams(text, self.n)

    def score(self, text: str) -> float:
        g = grams(text, self.n)
        if not g:
            return 0.0
        return len(g - self.seen) / len(g)


def load_reference(dirs, max_bytes: int = 20 * 2 ** 20) -> Novelty:
    nov, total = Novelty(), 0
    for d in dirs:
        for root, _, files in os.walk(d or ""):
            for f in sorted(files):
                if f.lower().endswith((".txt", ".md", ".rst")):
                    try:
                        t = fileio.read_text(os.path.join(root, f))
                    except (OSError, ValueError):
                        continue
                    total += len(t)
                    if total > max_bytes:
                        return nov
                    nov.add(t)
    return nov


def read_wishlist(path: str, state_path: str, limit: int):
    """The next `limit` URLs of the wishlist after the ones already attempted (a pointer in `state_path`, so a pass resumes where the last stopped)."""
    if not path or not os.path.exists(path):
        return [], 0
    with open(path, "r", encoding="utf-8") as f:
        urls = [ln.split("\t")[0].strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    pos = 0
    if os.path.exists(state_path):
        try:
            pos = int(fileio.read_json(state_path).get("pos", 0))
        except (OSError, ValueError):
            pos = 0
    return urls[pos:pos + limit], pos


def curious_pass(forager, urls, novelty: Novelty, final_dir: str, threshold: float = 0.4) -> list:
    """Fetch `urls` through `forager` (which writes to its own staging folder) and keep, in `final_dir`, only the pages whose novelty is >= threshold."""
    rows = []
    os.makedirs(final_dir, exist_ok=True)
    for r in forager.fetch_all(urls):
        r = dict(r, novelty=None, kept=False)
        if r["ok"] and r.get("path") and os.path.exists(r["path"]):
            text = fileio.read_text(r["path"])
            nov = novelty.score(text)
            r["novelty"] = round(nov, 3)
            if nov >= threshold:
                dst = os.path.join(final_dir, os.path.basename(r["path"]))
                shutil.move(r["path"], dst)
                novelty.add(text)
                r.update(kept=True, path=dst, reason=f"kept: {nov:.0%} of its 3-grams are new")
            else:
                os.remove(r["path"])
                r.update(reason=f"dropped: only {nov:.0%} of its 3-grams are new (threshold {threshold:.0%})")
        rows.append(r)
    return rows
