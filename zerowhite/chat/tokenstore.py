"""tokenstore.py - ingested records -> memory-mapped token shards -> packed training batches.

    state/corpus/<shard>.jsonl.gz  --TokenStore.sync()-->  state/tokens/<shard>.<tokenizer>.<ctx>/
                                                               ids.npy  mask.npy  doc_off.npy  doc_w.npy  doc_u.npy  meta.json
                                                               (+ pref_*.npy for preference records)

* A shard is re-tokenised only when its source shard, the tokenizer or the context length changed.
* Every document gets u in [0,1) from a hash of its record id; documents with u < data.val_fraction are validation.
  Because u never changes, appending data later never moves an old document between train and validation.
* Batches are packed rows of `ctx` positions holding several whole documents; `seg` keeps them from attending to each
  other and `pos` restarts RoPE for each, so packing costs no quality.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .. import fileio
from .config import Config
from .paths import Layout
from .tokenizer import ChatTemplate, Tokenizer

_W: Dict[str, object] = {}


def _u_of(rid: str) -> float:
    return int.from_bytes(hashlib.blake2b(rid.encode("utf-8"), digest_size=8).digest(), "big") / 2.0 ** 64


def _dtype_for(vocab: int):
    return np.uint16 if vocab < 65535 else np.uint32


# ---- worker ---------------------------------------------------------------------------------------------------------
def _init_worker(tok_path: str, cfg: Config):
    tok = Tokenizer.load(tok_path)
    _W["tok"], _W["tpl"], _W["cfg"] = tok, ChatTemplate(tok, cfg), cfg


def _tokenize_shard(job: dict) -> dict:
    tok: Tokenizer = _W["tok"]            # type: ignore[assignment]
    tpl: ChatTemplate = _W["tpl"]         # type: ignore[assignment]
    ctx = job["ctx"]
    max_len = ctx + 1
    ids_parts: List[List[int]] = []
    mask_parts: List[List[int]] = []
    doc_w: List[float] = []
    doc_u: List[float] = []
    pref = dict(p=[], c=[], r=[], w=[], u=[])
    n_rec = 0
    with gzip.open(job["src"], "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_rec += 1
            u, w = _u_of(rec["id"]), float(rec.get("weight", 1.0))
            kind = rec["kind"]
            if kind == "chat":
                for ids, mask in tpl.chunk_chat(rec.get("system"), rec["turns"], max_len):
                    ids_parts.append(ids); mask_parts.append(mask); doc_w.append(w); doc_u.append(u)
            elif kind == "text":
                for ids, mask in tpl.encode_text(rec["text"], max_len):
                    ids_parts.append(ids); mask_parts.append(mask); doc_w.append(w); doc_u.append(u)
            elif kind == "preference":
                p, c = tpl.encode_pair(rec.get("system"), rec["turns"], rec["chosen"], max_len)
                _, r = tpl.encode_pair(rec.get("system"), rec["turns"], rec["rejected"], max_len)
                pref["p"].append(p); pref["c"].append(c); pref["r"].append(r); pref["w"].append(w); pref["u"].append(u)
    out = job["dst"]
    tmp = out + ".building"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    dt = _dtype_for(tok.vocab_size)
    lens = np.array([len(x) for x in ids_parts], dtype=np.int64)
    off = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    ids = np.fromiter((i for part in ids_parts for i in part), dtype=dt, count=int(off[-1])) if len(lens) else np.zeros(0, dt)
    mask = np.fromiter((m for part in mask_parts for m in part), dtype=np.uint8, count=int(off[-1])) if len(lens) else np.zeros(0, np.uint8)
    np.save(os.path.join(tmp, "ids.npy"), ids)
    np.save(os.path.join(tmp, "mask.npy"), mask)
    np.save(os.path.join(tmp, "doc_off.npy"), off)
    np.save(os.path.join(tmp, "doc_w.npy"), np.array(doc_w, dtype=np.float32))
    np.save(os.path.join(tmp, "doc_u.npy"), np.array(doc_u, dtype=np.float64))
    n_pref = len(pref["p"])
    if n_pref:
        for key in ("p", "c", "r"):
            plens = np.array([len(x) for x in pref[key]], dtype=np.int64)
            poff = np.zeros(n_pref + 1, dtype=np.int64)
            np.cumsum(plens, out=poff[1:])
            np.save(os.path.join(tmp, f"pref_{key}_ids.npy"), np.fromiter((i for part in pref[key] for i in part), dtype=dt, count=int(poff[-1])))
            np.save(os.path.join(tmp, f"pref_{key}_off.npy"), poff)
        np.save(os.path.join(tmp, "pref_w.npy"), np.array(pref["w"], dtype=np.float32))
        np.save(os.path.join(tmp, "pref_u.npy"), np.array(pref["u"], dtype=np.float64))
    meta = dict(records=n_rec, docs=len(lens), tokens=int(off[-1]), target_tokens=int(mask.sum()), prefs=n_pref,
                src_sig=job["sig"], built=time.time(), created=job["created"])
    fileio.write_json(os.path.join(tmp, "meta.json"), meta)
    if os.path.isdir(out):
        shutil.rmtree(out)
    fileio.replace(tmp, out)
    return dict(dst=out, **meta)


# ---- the store --------------------------------------------------------------------------------------------------------
class TokenStore:
    def __init__(self, cfg: Config, layout: Layout, tok: Tokenizer, ctx: int):
        self.cfg, self.lay, self.tok, self.ctx = cfg, layout, tok, int(ctx)
        self.fp = tok.fingerprint()
        os.makedirs(layout.tokens_dir, exist_ok=True)

    def dir_for(self, shard_name: str) -> str:
        return os.path.join(self.lay.tokens_dir, f"{shard_name}.{self.fp}.{self.ctx}")

    def sync(self, shards, workers: int = 1, progress=None) -> dict:
        """Tokenise every shard that is missing or stale; delete token dirs of shards that no longer exist."""
        jobs, keep = [], set()
        for s in shards:
            st = os.stat(s.path)
            sig = f"{st.st_size}:{st.st_mtime_ns}"
            dst = self.dir_for(s.name)
            keep.add(os.path.basename(dst))
            mp = os.path.join(dst, "meta.json")
            if os.path.exists(mp):
                try:
                    if fileio.read_json(mp).get("src_sig") == sig:
                        continue
                except (OSError, ValueError):
                    pass
            jobs.append(dict(src=s.path, dst=dst, ctx=self.ctx, sig=sig, created=s.created))
        removed = 0
        for d in os.listdir(self.lay.tokens_dir):
            p = os.path.join(self.lay.tokens_dir, d)
            if os.path.isdir(p) and d not in keep:
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
        built = []
        if jobs:
            tok_path = self.lay.tokenizer_path
            if workers > 1 and len(jobs) > 1:
                with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), initializer=_init_worker,
                                         initargs=(tok_path, self.cfg)) as ex:
                    for r in ex.map(_tokenize_shard, jobs):
                        built.append(r)
                        if progress:
                            progress(r)
            else:
                _init_worker(tok_path, self.cfg)
                for j in jobs:
                    r = _tokenize_shard(j)
                    built.append(r)
                    if progress:
                        progress(r)
        return dict(built=len(built), removed=removed, tokens=sum(b["tokens"] for b in built))

    def token_counts(self, shards, since: Optional[float] = None) -> Tuple[int, int]:
        """(total tokens, tokens in shards ingested after `since`) from the built token shards; shards not tokenised yet are skipped."""
        total = new = 0
        for s in shards:
            mp = os.path.join(self.dir_for(s.name), "meta.json")
            if not os.path.exists(mp):
                continue
            try:
                m = fileio.read_json(mp)
            except (OSError, ValueError):
                continue
            total += int(m.get("tokens", 0))
            if since is None or m.get("created", 0) > since:
                new += int(m.get("tokens", 0))
        return total, new

    def load(self, shards, since: Optional[float] = None, new_boost: float = 1.0) -> "Corpus":
        return Corpus(self.cfg, [self.dir_for(s.name) for s in shards if os.path.exists(os.path.join(self.dir_for(s.name), "meta.json"))],
                      self.ctx, self.tok.vocab_size, since, new_boost)


# ---- the loaded corpus -------------------------------------------------------------------------------------------------
class Corpus:
    def __init__(self, cfg: Config, dirs: List[str], ctx: int, vocab: int, since: Optional[float] = None, new_boost: float = 1.0):
        self.ctx, self.vocab = ctx, vocab
        self.dirs = dirs
        self.ids: List[np.ndarray] = []
        self.mask: List[np.ndarray] = []
        sh, st, ln, wt, uu = [], [], [], [], []
        self.pref: List[dict] = []
        self.new_tokens = 0
        for si, d in enumerate(dirs):
            meta = fileio.read_json(os.path.join(d, "meta.json"))
            self.ids.append(np.load(os.path.join(d, "ids.npy"), mmap_mode="r"))
            self.mask.append(np.load(os.path.join(d, "mask.npy"), mmap_mode="r"))
            off = np.load(os.path.join(d, "doc_off.npy"))
            w = np.load(os.path.join(d, "doc_w.npy")).astype(np.float64)
            u = np.load(os.path.join(d, "doc_u.npy"))
            if since is not None and meta.get("created", 0) > since:
                w = w * new_boost
                self.new_tokens += int(meta["tokens"])
            n = len(off) - 1
            sh.append(np.full(n, si, dtype=np.int32)); st.append(off[:-1]); ln.append(np.diff(off).astype(np.int32)); wt.append(w); uu.append(u)
            if meta.get("prefs"):
                self.pref.append({k: np.load(os.path.join(d, f"pref_{k}.npy"), mmap_mode="r")
                                  for k in ("p_ids", "p_off", "c_ids", "c_off", "r_ids", "r_off", "w", "u")})
        cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)                    # noqa: E731
        self.d_shard, self.d_start, self.d_len = cat(sh, np.int32), cat(st, np.int64), cat(ln, np.int32)
        self.d_w, self.d_u = cat(wt, np.float64), cat(uu, np.float64)
        self.val_fraction = float(cfg.get("data.val_fraction"))
        min_docs = int(cfg.get("data.val_min_docs"))
        n = len(self.d_u)
        val = self.d_u < self.val_fraction
        # small corpora: hold out at least `val_min_docs` documents, but never more than a fifth of everything
        want = min(min_docs, max(1, n // 5)) if n else 0
        if int(val.sum()) < want:
            val = np.zeros(n, dtype=bool)
            val[np.argsort(self.d_u, kind="stable")[:want]] = True
        self.d_val = val
        self.train_idx = np.flatnonzero(~val)
        self.val_idx = np.flatnonzero(val)

    # ---- stats ------------------------------------------------------------------------------------------------------
    def counts(self) -> dict:
        tl = self.d_len
        tgt = np.array([int(self.mask[s][a:a + l].sum()) for s, a, l in zip(self.d_shard[self.train_idx[:20000]],
                                                                          self.d_start[self.train_idx[:20000]], tl[self.train_idx[:20000]])]) if len(self.train_idx) else np.zeros(0)
        return dict(docs=int(len(tl)), train_docs=int(len(self.train_idx)), val_docs=int(len(self.val_idx)),
                    train_tokens=int(tl[self.train_idx].sum()), val_tokens=int(tl[self.val_idx].sum()),
                    prefs=int(sum(len(p["w"]) for p in self.pref)), mean_doc_len=float(tl.mean()) if len(tl) else 0.0,
                    target_fraction=float(tgt.sum() / max(1, tl[self.train_idx[:20000]].sum())) if len(tgt) else 0.0)

    def doc(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        s, a, l = int(self.d_shard[i]), int(self.d_start[i]), int(self.d_len[i])
        return np.asarray(self.ids[s][a:a + l]), np.asarray(self.mask[s][a:a + l])

    def sample_text(self, n: int = 2000, seed: int = 0) -> Iterator[int]:
        rng = np.random.default_rng(seed)
        for i in rng.choice(self.train_idx, size=min(n, len(self.train_idx)), replace=False):
            yield int(i)

    def pref_pairs(self, val: bool = False) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        out = []
        for p in self.pref:
            n = len(p["w"])
            for j in range(n):
                is_val = float(p["u"][j]) < self.val_fraction
                if is_val != val:
                    continue
                out.append((np.asarray(p["p_ids"][p["p_off"][j]:p["p_off"][j + 1]]), np.asarray(p["c_ids"][p["c_off"][j]:p["c_off"][j + 1]]),
                            np.asarray(p["r_ids"][p["r_off"][j]:p["r_off"][j + 1]]), float(p["w"][j])))
        return out


# ---- packing ----------------------------------------------------------------------------------------------------------
class Packer:
    """Deterministic, resumable stream of packed training batches.
    state() / set_state() capture (epoch, cursor), so a resumed run continues the exact stream."""

    LOOKAHEAD = 8

    def __init__(self, corpus: Corpus, ctx: int, pad_id: int, seed: int = 0, split: str = "train", once: bool = False):
        self.c, self.ctx, self.pad, self.seed, self.split, self.once = corpus, ctx, pad_id, seed, split, once
        self.idx = corpus.train_idx if split == "train" else corpus.val_idx
        self.epoch, self.cursor = 0, 0
        self._order: Optional[np.ndarray] = None
        self._buf: List[int] = []
        self._exhausted = False
        self.docs_seen = 0

    # ---- ordering ---------------------------------------------------------------------------------------------------
    def _make_order(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed * 1_000_003 + self.epoch)
        w = self.c.d_w[self.idx]
        reps = np.floor(w).astype(np.int64) + (rng.random(len(w)) < (w - np.floor(w)))
        order = np.repeat(self.idx, reps)
        if self.split == "train":
            rng.shuffle(order)
        return order

    def _next_doc(self) -> int:
        """Next document id; -1 when `once` and the single pass is over."""
        if self.once and self._order is not None and self.cursor >= len(self._order):
            return -1
        if self._order is None or self.cursor >= len(self._order):
            if self._order is not None:
                self.epoch += 1
                self.cursor = 0
            self._order = self._make_order()
            if len(self._order) == 0:
                raise RuntimeError("no documents to train on")
        d = int(self._order[self.cursor])
        self.cursor += 1
        return d

    def state(self) -> dict:
        return dict(epoch=self.epoch, cursor=self.cursor, buf=list(self._buf), seed=self.seed)

    def set_state(self, st: dict):
        self.epoch, self.cursor = int(st["epoch"]), int(st["cursor"])
        self._buf = [int(x) for x in st.get("buf", [])]
        self._order = None
        if self.cursor:
            self._order = self._make_order()
            self.cursor = min(self.cursor, len(self._order))

    @property
    def epochs_done(self) -> float:
        n = len(self._order) if self._order is not None else 1
        return self.epoch + self.cursor / max(1, n)

    # ---- one batch --------------------------------------------------------------------------------------------------
    def next_batch(self, rows: int) -> dict:
        T = self.ctx
        ids = np.full((rows, T), self.pad, dtype=np.int64)
        tgt = np.zeros((rows, T), dtype=np.int64)
        w = np.zeros((rows, T), dtype=np.float32)
        pos = np.zeros((rows, T), dtype=np.int64)
        seg = np.zeros((rows, T), dtype=np.int64)
        for r in range(rows):
            a, k = 0, 0
            while True:
                while len(self._buf) < self.LOOKAHEAD and not self._exhausted:
                    d = self._next_doc()
                    if d < 0:
                        self._exhausted = True
                    else:
                        self._buf.append(d)
                pick = None
                for j, d in enumerate(self._buf):
                    if int(self.c.d_len[d]) - 1 <= T - a:
                        pick = j
                        break
                if pick is None:
                    break
                d = self._buf.pop(pick)
                toks, m = self.c.doc(d)
                L = len(toks) - 1
                if L < 1:
                    continue
                k += 1
                ids[r, a:a + L] = toks[:-1]
                tgt[r, a:a + L] = toks[1:]
                w[r, a:a + L] = m[1:]
                pos[r, a:a + L] = np.arange(L)
                seg[r, a:a + L] = k
                a += L
                self.docs_seen += 1
                if a >= T - 2:
                    break
        return dict(ids=ids, tgt=tgt, w=w, pos=pos, seg=seg)

    def fixed_batches(self, rows: int, max_target_tokens: int) -> List[dict]:
        """One reproducible pass over the split (never repeats a document), stopping after ~max_target_tokens."""
        p = Packer(self.c, self.ctx, self.pad, 0, self.split, once=True)
        out, seen = [], 0
        while seen < max_target_tokens:
            b = p.next_batch(rows)
            n = int((b["w"] != 0).sum())
            if n == 0:
                break
            out.append(b)
            seen += n
        return out
