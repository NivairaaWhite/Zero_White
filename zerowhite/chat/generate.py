"""generate.py - fast text generation: KV-cache decoding, sampling, and the Engine that owns the live model.

* forward_cached() runs only the new tokens through the network and keeps every layer's keys/values, so a reply of N tokens
  costs N small steps instead of N full forward passes. It is verified against the training forward pass in the tests.
* A Generator keeps its cache between turns of one conversation: when the next prompt starts with the tokens already cached
  (the usual case), only the new user message is processed.
* Engine = tokenizer + chat template + the live model. `reload_if_changed()` picks up newly trained weights, `swap()` is how the
  online learner publishes an update; running replies finish on the model they started with.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Iterator, List, Optional, Sequence

import numpy as np

from .config import AUTO, Config
from .model import GPT, GPTConfig, _moe_forward, _rms, _rope
from .paths import Layout
from .tokenizer import ChatTemplate, Tokenizer


class KVCache:
    def __init__(self, cfg: GPTConfig):
        shape = (cfg.n_layer, cfg.n_head, cfg.ctx, cfg.head_dim)
        self.K = np.zeros(shape, dtype=np.dtype(cfg.dtype))
        self.V = np.zeros(shape, dtype=np.dtype(cfg.dtype))
        self.n = 0


def forward_cached(model: GPT, cache: KVCache, ids: Sequence[int]) -> np.ndarray:
    """Feed `ids` after the `cache.n` tokens already cached; returns the logits (vocab,) for the last fed token."""
    c, p = model.cfg, model.p
    ids = np.asarray(ids, dtype=np.int64)
    n = len(ids)
    start, total = cache.n, cache.n + n
    if n == 0 or total > c.ctx:
        raise ValueError(f"cannot feed {n} tokens after {start} cached ones (context window {c.ctx})")
    H, hd, D = c.n_head, c.head_dim, c.d_model
    cos_t, sin_t = model.rope_tables(total)
    cos, sin = cos_t[start:total], sin_t[start:total]
    scale = 1.0 / np.sqrt(hd)
    x = p["tok_emb"][ids]
    mask = None
    if n > 1:
        mask = np.arange(total)[None, :] > (start + np.arange(n))[:, None]          # key j is in the future of query i
    for l in range(c.n_layer):
        pre = f"b{l}."
        n1, _, _ = _rms(x, p[pre + "n1"], c.eps)
        q = _rope((n1 @ p[pre + "wq"]).reshape(n, H, hd).transpose(1, 0, 2), cos, sin)
        k = _rope((n1 @ p[pre + "wk"]).reshape(n, H, hd).transpose(1, 0, 2), cos, sin)
        v = (n1 @ p[pre + "wv"]).reshape(n, H, hd).transpose(1, 0, 2)
        cache.K[l, :, start:total] = k
        cache.V[l, :, start:total] = v
        s = np.matmul(q, cache.K[l, :, :total].transpose(0, 2, 1)) * scale
        if mask is not None:
            s = np.where(mask[None], -1e30, s)
        s -= s.max(axis=-1, keepdims=True)
        np.exp(s, out=s)
        s /= s.sum(axis=-1, keepdims=True)
        ao = np.matmul(s, cache.V[l, :, :total]).transpose(1, 0, 2).reshape(n, D)
        x = x + ao @ p[pre + "wo"]
        n2, _, _ = _rms(x, p[pre + "n2"], c.eps)
        if c.n_experts > 1:
            x = x + _moe_forward(n2, p, pre, c, np, False)[0]
        else:
            a = n2 @ p[pre + "wg"]
            h = (a / (1.0 + np.exp(-a))) * (n2 @ p[pre + "wu"])
            x = x + h @ p[pre + "wd"]
    cache.n = total
    xf, _, _ = _rms(x[-1:], p["nf"], c.eps)
    E = p["tok_emb"] if c.tie else p["lm_head"]
    return (xf[0] @ E.T).astype(np.float64)


def sample_token(logits: np.ndarray, history: Sequence[int], rng: np.random.Generator, temperature: float, top_k: int, top_p: float,
                 rep_penalty: float, rep_window: int, banned: Sequence[int] = ()) -> int:
    lg = logits.copy()
    if rep_penalty and rep_penalty != 1.0 and len(history):
        recent = np.unique(np.asarray(history[-rep_window:] if rep_window else history, dtype=np.int64))
        v = lg[recent]
        lg[recent] = np.where(v > 0, v / rep_penalty, v * rep_penalty)
    if len(banned):
        lg[np.asarray(banned, dtype=np.int64)] = -np.inf
    if temperature <= 0.0:
        return int(np.argmax(lg))
    lg /= temperature
    if top_k and 0 < top_k < lg.size:
        kth = np.partition(lg, -top_k)[-top_k]
        lg[lg < kth] = -np.inf
    lg -= lg.max()
    pr = np.exp(lg)
    pr /= pr.sum()
    if 0.0 < top_p < 1.0:
        order = np.argsort(-pr)
        cum = np.cumsum(pr[order])
        keep = order[: int(np.searchsorted(cum, top_p)) + 1]
        m = np.zeros_like(pr)
        m[keep] = pr[keep]
        pr = m / m.sum()
    return int(rng.choice(pr.size, p=pr))


class Generator:
    """Generation state of ONE conversation (its KV cache). Create one per chat session."""

    def __init__(self):
        self.cache: Optional[KVCache] = None
        self.cached: List[int] = []
        self._model_id: Optional[int] = None

    def reset(self):
        self.cached = []
        if self.cache:
            self.cache.n = 0

    def stream_ids(self, model: GPT, prompt: Sequence[int], max_new: int, end_id: int, banned: Sequence[int], rng: np.random.Generator,
                   temperature: float, top_k: int, top_p: float, rep_penalty: float, rep_window: int) -> Iterator[int]:
        c = model.cfg
        if self.cache is None or self._model_id != id(model) or self.cache.K.shape[2] != c.ctx:
            self.cache, self.cached, self._model_id = KVCache(c), [], id(model)
        prompt = list(prompt)
        if len(prompt) >= c.ctx:
            raise ValueError(f"prompt of {len(prompt)} tokens does not fit the {c.ctx}-token window")
        lcp = 0
        for a, b in zip(self.cached, prompt):
            if a != b:
                break
            lcp += 1
        lcp = min(lcp, len(prompt) - 1)                         # always feed at least one token to get logits
        self.cache.n = lcp
        self.cached = prompt[:lcp]
        logits = forward_cached(model, self.cache, prompt[lcp:])
        self.cached = list(prompt)
        history = list(prompt)
        for _ in range(max(0, min(max_new, c.ctx - len(prompt)))):
            t = sample_token(logits, history, rng, temperature, top_k, top_p, rep_penalty, rep_window, banned)
            yield t
            if t == end_id:
                return
            history.append(t)
            self.cached.append(t)
            if self.cache.n + 1 > c.ctx:
                return
            logits = forward_cached(model, self.cache, [t])


class Engine:
    """Tokenizer + chat template + live model, with hot reload."""

    def __init__(self, cfg: Config, lay: Layout):
        self.cfg, self.lay = cfg, lay
        self.tok: Optional[Tokenizer] = None
        self.tpl: Optional[ChatTemplate] = None
        self.model: Optional[GPT] = None
        self.meta: dict = {}
        self.version: Optional[str] = None
        self.loaded_at = 0.0
        self._lock = threading.RLock()               # guards swapping
        self.gen_lock = threading.Lock()             # one generation at a time (CPU bound)

    # ---- loading --------------------------------------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> bool:
        """Load the tokenizer and the live model. Returns False when there is no trained model yet."""
        path = self.lay.current_weights()
        if not path or not os.path.exists(self.lay.tokenizer_path):
            return False
        with self._lock:
            tok = Tokenizer.load(self.lay.tokenizer_path)
            model, meta = GPT.load(path)
            if model.cfg.vocab != tok.vocab_size:
                raise ValueError(f"model vocab {model.cfg.vocab} != tokenizer vocab {tok.vocab_size}")
            self.tok, self.tpl = tok, ChatTemplate(tok, self.cfg)
            self.model, self.meta, self.version, self.loaded_at = model, meta, os.path.basename(path), time.time()
        return True

    def reload_if_changed(self) -> bool:
        cur = self.lay.current()
        if cur and cur.get("file") != self.version:
            return self.load()
        return False

    def swap(self, model: GPT, meta: Optional[dict] = None, version: Optional[str] = None):
        with self._lock:
            self.model = model
            if meta is not None:
                self.meta = meta
            if version:
                self.version = version
            self.loaded_at = time.time()

    # ---- prompts --------------------------------------------------------------------------------------------------
    def gen_params(self, **over) -> dict:
        g = self.cfg
        ctx = self.model.cfg.ctx
        mx = g.get("generation.max_new_tokens", None)
        d = dict(max_new_tokens=int(mx) if mx not in (None, AUTO) else max(16, ctx // 4),
                 temperature=float(g.get("generation.temperature")), top_k=int(g.get("generation.top_k")),
                 top_p=float(g.get("generation.top_p")), repetition_penalty=float(g.get("generation.repetition_penalty")),
                 repetition_window=int(g.get("generation.repetition_window")), stop=list(g.get("generation.stop")),
                 seed=g.get("generation.seed"))
        d.update({k: v for k, v in over.items() if v is not None})
        return d

    def prompt_for(self, system: Optional[str], turns: Sequence[dict], max_new: int) -> List[int]:
        ht = self.cfg.get("generation.history_turns", None)
        if ht not in (None, AUTO):
            turns = list(turns)[-int(ht):]
        return self.tpl.prompt_ids(system, turns, self.model.cfg.ctx - max_new)

    # ---- replies --------------------------------------------------------------------------------------------------
    def stream(self, gen: Generator, system: Optional[str], turns: Sequence[dict], **over) -> Iterator[str]:
        """Yield the assistant's reply as text pieces. `turns` must end with the user message being answered."""
        if not self.ready:
            raise RuntimeError("no trained model yet: run `zerowhite chat train` first")
        with self._lock:
            model, tok, tpl = self.model, self.tok, self.tpl
        p = self.gen_params(**over)
        max_new = max(1, min(int(p["max_new_tokens"]), model.cfg.ctx // 2))
        prompt = self.prompt_for(system, turns, max_new)
        banned = [i for i in range(tok.offset) if i != tpl.end_id]
        rng = np.random.default_rng(p["seed"])
        stops = [s for s in p["stop"] if s]
        hold = max((len(s) for s in stops), default=1) - 1
        dec = tok.stream_decoder()
        text, sent = "", 0
        with self.gen_lock:
            for t in gen.stream_ids(model, prompt, max_new, tpl.end_id, banned, rng, p["temperature"], p["top_k"], p["top_p"],
                                    p["repetition_penalty"], p["repetition_window"]):
                if t == tpl.end_id:
                    break
                text += dec.feed([t])
                cut = min((text.find(s) for s in stops if s in text), default=-1)
                if cut >= 0:
                    text = text[:cut]
                    yield text[sent:]
                    return
                safe = len(text) - hold
                if safe > sent:
                    yield text[sent:safe]
                    sent = safe
            text += dec.flush()
        if len(text) > sent:
            yield text[sent:]

    def reply(self, gen: Generator, system: Optional[str], turns: Sequence[dict], **over) -> str:
        return "".join(self.stream(gen, system, turns, **over)).strip()
