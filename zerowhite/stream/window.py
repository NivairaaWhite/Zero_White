"""window.py - a language model that reads (and writes) a stream LONGER than its context window.

The chat model has a fixed window `cfg.ctx`; its KV cache holds that many positions and `forward_cached` refuses to go past it. A live stream never ends, so the window has to
slide. Two ways, and the difference matters:

  mode="reprefill"  (exact)   when the window is full, drop the oldest `stride` tokens (never the first `sink` ones) and run the kept tokens through the network again from
                              position 0. The cache then holds EXACTLY what a fresh forward pass over the current window would produce (tested to 1e-9). The price is one
                              re-run of the window per `stride` new tokens, so a larger stride is cheaper (cost per token ~ window / stride extra token-steps).
  mode="rotate"     (cheap)   slide the cache itself: delete the dropped positions' keys/values and turn every kept key back by the number of dropped positions. That is exact
                              for rotary embeddings AT THE FIRST LAYER (a key is a rotation by its position; rotating by -d turns position j into j-d - tested), but a deeper
                              layer's keys/values were computed from hidden states that had attended to the dropped tokens, so they are stale by the dropped context.
                              Cheap and close, NOT the same as a fresh pass (the tests measure the difference; docs/STREAM.md says when to prefer which).

`sink` keeps the first tokens (a system prompt, a stream header) in the window for ever, the 'attention sink' trick of streaming LMs.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence

import numpy as np

from ..chat.generate import KVCache, forward_cached, sample_token
from ..chat.model import GPT, _rope_back


class WindowLM:
    MODES = ("reprefill", "rotate")

    def __init__(self, model: GPT, window: Optional[int] = None, stride: Optional[int] = None, mode: str = "reprefill", sink: int = 0):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        c = model.cfg
        self.model, self.mode = model, mode
        self.window = int(min(window or c.ctx, c.ctx))
        self.sink = int(sink)
        self.stride = int(stride or max(1, (self.window - self.sink) // 4))
        if self.sink < 0 or self.window - self.sink - self.stride < 1:
            raise ValueError(f"window {self.window}, sink {self.sink} and stride {self.stride} leave no room for new tokens")
        self.cache = KVCache(c)
        self.tokens: List[int] = []                                       # the tokens the cache currently represents (the window)
        self.seen = 0                                                      # all tokens ever fed
        self.slides = 0
        self.recomputed = 0                                                # token-steps spent re-running kept tokens (reprefill only)
        self._logits: Optional[np.ndarray] = None

    def reset(self):
        self.cache.n = 0
        self.tokens, self.seen, self.slides, self.recomputed, self._logits = [], 0, 0, 0, None

    # ---- feeding
    def feed(self, ids: Sequence[int]) -> Optional[np.ndarray]:
        """Read tokens; returns the next-token logits after the last of them (None if nothing was ever fed)."""
        ids = [int(i) for i in ids]
        step = self.window - self.sink - self.stride                       # the largest piece that always fits after one slide
        for s in range(0, len(ids), step):
            self._logits = self._feed(ids[s:s + step])
        return self._logits

    def _feed(self, part: List[int]) -> np.ndarray:
        n, c0 = len(part), self.cache.n
        if c0 + n > self.window:
            need = c0 + n - self.window
            d = min(-(-need // self.stride) * self.stride, c0 - self.sink)
            keep = self.tokens[:self.sink] + self.tokens[self.sink + d:]
            self.slides += 1
            if self.mode == "reprefill":
                self.cache.n = 0
                self.recomputed += len(keep)
                logits = forward_cached(self.model, self.cache, keep + part)
                self.tokens = keep + part
                self.seen += n
                return logits
            self._rotate_drop(d)
            self.tokens = keep
        logits = forward_cached(self.model, self.cache, part)
        self.tokens = self.tokens + part
        self.seen += n
        return logits

    def _rotate_drop(self, d: int):
        c0, s = self.cache.n, self.sink
        cos_t, sin_t = self.model.rope_tables(d + 1)
        cos, sin = cos_t[d], sin_t[d]
        m = c0 - s - d
        K = _rope_back(self.cache.K[:, :, s + d:c0].copy(), cos, sin)
        V = self.cache.V[:, :, s + d:c0].copy()
        self.cache.K[:, :, s:s + m] = K
        self.cache.V[:, :, s:s + m] = V
        self.cache.n = c0 - d

    # ---- reading the model
    def logprobs(self, ids: Sequence[int]) -> np.ndarray:
        """log p(token | the window before it) for each token of `ids`, feeding them one at a time. The first token of a brand-new stream has no context: nan."""
        out = np.empty(len(ids))
        for i, t in enumerate(ids):
            if self._logits is None:
                out[i] = np.nan
            else:
                lg = self._logits - self._logits.max()
                out[i] = lg[int(t)] - np.log(np.exp(lg).sum())
            self.feed([t])
        return out

    def perplexity(self, ids: Sequence[int]) -> float:
        lp = self.logprobs(ids)
        lp = lp[~np.isnan(lp)]
        return float(np.exp(-lp.mean())) if len(lp) else float("nan")

    def generate(self, prompt: Sequence[int], max_new: int, rng: Optional[np.random.Generator] = None, temperature: float = 0.8, top_k: int = 0, top_p: float = 1.0,
                 end_id: Optional[int] = None, banned: Sequence[int] = ()) -> Iterator[int]:
        """Continue the stream for `max_new` tokens. There is no length limit other than max_new: the window slides under the generation."""
        rng = rng or np.random.default_rng(0)
        logits = self.feed(prompt) if len(prompt) else self._logits
        if logits is None:
            raise ValueError("nothing has been fed yet: give a prompt")
        for _ in range(int(max_new)):
            t = sample_token(logits, self.tokens, rng, temperature, top_k, top_p, 1.0, 0, banned)
            yield t
            if end_id is not None and t == end_id:
                return
            logits = self.feed([t])

    def describe(self) -> dict:
        return dict(mode=self.mode, window=self.window, stride=self.stride, sink=self.sink, in_window=len(self.tokens), seen=self.seen, slides=self.slides,
                    recomputed_token_steps=self.recomputed,
                    exact=self.mode == "reprefill", note="the cache equals a fresh pass over the window" if self.mode == "reprefill"
                    else "deeper layers keep context from dropped tokens: close to, not equal to, a fresh pass")
