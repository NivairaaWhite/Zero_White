"""pipeline.py - wire live sources to a model: ring buffer -> streaming tokeniser -> one unified token stream -> a callback (typically `WindowLM.feed`).

    pipe = StreamPipeline(space)                                     # space: a TokenSpace
    pipe.add_text("chat", TextStream(tokenizer))
    pipe.add_audio("mic", AudioStream(codec), capacity=16000)
    pipe.add_video("cam", VideoStream(pixel_codec), shape=(32, 32))
    pipe.on_tokens = lm.feed                                         # receives lists of ids in `space`
    pipe.push("mic", samples)  ...  pipe.pump()                      # deterministic: one pass over everything buffered
    pipe.start(); ...; pipe.stop()                                   # or a background thread that pumps every `poll` seconds

What it does NOT do is hide loss: every source has a bounded `Ring`, a source that outruns the consumer drops (the oldest by default) and `stats()` says how many rows,
and for each source the latency from the moment a chunk was pushed to the moment its tokens were handed on (mean / max). Nothing here is a trained model; the ids are a layout
(`TokenSpace`), and what the callback does with them is the callback's business.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from .ring import Ring
from .tokens import AudioStream, TextStream, TokenSpace, VideoStream


class _Channel:
    def __init__(self, name: str, modality: str, ring: Ring, encode: Callable[[np.ndarray], List[int]], finish: Callable[[], List[int]]):
        self.name, self.modality, self.ring, self.encode, self.finish = name, modality, ring, encode, finish
        self.stamps: Deque[Tuple[float, int]] = deque()                  # (push time, rows) of what is still in the ring
        self.lat: List[float] = []
        self.tokens = 0


class StreamPipeline:
    def __init__(self, space: TokenSpace, clock: Callable[[], float] = time.monotonic, on_tokens: Optional[Callable[[List[int]], None]] = None):
        self.space, self.clock, self.on_tokens = space, clock, on_tokens
        self.channels: Dict[str, _Channel] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.error: Optional[BaseException] = None

    # ---- sources
    def add_text(self, name: str, stream: TextStream, capacity: int = 1 << 16, overflow: str = "drop_oldest") -> Ring:
        ring = Ring(capacity, dtype=np.uint8, overflow=overflow)
        def enc(rows): return self.space.text(stream.feed(rows.tobytes()))
        def fin(): return self.space.text(stream.finish())
        self.channels[name] = _Channel(name, "text", ring, enc, fin)
        return ring

    def add_audio(self, name: str, stream: AudioStream, capacity: int = 16000, overflow: str = "drop_oldest") -> Ring:
        ring = Ring(capacity, dtype=np.float64, overflow=overflow)
        def enc(rows): return self.space.audio(stream.feed(rows))
        def fin(): return self.space.audio(stream.finish())
        self.channels[name] = _Channel(name, "audio", ring, enc, fin)
        return ring

    def add_video(self, name: str, stream: VideoStream, shape: Tuple[int, ...], capacity: int = 8, overflow: str = "drop_oldest") -> Ring:
        ring = Ring(capacity, dtype=np.float64, shape=tuple(shape), overflow=overflow)
        def enc(rows):
            out: List[int] = []
            for fr in rows:
                tok, _ = stream.push(fr)
                out.extend(self.space.pixels(tok))
            return out
        self.channels[name] = _Channel(name, "pixel", ring, enc, lambda: [])
        return ring

    # ---- data in
    def push(self, name: str, data) -> int:
        ch = self.channels[name]
        if ch.modality == "text":
            data = np.frombuffer(data.encode("utf-8") if isinstance(data, str) else bytes(data), dtype=np.uint8)
        kept = ch.ring.push(data)
        if kept:
            with self._lock:
                ch.stamps.append((self.clock(), kept))
        return kept

    # ---- processing
    def pump(self, final: bool = False) -> int:
        """One pass: take everything buffered from every source, tokenise it, hand each source's tokens on as ONE framed chunk. Returns the number of ids handed on."""
        total = 0
        for ch in list(self.channels.values()):
            rows = ch.ring.read()
            now = self.clock()
            ids: List[int] = []
            if len(rows):
                with self._lock:
                    if ch.stamps:
                        ch.lat.append(now - ch.stamps[0][0])
                    ch.stamps.clear()
                ids = ch.encode(rows)
            if final:
                ids = ids + ch.finish()
            if ids:
                self._emit(ch, ids)
                total += len(ids)
        return total

    def _emit(self, ch: _Channel, ids: List[int]):
        mark = {"text": "<|text|>", "audio": "<|audio|>", "pixel": "<|video|>"}[ch.modality]
        framed = [self.space.marker(mark)] + ids + [self.space.marker("<|end_chunk|>")]
        ch.tokens += len(ids)
        if self.on_tokens is not None:
            self.on_tokens(framed)

    def flush(self) -> int:
        return self.pump(final=True)

    # ---- background thread
    def start(self, poll: float = 0.01):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def run():
            try:
                while not self._stop.is_set():
                    if not self.pump():
                        time.sleep(poll)
            except BaseException as e:                                   # a consumer that raises must not die silently
                self.error = e
        self._thread = threading.Thread(target=run, name="zerowhite-stream", daemon=True)
        self._thread.start()

    def stop(self, flush: bool = True, timeout: float = 5.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        if flush and self.error is None:
            self.pump(final=True)
        if self.error is not None:
            raise RuntimeError(f"the stream consumer failed: {self.error!r}") from self.error

    def stats(self) -> dict:
        out = {}
        for n, ch in self.channels.items():
            st = ch.ring.stats()
            st.update(modality=ch.modality, ids=ch.tokens, latency_mean=float(np.mean(ch.lat)) if ch.lat else None, latency_max=float(np.max(ch.lat)) if ch.lat else None)
            out[n] = st
        return out
