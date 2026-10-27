"""ring.py - a fixed-capacity ring buffer for a live stream (audio samples, pixel frames, token ids, observations).

A live source does not wait for its consumer. The two honest options when the consumer is slow are to throw the OLDEST data away (a microphone:
the newest audio matters) or to throw the NEWEST away / refuse it (a recording that must not lose its start), and either way the loss must be
COUNTED, never silent. `Ring` does exactly that and nothing else:

    r = Ring(capacity=16000, dtype=np.float32)           # one second of 16 kHz audio
    r.push(chunk)            -> how many rows were kept             (overflow="drop_oldest" | "drop_newest" | "error")
    r.read(n)                -> the n oldest rows, REMOVED          r.peek(n) the same, not removed        r.latest(n) the n newest rows, not removed
    r.read_blocking(n, timeout)  waits (condition variable) until n rows are there or the time is up

Rows are the first axis: a ring of frames has `shape=(H, W)`. Thread safe: one producer thread and one consumer thread is the intended use and needs no
other locking. `dropped` counts rows lost to overflow, `pushed` and `consumed` count what went in and out; pushed == consumed + dropped + len(ring) always.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np


class RingOverflow(RuntimeError):
    pass


class Ring:
    OVERFLOW = ("drop_oldest", "drop_newest", "error")

    def __init__(self, capacity: int, dtype=np.float64, shape: Tuple[int, ...] = (), overflow: str = "drop_oldest"):
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if overflow not in self.OVERFLOW:
            raise ValueError(f"overflow must be one of {self.OVERFLOW}")
        self.capacity, self.shape, self.overflow = capacity, tuple(shape), overflow
        self._a = np.zeros((capacity,) + self.shape, dtype=dtype)
        self._head = 0                                                  # index of the oldest row
        self._n = 0
        self.pushed = self.consumed = self.dropped = 0
        self._cv = threading.Condition()

    # ---- state
    def __len__(self) -> int:
        return self._n

    @property
    def dtype(self):
        return self._a.dtype

    def clear(self) -> int:
        """Discard everything buffered (counted as consumed-by-nobody: it goes to `dropped`). Returns how many rows."""
        with self._cv:
            k = self._n
            self.dropped += k
            self._head, self._n = 0, 0
            return k

    # ---- producer side
    def push(self, x) -> int:
        """Append rows. Returns how many of them were kept (rows refused by drop_newest, or older rows evicted by drop_oldest, go to `dropped`)."""
        x = np.asarray(x, dtype=self._a.dtype)
        if x.ndim == len(self.shape):                                   # a single row
            x = x[None]
        if x.shape[1:] != self.shape:
            raise ValueError(f"rows must have shape {self.shape}, got {x.shape[1:]}")
        k = len(x)
        if k == 0:
            return 0
        with self._cv:
            self.pushed += k
            free = self.capacity - self._n
            if k > free and self.overflow == "error":
                self.pushed -= k
                raise RingOverflow(f"{k} rows do not fit ({free} free of {self.capacity})")
            if k > free and self.overflow == "drop_newest":
                x = x[:free]
                self.dropped += k - free
                kept = free
            else:
                kept = k
                if k > self.capacity:                                   # more than the whole ring: only the newest `capacity` rows can survive
                    self.dropped += k - self.capacity
                    x = x[-self.capacity:]
                    k = self.capacity
                over = max(0, self._n + k - self.capacity)
                if over:                                                # evict the oldest rows to make room
                    self._head = (self._head + over) % self.capacity
                    self._n -= over
                    self.dropped += over
            m = len(x)
            end = (self._head + self._n) % self.capacity
            first = min(m, self.capacity - end)
            self._a[end:end + first] = x[:first]
            if m > first:
                self._a[:m - first] = x[first:]
            self._n += m
            self._cv.notify_all()
            return kept

    # ---- consumer side
    def _copy(self, n: int, offset: int = 0) -> np.ndarray:
        idx = (self._head + offset + np.arange(n)) % self.capacity
        return self._a[idx].copy()

    def peek(self, n: Optional[int] = None) -> np.ndarray:
        with self._cv:
            n = self._n if n is None else min(int(n), self._n)
            return self._copy(n)

    def latest(self, n: int) -> np.ndarray:
        """The n NEWEST rows (fewer if fewer are buffered), oldest first, without removing them."""
        with self._cv:
            n = min(int(n), self._n)
            return self._copy(n, self._n - n)

    def read(self, n: Optional[int] = None) -> np.ndarray:
        with self._cv:
            n = self._n if n is None else min(int(n), self._n)
            out = self._copy(n)
            self._head = (self._head + n) % self.capacity
            self._n -= n
            self.consumed += n
            return out

    def read_blocking(self, n: int, timeout: Optional[float] = None) -> np.ndarray:
        """Wait until `n` rows are buffered (or `timeout` seconds pass), then read what is there (up to n)."""
        n = min(int(n), self.capacity)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while self._n < n:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    break
                self._cv.wait(left)
            k = min(n, self._n)
            out = self._copy(k)
            self._head = (self._head + k) % self.capacity
            self._n -= k
            self.consumed += k
            return out

    def stats(self) -> dict:
        return dict(capacity=self.capacity, buffered=self._n, pushed=self.pushed, consumed=self.consumed, dropped=self.dropped, overflow=self.overflow)
