"""util.py - small shared helpers: cross-process lock files, durable JSONL appends, logging, time formatting."""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from typing import Iterator, Optional

from . import hardware


class LockBusy(RuntimeError):
    def __init__(self, path: str, holder: Optional[int]):
        super().__init__(f"{os.path.basename(path)} is held by process {holder}")
        self.path, self.holder = path, holder


class FileLock:
    """Exclusive cross-process lock implemented as an O_EXCL pid file. A lock whose owner process is gone (crash, kill -9,
    power loss) is detected and taken over, so a stale file never blocks anything forever."""

    def __init__(self, path: str):
        self.path = path
        self.held = False

    def _holder(self) -> Optional[int]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return int(f.read().strip().split()[0])
        except (OSError, ValueError, IndexError):
            return None

    def acquire(self, wait_s: float = 0.0) -> "FileLock":
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        deadline = time.time() + wait_s
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(f"{os.getpid()} {time.time():.0f}\n")
                self.held = True
                return self
            except FileExistsError:
                holder = self._holder()
                if holder is None or holder == os.getpid() or not hardware.pid_alive(holder):
                    try:
                        os.remove(self.path)
                    except OSError:
                        pass
                    continue
                if time.time() >= deadline:
                    raise LockBusy(self.path, holder)
                time.sleep(0.2)

    def release(self):
        if self.held:
            try:
                if self._holder() == os.getpid():
                    os.remove(self.path)
            except OSError:
                pass
            self.held = False

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


_append_lock = threading.Lock()


def append_jsonl(path: str, obj: dict, fsync: bool = False):
    """Append one JSON line. A single write() call per line keeps lines intact even with several writers."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
    with _append_lock:
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line)
            if fsync:
                f.flush()
                os.fsync(f.fileno())


def iter_jsonl(path: str) -> Iterator[dict]:
    """Yield the records of a JSONL file, skipping blank and torn (half-written) lines."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def setup_logging(path: Optional[str], max_mb: float = 10.0, backups: int = 5, console: bool = True,
                  name: str = "zerowhite", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(level)
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=int(max_mb * 1024 * 1024), backupCount=int(backups), encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        log.addHandler(sh)
    log.propagate = False
    return log


def fmt_time(s: float) -> str:
    s = int(max(0, s))
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def fmt_num(n: float) -> str:
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"
