"""memwatch.py - enforce a memory limit on a child process from OUTSIDE it, for the operating systems that do not enforce one themselves.

Why it exists: the sandbox limits a child's memory with RLIMIT_AS on Linux and with a Job Object on Windows. macOS enforces neither (setrlimit(RLIMIT_AS) is accepted or refused
and then ignored), so on the `macos-latest` CI legs of 6.4 a child that allocated 900 MB under a 256 MB limit simply succeeded, and `doctor` correctly reported "the memory
limit is not enforced here". The only portable remedy for such a system is to watch the child from the parent and kill its process group when it grows past the limit.

What it measures, per platform (the first that works; it never raises):
  linux     resident pages from /proc/<pid>/statm                                  (only used when forced: Linux enforces RLIMIT_AS itself)
  darwin    phys_footprint from libproc's proc_pid_rusage (what Activity Monitor calls "Memory": resident + compressed pages, so a hog the compressor squeezed still counts),
            then `ps -o rss=` if libproc cannot be loaded or returns nothing for this very process
  other     `ps -o rss= -p PID`

HONEST LIMITS: it is a poll (every 2 ms through libproc, every 20 ms through `ps`), not a hard wall: a child can overshoot by what it allocates in one interval (a single
900 MB allocation is caught only after it is touched, because untouched virtual memory is not resident). It watches the child's own process, not grandchildren (the whole
process GROUP is killed when it trips, so grandchildren die with it, but their memory is not summed). It measures resident / footprint memory, not address space, so the same
number means a looser limit than RLIMIT_AS does. It is containment for evolved-code accidents, like the rest of the sandbox, not a security boundary. The libproc structure
layout is written from Apple's header and checked against a fake library on every platform; it has not been run on a Mac by its author (the `macos-latest` CI legs do).
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading

RUSAGE_INFO_V0 = 0
# struct rusage_info_v0 { uint8_t ri_uuid[16]; uint64_t ri_user_time, ri_system_time, ri_pkg_idle_wkups, ri_interrupt_wkups, ri_pageins, ri_wired_size,
#                         ri_resident_size, ri_phys_footprint, ri_proc_start_abstime, ri_proc_exit_abstime; }
_OFF_RESIDENT, _OFF_FOOTPRINT, _V0_SIZE = 64, 72, 96
_BUF = 512                                                      # room for every later version of the structure too (the kernel fills only what the flavour needs)


def needed(platform: str | None = None) -> bool:
    """True where the OS does not enforce a memory limit on a child by itself (so the parent has to watch): everything POSIX except Linux and the BSDs that honour RLIMIT_AS."""
    p = platform or sys.platform                                # only the name that was asked about decides: needed("darwin") is True on a Windows host too
    if p == "win32":
        return False                                            # Windows has the Job Object (winjob.py)
    return not p.startswith(("linux", "freebsd", "openbsd", "netbsd"))


def parse_rusage_v0(buf: bytes):
    """(resident_size, phys_footprint) in bytes from a filled rusage_info_v0 buffer (little-endian uint64 fields, as on every Mac that has a libproc)."""
    if len(buf) < _V0_SIZE:
        raise ValueError("buffer too short for rusage_info_v0")
    return (int.from_bytes(buf[_OFF_RESIDENT:_OFF_RESIDENT + 8], sys.byteorder), int.from_bytes(buf[_OFF_FOOTPRINT:_OFF_FOOTPRINT + 8], sys.byteorder))


def _load_libproc():
    for name in (None, "libproc.dylib", "/usr/lib/libproc.dylib", "/usr/lib/libSystem.B.dylib"):
        try:
            lib = ctypes.CDLL(name)
            getattr(lib, "proc_pid_rusage")                        # dlsym: raises AttributeError when this library does not export it
            return lib
        except (OSError, AttributeError):
            continue
    return None


def libproc_sampler(lib=None):
    """pid -> phys_footprint in bytes through libproc, or None when libproc is not available. `lib` is injectable (tests pass a fake with the same C function)."""
    lib = lib if lib is not None else _load_libproc()
    if lib is None:
        return None

    def sample(pid: int):
        buf = (ctypes.c_uint8 * _BUF)()
        try:
            if lib.proc_pid_rusage(ctypes.c_int(int(pid)), ctypes.c_int(RUSAGE_INFO_V0), ctypes.byref(buf)) != 0:
                return None
            resident, footprint = parse_rusage_v0(bytes(buf))
        except (OSError, ValueError, AttributeError, ctypes.ArgumentError):
            return None
        return footprint or resident or None
    return sample


def proc_sampler():
    """pid -> resident bytes from /proc (Linux), or None when there is no /proc."""
    if not os.path.isdir("/proc/self"):
        return None
    try:
        page = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        page = 4096

    def sample(pid: int):
        try:
            with open(f"/proc/{int(pid)}/statm", "rb") as f:
                return int(f.read().split()[1]) * page
        except (OSError, ValueError, IndexError):
            return None
    return sample


def ps_sampler(run=None):
    """pid -> resident bytes by running `ps -o rss= -p PID` (macOS, BSDs, Linux with procps). Slow (a process per sample) but needs nothing else. `run` is injectable."""
    run = run or subprocess.run

    def sample(pid: int):
        try:
            r = run(["ps", "-o", "rss=", "-p", str(int(pid))], capture_output=True, text=True, timeout=5)
            return int(r.stdout.strip().split()[0]) * 1024 if r.returncode == 0 and r.stdout.strip() else None
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None
    return sample


def make_sampler(platform: str | None = None):
    """(sampler, name, interval_seconds) for this platform: the cheapest source that works. Never raises; the sampler may return None for a pid that has gone."""
    p = platform or sys.platform
    if p == "darwin":
        s = libproc_sampler()
        if s is not None and s(os.getpid()):                       # prove it on ourselves before trusting it: a wrong layout would read zeros
            return s, "libproc", 0.002
    if p.startswith("linux"):
        s = proc_sampler()
        if s is not None and s(os.getpid()):
            return s, "/proc", 0.002
    s = ps_sampler()
    if s(os.getpid()):
        return s, "ps", 0.02
    return None, "none", 0.05


class MemoryWatch(threading.Thread):
    """Polls a process's memory and calls `on_trip()` once when it exceeds `limit_bytes`. Start it right after the child is created, `stop()` it when the child is done."""

    def __init__(self, pid: int, limit_bytes: int, on_trip, sampler=None, interval: float | None = None):
        super().__init__(daemon=True, name="zw-memwatch")
        if sampler is None:
            sampler, self.source, auto_interval = make_sampler()
        else:
            self.source, auto_interval = "custom", 0.002
        self.pid, self.limit, self.on_trip, self.sampler = int(pid), int(limit_bytes), on_trip, sampler
        self.interval = auto_interval if interval is None else float(interval)
        self.tripped = None                                         # the measurement (bytes) that exceeded the limit, if any
        self.peak, self.samples = 0, 0
        self._stop_evt = threading.Event()

    @property
    def active(self) -> bool:
        return self.sampler is not None

    def run(self):
        if self.sampler is None:
            return
        while not self._stop_evt.is_set():
            v = self.sampler(self.pid)
            if v:
                self.samples += 1
                self.peak = max(self.peak, v)
                if v > self.limit:
                    self.tripped = v
                    try:
                        self.on_trip()
                    except Exception:                               # noqa: BLE001 - the watcher must never raise into the thread machinery
                        pass
                    return
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(2.0)
