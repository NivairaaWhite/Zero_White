"""hardware.py - measure the machine instead of assuming it.

Everything that depends on the computer (how many threads, how much memory a batch may use, how big a model is
worth training) is derived from what this module reports, so the same code runs on a laptop and on a server.
"""
from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import time

THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS")


def cpu_count() -> int:
    """Usable logical CPUs (respects affinity masks and container limits where the OS exposes them)."""
    try:
        n = len(os.sched_getaffinity(0))          # type: ignore[attr-defined]
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    # cgroup v2 / v1 CPU quota (Docker, WSL, CI): a quota below the visible core count is the real limit
    for p, kind in (("/sys/fs/cgroup/cpu.max", 2), ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", 1)):
        try:
            with open(p, "r", encoding="utf-8") as f:
                txt = f.read().split()
            if kind == 2 and txt and txt[0] != "max":
                n = min(n, max(1, int(round(int(txt[0]) / int(txt[1])))))
            elif kind == 1 and int(txt[0]) > 0:
                with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r", encoding="utf-8") as f:
                    n = min(n, max(1, int(round(int(txt[0]) / int(f.read().strip())))))
        except (OSError, ValueError, IndexError, ZeroDivisionError):
            pass
    return max(1, n)


def ram_bytes() -> int:
    """Total physical memory in bytes (0 when it cannot be determined)."""
    try:
        if sys.platform.startswith("win"):
            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))      # type: ignore[attr-defined]
            return int(ms.ullTotalPhys)
        if sys.platform == "darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                      timeout=5).stdout.strip())
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001 - a probe must never crash the program
        pass
    return 0


def ram_available_bytes() -> int:
    try:
        if sys.platform.startswith("win"):
            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))      # type: ignore[attr-defined]
            return int(ms.ullAvailPhys)
        if sys.platform != "darwin":
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001
        pass
    return ram_bytes()


def blas_threads() -> int:
    """Threads the BLAS will use right now (env override, else every CPU)."""
    for k in THREAD_ENV:
        v = os.environ.get(k)
        if v and v.isdigit() and int(v) > 0:
            return int(v)
    return cpu_count()


def thread_env(n: int | None) -> dict:
    """Environment variables that pin BLAS/OpenMP to `n` threads for a child process."""
    if not n:
        return {}
    return {k: str(int(n)) for k in THREAD_ENV}


def limit_threads(n: int | None) -> bool:
    """Best effort: cap BLAS threads in THIS process. Works when threadpoolctl is installed; otherwise the cap is
    applied through the environment (effective for child processes and for numpy imported afterwards).
    Returns True when the cap is certainly active."""
    if not n:
        return False
    try:
        from threadpoolctl import threadpool_limits          # optional
        threadpool_limits(int(n))
        return True
    except Exception:  # noqa: BLE001
        pass
    os.environ.update(thread_env(n))
    return False


def lower_priority() -> None:
    """Make this process a polite background citizen (used by training subprocesses started by the daemon)."""
    try:
        if sys.platform.startswith("win"):
            BELOW_NORMAL = 0x00004000
            h = ctypes.windll.kernel32.GetCurrentProcess()      # type: ignore[attr-defined]
            ctypes.windll.kernel32.SetPriorityClass(h, BELOW_NORMAL)   # type: ignore[attr-defined]
        else:
            os.nice(10)
    except Exception:  # noqa: BLE001
        pass


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform.startswith("win"):
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k = ctypes.windll.kernel32                              # type: ignore[attr-defined]
        h = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            if k.GetExitCodeProcess(h, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return True
        finally:
            k.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie child still answers kill(0); treat it as dead
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
            if f.read().rsplit(")", 1)[-1].split()[0] == "Z":
                return False
    except (OSError, IndexError):
        pass
    return True


def matmul_gflops(size: int = 512, repeat: int = 6) -> float:
    """Measured float32 matrix-multiply throughput of this machine, in GFLOP/s (about 0.1 s)."""
    import numpy as np
    a = np.random.default_rng(0).standard_normal((size * 2, size)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((size, size)).astype(np.float32)
    a @ b
    t = time.perf_counter()
    for _ in range(repeat):
        a @ b
    dt = max(time.perf_counter() - t, 1e-9)
    return repeat * 2.0 * a.shape[0] * size * size / dt / 1e9


def describe() -> dict:
    return dict(platform=platform.platform(), python=platform.python_version(), cpus=cpu_count(),
                ram_gb=round(ram_bytes() / 2 ** 30, 1), ram_available_gb=round(ram_available_bytes() / 2 ** 30, 1),
                blas_threads=blas_threads())
