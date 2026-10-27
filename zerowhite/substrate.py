"""substrate.py - which machine did I wake up on, and what follows from that?

The chat model already derives its own sizes from the hardware (`chat/hardware.py`, `chat/autoscale.py`). The general model
(v4) and the new ingestion layer (`omnivore/`) had no such awareness: every cap was a constant. This module closes that gap:

  probe()   measure the machine (cores, memory, matmul GFLOP/s, disk speed, optional accelerators)
  cached()  probe once per machine; a workspace that is copied to another computer notices and re-measures
  plan()    turn a probe into concrete limits (how big a file may be read, how many rows a table task keeps, ...)
            and write down WHY each number is what it is

Nothing here is required to be present: a probe that fails reports what it could not learn and the plan falls back to
conservative values. `accelerators` only *reports* what is installed. Since 6.2 the chat model's TRAINING can use a CUDA GPU through
CuPy (chat/backend.py, `zerowhite chat gpu` to check it); the general model and everything else still run on NumPy.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import tempfile
import time

from . import fileio
from .chat import hardware as hw

MiB = 1 << 20
GiB = 1 << 30

CACHE_NAME = "substrate.json"
CACHE_MAX_AGE_S = 24 * 3600

_ACCELERATOR_PACKAGES = ("cupy", "mlx", "jax", "torch")          # looked up with find_spec only: never imported


def _disk_mb_s(directory: str, megabytes: int = 4) -> float:
    """Sequential write speed (fsync'd) of the disk that holds `directory`, in MB/s; 0.0 when it cannot be measured."""
    try:
        os.makedirs(directory, exist_ok=True)
        blob = os.urandom(megabytes * MiB)
        fd, path = tempfile.mkstemp(dir=directory, prefix=".zw_disk_probe_")
        try:
            t = time.perf_counter()
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            dt = max(time.perf_counter() - t, 1e-6)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        return float(megabytes / dt)
    except Exception:                                      # noqa: BLE001 - a probe must never crash the program
        return 0.0


def accelerators() -> list:
    """Names of GPU-capable array libraries / tools that are installed. Detection only; nothing is imported or used."""
    found = []
    for name in _ACCELERATOR_PACKAGES:
        try:
            if importlib.util.find_spec(name) is not None:
                found.append(name)
        except (ImportError, ValueError, AttributeError):
            pass
    if shutil.which("nvidia-smi"):
        found.append("nvidia-smi")
    return found


def probe(workspace: str | None = None, bench: bool = True) -> dict:
    """Measure this machine. `bench=False` skips the timed parts (about 0.2 s in total when enabled)."""
    import numpy as np
    p = dict(t=time.time(), platform=platform.platform(), system=platform.system(), machine=platform.machine(),
             python=platform.python_version(), numpy=np.__version__, cpus=hw.cpu_count(),
             ram_gb=round(hw.ram_bytes() / GiB, 2), ram_available_gb=round(hw.ram_available_bytes() / GiB, 2),
             blas_threads=hw.blas_threads(), accelerators=accelerators(), gflops=0.0, disk_mb_s=0.0)
    if bench:
        try:
            p["gflops"] = round(hw.matmul_gflops(), 2)
        except Exception:                                  # noqa: BLE001
            pass
        p["disk_mb_s"] = round(_disk_mb_s(workspace or tempfile.gettempdir()), 1)
    return p


def fingerprint(p: dict) -> str:
    """What identifies 'the same machine' for caching purposes (not a serial number: just enough to notice a move)."""
    return f"{p.get('system')}/{p.get('machine')}/cpus={p.get('cpus')}/ram={round(float(p.get('ram_gb', 0)))}"


def cached(workspace: str, max_age_s: float = CACHE_MAX_AGE_S, force: bool = False) -> dict:
    """The probe stored in the workspace if it is fresh and was taken on this kind of machine, else a new one (which is stored)."""
    path = os.path.join(os.path.abspath(workspace), CACHE_NAME)
    quick = probe(workspace, bench=False)
    if not force and os.path.exists(path):
        try:
            old = fileio.read_json(path)
            if fingerprint(old) == fingerprint(quick) and time.time() - float(old.get("t", 0)) < max_age_s and old.get("gflops"):
                old["from_cache"] = True
                old["ram_available_gb"] = quick["ram_available_gb"]          # availability changes minute to minute
                return old
        except (ValueError, OSError):
            pass
    fresh = probe(workspace, bench=True)
    try:
        fileio.write_json(path, fresh, indent=2)
    except OSError:
        pass
    fresh["from_cache"] = False
    return fresh


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def plan(p: dict) -> dict:
    """Limits derived from a probe, plus `reasons`: one sentence per number saying where it came from."""
    avail = float(p.get("ram_available_gb") or p.get("ram_gb") or 0.0) * GiB
    gf = float(p.get("gflops") or 0.0)
    out, why = {}, {}
    if avail <= 0:
        avail = 1 * GiB
        why["memory"] = "available memory could not be measured: assuming 1 GiB"
    out["max_file_bytes"] = int(_clip(avail / 16, 8 * MiB, 256 * MiB))
    why["max_file_bytes"] = (f"one file may use about 1/16 of the {avail / GiB:.1f} GiB that is free "
                             f"(parsing makes several copies): {out['max_file_bytes'] / MiB:.0f} MiB")
    out["max_total_bytes"] = int(_clip(avail / 8, 32 * MiB, 1 * GiB))
    why["max_total_bytes"] = f"everything unpacked from one archive may use 1/8 of free memory: {out['max_total_bytes'] / MiB:.0f} MiB"
    if gf > 0:
        out["max_rows"] = int(_clip(gf * 800, 4_000, 250_000))
        why["max_rows"] = (f"{gf:.1f} GFLOP/s measured: a table task keeps at most {out['max_rows']:,} rows "
                           f"(more rows only slow the race down on this machine)")
        out["max_text_chars"] = int(_clip(gf * 20_000, 200_000, 4_000_000))
        why["max_text_chars"] = f"a text task trains on at most {out['max_text_chars']:,} characters at {gf:.1f} GFLOP/s"
    else:
        out["max_rows"], out["max_text_chars"] = 20_000, 1_000_000
        why["max_rows"] = why["max_text_chars"] = "speed could not be measured: conservative caps"
    out["cpus"] = int(p.get("cpus") or 1)
    out["reasons"] = why
    return out


def describe(p: dict, pl: dict | None = None) -> str:
    pl = pl or plan(p)
    acc = ", ".join(p.get("accelerators") or []) or "none found"
    lines = [f"machine      {p.get('platform')}  (python {p.get('python')}, numpy {p.get('numpy')})",
             f"cpus         {p.get('cpus')}   blas threads {p.get('blas_threads')}",
             f"memory       {p.get('ram_gb')} GiB total, {p.get('ram_available_gb')} GiB free",
             f"matmul       {p.get('gflops')} GFLOP/s (float32)   disk write {p.get('disk_mb_s')} MB/s",
             f"accelerators {acc}  (detected; chat training can use CuPy: `zerowhite chat gpu`; everything else runs on NumPy)", "derived limits:"]
    for k in ("max_file_bytes", "max_total_bytes", "max_rows", "max_text_chars"):
        lines.append(f"  {k:<16} {pl[k]:>12,}   {pl['reasons'].get(k, '')}")
    return "\n".join(lines)
