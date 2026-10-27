"""cjit.py - compile a string of C into a shared library at run time and call it through ctypes (no Python build system, no extension module).

This is the CPU half of the 'generate kernel source in Python, compile it in memory, run it on numpy buffers' design: it uses whatever C compiler the machine has
(`cc`, `gcc`, `clang`; on Windows `gcc`/`clang` if installed, MSVC `cl` is NOT supported) and falls back to 'unavailable' without one, in which case callers use numpy.
Flags: -O3 and IEEE semantics (no -ffast-math, -ffp-contract=off) so results are the ordinary double-precision results, not reassociated ones. The library is
cached under ~/.zero_white/kernels (or $ZERO_WHITE_DATA/kernels) by the hash of its source, so a graph that is evaluated again costs no compile.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

_libs = {}


def compiler():
    for c in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if c and shutil.which(c):
            return shutil.which(c)
    return None


def available() -> bool:
    return compiler() is not None


def cache_dir() -> str:
    base = os.environ.get("ZERO_WHITE_DATA") or os.path.join(os.path.expanduser("~"), ".zero_white")
    d = os.path.join(base, "kernels")
    os.makedirs(d, exist_ok=True)
    return d


def build(source: str, extra_flags=()) -> ctypes.CDLL:
    """Compile `source` (C) to a shared library and load it. Raises RuntimeError with the compiler's message on failure."""
    key = hashlib.sha256((source + "|" + " ".join(extra_flags)).encode("utf-8")).hexdigest()[:24]
    if key in _libs:
        return _libs[key]
    cc = compiler()
    if cc is None:
        raise RuntimeError("no C compiler found (looked for $CC, cc, gcc, clang)")
    ext = ".dll" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")
    out = os.path.join(cache_dir(), f"k_{key}{ext}")
    if not os.path.exists(out):
        d = tempfile.mkdtemp(prefix="zw_cjit_")
        src = os.path.join(d, "k.c")
        with open(src, "w", encoding="utf-8") as f:
            f.write(source)
        tmp_out = os.path.join(d, "k" + ext)
        cmd = [cc, "-O3", "-shared", "-fPIC", "-ffp-contract=off", "-std=c99", *extra_flags, "-o", tmp_out, src, "-lm"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError("C compilation failed:\n" + r.stderr[-2000:])
        shutil.move(tmp_out, out)
        shutil.rmtree(d, ignore_errors=True)
    lib = ctypes.CDLL(out)
    _libs[key] = lib
    return lib


def ptr(a):
    return ctypes.c_void_p(a.ctypes.data)
