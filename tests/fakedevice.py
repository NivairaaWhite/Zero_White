"""fakedevice.py - a strict stand-in for a GPU, so the device code path can be tested on a machine that has none.

It cannot tell you that a real card is fast or that CuPy behaves; it CAN catch the mistakes a NumPy -> CuPy port usually has:

  * a host array leaking into device math (CuPy refuses to mix them implicitly; so does this: HostLeak);
  * calling something CuPy does not provide (only the functions in FakeXP.ALLOWED exist; ufunc.reduceat / .at are refused);
  * hidden synchronisations: every device -> host transfer and every scalar read (float(), int(), item()) is counted, so a test can
    assert that a training step costs a handful of them, not one per parameter.

The "device memory" is ordinary NumPy memory behind a subclass, so results can be compared with a CPU run directly.
"""
from __future__ import annotations

import numpy as np

from zerowhite.chat import backend as bk


class HostLeak(AssertionError):
    """A host (plain numpy) array met device memory without an explicit transfer."""


class NotOnDevice(AttributeError):
    """The code used an API that CuPy does not provide."""


class FakeOOM(MemoryError):
    pass


class Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.h2d_calls = self.d2h_calls = self.h2d_bytes = self.d2h_bytes = self.scalar_reads = self.syncs = self.releases = 0

    def snapshot(self) -> dict:
        return dict(self.__dict__)


STATS = Stats()


def _flat(objs):
    for o in objs:
        if isinstance(o, (list, tuple)):
            yield from _flat(o)
        elif isinstance(o, dict):
            yield from _flat(o.values())
        else:
            yield o


def _check(objs):
    for o in _flat(objs):
        if isinstance(o, np.ndarray) and not isinstance(o, DevArray):
            raise HostLeak(f"a host numpy array (shape {o.shape}, {o.dtype}) was combined with device memory; transfer it with backend.asarray()")


def _raw(o):
    if isinstance(o, DevArray):
        return o.view(np.ndarray)
    if isinstance(o, (list, tuple)):
        return type(o)(_raw(x) for x in o)
    if isinstance(o, dict):
        return {k: _raw(v) for k, v in o.items()}
    return o


def _wrap(r):
    if isinstance(r, np.ndarray):
        return r.view(DevArray)
    if isinstance(r, np.generic):
        return np.asarray(r).view(DevArray)                 # a reduction leaves a 0-d device array, like CuPy; reading it is a sync
    if isinstance(r, tuple):
        return tuple(_wrap(x) for x in r)
    if isinstance(r, list):
        return [_wrap(x) for x in r]
    return r


class DevArray(np.ndarray):
    """Pretend device memory."""

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method in ("reduceat", "at"):
            raise NotOnDevice(f"CuPy has no ufunc.{method} (np.{ufunc.__name__}.{method}); use a backend method instead")
        _check(list(inputs) + list(kwargs.get("out") or ()) + [kwargs.get("where")])
        kwargs = dict(kwargs)
        if kwargs.get("out") is not None:
            kwargs["out"] = tuple(_raw(o) for o in kwargs["out"])
        return _wrap(getattr(ufunc, method)(*[_raw(i) for i in inputs], **kwargs))

    def __array_function__(self, func, types, args, kwargs):
        _check(list(args) + list((kwargs or {}).values()))
        return _wrap(func(*_raw(args), **_raw(kwargs or {})))

    def __getitem__(self, idx):
        _check(idx if isinstance(idx, tuple) else (idx,))
        return super().__getitem__(idx)

    def __setitem__(self, idx, val):
        _check(list(idx if isinstance(idx, tuple) else (idx,)) + [val])
        super().__setitem__(idx, val)

    def __float__(self):
        STATS.scalar_reads += 1
        return float(self.view(np.ndarray))

    def __int__(self):
        STATS.scalar_reads += 1
        return int(self.view(np.ndarray))

    def __bool__(self):
        STATS.scalar_reads += 1
        return bool(self.view(np.ndarray))

    def item(self, *a):
        STATS.scalar_reads += 1
        return self.view(np.ndarray).item(*a)


class FakeXP:
    """The slice of the CuPy API that zerowhite's training path uses. Anything else does not exist here, as it might not in CuPy."""
    ALLOWED = ("zeros_like", "zeros", "ones", "arange", "tri", "where", "exp", "log", "sqrt", "mean", "matmul", "concatenate",
               "flatnonzero", "ascontiguousarray", "vdot", "stack", "clip", "maximum", "sum", "abs", "max", "isfinite", "all")
    DTYPES = ("float32", "float64", "int64", "int32", "bool_")

    def __init__(self, backend):
        self._be = backend

    def asarray(self, a, dtype=None):
        return self._be.asarray(a, dtype)

    def __getattr__(self, name):
        if name in self.DTYPES:
            return getattr(np, name)
        if name in self.ALLOWED:
            fn = getattr(np, name)

            def call(*args, **kwargs):
                _check(list(args) + list(kwargs.values()))
                return _wrap(fn(*_raw(args), **_raw(kwargs)))
            return call
        raise NotOnDevice(f"cupy.{name} is not part of the API this project relies on (add it to FakeXP.ALLOWED only if CuPy really has it)")


class FakeGPU(bk.Backend):
    name = "gpu"
    is_gpu = True
    device = 0

    def __init__(self, free_gb: float = 8.0, total_gb: float = 8.0, label: str = "Fake GPU"):
        super().__init__(xp=None)
        self.xp = FakeXP(self)
        self._free, self._total, self._label = int(free_gb * 2 ** 30), int(total_gb * 2 ** 30), label

    def asarray(self, a, dtype=None):
        if isinstance(a, DevArray):
            return a if dtype is None or a.dtype == np.dtype(dtype) else a.astype(dtype)
        h = np.asarray(a, dtype=dtype)
        STATS.h2d_calls += 1
        STATS.h2d_bytes += h.nbytes
        return np.array(h, copy=True).view(DevArray)

    def to_host(self, a):
        if not isinstance(a, DevArray):
            return a
        STATS.d2h_calls += 1
        STATS.d2h_bytes += a.nbytes
        return np.array(a.view(np.ndarray), copy=True)

    def assign(self, dst, src):
        dst[...] = self.asarray(src)

    def rand(self, rng, shape):
        return self.asarray(rng.random(shape))

    def scatter_rows(self, dE, ids_flat, rows):
        _check([dE, ids_flat, rows])
        np.add.at(dE.view(np.ndarray), ids_flat.view(np.ndarray), rows.view(np.ndarray))

    def sync(self):
        STATS.syncs += 1

    def release(self):
        STATS.releases += 1

    def free_memory(self):
        return self._free

    def total_memory(self):
        return self._total

    def is_oom(self, exc):
        return isinstance(exc, FakeOOM)

    def matmul_gflops(self, size: int = 512, repeat: int = 6) -> float:
        return 5000.0                                          # a pretend card: only used where a measurement is skipped

    def describe(self) -> dict:
        return dict(backend="gpu", device=0, name=self._label, vram_gb=round(self._total / 2 ** 30, 1), vram_free_gb=round(self._free / 2 ** 30, 1))

    def label(self) -> str:
        return f"gpu:0 {self._label} ({self._total / 2 ** 30:.1f} GB)"
