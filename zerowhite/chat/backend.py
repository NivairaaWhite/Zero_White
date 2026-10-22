"""backend.py - where the chat model's arrays live: NumPy on the CPU (the default, unchanged) or CuPy on a GPU.

CuPy mirrors NumPy's API, so the model code is written once against `backend.xp` and runs on either. What differs is kept here:

  * transfers are EXPLICIT (`asarray` host -> device, `to_host` device -> host): nothing relies on CuPy's implicit conversions;
  * the one operation with no CuPy twin, the sorted segment-sum that scatters embedding gradients (`np.add.reduceat`), has a
    backend method (`scatter_rows`): the CPU keeps the exact legacy code, the GPU uses `cupyx.scatter_add`;
  * a GPU is asynchronous, so anything that is timed or read must `sync()` first;
  * memory is planned from the card (`free_memory`), not from RAM.

Selection: `compute.backend` = cpu | gpu | auto. `auto` (the default) uses the GPU when CuPy and a working device are present and
quietly falls back to the CPU otherwise (the reason is kept in `Backend.note`); `gpu` is a demand: if it cannot be honoured it
raises BackendError with the fix, because training for hours on the CPU by accident is worse than stopping.

Only TRAINING runs on the device. Serving, chat, export and online learning load weights from disk with the default CPU backend and
are unchanged. The same code path serves NVIDIA (CuPy `cupy-cuda12x` / `cupy-cuda11x`) and, experimentally, AMD (CuPy ROCm builds).
Apple Silicon has no CuPy and is not supported by this backend.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np


class BackendError(RuntimeError):
    pass


class Backend:
    name = "cpu"
    is_gpu = False
    device: Optional[int] = None

    def __init__(self, xp=np, note: Optional[str] = None):
        self.xp = xp
        self.note = note                      # why `auto` chose the CPU, when it did

    # ---- transfers (explicit, both ways) --------------------------------------------------------------------------
    def asarray(self, a, dtype=None):
        """Host (or already-device) data -> an array on this backend's device."""
        return self.xp.asarray(a, dtype=dtype)

    def to_host(self, a) -> np.ndarray:
        return np.asarray(a)

    def assign(self, dst, src):
        """dst[...] = src, where src may live on the host."""
        dst[...] = self.xp.asarray(src)

    def rand(self, rng: np.random.Generator, shape):
        """Uniform [0, 1) from the HOST generator, moved to the device: a run's dropout masks are the same on every backend."""
        return self.xp.asarray(rng.random(shape))

    # ---- the one operation CuPy has no twin for ------------------------------------------------------------------------
    def scatter_rows(self, dE, ids_flat, rows):
        """dE[id] += row for every (id, row): sorted segment sums, far faster than np.add.at for big batches."""
        order = np.argsort(ids_flat, kind="stable")
        sid = ids_flat[order]
        uniq, start = np.unique(sid, return_index=True)
        dE[uniq] += np.add.reduceat(rows[order], start, axis=0)

    # ---- device state ---------------------------------------------------------------------------------------------------
    def sync(self) -> None:
        """Wait for queued work (a no-op on the CPU). A GPU is asynchronous: timing without this measures nothing."""

    def free_memory(self) -> Optional[int]:
        return None

    def total_memory(self) -> Optional[int]:
        return None

    def release(self) -> None:
        """Hand cached memory back (a no-op on the CPU)."""

    def is_oom(self, exc: BaseException) -> bool:
        return isinstance(exc, MemoryError)

    def matmul_gflops(self, size: int = 512, repeat: int = 6) -> float:
        from . import hardware as hw
        return hw.matmul_gflops(size, repeat)

    def describe(self) -> dict:
        from . import hardware as hw
        return dict(backend="cpu", cores=hw.cpu_count(), note=self.note)

    def label(self) -> str:
        return "cpu"


class GpuBackend(Backend):
    name = "gpu"
    is_gpu = True

    def __init__(self, cp, cupyx, device: int = 0):
        super().__init__(cp)
        self.cp, self.cupyx, self.device = cp, cupyx, int(device)
        cp.cuda.Device(self.device).use()

    def to_host(self, a) -> np.ndarray:
        return a if isinstance(a, np.ndarray) else self.cp.asnumpy(a)

    def scatter_rows(self, dE, ids_flat, rows):
        self.cupyx.scatter_add(dE, ids_flat, rows)

    def sync(self) -> None:
        self.cp.cuda.runtime.deviceSynchronize()

    def free_memory(self) -> Optional[int]:
        return int(self.cp.cuda.Device(self.device).mem_info[0])

    def total_memory(self) -> Optional[int]:
        return int(self.cp.cuda.Device(self.device).mem_info[1])

    def release(self) -> None:
        try:
            self.cp.get_default_memory_pool().free_all_blocks()
            self.cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:                                    # noqa: BLE001 - releasing is best effort
            pass

    def is_oom(self, exc: BaseException) -> bool:
        return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()

    def matmul_gflops(self, size: int = 2048, repeat: int = 8) -> float:
        a = self.cp.asarray(np.random.default_rng(0).standard_normal((size, size)).astype(np.float32))
        b = a.T.copy()
        (a @ b)
        self.sync()
        t = time.perf_counter()
        for _ in range(repeat):
            a @ b
        self.sync()
        return 2.0 * size ** 3 * repeat / max(time.perf_counter() - t, 1e-9) / 1e9

    def properties(self) -> dict:
        try:
            p = self.cp.cuda.runtime.getDeviceProperties(self.device)
            name = p["name"]
            return dict(name=name.decode() if isinstance(name, bytes) else str(name), capability=f"{p.get('major', '?')}.{p.get('minor', '?')}")
        except Exception:                                    # noqa: BLE001
            return dict(name="unknown device", capability="?")

    def describe(self) -> dict:
        d = dict(backend="gpu", device=self.device, **self.properties(), cupy=getattr(self.cp, "__version__", "?"))
        try:
            free, total = self.cp.cuda.Device(self.device).mem_info
            d.update(vram_gb=round(total / 2 ** 30, 1), vram_free_gb=round(free / 2 ** 30, 1))
            d["cuda_runtime"] = int(self.cp.cuda.runtime.runtimeGetVersion())
        except Exception:                                    # noqa: BLE001
            pass
        return d

    def label(self) -> str:
        d = self.describe()
        return f"gpu:{self.device} {d.get('name', '')} ({d.get('vram_gb', '?')} GB)"


CPU = Backend()
_GPU_CACHE: dict = {}
_INSTALL_HINT = ("install CuPy for your CUDA version, e.g. `pip install cupy-cuda12x` (CUDA 12.x) or `cupy-cuda11x` (CUDA 11.x); "
                 "`nvidia-smi` shows the driver's CUDA version")


def _make_gpu(device: int) -> GpuBackend:
    try:
        import cupy as cp
    except Exception as ex:                                  # noqa: BLE001 - any import failure means "no CuPy here"
        raise BackendError(f"CuPy is not importable ({type(ex).__name__}: {ex}); {_INSTALL_HINT}") from ex
    try:
        import cupyx
        n = int(cp.cuda.runtime.getDeviceCount())
    except Exception as ex:                                  # noqa: BLE001
        raise BackendError(f"CuPy found no usable GPU ({type(ex).__name__}: {ex}); check the driver with `nvidia-smi`") from ex
    if not 0 <= int(device) < n:
        raise BackendError(f"compute.device={device} but only {n} GPU(s) are visible")
    be = GpuBackend(cp, cupyx, device)
    try:                                                     # a kernel must really compile and run: a broken CUDA install fails here, not mid-training
        a = cp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4))
        ok = np.allclose(cp.asnumpy(a @ a), np.arange(16, dtype=np.float32).reshape(4, 4) @ np.arange(16, dtype=np.float32).reshape(4, 4))
    except Exception as ex:                                  # noqa: BLE001
        raise BackendError(f"the GPU failed a 4x4 matrix product ({type(ex).__name__}: {ex}); the CUDA runtime and CuPy build probably do not match") from ex
    if not ok:
        raise BackendError("the GPU returned a wrong 4x4 matrix product; refusing to train on it")
    return be


def get(name: str = "auto", device: int = 0) -> Backend:
    """The backend for `compute.backend` / `compute.device`. See the module docstring for the cpu / gpu / auto contract."""
    name = (name or "auto").lower()
    if name == "cpu":
        return CPU
    if name not in ("gpu", "auto", "cuda"):
        raise BackendError(f"unknown compute backend '{name}' (use cpu, gpu or auto)")
    key = int(device)
    if key in _GPU_CACHE:
        return _GPU_CACHE[key]
    try:
        be = _make_gpu(key)
    except BackendError as ex:
        if name == "auto":
            return Backend(note=f"no GPU used: {ex}")
        raise
    _GPU_CACHE[key] = be
    return be


def from_config(cfg) -> Backend:
    return get(str(cfg.get("compute.backend")), int(cfg.get("compute.device")))


def is_array_of(be: Backend, a) -> bool:
    """True when `a` already lives on `be`'s device (host arrays are numpy.ndarray; device arrays are not)."""
    return isinstance(a, np.ndarray) if not be.is_gpu else not isinstance(a, np.ndarray)
