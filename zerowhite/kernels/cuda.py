"""cuda.py - run generated CUDA source through the CUDA driver API and NVRTC with ctypes only (no CuPy, no PyTorch, no nvcc).

The path, exactly as a CUDA programmer would do it by hand:  NVRTC compiles the generated source to PTX at run time  ->  cuModuleLoadData loads it  ->  cuMemAlloc /
cuMemcpyHtoD move the numpy buffers  ->  cuLaunchKernel runs it (grid-stride loop, so any size works)  ->  cuMemcpyDtoH brings the result back.

Layers, so that most of it can be tested without a GPU:
  RawDriver / RawNvrtc   thin ctypes wrappers: library lookup (Linux libcuda.so.1 / libnvrtc.so*, Windows nvcuda.dll / nvrtc64_*.dll), argtypes, error codes -> CudaError.
  CudaBackend            the sequence above, on any object with the same small Python API (RawDriver, or a fake).
  emulate_cuda           runs the generated CUDA source on the CPU by compiling it as C with a prelude that defines blockIdx / threadIdx / blockDim / gridDim and calling
                         the kernel once per (block, thread): it executes the ACTUAL generated text, including its index arithmetic and grid-stride loop.

HONEST STATUS: the author has not run this on a GPU. What is verified: the generated CUDA source computes the right numbers under the emulation above; CudaBackend
issues the right calls in the right order with the right values against a fake driver; RawDriver / RawNvrtc pass the right function names and arguments to a fake library.
What is NOT: that real NVRTC accepts the source, that cuLaunchKernel's parameter packing matches a real driver, performance (it is not tuned; one thread per element,
no shared memory), and anything on Windows or with multiple GPUs. `python -m zerowhite kernels check` runs the real thing on your machine and says what happened.
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

from . import cjit


class CudaError(RuntimeError):
    pass


def driver_names(platform=None):
    p = platform or sys.platform
    if p == "win32":
        return ["nvcuda.dll"]
    if p == "darwin":
        return []                                           # CUDA is not supported on current macOS
    return ["libcuda.so.1", "libcuda.so"]


def nvrtc_names(platform=None):
    p = platform or sys.platform
    if p == "win32":
        return [f"nvrtc64_{v}_0.dll" for v in ("130", "129", "128", "127", "126", "125", "124", "123", "122", "121", "120", "118", "117", "116", "115", "114", "113", "112")] + ["nvrtc64_111_0.dll", "nvrtc64_110_0.dll"]
    if p == "darwin":
        return []
    return ["libnvrtc.so"] + [f"libnvrtc.so.{v}" for v in ("13", "12", "11.8", "11.7", "11.6", "11.5", "11.4", "11.3", "11.2")]


def load_first(names, loader=None):
    loader = loader or ctypes.CDLL
    for n in names:
        try:
            return loader(n), n
        except OSError:
            continue
    return None, None


def _check(code, what):
    if code != 0:
        raise CudaError(f"{what} failed with code {code}")


class RawDriver:
    """ctypes wrapper over libcuda. `lib` is injectable (tests pass a fake library object with the same C function names)."""

    def __init__(self, lib=None):
        if lib is None:
            lib, _ = load_first(driver_names())
        if lib is None:
            raise CudaError("the CUDA driver library was not found (libcuda.so.1 / nvcuda.dll): no NVIDIA driver here")
        self.lib = lib
        self.ctx = None

    def init(self, device=0):
        _check(self.lib.cuInit(0), "cuInit")
        dev = ctypes.c_int(0)
        _check(self.lib.cuDeviceGet(ctypes.byref(dev), int(device)), "cuDeviceGet")
        ctx = ctypes.c_void_p()
        _check(self.lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
        self.ctx = ctx

    def module_load(self, image: bytes):
        mod = ctypes.c_void_p()
        _check(self.lib.cuModuleLoadData(ctypes.byref(mod), image), "cuModuleLoadData")
        return mod

    def get_function(self, mod, name: str):
        fn = ctypes.c_void_p()
        _check(self.lib.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode("ascii")), "cuModuleGetFunction")
        return fn

    def mem_alloc(self, nbytes: int):
        p = ctypes.c_uint64()
        _check(self.lib.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(nbytes)), "cuMemAlloc")
        return p

    def mem_free(self, p):
        _check(self.lib.cuMemFree_v2(p), "cuMemFree")

    def htod(self, dptr, arr: np.ndarray):
        _check(self.lib.cuMemcpyHtoD_v2(dptr, ctypes.c_void_p(arr.ctypes.data), ctypes.c_size_t(arr.nbytes)), "cuMemcpyHtoD")

    def dtoh(self, arr: np.ndarray, dptr):
        _check(self.lib.cuMemcpyDtoH_v2(ctypes.c_void_p(arr.ctypes.data), dptr, ctypes.c_size_t(arr.nbytes)), "cuMemcpyDtoH")

    def launch(self, fn, grid, block, dptrs, n=None):
        params = (ctypes.c_void_p * len(dptrs))(*[ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in dptrs])
        _check(self.lib.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, None, params, None), "cuLaunchKernel")

    def sync(self):
        _check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")


class RawNvrtc:
    def __init__(self, lib=None):
        if lib is None:
            lib, _ = load_first(nvrtc_names())
        if lib is None:
            raise CudaError("the NVRTC library was not found (libnvrtc.so* / nvrtc64_*.dll): install the CUDA toolkit's runtime compiler")
        self.lib = lib

    def compile(self, src: str, name: str = "k.cu", options=("--gpu-architecture=compute_70",)) -> bytes:
        prog = ctypes.c_void_p()
        _check(self.lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode("utf-8"), name.encode("ascii"), 0, None, None), "nvrtcCreateProgram")
        opts = (ctypes.c_char_p * len(options))(*[o.encode("ascii") for o in options])
        rc = self.lib.nvrtcCompileProgram(prog, len(options), opts)
        if rc != 0:
            size = ctypes.c_size_t()
            self.lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
            buf = ctypes.create_string_buffer(size.value)
            self.lib.nvrtcGetProgramLog(prog, buf)
            raise CudaError(f"nvrtcCompileProgram failed ({rc}): {buf.value.decode('utf-8', 'replace')[:2000]}")
        size = ctypes.c_size_t()
        _check(self.lib.nvrtcGetPTXSize(prog, ctypes.byref(size)), "nvrtcGetPTXSize")
        ptx = ctypes.create_string_buffer(size.value)
        _check(self.lib.nvrtcGetPTX(prog, ptx), "nvrtcGetPTX")
        self.lib.nvrtcDestroyProgram(ctypes.byref(prog))
        return ptx.raw


class CudaBackend:
    BLOCK = 256

    def __init__(self, driver=None, nvrtc=None, device=0):
        self.driver = driver if driver is not None else RawDriver()
        self.nvrtc = nvrtc if nvrtc is not None else RawNvrtc()
        self.driver.init(device)
        self._fn = {}

    def run(self, src: str, arrs, out: np.ndarray):
        if src not in self._fn:
            image = self.nvrtc.compile(src)
            mod = self.driver.module_load(image)
            self._fn[src] = self.driver.get_function(mod, "k")
        fn = self._fn[src]
        bufs = []
        try:
            for a in arrs:
                p = self.driver.mem_alloc(max(8, a.nbytes))
                self.driver.htod(p, a)
                bufs.append(p)
            po = self.driver.mem_alloc(max(8, out.nbytes))
            bufs.append(po)
            n = max(1, out.size)
            grid = min(65535, (n + self.BLOCK - 1) // self.BLOCK)
            self.driver.launch(fn, grid, self.BLOCK, bufs, n)
            self.driver.sync()
            self.driver.dtoh(out, po)
        finally:
            for p in bufs:
                self.driver.mem_free(p)
        return out


def emulate_cuda(src: str, arrs, out: np.ndarray, grid: int = 3, block: int = 8):
    """Run generated CUDA source on the CPU, one (block, thread) at a time, by compiling it as C with CUDA's built-in variables defined."""
    k = len(arrs)
    body = src.replace('extern "C" ', "").replace("__global__ ", "")
    params = ", ".join([f"const double* a{i}" for i in range(k)] + ["double* out"])
    call = ", ".join([f"a{i}" for i in range(k)] + ["out"])
    c = ("#include <math.h>\ntypedef struct { unsigned x, y, z; } dim3_;\nstatic dim3_ blockIdx, threadIdx, blockDim, gridDim;\n" + body +
         f"\nvoid run(long gx, long bx, {params}) {{\n  gridDim.x = gx; blockDim.x = bx;\n  for (long b = 0; b < gx; b++) for (long t = 0; t < bx; t++) {{ blockIdx.x = b; threadIdx.x = t; k({call}); }}\n}}\n")
    lib = cjit.build(c)
    lib.run.restype = None
    lib.run.argtypes = [ctypes.c_long, ctypes.c_long] + [ctypes.c_void_p] * (k + 1)
    lib.run(grid, block, *[cjit.ptr(np.ascontiguousarray(a)) for a in arrs], cjit.ptr(out))
    return out


def probe() -> dict:
    """What this machine offers, without raising: driver / NVRTC found, device count."""
    info = dict(driver=None, nvrtc=None, devices=None)
    lib, name = load_first(driver_names())
    info["driver"] = name
    if lib is not None:
        try:
            if lib.cuInit(0) == 0:
                n = ctypes.c_int(0)
                if lib.cuDeviceGetCount(ctypes.byref(n)) == 0:
                    info["devices"] = n.value
        except Exception:                                    # noqa: BLE001
            pass
    info["nvrtc"] = load_first(nvrtc_names())[1]
    return info
