"""cuda.py - run generated CUDA source through the CUDA driver API and NVRTC with ctypes only (no CuPy, no PyTorch, no nvcc).

The path, exactly as a CUDA programmer would do it by hand:  NVRTC compiles the generated source to PTX at run time  ->  cuModuleLoadData loads it  ->  cuMemAlloc /
cuMemcpyHtoD move the numpy buffers  ->  cuLaunchKernel runs it (grid-stride loop, so any size works)  ->  cuMemcpyDtoH brings the result back.

Layers, so that most of it can be tested without a GPU:
  RawDriver / RawNvrtc   thin ctypes wrappers: library lookup (Linux libcuda.so.1 / libnvrtc.so*, Windows nvcuda.dll / nvrtc64_*.dll), argtypes, error codes -> CudaError.
  CudaBackend            the sequence above, on any object with the same small Python API (RawDriver, or a fake).
  emulate_cuda           runs the generated CUDA source on the CPU by compiling it as C with a prelude that defines blockIdx / threadIdx / blockDim / gridDim and calling
                         the kernel once per (block, thread): it executes the ACTUAL generated text, including its index arithmetic and grid-stride loop.

HONEST STATUS (6.5): a real Tesla T4 run (Colab, 6.4) proved that the driver, NVRTC and device discovery work, and that the ONE thing that failed was the hard-coded
`--gpu-architecture=compute_70`: NVRTC from CUDA 13 no longer targets anything below sm_75 ("invalid value for --gpu-architecture"), which I reproduced here against the real
NVRTC 13.4 and 12.9 libraries. 6.5 therefore never guesses an architecture: it asks the device (cuDeviceGetAttribute), asks NVRTC which architectures it supports
(nvrtcGetSupportedArchs), prefers a real-architecture CUBIN (no driver JIT, so an NVRTC newer than the driver cannot produce PTX the driver rejects) and falls back through PTX
targets and finally NVRTC's own default, recording what it tried. What is verified without a GPU: the generated source computes the right numbers under the CPU emulation below;
the target planner against the supported-architecture lists of NVRTC 12.9 and 13.4; the real NVRTC compiling the generated source to CUBIN and PTX for every supported architecture;
CudaBackend's call order and fallback behaviour against fakes. What is NOT verified here: loading and launching that code on a real device after the 6.5 change (6.4's T4 run
stopped at compilation), performance (one thread per element, no shared memory, float64: a T4 has 1/32 double-precision rate, so do not expect it to beat CuPy's float32 path),
Windows, multiple GPUs. `python -m zerowhite kernels` runs the real thing on your machine and says what happened, including which target it chose.
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

from . import cjit


CU_ATTR_CC_MAJOR, CU_ATTR_CC_MINOR = 75, 76          # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR


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
        self.dev = None

    def init(self, device=0):
        _check(self.lib.cuInit(0), "cuInit")
        dev = ctypes.c_int(0)
        _check(self.lib.cuDeviceGet(ctypes.byref(dev), int(device)), "cuDeviceGet")
        ctx = ctypes.c_void_p()
        _check(self.lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
        self.ctx = ctx
        self.dev = dev

    def compute_capability(self):
        """(major, minor) of the device this context runs on, e.g. (7, 5) for a Tesla T4; None when the driver cannot say."""
        if self.dev is None:
            return None
        try:
            vals = []
            for attr in (CU_ATTR_CC_MAJOR, CU_ATTR_CC_MINOR):
                v = ctypes.c_int(0)
                _check(self.lib.cuDeviceGetAttribute(ctypes.byref(v), attr, self.dev), "cuDeviceGetAttribute")
                vals.append(int(v.value))
            return (vals[0], vals[1]) if vals[0] > 0 else None
        except (AttributeError, CudaError, OSError):
            return None

    def device_name(self):
        if self.dev is None:
            return None
        try:
            buf = ctypes.create_string_buffer(256)
            _check(self.lib.cuDeviceGetName(buf, 256, self.dev), "cuDeviceGetName")
            return buf.value.decode("utf-8", "replace") or None
        except (AttributeError, CudaError, OSError):
            return None

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

    def version(self):
        """(major, minor) of this NVRTC, or None."""
        try:
            a, b = ctypes.c_int(0), ctypes.c_int(0)
            if self.lib.nvrtcVersion(ctypes.byref(a), ctypes.byref(b)) == 0:
                return (int(a.value), int(b.value))
        except (AttributeError, OSError):
            pass
        return None

    def supported_archs(self):
        """The architectures this NVRTC can target, as integers (75 = sm_75), or None for an NVRTC older than 11.2 that cannot say."""
        try:
            n = ctypes.c_int(0)
            if self.lib.nvrtcGetNumSupportedArchs(ctypes.byref(n)) != 0 or n.value <= 0:
                return None
            arr = (ctypes.c_int * n.value)()
            if self.lib.nvrtcGetSupportedArchs(arr) != 0:
                return None
            return sorted(int(v) for v in arr)
        except (AttributeError, OSError):
            return None

    def compile(self, src: str, name: str = "k.cu", options=(), cubin: bool = False) -> bytes:
        """Compile `src` with the given option strings and return PTX (default) or a CUBIN (`cubin=True`, needs a real `sm_XX` architecture option)."""
        prog = ctypes.c_void_p()
        _check(self.lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode("utf-8"), name.encode("ascii"), 0, None, None), "nvrtcCreateProgram")
        try:
            opts = (ctypes.c_char_p * max(1, len(options)))(*[o.encode("ascii") for o in options])
            rc = self.lib.nvrtcCompileProgram(prog, len(options), opts)
            if rc != 0:
                size = ctypes.c_size_t()
                self.lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
                buf = ctypes.create_string_buffer(max(1, size.value))
                self.lib.nvrtcGetProgramLog(prog, buf)
                raise CudaError(f"nvrtcCompileProgram failed ({rc}): {buf.value.decode('utf-8', 'replace').strip()[:2000]}")
            size = ctypes.c_size_t()
            get_size, get = ("nvrtcGetCUBINSize", "nvrtcGetCUBIN") if cubin else ("nvrtcGetPTXSize", "nvrtcGetPTX")
            _check(getattr(self.lib, get_size)(prog, ctypes.byref(size)), get_size)
            image = ctypes.create_string_buffer(size.value)
            _check(getattr(self.lib, get)(prog, image), get)
            return image.raw
        finally:
            try:
                self.lib.nvrtcDestroyProgram(ctypes.byref(prog))
            except Exception:                                    # noqa: BLE001 - cleanup must not hide the real error
                pass


class Target:
    """One way of asking NVRTC for code: `arch` (e.g. 75, or None for NVRTC's own default) as a real-architecture CUBIN or as PTX for the driver to JIT."""
    __slots__ = ("arch", "cubin")

    def __init__(self, arch, cubin):
        self.arch, self.cubin = arch, bool(cubin)

    @property
    def options(self):
        if self.arch is None:
            return ()
        return (f"--gpu-architecture={'sm' if self.cubin else 'compute'}_{self.arch}",)

    def __str__(self):
        if self.arch is None:
            return "NVRTC default PTX"
        return f"sm_{self.arch} CUBIN" if self.cubin else f"compute_{self.arch} PTX"

    def __eq__(self, o):
        return isinstance(o, Target) and (self.arch, self.cubin) == (o.arch, o.cubin)

    def __hash__(self):
        return hash((self.arch, self.cubin))

    __repr__ = __str__


_PTX_FALLBACKS = (90, 89, 86, 80, 75, 70, 61, 60, 52, 50)


def plan_targets(cc, supported=None):
    """The ordered list of Targets to try for a device of compute capability `cc` (an int such as 75; None = unknown) given the architectures `supported` by NVRTC (None = unknown).

    Rules (each one is a failure that was seen or is documented):
      * never ask for an architecture the device is older than (PTX for compute_X cannot run on a device below X);
      * never ask for one NVRTC does not know (CUDA 13's NVRTC rejects compute_70 and below: the 6.4 Colab failure);
      * a real-architecture CUBIN first: it needs no driver JIT, so an NVRTC newer than the driver cannot hand the driver PTX it does not understand;
      * then PTX for the same architecture, then PTX for lower supported architectures (forward-compatible), and last NVRTC's own default.
    """
    out = []

    def add(t):
        if t not in out:
            out.append(t)
    if cc is not None:
        if supported:
            usable = sorted((a for a in supported if a <= cc), reverse=True)
            if usable:
                add(Target(usable[0], True))
                add(Target(usable[0], False))
                for a in usable[1:3]:
                    add(Target(a, False))
        else:
            add(Target(cc, True))
            add(Target(cc, False))
            for a in _PTX_FALLBACKS:
                if a < cc:
                    add(Target(a, False))
                    break
    add(Target(None, False))
    return out


class CudaBackend:
    BLOCK = 256

    def __init__(self, driver=None, nvrtc=None, device=0):
        self.driver = driver if driver is not None else RawDriver()
        self.nvrtc = nvrtc if nvrtc is not None else RawNvrtc()
        self.driver.init(device)
        self._fn = {}
        self.target = None                                          # the Target that worked (set by the first compile)
        self.attempts = []                                          # [(Target, error message or "ok")] of the first compile: what was tried and why it failed
        cc = getattr(self.driver, "compute_capability", lambda: None)()
        self.cc = (cc[0] * 10 + cc[1]) if cc else None
        self.device_name = getattr(self.driver, "device_name", lambda: None)()
        self.nvrtc_version = getattr(self.nvrtc, "version", lambda: None)()
        self.supported = getattr(self.nvrtc, "supported_archs", lambda: None)()

    def describe(self):
        """One line for `zerowhite kernels`: the device, the compiler, and the target that is in use."""
        dev = (self.device_name or "device") + (f" (compute capability {self.cc // 10}.{self.cc % 10})" if self.cc else "")
        nv = f"NVRTC {self.nvrtc_version[0]}.{self.nvrtc_version[1]}" if self.nvrtc_version else "NVRTC"
        return f"{dev}, {nv}, target {self.target if self.target is not None else 'not chosen yet'}"

    def _build(self, src: str):
        """Compile `src` for this device and load it: the first planned target that both compiles AND loads wins; the reasons for the others are kept in `self.attempts`."""
        tried = []
        for t in plan_targets(self.cc, self.supported):
            try:
                image = self.nvrtc.compile(src, "k.cu", options=t.options, cubin=t.cubin)
                mod = self.driver.module_load(image)
                fn = self.driver.get_function(mod, "k")
            except CudaError as ex:
                tried.append((t, str(ex).replace("\n", " ")[:300]))
                continue
            tried.append((t, "ok"))
            self.attempts = tried
            if self.target is None:
                self.target = t
            return fn
        self.attempts = tried
        hint = ""
        if self.cc and self.supported and min(self.supported) > self.cc:
            hint = (f" This device is sm_{self.cc} but the installed NVRTC {self.nvrtc_version[0]}.{self.nvrtc_version[1] if self.nvrtc_version else ''} only targets sm_{min(self.supported)} and newer "
                    f"(CUDA 13 dropped Maxwell, Pascal and Volta): install an NVRTC from CUDA 12.")
        raise CudaError("no NVRTC target worked: " + "; ".join(f"{t}: {why}" for t, why in tried) + "." + hint)

    def run(self, src: str, arrs, out: np.ndarray):
        if src not in self._fn:
            self._fn[src] = self._build(src)
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
