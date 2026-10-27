"""Tests for 6.5: the three failures from the 6.4 CI / Colab logs (NVRTC architecture, macOS memory limit; the settle-wait tests live in test_chat.py), and the new abilities.
Same rule as test_v63 / test_v64: each test must fail when the property it names breaks."""
import ast
import ctypes
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ZERO_WHITE_DATA", tempfile.mkdtemp(prefix="zw_test_data_"))

from zerowhite import fileio                                           # noqa: E402

_DIRS = []


def tmp():
    d = tempfile.mkdtemp(prefix="zw_t65_")
    _DIRS.append(d)
    return d


import atexit                                                          # noqa: E402
atexit.register(lambda: [shutil.rmtree(d, ignore_errors=True) for d in _DIRS])


# ================================================================================================================ CUDA architecture (the Colab T4 failure)
NVRTC_13_4 = [75, 80, 86, 87, 88, 89, 90, 100, 103, 107, 110, 120, 121]          # what the real NVRTC 13.4 reported (nvrtcGetSupportedArchs), measured in the build environment
NVRTC_12_9 = [50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 101, 103, 120, 121]   # ... and NVRTC 12.9


def test_the_target_planner_never_asks_for_an_architecture_the_device_or_the_compiler_cannot_use():
    from zerowhite.kernels import cuda
    T = cuda.Target
    # the exact 6.4 failure: a Tesla T4 (7.5) with CUDA 13's NVRTC, which rejects compute_70 and everything below sm_75
    plan = cuda.plan_targets(75, NVRTC_13_4)
    assert plan[0] == T(75, True) and plan[1] == T(75, False) and plan[-1] == T(None, False)
    for supported, cc in ((NVRTC_13_4, 75), (NVRTC_13_4, 86), (NVRTC_13_4, 120), (NVRTC_12_9, 70), (NVRTC_12_9, 75), (NVRTC_12_9, 89), (NVRTC_12_9, 60)):
        for t in cuda.plan_targets(cc, supported):
            assert t.arch is None or (t.arch in supported and t.arch <= cc), (cc, t)       # never newer than the device, never unknown to the compiler
    assert cuda.plan_targets(70, NVRTC_12_9)[0] == T(70, True) and "--gpu-architecture=sm_70" in cuda.plan_targets(70, NVRTC_12_9)[0].options
    assert T(75, False).options == ("--gpu-architecture=compute_75",) and T(None, False).options == ()
    # a compiler that cannot list its architectures: start from the device's own, then lower PTX, then NVRTC's default
    assert cuda.plan_targets(75, None)[:2] == [T(75, True), T(75, False)] and cuda.plan_targets(75, None)[-1] == T(None, False)
    # a device we could not identify: only NVRTC's own default is safe to ask for
    assert cuda.plan_targets(None, NVRTC_13_4) == [T(None, False)]
    # a device older than everything the compiler targets: nothing sensible but the default (which the backend then reports as unusable, see below)
    assert cuda.plan_targets(70, NVRTC_13_4) == [T(None, False)]
    assert 70 not in [t.arch for t in cuda.plan_targets(75, NVRTC_13_4)]                    # compute_70 was the 6.4 hard-coded value


class _CcDriver:
    """tests/test_v64's fake driver, plus what 6.5 asks of a real one: the device's compute capability and name."""
    def __init__(self, cc=(7, 5), name="Tesla T4", fail_load=()):
        from test_v64 import _FakeDriver
        self.inner = _FakeDriver()
        self.cc, self.name, self.fail_load = cc, name, set(fail_load)
        self.loaded = []

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def compute_capability(self):
        return self.cc

    def device_name(self):
        return self.name

    def module_load(self, image):
        from zerowhite.kernels import cuda
        head = image.split(b"\n", 2)
        arch = int(head[0].split()[1])
        self.loaded.append(head[0])
        if arch > self.cc[0] * 10 + self.cc[1]:
            raise cuda.CudaError("cuModuleLoadData failed with code 209")                       # CUDA_ERROR_NO_BINARY_FOR_GPU: the image targets a newer architecture than this device
        if head[1].startswith(b"//PTXNEW"):
            raise cuda.CudaError("cuModuleLoadData failed with code 222")                       # CUDA_ERROR_UNSUPPORTED_PTX_VERSION: the driver is older than the compiler
        return self.inner.module_load(image)


class _StrictNvrtc:
    """Behaves like CUDA 13's NVRTC: only the listed architectures exist, everything else is 'invalid value for --gpu-architecture'."""
    def __init__(self, supported, version=(13, 4), cubin_ok=True, ptx_marker=False):
        self.supported, self._version, self.cubin_ok, self.ptx_marker = list(supported), version, cubin_ok, ptx_marker
        self.requests = []

    def version(self):
        return self._version

    def supported_archs(self):
        return list(self.supported)

    def compile(self, src, name="k.cu", options=(), cubin=False):
        from zerowhite.kernels import cuda
        self.requests.append((tuple(options), cubin))
        for o in options:
            arch = int(o.split("_")[-1])
            if arch not in self.supported:
                raise cuda.CudaError("nvrtcCompileProgram failed (5): nvrtc: error: invalid value for --gpu-architecture (-arch)")
        if cubin and not self.cubin_ok:
            raise cuda.CudaError("nvrtcGetCUBINSize failed with code 1")
        assert '__global__' in src and "gridDim.x" in src
        arch = int(options[0].split("_")[-1]) if options else min(self.supported)                # NVRTC's default architecture is its oldest supported one here
        head = f"//ARCH {arch} {'cubin' if cubin else 'ptx'}\n" + ("//PTXNEW\n" if (self.ptx_marker and not cubin) else "//-\n")
        return head.encode() + src.encode()


def _small_graph():
    from zerowhite.kernels import graph as G
    r = np.random.default_rng(3)
    a, b = r.normal(size=(6, 11)), r.normal(size=(11,))
    return ((G.lazy(a) * G.lazy(b) + 1.0).tanh() - 0.25).abs()


def test_a_t4_with_cuda13s_compiler_gets_a_sm75_cubin_and_the_right_numbers_where_6_4_failed():
    from zerowhite.kernels import cjit, cuda, graph as G
    if not cjit.available():
        return
    g = _small_graph()
    nv, drv = _StrictNvrtc(NVRTC_13_4), _CcDriver((7, 5))
    be = cuda.CudaBackend(driver=drv, nvrtc=nv)
    assert be.cc == 75 and be.device_name == "Tesla T4" and be.nvrtc_version == (13, 4) and be.supported == NVRTC_13_4
    got = G.evaluate(g, "cuda", cuda=be)
    assert np.allclose(got, G.evaluate(g, "numpy"), atol=1e-12)
    assert nv.requests[0] == (("--gpu-architecture=sm_75",), True), nv.requests                # the first (and only) request was the device's own real architecture
    assert all("compute_70" not in o for opts, _ in nv.requests for o in opts)
    assert str(be.target) == "sm_75 CUBIN" and "Tesla T4" in be.describe() and "7.5" in be.describe() and "sm_75 CUBIN" in be.describe()
    G.evaluate(g, "cuda", cuda=be)
    assert len(nv.requests) == 1                                                                # compiled once, cached by source


def test_when_a_target_compiles_but_will_not_load_the_next_one_is_tried_and_every_failure_is_kept():
    from zerowhite.kernels import cjit, cuda, graph as G
    if not cjit.available():
        return
    g = _small_graph()
    # a compiler without CUBIN output (NVRTC < 11.1), and a driver that rejects the compiler's PTX version: the default PTX target is not used because
    # the compute_75 PTX loads... so make PTX for compute_75 fail at load and let the NVRTC-default target win
    nv = _StrictNvrtc(NVRTC_13_4, cubin_ok=False)
    be = cuda.CudaBackend(driver=_CcDriver((7, 5)), nvrtc=nv)
    got = G.evaluate(g, "cuda", cuda=be)
    assert np.allclose(got, G.evaluate(g, "numpy"), atol=1e-12)
    assert str(be.target) == "compute_75 PTX"
    assert [str(t) for t, why in be.attempts] == ["sm_75 CUBIN", "compute_75 PTX"] and be.attempts[0][1] != "ok" and "CUBIN" in be.attempts[0][1] and be.attempts[1][1] == "ok"
    # the driver refuses compiler-new PTX (CUDA_ERROR_UNSUPPORTED_PTX_VERSION): with a CUBIN available that never happens, which is why the CUBIN comes first
    nv2 = _StrictNvrtc(NVRTC_13_4, cubin_ok=True, ptx_marker=True)
    be2 = cuda.CudaBackend(driver=_CcDriver((7, 5)), nvrtc=nv2)
    assert np.allclose(G.evaluate(g, "cuda", cuda=be2), G.evaluate(g, "numpy"), atol=1e-12) and str(be2.target) == "sm_75 CUBIN"
    nv3 = _StrictNvrtc(NVRTC_13_4, cubin_ok=False, ptx_marker=True)
    try:
        G.evaluate(g, "cuda", cuda=cuda.CudaBackend(driver=_CcDriver((7, 5)), nvrtc=nv3))
        raise AssertionError("every target failing must raise")
    except cuda.CudaError as ex:
        assert "no NVRTC target worked" in str(ex) and "sm_75 CUBIN" in str(ex) and "compute_75 PTX" in str(ex) and "code 222" in str(ex), str(ex)


def test_a_device_older_than_the_installed_compiler_is_told_what_to_install():
    from zerowhite.kernels import cjit, cuda, graph as G
    if not cjit.available():
        return
    be = cuda.CudaBackend(driver=_CcDriver((7, 0), "Tesla V100"), nvrtc=_StrictNvrtc(NVRTC_13_4))
    try:
        G.evaluate(_small_graph(), "cuda", cuda=be)
        raise AssertionError("a V100 with CUDA 13's NVRTC cannot work")
    except cuda.CudaError as ex:
        assert "sm_70" in str(ex) and "only targets sm_75 and newer" in str(ex) and "CUDA 12" in str(ex), str(ex)
    ok = cuda.CudaBackend(driver=_CcDriver((7, 0), "Tesla V100"), nvrtc=_StrictNvrtc(NVRTC_12_9, version=(12, 9)))
    assert np.allclose(G.evaluate(_small_graph(), "cuda", cuda=ok), G.evaluate(_small_graph(), "numpy"), atol=1e-12) and str(ok.target) == "sm_70 CUBIN"


class _AttrLib:
    """libcuda / libnvrtc stand-in for the new queries: fills out-parameters the way the C API does."""
    def __init__(self, cc=(7, 5), archs=(75, 80), version=(13, 4), fail=()):
        self.cc, self.archs, self.version, self.fail, self.calls = cc, list(archs), version, set(fail), []

    def cuInit(self, f): return 0                                                              # noqa: E704
    def cuDeviceGet(self, ref, i): ref._obj.value = 3; return 0                                # noqa: E704
    def cuCtxCreate_v2(self, ref, flags, dev): ref._obj.value = 9; return 0                    # noqa: E704

    def cuDeviceGetAttribute(self, ref, attr, dev):
        self.calls.append(("attr", attr, dev.value))
        if "attr" in self.fail:
            return 1
        ref._obj.value = {75: self.cc[0], 76: self.cc[1]}[attr]
        return 0

    def cuDeviceGetName(self, buf, n, dev):
        ctypes.memmove(buf, b"Fake GPU\0", 9)
        return 0

    def nvrtcVersion(self, a, b): a._obj.value, b._obj.value = self.version; return 0         # noqa: E704
    def nvrtcGetNumSupportedArchs(self, ref): ref._obj.value = len(self.archs); return 0       # noqa: E704

    def nvrtcGetSupportedArchs(self, arr):
        for i, v in enumerate(self.archs):
            arr[i] = v
        return 0


def test_the_ctypes_layers_query_compute_capability_name_version_and_supported_architectures():
    from zerowhite.kernels import cuda
    d = cuda.RawDriver(_AttrLib(cc=(8, 6)))
    assert d.compute_capability() is None and d.device_name() is None                           # before init there is no device
    d.init(0)
    assert d.compute_capability() == (8, 6) and d.device_name() == "Fake GPU"
    assert d.lib.calls == [("attr", 75, 3), ("attr", 76, 3)]                                    # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR on the device handle
    bad = cuda.RawDriver(_AttrLib(fail=("attr",)))
    bad.init(0)
    assert bad.compute_capability() is None                                                    # an error is "unknown", never an exception
    nv = cuda.RawNvrtc(_AttrLib(archs=(75, 80, 86), version=(13, 4)))
    assert nv.version() == (13, 4) and nv.supported_archs() == [75, 80, 86]
    old = cuda.RawNvrtc(object())                                                              # an NVRTC that predates the query (no such functions)
    assert old.version() is None and old.supported_archs() is None


def test_the_nvrtc_program_is_destroyed_even_when_compilation_fails():
    from zerowhite.kernels import cuda

    class Lib:
        def __init__(self): self.destroyed = 0; self.ptx = b"PTX\0"
        def nvrtcCreateProgram(self, ref, src, name, n, h, i): ref._obj.value = 5; return 0     # noqa: E704
        def nvrtcCompileProgram(self, prog, n, opts): return 5 if b"bad" in [o for o in opts] else 0   # noqa: E704
        def nvrtcGetProgramLogSize(self, prog, ref): ref._obj.value = 11; return 0               # noqa: E704
        def nvrtcGetProgramLog(self, prog, buf): ctypes.memmove(buf, b"bad option\0", 11); return 0   # noqa: E704
        def nvrtcGetPTXSize(self, prog, ref): ref._obj.value = 4; return 0                       # noqa: E704
        def nvrtcGetPTX(self, prog, buf): ctypes.memmove(buf, self.ptx, 4); return 0             # noqa: E704
        def nvrtcGetCUBINSize(self, prog, ref): ref._obj.value = 3; return 0                     # noqa: E704
        def nvrtcGetCUBIN(self, prog, buf): ctypes.memmove(buf, b"CUB", 3); return 0             # noqa: E704
        def nvrtcDestroyProgram(self, ref): self.destroyed += 1; return 0                        # noqa: E704
    lib = Lib()
    nv = cuda.RawNvrtc(lib)
    assert nv.compile("x", options=("--ok",)) == b"PTX\0" and nv.compile("x", options=("--ok",), cubin=True) == b"CUB" and lib.destroyed == 2
    lib.nvrtcCompileProgram = lambda prog, n, opts: 5
    try:
        nv.compile("x", options=("--gpu-architecture=compute_70",))
        raise AssertionError("must raise")
    except cuda.CudaError as ex:
        assert "bad option" in str(ex)
    assert lib.destroyed == 3, lib.destroyed                                                    # 6.4 leaked the program on every failed compile


def _real_nvrtc():
    """A real NVRTC if this machine has one (the CUDA toolkit, or `ZW_TEST_NVRTC=/path/to/libnvrtc.so.N`); None otherwise, which is the case on CI runners."""
    from zerowhite.kernels import cuda
    path = os.environ.get("ZW_TEST_NVRTC")
    if path and os.path.exists(path):
        return cuda.RawNvrtc(ctypes.CDLL(path))
    try:
        return cuda.RawNvrtc()
    except cuda.CudaError:
        return None


def test_a_real_nvrtc_if_present_compiles_the_generated_source_for_every_architecture_it_supports():
    from zerowhite.kernels import cuda, graph as G
    nv = _real_nvrtc()
    if nv is None:
        return
    g = _small_graph()
    order, cons, is_root = G.plan_regions(g)
    _, src = G.emit_cuda(g, cons, is_root)
    archs = nv.supported_archs()
    assert archs and nv.version()
    for a in archs:
        cub = nv.compile(src, options=(f"--gpu-architecture=sm_{a}",), cubin=True)
        assert cub[:4] == b"\x7fELF", a                                                         # a real CUBIN is an ELF file
        assert b".entry" in nv.compile(src, options=(f"--gpu-architecture=compute_{a}",)) or b"visible .entry k" in nv.compile(src, options=(f"--gpu-architecture=compute_{a}",))
    for cc in (60, 70, 75, 80, 86, 89, 90):                                                     # whatever the device, the planned FIRST usable target compiles
        plan = cuda.plan_targets(cc, archs)
        ok = False
        for t in plan:
            try:
                nv.compile(src, options=t.options, cubin=t.cubin)
                ok = True
                break
            except cuda.CudaError:
                continue
        assert ok or (min(archs) > cc), (cc, plan)


# ================================================================================================================ memory limit (the macOS failure)
def test_which_systems_need_the_parents_watcher_and_which_enforce_a_limit_themselves():
    from zerowhite.selfmod import memwatch
    assert memwatch.needed("darwin") and not memwatch.needed("linux") and not memwatch.needed("freebsd13") and not memwatch.needed("win32")
    import types
    real_os = memwatch.os
    try:
        memwatch.os = types.SimpleNamespace(name="nt")                                          # a Windows HOST must still answer for the platform it is asked about (it used to say False for "darwin")
        assert memwatch.needed("darwin") and not memwatch.needed("linux") and not memwatch.needed("win32")
    finally:
        memwatch.os = real_os


class _FakeProcLib:
    """libproc stand-in: proc_pid_rusage fills the buffer the way the kernel does, with distinct values in distinct fields so a wrong offset is visible."""
    def __init__(self, resident=111 * 2 ** 20, footprint=222 * 2 ** 20, rc=0):
        self.resident, self.footprint, self.rc, self.calls = resident, footprint, rc, []

    def proc_pid_rusage(self, pid, flavor, ref):
        self.calls.append((pid.value, flavor.value))
        if self.rc:
            return self.rc
        buf = ref._obj
        off = 16
        for field in range(10):                                        # ri_user_time ... ri_proc_exit_abstime, in header order
            v = {6: self.resident, 7: self.footprint}.get(field, 1000 + field)
            for i, byte in enumerate(int(v).to_bytes(8, sys.byteorder)):
                buf[off + 8 * field + i] = byte
        return 0


def test_the_libproc_structure_layout_is_read_at_the_right_offsets_and_failures_are_none_not_exceptions():
    from zerowhite.selfmod import memwatch
    lib = _FakeProcLib()
    s = memwatch.libproc_sampler(lib)
    assert s(4242) == 222 * 2 ** 20                                   # phys_footprint (offset 72), not resident_size (offset 64), not a neighbouring counter
    assert lib.calls == [(4242, 0)]                                  # RUSAGE_INFO_V0
    assert memwatch.libproc_sampler(_FakeProcLib(footprint=0))(1) == 111 * 2 ** 20            # no footprint reported: resident size
    assert memwatch.libproc_sampler(_FakeProcLib(rc=-1))(1) is None                            # the process is gone / not allowed: unknown
    assert memwatch.parse_rusage_v0(bytes(range(96)))[0] == int.from_bytes(bytes(range(64, 72)), sys.byteorder)
    try:
        memwatch.parse_rusage_v0(b"\0" * 10)
        raise AssertionError("a short buffer must be refused")
    except ValueError:
        pass


def test_the_watcher_trips_once_at_the_limit_and_never_on_a_process_below_it():
    from zerowhite.selfmod import memwatch
    calls = []
    seq = iter([10, 50, 100, 300, 900, 900])
    w = memwatch.MemoryWatch(1, 200, lambda: calls.append(1), sampler=lambda pid: next(seq), interval=0.001)
    w.start()
    w.join(3)
    assert calls == [1] and w.tripped == 300 and w.peak == 300 and w.samples == 4          # stops at the first sample over the limit
    quiet = memwatch.MemoryWatch(1, 200, lambda: calls.append(2), sampler=lambda pid: 150, interval=0.001)
    quiet.start()
    deadline = time.time() + 10                                                              # Windows timers tick at about 15 ms, so 1 ms waits take far longer than 1 ms: wait for the samples, not for the clock
    while quiet.samples <= 5 and time.time() < deadline:
        time.sleep(0.01)
    quiet.stop()
    assert calls == [1] and quiet.tripped is None and quiet.samples > 5
    gone = memwatch.MemoryWatch(1, 200, lambda: calls.append(3), sampler=lambda pid: None, interval=0.001)
    gone.start(); time.sleep(0.1); gone.stop()
    assert calls == [1] and gone.tripped is None                                              # a process that has exited reads as None: no kill, no crash


_HOG = "b = bytearray(900 * 2**20); b[::4096] = bytes(len(b[::4096])); print('alive')"


def test_with_only_the_parents_watcher_a_900_mb_child_under_a_256_mb_limit_is_killed_and_a_small_one_is_not():
    from zerowhite.selfmod import sandbox
    ok = sandbox.run_code("b = bytearray(16 * 2**20); print('alive')", mem_mb=256, memory="watcher")
    hog = sandbox.run_code(_HOG, mem_mb=256, timeout=30, memory="watcher")                  # RLIMIT_AS is NOT set in this mode: only the watcher can stop it
    assert ok.ok and ok.stdout.strip() == "alive"
    assert hog.stdout.strip() != "alive" and not hog.ok and hog.returncode != 0 and not hog.timed_out
    if sandbox.IS_WINDOWS:
        return                                                                                  # Windows has no parent watcher: the Job Object (winjob.py) stops the hog, silently
    assert "[sandbox] killed: memory limit" in hog.stderr and "> 256 MB" in hog.stderr, hog.stderr[-300:]


def test_the_watcher_also_stops_a_hog_that_keeps_its_memory_when_the_slow_ps_fallback_is_the_only_sensor():
    from zerowhite.selfmod import memwatch
    if not memwatch.ps_sampler()(os.getpid()):
        return                                                                                  # no usable `ps` here
    p = subprocess.Popen([sys.executable, "-c", "b = bytearray(600 * 2**20); import time; time.sleep(20); print('alive')"], stdout=subprocess.PIPE, text=True, start_new_session=True)
    w = memwatch.MemoryWatch(p.pid, 256 * 2 ** 20, p.kill, sampler=memwatch.ps_sampler(), interval=0.02)
    w.start()
    t0 = time.time()
    out = p.communicate(timeout=30)[0]
    w.stop()
    assert out.strip() != "alive" and w.tripped and w.tripped > 256 * 2 ** 20 and time.time() - t0 < 15


def test_the_report_names_the_mechanism_and_says_none_when_nothing_can_measure():
    from zerowhite.selfmod import sandbox, memwatch
    assert sandbox.sandbox_report()["memory_limit"] in ("rlimit_as", "job_object", "watcher")
    if sandbox.IS_WINDOWS:                                                                      # the Job Object is the only mechanism there, whatever is asked for
        assert sandbox.memory_mechanism("watcher") == sandbox.memory_mechanism("rlimit") == sandbox.memory_mechanism("auto") and sandbox.memory_mechanism() in ("job_object", "none")
    else:
        assert sandbox.memory_mechanism("watcher") == "watcher" and (sandbox.memory_mechanism("rlimit") in ("rlimit_as", "none"))
    real_needed, real_make = memwatch.needed, memwatch.make_sampler
    try:
        memwatch.needed = lambda platform=None: True                                            # pretend to be macOS
        memwatch.make_sampler = lambda platform=None: (lambda pid: 1, "fake", 0.01)
        if not sandbox.IS_WINDOWS:
            assert sandbox.memory_mechanism("auto") == "watcher"
        memwatch.make_sampler = lambda platform=None: (None, "none", 0.05)
        if not sandbox.IS_WINDOWS:
            assert sandbox.memory_mechanism("auto") == "none"                                   # honest: no sensor, no claim
    finally:
        memwatch.needed, memwatch.make_sampler = real_needed, real_make


def test_the_default_memory_limit_check_of_doctor_passes_on_this_machine_and_names_its_mechanism():
    from zerowhite import doctor
    out = []
    doctor.check_sandbox(out)
    assert not [f for f in out if f.level in ("fail", "warn") and "memory" in f.msg], [f.msg for f in out]
    assert any("memory limit verified" in f.msg and "mechanism:" in f.msg for f in out), [f.msg for f in out]
