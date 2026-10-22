"""winjob.py - hard resource limits for sandboxed child processes on Windows, through a Job Object (ctypes only, no pywin32).

POSIX gets CPU-time / memory / file-size limits from `resource.setrlimit`. Windows has no `resource` module, so until 6.3 a runaway
evolved program there was limited by the wall-clock timeout alone. A Job Object is the Windows mechanism that does the same job:
every process assigned to it (and every process THEY start) is counted together, and the kernel enforces

    JOB_OBJECT_LIMIT_JOB_MEMORY        committed memory of the whole tree <= mem_mb
    JOB_OBJECT_LIMIT_PROCESS_TIME      user CPU time of each process <= cpu_s
    JOB_OBJECT_LIMIT_ACTIVE_PROCESS    optional cap on how many processes may exist at once
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE the whole tree dies when we close the handle (also if this Python process crashes)

Like safety.py and sandbox.py this file is part of the CONSTITUTION: the self-modifier may not edit it (see CONSTITUTION.sha256).

HONEST LIMITS (also in docs/SELF_MODIFICATION.md)
  * The structure layouts and the call sequence are tested on every OS (struct sizes against the documented Windows values, the calls
    against a fake kernel32). The enforcement itself has NOT been run on a Windows machine by the author. `doctor` runs a live
    self-test there: a child that allocates past the limit must be killed.
  * Between Popen() returning and the assignment the child runs unconstrained for a few milliseconds; a child that spawns another
    process in that window escapes the job. Python start-up takes longer than that, so for `python -B script.py` children this does not
    happen in practice, but it is a window, not a guarantee.
  * Job memory counts committed memory, not address space (RLIMIT_AS), so the two OSes' limits are comparable, not identical.
  * Nested jobs need Windows 8 / Server 2012 or later; on older systems assignment fails and the caller falls back to wall-clock only.
"""
from __future__ import annotations

import ctypes
import os

# ---- constants (winnt.h)
JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

# Fixed-width types, NOT ctypes.wintypes: wintypes.DWORD is c_ulong, which is 8 bytes on 64-bit Linux, so the layout could not be
# checked anywhere but Windows. These have the Windows sizes on every 64-bit (and 32-bit) platform with natural alignment.
_DWORD = ctypes.c_uint32
_LARGE_INTEGER = ctypes.c_int64
_ULONGLONG = ctypes.c_uint64
_SIZE_T = ctypes.c_size_t                       # pointer-sized, like SIZE_T and ULONG_PTR


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [("ReadOperationCount", _ULONGLONG), ("WriteOperationCount", _ULONGLONG), ("OtherOperationCount", _ULONGLONG),
                ("ReadTransferCount", _ULONGLONG), ("WriteTransferCount", _ULONGLONG), ("OtherTransferCount", _ULONGLONG)]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", _LARGE_INTEGER), ("PerJobUserTimeLimit", _LARGE_INTEGER), ("LimitFlags", _DWORD),
                ("MinimumWorkingSetSize", _SIZE_T), ("MaximumWorkingSetSize", _SIZE_T), ("ActiveProcessLimit", _DWORD),
                ("Affinity", _SIZE_T), ("PriorityClass", _DWORD), ("SchedulingClass", _DWORD)]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", _SIZE_T), ("JobMemoryLimit", _SIZE_T), ("PeakProcessMemoryUsed", _SIZE_T), ("PeakJobMemoryUsed", _SIZE_T)]


def build_limits(mem_mb=None, cpu_s=None, active_processes=None, kill_on_close=True) -> JOBOBJECT_EXTENDED_LIMIT_INFORMATION:
    """The limit structure for the given budget (pure data: usable, and tested, on any OS)."""
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    flags = JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
    if kill_on_close:
        flags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if mem_mb:
        flags |= JOB_OBJECT_LIMIT_JOB_MEMORY
        info.JobMemoryLimit = int(mem_mb) * 2 ** 20
    if cpu_s:
        flags |= JOB_OBJECT_LIMIT_PROCESS_TIME
        info.BasicLimitInformation.PerProcessUserTimeLimit = int(cpu_s) * 10_000_000          # 100-nanosecond units
    if active_processes:
        flags |= JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        info.BasicLimitInformation.ActiveProcessLimit = int(active_processes)
    info.BasicLimitInformation.LimitFlags = flags
    return info


def _kernel32():
    if os.name != "nt":
        return None
    try:
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.restype = ctypes.c_void_p
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        k.SetInformationJobObject.restype = ctypes.c_int
        k.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, _DWORD]
        k.AssignProcessToJobObject.restype = ctypes.c_int
        k.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        k.TerminateJobObject.restype = ctypes.c_int
        k.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k.CloseHandle.restype = ctypes.c_int
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        return k
    except (OSError, AttributeError):
        return None


def available(api=None) -> bool:
    """True when Job Objects can be used here (Windows, kernel32 loadable). `api` lets tests inject a fake kernel32."""
    return (api if api is not None else _kernel32()) is not None


class JobObject:
    """One job. `JobObject(...).assign(popen)` puts the child (and everything it starts) under the limits. Never raises:
    every failure returns False, because a sandbox that cannot add a limit must still run the child with the limits it has."""

    def __init__(self, mem_mb=None, cpu_s=None, active_processes=None, kill_on_close=True, api=None):
        self.api = api if api is not None else _kernel32()
        self.handle = None
        self.error = None
        if self.api is None:
            self.error = "Job Objects are not available on this platform"
            return
        try:
            h = self.api.CreateJobObjectW(None, None)
            if not h:
                self.error = f"CreateJobObjectW failed ({ctypes.get_last_error() if os.name == 'nt' else '?'})"
                return
            info = build_limits(mem_mb, cpu_s, active_processes, kill_on_close)
            if not self.api.SetInformationJobObject(h, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)):
                self.error = "SetInformationJobObject refused the limits"
                self.api.CloseHandle(h)
                return
            self.handle = h
        except Exception as ex:                                  # noqa: BLE001 - never let the sandbox's own safety net crash a run
            self.error = f"{type(ex).__name__}: {ex}"
            self.handle = None

    @property
    def ok(self) -> bool:
        return self.handle is not None

    def assign(self, popen) -> bool:
        if not self.ok:
            return False
        try:
            ph = getattr(popen, "_handle", None)                 # subprocess.Popen on Windows keeps the process handle here
            if ph is None:
                self.error = "the child has no process handle"
                return False
            if not self.api.AssignProcessToJobObject(self.handle, int(ph)):
                self.error = "AssignProcessToJobObject failed (the child may already be in a job on a pre-Windows-8 system)"
                return False
            return True
        except Exception as ex:                                  # noqa: BLE001
            self.error = f"{type(ex).__name__}: {ex}"
            return False

    def terminate(self, exit_code: int = 1) -> bool:
        if not self.ok:
            return False
        try:
            return bool(self.api.TerminateJobObject(self.handle, exit_code))
        except Exception:                                        # noqa: BLE001
            return False

    def close(self):
        if self.ok:
            try:
                self.api.CloseHandle(self.handle)                # with KILL_ON_JOB_CLOSE this also ends anything still running
            except Exception:                                    # noqa: BLE001
                pass
            self.handle = None
