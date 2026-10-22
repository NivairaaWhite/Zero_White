"""sandbox.py - run self-written or self-patched code somewhere it can do little harm.

Ideas taken from nsjail / judge0 / E2B (process isolation, wall-clock and resource limits, a
scrubbed environment, a throw-away working directory) at the level a pure-Python, cross-platform
package can honestly provide:

  * every run is a separate Python process in a temporary directory, with a minimal environment;
  * hard wall-clock timeout; on timeout the WHOLE process tree is killed (taskkill /T on Windows,
    a process group on POSIX);
  * on POSIX: CPU-time, memory and file-size rlimits (the `resource` module does not exist on Windows);
  * on Windows (6.3): a Job Object (selfmod/winjob.py) gives the whole process tree a memory and a CPU-time limit and kills it when the run
    ends; if the job cannot be created or assigned the run still happens, under the wall-clock timeout alone (`sandbox_report()` says which);
  * on Linux with `unshare` permitted: no network namespace (best effort, detected at runtime).

HONEST LIMIT: this is containment against evolved-code accidents (runaway loops, huge allocations,
stray writes), NOT a security boundary against a determined attacker. A real boundary needs
nsjail, gVisor or a container. `sandbox_report()` says exactly what is enforced on this machine.
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

try:                                    # POSIX only
    import resource
except ImportError:                     # Windows
    resource = None

IS_WINDOWS = os.name == "nt"
from . import winjob                    # noqa: E402  (stdlib + ctypes only; harmless to import on every OS)
_COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".tmp_*", ".zw_promotion", "runs", "zw", "workspace*",
                                      "*.egg-info", "build", "dist", ".pytest_cache")


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float

    @property
    def ok(self):
        return self.returncode == 0 and not self.timed_out


_UNSHARE = {}


def _can_unshare_net():
    if "ok" not in _UNSHARE:
        ok = False
        exe = shutil.which("unshare")
        if exe and not IS_WINDOWS:
            try:
                ok = subprocess.run([exe, "-rn", "true"], capture_output=True, timeout=5).returncode == 0
            except Exception:                                    # noqa: BLE001
                ok = False
        _UNSHARE["ok"] = ok
    return _UNSHARE["ok"]


def sandbox_report():
    job = IS_WINDOWS and winjob.available()
    return dict(platform=sys.platform, process_isolation=True, wall_clock_timeout=True, kills_process_tree=True,
                rlimits=resource is not None or job,                  # "resource limits are enforced": rlimits on POSIX, a Job Object on Windows
                job_object=job, network_namespace=_can_unshare_net(),
                note="containment for evolved-code accidents, not a security boundary")


def _minimal_env(workdir, extra=None):
    keep = ("PATH", "SYSTEMROOT", "SystemRoot", "COMSPEC", "PATHEXT", "TEMP", "TMP", "LANG", "LC_ALL")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update(HOME=workdir, USERPROFILE=workdir, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1",
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1", ZERO_WHITE_DATA=os.environ.get("ZERO_WHITE_DATA", ""))
    env.update(extra or {})
    return {k: v for k, v in env.items() if v != ""}


def _limits(cpu_s, mem_mb, fsize_mb):
    def apply():
        os.setsid()
        if cpu_s:
            resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_s), int(cpu_s) + 5))
        if mem_mb:
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2 ** 20, mem_mb * 2 ** 20))
            except (ValueError, OSError):
                pass
        if fsize_mb:
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_mb * 2 ** 20, fsize_mb * 2 ** 20))
    return apply


def _kill_tree(p):
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True, timeout=15)
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:                                            # noqa: BLE001
        pass
    try:
        p.kill()
    except Exception:                                            # noqa: BLE001
        pass


def run_python(args, cwd, timeout=120, env=None, mem_mb=4096, fsize_mb=256, no_network=True):
    """Run `python <args>` in `cwd` under the limits above and return a RunResult."""
    cmd = [sys.executable, "-B"] + list(args)
    if no_network and _can_unshare_net():
        cmd = [shutil.which("unshare"), "-rn"] + cmd
    kw = {}
    if IS_WINDOWS:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    elif resource is not None:
        kw["preexec_fn"] = _limits(int(timeout * 2) + 10, mem_mb, fsize_mb)
    else:
        kw["start_new_session"] = True
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=_minimal_env(cwd, env), stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         encoding="utf-8", errors="replace", **kw)
    job = None
    if IS_WINDOWS:                                               # 6.3: memory / CPU limits for the whole tree (best effort, see winjob.py)
        job = winjob.JobObject(mem_mb=mem_mb, cpu_s=int(timeout * 2) + 10)
        job.assign(p)
    try:
        out, err = p.communicate(timeout=timeout)
        return RunResult(p.returncode, out, err, False, time.time() - t0)
    except subprocess.TimeoutExpired:
        if job is not None:
            job.terminate()
        _kill_tree(p)
        try:
            out, err = p.communicate(timeout=10)
        except Exception:                                        # noqa: BLE001
            out, err = "", ""
        return RunResult(-9, out or "", (err or "") + "\n[sandbox] killed: wall-clock timeout", True, time.time() - t0)
    finally:
        if job is not None:
            job.close()


def run_code(source, timeout=20, **kw):
    """Run a snippet of Python source in a fresh temp directory."""
    with tempfile.TemporaryDirectory(prefix="zw_sbx_") as d:
        path = os.path.join(d, "snippet.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        return run_python([path], cwd=d, timeout=timeout, **kw)


class SandboxCopy:
    """A throw-away copy of a source tree: patch and test it here, never in the real tree."""

    def __init__(self, src_root):
        self.src_root = os.path.abspath(src_root)
        self.dir = tempfile.mkdtemp(prefix="zw_copy_")
        self.root = os.path.join(self.dir, "tree")
        shutil.copytree(self.src_root, self.root, ignore=_COPY_IGNORE)

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
