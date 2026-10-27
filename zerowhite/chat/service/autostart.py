"""autostart.py - start the background service when you log in (Windows Task Scheduler, systemd user unit, macOS LaunchAgent).

`show` prints exactly what would be installed; `install` writes it and registers it; `remove` undoes it.
Honest note: these were written from the platforms' documentation and could not be run on Windows or macOS in the build environment.
"""
from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from typing import List, Tuple

from ... import fileio
from ..config import Config
from ..paths import Layout, safe_name
from .daemon import PACKAGE_PARENT


def _name(lay: Layout) -> str:
    return "ZeroWhite-" + safe_name(os.path.basename(lay.workspace.rstrip("/\\")) or "chat", 40)


def _argv(cfg: Config, pythonw: bool = False) -> List[str]:
    exe = sys.executable
    if pythonw and os.name == "nt":
        cand = os.path.join(os.path.dirname(exe), "pythonw.exe")
        exe = cand if os.path.exists(cand) else exe
    return [exe, "-m", "zerowhite", "chat", "serve", "--supervise"] + cfg.cli_args()


def _quote_cmd(a: str) -> str:
    return '"' + a.replace('"', '""') + '"' if (" " in a or not a) else a


def _paths(lay: Layout) -> Tuple[str, str, str]:
    n = _name(lay)
    home = os.path.expanduser("~")
    return (os.path.join(lay.run, "start-service.cmd"),
            os.path.join(home, ".config", "systemd", "user", n.lower() + ".service"),
            os.path.join(home, "Library", "LaunchAgents", "com.zerowhite." + n.lower() + ".plist"))


def render(cfg: Config, lay: Layout) -> Tuple[str, str]:
    """(target path, file content) for this platform."""
    cmd_path, unit_path, plist_path = _paths(lay)
    if os.name == "nt":
        body = ("@echo off\r\n"
                f'set "PYTHONPATH={PACKAGE_PARENT}"\r\n'
                "set PYTHONUTF8=1\r\n"
                f'start "" /B {" ".join(_quote_cmd(a) for a in _argv(cfg, pythonw=True))}\r\n')
        return cmd_path, body
    if sys.platform == "darwin":
        pl = {"Label": "com.zerowhite." + _name(lay).lower(), "ProgramArguments": _argv(cfg), "RunAtLoad": True, "KeepAlive": False,
              "EnvironmentVariables": {"PYTHONPATH": PACKAGE_PARENT, "PYTHONUTF8": "1"}, "WorkingDirectory": PACKAGE_PARENT,
              "StandardOutPath": lay.log_path("daemon.out"), "StandardErrorPath": lay.log_path("daemon.out")}
        return plist_path, plistlib.dumps(pl).decode("utf-8")
    exec_line = " ".join('"' + a + '"' if " " in a else a for a in _argv(cfg))
    unit = ("[Unit]\nDescription=Zero White assistant (" + lay.workspace + ")\nAfter=network.target\n\n[Service]\n"
            f"Environment=PYTHONPATH={PACKAGE_PARENT}\nEnvironment=PYTHONUTF8=1\nWorkingDirectory={PACKAGE_PARENT}\n"
            f"ExecStart={exec_line}\nRestart=on-failure\nNice=5\n\n[Install]\nWantedBy=default.target\n")
    return unit_path, unit


def _run(cmd: List[str]) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def show(cfg: Config, lay: Layout) -> str:
    path, body = render(cfg, lay)
    head = f"# {path}\n"
    if os.name == "nt":
        head += f'# registered with: schtasks /Create /TN "{_name(lay)}" /SC ONLOGON /TR "{path}" /RL LIMITED /F\n'
    elif sys.platform == "darwin":
        head += "# registered with: launchctl load -w <that file>\n"
    else:
        head += f"# registered with: systemctl --user daemon-reload && systemctl --user enable --now {os.path.basename(path)}\n"
    return head + body


def install(cfg: Config, lay: Layout) -> str:
    path, body = render(cfg, lay)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fileio.write_text(path, body)
    if os.name == "nt":
        rc, out = _run(["schtasks", "/Create", "/TN", _name(lay), "/SC", "ONLOGON", "/TR", f'"{path}"', "/RL", "LIMITED", "/F"])
    elif sys.platform == "darwin":
        rc, out = _run(["launchctl", "load", "-w", path])
    else:
        _run(["systemctl", "--user", "daemon-reload"])
        rc, out = _run(["systemctl", "--user", "enable", "--now", os.path.basename(path)])
    if rc != 0:
        return f"wrote {path} but registering it failed: {out}\nRegister it by hand with the commands shown by `chat autostart show`."
    return f"installed: {path}\nThe service will start at your next login (and `zerowhite chat start` starts it now)."


def remove(cfg: Config, lay: Layout) -> str:
    path, _ = render(cfg, lay)
    if os.name == "nt":
        rc, out = _run(["schtasks", "/Delete", "/TN", _name(lay), "/F"])
    elif sys.platform == "darwin":
        rc, out = _run(["launchctl", "unload", "-w", path])
    else:
        rc, out = _run(["systemctl", "--user", "disable", "--now", os.path.basename(path)])
    try:
        os.remove(path)
    except OSError:
        pass
    return "removed" if rc == 0 else f"removed the file; unregistering reported: {out}"
