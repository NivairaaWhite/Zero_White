"""cli_common.py - argument plumbing shared by every command that works on a chat workspace."""
from __future__ import annotations

import argparse
import os
import sys

from .config import Config, ConfigError
from .paths import Layout


def add_common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--workspace", "-w", default=None,
                   help="workspace folder (default: $ZW_WORKSPACE, else ./zw-chat)")
    p.add_argument("--config", "-c", default=None, help="config file (default: <workspace>/config.json if present)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any option for this run, e.g. --set train.lr=0.002 (repeatable)")
    return p


def load_cfg(args) -> Config:
    try:
        return Config.load(getattr(args, "workspace", None), getattr(args, "config", None),
                           getattr(args, "overrides", None) or [])
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        raise SystemExit(2)


def load_layout(args, create: bool = True):
    cfg = load_cfg(args)
    lay = Layout(cfg)
    if create:
        lay.ensure()
    return cfg, lay


def fail(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def human_time(s: float) -> str:
    s = int(max(0, s))
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def isatty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:  # noqa: BLE001
        return False


def workspace_hint(cfg: Config) -> str:
    return os.path.relpath(cfg.workspace) if os.path.commonpath([cfg.workspace, os.getcwd()]) == os.getcwd() else cfg.workspace
