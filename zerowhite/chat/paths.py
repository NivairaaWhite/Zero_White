"""paths.py - the on-disk layout of a chat workspace. Every path in the chat stack comes from here (which in turn
comes from the configuration), so moving the data folder, or the whole workspace, is a one-line config change.

    <workspace>/
      config.json               your settings (optional)
      data/                     DROP DATASETS HERE (any sub-folders; see docs/CHAT.md)
      models/
        tokenizer.json
        current.json            pointer to the live model {file, step, val_loss, ...}
        weights/<id>.npz        every saved model version
        train_state.npz         optimizer + data cursor (resume)
      state/
        manifest.json           which data files were ingested, with hashes
        corpus/<shard>.jsonl.gz normalised records
        tokens/<shard>.<tok>/   token shards (ids.npy, mask.npy, ...)
        online/experience.jsonl live feedback log
        sessions/<id>.jsonl     chat transcripts
        resolved_config.json    every derived value and why
      logs/                     train.jsonl, daemon.log, ...
      run/                      daemon.pid, server.json, STOP
      exports/                  converted models / datasets
"""
from __future__ import annotations

import os
import re
import time

from .. import fileio
from .config import Config


def safe_name(s: str, limit: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._") [:limit] or "x"


class Layout:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.workspace = cfg.workspace
        self.data = cfg.path("data")
        self.models = cfg.path("models")
        self.state = cfg.path("state")
        self.logs = cfg.path("logs")
        self.run = cfg.path("run")
        self.exports = cfg.path("exports")

    # ---- creation --------------------------------------------------------------------------------------------
    def ensure(self) -> "Layout":
        for d in (self.workspace, self.data, self.models, self.weights_dir, self.state, self.corpus_dir, self.tokens_dir,
                  self.online_dir, self.sessions_dir, self.logs, self.run, self.exports, self.voice_models, self.voice_state):
            os.makedirs(d, exist_ok=True)
        return self

    # ---- models ----------------------------------------------------------------------------------------------
    @property
    def tokenizer_path(self) -> str:
        p = self.cfg.get("tokenizer.path")
        return self.cfg.resolve_path(p) if p else os.path.join(self.models, "tokenizer.json")

    @property
    def weights_dir(self) -> str:
        return os.path.join(self.models, "weights")

    @property
    def current_pointer(self) -> str:
        return os.path.join(self.models, "current.json")

    @property
    def train_state(self) -> str:
        return os.path.join(self.models, "train_state.npz")

    def new_weights_path(self, kind: str, step: int) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return os.path.join(self.weights_dir, f"{kind}-{stamp}-s{int(step):07d}.npz")

    def current(self):
        """The live-model pointer dict, or None when nothing has been trained yet."""
        if not os.path.exists(self.current_pointer):
            return None
        try:
            return fileio.read_json(self.current_pointer)
        except (OSError, ValueError):
            return None

    def current_weights(self):
        cur = self.current()
        if not cur:
            return None
        p = os.path.join(self.weights_dir, cur["file"])
        return p if os.path.exists(p) else None

    def set_current(self, file: str, **meta):
        rec = dict(file=os.path.basename(file), updated=time.time(), **meta)
        fileio.write_json(self.current_pointer, rec, indent=2)
        return rec

    # ---- state -----------------------------------------------------------------------------------------------
    def versions(self, kind: str = None):
        """Saved model files, newest first: [(path, kind, mtime)]. kind = 'train' | 'online' | 'best' | None (all)."""
        out = []
        if os.path.isdir(self.weights_dir):
            for fn in os.listdir(self.weights_dir):
                if fn.endswith(".npz") and not fn.startswith("."):
                    k = fn.split("-", 1)[0]
                    if kind is None or k == kind:
                        p = os.path.join(self.weights_dir, fn)
                        out.append((p, k, os.path.getmtime(p)))
        return sorted(out, key=lambda x: -x[2])

    def prune_versions(self, kind: str, keep: int) -> int:
        """Delete the oldest files of `kind`, keeping the newest `keep` and never the live model. Returns how many were removed."""
        live = self.current_weights()
        removed = 0
        for i, (p, _, _) in enumerate(self.versions(kind)):
            if i >= max(0, keep) and p != live:
                try:
                    os.remove(p)
                    removed += 1
                except OSError:
                    pass
        return removed

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.state, "manifest.json")

    @property
    def corpus_dir(self) -> str:
        return os.path.join(self.state, "corpus")

    @property
    def tokens_dir(self) -> str:
        return os.path.join(self.state, "tokens")

    @property
    def dedupe_db(self) -> str:
        return os.path.join(self.state, "dedupe.sqlite")

    @property
    def online_dir(self) -> str:
        return os.path.join(self.state, "online")

    @property
    def experience_path(self) -> str:
        return os.path.join(self.online_dir, "experience.jsonl")

    @property
    def sessions_dir(self) -> str:
        return os.path.join(self.state, "sessions")

    @property
    def resolved_config_path(self) -> str:
        return os.path.join(self.state, "resolved_config.json")

    @property
    def voice_models(self) -> str:
        return os.path.join(self.models, "voice")

    @property
    def voice_state(self) -> str:
        return os.path.join(self.state, "voice")

    @property
    def voice_cache(self) -> str:
        return os.path.join(self.voice_state, "cache")

    @property
    def audio_data(self) -> str:
        return os.path.join(self.data, self.cfg.get("voice.audio_folder") or "audio")

    @property
    def token_file(self) -> str:
        return os.path.join(self.state, "token")

    # ---- run -------------------------------------------------------------------------------------------------
    @property
    def pid_file(self) -> str:
        return os.path.join(self.run, "daemon.pid")

    @property
    def supervisor_file(self) -> str:
        return os.path.join(self.run, "supervisor.pid")

    @property
    def daemon_state(self) -> str:
        return os.path.join(self.run, "daemon_state.json")

    @property
    def server_file(self) -> str:
        return os.path.join(self.run, "server.json")

    @property
    def stop_file(self) -> str:
        return os.path.join(self.run, "STOP")

    @property
    def train_lock(self) -> str:
        return os.path.join(self.run, "train.lock")

    @property
    def train_status_path(self) -> str:
        return os.path.join(self.run, "train_status.json")

    def log_path(self, name: str) -> str:
        return os.path.join(self.logs, name)
