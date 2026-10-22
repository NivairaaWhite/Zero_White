"""daemon.py - the background process.

One process that
  * serves the HTTP API + web chat (server.py)
  * watches the data folder: new / changed / removed files are ingested and tokenised automatically (settle time: half-copied files wait)
  * trains by itself when enough new data has arrived (daemon.train_min_new_tokens, daemon.train_cooldown_minutes), in a separate
    low-priority process, and hot-swaps the new model into the running server once it is validated and better
  * applies online learning from chat feedback (online.py)
and a supervisor (`supervise`) that restarts it with backoff if it ever crashes. `start()` / `stop()` / `status()` are what the CLI uses.
Stopping is cooperative: a STOP file in run/ and (where available) SIGTERM; a running training job checkpoints and can be resumed.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import List, Optional

from ... import fileio
from .. import autoscale as AS
from .. import hardware as hw
from ..config import Config
from ..corpus.store import Ingestor
from ..generate import Engine
from ..online import OnlineLearner
from ..paths import Layout
from ..prepare import load_corpus, load_tokenizer, resolve_light
from ..session import ChatService
from ..tokenstore import TokenStore
from ..util import FileLock, LockBusy, setup_logging
from .server import App, Server

PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------------------------------------------------
# process helpers
# ---------------------------------------------------------------------------------------------------------------------
def child_env(cfg: Config) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = PACKAGE_PARENT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def spawn_detached(cmd: List[str], env: dict, out_path: str) -> subprocess.Popen:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    kw: dict = {}
    if os.name == "nt":
        kw["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000          # DETACHED_PROCESS | NEW_PROCESS_GROUP | NO_WINDOW
    else:
        kw["start_new_session"] = True
    out = open(out_path, "ab")
    try:
        return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=out, close_fds=True, env=env, cwd=PACKAGE_PARENT, **kw)
    finally:
        out.close()


def _read_pid(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def running(lay: Layout) -> Optional[dict]:
    """{'pid', 'url'} of the live daemon of this workspace, or None."""
    pid = _read_pid(lay.pid_file)
    if pid and hw.pid_alive(pid):
        info = {}
        try:
            info = fileio.read_json(lay.server_file)
        except (OSError, ValueError):
            pass
        return dict(pid=pid, url=info.get("url"), supervisor=_read_pid(lay.supervisor_file))
    return None


def log_tails(lay: Layout, lines: int = 20) -> str:
    """The end of every log that can say why the service did not come up: the worker's stdout/stderr (import errors and tracebacks land
    there, before logging exists), its own log, and the supervisor's (which records each crash and restart of the worker)."""
    parts = []
    for name in ("daemon.out", "daemon.log", "supervisor.log"):
        path = lay.log_path(name)
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 16_000))
                tail = f.read().decode("utf-8", "replace").splitlines()[-lines:]
        except OSError:
            continue
        if tail:
            parts.append(f"--- {path} (last {len(tail)} lines) ---\n" + "\n".join(tail))
    return "\n".join(parts)


def start(cfg: Config, lay: Layout, wait_s: float = 30.0) -> dict:
    """Start the background service (detached). Returns {'started': bool, 'url': str, ...}."""
    lay.ensure()
    cur = running(lay)
    if cur:
        return dict(started=False, already=True, **cur)
    try:
        os.remove(lay.stop_file)
    except OSError:
        pass
    mode = "supervise" if cfg.get("daemon.supervise") else "worker"
    cmd = [sys.executable, "-m", "zerowhite", "chat", "serve", "--" + mode] + cfg.cli_args()
    p = spawn_detached(cmd, child_env(cfg), lay.log_path("daemon.out"))
    deadline = time.time() + wait_s
    while time.time() < deadline:
        time.sleep(0.25)
        cur = running(lay)
        if cur and cur.get("url"):
            return dict(started=True, **cur)
        if p.poll() is not None:
            return dict(started=False, error=f"the service exited immediately (code {p.returncode}); see {lay.log_path('daemon.out')}\n" + log_tails(lay))
    # With the supervisor (the default) `p` is only the supervisor: a worker that keeps crashing leaves it alive and restarting, so the
    # logs, not the process state, are what say why there is no URL yet.
    return dict(started=False, error=f"the service did not become ready within {wait_s:.0f}s; see {lay.log_path('daemon.log')}\n" + log_tails(lay))


def stop(cfg: Config, lay: Layout, wait_s: float = 60.0) -> dict:
    """Ask the service (and any training job) to stop; escalate to terminate / kill if it does not."""
    pids = [p for p in (_read_pid(lay.supervisor_file), _read_pid(lay.pid_file)) if p and hw.pid_alive(p)]
    if not pids:
        return dict(stopped=False, message="the service is not running")
    fileio.write_text(lay.stop_file, str(time.time()))
    deadline = time.time() + wait_s
    while time.time() < deadline and any(hw.pid_alive(p) for p in pids):
        time.sleep(0.3)
    left = [p for p in pids if hw.pid_alive(p)]
    for sig, grace in ((signal.SIGTERM, 10.0), (getattr(signal, "SIGKILL", signal.SIGTERM), 3.0)):
        for p in left:
            try:
                os.kill(p, sig)
            except OSError:
                pass
        end = time.time() + grace
        while time.time() < end and any(hw.pid_alive(p) for p in left):
            time.sleep(0.2)
        left = [p for p in left if hw.pid_alive(p)]
        if not left:
            break
    try:
        os.remove(lay.stop_file)
    except OSError:
        pass
    return dict(stopped=not left, message="stopped" if not left else f"could not stop process(es) {left}")


# ---------------------------------------------------------------------------------------------------------------------
class Daemon:
    def __init__(self, cfg: Config, lay: Layout, console: bool = False):
        self.cfg0, self.lay = cfg, lay
        self.cfg = cfg
        self.console = console
        self.log: logging.Logger = logging.getLogger("zerowhite")
        self.stop_evt = threading.Event()
        self.started = time.time()
        self.engine: Optional[Engine] = None
        self.chat: Optional[ChatService] = None
        self.learner: Optional[OnlineLearner] = None
        self.server: Optional[Server] = None
        self.train_proc: Optional[subprocess.Popen] = None
        self.train_started = 0.0
        self.train_reason = ""
        self.gate = threading.Lock()                       # learner update  <->  starting a training job
        self.state = dict(last_train_end=0.0, last_attempt_cutoff=0.0, last_result=None, trainings=0)
        try:
            self.state.update(fileio.read_json(lay.daemon_state))
        except (OSError, ValueError):
            pass

    # ---- bookkeeping ----------------------------------------------------------------------------------------------
    def _save_state(self):
        fileio.write_json(self.lay.daemon_state, self.state, indent=2)

    def info(self, msg: str, *a):
        self.log.info(msg, *a)

    def _stop_requested(self) -> bool:
        if self.stop_evt.is_set():
            return True
        try:
            return os.path.getmtime(self.lay.stop_file) >= self.started - 1.0
        except OSError:
            return False

    def _resolve(self):
        """(Re)derive the auto settings from the current tokenizer / model; keeps the engine and learner in step."""
        try:
            self.cfg = resolve_light(self.cfg0, self.lay)
        except Exception as e:                                       # noqa: BLE001
            self.log.warning("could not derive automatic settings yet: %s", e)
            self.cfg = self.cfg0
        if self.engine:
            self.engine.cfg = self.cfg
        if self.chat:
            self.chat.cfg = self.cfg

    def _ensure_learner(self):
        if self.learner or not self.engine or not self.engine.ready or not self.cfg.get("online.enabled"):
            return
        corpus = load_corpus(self.cfg, self.lay, self.engine.tok, self.engine.model.cfg.ctx)
        self.learner = OnlineLearner(self.cfg, self.lay, self.engine, corpus,
                                     on_event=lambda ev: self.log.info("online: %s", json.dumps(ev, default=str)[:400]))
        self.app.learner = self.learner
        self.info("online learning ready (%s)", "with replay corpus" if corpus is not None else "WITHOUT a corpus: updates are blocked until `chat prepare`")

    # ---- run ------------------------------------------------------------------------------------------------------
    def run(self) -> int:
        lay, cfg = self.lay.ensure(), self.cfg0
        lock = FileLock(lay.pid_file)
        try:
            lock.acquire()
        except LockBusy as e:
            print(f"the service is already running (process {e.holder}). Use `zerowhite chat status`.", file=sys.stderr)
            return 1
        self.log = setup_logging(lay.log_path("daemon.log"), float(cfg.get("daemon.log_max_mb")), int(cfg.get("daemon.log_backups")), console=self.console)
        try:
            return self._run()
        except Exception:                                            # noqa: BLE001
            self.log.exception("the service crashed")
            return 2
        finally:
            self._shutdown()
            lock.release()

    def _run(self) -> int:
        lay, cfg = self.lay, self.cfg0
        self.info("starting (pid %d, workspace %s)", os.getpid(), lay.workspace)
        self.engine = Engine(cfg, lay)
        try:
            if self.engine.load():
                self.info("model loaded: %s", self.engine.version)
            else:
                self.info("no trained model yet")
        except Exception as e:                                       # noqa: BLE001
            self.log.error("could not load the model: %s", e)
        self._resolve()
        self.chat = ChatService(self.cfg, lay, self.engine)
        hooks = dict(request_train=self.request_train, status=self.status, log=lambda lvl, m: self.log.error(m))
        self.app = App(self.cfg, lay, self.engine, self.chat, None, hooks)
        self._ensure_learner()
        self.server = Server(self.app)
        self.server.start()
        self.info("serving %s (token in %s)", self.server.url, lay.token_file)
        if self.console:
            print(f"Zero White is running: {self.server.url}   (Ctrl-C to stop)\nAccess token: {lay.token_file}", flush=True)
        self._signals()
        threads = [threading.Thread(target=self._watch_loop, name="watch", daemon=True),
                   threading.Thread(target=self._learn_loop, name="learn", daemon=True)]
        for t in threads:
            t.start()
        while not self._stop_requested():
            time.sleep(0.5)
        self.info("stop requested")
        self.stop_evt.set()
        return 0

    def _signals(self):
        if threading.current_thread() is not threading.main_thread():
            return
        for n in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, n, None)
            if sig is not None:
                try:
                    signal.signal(sig, lambda *_: self.stop_evt.set())
                except (ValueError, OSError):
                    pass

    def _shutdown(self):
        self.stop_evt.set()
        if self.server:
            try:
                self.server.stop()
            except Exception:                                        # noqa: BLE001
                pass
        p = self.train_proc
        if p and p.poll() is None:
            self.info("asking the training job to checkpoint and stop")
            try:
                p.terminate()
                p.wait(timeout=90)
            except Exception:                                        # noqa: BLE001
                p.kill()
        self._save_state()
        self.info("stopped")

    # ---- status ---------------------------------------------------------------------------------------------------
    def status(self) -> dict:
        return dict(pid=os.getpid(), started=self.started, autotrain=bool(self.cfg.get("daemon.autotrain")),
                    training=bool(self.train_proc and self.train_proc.poll() is None), training_reason=self.train_reason,
                    last_train=self.state.get("last_result"), trainings=self.state.get("trainings", 0), url=self.server.url if self.server else None)

    # ---- watching the data folder ---------------------------------------------------------------------------------
    def _watch_loop(self):
        interval = float(self.cfg.get("data.watch_interval_s"))
        while not self.stop_evt.wait(interval):
            try:
                self._tick()
            except Exception:                                        # noqa: BLE001
                self.log.exception("watcher error (continuing)")

    def _tick(self):
        if self.train_proc is not None:
            rc = self.train_proc.poll()
            if rc is None:
                return                                               # never touch shards or tokens while a job reads them
            self._training_finished(rc)
        cfg, lay = self.cfg, self.lay
        ing = Ingestor(cfg, lay)
        rep = ing.ingest()
        if rep.changed_anything:
            self.info("data folder: %s", rep.summary())
        if rep.errors:
            for rel, err in rep.errors[:3]:
                self.log.warning("cannot use %s: %s", rel, err)
        if self.engine and self.engine.reload_if_changed():
            self.info("hot-swapped to %s", self.engine.version)
            self._resolve()
        if not ing.shards():
            return
        why = self._should_train(ing)
        if why:
            self._start_training(why)

    def _token_counts(self, ing: Ingestor):
        """(total tokens, tokens of data newer than the live model's training data)."""
        cur = self.lay.current() or {}
        since = max(float(cur.get("data_cutoff", 0.0)), float(self.state.get("last_attempt_cutoff", 0.0)))
        tok = load_tokenizer(self.cfg, self.lay)
        shards = ing.shards()
        if tok is not None and self.engine and self.engine.ready:
            ts = TokenStore(self.cfg, self.lay, tok, self.engine.model.cfg.ctx)
            ts.sync(shards, workers=1)
            return ts.token_counts(shards, since)
        est = AS.estimate_corpus(ing, int(self.cfg.get("autoscale.sample_records")))["bytes"] // 4          # ~4 bytes per token
        return est, est

    def _should_train(self, ing: Ingestor) -> Optional[str]:
        cfg = self.cfg
        if not cfg.get("daemon.autotrain") or self.train_proc is not None:
            return None
        has_model = bool(self.engine and self.engine.ready)
        if has_model and time.time() - float(self.state.get("last_train_end", 0)) < float(cfg.get("daemon.train_cooldown_minutes")) * 60:
            return None
        total, new = self._token_counts(ing)
        thr = cfg.get("daemon.train_min_new_tokens", None)
        thr = max(20000, int(0.02 * total)) if thr in (None, "auto") else int(thr)
        if new >= thr:
            return (f"first model: {total:,} tokens of data" if not has_model else f"{new:,} new tokens of data (threshold {thr:,})")
        return None

    # ---- training jobs --------------------------------------------------------------------------------------------
    def request_train(self) -> dict:
        if self.train_proc is not None and self.train_proc.poll() is None:
            return dict(started=False, message="a training run is already in progress")
        if not Ingestor(self.cfg, self.lay).shards():
            ing = Ingestor(self.cfg, self.lay)
            ing.ingest()
            if not ing.shards():
                return dict(started=False, message=f"there is no usable data in {self.lay.data} yet")
        self._start_training("requested from the UI / API")
        return dict(started=True, message="Training started. The new model replaces the current one only if it scores better.")

    def _start_training(self, why: str):
        with self.gate:
            cmd = [sys.executable, "-m", "zerowhite", "chat", "train", "--background", "--wait", "600"] + self.cfg0.cli_args()
            self.info("starting training: %s", why)
            out = open(self.lay.log_path("train_run.log"), "ab")
            kw = {"creationflags": 0x08000000} if os.name == "nt" else {}
            try:
                self.train_proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=out, env=child_env(self.cfg0), cwd=PACKAGE_PARENT, **kw)
            finally:
                out.close()
            self.train_started, self.train_reason = time.time(), why

    def _training_finished(self, rc: int):
        self.train_proc = None
        res = None
        for ev in reversed(list(_tail_jsonl(self.lay.log_path("train.jsonl"), 80))):
            if ev.get("event") == "done":
                res = ev
                break
        self.state["last_train_end"] = time.time()
        self.state["trainings"] = int(self.state.get("trainings", 0)) + 1
        ing = Ingestor(self.cfg, self.lay)
        self.state["last_attempt_cutoff"] = max((s.created for s in ing.shards()), default=0.0)
        self.state["last_result"] = dict(rc=rc, finished=time.time(), **{k: res.get(k) for k in ("reason", "steps", "val_nll", "promoted", "promote_reason")}) if res else dict(rc=rc, finished=time.time())
        self._save_state()
        if rc == 0 and res and res.get("promoted"):
            self.info("training finished: model promoted (val loss %.4f)", res.get("val_nll") or float("nan"))
            if self.cfg.get("daemon.hot_swap") and self.engine:
                self.engine.load()
                self._resolve()
                if self.learner:
                    self.learner.corpus = load_corpus(self.cfg, self.lay, self.engine.tok, self.engine.model.cfg.ctx)
                    self.learner.engine = self.engine
                else:
                    self._ensure_learner()
        elif rc == 0:
            self.info("training finished but the result was not promoted: %s", (res or {}).get("promote_reason"))
        else:
            self.log.warning("the training job exited with code %s; see %s", rc, self.lay.log_path("train_run.log"))

    # ---- online learning ------------------------------------------------------------------------------------------
    def _learn_loop(self):
        interval = float(self.cfg.get("online.interval_s"))
        while not self.stop_evt.is_set():
            if self.learner is None:
                if self.stop_evt.wait(interval):
                    return
                try:
                    self._ensure_learner()
                except Exception:                                    # noqa: BLE001
                    self.log.exception("could not start online learning")
                continue
            self.learner.poke.wait(timeout=interval)
            self.learner.poke.clear()
            if self.stop_evt.is_set():
                return
            if self.train_proc is not None and self.train_proc.poll() is None:
                continue
            try:
                with self.gate:
                    self.learner.update()
            except Exception:                                        # noqa: BLE001
                self.log.exception("online update failed (continuing)")


def _tail_jsonl(path: str, n: int):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 200_000))
            lines = f.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------------------------------------------------------------
def supervise(cfg: Config, lay: Layout) -> int:
    """Keep the worker alive: restart it after a crash (backoff from daemon.restart_backoff_s, at most
    daemon.max_restarts_per_hour), exit when it stops cleanly or a STOP file appears."""
    lay.ensure()
    lock = FileLock(lay.supervisor_file)
    try:
        lock.acquire()
    except LockBusy as e:
        print(f"a supervisor is already running (process {e.holder})", file=sys.stderr)
        return 1
    log = setup_logging(lay.log_path("supervisor.log"), 2.0, 3, console=False, name="zerowhite.supervisor")
    backoff = [float(x) for x in cfg.get("daemon.restart_backoff_s")] or [1.0]
    limit = int(cfg.get("daemon.max_restarts_per_hour"))
    crashes: List[float] = []
    streak = 0
    proc: Optional[subprocess.Popen] = None
    started = time.time()
    stopping = threading.Event()

    def on_signal(*_):
        stopping.set()
    for n in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, n, None)
        if sig is not None:
            try:
                signal.signal(sig, on_signal)
            except (ValueError, OSError):
                pass

    def stop_asked() -> bool:
        if stopping.is_set():
            return True
        try:
            return os.path.getmtime(lay.stop_file) >= started - 1.0
        except OSError:
            return False

    try:
        while not stop_asked():
            cmd = [sys.executable, "-m", "zerowhite", "chat", "serve", "--worker"] + cfg.cli_args()
            t0 = time.time()
            out = open(lay.log_path("daemon.out"), "ab")
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=out, env=child_env(cfg), cwd=PACKAGE_PARENT,
                                        **({"creationflags": 0x08000000} if os.name == "nt" else {}))
            finally:
                out.close()
            log.info("worker started (pid %d)", proc.pid)
            while proc.poll() is None:
                if stop_asked():
                    break
                time.sleep(0.5)
            if proc.poll() is None:                                   # a stop was asked: let the worker leave by itself
                try:
                    proc.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    proc.kill()
            rc = proc.returncode
            if stop_asked() or rc == 0:
                log.info("worker exited (code %s); supervisor leaving", rc)
                return 0
            now = time.time()
            crashes = [c for c in crashes if now - c < 3600] + [now]
            streak = 0 if now - t0 > 2 * max(backoff) else streak + 1
            if len(crashes) > limit:
                log.error("worker crashed %d times within an hour; giving up (see %s)", len(crashes), lay.log_path("daemon.log"))
                return 3
            wait = backoff[min(streak, len(backoff)) - 1] if streak else backoff[0]
            log.warning("worker exited with code %s after %.0fs; restarting in %.0fs", rc, now - t0, wait)
            end = time.time() + wait
            while time.time() < end and not stop_asked():
                time.sleep(0.2)
        return 0
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
        lock.release()
