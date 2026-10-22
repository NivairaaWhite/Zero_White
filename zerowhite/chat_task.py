"""chat_task.py - the chat model (v5) as one more task the organism (v6) schedules.

Until now the organism drove only the general model; the chat model had its own loop (the background service). This module is the
bridge that lets one scheduler weigh both. It adds no new learning code: an "attempt" is the existing `prepare()` + `Trainer`, so
every guarantee of the chat trainer (validation split, promote only if better, resumable checkpoints) still holds.

What the organism needs from a task, and how the chat model supplies it:

  skill     in [0, 1], 1.0 = perfect.  skill = 1 - validation_nll / ln(vocab): the share of a uniform guess's uncertainty that the model has
            removed. 0 = no better than guessing, and 1 is unreachable on real text, so a chat task is never "at its ceiling": it
            plateaus only when attempts stop paying (the `patience` rule).
  attempt   one training run with a wall-clock budget in MINUTES. Unlike the general model there is no cost model to learn: the trainer
            honours the budget, so the cost of an attempt is the budget plus a measured overhead (ingest, tokenise, benchmark, save).
  gain      measured on ONE validation set: the live model is scored before and after on the same documents. New data changes the
            validation set, so comparing against a number recorded earlier would mix "learned more" with "the exam changed".
  data      the chat workspace's data folder, plus (optionally) the prose the organism's inbox digested: the text pool is mirrored into
            `<chat>/data/_omnivore/` so a file dropped in the inbox can teach both models.

The first run of a chat model decides its SIZE (autoscale picks the largest model the time budget and data can support), so the first
attempt is a real-sized run, not a cheap probe: a one-minute probe would fix a tiny model for good.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from typing import Callable, Optional

from . import fileio
from .chat import hardware as hw
from .chat.config import Config
from .chat.paths import Layout
from .chat.util import FileLock

NAME = "chat"
FEED_DIR = "_omnivore"


def skill_from_nll(nll, vocab) -> Optional[float]:
    """Normalised skill of a language model from its validation loss (None when it cannot be computed)."""
    try:
        nll, vocab = float(nll), int(vocab)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(nll) or vocab < 2:
        return None
    return float(min(1.0, max(0.0, 1.0 - nll / math.log(vocab))))


def summarise(res: dict, vocab: int, had_model: bool) -> dict:
    """Turn the trainer's result into what the organism's `learn` step reads. Pure, so it can be tested without training anything."""
    promoted = bool(res.get("promoted"))
    interrupted = res.get("reason") == "stopped"
    before_nll = res.get("baseline_val")
    live_nll = res.get("val_nll") if promoted else before_nll          # what the LIVE model scores on the current validation set
    skill, before = skill_from_nll(live_nll, vocab), skill_from_nll(before_nll, vocab)
    measured = skill is not None
    improved = bool(promoted and (not had_model or before is None or (skill is not None and skill > before + 1e-12)))
    ppl = math.exp(min(50.0, float(live_nll))) if measured and live_nll is not None else None
    return dict(ok=True, skill=skill if measured else 0.0, skill_before=before, improved=improved, family="chat-transformer", test=None,
                measured=measured, interrupted=interrupted, promoted=promoted, promote_reason=res.get("promote_reason"),
                val_nll=live_nll, ppl=ppl, params=res.get("params"), tokens=res.get("tokens"), steps=res.get("steps"),
                reason=res.get("reason"), train_seconds=res.get("elapsed_s"), vocab=int(vocab) if vocab else None,
                description=(f"{res.get('params') or 0:,} parameters, validation perplexity {ppl:.1f}" if ppl else "no validation data to measure"))


class ChatTask:
    def __init__(self, path: str, overrides=(), backend=None):
        self.path = os.path.abspath(path)
        self.overrides = list(overrides or ())
        self.backend = backend                      # None: compute.backend in the chat config decides (cpu / gpu / auto)

    # ------------------------------------------------------------------------------------------ configuration / layout
    def config(self, **kv) -> Config:
        cfg = Config.load(self.path, overrides=self.overrides)
        return cfg.replace(**kv) if kv else cfg

    def layout(self, cfg: Optional[Config] = None) -> Layout:
        return Layout(cfg or self.config()).ensure()

    # ------------------------------------------------------------------------------------------ who else is training?
    def busy(self) -> Optional[int]:
        """The pid of another live process that is training this chat model (its service, or a `chat train`), else None.
        Training takes a lock, so two trainers can never corrupt each other; this lets the organism wait instead of failing."""
        lock = FileLock(self.layout().train_lock)
        pid = lock._holder()
        return pid if pid and pid != os.getpid() and hw.pid_alive(pid) else None

    def service_running(self) -> Optional[dict]:
        try:
            from .chat.service import daemon
            return daemon.running(self.layout())
        except Exception:                                    # noqa: BLE001 - a status probe must never break the loop
            return None

    # ------------------------------------------------------------------------------------------ data
    def feed_prose(self, library, pool: str) -> bool:
        """Mirror the omnivore text pool into the chat data folder. True when the mirror changed.
        Written atomically from outside the data folder, so the ingestor never sees a half-written file."""
        from .omnivore.library import _safe
        src = os.path.join(library.task_dir(_safe(f"corpus-{pool}")), "corpus.txt")
        if not os.path.exists(src):
            return False
        lay = self.layout()
        dst_dir = os.path.join(lay.data, FEED_DIR)
        dst = os.path.join(dst_dir, f"corpus-{_safe(pool)}.txt")
        with open(src, "rb") as f:
            data = f.read()
        if os.path.exists(dst) and os.path.getsize(dst) == len(data):
            with open(dst, "rb") as f:
                if hashlib.sha256(f.read()).digest() == hashlib.sha256(data).digest():
                    return False
        os.makedirs(dst_dir, exist_ok=True)
        tmp = os.path.join(lay.state, ".feed_tmp")
        with open(tmp, "wb") as f:
            f.write(data)
        fileio._replace_with_retry(tmp, dst)
        return True

    def sync(self) -> dict:
        """Read new / changed data files into the chat corpus (incremental: unchanged files cost a stat)."""
        from .chat.corpus.store import Ingestor
        cfg = self.config()
        lay = self.layout(cfg)
        ing = Ingestor(cfg, lay)
        rep = ing.ingest()
        return dict(changed=rep.changed_anything, has_data=bool(ing.shards()), summary=rep.summary(), unsettled=rep.files_unsettled,
                    errors=[f"{r}: {e}" for r, e in rep.errors[:3]])

    # ------------------------------------------------------------------------------------------ the live model
    def snapshot(self) -> dict:
        """What is known about the live model from files alone (nothing is loaded or trained)."""
        lay = self.layout()
        cur = lay.current()
        if not cur:
            return dict(has_model=False)
        vocab = None
        try:
            from .chat.prepare import load_tokenizer
            tok = load_tokenizer(self.config(), lay)
            vocab = tok.vocab_size if tok is not None else None
        except Exception:                                    # noqa: BLE001
            pass
        nll = cur.get("val_nll")
        return dict(has_model=True, val_nll=nll, vocab=vocab, params=cur.get("params"), runs=cur.get("runs"),
                    skill=skill_from_nll(nll, vocab) if vocab else None,
                    ppl=math.exp(min(50.0, float(nll))) if nll is not None else None)

    # ------------------------------------------------------------------------------------------ one attempt
    def attempt(self, minutes: float, should_stop: Optional[Callable[[], bool]] = None) -> dict:
        from .chat.prepare import prepare
        from .chat.train import Trainer, TrainingBusy
        t0 = time.time()
        cfg = self.config(**{"train.budget_minutes": float(minutes)})
        lay = self.layout(cfg)
        had_model = lay.current_weights() is not None
        P = prepare(cfg, lay, progress=None, do_bench=not had_model, backend=self.backend)
        try:
            res = Trainer(P, progress=None, should_stop=should_stop).run()
        except TrainingBusy as ex:
            return dict(ok=True, deferred=True, reason=str(ex), seconds=time.time() - t0)
        out = summarise(res, P.tok.vocab_size, had_model)
        total = time.time() - t0
        out.update(seconds=total, overhead_s=max(0.0, total - float(res.get("elapsed_s") or 0.0)), minutes=float(minutes),
                   compute=P.backend.label() if P.backend is not None else "cpu")
        return out
