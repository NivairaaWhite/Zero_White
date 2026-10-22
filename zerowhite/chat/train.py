"""train.py - the real trainer.

    new model        random init, cosine schedule over the planned steps (autoscale.py), validation + early stopping
    continual        the live model is the starting point; LR = previous peak x train.continual.lr_scale; documents added
                     since the last run are sampled train.continual.new_data_weight times more often (old data is still replayed)
    resume           an interrupted run (Ctrl-C, crash, power loss, the daemon stopping) continues from its checkpoint with the
                     identical data stream, optimiser state and schedule
    preference phase after supervised training, if the corpus has (chosen, rejected) pairs: DPO against a frozen copy of the model,
                     discarded when it hurts validation loss by more than train.preference_max_regression

A finished run is only *promoted* (becomes the live model) when its validation loss is not worse than the previous live model's,
measured on the same current validation set - training can never make the assistant worse than it was.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .. import fileio
from . import hardware as hw
from . import preference as PF
from .config import AUTO
from .model import GPT, GPTConfig
from . import backend as bk
from .optim import AdamW, cosine_lr
from .prepare import Prepared
from .tokenstore import Packer
from .util import FileLock, LockBusy, append_jsonl, fmt_num, fmt_time

STATE_VERSION = 1


class TrainingBusy(RuntimeError):
    pass


class TrainingDiverged(RuntimeError):
    pass


class GpuOutOfMemory(RuntimeError):
    pass


def evaluate(model: GPT, batches: List[dict]) -> Tuple[float, int]:
    """Mean negative log-likelihood per target token over fixed batches (nan when there are none)."""
    tot, n = 0.0, 0
    for b in batches:
        l, k = model.eval_nll(b)
        tot += l
        n += k
    return (tot / n if n else float("nan")), n


def pairwise_accuracy(model: GPT, pairs: List[PF.Pair], ctx: int, pad: int, limit: int = 200, chunk: int = 16) -> Optional[float]:
    """Share of pairs where the model finds the chosen answer more likely than the rejected one."""
    pairs = pairs[:limit]
    if not pairs:
        return None
    hit = 0
    for s in range(0, len(pairs), chunk):
        lc, lr = PF.sequence_logprobs(model, pairs[s:s + chunk], ctx, pad)
        hit += int((lc > lr).sum())
    return hit / len(pairs)


def console_progress(ev: dict):
    """Default progress printer used by the CLI."""
    k = ev.get("event")
    if k == "start":
        print(f"[train] {ev['mode']} model: {fmt_num(ev['params'])} params ({ev['d_model']}d x {ev['n_layer']}L, ctx {ev['ctx']}), "
              f"{ev['planned_steps']:,} steps x {ev['eff_tokens']:,} tokens, peak lr {ev['peak_lr']:.2e}"
              + (f", resuming at step {ev['step']}" if ev.get("step") else ""), flush=True)
    elif k == "log":
        print(f"[train] step {ev['step']:>6}/{ev['planned']}  loss {ev['loss']:.4f}  lr {ev['lr']:.2e}  gnorm {ev['grad_norm']:.2f}  "
              f"{ev['tok_s']:,.0f} tok/s  epoch {ev['epoch']:.2f}  eta {fmt_time(ev['eta_s'])}", flush=True)
    elif k == "eval":
        print(f"[train] step {ev['step']:>6}  validation loss {ev['val_nll']:.4f} (ppl {ev['val_ppl']:.1f})"
              + ("  * best" if ev.get("improved") else f"  (no improvement x{ev['bad_evals']})"), flush=True)
    elif k == "note":
        print(f"[train] {ev['message']}", flush=True)
    elif k == "done":
        r = ev
        print(f"[train] finished: {r['reason']} after {r['steps']:,} steps / {fmt_num(r['tokens'])} tokens in {fmt_time(r['elapsed_s'])}. "
              + (f"validation loss {r['val_nll']:.4f} (ppl {math.exp(min(50.0, r['val_nll'])):.1f}); " if r.get('val_nll') is not None else "")
              + ("promoted to the live model." if r["promoted"] else f"NOT promoted: {r['promote_reason']}"), flush=True)


class Trainer:
    def __init__(self, prep: Prepared, progress: Optional[Callable[[dict], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None, background: bool = False, wait_s: float = 0.0,
                 backend: Optional[bk.Backend] = None):
        self.P = prep
        self.be = backend or prep.backend or bk.CPU                 # where the model lives while it trains
        self.cfg, self.lay = prep.cfg, prep.lay
        self.progress = progress
        self.should_stop = should_stop
        self.background = background
        self.wait_s = float(wait_s)
        self._stop = threading.Event()
        self._t_start = time.time()

    # ---- control --------------------------------------------------------------------------------------------------
    def request_stop(self):
        self._stop.set()

    def _stopping(self) -> bool:
        if self._stop.is_set() or (self.should_stop and self.should_stop()):
            return True
        try:
            return os.path.getmtime(self.lay.stop_file) > self._t_start
        except OSError:
            return False

    def _emit(self, ev: dict, log: bool = True):
        ev = dict(ev, t=round(time.time(), 2))
        if log:
            append_jsonl(self.lay.log_path("train.jsonl"), ev)
        if self.progress:
            self.progress(ev)

    def _status(self, **kv):
        try:
            fileio.write_json(self.lay.train_status_path, dict(kv, updated=time.time(), pid=os.getpid()))
        except OSError:
            pass

    # ---- checkpoint -----------------------------------------------------------------------------------------------
    def _corpus_sig(self) -> str:
        c = self.P.corpus
        blob = "|".join(os.path.basename(d) for d in c.dirs) + f"|{len(c.train_idx)}|{len(c.val_idx)}"
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

    def _save_state(self, model: Optional[GPT], opt: Optional[AdamW], meta: dict):
        arrays: Dict[str, np.ndarray] = {}
        if model is not None:
            arrays.update({f"p.{k}": v for k, v in model.host_params().items()})
        if opt is not None:
            arrays.update(opt.state_arrays())
        arrays["__state__"] = np.frombuffer(json.dumps(dict(meta, version=STATE_VERSION), default=fileio._json_default).encode("utf-8"), np.uint8)
        fileio.save_npz(self.lay.train_state, arrays)

    def _load_state(self) -> Optional[Tuple[dict, dict]]:
        p = self.lay.train_state
        if not os.path.exists(p):
            return None
        try:
            z = fileio.load_npz(p)
            meta = json.loads(bytes(z["__state__"]).decode("utf-8"))
        except (OSError, ValueError, KeyError):
            return None
        return (meta, z) if meta.get("version") == STATE_VERSION else None

    # ---- entry point ----------------------------------------------------------------------------------------------
    def run(self) -> dict:
        lock = FileLock(self.lay.train_lock)
        try:
            lock.acquire(wait_s=self.wait_s)
        except LockBusy as e:
            raise TrainingBusy(f"another training run is active (process {e.holder}). Wait for it, or stop it with `zerowhite chat stop`.") from None
        old = self._install_signals()
        try:
            self._emit(dict(event="note", message=f"compute: {self.be.label()}" + (f" ({self.be.note})" if self.be.note else "")), log=True)
            return self._run()
        except Exception as ex:                                    # noqa: BLE001
            if self.be.is_gpu and self.be.is_oom(ex):
                free = self.be.free_memory()
                raise GpuOutOfMemory(
                    f"the GPU ran out of memory ({self.be.label()}; {(free or 0) / 2 ** 30:.1f} GB free now). Lower train.batch_tokens (the micro-batch), "
                    f"compute.vram_fraction, or the model size (model.target_params); the run's last checkpoint is kept and `train` resumes from it.") from ex
            raise
        finally:
            self._restore_signals(old)
            self.be.release()
            lock.release()
            try:
                os.remove(self.lay.train_status_path)
            except OSError:
                pass

    def _install_signals(self):
        saved = {}
        if threading.current_thread() is not threading.main_thread():
            return saved
        names = ["SIGINT", "SIGTERM"] + (["SIGBREAK"] if os.name == "nt" else [])
        for n in names:
            sig = getattr(signal, n, None)
            if sig is None:
                continue
            try:
                saved[sig] = signal.signal(sig, lambda *_: self._stop.set())
            except (ValueError, OSError):
                pass
        return saved

    def _restore_signals(self, saved):
        for sig, h in saved.items():
            try:
                signal.signal(sig, h)
            except (ValueError, OSError):
                pass

    # ---- the run --------------------------------------------------------------------------------------------------
    def _run(self) -> dict:
        P, cfg, lay = self.P, self.cfg, self.lay
        mc: GPTConfig = P.plan.model
        ctx, pad = mc.ctx, P.tpl.pad_id
        corpus = P.corpus
        if not len(corpus.train_idx):
            raise RuntimeError("the corpus has no training documents")
        hw.limit_threads(int(cfg.get("train.threads")))
        if self.background and cfg.get("daemon.low_priority_training"):
            hw.lower_priority()

        micro = int(cfg.get("train.batch_tokens"))
        rows = max(1, micro // ctx)
        micro = rows * ctx
        eff = int(cfg.get("train.effective_batch_tokens"))
        accum = max(1, eff // micro)
        eff = accum * micro
        planned = int(P.plan.info["planned_steps"])
        peak = float(cfg.get("train.lr"))
        warmup = int(cfg.get("train.warmup_steps"))
        min_ratio = float(cfg.get("train.min_lr_ratio"))
        clip = float(cfg.get("train.grad_clip"))
        dropout = float(cfg.get("train.dropout"))
        eval_every = max(1, int(cfg.get("train.eval_every")))
        patience = int(cfg.get("train.patience"))
        log_every = max(1, int(cfg.get("train.log_every")))
        save_every_s = float(cfg.get("train.save_every_minutes")) * 60.0
        budget_min = cfg.get("train.budget_minutes")
        budget_s = float(budget_min) * 60.0 if budget_min else None
        epochs_goal = cfg.get("train.epochs")
        seed = int(cfg.get("train.seed"))
        max_skipped = int(cfg.get("train.max_skipped_steps"))

        # ---- decide where to start from
        live_path = lay.current_weights() if cfg.get("train.resume") else None
        live_meta: dict = lay.current() or {}
        runs_before = int(live_meta.get("runs", 0)) if live_path else 0
        st = self._load_state() if cfg.get("train.resume") else None
        resumed = None
        if st and st[0].get("status") == "running" and st[0].get("model") == mc.to_dict() and st[0].get("tokenizer") == P.tok.fingerprint():
            resumed = st
        baseline_val: Optional[float] = None
        step = 0
        elapsed_before = 0.0
        tokens_seen = 0
        best_val, best_step, bad_evals, best_file = math.inf, 0, 0, None
        packer_state = None

        if resumed:
            meta, z = resumed
            model = GPT(mc, params={k[2:]: v for k, v in z.items() if k.startswith("p.")}, backend=self.be)
            opt = AdamW(model.p, float(cfg.get("train.beta1")), float(cfg.get("train.beta2")), weight_decay=float(cfg.get("train.weight_decay")), backend=self.be)
            if not opt.load_state(z):
                self._emit(dict(event="note", message="optimiser state in the checkpoint was unreadable; restarting the optimiser"))
            step, planned, peak, warmup = int(meta["step"]), int(meta["planned"]), float(meta["peak_lr"]), int(meta["warmup"])
            elapsed_before, tokens_seen = float(meta["elapsed"]), int(meta["tokens"])
            best_val, best_step, bad_evals, best_file = float(meta["best_val"]) if meta["best_val"] is not None else math.inf, int(meta["best_step"]), int(meta["bad_evals"]), meta.get("best_file")
            baseline_val = meta.get("baseline_val")
            mode, runs_before = meta["mode"], int(meta.get("runs_before", runs_before))
            if meta.get("corpus_sig") == self._corpus_sig():
                packer_state = meta.get("packer")
            else:
                self._emit(dict(event="note", message="the data changed since the checkpoint: continuing with the saved weights but a fresh data order"))
        elif live_path and P.model_cfg is not None:
            model, lm = GPT.load(live_path, backend=self.be)
            peak = float(lm.get("peak_lr", peak)) * float(cfg.get("train.continual.lr_scale"))
            warmup = max(1, min(warmup, planned // 10))
            opt = AdamW(model.p, float(cfg.get("train.beta1")), float(cfg.get("train.beta2")), weight_decay=float(cfg.get("train.weight_decay")), backend=self.be)
            mode = "continual"
        else:
            model = GPT(mc, seed=seed, init_std=float(cfg.get("model.init_std")), backend=self.be)
            opt = AdamW(model.p, float(cfg.get("train.beta1")), float(cfg.get("train.beta2")), weight_decay=float(cfg.get("train.weight_decay")), backend=self.be)
            mode = "new"

        packer = Packer(corpus, ctx, pad, seed=seed + runs_before, split="train")
        if packer_state:
            packer.set_state(packer_state)
        val_batches = Packer(corpus, ctx, pad, split="val", once=True).fixed_batches(rows, int(cfg.get("train.eval_tokens"))) if len(corpus.val_idx) else []
        if mode == "continual" and val_batches:
            baseline_val, _ = evaluate(model, val_batches)
            self._emit(dict(event="note", message=f"the live model scores {baseline_val:.4f} on the current validation set; training must beat that to replace it"))
        if P.plan.info.get("notice"):
            self._emit(dict(event="note", message=P.plan.info["notice"]))
        if P.plan.info.get("warning"):
            self._emit(dict(event="note", message=P.plan.info["warning"]))
        self._emit(dict(event="start", mode=mode, params=model.num_params(), d_model=mc.d_model, n_layer=mc.n_layer, ctx=ctx,
                        planned_steps=planned, eff_tokens=eff, peak_lr=peak, step=step, train_docs=len(corpus.train_idx), val_docs=len(corpus.val_idx)))

        t_session = time.time()
        last_save = t_session
        last_log_t, last_log_tokens = t_session, tokens_seen
        skipped = 0
        reason = "done"
        loss_ema: Optional[float] = None

        def elapsed() -> float:
            return elapsed_before + (time.time() - t_session)

        def meta_now(status: str) -> dict:
            return dict(status=status, step=step, planned=planned, peak_lr=peak, warmup=warmup, elapsed=elapsed(), tokens=tokens_seen,
                        best_val=None if best_val == math.inf else best_val, best_step=best_step, bad_evals=bad_evals, best_file=best_file,
                        baseline_val=baseline_val, mode=mode, runs_before=runs_before, model=mc.to_dict(), tokenizer=P.tok.fingerprint(),
                        corpus_sig=self._corpus_sig(), packer=packer.state())

        def do_eval(final: bool = False):
            nonlocal best_val, best_step, bad_evals, best_file
            v, _ = evaluate(model, val_batches)
            improved = v < best_val - 1e-9
            if improved:
                best_val, best_step, bad_evals = v, step, 0
                old = best_file
                best_file = lay.new_weights_path("best", step)
                model.save(best_file, meta=dict(step=step, val_nll=v))
                if old and old != best_file and os.path.exists(old):
                    os.remove(old)
            else:
                bad_evals += 1
            self._emit(dict(event="eval", step=step, val_nll=v, val_ppl=math.exp(min(50.0, v)), improved=improved, bad_evals=bad_evals))
            return improved

        try:
            while step < planned:
                if self._stopping():
                    reason = "stopped"
                    break
                if budget_s and elapsed() >= budget_s:
                    reason = "time budget"
                    break
                if epochs_goal and packer.epochs_done >= float(epochs_goal):
                    reason = "epochs"
                    break
                frac = max(step / planned, (elapsed() / budget_s) if budget_s else 0.0)
                lr = cosine_lr(peak, min_ratio, warmup, step, frac)
                rng = np.random.default_rng((seed, runs_before, step))
                acc: Optional[Dict[str, np.ndarray]] = None
                n_tok, nll_sum = 0, 0.0
                for _ in range(accum):
                    batch = packer.next_batch(rows)
                    _, g, info = model.loss_and_grads(batch, dropout, rng)
                    n_tok += info["n"]
                    nll_sum += info["nll_sum"]
                    if acc is None:
                        acc = g
                    else:
                        for k in acc:
                            acc[k] += g[k]
                if n_tok == 0:
                    step += 1
                    continue
                inv = 1.0 / n_tok
                for k in acc:
                    acc[k] *= inv
                res = opt.step(model.p, acc, lr, clip)
                del acc
                if not res["applied"]:
                    skipped += 1
                    self._emit(dict(event="note", message=f"step {step}: non-finite gradient, step skipped ({skipped}/{max_skipped})"))
                    if skipped >= max_skipped:
                        raise TrainingDiverged(f"{skipped} consecutive non-finite gradients at step {step}; the run diverged. Lower train.lr "
                                               f"(currently {peak:.2e}) and run again - the best checkpoint so far is kept.")
                else:
                    skipped = 0
                step += 1
                tokens_seen += n_tok
                loss = nll_sum / n_tok
                loss_ema = loss if loss_ema is None else 0.9 * loss_ema + 0.1 * loss

                now = time.time()
                if step % log_every == 0 or step == planned:
                    tok_s = (tokens_seen - last_log_tokens) / max(1e-9, now - last_log_t)
                    left = max(0, planned - step)
                    eta = min(left * eff / max(tok_s, 1e-9), (budget_s - elapsed()) if budget_s else 1e18)
                    last_log_t, last_log_tokens = now, tokens_seen
                    self._emit(dict(event="log", step=step, planned=planned, loss=loss_ema, lr=lr, grad_norm=res["grad_norm"], tok_s=tok_s,
                                    epoch=packer.epochs_done, eta_s=max(0.0, eta), tokens=tokens_seen))
                    self._status(state="training", step=step, planned=planned, loss=loss_ema, tok_s=tok_s, eta_s=max(0.0, eta), mode=mode)
                if val_batches and step % eval_every == 0 and step < planned:
                    do_eval()
                    if patience and bad_evals >= patience:
                        reason = "early stop"
                        break
                if save_every_s and now - last_save >= save_every_s:
                    self._save_state(model, opt, meta_now("running"))
                    last_save = now
                    self._emit(dict(event="checkpoint", step=step), log=False)
            else:
                reason = "done"
            if reason in ("done", "epochs", "time budget") and val_batches:
                do_eval(final=True)
        except BaseException:
            self._save_state(model, opt, meta_now("running"))                       # keep progress on crash / divergence
            raise
        if reason == "stopped":
            self._save_state(model, opt, meta_now("running"))
            self._emit(dict(event="note", message=f"stopped at step {step}; run the same command again to resume exactly here"))
            bv = None if best_val == math.inf else best_val
            out = dict(reason=reason, steps=step, tokens=tokens_seen, elapsed_s=elapsed(), promoted=False, promote_reason="the run is unfinished (resume it)",
                       val_nll=bv, val_ppl=math.exp(min(50.0, bv)) if bv is not None else None, mode=mode, file=None)
            self._emit(dict(event="done", **out))
            return out

        # ---- take the best supervised weights
        if val_batches and best_file and os.path.exists(best_file):
            w = fileio.load_npz(best_file)
            model.load_params({k[2:]: v for k, v in w.items() if k.startswith("p.")})
        final_val = best_val if (val_batches and best_val != math.inf) else None

        # ---- preference phase
        pref_info = self._preference_phase(model, ctx, pad, planned, peak, clip, seed, budget_s, elapsed, final_val, val_batches)
        if pref_info.get("val_nll") is not None:
            final_val = pref_info["val_nll"]

        # ---- promotion
        promoted, why = True, "first model" if mode == "new" else "no validation data to compare against"
        if final_val is not None and baseline_val is not None:
            promoted = final_val <= baseline_val
            why = (f"validation loss {final_val:.4f} vs the live model's {baseline_val:.4f}" if not promoted else "better than the live model")
        elif final_val is not None:
            why = "first model"
        file = None
        if promoted:
            file = lay.new_weights_path("train", step)
            cutoff = max((s.created for s in P.ing.shards()), default=time.time())
            meta = dict(step=step, tokens=tokens_seen, val_nll=final_val, params=model.num_params(), mode=mode, runs=runs_before + 1, peak_lr=peak,
                        data_cutoff=cutoff, trained=time.time(), tokenizer=P.tok.fingerprint(), reason=reason, preference=pref_info)
            model.save(file, meta=meta)
            lay.set_current(file, **{k: v for k, v in meta.items() if k != "preference"})
            lay.prune_versions("train", int(cfg.get("train.keep_checkpoints")))
        for p, kind, _ in lay.versions("best"):
            try:
                os.remove(p)
            except OSError:
                pass
        self._save_state(None, None, dict(status="finished", step=step, finished=time.time()))
        out = dict(reason=reason, steps=step, tokens=tokens_seen, elapsed_s=elapsed(), promoted=promoted, promote_reason=why,
                   val_nll=final_val, val_ppl=math.exp(min(50.0, final_val)) if final_val is not None else None, baseline_val=baseline_val,
                   mode=mode, file=file, preference=pref_info, params=model.num_params())
        self._emit(dict(event="done", **out))
        return out

    # ---- preference phase -------------------------------------------------------------------------------------------
    def _preference_phase(self, model: GPT, ctx: int, pad: int, planned: int, peak: float, clip: float, seed: int,
                          budget_s: Optional[float], elapsed: Callable[[], float], sft_val: Optional[float], val_batches: List[dict]) -> dict:
        cfg, corpus = self.cfg, self.P.corpus
        pairs = corpus.pref_pairs(False)
        if not pairs:
            return dict(ran=False)
        steps = cfg.get("train.preference_steps", AUTO)
        steps = max(1, int(round(0.1 * planned))) if steps == AUTO else int(steps)
        if steps <= 0:
            return dict(ran=False, reason="train.preference_steps = 0")
        batch_n = max(1, min(int(cfg.get("train.preference_batch")), len(pairs)))
        beta = float(cfg.get("train.preference_beta"))
        lr0 = peak * float(cfg.get("train.preference_lr_ratio"))
        val_pairs = corpus.pref_pairs(True)
        acc_before = pairwise_accuracy(model, val_pairs, ctx, pad)
        ref = model.clone()
        backup = {k: v.copy() for k, v in model.p.items()}
        opt = AdamW(model.p, float(cfg.get("train.beta1")), float(cfg.get("train.beta2")), weight_decay=0.0, backend=self.be)
        rng = np.random.default_rng((seed, 7))
        last = None
        done = 0
        for s in range(steps):
            if self._stopping() or (budget_s and elapsed() > 1.2 * budget_s):
                break
            idx = rng.choice(len(pairs), size=batch_n, replace=False)
            bp = [pairs[i] for i in idx]
            rc, rr = PF.sequence_logprobs(ref, bp, ctx, pad)
            g, stt = PF.dpo_grads(model, rc, rr, bp, beta, ctx, pad)
            opt.step(model.p, g, cosine_lr(lr0, 0.1, max(1, steps // 10), s, s / steps), clip)
            last = stt
            done += 1
            if (s + 1) % max(1, steps // 5) == 0 or s + 1 == steps:
                self._emit(dict(event="note", message=f"preference step {s + 1}/{steps}: loss {stt['loss']:.4f}, pairs ranked correctly {stt['accuracy']:.0%}"), log=True)
        info = dict(ran=True, steps=done, pairs=len(pairs), last=last, accuracy_before=acc_before)
        if val_batches:
            v, _ = evaluate(model, val_batches)
            info["val_nll"] = v
            limit = sft_val * (1.0 + float(cfg.get("train.preference_max_regression"))) if sft_val is not None else math.inf
            if v > limit:
                model.load_params(backup)
                info.update(ran=True, accepted=False, val_nll=sft_val,
                            reason=f"validation loss {v:.4f} > {limit:.4f} (train.preference_max_regression); kept the supervised model")
                self._emit(dict(event="note", message="preference phase discarded: " + info["reason"]))
                return info
        info["accepted"] = True
        info["accuracy_after"] = pairwise_accuracy(model, val_pairs, ctx, pad)
        return info
