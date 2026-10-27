"""online.py - the assistant keeps learning from the conversations it has.

    chat reply  --log_reply-->  experience log (state/online/experience.jsonl)
    thumbs-up / thumbs-down / typed correction / "regenerate" --log_feedback--> same log

The learner (run by the daemon every online.interval_s, or on demand) turns pending feedback into two kinds of signal:

    supervised   thumbs-up reply, or the user's correction, becomes a training example (loss only on the answer tokens)
    preference   correction vs. the reply it fixed, or a good regenerated reply vs. the one the user regenerated away from,
                 becomes a (chosen, rejected) pair for the DPO loss of preference.py

and applies a small number of gradient steps to a COPY of the live model, mixing in replay documents from the original corpus so
old skills are not forgotten. The copy only replaces the live model when it passes a gate: validation loss on probe documents may not
rise by more than online.max_regression. Every accepted update is a versioned file (rollback = point `current.json` at the parent),
and accepted examples are written to the data folder so that the next full training run keeps them.

A thumbs-down on its own is logged but changes nothing: "this was bad" does not say what would have been good. It becomes a training
signal once a correction or a better regenerated answer exists.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .. import fileio
from . import hardware as hw
from . import preference as PF
from .config import AUTO, Config
from .corpus import records as R
from .corpus.filters import RecordFilter
from .generate import Engine
from .model import GPT
from .optim import AdamW
from .paths import Layout
from .tokenstore import Corpus
from .util import FileLock, LockBusy, append_jsonl


# ---------------------------------------------------------------------------------------------------------------------
# experience log
# ---------------------------------------------------------------------------------------------------------------------
class ExperienceLog:
    def __init__(self, lay: Layout):
        self.path = lay.experience_path
        self._tl = threading.RLock()
        self._lockfile = self.path + ".lock"

    def _locked(self):
        class _Both:
            def __enter__(s):
                self._tl.acquire()
                s.fl = FileLock(self._lockfile).acquire(wait_s=10.0)
                return s

            def __exit__(s, *exc):
                s.fl.release()
                self._tl.release()
        return _Both()

    def log_reply(self, session: str, system: Optional[str], context: Sequence[dict], reply: str, version: Optional[str],
                  regen_of: Optional[str] = None) -> str:
        rid = uuid.uuid4().hex[:12]
        with self._locked():
            append_jsonl(self.path, dict(ev="reply", id=rid, t=time.time(), session=session, system=system,
                                         context=[dict(role=t["role"], content=t["content"]) for t in context],
                                         reply=reply, version=version, regen_of=regen_of))
        return rid

    def log_feedback(self, reply_id: str, rating: int = 0, correction: Optional[str] = None) -> dict:
        ev = dict(ev="feedback", id=reply_id, t=time.time(), rating=int(rating), correction=(correction or "").strip() or None)
        with self._locked():
            append_jsonl(self.path, ev)
        return ev

    def read_events(self, from_offset: int = 0) -> Iterator[Tuple[int, dict]]:
        """(offset_after_event, event) for complete lines from `from_offset`; a torn last line is left for later."""
        if not os.path.exists(self.path):
            return
        with open(self.path, "rb") as f:
            f.seek(from_offset)
            off = from_offset
            for line in f:
                if not line.endswith(b"\n"):
                    return
                off += len(line)
                try:
                    yield off, json.loads(line.decode("utf-8"))
                except ValueError:
                    continue

    def size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def compact(self, keep_replies: int, upto: int) -> Optional[int]:
        """Rewrite the log keeping the newest `keep_replies` replies and their feedback (only events before `upto` are judged; later
        ones are copied untouched). Returns the new offset equivalent to `upto`, or None when nothing was done."""
        with self._locked():
            events = list(self.read_events(0))
            reply_ids = [e["id"] for _, e in events if e["ev"] == "reply"]
            if len(reply_ids) <= keep_replies:
                return None
            keep = set(reply_ids[-keep_replies:])
            out, new_upto = [], 0
            for off, e in events:
                if e["ev"] == "reply" and e["id"] not in keep:
                    pass
                elif e["ev"] == "feedback" and e["id"] not in keep:
                    pass
                else:
                    out.append(json.dumps(e, ensure_ascii=False))
                if off <= upto:
                    new_upto = sum(len(x.encode("utf-8")) + 1 for x in out)
            fileio.write_text(self.path, ("\n".join(out) + "\n") if out else "")
            return new_upto

    def stats(self) -> dict:
        n_reply = n_fb = n_up = n_down = n_corr = 0
        for _, e in self.read_events(0):
            if e["ev"] == "reply":
                n_reply += 1
            else:
                n_fb += 1
                n_up += e.get("rating", 0) > 0
                n_down += e.get("rating", 0) < 0
                n_corr += bool(e.get("correction"))
        return dict(replies=n_reply, feedback=n_fb, thumbs_up=n_up, thumbs_down=n_down, corrections=n_corr)


# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Example:
    system: Optional[str]
    context: List[dict]
    reply: str
    weight: float
    source: str
    reply_id: str


@dataclass
class PairEx:
    system: Optional[str]
    context: List[dict]
    chosen: str
    rejected: str
    weight: float
    reply_id: str


@dataclass
class Pending:
    sft: List[Example] = field(default_factory=list)
    pairs: List[PairEx] = field(default_factory=list)
    end_offset: int = 0
    events: int = 0
    corrections: int = 0
    implicit_ids: List[str] = field(default_factory=list)
    skipped: Dict[str, int] = field(default_factory=dict)
    implicit_offset: int = 0


def _h(*parts) -> str:
    return hashlib.blake2b("\x1f".join(str(p) for p in parts).encode("utf-8"), digest_size=10).hexdigest()


def rows_to_batch(docs: Sequence[Tuple[np.ndarray, np.ndarray, float]], pad: int) -> dict:
    """Unpacked batch: one document per row, loss weight = mask x document weight on the shifted targets."""
    L = max(len(d[0]) - 1 for d in docs)
    B = len(docs)
    ids = np.full((B, L), pad, np.int64)
    tgt = np.zeros((B, L), np.int64)
    w = np.zeros((B, L), np.float32)
    pos = np.zeros((B, L), np.int64)
    seg = np.zeros((B, L), np.int64)
    for b, (t, m, wt) in enumerate(docs):
        n = len(t) - 1
        ids[b, :n] = t[:-1]
        tgt[b, :n] = t[1:]
        w[b, :n] = np.asarray(m[1:], dtype=np.float32) * wt
        pos[b, :n] = np.arange(n)
        seg[b, :n] = 1
    return dict(ids=ids, tgt=tgt, w=w, pos=pos, seg=seg)


# ---------------------------------------------------------------------------------------------------------------------
class OnlineLearner:
    def __init__(self, cfg: Config, lay: Layout, engine: Engine, corpus: Optional[Corpus] = None,
                 on_event: Optional[Callable[[dict], None]] = None):
        self.cfg, self.lay, self.engine, self.corpus = cfg, lay, engine, corpus
        self.log = ExperienceLog(lay)
        # quality + safety + PII filters apply to live feedback; the persona-keyword filter is for dataset folders only
        self.filter = RecordFilter(cfg.replace(**{"filter.persona.enabled": False}))
        self.on_event = on_event
        self.state_path = os.path.join(lay.online_dir, "state.json")
        self.state = dict(offset=0, implicit_offset=0, implicit_done=[], updates=0, last_update=None, last_result=None)
        if os.path.exists(self.state_path):
            try:
                self.state.update(fileio.read_json(self.state_path))
            except (OSError, ValueError):
                pass
        self._lock = threading.Lock()
        self._read_off = 0
        self._replies: Dict[str, dict] = {}
        self._reply_off: Dict[str, int] = {}
        self._order: List[str] = []
        self._feedback: List[Tuple[int, dict]] = []
        self._rated: set = set()
        self._regen: set = set()
        self._by_session: Dict[str, List[str]] = {}
        self._probe: Optional[Tuple[str, List[dict]]] = None
        self._probe_loss: Dict[str, float] = {}
        self.poke = threading.Event()

    # ---- bookkeeping ------------------------------------------------------------------------------------------------
    def _save_state(self):
        self.state.pop("learned", None)
        self.state["implicit_done"] = self.state["implicit_done"][-20000:]
        fileio.write_json(self.state_path, self.state, indent=2)

    def _emit(self, ev: dict):
        ev = dict(ev, t=round(time.time(), 2))
        append_jsonl(self.lay.log_path("online.jsonl"), ev)
        if self.on_event:
            self.on_event(ev)

    def _sync(self):
        """Read events appended since last time into the in-memory index."""
        for off, e in self.log.read_events(self._read_off):
            self._read_off = off
            if e.get("ev") == "reply":
                rid = e["id"]
                self._replies[rid] = e
                self._reply_off[rid] = off
                self._order.append(rid)
                self._by_session.setdefault(e.get("session") or "", []).append(rid)
                if e.get("regen_of"):
                    self._regen.add(e["regen_of"])
            elif e.get("ev") == "feedback":
                self._feedback.append((off, e))
                self._rated.add(e["id"])

    # ---- turning feedback into examples -----------------------------------------------------------------------------
    def _clean(self, system, context, reply) -> Optional[Tuple[Optional[str], List[dict], str]]:
        rec = R.make_chat(_h(system, json.dumps(context), reply), system, list(context) + [dict(role="assistant", content=reply)], 1.0, {})
        if rec is None:
            return None
        rec, _ = self.filter.apply(rec)
        if rec is None:
            return None
        return rec["system"], rec["turns"][:-1], rec["turns"][-1]["content"]

    def collect(self) -> Pending:
        self._sync()
        P = Pending(end_offset=self._read_off)
        learned: set = set()                                    # duplicates within this batch only: repeating a correction is a new event
        implicit_done = set(self.state["implicit_done"])
        cutoff = self.state["offset"]

        def skip(why: str):
            P.skipped[why] = P.skipped.get(why, 0) + 1

        def add_sft(r, reply, w, src, fid):
            c = self._clean(r.get("system"), r["context"], reply)
            if not c:
                return skip("filtered")
            key = _h("sft", *[json.dumps(x, sort_keys=True) for x in (c[0], c[1], c[2])])
            if key in learned:
                return skip("duplicate_in_batch")
            learned.add(key)
            P.sft.append(Example(c[0], c[1], c[2], w, src, fid))

        def add_pair(r, chosen, rejected, w, fid):
            c = self._clean(r.get("system"), r["context"], chosen)
            d = self._clean(r.get("system"), r["context"], rejected)
            if not c or not d or c[2] == d[2]:
                return skip("filtered")
            key = _h("pair", json.dumps(c[1], sort_keys=True), c[2], d[2])
            if key in learned:
                return skip("duplicate_in_batch")
            learned.add(key)
            P.pairs.append(PairEx(c[0], c[1], c[2], d[2], w, fid))

        for off, f in self._feedback:
            if off <= cutoff:
                continue
            P.events += 1
            r = self._replies.get(f["id"])
            if r is None:
                skip("unknown_reply")
                continue
            corr, rating = f.get("correction"), int(f.get("rating") or 0)
            if corr:
                P.corrections += 1
                add_sft(r, corr, 1.0, "correction", f["id"])
                add_pair(r, corr, r["reply"], 1.0, f["id"])
            elif rating > 0:
                add_sft(r, r["reply"], 1.0, "thumbs_up", f["id"])
                prev = self._replies.get(r.get("regen_of") or "")
                if prev is not None:
                    add_pair(r, r["reply"], prev["reply"], 1.0, f["id"])
            else:
                skip("thumbs_down_alone" if rating < 0 else "empty_feedback")

        T = self.cfg.get("online.implicit_accept_s")
        if T is not None:
            now = time.time()
            wi = float(self.cfg.get("online.implicit_weight"))
            first_open = None
            for i, rid in enumerate(self._order):
                if self._reply_off[rid] <= self.state["implicit_offset"]:
                    continue
                r = self._replies[rid]
                sess = self._by_session.get(r.get("session") or "", [])
                followed = sess and sess[-1] != rid
                decided = rid in self._rated or rid in self._regen or rid in implicit_done
                if not decided and followed and now - r["t"] >= float(T):
                    add_sft(r, r["reply"], wi, "implicit", rid)
                    P.implicit_ids.append(rid)
                    decided = True
                if not decided and first_open is None:
                    first_open = self._reply_off[rid]
            P.implicit_offset = (first_open - 1) if first_open is not None else self._read_off
        return P

    def should_update(self, P: Pending) -> bool:
        if not (P.sft or P.pairs):
            return False
        if self.cfg.get("online.immediate_corrections") and P.corrections:
            return True
        return P.events >= int(self.cfg.get("online.min_events"))

    # ---- probe documents for the regression gate --------------------------------------------------------------------
    def _probe_batches(self, ctx: int, pad: int) -> List[dict]:
        key = f"{ctx}:{id(self.corpus)}"
        if self._probe and self._probe[0] == key:
            return self._probe[1]
        c = self.corpus
        pool = c.val_idx if len(c.val_idx) else c.train_idx
        order = pool[np.argsort(c.d_u[pool], kind="stable")][: int(self.cfg.get("online.probe_docs"))]
        docs = [(*c.doc(int(i)), 1.0) for i in order]
        batches = [rows_to_batch(docs[s:s + 16], pad) for s in range(0, len(docs), 16)]
        self._probe = (key, batches)
        self._probe_loss = {}
        return batches

    @staticmethod
    def _nll(model: GPT, batches: List[dict]) -> float:
        tot = n = 0
        for b in batches:
            l, k = model.eval_nll(b)
            tot += l
            n += k
        return tot / max(1, n)

    # ---- the update -------------------------------------------------------------------------------------------------
    def update(self, force: bool = False) -> dict:
        """Run one online update if there is enough feedback (or force=True and there is any). Always returns a result dict."""
        if not self.cfg.get("online.enabled"):
            return dict(updated=False, reason="online learning is disabled (online.enabled)")
        if not self.engine.ready:
            return dict(updated=False, reason="no trained model yet")
        if not self._lock.acquire(blocking=False):
            return dict(updated=False, reason="an update is already running")
        try:
            lk = FileLock(self.lay.train_lock)
            try:
                lk.acquire()
            except LockBusy:
                return dict(updated=False, reason="a training run is active; feedback is kept and will be learned afterwards")
            try:
                return self._update(force)
            finally:
                lk.release()
        finally:
            self._lock.release()

    def _update(self, force: bool) -> dict:
        t0 = time.time()
        P = self.collect()
        if not (P.sft or P.pairs):
            self._advance(P)
            return dict(updated=False, reason="nothing to learn yet", skipped=P.skipped)
        if not force and not self.should_update(P):
            return dict(updated=False, reason=f"{P.events} of {self.cfg.get('online.min_events')} feedback events needed", pending=P.events)
        eng, cfg = self.engine, self.cfg
        with eng._lock:
            live, tpl, parent = eng.model, eng.tpl, eng.version
        ctx, pad = live.cfg.ctx, tpl.pad_id
        if self.corpus is None or not len(self.corpus.train_idx):
            return dict(updated=False, reason="the training corpus (state/tokens) is needed as replay and as the regression gate; "
                                              "run `zerowhite chat prepare` first")
        threads = cfg.get("online.threads", None)
        if threads not in (None, AUTO):
            hw.limit_threads(int(threads))

        # encode
        sft_docs = []
        for e in P.sft:
            prompt, ans = tpl.encode_pair(e.system, e.context, e.reply, ctx + 1)
            ids = np.asarray(prompt + ans, dtype=np.int64)
            m = np.concatenate((np.zeros(len(prompt), np.int8), np.ones(len(ans), np.int8)))
            sft_docs.append((ids, m, e.weight))
        pairs: List[PF.Pair] = []
        for p in P.pairs:
            pr, ch = tpl.encode_pair(p.system, p.context, p.chosen, ctx + 1)
            _, rj = tpl.encode_pair(p.system, p.context, p.rejected, ctx + 1)
            pairs.append((np.asarray(pr, np.int64), np.asarray(ch, np.int64), np.asarray(rj, np.int64), p.weight))

        probe = self._probe_batches(ctx, pad)
        cached = self._probe_loss.get(parent or "")
        before = cached if cached is not None else self._nll(live, probe)
        self._probe_loss[parent or ""] = before
        fit_batch = rows_to_batch(sft_docs, pad) if sft_docs else None
        fit_before = self._nll(live, [fit_batch]) if fit_batch else None
        if pairs:
            ref_c, ref_r = PF.sequence_logprobs(live, pairs, ctx, pad)

        cand = live.clone()
        lr = float(cfg.get("online.lr"))
        opt = AdamW(cand.p, float(cfg.get("train.beta1")), float(cfg.get("train.beta2")), weight_decay=0.0)
        rng = np.random.default_rng(int(self.state["updates"]) + 1)
        steps = int(cfg.get("online.steps_per_update"))
        bs = max(1, int(cfg.get("online.batch_examples")))
        replay_n = int(round(float(cfg.get("online.replay_ratio")) * bs)) if sft_docs else 0
        pw = float(cfg.get("online.preference_weight"))
        beta = float(cfg.get("online.dpo_beta"))
        clip = float(cfg.get("train.grad_clip"))
        train_idx = self.corpus.train_idx
        for s in range(steps):
            grads = None
            if sft_docs:
                pick = rng.choice(len(sft_docs), size=min(bs, len(sft_docs)), replace=len(sft_docs) < bs)
                docs = [sft_docs[i] for i in pick]
                if replay_n:
                    for i in rng.choice(train_idx, size=replay_n, replace=len(train_idx) < replay_n):
                        t, m = self.corpus.doc(int(i))
                        docs.append((t, m, 1.0))
                _, g, info = cand.loss_and_grads(rows_to_batch(docs, pad))
                inv = 1.0 / max(1, info["n"])
                grads = {k: v * inv for k, v in g.items()}
            if pairs:
                pick = rng.choice(len(pairs), size=min(bs, len(pairs)), replace=False)
                pg, _ = PF.dpo_grads(cand, ref_c[pick], ref_r[pick], [pairs[i] for i in pick], beta, ctx, pad, pair_weight=pw)
                if grads is None:
                    grads = pg
                else:
                    for k in grads:
                        grads[k] += pg[k]
            res = opt.step(cand.p, grads, lr, clip)
            if not res["applied"]:
                break
        after = self._nll(cand, probe)
        fit_after = self._nll(cand, [fit_batch]) if fit_batch else None
        limit = before * (1.0 + float(cfg.get("online.max_regression")))
        ok = math.isfinite(after) and after <= limit
        out = dict(events=P.events, sft=len(P.sft), pairs=len(P.pairs), steps=steps, probe_before=before, probe_after=after,
                   fit_before=fit_before, fit_after=fit_after, skipped=P.skipped, seconds=round(time.time() - t0, 2), parent=parent)
        if not ok:
            out.update(updated=False, reason=f"rejected by the regression gate: probe loss {after:.4f} > {limit:.4f} "
                                             f"(online.max_regression); the model was left unchanged")
            self._reject(P, out)
            self._advance(P)
            self._emit(dict(event="rejected", **out))
            return out
        # accept: save version, switch live pointer, hot swap
        cur = self.lay.current() or {}
        meta = {k: v for k, v in cur.items() if k not in ("file", "updated")}
        n = int(self.state["updates"]) + 1
        file = self.lay.new_weights_path("online", n)
        meta.update(online_updates=n, parent=parent, online_events=P.events, probe_before=before, probe_after=after,
                    fit_before=fit_before, fit_after=fit_after, trained_online=time.time())
        cand.save(file, meta=meta)
        self.lay.set_current(file, **meta)
        if eng.model is live:
            eng.swap(cand, meta, os.path.basename(file))
        self._probe_loss[os.path.basename(file)] = after
        self.lay.prune_versions("online", int(cfg.get("online.keep_versions")))
        self._persist_examples(P)
        self.state["updates"] = n
        self.state["last_update"] = time.time()
        out.update(updated=True, version=os.path.basename(file), reason="accepted")
        self.state["last_result"] = {k: v for k, v in out.items() if k != "skipped"}
        self._advance(P)
        self._emit(dict(event="updated", **out))
        return out

    # ---- helpers ---------------------------------------------------------------------------------------------------
    def _advance(self, P: Pending):
        self.state["offset"] = max(self.state["offset"], P.end_offset)
        self.state["implicit_offset"] = max(self.state["implicit_offset"], P.implicit_offset)
        self.state["implicit_done"] = list(self.state["implicit_done"]) + P.implicit_ids
        keep = int(self.cfg.get("online.buffer_max"))
        if len(self._order) > keep * 1.2:
            new_off = self.log.compact(keep, self.state["offset"])
            if new_off is not None:
                self.state["offset"] = self.state["implicit_offset"] = new_off
                self._reset_index()
        self._save_state()

    def _reset_index(self):
        self._read_off = 0
        self._replies.clear(); self._reply_off.clear(); self._order.clear(); self._feedback.clear()
        self._rated.clear(); self._regen.clear(); self._by_session.clear()

    def _reject(self, P: Pending, info: dict):
        append_jsonl(os.path.join(self.lay.online_dir, "rejected.jsonl"),
                     dict(t=time.time(), info={k: v for k, v in info.items() if k != "skipped"},
                          sft=[dict(system=e.system, context=e.context, reply=e.reply, source=e.source) for e in P.sft],
                          pairs=[dict(system=p.system, context=p.context, chosen=p.chosen, rejected=p.rejected) for p in P.pairs]))

    def _persist_examples(self, P: Pending):
        """Learned examples go to the data folder (as ordinary dataset files) so future full training runs keep them."""
        if not self.cfg.get("online.save_to_data"):
            return
        folder = os.path.join(self.lay.data, str(self.cfg.get("online.data_folder")))
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if P.sft:
            lines = [json.dumps(dict(system=e.system, messages=list(e.context) + [dict(role="assistant", content=e.reply)]), ensure_ascii=False)
                     for e in P.sft]
            fileio.write_text(os.path.join(folder, f"feedback-{stamp}.jsonl"), "\n".join(lines) + "\n")
        if P.pairs:
            lines = [json.dumps(dict(system=p.system, chosen=list(p.context) + [dict(role="assistant", content=p.chosen)],
                                     rejected=list(p.context) + [dict(role="assistant", content=p.rejected)]), ensure_ascii=False) for p in P.pairs]
            fileio.write_text(os.path.join(folder, f"preferences-{stamp}.jsonl"), "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------------------------------------
def rollback(lay: Layout, to: Optional[str] = None) -> Optional[str]:
    """Point the live model back at `to` (a file in models/weights) or at the parent of the current version.
    Returns the file now live, or None when there is nothing to roll back to."""
    cur = lay.current()
    if not cur:
        return None
    target = to or cur.get("parent")
    if not target:
        return None
    path = target if os.path.isabs(target) else os.path.join(lay.weights_dir, os.path.basename(target))
    if not os.path.exists(path):
        return None
    from .model import read_meta
    meta = read_meta(path).get("meta", {})
    lay.set_current(path, **{k: v for k, v in meta.items() if k not in ("file", "updated")})
    return os.path.basename(path)
