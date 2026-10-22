"""sleep.py - a consolidation pass for the chat model: learn what the live feedback taught, slowly, with heavy replay, and refuse if ANY part of the old corpus got worse.

The online learner already learns from /good /bad /fix feedback: a few gradient steps on a COPY of the model, 3 replay documents per new example, and a regression gate on
a set of probe documents. Two things make that a poor fit for a long idle period, which is when consolidation should happen:

  * the gate averages over ONE pile of probe documents, so a model can forget one kind of document badly while the mean barely moves;
  * the update is sized for a chat in progress (6 steps at 0.1 x the training learning rate), not for catching up on a day of feedback.

`ConsolidationLearner` is the online learner with two changes: the probe documents are STRATIFIED (the training corpus is cut into four age bands in the order it was
written, and each band is probed separately), and the loss the gate watches is the WORST band, not the mean. `consolidate()` runs it with a longer, gentler, more
replay-heavy schedule (4x the steps, half the learning rate, twice the replay, half the tolerated regression) and reports the per-band loss before and after. The
organism runs it as its `sleep` arm (opt-in, `--sleep`) when every task has plateaued.

HONEST SCOPE: the bands are age bands of the corpus, not topics (the corpus does not label topics); 'worst band' is a stricter gate, not a proof that nothing was forgotten; and
what is learned is the same feedback the online learner would have learned, just in one longer, safer pass. There is no separate 'dream' generation or distillation step.
"""
from __future__ import annotations

import math
import time
from typing import List

import numpy as np

from .generate import Engine
from .model import GPT
from .online import OnlineLearner, rows_to_batch

BANDS = 4


class ConsolidationLearner(OnlineLearner):
    def _probe_batches(self, ctx: int, pad: int) -> List[dict]:
        key = f"{ctx}:{id(self.corpus)}:bands"
        if self._probe and self._probe[0] == key:
            return self._probe[1]
        c = self.corpus
        pool = np.sort(np.concatenate([c.train_idx, c.val_idx])) if len(c.val_idx) else np.sort(c.train_idx)
        per = max(1, int(self.cfg.get("online.probe_docs")) // BANDS)
        batches = []
        for band in np.array_split(pool, BANDS):
            if not len(band):
                continue
            pick = band[np.linspace(0, len(band) - 1, num=min(per, len(band))).round().astype(int)]
            docs = [(*c.doc(int(i)), 1.0) for i in pick]
            batches.append(rows_to_batch(docs, pad))
        self._probe = (key, batches)
        self._probe_loss = {}
        return batches

    def _nll(self, model: GPT, batches: List[dict]) -> float:
        """The gate watches the WORST band."""
        per = self.band_losses(model, batches)
        return max(per) if per else float("inf")

    @staticmethod
    def band_losses(model: GPT, batches: List[dict]) -> List[float]:
        out = []
        for b in batches:
            l, k = model.eval_nll(b)
            out.append(l / max(1, k))
        return out


def consolidate(cfg, lay, steps_mult: float = 4.0, lr_scale: float = 0.5, replay_mult: float = 2.0, regression_scale: float = 0.5) -> dict:
    """One sleep pass over the pending feedback of a chat workspace. Always returns a dict: updated (bool), reason, bands_before / bands_after (loss per age band)."""
    from . import prepare as PR
    t0 = time.time()
    P = PR.prepare(cfg, lay, do_bench=False)
    eng = Engine(P.cfg, lay)
    if not eng.load():
        return dict(updated=False, reason="there is no trained chat model to consolidate", seconds=0.0)
    base = P.cfg
    lr = base.get("online.lr")
    over = {"online.steps_per_update": int(max(1, round(float(base.get("online.steps_per_update")) * steps_mult))),
            "online.replay_ratio": float(base.get("online.replay_ratio")) * replay_mult,
            "online.max_regression": float(base.get("online.max_regression")) * regression_scale}
    if isinstance(lr, (int, float)):
        over["online.lr"] = float(lr) * lr_scale
    else:
        peak = (lay.current() or {}).get("lr")
        over["online.lr_ratio"] = float(base.get("online.lr_ratio")) * lr_scale
    learner = ConsolidationLearner(base.replace(**over), lay, eng, P.corpus)
    if P.corpus is None or not len(P.corpus.train_idx):
        return dict(updated=False, reason="the training corpus (state/tokens) is needed as replay and as the forgetting gate: run `zerowhite chat prepare` first", seconds=0.0)
    ctx, pad = eng.model.cfg.ctx, eng.tpl.pad_id
    bands = learner._probe_batches(ctx, pad)
    before = learner.band_losses(eng.model, bands)
    out = learner.update(force=True)
    after = learner.band_losses(eng.model, bands)                                    # the engine holds the accepted model, or the untouched live one
    worst = max(range(len(before)), key=lambda i: after[i] / max(before[i], 1e-9)) if before else None
    return dict(out, bands_before=before, bands_after=after, worst_band=worst, seconds=round(time.time() - t0, 2),
                forgotten=bool(before) and max(a / max(b, 1e-9) for a, b in zip(after, before)) > 1.0 + float(over["online.max_regression"]))
