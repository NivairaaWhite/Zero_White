"""vad.py - voice activity detection that calibrates itself to the room.

The detector works on 20 ms frame energies. It measures the noise floor (the first `voice.vad.calibration_s` seconds, then a slowly
adapting estimate during silence) and calls a frame speech when it is `margin_db` above that floor - or above the absolute
`voice.vad.threshold_db` if you set one. Nothing is tuned to one microphone.

    seg = Segmenter(cfg, sr)            # live: feed audio chunks as they arrive
    for utterance in seg.feed(chunk): ...
    split_utterances(x, sr, cfg)        # offline: [(start, end), ...] in samples
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from .features import frame_db

FRAME_MS = 20.0
MIN_FLOOR_DB = -75.0            # digital silence is not a room; never calibrate below this
DIGITAL_SILENCE_DB = -100.0


class VadParams:
    def __init__(self, cfg):
        g = cfg.get
        thr = g("voice.vad.threshold_db", None)
        self.abs_threshold = None if thr in (None, "auto") else float(thr)
        self.margin_db = float(g("voice.vad.margin_db"))
        self.calibration_s = float(g("voice.vad.calibration_s"))
        self.min_speech_ms = float(g("voice.vad.min_speech_ms"))
        self.end_silence_ms = float(g("voice.vad.end_silence_ms"))
        self.pre_roll_ms = float(g("voice.vad.pre_roll_ms"))
        self.max_utterance_s = float(g("voice.vad.max_utterance_s"))


class Segmenter:
    """Streaming utterance splitter."""

    def __init__(self, cfg, sr: int, params: Optional[VadParams] = None):
        self.p = params or VadParams(cfg)
        self.sr = int(sr)
        self.n = max(1, int(round(self.sr * FRAME_MS / 1000.0)))
        self._buf = np.zeros(0, np.float32)
        self._calib: List[float] = []
        self._need_calib = int(round(self.p.calibration_s * 1000.0 / FRAME_MS))
        self.floor_db: Optional[float] = None
        self._pre = deque(maxlen=max(1, int(self.p.pre_roll_ms / FRAME_MS)))
        self._utt: List[np.ndarray] = []
        self._speech = 0
        self._silence = 0
        self._active = False
        self._min_f = max(1, int(round(self.p.min_speech_ms / FRAME_MS)))
        self._end_f = max(1, int(round(self.p.end_silence_ms / FRAME_MS)))
        self._max_f = max(self._min_f + 1, int(round(self.p.max_utterance_s * 1000.0 / FRAME_MS)))

    @property
    def threshold_db(self) -> Optional[float]:
        if self.p.abs_threshold is not None:
            return self.p.abs_threshold
        return None if self.floor_db is None else self.floor_db + self.p.margin_db

    @property
    def calibrated(self) -> bool:
        return self.p.abs_threshold is not None or self.floor_db is not None

    def _calibrate(self, db: float):
        self._calib.append(db)
        if len(self._calib) >= self._need_calib:
            self.floor_db = max(MIN_FLOOR_DB, float(np.percentile(self._calib, 75)))      # a bit of speech in the window must not raise it much

    def reset(self):
        self._utt, self._pre = [], deque(maxlen=self._pre.maxlen)
        self._speech = self._silence = 0
        self._active = False

    def feed(self, chunk: np.ndarray) -> List[np.ndarray]:
        """Add audio; returns the utterances (float32 arrays) completed by it."""
        out: List[np.ndarray] = []
        self._buf = np.concatenate([self._buf, np.asarray(chunk, np.float32).reshape(-1)])
        while len(self._buf) >= self.n:
            fr, self._buf = self._buf[: self.n], self._buf[self.n:]
            db = float(frame_db(fr, self.sr, FRAME_MS)[0])
            if not self.calibrated:
                self._calibrate(db)
                continue
            if self.floor_db is None:
                self.floor_db = max(MIN_FLOOR_DB, db)                      # absolute threshold: still track the floor for reporting
            thr = self.threshold_db
            is_speech = db >= thr
            if not self._active:
                if not is_speech and self.p.abs_threshold is None:
                    self.floor_db = max(MIN_FLOOR_DB, 0.97 * self.floor_db + 0.03 * db)        # follow slow changes in the room
                self._pre.append(fr)
                if is_speech:
                    self._speech += 1
                    if self._speech >= self._min_f:
                        self._active, self._silence = True, 0
                        self._utt = list(self._pre)
                        self._pre.clear()
                else:
                    self._speech = 0
            else:
                self._utt.append(fr)
                self._silence = 0 if is_speech else self._silence + 1
                if self._silence >= self._end_f or len(self._utt) >= self._max_f:
                    keep = len(self._utt) - self._silence + min(self._silence, 3)           # keep a short tail, drop the rest of the silence
                    out.append(np.concatenate(self._utt[: max(keep, 1)]))
                    self.reset()
        return out

    def flush(self) -> Optional[np.ndarray]:
        """Whatever speech is pending when the stream ends (None if it is shorter than min_speech_ms)."""
        if self._active and len(self._utt) - self._silence >= self._min_f:
            out = np.concatenate(self._utt[: max(1, len(self._utt) - self._silence + 3)])
        else:
            out = None
        self.reset()
        return out


def split_utterances(x: np.ndarray, sr: int, cfg, params: Optional[VadParams] = None) -> List[Tuple[int, int]]:
    """Offline segmentation of a recording: [(start_sample, end_sample), ...]."""
    p = params or VadParams(cfg)
    db = frame_db(x, sr, FRAME_MS)
    n = max(1, int(round(sr * FRAME_MS / 1000.0)))
    if len(db) == 0:
        return []
    if p.abs_threshold is not None:
        thr = p.abs_threshold
    else:
        live = db[db > DIGITAL_SILENCE_DB]                                    # exact zeros (a muted gap, a synthesised pause) are no noise floor
        floor = max(MIN_FLOOR_DB, float(np.percentile(live if live.size else db, 10)))
        if float(db.max()) - floor < p.margin_db:
            return [(0, len(x))] if float(db.max()) > -45.0 else []          # flat signal: all one sound, or silence
        thr = floor + p.margin_db
    speech = db >= thr
    min_f = max(1, int(round(p.min_speech_ms / FRAME_MS)))
    end_f = max(1, int(round(p.end_silence_ms / FRAME_MS)))
    pre = max(0, int(round(p.pre_roll_ms / FRAME_MS)))
    max_f = max(min_f + 1, int(round(p.max_utterance_s * 1000.0 / FRAME_MS)))
    segs: List[Tuple[int, int]] = []
    i, T = 0, len(speech)
    while i < T:
        if not speech[i]:
            i += 1
            continue
        j, last, silent = i, i, 0
        while j < T and (j - i) < max_f:
            if speech[j]:
                last, silent = j, 0
            else:
                silent += 1
                if silent >= end_f:
                    break
            j += 1
        if last - i + 1 >= min_f:
            segs.append((max(0, i - pre) * n, min(len(x), (last + 1 + 3) * n)))
        i = max(j, last + 1)
    merged: List[Tuple[int, int]] = []
    for s, e in segs:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged
