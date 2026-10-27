"""service.py - one object that gives the rest of the program speech in and speech out.

    v = VoiceService(cfg, lay)
    v.describe()                  which engines are in use / available, and why not
    v.transcribe(wav_bytes)       -> text            (any audio ffmpeg can read; WAV needs nothing)
    v.synthesize("Hello")         -> WAV bytes       (cached on disk)
Engines are chosen from the configuration (`auto` = best available) and created on first use.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Optional, Tuple, Union

import numpy as np

from . import audio as A
from . import stt as STT
from . import tts as TTS


def clean_for_speech(text: str, max_chars: int = 1500) -> str:
    """Make chat text speakable: drop code blocks, markdown marks and URLs."""
    t = re.sub(r"```.*?```", " (code omitted) ", text, flags=re.S)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"https?://\S+", " link ", t)
    t = re.sub(r"[*_#>~|]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_chars:
        cut = t.rfind(". ", 0, max_chars)
        t = t[: cut + 1 if cut > max_chars // 2 else max_chars]
    return t


class VoiceService:
    def __init__(self, cfg, lay):
        self.cfg, self.lay = cfg, lay
        self._tts: Optional[TTS.Engine] = None
        self._stt: Optional[STT.STTEngine] = None
        self._tts_err: Optional[str] = None
        self._stt_err: Optional[str] = None
        self._tlock, self._slock, self._init = threading.Lock(), threading.Lock(), threading.Lock()
        self.profile_path = os.path.join(lay.voice_models, "tts_profile.json")

    # ---- engines ---------------------------------------------------------------------------------------------------
    def tts_engine(self) -> TTS.Engine:
        with self._init:
            if self._tts is None:
                try:
                    self._tts = TTS.make(self.cfg, self.profile_path)
                    self._tts_err = None
                except TTS.TTSError as e:
                    self._tts_err = str(e)
                    raise
            return self._tts

    def stt_engine(self) -> STT.STTEngine:
        with self._init:
            if self._stt is None:
                try:
                    self._stt = STT.make(self.cfg, self.lay)
                    self._stt_err = None
                except STT.STTError as e:
                    self._stt_err = str(e)
                    raise
            return self._stt

    def reset(self):
        """Forget the chosen engines (after a config change or after training a new built-in model)."""
        with self._init:
            self._tts = self._stt = None

    def describe(self) -> dict:
        tts_rep = TTS.report(self.cfg, self.profile_path)
        stt_rep = STT.report(self.cfg, self.lay)
        out = dict(tts=None, stt=None, tts_engines={k: dict(available=v[0], note=v[1]) for k, v in tts_rep.items()},
                   stt_engines={k: dict(available=v[0], note=v[1]) for k, v in stt_rep.items()}, sample_rate=int(self.cfg.get("voice.sample_rate")))
        try:
            out["tts"] = self.tts_engine().name
        except TTS.TTSError as e:
            out["tts_error"] = str(e)
        try:
            out["stt"] = self.stt_engine().name
        except STT.STTError as e:
            out["stt_error"] = str(e)
        out["available"] = bool(out["tts"] or out["stt"])
        mp = STT.builtin_path(self.cfg, self.lay)
        if os.path.exists(mp):
            try:
                m = STT.BuiltinModel.load(mp)
                out["builtin_stt"] = dict(path=mp, val_cer=m.meta.get("val_cer"), hours=m.meta.get("hours"), params=m.model.num_params())
            except Exception as e:                                         # noqa: BLE001
                out["builtin_stt"] = dict(path=mp, error=str(e))
        return out

    # ---- speech to text --------------------------------------------------------------------------------------------
    def transcribe_array(self, x: np.ndarray, sr: int) -> str:
        eng = self.stt_engine()
        with self._slock:
            return eng.transcribe(x, sr).strip()

    def transcribe(self, audio: Union[bytes, str, np.ndarray], sr: Optional[int] = None) -> str:
        if isinstance(audio, np.ndarray):
            return self.transcribe_array(audio, sr or STT.SR)
        x, rate = A.load_audio(audio, STT.SR, True, self.cfg)
        return self.transcribe_array(x, rate)

    # ---- text to speech --------------------------------------------------------------------------------------------
    def synthesize_array(self, text: str) -> Tuple[np.ndarray, int]:
        text = clean_for_speech(text, int(self.cfg.get("voice.tts.max_chars")))
        if not text:
            return np.zeros(0, np.float32), int(self.cfg.get("voice.sample_rate"))
        eng = self.tts_engine()
        use_cache = bool(self.cfg.get("voice.tts.cache"))
        parts, rate = [], None
        for piece in TTS.split_sentences(text):
            key = TTS.cache_key(eng, piece)
            cp = os.path.join(self.lay.voice_cache, key + ".wav")
            x = None
            if use_cache and os.path.exists(cp):
                try:
                    x, r = A.load_audio(cp, None, True)
                    os.utime(cp, None)
                    rate = rate or r
                except A.AudioError:
                    x = None
            if x is None:
                with self._tlock:
                    x, r = eng.synth(piece)
                rate = rate or r
                if r != rate:
                    x, r = A.resample(x, r, rate), rate
                if use_cache and len(x):
                    try:
                        A.save_audio(cp, x, r)
                    except OSError:
                        pass
            parts.append(x)
            parts.append(np.zeros(int(0.12 * (rate or 16000)), np.float32))
        if use_cache:
            self._prune_cache()
        return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), int(rate or 16000)

    def synthesize(self, text: str) -> bytes:
        x, sr = self.synthesize_array(text)
        return A.encode_wav(x, sr)

    def _prune_cache(self):
        limit = float(self.cfg.get("voice.tts.cache_max_mb")) * 1024 * 1024
        try:
            files = [(os.path.join(self.lay.voice_cache, f),) for f in os.listdir(self.lay.voice_cache) if f.endswith(".wav")]
            info = sorted(((os.path.getmtime(p), os.path.getsize(p), p) for (p,) in files))
        except OSError:
            return
        total = sum(s for _, s, _ in info)
        for _, size, p in info:
            if total <= limit:
                break
            try:
                os.remove(p)
                total -= size
            except OSError:
                pass
