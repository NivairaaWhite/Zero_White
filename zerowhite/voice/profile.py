"""profile.py - measure a speaker from a speech dataset so the built-in synthesizer can imitate pitch and pace.

`zerowhite chat voice profile` reads clips from the audio data folder (LJ Speech, VCTK, your own recordings), measures the median
fundamental frequency of the voiced frames and the speaking rate (phonemes per second of speech, from the transcripts), and writes
<models>/voice/tts_profile.json. The built-in voice uses it unless voice.tts.builtin_f0_hz / builtin_tempo are set.
This copies pitch and tempo only - the built-in synthesizer is a formant synthesizer, not a voice-cloning model.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np

from .. import fileio
from . import audio as A
from . import datasets as D
from . import features as F
from . import g2p
from .vad import split_utterances

SR = 16000


CALIBRATION_TEXTS = ("The old man likes two apples every day.", "My friend wants some water here.", "A young girl takes the small boat now.",
                     "She is going to the market to buy some bread.")


def _analyse(x: np.ndarray, cfg, text: str):
    """(median f0 of voiced frames, phonemes per second of speech) of one clip, or None."""
    f0 = F.estimate_f0(x, SR)
    f0 = f0[~np.isnan(f0)] if len(f0) else f0
    if len(f0) < 10:
        return None
    spans = split_utterances(x, SR, cfg)
    speech_s = sum(e - s for s, e in spans) / SR if spans else len(x) / SR
    n_ph = g2p.text_to_phones(text)
    if speech_s < 0.5 or n_ph < 3:
        return None
    return float(np.median(f0)), n_ph / speech_s


def _synth_response(cfg) -> tuple:
    """How the analysis above reads the built-in voice at its nominal settings: (measured f0 / f0 setting, measured rate / tempo setting).
    Measuring real speech through the same pipeline and dividing by this maps it onto the synthesizer's own scale."""
    from . import formant as FM
    f0_set, tempo_set = 120.0, FM.BASE_TEMPO
    fs, ts = [], []
    for text in CALIBRATION_TEXTS:
        got = _analyse(FM.FormantSynth(FM.Voice(f0_hz=f0_set, tempo=tempo_set)).synthesize(text), cfg, text)
        if got:
            fs.append(got[0] / f0_set)
            ts.append(got[1] / tempo_set)
    return (float(np.median(fs)), float(np.median(ts))) if fs else (1.0, 1.0)


def measure(cfg, lay, source: Optional[str] = None, clips: int = 200, progress: Optional[Callable[[str], None]] = None) -> dict:
    root = source or lay.audio_data
    pairs, rep = D.discover(root, clips * 3)
    if not pairs:
        raise RuntimeError(f"no (audio, transcript) pairs found in {root}")
    f0s, rates, used = [], [], 0
    for path, text in pairs:
        if used >= clips:
            break
        try:
            x, _ = A.load_audio(path, SR, True, cfg)
        except A.AudioError:
            continue
        if not 1.0 <= len(x) / SR <= 20.0:
            continue
        got = _analyse(x, cfg, text)
        if got is None:
            continue
        f0s.append(got[0])
        rates.append(got[1])
        used += 1
        if progress and used % 25 == 0:
            progress(f"measured {used} clips")
    if used < 5:
        raise RuntimeError(f"only {used} usable clips; need at least 5")
    kf, kt = _synth_response(cfg)
    prof = dict(f0_hz=round(float(np.median(f0s)) / kf, 1), tempo=round(float(np.median(rates)) / kt, 2), clips=used, source=root,
                f0_spread_hz=round(float(np.std(f0s)) / kf, 1), measured_f0_hz=round(float(np.median(f0s)), 1),
                measured_phonemes_per_s=round(float(np.median(rates)), 2))
    fileio.write_json(os.path.join(lay.voice_models, "tts_profile.json"), prof, indent=2)
    return prof
