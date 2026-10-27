"""features.py - spectral features in numpy: log-mel filterbank, frame energy, pitch."""
from __future__ import annotations

import numpy as np


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


_FB_CACHE: dict = {}


def mel_filterbank(n_mels: int, n_fft: int, sr: int, fmin: float = 40.0, fmax: float = None) -> np.ndarray:
    """(n_mels, n_fft//2+1) triangular filters with unit-area-free peak 1 (Slaney-style spacing on the mel scale)."""
    fmax = fmax or sr / 2.0
    key = (n_mels, n_fft, sr, fmin, fmax)
    fb = _FB_CACHE.get(key)
    if fb is None:
        pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
        bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
        fb = np.zeros((n_mels, len(bins)), dtype=np.float32)
        for i in range(n_mels):
            lo, mid, hi = pts[i], pts[i + 1], pts[i + 2]
            up = (bins - lo) / max(mid - lo, 1e-9)
            down = (hi - bins) / max(hi - mid, 1e-9)
            fb[i] = np.maximum(0.0, np.minimum(up, down))
        _FB_CACHE[key] = fb
    return fb


def frame_signal(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """(n_frames, win) view-copy of overlapping frames; the tail is zero-padded to a whole frame."""
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win + hop - 1) // hop
    need = (n - 1) * hop + win
    if need > len(x):
        x = np.pad(x, (0, need - len(x)))
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def log_mel(x: np.ndarray, sr: int, n_mels: int = 40, win_ms: float = 25.0, hop_ms: float = 10.0, preemph: float = 0.97) -> np.ndarray:
    """(frames, n_mels) float32 log-mel energies."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) > 1 and preemph:
        x = np.concatenate([x[:1], x[1:] - preemph * x[:-1]])
    win = max(8, int(round(sr * win_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    n_fft = 1 << int(np.ceil(np.log2(win)))
    fr = frame_signal(x, win, hop) * np.hanning(win).astype(np.float32)[None, :]
    spec = np.abs(np.fft.rfft(fr, n=n_fft, axis=1)) ** 2
    mel = spec @ mel_filterbank(n_mels, n_fft, sr).T
    return np.log(mel + 1e-6).astype(np.float32)


def frame_db(x: np.ndarray, sr: int, frame_ms: float = 20.0) -> np.ndarray:
    """RMS level of consecutive non-overlapping frames, in dB relative to full scale."""
    n = max(1, int(round(sr * frame_ms / 1000.0)))
    m = len(x) // n
    if m == 0:
        return np.array([-120.0])
    fr = np.asarray(x[: m * n], dtype=np.float64).reshape(m, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-6))


def estimate_f0(x: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 400.0, frame_ms: float = 40.0, hop_ms: float = 20.0,
                voiced_threshold: float = 0.45) -> np.ndarray:
    """Per-frame fundamental frequency (Hz) by normalised autocorrelation; unvoiced frames are NaN."""
    win = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    if len(x) < win:
        return np.array([])
    fr = frame_signal(np.asarray(x, np.float32), win, hop)
    fr = fr - fr.mean(axis=1, keepdims=True)
    lo, hi = int(sr / fmax), min(int(sr / fmin), win - 1)
    energy = (fr ** 2).sum(axis=1)
    n_fft = 1 << int(np.ceil(np.log2(2 * win)))
    F = np.fft.rfft(fr, n=n_fft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), n=n_fft, axis=1)[:, : hi + 1]
    ac = ac / np.maximum(ac[:, :1], 1e-12)
    seg = ac[:, lo: hi + 1]
    k = seg.argmax(axis=1)
    peak = seg[np.arange(len(seg)), k]
    f0 = sr / (k + lo).astype(np.float64)
    gate = energy > 0.05 * np.median(energy[energy > 0]) if np.any(energy > 0) else np.zeros(len(energy), bool)
    f0[(peak < voiced_threshold) | ~gate] = np.nan
    return f0
