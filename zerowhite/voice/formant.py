"""formant.py - the built-in speech synthesizer: rules -> formant tracks -> waveform, in numpy only.

Text goes through g2p.py to phonemes. Each phoneme has formant targets, a duration and a source type (voiced / noise / burst); the
targets are joined by linear interpolation (this makes the formant transitions into and out of consonants, which is what makes
speech intelligible). A pitch contour is added (declination, stress accents, question rise). The waveform is built in the
frequency domain: every pitch period is the inverse FFT of (glottal tilt x formant resonances), overlap-added at the pitch marks;
fricatives and bursts are noise shaped by band filters, frame by frame. Fast (much faster than real time) and dependency-free.

It sounds like a robot from the 1980s. It exists so that voice works with nothing installed, and as the reference voice the
recogniser is tested against. Real voices: `voice.tts.engine` = espeak / say / windows_sapi / piper.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import g2p

HOP_MS = 5.0
BASE_TEMPO = 12.0              # phonemes per second the duration table below corresponds to

# kind, duration ms, formants (F1,F2,F3) or [two knots] for diphthongs, amplitude
VOWELS: Dict[str, tuple] = {
    "IY": (120, (270, 2290, 3010), 1.0), "IH": (80, (390, 1990, 2550), 1.0), "EH": (95, (530, 1840, 2480), 1.0),
    "AE": (125, (660, 1720, 2410), 1.0), "AA": (125, (730, 1090, 2440), 1.0), "AO": (125, (570, 840, 2410), 1.0),
    "UH": (85, (440, 1020, 2240), 1.0), "UW": (120, (300, 870, 2240), 1.0), "AH": (85, (640, 1190, 2390), 1.0),
    "ER": (125, (490, 1350, 1690), 1.0), "AX": (60, (500, 1500, 2500), 0.8),
    "EY": (170, [(480, 2000, 2600), (330, 2250, 2950)], 1.0), "AY": (180, [(730, 1100, 2440), (340, 2100, 2800)], 1.0),
    "OY": (190, [(570, 850, 2410), (340, 2000, 2800)], 1.0), "AW": (180, [(730, 1100, 2440), (380, 900, 2300)], 1.0),
    "OW": (170, [(480, 950, 2400), (340, 800, 2300)], 1.0),
}
SONORANTS: Dict[str, tuple] = {            # dur, formants, amplitude, nasal?
    "L": (70, (360, 1100, 2800), 0.9, False), "R": (75, (400, 1200, 1600), 0.9, False), "W": (65, (300, 700, 2300), 0.9, False),
    "Y": (60, (280, 2200, 3000), 0.9, False), "M": (75, (250, 1000, 2200), 0.7, True), "N": (70, (250, 1500, 2400), 0.7, True),
    "NG": (75, (250, 2000, 2400), 0.7, True),
}
FRICATIVES: Dict[str, tuple] = {            # dur, noise band (Hz), noise amp, voiced amp, formant locus
    "S": (110, (4200, 8000), 0.55, 0.0, (300, 1700, 2500)), "SH": (110, (2200, 5500), 0.6, 0.0, (300, 1900, 2600)),
    "F": (90, (1200, 8000), 0.16, 0.0, (300, 1200, 2500)), "TH": (90, (1200, 8000), 0.12, 0.0, (300, 1500, 2500)),
    "Z": (85, (4200, 8000), 0.32, 0.45, (300, 1700, 2500)), "ZH": (85, (2200, 5500), 0.35, 0.45, (300, 1900, 2600)),
    "V": (65, (1200, 8000), 0.1, 0.55, (300, 1200, 2500)), "DH": (55, (1200, 8000), 0.08, 0.55, (300, 1500, 2500)),
}
STOPS: Dict[str, tuple] = {                 # closure ms, burst band, burst amp, voiced, aspiration ms, locus formants
    "P": (60, (400, 3500), 0.35, False, 40, (200, 800, 2200)), "T": (55, (3500, 8000), 0.4, False, 40, (200, 1700, 2600)),
    "K": (60, (1200, 4000), 0.4, False, 45, (200, 1900, 2400)),
    "B": (50, (400, 3500), 0.18, True, 8, (200, 800, 2200)), "D": (45, (3500, 8000), 0.2, True, 8, (200, 1700, 2600)),
    "G": (50, (1200, 4000), 0.2, True, 10, (200, 1900, 2400)),
}
AFFRICATES: Dict[str, tuple] = {"CH": (45, (2200, 6000), 0.5, False, 60), "JH": (40, (2200, 6000), 0.4, True, 55)}
PAUSES = {",": 220, ";": 260, ".": 420, "?": 440, "!": 440}
FORMANT_BW = (60.0, 90.0, 140.0, 200.0, 250.0)
UPPER_FORMANTS = (3500.0, 4500.0)


def _smooth(a: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return a
    k = np.ones(n) / n
    return np.convolve(np.pad(a, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


class Voice:
    """Parameters of the built-in voice."""

    def __init__(self, f0_hz: float = 118.0, tempo: float = BASE_TEMPO, pitch: float = 1.0, rate: float = 1.0, volume: float = 1.0,
                 formant_scale: Optional[float] = None, sr: int = 16000, breathiness: float = 0.03):
        self.f0, self.tempo, self.pitch, self.rate, self.volume, self.sr = float(f0_hz), float(tempo), float(pitch), float(rate), float(volume), int(sr)
        self.formant_scale = float(formant_scale) if formant_scale else float(pitch) ** 0.35
        self.breath = float(breathiness)


class FormantSynth:
    def __init__(self, voice: Optional[Voice] = None, seed: int = 0):
        self.v = voice or Voice()
        self.rng = np.random.default_rng(seed)

    # ---- text -> timeline -------------------------------------------------------------------------------------------
    def _units(self, text: str) -> List[dict]:
        toks = g2p.expand_text(text)
        units: List[dict] = []
        for i, (t, kind) in enumerate(toks):
            if kind == "w":
                ph = g2p.word_to_phones(t)
                if not ph:
                    continue
                func = g2p.is_function_word(t)
                first_v = next((k for k, p in enumerate(ph) if p in VOWELS), None)
                for k, p in enumerate(ph):
                    units.append(dict(ph=p, stress=(0 if func else (2 if k == first_v else 1)) if p in VOWELS else 0, word_start=k == 0, word_end=k == len(ph) - 1))
                units.append(dict(ph="_", ms=25.0, soft=True))
            else:
                units.append(dict(ph="_", ms=float(PAUSES.get(kind, 250)), mark=kind))
        return units

    def synthesize(self, text: str) -> np.ndarray:
        v = self.v
        units = self._units(text)
        if not any(u["ph"] != "_" for u in units):
            return np.zeros(0, np.float32)
        if units[-1]["ph"] != "_" or units[-1].get("soft"):
            units.append(dict(ph="_", ms=300.0, mark="."))                  # an utterance ends like a sentence: final lengthening + a pause
        dscale = BASE_TEMPO / max(1.0, v.tempo) / max(0.2, v.rate)
        sr = v.sr
        hop = HOP_MS / 1000.0
        # --- schedule ------------------------------------------------------------------------------------------------
        ev: List[dict] = []
        t = 0.04
        n_units = len(units)
        for idx, u in enumerate(units):
            p = u["ph"]
            nxt_pause = idx + 1 < n_units and units[idx + 1]["ph"] == "_" and not units[idx + 1].get("soft")
            if p == "_":
                ev.append(dict(kind="pause", t0=t, t1=t + u["ms"] / 1000.0 * dscale, mark=u.get("mark")))
                t = ev[-1]["t1"]
                continue
            if p in VOWELS:
                dur, f, amp = VOWELS[p]
                st = u.get("stress", 1)
                d = dur * dscale * (1.2 if st == 2 else 0.75 if st == 0 else 1.0) * (1.35 if nxt_pause else 1.0)
                ev.append(dict(kind="vowel", t0=t, t1=t + d / 1000.0, f=f, amp=amp * (1.0 if st else 0.8), stress=st, ph=p))
            elif p in SONORANTS:
                dur, f, amp, nasal = SONORANTS[p]
                d = dur * dscale * (1.3 if nxt_pause else 1.0)
                ev.append(dict(kind="son", t0=t, t1=t + d / 1000.0, f=f, amp=amp, nasal=nasal, ph=p))
            elif p in FRICATIVES:
                dur, band, namp, vamp, loc = FRICATIVES[p]
                d = dur * dscale
                ev.append(dict(kind="fric", t0=t, t1=t + d / 1000.0, band=band, namp=namp, vamp=vamp, f=loc, ph=p))
            elif p in STOPS:
                clo, band, bamp, voiced, asp, loc = STOPS[p]
                d = (clo + 12 + asp) * dscale
                ev.append(dict(kind="stop", t0=t, t1=t + d / 1000.0, clo=clo * dscale / 1000.0, asp=asp * dscale / 1000.0, band=band, bamp=bamp, voiced=voiced, f=loc, ph=p))
            elif p in AFFRICATES:
                clo, band, namp, voiced, fr = AFFRICATES[p]
                d = (clo + fr) * dscale
                ev.append(dict(kind="affr", t0=t, t1=t + d / 1000.0, clo=clo * dscale / 1000.0, band=band, namp=namp, voiced=voiced, f=(300, 1900, 2600), ph=p))
            elif p == "HH":
                ev.append(dict(kind="asp", t0=t, t1=t + 0.06 * dscale, ph=p))
            else:
                continue
            t = ev[-1]["t1"]
        total = t + 0.15
        N = int(np.ceil(total / hop)) + 2
        tt = np.arange(N) * hop
        # --- tracks --------------------------------------------------------------------------------------------------
        AV = np.zeros(N)
        AF = np.zeros(N)
        AH = np.zeros(N)
        NASAL = np.zeros(N)
        band_lo = np.full(N, 3000.0)
        band_hi = np.full(N, 8000.0)
        knots: List[Tuple[float, Tuple[float, float, float]]] = []
        accents: List[Tuple[float, float]] = []

        def sl(t0, t1):
            return slice(max(0, int(round(t0 / hop))), min(N, max(int(round(t1 / hop)), int(round(t0 / hop)) + 1)))

        for e in ev:
            k = e["kind"]
            t0, t1 = e["t0"], e["t1"]
            mid = 0.5 * (t0 + t1)
            if k == "vowel":
                f = e["f"]
                if isinstance(f, list):
                    knots += [(t0 + 0.2 * (t1 - t0), f[0]), (t0 + 0.8 * (t1 - t0), f[1])]
                else:
                    knots.append((mid, f))
                AV[sl(t0, t1)] = e["amp"]
                if e["stress"] == 2:
                    accents.append((mid, t1 - t0))
            elif k == "son":
                knots.append((mid, e["f"]))
                AV[sl(t0, t1)] = e["amp"]
                if e["nasal"]:
                    NASAL[sl(t0, t1)] = 1.0
            elif k == "fric":
                knots.append((mid, e["f"]))
                AF[sl(t0, t1)] = e["namp"]
                AV[sl(t0, t1)] = e["vamp"]
                band_lo[sl(t0, t1)], band_hi[sl(t0, t1)] = e["band"]
            elif k == "stop":
                clo_end = t0 + e["clo"]
                knots.append((clo_end, e["f"]))
                if e["voiced"]:
                    AV[sl(t0, clo_end)] = 0.18                                    # voice bar
                b_end = clo_end + 0.012
                AF[sl(clo_end, b_end)] = e["bamp"]
                band_lo[sl(clo_end, b_end)], band_hi[sl(clo_end, b_end)] = e["band"]
                AH[sl(b_end, t1)] = 0.45 if not e["voiced"] else 0.12
            elif k == "affr":
                clo_end = t0 + e["clo"]
                knots.append((mid, e["f"]))
                if e["voiced"]:
                    AV[sl(t0, clo_end)] = 0.18
                    AV[sl(clo_end, t1)] = 0.4
                AF[sl(clo_end, t1)] = e["namp"]
                band_lo[sl(clo_end, t1)], band_hi[sl(clo_end, t1)] = e["band"]
            elif k == "asp":
                AH[sl(t0, t1)] = 0.55
        AV, AF, AH = _smooth(AV, 3), _smooth(AF, 2), _smooth(AH, 3)
        NASAL = _smooth(NASAL, 3)
        if not knots:
            knots = [(0.0, (500, 1500, 2500))]
        knots.sort(key=lambda x: x[0])
        kt = np.array([k[0] for k in knots])
        kf = np.array([k[1] for k in knots], dtype=np.float64)
        F = np.stack([np.interp(tt, kt, kf[:, i]) for i in range(3)], axis=1) * v.formant_scale
        # --- pitch ---------------------------------------------------------------------------------------------------
        f0 = np.full(N, v.f0 * v.pitch)
        # sentence-wise declination, reset after . ? !
        starts = [0.0] + [e["t1"] for e in ev if e["kind"] == "pause" and e.get("mark") in (".", "?", "!")]
        for a, b in zip(starts, starts[1:] + [total]):
            s = sl(a, b)
            n = s.stop - s.start
            if n > 0:
                f0[s] *= np.linspace(1.14, 0.84, n)
        for tm, du in accents:
            f0 *= 1.0 + 0.11 * np.exp(-0.5 * ((tt - tm) / max(0.05, du * 0.6)) ** 2)
        for e in ev:
            if e["kind"] == "pause" and e.get("mark") == "?":
                s = sl(e["t0"] - 0.35, e["t0"])
                n = s.stop - s.start
                if n > 0:
                    f0[s] *= np.linspace(1.0, 1.28, n)
        # --- render --------------------------------------------------------------------------------------------------
        out = np.zeros(int((total + 0.3) * sr) + 1024, np.float64)
        self._voiced(out, tt, F, f0, AV, NASAL, AH)
        self._noise(out, tt, F, AF, AH, band_lo, band_hi, hop)
        out = out[: int(total * sr)]
        peak = float(np.abs(out).max())
        if peak > 1e-9:
            out *= 0.8 * v.volume / peak
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    # ---- rendering --------------------------------------------------------------------------------------------------
    def _resonance(self, freqs: np.ndarray, F: np.ndarray, nasal: float) -> np.ndarray:
        """Magnitude response of the vocal tract at `freqs` for formants F (3,), plus fixed upper formants, glottal tilt and nasalisation."""
        fs = list(F) + list(UPPER_FORMANTS)
        H = np.ones_like(freqs)
        for fi, bw in zip(fs, FORMANT_BW):
            H = H / np.sqrt((1.0 - (freqs / fi) ** 2) ** 2 + (freqs * bw / fi ** 2) ** 2)
        H = H * fs[0] ** 0 * 1.0
        tilt = 1.0 / (1.0 + (freqs / 1100.0) ** 2)
        if nasal > 0:
            nas = 1.0 / np.sqrt((1.0 - (freqs / 270.0) ** 2) ** 2 + (freqs * 120.0 / 270.0 ** 2) ** 2)
            zero = np.sqrt((1.0 - (freqs / 900.0) ** 2) ** 2 + (freqs * 200.0 / 900.0 ** 2) ** 2)
            H = H * (1.0 - nasal + nasal * 0.6 * nas * zero / 3.0)
        return H * tilt

    def _voiced(self, out: np.ndarray, tt: np.ndarray, F: np.ndarray, f0: np.ndarray, AV: np.ndarray, NASAL: np.ndarray, AH: np.ndarray):
        sr = self.v.sr
        n_fft = 1024
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        pos = 0.04
        end = tt[-1]
        cache_key, cache_val = None, None
        while pos < end - 0.01:
            i = min(len(tt) - 1, int(pos / HOP_MS * 1000.0))
            period = 1.0 / max(60.0, f0[i])
            av = AV[i]
            if av > 0.02:
                key = (int(F[i, 0] / 15), int(F[i, 1] / 30), int(F[i, 2] / 40), round(float(NASAL[i]), 1))
                if key != cache_key:
                    H = self._resonance(freqs, F[i], float(NASAL[i]))
                    pulse = np.fft.irfft(H, n=n_fft)
                    pulse = np.roll(pulse, n_fft // 2)
                    cache_key, cache_val = key, pulse
                pulse = cache_val
                half = int(min(n_fft // 2 - 1, max(24, period * sr * 1.0)))
                seg = pulse[n_fft // 2 - half: n_fft // 2 + half] * np.hanning(2 * half)
                # derivative-like glottal shape: differentiate to avoid a DC bias and emphasise the closure instant
                seg = np.diff(seg, prepend=0.0)
                s = int(pos * sr) - half
                a = max(0, s)
                b = min(len(out), s + 2 * half)
                if b > a:
                    gain = av * (1.0 + self.v.breath * 0.0)
                    out[a:b] += gain * seg[a - s: b - s] * 3.2
            pos += period

    def _noise(self, out: np.ndarray, tt: np.ndarray, F: np.ndarray, AF: np.ndarray, AH: np.ndarray, lo: np.ndarray, hi: np.ndarray, hop: float):
        sr = self.v.sr
        win = int(round(2 * hop * sr))
        n_fft = 256
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        w = np.hanning(win + 1)[:win]
        for m in range(len(tt)):
            af, ah = AF[m], AH[m]
            if af < 0.01 and ah < 0.01:
                continue
            noise = self.rng.standard_normal(win)
            X = np.fft.rfft(noise, n=n_fft)
            shape = np.zeros_like(freqs)
            if af >= 0.01:
                edge = 600.0
                band = (1.0 / (1.0 + np.exp(-(freqs - lo[m]) / (edge / 4.0)))) * (1.0 / (1.0 + np.exp((freqs - hi[m]) / (edge / 4.0))))
                shape += af * 1.7 * band
            if ah >= 0.01:
                H = self._resonance(freqs, F[m], 0.0)
                shape += ah * 1.1 * H / max(1e-9, H.max()) * 1.6
            y = np.fft.irfft(X * shape, n=n_fft)[:win] * w
            s = int(round(tt[m] * sr)) - win // 2
            a, b = max(0, s), min(len(out), s + win)
            if b > a:
                out[a:b] += y[a - s: b - s] * 0.9


def synthesize(text: str, sr: int = 16000, **voice_kw) -> np.ndarray:
    return FormantSynth(Voice(sr=sr, **voice_kw)).synthesize(text)
