"""stt.py - speech recognition: the built-in recogniser you train yourself, plus engines the machine may already have.

Engines (voice.stt.engine; `auto` picks the first usable of faster_whisper, vosk, whisper_cpp, windows_sapi, builtin):
    builtin          the numpy CTC model (ctc.py) trained by `zerowhite chat voice train-stt` on the speech in data/audio
    faster_whisper   `pip install faster-whisper`; voice.stt.model = a model name or folder
    vosk             `pip install vosk`;           voice.stt.model = the unpacked model folder
    whisper_cpp      whisper.cpp's `whisper-cli`;  voice.stt.model = a ggml .bin file
    windows_sapi     Windows' own recogniser through PowerShell (nothing to install)
    command          your own program: voice.stt.command with {wav}; it prints the transcript
All take 16 kHz mono float32 audio and return text.
"""
from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
from typing import List, Optional, Tuple

import numpy as np

from .. import fileio
from . import audio as A
from . import ctc as C
from . import features as F
from .vad import VadParams, split_utterances

SR = 16000


class STTError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------------------------
# the built-in model file
# ---------------------------------------------------------------------------------------------------------------------
def raw_features(x: np.ndarray, sr: int, n_mels: int, win_ms: float, hop_ms: float, floor_db: float = 50.0) -> np.ndarray:
    """Log-mel frames (T, n_mels) with the dynamic range limited to `floor_db` below the loudest frame, so digital silence and a
    quiet room look the same."""
    if sr != SR:
        x = A.resample(x, sr, SR)
    f = F.log_mel(x, SR, n_mels, win_ms, hop_ms)
    if floor_db:
        f = np.maximum(f, f.max() - floor_db * math.log(10.0) / 10.0)
    return f.astype(np.float32)


def silence_value(f: np.ndarray) -> float:
    return float(f.min())                                                   # the floor: raw_features never goes below it


def pad_silence(f: np.ndarray, lead: int, tail: int) -> np.ndarray:
    v = silence_value(f)
    return np.concatenate([np.full((lead, f.shape[1]), v, np.float32), f, np.full((tail, f.shape[1]), v, np.float32)])


def normalize(f: np.ndarray) -> np.ndarray:
    """Per-utterance mean / variance normalisation."""
    return ((f - f.mean(axis=0, keepdims=True)) / (f.std(axis=0, keepdims=True) + 1e-3)).astype(np.float32)


def featurize(x: np.ndarray, sr: int, n_mels: int, win_ms: float, hop_ms: float, floor_db: float = 50.0) -> np.ndarray:
    return normalize(raw_features(x, sr, n_mels, win_ms, hop_ms, floor_db))


class BuiltinModel:
    def __init__(self, model: C.CTCModel, alphabet: List[str], meta: dict):
        self.model, self.alphabet, self.meta = model, list(alphabet), dict(meta)
        self.max_s = float(meta.get("max_seconds", 12.0))

    @property
    def n_mels(self) -> int:
        return self.model.cfg.n_in

    def ids_to_text(self, ids: List[int]) -> str:
        return "".join(self.alphabet[i - 1] for i in ids if 1 <= i <= len(self.alphabet)).strip()

    def text_to_ids(self, text: str) -> List[int]:
        index = {c: i + 1 for i, c in enumerate(self.alphabet)}
        return [index[c] for c in text if c in index]

    def feats(self, x: np.ndarray, sr: int = SR) -> np.ndarray:
        """Features exactly as the model was trained on them: floored log-mel, a stretch of silence either side, normalised."""
        f = raw_features(x, sr, self.n_mels, float(self.meta.get("win_ms", 25.0)), float(self.meta.get("hop_ms", 10.0)), float(self.meta.get("floor_db", 0.0)))
        n = self.pad_frames()
        return normalize(pad_silence(f, n, n))

    def pad_frames(self) -> int:
        return int(round(float(self.meta.get("pad_s", 0.25)) * 1000.0 / float(self.meta.get("hop_ms", 10.0))))

    def _one(self, x: np.ndarray) -> str:
        X = self.feats(x)[None]
        logits, lo, _ = self.model.forward(X, np.array([X.shape[1]]))
        return " ".join(self.ids_to_text(C.greedy_decode(logits[0, : lo[0]])).split())

    def transcribe(self, x: np.ndarray, sr: int = SR, cfg=None) -> str:
        if sr != SR:
            x = A.resample(x, sr, SR)
        if len(x) < SR // 5:
            return ""
        limit = int(self.max_s * 1.5 * SR)
        if len(x) <= limit:
            return self._one(x)
        spans = split_utterances(x, SR, cfg, VadParams(cfg)) if cfg is not None else []
        pieces: List[np.ndarray] = []
        for s, e in (spans or [(0, len(x))]):
            seg = x[s:e]
            n = max(1, int(np.ceil(len(seg) / limit)))
            step = len(seg) // n
            pieces += [seg[i * step:(i + 1) * step if i < n - 1 else len(seg)] for i in range(n)]
        return " ".join(t for t in (self._one(p) for p in pieces if len(p) >= SR // 5) if t)

    # ---- files ----------------------------------------------------------------------------------------------------
    def save(self, path: str):
        arrays = dict(self.model.p)
        info = dict(config=self.model.cfg.to_dict(), alphabet=self.alphabet, meta=self.meta)
        arrays["__meta__"] = np.frombuffer(json.dumps(info, default=fileio._json_default).encode("utf-8"), np.uint8)
        fileio.save_npz(path, arrays)

    @classmethod
    def load(cls, path: str) -> "BuiltinModel":
        z = fileio.load_npz(path)
        info = json.loads(bytes(z["__meta__"]).decode("utf-8"))
        cfg = C.CTCConfig.from_dict(info["config"])
        model = C.CTCModel(cfg, params={k: z[k] for k in z if not k.startswith("__")})
        return cls(model, info["alphabet"], dict(info.get("meta", {})))


def builtin_path(cfg, lay) -> str:
    m = cfg.get("voice.stt.model")
    if m and cfg.get("voice.stt.engine") in ("builtin", "auto") and str(m).lower().endswith(".npz"):
        return cfg.resolve_path(str(m))
    return os.path.join(lay.voice_models, "stt.npz")


# ---------------------------------------------------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------------------------------------------------
class STTEngine:
    name = "?"

    def __init__(self, cfg, lay):
        self.cfg, self.lay = cfg, lay
        self.lang = str(cfg.get("voice.stt.language"))
        self.model_ref = cfg.get("voice.stt.model")

    def available(self) -> Tuple[bool, str]:
        return True, ""

    def transcribe(self, x: np.ndarray, sr: int = SR) -> str:
        raise NotImplementedError


def _wav_tmp(x: np.ndarray, sr: int) -> str:
    fd, p = tempfile.mkstemp(suffix=".wav", prefix="zw_stt_")
    os.close(fd)
    with open(p, "wb") as f:
        f.write(A.encode_wav(x, sr))
    return p


def _rm(p: str):
    try:
        os.remove(p)
    except OSError:
        pass


class BuiltinSTT(STTEngine):
    name = "builtin"
    _cache: dict = {}

    def _path(self) -> str:
        return builtin_path(self.cfg, self.lay)

    def available(self):
        p = self._path()
        return (True, "") if os.path.isfile(p) else (False, f"no trained model at {p}: put speech + transcripts in {self.lay.audio_data} and run `zerowhite chat voice train-stt`")

    def model(self) -> BuiltinModel:
        p = self._path()
        key = (p, os.path.getmtime(p))
        m = BuiltinSTT._cache.get(p)
        if m is None or m[0] != key:
            m = (key, BuiltinModel.load(p))
            BuiltinSTT._cache[p] = m
        return m[1]

    def transcribe(self, x, sr=SR):
        return self.model().transcribe(x, sr, self.cfg)


class FasterWhisperSTT(STTEngine):
    name = "faster_whisper"
    _m = None

    def available(self):
        try:
            import faster_whisper  # noqa: F401
        except Exception as e:                                             # noqa: BLE001
            return False, f"faster-whisper is not installed ({type(e).__name__}); pip install faster-whisper"
        return (True, "") if self.model_ref else (False, "set voice.stt.model (a model name such as 'base', or a folder)")

    def transcribe(self, x, sr=SR):
        from faster_whisper import WhisperModel
        if FasterWhisperSTT._m is None:
            ref = str(self.model_ref)
            FasterWhisperSTT._m = WhisperModel(self.cfg.resolve_path(ref) if os.path.exists(self.cfg.resolve_path(ref)) else ref, compute_type="int8")
        if sr != SR:
            x = A.resample(x, sr, SR)
        segs, _ = FasterWhisperSTT._m.transcribe(x.astype(np.float32), language=self.lang or None, vad_filter=False)
        return " ".join(s.text.strip() for s in segs).strip()


class VoskSTT(STTEngine):
    name = "vosk"
    _m = None

    def available(self):
        try:
            import vosk  # noqa: F401
        except Exception as e:                                             # noqa: BLE001
            return False, f"vosk is not installed ({type(e).__name__}); pip install vosk"
        if not self.model_ref or not os.path.isdir(self.cfg.resolve_path(str(self.model_ref))):
            return False, "set voice.stt.model to the unpacked Vosk model folder"
        return True, ""

    def transcribe(self, x, sr=SR):
        import vosk
        if VoskSTT._m is None:
            vosk.SetLogLevel(-1)
            VoskSTT._m = vosk.Model(self.cfg.resolve_path(str(self.model_ref)))
        if sr != SR:
            x = A.resample(x, sr, SR)
        rec = vosk.KaldiRecognizer(VoskSTT._m, SR)
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        for i in range(0, len(pcm), 8000):
            rec.AcceptWaveform(pcm[i:i + 8000])
        return json.loads(rec.FinalResult()).get("text", "").strip()


class WhisperCppSTT(STTEngine):
    name = "whisper_cpp"

    def _exe(self):
        for n in ("whisper-cli", "whisper-cpp", "whisper"):
            p = shutil.which(n)
            if p:
                return p
        return None

    def available(self):
        if not self._exe():
            return False, "whisper.cpp's `whisper-cli` was not found on PATH"
        return (True, "") if self.model_ref and os.path.isfile(self.cfg.resolve_path(str(self.model_ref))) else (False, "set voice.stt.model to a ggml .bin model")

    def transcribe(self, x, sr=SR):
        if sr != SR:
            x = A.resample(x, sr, SR)
        wav = _wav_tmp(x, SR)
        try:
            cmd = [self._exe(), "-m", self.cfg.resolve_path(str(self.model_ref)), "-f", wav, "-nt", "-np"] + (["-l", self.lang] if self.lang else [])
            r = subprocess.run(cmd, capture_output=True, timeout=600)
            if r.returncode != 0:
                raise STTError("whisper.cpp failed: " + r.stderr.decode("utf-8", "replace")[-300:])
            return " ".join(ln.strip() for ln in r.stdout.decode("utf-8", "replace").splitlines() if ln.strip()).strip()
        except (OSError, subprocess.TimeoutExpired) as e:
            raise STTError(f"whisper.cpp: {e}") from None
        finally:
            _rm(wav)


_SAPI_STT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$culture = if ($env:ZW_LANG) { New-Object System.Globalization.CultureInfo($env:ZW_LANG) } else { [System.Globalization.CultureInfo]::CurrentCulture }
try { $r = New-Object System.Speech.Recognition.SpeechRecognitionEngine($culture) } catch { $r = New-Object System.Speech.Recognition.SpeechRecognitionEngine }
$r.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
$r.SetInputToWaveFile($env:ZW_WAV)
$parts = @()
while ($true) { $res = $r.Recognize(); if ($res -eq $null) { break }; $parts += $res.Text }
$r.Dispose()
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Write-Output ($parts -join ' ')
"""


class SapiSTT(STTEngine):
    name = "windows_sapi"

    def available(self):
        if os.name != "nt":
            return False, "Windows speech recognition is only on Windows"
        return (True, "") if (shutil.which("powershell") or shutil.which("pwsh")) else (False, "PowerShell was not found")

    def transcribe(self, x, sr=SR):
        ps = shutil.which("powershell") or shutil.which("pwsh")
        if sr != SR:
            x = A.resample(x, sr, SR)
        wav = _wav_tmp(x, SR)
        try:
            env = dict(os.environ, ZW_WAV=wav, ZW_LANG=self.lang if "-" in self.lang else "")
            r = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", _SAPI_STT], capture_output=True, env=env, timeout=300)
            if r.returncode != 0:
                raise STTError("Windows speech recognition failed: " + r.stderr.decode("utf-8", "replace")[-300:])
            return r.stdout.decode("utf-8", "replace").strip()
        except (OSError, subprocess.TimeoutExpired) as e:
            raise STTError(f"windows_sapi: {e}") from None
        finally:
            _rm(wav)


class CommandSTT(STTEngine):
    name = "command"

    def available(self):
        return (True, "") if self.cfg.get("voice.stt.command") else (False, "set voice.stt.command (a template with {wav})")

    def transcribe(self, x, sr=SR):
        if sr != SR:
            x = A.resample(x, sr, SR)
        wav = _wav_tmp(x, SR)
        try:
            tpl = str(self.cfg.get("voice.stt.command"))
            cmd = tpl.replace("{wav}", shlex.quote(wav) if os.name != "nt" else f'"{wav}"')
            r = subprocess.run(cmd, shell=True, capture_output=True, timeout=600)
            if r.returncode != 0:
                raise STTError("stt command failed: " + r.stderr.decode("utf-8", "replace")[-300:])
            return r.stdout.decode("utf-8", "replace").strip()
        except (OSError, subprocess.TimeoutExpired) as e:
            raise STTError(f"stt command: {e}") from None
        finally:
            _rm(wav)


ENGINES = {c.name: c for c in (BuiltinSTT, FasterWhisperSTT, VoskSTT, WhisperCppSTT, SapiSTT, CommandSTT)}
AUTO_ORDER = ["faster_whisper", "vosk", "whisper_cpp", "windows_sapi", "builtin"]


def report(cfg, lay) -> dict:
    return {n: c(cfg, lay).available() for n, c in ENGINES.items()}


def make(cfg, lay, name: Optional[str] = None) -> STTEngine:
    want = name or cfg.get("voice.stt.engine")
    if want != "auto":
        if want not in ENGINES:
            raise STTError(f"unknown stt engine '{want}'; choose from {sorted(ENGINES)}")
        e = ENGINES[want](cfg, lay)
        ok, why = e.available()
        if not ok:
            raise STTError(f"stt engine '{want}' is not usable: {why}")
        return e
    reasons = []
    for n in AUTO_ORDER:
        e = ENGINES[n](cfg, lay)
        ok, why = e.available()
        if ok:
            return e
        reasons.append(f"{n}: {why}")
    raise STTError("no speech recogniser is available. Options:\n  - " + "\n  - ".join(reasons))
