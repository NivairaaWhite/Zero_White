"""tts.py - speech synthesis: the built-in synthesizer plus whatever the machine already has.

Engines (voice.tts.engine; `auto` picks the best available in this order for your OS):
    windows_sapi   Windows' own voices through PowerShell + System.Speech   (nothing to install)
    say            macOS `say`                                              (nothing to install)
    piper          Piper neural voices: the `piper` program + voice.tts.model (.onnx)
    espeak         espeak-ng / espeak                                       (apt install espeak-ng)
    pyttsx3        the pyttsx3 package (SAPI / NSSpeech / eSpeak)
    command        your own program: voice.tts.command with {text} and {wav}
    builtin        the numpy formant synthesizer in formant.py             (always available, robotic)
Every engine returns (float32 samples, sample_rate). Nothing here is hard-wired to one voice: voice, rate, pitch and volume are settings.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

import numpy as np

from .. import fileio
from . import audio as A
from . import formant as FM


class TTSError(RuntimeError):
    pass


def _run(cmd, **kw) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=kw.pop("timeout", 180), **kw)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise TTSError(f"{cmd[0] if isinstance(cmd, list) else cmd}: {e}") from None


def _powershell() -> Optional[str]:
    return shutil.which("powershell") or shutil.which("pwsh")


def _read_wav(path: str) -> Tuple[np.ndarray, int]:
    x, sr = A.load_audio(path, None, True)
    return x, sr


def _tmp_wav() -> str:
    fd, p = tempfile.mkstemp(suffix=".wav", prefix="zw_tts_")
    os.close(fd)
    return p


class Engine:
    name = "?"

    def __init__(self, cfg):
        self.cfg = cfg
        g = cfg.get
        self.voice = g("voice.tts.voice")
        self.rate = float(g("voice.tts.rate"))
        self.pitch = float(g("voice.tts.pitch"))
        self.volume = float(g("voice.tts.volume"))

    def available(self) -> Tuple[bool, str]:
        return True, ""

    def voices(self) -> List[str]:
        return []

    def synth(self, text: str) -> Tuple[np.ndarray, int]:
        raise NotImplementedError


class BuiltinTTS(Engine):
    name = "builtin"

    def _profile(self) -> dict:
        p = getattr(self, "profile_path", None)
        try:
            return fileio.read_json(p) if p and os.path.exists(p) else {}
        except (OSError, ValueError):
            return {}

    def synth(self, text: str) -> Tuple[np.ndarray, int]:
        g = self.cfg.get
        prof = self._profile()
        f0 = g("voice.tts.builtin_f0_hz", None) or prof.get("f0_hz") or 118.0
        tempo = g("voice.tts.builtin_tempo", None) or prof.get("tempo") or FM.BASE_TEMPO
        v = FM.Voice(f0_hz=f0, tempo=tempo, pitch=self.pitch, rate=self.rate, volume=self.volume)
        return FM.FormantSynth(v).synthesize(text), v.sr


class EspeakTTS(Engine):
    name = "espeak"

    def _exe(self) -> Optional[str]:
        return shutil.which("espeak-ng") or shutil.which("espeak")

    def available(self):
        return (True, "") if self._exe() else (False, "espeak-ng is not installed (apt install espeak-ng / brew install espeak-ng)")

    def voices(self):
        exe = self._exe()
        if not exe:
            return []
        out = _run([exe, "--voices"]).stdout.decode("utf-8", "replace").splitlines()[1:]
        return [ln.split()[3] for ln in out if len(ln.split()) >= 4]

    def synth(self, text):
        exe = self._exe()
        if not exe:
            raise TTSError("espeak is not installed")
        wav = _tmp_wav()
        try:
            cmd = [exe, "-w", wav, "-s", str(int(max(80, min(450, 175 * self.rate)))), "-p", str(int(max(0, min(99, 50 * self.pitch)))),
                   "-a", str(int(max(0, min(200, 100 * self.volume))))]
            if self.voice:
                cmd += ["-v", str(self.voice)]
            r = _run(cmd + ["--", text])
            if r.returncode != 0:
                raise TTSError("espeak failed: " + r.stderr.decode("utf-8", "replace")[-200:])
            return _read_wav(wav)
        finally:
            _rm(wav)


class SayTTS(Engine):
    name = "say"

    def available(self):
        return (True, "") if sys.platform == "darwin" and shutil.which("say") else (False, "macOS `say` is only on macOS")

    def voices(self):
        if not shutil.which("say"):
            return []
        out = _run(["say", "-v", "?"]).stdout.decode("utf-8", "replace").splitlines()
        return [ln.split()[0] for ln in out if ln.strip()]

    def synth(self, text):
        wav = _tmp_wav()
        try:
            cmd = ["say", "-o", wav, "--file-format=WAVE", "--data-format=LEI16@22050", "-r", str(int(175 * self.rate))]
            if self.voice:
                cmd += ["-v", str(self.voice)]
            r = _run(cmd + ["--", text])
            if r.returncode != 0:
                raise TTSError("say failed: " + r.stderr.decode("utf-8", "replace")[-200:])
            x, sr = _read_wav(wav)
            return x * min(1.0, self.volume), sr
        finally:
            _rm(wav)


_SAPI_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($env:ZW_VOICE) { try { $s.SelectVoice($env:ZW_VOICE) } catch { } }
$s.Rate = [int]$env:ZW_RATE
$s.Volume = [int]$env:ZW_VOLUME
$s.SetOutputToWaveFile($env:ZW_WAV)
$s.Speak([System.IO.File]::ReadAllText($env:ZW_TEXTFILE, [System.Text.Encoding]::UTF8))
$s.Dispose()
"""


class SapiTTS(Engine):
    name = "windows_sapi"

    def available(self):
        if os.name != "nt":
            return False, "Windows SAPI is only on Windows"
        return (True, "") if _powershell() else (False, "PowerShell was not found")

    def voices(self):
        ps = _powershell()
        if os.name != "nt" or not ps:
            return []
        r = _run([ps, "-NoProfile", "-Command", "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }"])
        return [ln.strip() for ln in r.stdout.decode("utf-8", "replace").splitlines() if ln.strip()]

    def synth(self, text):
        ps = _powershell()
        if not ps:
            raise TTSError("PowerShell not found")
        wav = _tmp_wav()
        fd, txt = tempfile.mkstemp(suffix=".txt", prefix="zw_tts_")
        os.close(fd)
        try:
            with open(txt, "w", encoding="utf-8") as f:
                f.write(text)
            env = dict(os.environ, ZW_WAV=wav, ZW_TEXTFILE=txt, ZW_VOICE=str(self.voice or ""),
                       ZW_RATE=str(int(max(-10, min(10, round((self.rate - 1.0) * 10))))), ZW_VOLUME=str(int(max(0, min(100, 100 * self.volume)))))
            r = _run([ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", _SAPI_SCRIPT], env=env)
            if r.returncode != 0 or not os.path.getsize(wav):
                raise TTSError("SAPI failed: " + r.stderr.decode("utf-8", "replace")[-300:])
            return _read_wav(wav)
        finally:
            _rm(wav)
            _rm(txt)


class PiperTTS(Engine):
    name = "piper"

    def _exe(self):
        return shutil.which("piper") or shutil.which("piper-tts")

    def available(self):
        if not self._exe():
            return False, "the `piper` program is not installed (https://github.com/rhasspy/piper)"
        m = self.cfg.get("voice.tts.model")
        if not m or not os.path.isfile(self.cfg.resolve_path(str(m))):
            return False, "set voice.tts.model to a Piper voice (.onnx file)"
        return True, ""

    def synth(self, text):
        exe = self._exe()
        model = self.cfg.resolve_path(str(self.cfg.get("voice.tts.model")))
        wav = _tmp_wav()
        try:
            r = _run([exe, "--model", model, "--output_file", wav, "--length_scale", f"{1.0 / max(0.2, self.rate):.3f}"], input=text.encode("utf-8"))
            if r.returncode != 0:
                raise TTSError("piper failed: " + r.stderr.decode("utf-8", "replace")[-300:])
            x, sr = _read_wav(wav)
            return x * min(1.0, self.volume), sr
        finally:
            _rm(wav)


class Pyttsx3TTS(Engine):
    name = "pyttsx3"

    def available(self):
        try:
            import pyttsx3  # noqa: F401
            return True, ""
        except Exception as e:                                             # noqa: BLE001
            return False, f"pyttsx3 is not installed ({type(e).__name__}); pip install pyttsx3"

    def synth(self, text):
        import pyttsx3
        wav = _tmp_wav()
        try:
            eng = pyttsx3.init()
            eng.setProperty("rate", int(200 * self.rate))
            eng.setProperty("volume", max(0.0, min(1.0, self.volume)))
            if self.voice:
                eng.setProperty("voice", str(self.voice))
            eng.save_to_file(text, wav)
            eng.runAndWait()
            if not os.path.exists(wav) or not os.path.getsize(wav):
                raise TTSError("pyttsx3 produced no audio")
            return _read_wav(wav)
        finally:
            _rm(wav)


class CommandTTS(Engine):
    name = "command"

    def available(self):
        return (True, "") if self.cfg.get("voice.tts.command") else (False, "set voice.tts.command (a template with {text} and {wav})")

    def synth(self, text):
        tpl = str(self.cfg.get("voice.tts.command"))
        wav = _tmp_wav()
        try:
            cmd = tpl.replace("{wav}", shlex.quote(wav) if os.name != "nt" else f'"{wav}"').replace("{text}", shlex.quote(text) if os.name != "nt" else '"' + text.replace('"', '\\"') + '"')
            r = _run(cmd, shell=True)
            if r.returncode != 0:
                raise TTSError("tts command failed: " + r.stderr.decode("utf-8", "replace")[-300:])
            return _read_wav(wav)
        finally:
            _rm(wav)


def _rm(p: str):
    try:
        os.remove(p)
    except OSError:
        pass


ENGINES = {c.name: c for c in (BuiltinTTS, EspeakTTS, SayTTS, SapiTTS, PiperTTS, Pyttsx3TTS, CommandTTS)}


def auto_order() -> List[str]:
    if os.name == "nt":
        return ["windows_sapi", "piper", "pyttsx3", "builtin"]
    if sys.platform == "darwin":
        return ["say", "piper", "espeak", "pyttsx3", "builtin"]
    return ["piper", "espeak", "pyttsx3", "builtin"]


def report(cfg, profile_path: Optional[str] = None) -> dict:
    """{engine: (available, reason)} for every engine."""
    out = {}
    for n, c in ENGINES.items():
        e = c(cfg)
        if profile_path and isinstance(e, BuiltinTTS):
            e.profile_path = profile_path
        out[n] = e.available()
    return out


def make(cfg, profile_path: Optional[str] = None, name: Optional[str] = None) -> Engine:
    """The engine named by voice.tts.engine (or `name`); `auto` takes the first available in auto_order()."""
    want = name or cfg.get("voice.tts.engine")
    if want != "auto":
        if want not in ENGINES:
            raise TTSError(f"unknown tts engine '{want}'; choose from {sorted(ENGINES)}")
        e = ENGINES[want](cfg)
        ok, why = e.available()
        if not ok:
            raise TTSError(f"tts engine '{want}' is not usable: {why}")
    else:
        e = None
        for n in auto_order():
            cand = ENGINES[n](cfg)
            if cand.available()[0]:
                e = cand
                break
        if e is None:                                                      # builtin is always available; this is unreachable
            e = BuiltinTTS(cfg)
    if profile_path and isinstance(e, BuiltinTTS):
        e.profile_path = profile_path
    return e


def split_sentences(text: str, limit: int = 300) -> List[str]:
    """Pieces short enough for engines that stumble on long input; keeps sentence boundaries."""
    import re
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
    out: List[str] = []
    for p in parts:
        while len(p) > limit:
            cut = p.rfind(" ", 0, limit)
            cut = cut if cut > 40 else limit
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


def cache_key(engine: Engine, text: str) -> str:
    c = engine.cfg.get
    sig = json.dumps([engine.name, c("voice.tts.voice"), c("voice.tts.rate"), c("voice.tts.pitch"), c("voice.tts.volume"),
                      c("voice.tts.model"), c("voice.tts.builtin_f0_hz", None), c("voice.tts.builtin_tempo", None), text], sort_keys=True, default=str)
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:32]
