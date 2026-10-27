"""devices.py - microphone capture and speaker playback with whatever the machine offers.

Playback:  sounddevice (if installed) -> winsound (Windows, standard library) -> afplay (macOS) -> paplay / aplay / ffplay (Linux)
Capture:   sounddevice (if installed) -> ffmpeg (alsa / pulse / avfoundation / dshow)
Select a device with voice.input_device / voice.output_device (a name fragment or an index).
These code paths need real audio hardware; they could not be exercised in the cloud sandbox this was written in
(only the selection logic and the error paths are covered by tests). `voice devices` shows what was found.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import List, Optional

import numpy as np

from . import audio as A


class DeviceError(RuntimeError):
    pass


def _sd():
    try:
        import sounddevice as sd
        sd.query_devices()                                                 # raises when PortAudio is missing
        return sd
    except Exception:                                                      # noqa: BLE001
        return None


def _pick_device(sd, ref, kind: str):
    """Index of the device matching `ref` (name fragment or number); None = system default."""
    if ref in (None, "", "default"):
        return None
    s = str(ref)
    if s.isdigit():
        return int(s)
    for i, d in enumerate(sd.query_devices()):
        if s.lower() in d["name"].lower() and d["max_" + kind + "_channels"] > 0:
            return i
    raise DeviceError(f"no {kind} device matching '{ref}'. `zerowhite chat voice devices` lists them.")


# ---------------------------------------------------------------------------------------------------------------------
# playback
# ---------------------------------------------------------------------------------------------------------------------
class Speaker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ref = cfg.get("voice.output_device")
        self._proc: Optional[subprocess.Popen] = None
        self._stop = threading.Event()

    def backend(self) -> Optional[str]:
        if _sd():
            return "sounddevice"
        if os.name == "nt":
            return "winsound"
        if sys.platform == "darwin" and shutil.which("afplay"):
            return "afplay"
        for n in ("paplay", "aplay", "ffplay"):
            if shutil.which(n):
                return n
        return None

    def play(self, x: np.ndarray, sr: int, block: bool = True):
        if len(x) == 0:
            return
        be = self.backend()
        if be is None:
            raise DeviceError("no way to play audio: install `sounddevice` (pip install sounddevice; needs PortAudio) "
                              "or, on Linux, paplay / aplay / ffplay")
        self._stop.clear()
        if be == "sounddevice":
            sd = _sd()
            sd.play(x.astype(np.float32), sr, device=_pick_device(sd, self.ref, "output"))
            if block:
                sd.wait()
            return
        wav = A.encode_wav(x, sr)
        if be == "winsound":
            import winsound
            winsound.PlaySound(wav, winsound.SND_MEMORY | (0 if block else winsound.SND_ASYNC))
            return
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="zw_play_")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(wav)
        cmd = {"afplay": ["afplay", path], "paplay": ["paplay", path], "aplay": ["aplay", "-q", path],
               "ffplay": ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]}[be]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if block:
            try:
                self._proc.wait()
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def stop(self):
        self._stop.set()
        sd = _sd()
        if sd:
            try:
                sd.stop()
            except Exception:                                              # noqa: BLE001
                pass
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def beep(self, sr: int = 16000):
        t = np.arange(int(0.09 * sr)) / sr
        env = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)
        try:
            self.play((0.25 * np.sin(2 * np.pi * 880 * t) * env).astype(np.float32), sr)
        except DeviceError:
            pass


# ---------------------------------------------------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------------------------------------------------
def _ffmpeg_input_args(cfg, exe: str) -> List[str]:
    ref = cfg.get("voice.input_device")
    if sys.platform.startswith("linux"):
        fmt = "pulse" if shutil.which("pactl") else "alsa"
        return ["-f", fmt, "-i", str(ref) if ref not in (None, "") else "default"]
    if sys.platform == "darwin":
        return ["-f", "avfoundation", "-i", ":" + (str(ref) if ref not in (None, "") else "0")]
    name = ref if ref not in (None, "") else (_dshow_devices(exe) or [None])[0]
    if not name:
        raise DeviceError("no microphone found through ffmpeg/dshow; set voice.input_device to its name (`voice devices` lists them)")
    return ["-f", "dshow", "-i", f"audio={name}"]


def _dshow_devices(exe: str) -> List[str]:
    try:
        r = subprocess.run([exe, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    text = r.stderr.decode("utf-8", "replace")
    return re.findall(r'"([^"]+)"\s+\(audio\)', text)


class Microphone:
    """Context manager yielding fixed-size chunks of mono float32 audio at `sr`."""

    def __init__(self, cfg, sr: Optional[int] = None, chunk_ms: float = 100.0):
        self.cfg = cfg
        self.sr = int(sr or cfg.get("voice.sample_rate"))
        self.n = max(1, int(self.sr * chunk_ms / 1000.0))
        self._stream = None
        self._proc: Optional[subprocess.Popen] = None

    def backend(self) -> Optional[str]:
        if _sd():
            return "sounddevice"
        exe = A.find_ffmpeg(self.cfg)
        return "ffmpeg" if exe else None

    def __enter__(self) -> "Microphone":
        be = self.backend()
        if be == "sounddevice":
            sd = _sd()
            self._stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32", device=_pick_device(sd, self.cfg.get("voice.input_device"), "input"))
            self._stream.start()
        elif be == "ffmpeg":
            exe = A.find_ffmpeg(self.cfg)
            cmd = [exe, "-hide_banner", "-loglevel", "error"] + _ffmpeg_input_args(self.cfg, exe) + ["-ac", "1", "-ar", str(self.sr), "-f", "s16le", "pipe:1"]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        else:
            raise DeviceError("no way to record audio: install `sounddevice` (pip install sounddevice; needs PortAudio) or ffmpeg")
        return self

    def read(self) -> np.ndarray:
        if self._stream is not None:
            data, _ = self._stream.read(self.n)
            return data[:, 0].astype(np.float32)
        raw = self._proc.stdout.read(self.n * 2)
        if not raw:
            raise DeviceError("the microphone stream ended (is the device in use or unplugged?)")
        return np.frombuffer(raw[: len(raw) // 2 * 2], "<i2").astype(np.float32) / 32768.0

    def __exit__(self, *exc):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:                                              # noqa: BLE001
                pass
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def list_devices(cfg) -> str:
    lines = []
    sd = _sd()
    if sd:
        lines.append("sounddevice (PortAudio) devices:")
        for i, d in enumerate(sd.query_devices()):
            kinds = ("in" if d["max_input_channels"] else "") + ("/out" if d["max_output_channels"] else "")
            lines.append(f"  [{i}] {d['name']}  ({kinds.strip('/')})")
    else:
        lines.append("sounddevice: not available (pip install sounddevice; it needs the PortAudio library)")
    exe = A.find_ffmpeg(cfg)
    lines.append(f"ffmpeg: {exe or 'not found'}")
    if exe and os.name == "nt":
        names = _dshow_devices(exe)
        lines += ["  dshow microphones: " + (", ".join(names) if names else "none found")]
    spk = Speaker(cfg).backend()
    lines.append(f"playback backend: {spk or 'none'}")
    lines.append(f"capture backend: {Microphone(cfg).backend() or 'none'}")
    return "\n".join(lines)
