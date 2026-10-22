"""audio.py - read, write, resample and convert audio with numpy and the standard library.

* WAV (PCM 8/16/24/32-bit, float 32/64, WAVE_FORMAT_EXTENSIBLE) is parsed here, with no dependency.
* Anything else (mp3, flac, ogg, opus, m4a, ...) is decoded by ffmpeg when it is installed (`voice.convert.ffmpeg` or PATH).
* Sample-rate conversion is a windowed-sinc resampler (band-limited, so down-sampling does not alias).

Samples are float32 in [-1, 1]; mono is shape (n,), multi-channel (n, channels).
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
from typing import Optional, Tuple, Union

import numpy as np
from .. import fileio

AUDIO_EXTENSIONS = (".wav", ".wave", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".aif", ".webm", ".mp4")


class AudioError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------------------------
# WAV
# ---------------------------------------------------------------------------------------------------------------------
def is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] in (b"RIFF", b"RF64") and data[8:12] == b"WAVE"


def parse_wav(data: bytes) -> Tuple[np.ndarray, int]:
    """(samples float32 (n,) or (n, ch), sample_rate) from the bytes of a WAV file."""
    if not is_wav(data):
        raise AudioError("not a RIFF/WAVE file")
    pos, fmt, pcm = 12, None, None
    n = len(data)
    while pos + 8 <= n:
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            if size < 16:
                raise AudioError("damaged WAV header")
            tag, ch, sr, _, _, bits = struct.unpack("<HHIIHH", data[body:body + 16])
            if tag == 0xFFFE and size >= 26:                           # WAVE_FORMAT_EXTENSIBLE: real tag is the first 2 bytes of the GUID
                tag = struct.unpack("<H", data[body + 24:body + 26])[0]
            fmt = (tag, ch, sr, bits)
        elif cid == b"data":
            end = n if (size == 0xFFFFFFFF or body + size > n) else body + size      # streamed / truncated files
            pcm = data[body:end]
            break
        pos = body + size + (size & 1)
    if fmt is None or pcm is None:
        raise AudioError("WAV file without a fmt or data chunk")
    tag, ch, sr, bits = fmt
    if ch < 1 or sr < 1:
        raise AudioError("damaged WAV header")
    if tag == 1:
        if bits == 8:
            x = (np.frombuffer(pcm, np.uint8).astype(np.float32) - 128.0) / 128.0
        elif bits == 16:
            x = np.frombuffer(pcm[: len(pcm) // 2 * 2], "<i2").astype(np.float32) / 32768.0
        elif bits == 24:
            b = np.frombuffer(pcm[: len(pcm) // 3 * 3], np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            v = np.where(v & 0x800000, v - 0x1000000, v)
            x = v.astype(np.float32) / 8388608.0
        elif bits == 32:
            x = np.frombuffer(pcm[: len(pcm) // 4 * 4], "<i4").astype(np.float32) / 2147483648.0
        else:
            raise AudioError(f"unsupported PCM width: {bits} bits")
    elif tag == 3:
        if bits == 32:
            x = np.frombuffer(pcm[: len(pcm) // 4 * 4], "<f4").astype(np.float32)
        elif bits == 64:
            x = np.frombuffer(pcm[: len(pcm) // 8 * 8], "<f8").astype(np.float32)
        else:
            raise AudioError(f"unsupported float width: {bits} bits")
    else:
        raise AudioError(f"unsupported WAV encoding (format tag {tag}); convert it with ffmpeg or `zerowhite chat voice convert`")
    x = x[: len(x) // ch * ch]
    return (x.reshape(-1, ch) if ch > 1 else x), int(sr)


def encode_wav(x: np.ndarray, sr: int, bits: int = 16) -> bytes:
    """Bytes of a WAV file. bits: 16 (default), 24, 32 (integer PCM) or 'f32'-style 33 for float."""
    x = np.asarray(x, dtype=np.float32)
    ch = 1 if x.ndim == 1 else x.shape[1]
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    if bits == 16:
        pcm = (np.clip(x, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()
        tag, width = 1, 16
    elif bits == 24:
        v = (np.clip(x, -1.0, 1.0) * 8388607.0).round().astype(np.int32).reshape(-1)
        pcm = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=1).astype(np.uint8).tobytes()
        tag, width = 1, 24
    elif bits == 32:
        pcm = (np.clip(x, -1.0, 1.0) * 2147483647.0).round().astype("<i4").tobytes()
        tag, width = 1, 32
    elif bits == 33:
        pcm = x.astype("<f4").tobytes()
        tag, width = 3, 32
    else:
        raise AudioError("bits must be 16, 24, 32 (integer) or 33 (float32)")
    block = ch * width // 8
    hdr = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE" + b"fmt " + struct.pack("<IHHIIHH", 16, tag, ch, int(sr), int(sr) * block, block, width)
           + b"data" + struct.pack("<I", len(pcm)))
    return hdr + pcm


# ---------------------------------------------------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------------------------------------------------
def find_ffmpeg(cfg=None) -> Optional[str]:
    p = cfg.get("voice.convert.ffmpeg", None) if cfg is not None else None
    if p:
        p = os.path.expanduser(str(p))
        return p if os.path.isfile(p) else shutil.which(p)
    return shutil.which("ffmpeg")


def _ffmpeg_decode(src: Union[str, bytes], sr: int, mono: bool, cfg=None) -> Tuple[np.ndarray, int]:
    exe = find_ffmpeg(cfg)
    if not exe:
        raise AudioError("this audio format needs ffmpeg (https://ffmpeg.org): install it, or set voice.convert.ffmpeg to its path. "
                         "WAV files need nothing.")
    cmd = [exe, "-v", "error", "-nostdin", "-i", src if isinstance(src, str) else "pipe:0", "-f", "f32le", "-acodec", "pcm_f32le"]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-ar", str(int(sr)), "pipe:1"]
    try:
        p = subprocess.run(cmd, input=None if isinstance(src, str) else src, capture_output=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise AudioError(f"ffmpeg failed: {e}") from None
    if p.returncode != 0 or not p.stdout:
        raise AudioError("ffmpeg could not decode the audio: " + p.stderr.decode("utf-8", "replace").strip()[-300:])
    x = np.frombuffer(p.stdout[: len(p.stdout) // 4 * 4], "<f4").astype(np.float32)
    return x, int(sr)


def _probe_rate(path: str, cfg=None) -> Optional[int]:
    exe = find_ffmpeg(cfg)
    if not exe:
        return None
    probe = shutil.which("ffprobe") or os.path.join(os.path.dirname(exe), "ffprobe")
    try:
        p = subprocess.run([probe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate", "-of", "csv=p=0", path],
                           capture_output=True, text=True, timeout=60)
        return int(p.stdout.strip().split()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------------------------------------------------
def to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1).astype(np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int, zeros: int = 24, chunk: int = 1 << 16) -> np.ndarray:
    """Band-limited resampling by windowed-sinc interpolation. Accurate to ~-80 dB for in-band content."""
    x = np.asarray(x, dtype=np.float32)
    if sr_in == sr_out or len(x) == 0:
        return x.copy()
    if x.ndim == 2:
        return np.stack([resample(x[:, c], sr_in, sr_out, zeros, chunk) for c in range(x.shape[1])], axis=1)
    ratio = sr_out / sr_in
    cutoff = min(1.0, ratio)                                          # relative to the input Nyquist: filter below the lower of the two rates
    half = int(np.ceil(zeros / cutoff))
    n_out = int(round(len(x) * ratio))
    pad = np.concatenate([np.zeros(half, np.float32), x, np.zeros(half + 1, np.float32)])
    taps = np.arange(-half + 1, half + 1)                             # 2*half taps around each position
    out = np.empty(n_out, np.float32)
    for s in range(0, n_out, chunk):
        idx = np.arange(s, min(n_out, s + chunk))
        pos = idx / ratio                                             # position in input samples
        base = np.floor(pos).astype(np.int64)
        frac = (pos - base)[:, None]
        t = taps[None, :] - frac                                      # distance of each tap from the exact position
        w = cutoff * np.sinc(cutoff * t)
        w *= 0.5 * (1.0 + np.cos(np.pi * np.clip(t / half, -1.0, 1.0)))     # Hann window over the support
        gather = pad[(base[:, None] + taps[None, :] + half)]
        out[idx] = (gather * w).sum(axis=1)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# front doors
# ---------------------------------------------------------------------------------------------------------------------
def load_audio(src: Union[str, bytes, os.PathLike], sr: Optional[int] = None, mono: bool = True, cfg=None) -> Tuple[np.ndarray, int]:
    """Decode a file (path) or the bytes of one. Returns (samples, sample_rate); resampled to `sr` when given."""
    data = None
    path = None
    if isinstance(src, (bytes, bytearray, memoryview)):
        data = bytes(src)
    else:
        path = os.fspath(src)
        if not os.path.isfile(path):
            raise AudioError(f"audio file not found: {path}")
        with open(path, "rb") as f:
            head = f.read(12)
        if is_wav(head):
            with open(path, "rb") as f:
                data = f.read()
    if data is not None and is_wav(data):
        x, rate = parse_wav(data)
        if mono:
            x = to_mono(x)
    else:
        target = sr or (_probe_rate(path, cfg) if path else None) or 16000
        x, rate = _ffmpeg_decode(path if path else data, target, mono, cfg)
    if sr and rate != sr:
        x, rate = resample(x, rate, sr), sr
    return x.astype(np.float32, copy=False), rate


def save_audio(path: str, x: np.ndarray, sr: int, bits: int = 16, cfg=None):
    """Write WAV directly; any other extension goes through ffmpeg."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    wav = encode_wav(x, sr, bits)
    if ext in ("", ".wav", ".wave"):
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(wav)
        fileio.replace(tmp, path)
        return
    exe = find_ffmpeg(cfg)
    if not exe:
        raise AudioError(f"writing {ext} needs ffmpeg; write a .wav instead or install ffmpeg")
    p = subprocess.run([exe, "-v", "error", "-y", "-i", "pipe:0", path], input=wav, capture_output=True, timeout=600)
    if p.returncode != 0:
        raise AudioError("ffmpeg could not encode the audio: " + p.stderr.decode("utf-8", "replace").strip()[-300:])


def convert_file(src: str, dst: str, sr: Optional[int] = 16000, mono: bool = True, bits: int = 16, normalize: Optional[float] = None, cfg=None) -> dict:
    """Convert one file. normalize = target peak (e.g. 0.95) or None. Returns {seconds, sr_in, sr_out}."""
    x, rate_in = _load_native(src, mono, cfg)
    rate_out = sr or rate_in
    if rate_out != rate_in:
        x = resample(x, rate_in, rate_out)
    if normalize:
        peak = float(np.abs(x).max()) or 1.0
        x = x * (normalize / peak)
    save_audio(dst, x, rate_out, bits, cfg)
    return dict(seconds=len(x) / rate_out, sr_in=rate_in, sr_out=rate_out)


def _load_native(src: str, mono: bool, cfg=None) -> Tuple[np.ndarray, int]:
    """Decode without resampling (keeps the file's own rate)."""
    with open(src, "rb") as f:
        head = f.read(12)
    if is_wav(head):
        with open(src, "rb") as f:
            x, rate = parse_wav(f.read())
        return (to_mono(x) if mono else x), rate
    rate = _probe_rate(src, cfg) or 44100
    return _ffmpeg_decode(src, rate, mono, cfg)


def duration_s(path: str, cfg=None) -> Optional[float]:
    """Length of an audio file in seconds without decoding it fully when it is a WAV."""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
        if is_wav(head):
            pos = 12
            fmt = None
            while pos + 8 <= len(head):
                cid, size = head[pos:pos + 4], struct.unpack("<I", head[pos + 4:pos + 8])[0]
                if cid == b"fmt ":
                    _, ch, sr, byterate, block, bits = struct.unpack("<HHIIHH", head[pos + 8:pos + 24])
                    fmt = (byterate or sr * block)
                elif cid == b"data" and fmt:
                    n = os.path.getsize(path) - (pos + 8) if size in (0, 0xFFFFFFFF) else size
                    return n / fmt
                pos += 8 + size + (size & 1)
        x, rate = load_audio(path, None, True, cfg)
        return len(x) / rate
    except (OSError, AudioError, struct.error, ZeroDivisionError):
        return None


def peak_normalize(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.abs(x).max()) if len(x) else 0.0
    return x if m < 1e-9 else (x * (peak / m)).astype(np.float32)
