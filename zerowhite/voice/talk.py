"""talk.py - the spoken conversation: listen, think, answer aloud.

    zerowhite chat talk --voice

Listening uses the self-calibrating VAD (vad.py), recognition and synthesis come from VoiceService, the answers from the chat model
(through the background service when it is running, otherwise the model directly). While the assistant speaks the microphone is
closed, so it never hears itself. Say one of voice.talk.stop_phrases (or press Ctrl-C) to end.
"""
from __future__ import annotations

import re
import time
from typing import Iterator, Optional

import numpy as np

from ..chat.cli_common import fail
from .devices import DeviceError, Microphone, Speaker
from .service import VoiceService
from .vad import Segmenter

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def listen_once(cfg, say=print, timeout_s: Optional[float] = None) -> Optional[np.ndarray]:
    """Record one utterance from the microphone (None on timeout)."""
    with Microphone(cfg) as mic:
        seg = Segmenter(cfg, mic.sr)
        t0 = time.time()
        announced = False
        while True:
            if timeout_s and time.time() - t0 > timeout_s:
                return None
            for utt in seg.feed(mic.read()):
                return utt
            if seg.calibrated and not announced:
                say("(listening)")
                announced = True


def sentences(deltas: Iterator[str]) -> Iterator[str]:
    """Group streamed text pieces into whole sentences (the tail is flushed at the end)."""
    buf = ""
    for d in deltas:
        buf += d
        parts = _SENTENCE_END.split(buf)
        for p in parts[:-1]:
            if p.strip():
                yield p.strip()
        buf = parts[-1]
    if buf.strip():
        yield buf.strip()


def run_talk(cfg, lay, args) -> int:
    from ..chat.cli import _gen_opts, _open_chat
    voice = VoiceService(cfg, lay)
    try:
        voice.stt_engine()
    except Exception as e:                                                  # noqa: BLE001
        fail(f"cannot listen: {e}")
    speak = bool(cfg.get("voice.talk.speak")) and not getattr(args, "no_speak", False)
    speaker = Speaker(cfg)
    if speak:
        try:
            voice.tts_engine()
        except Exception as e:                                              # noqa: BLE001
            print(f"(answers will be text only: {e})")
            speak = False
    if speak and speaker.backend() is None:
        print("(answers will be text only: no audio playback backend; see `zerowhite chat voice devices`)")
        speak = False
    try:
        mic_probe = Microphone(cfg).backend()
    except DeviceError as e:
        fail(str(e))
    if mic_probe is None:
        fail("no way to record audio: pip install sounddevice (needs PortAudio) or install ffmpeg. `zerowhite chat voice devices` shows what was found.")
    chat = _open_chat(args, cfg, lay)
    name = cfg.get("persona.name") or "Zero White"
    stops = [s.lower() for s in cfg.get("voice.talk.stop_phrases")]
    timeout = cfg.get("voice.talk.listen_timeout_s", None)
    opts = _gen_opts(args)
    print(f"{name} ({chat.where}) - speech: listen={voice.stt_engine().name}, speak={voice.tts_engine().name if speak else 'off'}. "
          f"Say \"{stops[0]}\" or press Ctrl-C to stop.", flush=True)
    while True:
        try:
            if cfg.get("voice.talk.beep") and speak:
                speaker.beep()
            utt = listen_once(cfg, print, float(timeout) if timeout else None)
            if utt is None:
                print("(no speech heard; stopping)")
                return 0
            text = voice.transcribe_array(utt, int(cfg.get("voice.sample_rate")))
            if not text:
                continue
            print(f"you: {text}", flush=True)
            if any(p and p in text.lower() for p in stops):
                print("(ok, stopping)")
                return 0
            print(f"{name}: ", end="", flush=True)
            deltas = (ev["delta"] for ev in chat.send(text, **opts) if "delta" in ev)
            for sent in sentences(deltas):
                print(sent + " ", end="", flush=True)
                if speak:
                    x, sr = voice.synthesize_array(sent)
                    speaker.play(x, sr)
            print()
        except KeyboardInterrupt:
            speaker.stop()
            print("\n(stopped)")
            return 0
        except DeviceError as e:
            fail(str(e))
        except Exception as e:                                              # noqa: BLE001
            print(f"\n({type(e).__name__}: {e})")
            time.sleep(0.5)
