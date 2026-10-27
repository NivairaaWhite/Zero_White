"""zerowhite.voice - speech in and speech out.

    audio.py       WAV parsing/writing, resampling, conversion (ffmpeg for other formats)
    features.py    log-mel, frame energy, pitch
    vad.py         voice activity detection that calibrates itself to the room
    g2p.py         English text -> phonemes
    formant.py     the built-in speech synthesizer (numpy)
    tts.py         synthesis engines: built-in, SAPI, say, piper, espeak, pyttsx3, command
    ctc.py         the built-in recogniser: dilated conv + CTC with hand-written backward pass
    datasets.py    find (audio, transcript) pairs in LJ Speech / VCTK / Common Voice / your own folders
    train_stt.py   train the built-in recogniser
    stt.py         recognition engines: built-in, faster-whisper, vosk, whisper.cpp, SAPI, command
    devices.py     microphone capture and speaker playback
    service.py     VoiceService: what the web UI / daemon / CLI call
    talk.py        the spoken conversation loop
    cli.py         `zerowhite chat voice ...`
"""
from .service import VoiceService, clean_for_speech  # noqa: F401
