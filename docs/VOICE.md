# Voice: speech in, speech out

```
microphone -> voice-activity detection -> speech recogniser -> chat model -> speech synthesiser -> speaker
```

```bash
zerowhite chat talk --voice -w my-bot             # hands-free conversation (say "stop listening" to end it)
zerowhite chat voice status  -w my-bot            # which engines are in use, which are available and why not
zerowhite chat voice say "Hello there" --out hello.wav
zerowhite chat voice listen recording.wav         # transcribe a file (no file: one utterance from the microphone)
zerowhite chat voice ask question.wav --out answer.wav     # spoken question -> chat model -> spoken answer
zerowhite chat voice test                         # speak a phrase and recognise it again: checks the whole plumbing
zerowhite chat voice devices                      # microphones, speakers, and which backends were found
```

The web UI of `chat start` has a microphone button and a "speak replies" switch (it records in the browser, sends WAV to `/api/stt`, and plays what `/api/tts`
returns). The same endpoints are documented in [CHAT.md](CHAT.md).

Everything works with the standard library + NumPy. Nothing is *required* beyond that; better engines are plug-ins you can add.

## Speech out (`voice.tts.*`)

`voice.tts.engine = auto` takes the first engine that is available, in this order (the order depends on your OS):

| OS | order |
|---|---|
| Windows | `windows_sapi` (built-in system voices), `piper`, `pyttsx3`, **`builtin`** |
| macOS | `say`, `piper`, `espeak`, `pyttsx3`, **`builtin`** |
| Linux | `piper`, `espeak` / `espeak-ng`, `pyttsx3`, **`builtin`** |

Plus `command`: your own program, `voice.tts.command = "mytts --text {text} --out {wav}"`.
`voice.tts.voice`, `.rate`, `.pitch`, `.volume`, `.model` (for piper) apply where the engine supports them. Phrases are cached on disk (`voice.tts.cache`).
Long answers are spoken sentence by sentence, so speech starts before the whole reply is synthesised.

**The built-in voice** is a formant synthesiser: text -> rule-based letters-to-phonemes (numbers, punctuation and unknown words handled) -> a source-filter model with
vowel/consonant formants, voicing, aspiration and prosody. It is deterministic, needs nothing installed, and is **intelligible but robotic**. It is also what the built-in recogniser
is tested against. `chat voice profile` can measure the pitch and pace of a speech dataset and set the built-in voice to match (`voice.tts.builtin_f0_hz`, `voice.tts.builtin_tempo`).
For a natural voice install `piper` (set `voice.tts.model`) or use your OS voices.

## Speech in (`voice.stt.*`)

`voice.stt.engine = auto`: `faster_whisper`, `vosk`, `whisper_cpp`, `windows_sapi`, then **`builtin`**; or `command` (`voice.stt.command = "mystt {wav}"`, transcript on stdout).

> Whisper / Vosk / whisper.cpp are *pre-trained* neural recognisers from other projects (speech recognisers, not language models). Zero White never installs them and never calls a
> service; if you have one installed, `auto` will use it because it is far more accurate than anything that can be trained on a laptop. If you want a fully self-contained system,
> set `voice.stt.engine = builtin`.

### The built-in recogniser: train it on your own audio

A small dilated-convolution network with CTC loss (log-mel features -> strided conv -> residual dilated convs -> characters), with forward and backward passes written in NumPy.
It learns the alphabet from your transcripts, so any language written with a character set works. Decoding is greedy.

1. Put speech data in `data/audio/` (`voice.audio_folder`). Layouts understood without configuration:
   * **LJ Speech**: `metadata.csv` with `id|text|normalised text` and `wavs/id.wav`
   * **Common Voice**: `validated.tsv` (`path<TAB>sentence`) and `clips/*.mp3`
   * **VCTK**: `wav48/p225/p225_001.flac` with `txt/p225/p225_001.txt`
   * any `clip.wav` + `clip.txt`, or any csv/tsv/jsonl with an audio-path column and a text column (`dataset.json` can name them)
2. `zerowhite chat voice data` shows which (audio, transcript) pairs were found and what could not be matched.
3. `zerowhite chat voice train-stt --minutes 60` trains. Network size follows the amount of audio (about 100k parameters for minutes of data, a few million for tens of hours),
   features are cached after the first pass, training uses speed/silence/mask augmentation, a stable-hash validation split, a wall-clock budget and early stopping, and **continues from
   the saved model** on the next run. mp3/flac/ogg/m4a need `ffmpeg` (`voice.convert.ffmpeg`); WAV is handled natively.

Models and settings: `voice.builtin_stt.*` ([CONFIG.md](CONFIG.md)).

**What accuracy to expect — measured, and what is not.** All numbers are from the build sandbox, on synthetic speech made by the built-in voice, because it had no real recordings:

| experiment | result |
|---|---|
| 10 command words, 100 clips, 40 s of training, tested on a pitch/tempo it had not heard | 7/10 exactly right, character error about 11 % |
| 700 synthetic sentences (about 35 min), minutes of training | character error about 23 %, word error about 47 % on held-out sentences, no further gain from longer training |

So: it is good at **small vocabularies** (commands, names, digits) and, as is true of every recogniser, needs **hours** of varied speech and hours of CPU time to do open-vocabulary dictation.
**It has not been trained or measured on real human recordings** (LJ Speech, VCTK, Common Voice were not available to the build), so its accuracy on your voice is unknown until you train it on yours.
`voice test` and the loopback tests check the pipeline, not accuracy on people.

## Voice-activity detection

Self-calibrating: it measures your room's noise for `voice.vad.calibration_s`, sets the speech threshold `voice.vad.margin_db` above it, and keeps tracking it. Utterances end after
`voice.vad.end_silence_ms` of quiet and keep `voice.vad.pre_roll_ms` before the first word. Set `voice.vad.threshold_db` to a number to fix the threshold.

## Converting audio

```bash
zerowhite chat voice convert recordings/ converted/ --rate 16000 --bits 16          # whole folder, same layout, mono 16 kHz WAV
zerowhite chat voice convert song.mp3 speech.wav --normalize 0.95
```

Rate conversion uses a windowed-sinc resampler (alias-free; tested against known tones). Originals are never modified. Dataset *text* conversion is in [DATA.md](DATA.md).

## Devices

Playback: `sounddevice` if installed, else `winsound` (Windows), `afplay` (macOS), `paplay` / `aplay` / `ffplay` (Linux).
Capture: `sounddevice` if installed (`pip install sounddevice`; needs PortAudio), else `ffmpeg` (ALSA/Pulse, AVFoundation, DirectShow).
Choose with `voice.input_device` / `voice.output_device` (a name fragment or an index; `voice devices` lists them).

## A learned audio codec (6.4, `voice/codec.py`, `zerowhite codec train|encode|decode`)

A small neural codec in plain numpy: 320-sample sqrt-Hann frames (hop 160, perfect reconstruction without coding), an MLP autoencoder to a 48-number latent, then **residual vector quantisation** (6 stages x 128 codewords: a frame becomes 6 integers, 4.2 kbps; any prefix of the stages is a
valid coarser code, 0.7 kbps per stage; the decoder is fine-tuned with quantiser dropout so a prefix decodes well: without it prefixes were 0.1-0.25 dB worse on a test-sized codec in two seeds (an earlier, smaller test codec was not even monotone in stages before I added it)). `tokens_to_text` / `text_to_tokens` write the codes as text for the chat corpus.

Measured on **held-out synthetic speech** (trained on 32 clips, ~97 s, 78 s of CPU): SNR 1.6 / 3.3 / 3.8 dB at 1 / 3 / 6 stages against 1.1 / 2.0 / 2.5 dB for a plain PCA transform with the same quantiser and bitrate.
A small test-sized codec (about 10 s to train) does NOT beat PCA clearly; the claim holds for the properly trained one (`ZW_SLOW_TESTS=1` enforces it, about 80 s). **These are low absolute numbers**: waveform SNR at a few kbps is hard, and the figures say the learned transform helps, not that the codec is good.

**What this does not show.** It was trained and tested only on the project's own formant synthesiser: it has never seen a real voice, room noise or music, so nothing here addresses the recogniser's real-speech weakness; "unified audio tokens in the chat backbone" is the text serialisation only (`chat/modal.py`), with no model trained on it.

## What has and has not been exercised

Tested (automatically, on Linux): WAV parsing/encoding at all bit depths, the resampler, VAD (offline and streaming), letters-to-phonemes, the synthesiser's pitch/tempo,
CTC loss against brute-force enumeration of alignments and its gradient by finite differences, the acoustic model's backward pass, speech-dataset discovery in all the layouts above, recogniser
training on a real (synthetic-audio) folder and decoding a voice it did not hear, the voice service, `/api/stt` `/api/tts` with token checks, folder conversion.

**Not exercised: microphone capture, speaker playback, the browser microphone button, Windows SAPI, macOS `say`, piper, Whisper, Vosk** — the build environment had no audio hardware and none of those installed.
Their code follows the documented interfaces and fails with an actionable message (`voice devices`, `voice status`), but treat the first run on your machine as a test.
