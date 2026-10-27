"""cli.py - `zerowhite chat voice ...`"""
from __future__ import annotations

import os
import sys
import time

from ..chat.cli_common import add_common, fail, load_cfg, load_layout
from . import audio as A


def _print_engines(rep: dict):
    for n, v in rep.items():
        print(f"    {n:<15} {'ok' if v['available'] else '--'}  {v['note']}")


def cmd_status(a):
    from .service import VoiceService
    cfg, lay = load_layout(a)
    d = VoiceService(cfg, lay).describe()
    print(f"speech out : {d['tts'] or 'NOT AVAILABLE'}" + (f"   ({d['tts_error']})" if d.get("tts_error") else ""))
    _print_engines(d["tts_engines"])
    print(f"speech in  : {d['stt'] or 'NOT AVAILABLE'}")
    if d.get("stt_error"):
        print("    " + d["stt_error"].replace("\n", "\n    "))
    _print_engines(d["stt_engines"])
    b = d.get("builtin_stt")
    if b:
        print(f"built-in recogniser: {b.get('params', 0):,} parameters, trained on {b.get('hours')} h, validation character error {b.get('val_cer')}")
    print(f"audio data folder: {lay.audio_data}")


def cmd_say(a):
    from .devices import DeviceError, Speaker
    from .service import VoiceService
    cfg, lay = load_layout(a)
    text = " ".join(a.text) if a.text else sys.stdin.read()
    if not text.strip():
        fail("say what? e.g.  zerowhite chat voice say \"hello\"")
    v = VoiceService(cfg, lay)
    try:
        x, sr = v.synthesize_array(text)
    except Exception as e:                                                  # noqa: BLE001
        fail(f"cannot speak: {e}")
    if a.out:
        A.save_audio(a.out, x, sr, cfg=cfg)
        print(f"wrote {a.out} ({len(x) / sr:.1f} s, {sr} Hz, engine {v.tts_engine().name})")
    if a.play or not a.out:
        try:
            Speaker(cfg).play(x, sr)
        except DeviceError as e:
            if not a.out:
                fail(f"{e}\nTip: add --out reply.wav to save the audio instead.")
            print(f"(not played: {e})")


def cmd_listen(a):
    from .service import VoiceService
    from .talk import listen_once
    cfg, lay = load_layout(a)
    v = VoiceService(cfg, lay)
    try:
        if a.file:
            print(v.transcribe(a.file))
            return
        from .devices import DeviceError
        try:
            x = listen_once(cfg, lambda m: print(m, file=sys.stderr), a.seconds)
        except DeviceError as e:
            fail(str(e))
        if x is None:
            fail("no speech heard")
        print(v.transcribe_array(x, int(cfg.get("voice.sample_rate"))))
    except Exception as e:                                                  # noqa: BLE001
        if isinstance(e, SystemExit):
            raise
        fail(f"cannot transcribe: {e}")


def cmd_ask(a):
    from ..chat.cli import _gen_opts, _open_chat
    from .service import VoiceService
    cfg, lay = load_layout(a)
    v = VoiceService(cfg, lay)
    try:
        heard = v.transcribe(a.file)
    except Exception as e:                                                  # noqa: BLE001
        fail(f"cannot transcribe: {e}")
    print(f"heard : {heard}")
    if not heard.strip():
        fail("nothing was recognised in that recording")
    chat = _open_chat(a, cfg, lay)
    reply = ""
    for ev in chat.send(heard, **_gen_opts(a)):
        if ev.get("done"):
            reply = ev.get("text", "")
    print(f"reply : {reply}")
    if a.out:
        x, sr = v.synthesize_array(reply)
        A.save_audio(a.out, x, sr, cfg=cfg)
        print(f"spoken reply written to {a.out}")


def cmd_train_stt(a):
    from .train_stt import TrainSTTError, train_stt
    cfg, lay = load_layout(a)
    if a.minutes is not None:
        cfg = cfg.replace(**{"voice.builtin_stt.budget_minutes": float(a.minutes)})
    if a.epochs is not None:
        cfg = cfg.replace(**{"voice.builtin_stt.epochs": int(a.epochs)})

    def show(ev):
        k = ev["event"]
        if k == "found":
            print(f"[stt] found {ev['pairs']:,} (audio, transcript) pairs; metadata: {', '.join(ev['tables']) or '-'}; sidecar transcripts: {ev['sidecars']}", flush=True)
        elif k == "features":
            print(f"[stt] features {ev['done']:,}/{ev['total']:,}", flush=True)
        elif k == "data":
            print(f"[stt] {ev['clips']:,} clips, {ev['hours']} h" + (f"; skipped {ev['skipped']}" if ev["skipped"] else ""), flush=True)
        elif k == "model":
            print(f"[stt] model: {ev['params']:,} parameters ({ev['hidden']} wide x {ev['layers']} layers, {ev['alphabet']} characters; {ev['reason']})"
                  + ("; continuing the existing model" if ev["resumed"] else ""), flush=True)
        elif k == "step":
            print(f"[stt] step {ev['step']}/{ev['planned']}  loss {ev['loss']:.3f}  lr {ev['lr']:.2e}", flush=True)
        elif k == "eval":
            extra = "" if ev["epoch"] == 0 else f"  train loss {ev.get('loss', 0):.3f}" + ("  * best" if ev.get("improved") else "")
            print(f"[stt] epoch {ev['epoch']:>3}: validation character error {ev['cer'] * 100:5.1f}%  word error {ev['wer'] * 100:5.1f}%{extra}", flush=True)
        elif k == "done":
            print(f"[stt] finished ({ev['reason']}) in {ev['seconds']}s: character error {ev['val_cer'] * 100:.1f}%, word error {ev['val_wer'] * 100:.1f}% on {ev['val_clips']} held-out clips. "
                  f"Saved {ev['path']}", flush=True)
    stop_file = lay.stop_file
    t0 = time.time()
    try:
        train_stt(cfg, lay, show, should_stop=lambda: os.path.exists(stop_file) and os.path.getmtime(stop_file) > t0, source_dir=a.source)
    except TrainSTTError as e:
        fail(str(e), 2)
    except KeyboardInterrupt:
        print("interrupted; the best model so far was kept")


def cmd_profile(a):
    from . import profile
    cfg, lay = load_layout(a)
    try:
        p = profile.measure(cfg, lay, a.source, a.clips, lambda m: print(m, flush=True))
    except RuntimeError as e:
        fail(str(e), 2)
    print(f"measured {p['clips']} clips: pitch {p['f0_hz']} Hz (spread {p['f0_spread_hz']}), pace {p['tempo']} phonemes/s -> saved; the built-in voice now uses them")


def cmd_data(a):
    from . import datasets as D
    cfg, lay = load_layout(a)
    root = a.source or lay.audio_data
    if not os.path.isdir(root):
        fail(f"{root} does not exist. Create it and put a speech dataset in it.", 2)
    pairs, rep = D.discover(root)
    print(f"folder   : {root}\naudio    : {rep['audio_files']:,} files\npairs    : {len(pairs):,} with transcripts "
          f"({rep['sidecars']:,} from .txt/.lab files; tables: {', '.join(rep['tables']) or 'none'})")
    if rep["missing_audio"]:
        print(f"missing  : {rep['missing_audio']:,} table rows name audio that is not there")
    secs = 0.0
    for p, t in pairs[: a.sample or 0]:
        d = A.duration_s(p, cfg)
        secs += d or 0
        print(f"  {os.path.relpath(p, root)}  ({(d or 0):.1f}s)  ->  {t[:70]}")
    if not pairs:
        print("Nothing usable. A metadata table needs an audio column and a text column (or put a dataset.json next to it naming them); "
              "otherwise put a .txt next to every clip. See docs/VOICE.md.")


def cmd_convert(a):
    cfg = load_cfg(a)
    src, dst = a.src, a.dst
    sr = None if a.rate == 0 else a.rate
    ext = "." + a.format.lstrip(".")
    jobs = []
    if os.path.isdir(src):
        for dp, dn, fn in os.walk(src):
            dn[:] = [d for d in dn if not d.startswith(".")]
            for f in sorted(fn):
                if os.path.splitext(f)[1].lower() in A.AUDIO_EXTENSIONS:
                    rel = os.path.relpath(os.path.join(dp, f), src)
                    jobs.append((os.path.join(dp, f), os.path.join(dst, os.path.splitext(rel)[0] + ext)))
        if not jobs:
            fail(f"no audio files under {src}", 2)
    else:
        if not os.path.isfile(src):
            fail(f"not found: {src}", 2)
        out = dst if os.path.splitext(dst)[1] else os.path.join(dst, os.path.splitext(os.path.basename(src))[0] + ext)
        jobs.append((src, out))
    ok = bad = 0
    total = 0.0
    for i, (s, d) in enumerate(jobs, 1):
        try:
            r = A.convert_file(s, d, sr, not a.keep_channels, a.bits, a.normalize, cfg)
            total += r["seconds"]
            ok += 1
        except A.AudioError as e:
            bad += 1
            print(f"  skipped {s}: {e}")
        if len(jobs) > 1 and i % 50 == 0:
            print(f"  {i}/{len(jobs)}", flush=True)
    print(f"converted {ok} file(s) ({total / 60:.1f} min of audio) to {dst}" + (f"; {bad} failed" if bad else ""))
    return 0 if not bad else 1


def cmd_devices(a):
    from .devices import list_devices
    cfg = load_cfg(a)
    print(list_devices(cfg))


def cmd_test(a):
    """Round trip through the configured engines: speak a phrase, transcribe it, compare."""
    from . import ctc as C
    from .datasets import normalize_text
    from .service import VoiceService
    cfg, lay = load_layout(a)
    v = VoiceService(cfg, lay)
    text = a.text or "hello how are you today"
    try:
        x, sr = v.synthesize_array(text)
        print(f"speech out ({v.tts_engine().name}): {len(x) / sr:.1f} s")
        heard = v.transcribe_array(x, sr)
    except Exception as e:                                                  # noqa: BLE001
        fail(f"round trip failed: {e}", 1)
    ref, hyp = normalize_text(text), normalize_text(heard)
    print(f"speech in  ({v.stt_engine().name}): '{heard}'\ncharacter error: {C.error_rate([ref], [hyp]) * 100:.0f}%")
    print("Note: this tests the plumbing and how well the recogniser copes with the synthesised voice, not its accuracy on real speakers.")


def register(cs, add_common_fn=add_common):
    v = cs.add_parser("voice", help="speech in / speech out: say, listen, train the built-in recogniser, convert audio")
    vs = v.add_subparsers(dest="voice_cmd", required=True, metavar="COMMAND")

    def mk(name, fn, help):
        p = vs.add_parser(name, help=help)
        add_common_fn(p)
        p.set_defaults(fn=fn)
        return p

    mk("status", cmd_status, "which speech engines are in use / available")
    p = mk("say", cmd_say, "speak text (or write it to an audio file)")
    p.add_argument("text", nargs="*")
    p.add_argument("--out", help="write the speech to this file (.wav, or any format ffmpeg can encode)")
    p.add_argument("--play", action="store_true", help="also play it when --out is given")
    p = mk("listen", cmd_listen, "transcribe an audio file, or one utterance from the microphone")
    p.add_argument("file", nargs="?")
    p.add_argument("--seconds", type=float, help="give up waiting for speech after this long")
    p = mk("ask", cmd_ask, "voice question in a file -> text answer (and a spoken answer with --out)")
    p.add_argument("file")
    p.add_argument("--out", help="write the spoken answer here")
    p.add_argument("--session")
    p.add_argument("--system")
    p.add_argument("--local", action="store_true")
    p.add_argument("--temperature", type=float)
    p.add_argument("--max-tokens", type=int, dest="max_tokens")
    p = mk("train-stt", cmd_train_stt, "train the built-in recogniser on the speech in the data/audio folder")
    p.add_argument("--source", help="speech dataset folder (default: <data>/<voice.audio_folder>)")
    p.add_argument("--minutes", type=float, help="wall-clock budget")
    p.add_argument("--epochs", type=int)
    p = mk("profile", cmd_profile, "measure pitch and pace from a speech dataset for the built-in voice")
    p.add_argument("--source")
    p.add_argument("--clips", type=int, default=200)
    p = mk("data", cmd_data, "show which (audio, transcript) pairs the speech dataset folder yields")
    p.add_argument("--source")
    p.add_argument("--sample", type=int, default=5, help="print this many example pairs")
    p = mk("convert", cmd_convert, "convert audio files / folders (wav, mp3, flac, ogg, m4a ...) to training-ready audio")
    p.add_argument("src")
    p.add_argument("dst", help="output file or folder")
    p.add_argument("--format", default="wav", help="output format when dst is a folder (default wav)")
    p.add_argument("--rate", type=int, default=16000, help="sample rate in Hz (0 = keep)")
    p.add_argument("--keep-channels", action="store_true", help="do not mix down to mono")
    p.add_argument("--bits", type=int, choices=[16, 24, 32], default=16)
    p.add_argument("--normalize", type=float, metavar="PEAK", help="scale to this peak, e.g. 0.95")
    mk("devices", cmd_devices, "list microphones, speakers and the backends found")
    p = mk("test", cmd_test, "speak a phrase and recognise it again through the configured engines")
    p.add_argument("--text")
    return v
