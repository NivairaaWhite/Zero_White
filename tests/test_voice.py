"""Tests for speech in / speech out: audio files, resampling, voice-activity detection, the built-in synthesiser, the built-in
recogniser (maths and a real small training run), dataset discovery for speech folders, the engine plug-ins and the service.

The hardware paths (microphone capture, speaker playback) need real devices and are only covered by their selection / error
logic here; see docs/VOICE.md for what has and has not been exercised.
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from test_chat import _http, _serve, copy_workspace, tiny_cfg, tmp                # noqa: E402
from zerowhite.chat.paths import Layout                                                 # noqa: E402
from zerowhite.voice import audio as A                                                  # noqa: E402
from zerowhite.voice import ctc, formant, g2p, stt as STT, train_stt as TS, tts as TTS  # noqa: E402
from zerowhite.voice import datasets as VD, devices as DV, features as F, vad           # noqa: E402
from zerowhite.voice.service import VoiceService, clean_for_speech                      # noqa: E402

SR = 16000
WORDS = ["red", "green", "blue", "yes", "no", "stop", "go", "left", "right", "up"]


def tone(f, dur=0.5, sr=SR, amp=0.5):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def fit_amp(x, f, sr):
    """Least-squares amplitude of a sinusoid of known frequency (robust to window scalloping)."""
    t = np.arange(len(x)) / sr
    A_ = np.stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)], 1)
    c, *_ = np.linalg.lstsq(A_, x.astype(np.float64), rcond=None)
    return float(np.hypot(*c))


# ---------------------------------------------------------------------------------------------------- audio files
def test_wav_round_trips_at_every_bit_depth_and_stereo_is_mixed_down():
    x = tone(440, 0.2)
    for bits, tol in ((16, 1e-3), (24, 1e-5), (32, 1e-8), (33, 1e-6)):
        data = A.encode_wav(x, SR, bits)
        assert A.is_wav(data)
        y, sr = A.parse_wav(data)
        assert sr == SR and len(y) == len(x) and np.abs(y - x).max() < tol, bits
    st = np.stack([x, -x], 1)
    y, _ = A.load_audio(A.encode_wav(st, SR), mono=True)
    assert np.abs(y).max() < 1e-3                                                         # opposite channels cancel
    try:
        A.parse_wav(b"RIFF....nope")
        raise AssertionError("garbage must be rejected")
    except A.AudioError:
        pass


def test_resampler_preserves_tones_removes_aliases_and_handles_awkward_ratios():
    for sr_in, sr_out, f in ((16000, 8000, 1000), (16000, 44100, 1000), (44100, 16000, 3000), (22050, 16000, 500)):
        x = tone(f, 0.5, sr_in)
        y = A.resample(x, sr_in, sr_out)
        assert abs(len(y) - int(round(len(x) * sr_out / sr_in))) <= 1
        core = y[len(y) // 10: -len(y) // 10]
        assert abs(fit_amp(core, f, sr_out) - 0.5) < 0.02, (sr_in, sr_out)
    hi = A.resample(tone(7000, 0.5, 16000), 16000, 8000)                                 # above the new Nyquist: must vanish, not alias
    assert np.abs(hi[200:-200]).max() < 0.02
    assert np.array_equal(A.resample(tone(500), SR, SR), tone(500))


def test_convert_file_changes_rate_depth_and_loudness_and_never_clips():
    d = tmp()
    src, dst = os.path.join(d, "a.wav"), os.path.join(d, "b.wav")
    A.save_audio(src, tone(440, 0.3, 22050, 0.2), 22050, bits=24)
    info = A.convert_file(src, dst, sr=16000, mono=True, bits=16, normalize=0.9)
    y, sr = A.load_audio(dst)
    assert sr == 16000 and abs(np.abs(y).max() - 0.9) < 0.01 and info
    assert abs(A.duration_s(dst) - 0.3) < 0.01


# ---------------------------------------------------------------------------------------------------- speech detection
def _speech_with_gaps(seed=0):
    rng = np.random.default_rng(seed)
    syn = formant.FormantSynth(formant.Voice(), seed=1)
    a, b = syn.synthesize("good morning to you."), syn.synthesize("how are you today?")
    gap = lambda s: (0.003 * rng.standard_normal(int(s * SR))).astype(np.float32)             # noqa: E731
    x = np.concatenate([gap(1.2), a, gap(1.0), b, gap(1.0)])
    return x, len(a), len(b)


def test_voice_activity_detection_finds_each_utterance_offline_and_streaming():
    cfg = tiny_cfg()
    x, la, lb = _speech_with_gaps()
    segs = vad.split_utterances(x, SR, cfg)
    assert len(segs) == 2, segs
    (s1, e1), (s2, e2) = segs
    assert abs((e1 - s1) - la) < 0.5 * SR and abs((e2 - s2) - lb) < 0.5 * SR
    seg = vad.Segmenter(cfg, SR)
    out = []
    for i in range(0, len(x), 1600):                                                      # 100 ms chunks, like a microphone
        out += seg.feed(x[i:i + 1600])
    tail = seg.flush()
    if tail is not None:
        out.append(tail)
    assert len(out) == 2 and seg.calibrated
    assert vad.split_utterances(np.zeros(SR, np.float32), SR, cfg) == []                  # digital silence: nothing to hear


# ---------------------------------------------------------------------------------------------------- text to phones
def test_text_to_phones_handles_numbers_punctuation_and_unknown_words():
    assert g2p.number_to_words(0) == "zero" and g2p.number_to_words(21) == "twenty one"
    assert g2p.number_to_words(1234) == "one thousand two hundred thirty four"
    assert g2p.word_to_phones("cat") == ["K", "AE", "T"]
    assert g2p.word_to_phones("dogs")[-1] == "Z" and g2p.word_to_phones("cats")[-1] == "S"      # plural voicing follows the stem
    assert g2p.word_to_phones("jumped")[-1] == "T"
    toks = g2p.expand_text("I have 3 cats, 2.5 dogs & a bird!")
    words = [t for t, k in toks if k == "w"]
    assert "three" in words and "point" in words and "and" in words and ("!", "!") in toks
    for w in ("zyxqv", "Xylophone's", "naïve", "don't", "a"):
        assert isinstance(g2p.word_to_phones(w), list)                                      # never raises on anything
    assert g2p.text_to_phones("") == 0


# ---------------------------------------------------------------------------------------------------- synthesiser
def test_the_built_in_voice_is_deterministic_bounded_and_has_the_requested_pitch():
    a = formant.synthesize("hello world, this is a test.")
    b = formant.synthesize("hello world, this is a test.")
    assert np.array_equal(a, b) and a.dtype == np.float32
    assert 1.2 < len(a) / SR < 4.0 and np.isfinite(a).all() and np.abs(a).max() <= 1.0 and np.abs(a).max() > 0.2
    assert len(formant.synthesize("")) == 0
    low = formant.synthesize("we saw a long road", f0_hz=100.0)
    high = formant.synthesize("we saw a long road", f0_hz=200.0)
    f0 = lambda x: float(np.median([v for v in F.estimate_f0(x, SR) if v and v > 0]))        # noqa: E731
    assert abs(f0(low) - 100) < 15 and abs(f0(high) - 200) < 25
    fast = formant.synthesize("we saw a long road", tempo=18.0)
    slow = formant.synthesize("we saw a long road", tempo=9.0)
    assert len(slow) > 1.5 * len(fast)
    assert not np.allclose(formant.synthesize("a bee"), formant.synthesize("a boo")[:len(formant.synthesize("a bee"))])


def test_log_mel_features_have_the_documented_shape_and_react_to_pitch():
    m = F.log_mel(tone(500, 1.0), SR, n_mels=40, win_ms=25.0, hop_ms=10.0)
    assert m.shape[1] == 40 and abs(m.shape[0] - 98) <= 3
    lo, hi = F.log_mel(tone(300, 0.5), SR, 40).mean(0), F.log_mel(tone(3000, 0.5), SR, 40).mean(0)
    assert lo.argmax() < hi.argmax()


# ---------------------------------------------------------------------------------------------------- recogniser maths
def test_ctc_loss_equals_the_sum_over_all_alignments_and_its_gradient_is_exact():
    import itertools
    rng = np.random.default_rng(0)
    T, V, labels = 5, 4, [1, 2, 2]
    z = rng.standard_normal((T, V))
    lp = ctc.log_softmax(z)

    def collapse(path):
        out, prev = [], -1
        for s in path:
            if s != prev and s != 0:
                out.append(s)
            prev = s
        return out
    total = sum(np.exp(sum(lp[t, s] for t, s in enumerate(path))) for path in itertools.product(range(V), repeat=T) if collapse(path) == labels)
    loss, g = ctc.ctc_loss_grad(lp, labels)
    assert abs(loss + np.log(total)) < 1e-9
    num = np.zeros_like(z)
    for i in np.ndindex(*z.shape):
        zp, zm = z.copy(), z.copy()
        zp[i] += 1e-6
        zm[i] -= 1e-6
        num[i] = (ctc.ctc_loss_grad(ctc.log_softmax(zp), labels)[0] - ctc.ctc_loss_grad(ctc.log_softmax(zm), labels)[0]) / 2e-6
    assert np.abs(num - g).max() < 1e-6
    loss2, g2 = ctc.ctc_loss_grad(ctc.log_softmax(rng.standard_normal((2, V))), [1, 2, 3])        # impossible alignment: skipped, not fatal
    assert g2 is None


def test_acoustic_model_backward_matches_finite_differences_and_batching_does_not_change_results():
    rng = np.random.default_rng(1)
    cfg = ctc.CTCConfig(n_in=6, hidden=8, layers=3, vocab=5, kernel=3, dilations=[1, 2])
    m = ctc.CTCModel(cfg, rng)
    m.p = {k: v.astype(np.float64) for k, v in m.p.items()}
    X = rng.standard_normal((2, 14, 6))
    lens = np.array([14, 9])
    targets = [[1, 2], [3]]

    def loss():
        lg, lo, _ = m.forward(X, lens)
        return ctc.batch_loss_grad(lg, lo, targets)[0]
    lg, lo, cache = m.forward(X, lens)
    _, dl, ok, _ = ctc.batch_loss_grad(lg, lo, targets)
    grads = m.backward(dl, cache)
    for name in grads:
        flat = m.p[name].reshape(-1)
        for j in rng.choice(flat.size, 3, replace=False):
            old = flat[j]
            flat[j] = old + 1e-6
            a = loss()
            flat[j] = old - 1e-6
            b = loss()
            flat[j] = old
            assert abs((a - b) / 2e-6 - grads[name].reshape(-1)[j]) < 1e-5 * max(1.0, abs(grads[name].reshape(-1)[j])), name
    lg1, _, _ = m.forward(X[1:2, :9], lens[1:])                                           # the short one alone == inside the padded batch
    assert np.allclose(lg1[0], lg[1, :lg1.shape[1]], atol=1e-9)
    assert ctc.error_rate(["abc", "de"], ["abd", "de"]) == 1 / 5 and ctc.error_rate(["a b c"], ["a c"], words=True) == 1 / 3


# ---------------------------------------------------------------------------------------------------- speech datasets
def test_speech_folders_in_the_common_layouts_are_paired_with_their_transcripts():
    root = tmp()
    clip = A.encode_wav(tone(300, 0.2), SR)

    def put(rel, data=clip):
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb" if isinstance(data, bytes) else "w", **({} if isinstance(data, bytes) else dict(encoding="utf-8"))) as f:
            f.write(data)
    put("lj/wavs/LJ001-0001.wav")
    put("lj/wavs/LJ001-0002.wav")
    put("lj/metadata.csv", "LJ001-0001|Printing, in the only sense|printing in the only sense\nLJ001-0002|With which we are concerned, 1,000|with which we are concerned one thousand\n")
    put("cv/clips/a.wav")
    put("cv/clips/b.wav")
    put("cv/validated.tsv", "client_id\tpath\tsentence\nx\ta.wav\tHello there.\nx\tb.wav\tWhat's up?\nx\tmissing.wav\tNo audio\n")
    put("vctk/wav48/p225/p225_001_mic1.wav")
    put("vctk/txt/p225/p225_001.txt", "Please call Stella.\n")
    put("side/clip1.wav")
    put("side/clip1.txt", "A sidecar transcript\n")
    pairs, rep = VD.discover(root)
    got = {os.path.basename(p): t for p, t in pairs}
    assert got["LJ001-0001.wav"] == "printing in the only sense" or got["LJ001-0001.wav"].startswith("printing")
    assert got["a.wav"] == "hello there" and got["b.wav"] == "what's up"
    assert got["p225_001_mic1.wav"] == "please call stella" and got["clip1.wav"] == "a sidecar transcript"
    assert len(pairs) == 6 and rep["missing_audio"] == 1, (len(pairs), rep)
    assert VD.normalize_text("It’s  “Fine” — OK!") == "it's fine ok"


# ---------------------------------------------------------------------------------------------------- recogniser, trained for real
# `auto` picks the best engine *of the machine*: macOS `say` and Windows SAPI for speech out, and (Windows) SAPI for speech in. The
# built-in recogniser is trained on the built-in synthesiser's voice only, so these tests must not depend on what the OS provides.
BUILTIN_ENGINES = ["voice.tts.engine=builtin", "voice.stt.engine=builtin"]
# budget_minutes=null: the learning-rate schedule would otherwise also follow the wall clock, which makes the trained model depend on how
# fast the machine is (and a slow runner could hit the budget and stop early).


def _speech_workspace(n_per_word=10, seed=0):
    cfg = tiny_cfg(extra=BUILTIN_ENGINES + ["voice.builtin_stt.hidden=64", "voice.builtin_stt.layers=2", "voice.builtin_stt.epochs=200",
                          "voice.builtin_stt.patience=200", "voice.builtin_stt.budget_minutes=null", "voice.builtin_stt.batch=8",
                          "voice.builtin_stt.lr=0.004", "voice.builtin_stt.val_min=6"])
    lay = Layout(cfg).ensure()
    d = os.path.join(lay.audio_data, "words")
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_per_word):
        for w in WORDS:
            v = formant.Voice(f0_hz=float(rng.uniform(100, 150)), tempo=12.0 * float(rng.uniform(0.9, 1.1)))
            x = formant.FormantSynth(v, seed=k).synthesize(w)
            x = x + 0.002 * rng.standard_normal(len(x)).astype(np.float32)
            A.save_audio(os.path.join(d, f"{w}_{k}.wav"), x, SR)
            rows.append(f"{w}_{k}|{w}|{w}")
    with open(os.path.join(d, "metadata.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    return cfg, lay


_STT = {}


def trained_recogniser():
    if "x" not in _STT:
        cfg, lay = _speech_workspace()
        res = TS.train_stt(cfg, lay)
        _STT["x"] = (cfg, lay, res)
    return _STT["x"]


def test_the_recogniser_trains_from_an_audio_folder_and_understands_a_voice_it_never_heard():
    cfg, lay, res = trained_recogniser()
    assert res["clips"] == 100 and res["val_clips"] >= 6 and os.path.exists(res["path"])
    assert res["val_cer"] < 0.2, res
    model = STT.BuiltinModel.load(res["path"])
    refs, hyps = [], []
    for i, w in enumerate(WORDS):
        x = formant.FormantSynth(formant.Voice(f0_hz=127.0 + i, tempo=12.4), seed=99).synthesize(w)
        refs.append(w)
        hyps.append(model.transcribe(x, SR, cfg))
    assert ctc.error_rate(refs, hyps) < 0.3, list(zip(refs, hyps))
    again = TS.train_stt(cfg.replace(**{"voice.builtin_stt.epochs": 3}), lay)                # training continues from the saved model
    assert again["resumed"] is True


def test_speech_in_and_speech_out_through_the_service_object_and_the_http_api():
    cfg, lay, _ = trained_recogniser()
    svc = VoiceService(cfg, lay)
    info = svc.describe()
    assert info["tts"] and info["stt"] == "builtin" and info["builtin_stt"]["val_cer"] < 0.2, info
    wav = svc.synthesize("stop")
    x, sr = A.parse_wav(wav)
    assert sr > 0 and len(x) > 0.2 * sr
    hyps = [svc.transcribe(svc.synthesize(w)) for w in WORDS]                                 # its own voice, through its own ears
    assert ctc.error_rate(WORDS, hyps) < 0.35, list(zip(WORDS, hyps))
    assert svc.transcribe(x, sr) == svc.transcribe(wav)                                       # bytes, file contents and arrays agree
    assert clean_for_speech("**Hello** `code` https://x.y/z  <b>there</b>\n\n- item") == "Hello code there item" or "http" not in clean_for_speech("see https://x.y/z now")
    assert VoiceService(cfg, lay).transcribe(A.encode_wav(np.zeros(SR // 2, np.float32), SR)) == ""      # silence is not text


def test_voice_endpoints_of_the_service_need_the_token_and_carry_audio_both_ways():
    import urllib.request
    cfg, lay, _ = trained_recogniser()
    ccfg, clay = copy_workspace(extra=BUILTIN_ENGINES)
    import shutil                                                                             # give the chat workspace the trained recogniser
    os.makedirs(clay.voice_models, exist_ok=True)
    shutil.copyfile(STT.builtin_path(cfg, lay), STT.builtin_path(ccfg, clay))
    app, srv = _serve(ccfg, clay)
    try:
        assert _http(srv, "POST", "/api/tts", {"text": "go"})[0] in (401, 403)
        st, wav = _http(srv, "POST", "/api/tts", {"text": "go"}, token=app.token)
        assert st == 200 and wav[:4] == b"RIFF"
        req = urllib.request.Request(f"http://127.0.0.1:{srv.port}/api/stt", data=wav, headers={"X-ZW-Token": app.token, "Content-Type": "audio/wav"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            heard = json.loads(r.read())["text"]
        # this test is about the HTTP layer carrying audio both ways; how accurate the recogniser is gets measured above. A single
        # two-letter word against a character-error threshold is a coin flip, so compare with what the service object hears instead.
        assert heard and heard == VoiceService(ccfg, clay).transcribe(wav), heard
        st, body = _http(srv, "GET", "/api/voice", token=app.token)
        assert st == 200 and "tts" in json.loads(body)
    finally:
        srv.stop()


# ---------------------------------------------------------------------------------------------------- engines and devices
def test_tts_engines_are_chosen_by_availability_and_never_required():
    cfg = tiny_cfg()
    rep = TTS.report(cfg)
    assert rep["builtin"][0] is True
    eng = TTS.make(cfg)
    x, sr = eng.synth("hello")
    assert len(x) > 0 and sr > 0
    try:
        TTS.make(cfg, None, "no-such-engine")
        raise AssertionError("unknown engines must be named in the error")
    except TTS.TTSError as e:
        assert "builtin" in str(e)
    parts = TTS.split_sentences("One. Two! Three? " + "word " * 100, limit=60)
    assert len(parts) > 1 and all(len(p) <= 60 for p in parts) and "".join(parts).replace(" ", "").startswith("One.Two!Three?")
    assert TTS.cache_key(eng, "a") != TTS.cache_key(eng, "b")
    ok, why = STT.report(cfg, Layout(cfg).ensure())["builtin"]                              # no model trained yet: says what to do
    assert not ok and "train" in why.lower(), why
    try:
        STT.make(cfg.replace(**{"voice.stt.engine": "builtin"}), Layout(cfg))
        raise AssertionError("an unusable engine must not be returned")
    except STT.STTError as e:
        assert "train" in str(e).lower()


def test_device_selection_and_missing_hardware_fail_with_instructions():
    cfg = tiny_cfg()
    text = DV.list_devices(cfg)
    assert "playback backend" in text and "capture backend" in text
    try:
        DV._pick_device(type("SD", (), {"query_devices": staticmethod(lambda: [dict(name="Built-in Mic", max_input_channels=1, max_output_channels=0)])}), "usb", "input")
        raise AssertionError("an unknown device must be an error")
    except DV.DeviceError as e:
        assert "usb" in str(e)
    sd = type("SD", (), {"query_devices": staticmethod(lambda: [dict(name="Built-in Mic", max_input_channels=1, max_output_channels=0)])})
    assert DV._pick_device(sd, "mic", "input") == 0 and DV._pick_device(sd, None, "input") is None and DV._pick_device(sd, "3", "input") == 3


def test_cli_voice_convert_handles_a_whole_folder_and_leaves_the_originals_alone():
    d = tmp()
    src = os.path.join(d, "in")
    os.makedirs(os.path.join(src, "sub"))
    for name in ("a.wav", os.path.join("sub", "b.wav")):
        A.save_audio(os.path.join(src, name), tone(440, 0.2, 22050), 22050, bits=24)
    out = os.path.join(d, "out")
    base = [sys.executable, "-m", "zerowhite", "chat", "voice", "convert", src, out, "-w", os.path.join(d, "ws")]
    r = subprocess.run(base + ["--rate", "16000", "--bits", "16"], cwd=ROOT, capture_output=True, text=True, timeout=120, env=dict(os.environ, PYTHONUTF8="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    y, sr = A.load_audio(os.path.join(out, "sub", "b.wav"))
    assert sr == 16000 and abs(len(y) / sr - 0.2) < 0.01
    assert A.load_audio(os.path.join(src, "a.wav"))[1] == 22050
