"""Tests for 6.5 real-time multimodal streaming (zerowhite/stream/)."""
import os
import shutil
import sys
import tempfile
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ZERO_WHITE_DATA", tempfile.mkdtemp(prefix="zw_test_data_"))

from zerowhite.chat.generate import KVCache, forward_cached            # noqa: E402
from zerowhite.chat.model import GPT, GPTConfig                          # noqa: E402
from zerowhite.chat.tokenizer import Tokenizer                           # noqa: E402
from zerowhite.stream import (AudioDecoder, AudioStream, FrameStackEnv, MaskObsEnv, PixelCodec, Ring, RingOverflow, StreamPipeline,   # noqa: E402
                              TextStream, TokenSpace, VideoStream, WindowLM, interleave)
from zerowhite.voice import codec as C                                   # noqa: E402

_DIRS = []


def tmp():
    d = tempfile.mkdtemp(prefix="zw_t65s_")
    _DIRS.append(d)
    return d


import atexit                                                           # noqa: E402
atexit.register(lambda: [shutil.rmtree(d, ignore_errors=True) for d in _DIRS])

_CACHE = {}


def tokenizer():
    if "tok" not in _CACHE:
        corpus = ["the quick brown fox jumps over the lazy dog. " * 20, "def f(x):\n    return x * 2  # it's fine, they'll see\n" * 20,
                  "naïve café — 日本語のテキスト 🙂 don't we've I'm you're " * 10, "numbers 12345 and 3.14159 and   spaces   here\n\n" * 10]
        _CACHE["tok"] = Tokenizer.train(corpus, 380, ["<|pad|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"], min_frequency=2)
    return _CACHE["tok"]


def small_codec():
    if "codec" not in _CACHE:
        r = np.random.default_rng(0)
        waves = [np.sin(2 * np.pi * f * np.arange(8000) / 16000.0) * r.uniform(0.3, 0.8) + 0.05 * r.normal(size=8000) for f in (110, 220, 330, 440)]
        _CACHE["codec"] = C.train(waves, stages=3, codewords=16, hidden=32, latent=8, epochs=4, finetune_epochs=2, seed=0)
    return _CACHE["codec"]


def chunks(n, rng, lo=1, hi=300):
    out, i = [], 0
    while i < n:
        k = int(rng.integers(lo, hi))
        out.append((i, min(n, i + k)))
        i += k
    return out


# =========================================================================== ring buffer
def test_a_ring_keeps_its_books_straight_under_every_overflow_policy_and_wraparound():
    r = Ring(8, dtype=np.int64)
    r.push(np.arange(5))
    assert len(r) == 5 and list(r.peek()) == [0, 1, 2, 3, 4] and list(r.latest(2)) == [3, 4]
    assert list(r.read(3)) == [0, 1, 2]
    r.push(np.arange(5, 11))                                          # wraps around the end of the array
    assert list(r.read()) == [3, 4, 5, 6, 7, 8, 9, 10] and len(r) == 0
    r.push(np.arange(20))                                             # more than the capacity: only the newest 8 survive, the loss is counted
    assert list(r.peek()) == list(range(12, 20)) and r.dropped == 12
    assert r.pushed == r.consumed + r.dropped + len(r)
    old = Ring(4, dtype=np.int64, overflow="drop_oldest")
    new = Ring(4, dtype=np.int64, overflow="drop_newest")
    for ring in (old, new):
        ring.push([1, 2, 3])
        ring.push([4, 5, 6])
    assert list(old.read()) == [3, 4, 5, 6] and list(new.read()) == [1, 2, 3, 4] and old.dropped == new.dropped == 2
    err = Ring(4, dtype=np.int64, overflow="error")
    err.push([1, 2, 3])
    try:
        err.push([4, 5])
        raise AssertionError("should have refused")
    except RingOverflow:
        pass
    assert list(err.peek()) == [1, 2, 3] and err.pushed == 3                                           # a refused push changes nothing
    fr = Ring(3, shape=(2, 2))
    fr.push(np.ones((2, 2)))                                          # a single row
    fr.push(np.zeros((2, 2, 2)))
    assert len(fr) == 3 and fr.read(1).shape == (1, 2, 2)
    try:
        fr.push(np.ones((3, 3)))
        raise AssertionError("shape")
    except ValueError:
        pass
    rr = np.random.default_rng(0)                                     # random pushes and reads against a plain list model
    ring, model, dropped = Ring(13, dtype=np.int64), [], 0
    for i in range(300):
        x = rr.integers(0, 1000, int(rr.integers(0, 9)))
        ring.push(x)
        model.extend(int(v) for v in x)
        if len(model) > 13:
            dropped += len(model) - 13
            model = model[-13:]
        if rr.random() < 0.5:
            k = int(rr.integers(0, 10))
            got, model = list(ring.read(k)), model[k:]
        assert list(ring.peek()) == model, i
    assert ring.dropped == dropped


def test_a_ring_works_between_a_producer_thread_and_a_consumer_thread_without_losing_a_row():
    n = 20000
    r = Ring(1 << 15, dtype=np.int64)                                 # larger than the whole stream, so even a slow consumer loses nothing
    got = []

    def producer():
        for i in range(0, n, 100):
            r.push(np.arange(i, i + 100))

    t = threading.Thread(target=producer)
    t.start()
    while len(got) < n:
        got.extend(r.read_blocking(500, timeout=2.0).tolist())
        if not t.is_alive() and len(r) == 0 and len(got) < n:
            break
    t.join()
    assert got == list(range(n)) and r.dropped == 0 and r.pushed == r.consumed == n
    assert len(Ring(4).read_blocking(2, timeout=0.05)) == 0                      # a timeout returns what is there (nothing), it does not hang


# =========================================================================== text
def _text_chunked(tok, data, cuts, **kw):
    s = TextStream(tok, **kw)
    out = []
    for a, b in cuts:
        out.extend(s.feed(data[a:b]))
    out.extend(s.finish())
    return out, s


def test_streaming_text_gives_exactly_the_batch_tokens_for_any_chunking_including_cuts_inside_a_utf8_character():
    tok = tokenizer()
    rng = np.random.default_rng(1)
    pieces = ["the quick brown fox", "  leading", "it's", "they'll", "we've", "I'm", "don't!", "12345", "3.14", "\n\n", "\t", "   ", "日本語", "🙂", "naïve", "café", "_x_",
              "'", "'l", "'ll", "x'", "ab  ", " cd", "snake_case", "!!!", "...", "a" * 30, " " * 40]
    texts = ["".join(rng.choice(pieces, size=int(rng.integers(1, 40)))) for _ in range(60)] + ["", " ", "'", "'l", "a'l", "x'll", "   ", " a", "\n"]
    for text in texts:
        want = tok.encode(text)
        data = text.encode("utf-8")
        for _ in range(6):
            cuts = chunks(len(data), rng, 1, 12)                         # byte chunks: cuts land inside multi-byte characters
            assert _text_chunked(tok, data, cuts)[0] == want, (text, cuts)
        assert _text_chunked(tok, text, chunks(len(text), rng, 1, 5))[0] == want    # str chunks too
        assert _text_chunked(tok, data, [(0, len(data))])[0] == want
    bad = bytes(rng.integers(0, 256, 400).tolist())                       # arbitrary bytes: invalid UTF-8 is replaced exactly as the batch decode replaces it
    for _ in range(5):
        assert _text_chunked(tok, bad, chunks(len(bad), rng, 1, 20))[0] == tok.encode(bad.decode("utf-8", errors="replace"))


def test_the_text_stream_holds_back_only_a_short_tail_and_a_giant_word_is_cut_exactly_where_the_tokenizer_cuts_it():
    tok = tokenizer()
    s = TextStream(tok)
    out = s.feed("hello world, this is a stream. and more words")
    assert s.held < 12 and len(out) > 5                                   # the unfinished last word waits; everything before it is already out
    s.finish()
    assert s.held == 0
    giant = "x" * 5000 + " tail " + "y" * 3000                           # one word far longer than max_hold
    want = tok.encode(giant)
    got, st = _text_chunked(tok, giant, chunks(len(giant), np.random.default_rng(2), 1, 60), max_hold=200)
    assert got == want and st.forced > 0
    got2, st2 = _text_chunked(tok, "é" * 700, chunks(1400, np.random.default_rng(3), 1, 9), max_hold=128)      # a giant word of multi-byte characters
    assert got2 == tok.encode("é" * 700) and st2.forced > 0                  # cut positions are byte aligned, so this holds exactly


# =========================================================================== audio
def test_streaming_audio_encode_equals_the_batch_encoder_for_any_chunking_and_the_first_token_appears_after_one_hop():
    cd = small_codec()
    rng = np.random.default_rng(0)
    x = rng.normal(size=6000) * 0.3 + np.sin(np.arange(6000) * 0.05) * 0.4
    want = cd.encode(x)
    for lo, hi in ((1, 50), (1, 400), (100, 1500), (160, 161), (7, 8)):
        s = AudioStream(cd)
        got = [s.feed(x[a:b]) for a, b in chunks(len(x), rng, lo, hi)] + [s.finish()]
        got = np.concatenate(got)
        assert got.shape == want.shape and np.array_equal(got, want), (lo, hi)
    s = AudioStream(cd)
    assert len(s.feed(x[:AudioStream.latency_samples - 1])) == 0           # 159 samples: not yet
    assert len(s.feed(x[AudioStream.latency_samples - 1:AudioStream.latency_samples])) == 1                           # the 160th sample completes frame 0 (10 ms at 16 kHz)
    for st in (1, 2):                                                       # a prefix of the stages is a valid coarser code, streamed the same way
        s2 = AudioStream(cd, stages=st)
        got = np.concatenate([s2.feed(x), s2.finish()])
        assert np.array_equal(got, cd.encode(x, stages=st)) and got.shape[1] == st
    e = AudioStream(cd)
    e.finish()
    try:
        e.feed(x[:10])
        raise AssertionError("feeding after finish must fail")
    except RuntimeError:
        pass
    assert AudioStream(cd).finish().shape == (cd.encode(np.zeros(0)).shape[0], 3)                                       # an empty stream matches the batch encoder too


def test_streaming_audio_decode_equals_the_batch_decoder_and_a_full_stream_round_trip_matches_the_codec():
    cd = small_codec()
    rng = np.random.default_rng(4)
    x = rng.normal(size=7000) * 0.3
    toks = cd.encode(x)
    want = cd.decode(toks, len(x))
    for lo, hi in ((1, 3), (1, 20), (5, 60)):
        d = AudioDecoder(cd, n_samples=len(x))
        parts = [d.feed(toks[a:b]) for a, b in chunks(len(toks), rng, lo, hi)] + [d.finish()]
        got = np.concatenate(parts)
        assert got.shape == want.shape and np.allclose(got, want, atol=1e-9), (lo, hi)
    d = AudioDecoder(cd)
    assert len(d.feed(toks[:1])) == 0                                      # one frame: nothing is final yet except the padding
    first = d.feed(toks[1:2])
    assert len(first) == 160 - 0 and np.allclose(first, want[:160], atol=1e-9)         # two frames make the first hop of audio final
    dd = AudioDecoder(cd)
    got = np.concatenate([dd.feed(toks), dd.finish()])                      # no length given: HOP * (frames - 1), the batch default
    assert np.allclose(got, cd.decode(toks), atol=1e-9)
    d3 = AudioDecoder(cd)                                                   # without a length the output runs to the end of the last frame; trimming it gives the same samples
    over = np.concatenate([d3.feed(toks), d3.finish(len(x) + 100)])
    assert len(over) >= len(x) and np.allclose(over[:len(x)], want, atol=1e-9)
    enc, dec, out = AudioStream(cd), AudioDecoder(cd, n_samples=len(x)), []                   # the whole live loop
    for a, b in chunks(len(x), rng, 50, 700):
        out.append(dec.feed(enc.feed(x[a:b])))
    out.append(dec.feed(enc.finish()))
    out.append(dec.finish())
    live = np.concatenate(out)
    assert live.shape == x.shape and np.allclose(live, cd.roundtrip(x), atol=1e-9)


# =========================================================================== pixels / video
def _scene(t, h=16, w=16):
    f = np.full((h, w), 0.2)
    f[2:6, 2:6] = 0.9                                                   # a static block
    x = (t * 2) % (w - 4)
    f[10:14, x:x + 4] = 0.6                                             # a moving block
    return f


def test_the_pixel_codec_improves_with_stages_round_trips_through_a_file_and_the_video_stream_reports_what_changed():
    frames = [_scene(t) for t in range(40)]
    pc = PixelCodec(patch=4).fit(frames, stages=3, codewords=12, seed=0)
    e1, e3 = pc.mse(frames, 1), pc.mse(frames, 3)
    assert e3 <= e1 + 1e-12 and e3 < 0.01
    tok = pc.encode(frames[3])
    assert tok.shape == (4, 4, 3) and tok.dtype == np.int32 and tok.max() < 12
    assert np.array_equal(pc.encode(frames[3]), tok)
    assert pc.decode(tok).shape == (16, 16) and pc.bits_per_frame((16, 16)) == 16 * 3 * np.log2(12)
    p = os.path.join(tmp(), "px.npz")
    pc.save(p)
    pc2 = PixelCodec.load(p)
    assert np.array_equal(pc2.encode(frames[7]), pc.encode(frames[7]))
    vs = VideoStream(pc)
    tok0, ch0 = vs.push(frames[0])
    assert ch0.all()                                                    # the first frame: everything is new
    tok1, ch1 = vs.push(frames[1])
    assert 0 < ch1.sum() < ch1.size and not ch1[0, 0]                   # only the patches under the moving block change; the static corner does not
    assert len(vs.changes(tok1, ch1)) == int(ch1.sum())
    for t in range(2, 20):
        vs.push(frames[t])
    assert 0.3 < vs.static_fraction < 1.0 and len(vs.recent(3)) == 3
    rgb = PixelCodec(patch=2, channels=3).fit([np.random.default_rng(0).random((4, 4, 3)) for _ in range(5)], stages=1, codewords=4)
    assert rgb.decode(rgb.encode(np.ones((4, 4, 3)) * 0.5)).shape == (4, 4, 3)
    try:
        pc.encode(np.zeros((15, 16)))
        raise AssertionError("shape must be checked")
    except ValueError:
        pass


# =========================================================================== one id space
def test_the_token_space_keeps_modalities_apart_and_interleaving_is_time_ordered_and_reversible():
    tok, cd = tokenizer(), small_codec()
    pc = PixelCodec(patch=4).fit([_scene(t) for t in range(10)], stages=2, codewords=8)
    sp = TokenSpace.for_models(tok, cd, pc)
    assert sp.vocab_size == tok.vocab_size + len(TokenSpace.MARKERS) + 3 * 16 + 2 * 8
    ids = sp.audio(np.array([[0, 1, 2], [15, 15, 15]]))
    assert [sp.classify(i)[:1] for i in ids] == [("audio",)] * 6 and [sp.classify(i)[1:] for i in ids[:3]] == [(0, 0), (1, 1), (2, 2)]
    assert np.array_equal(sp.audio_back(ids), [[0, 1, 2], [15, 15, 15]])
    px = sp.pixels(np.array([[[0, 7], [3, 4]]]))
    assert [sp.classify(i)[0] for i in px] == ["pixel"] * 4 and sp.classify(px[1])[1:] == (1, 7)
    seen = set()
    for i in range(sp.vocab_size):
        c = sp.classify(i)
        assert c not in seen or c[0] == "text"
        seen.add(c)
    try:
        sp.classify(sp.vocab_size)
        raise AssertionError("outside")
    except ValueError:
        pass
    try:
        sp.audio(np.array([[16, 0, 0]]))
        raise AssertionError("code out of range")
    except ValueError:
        pass
    text = sp.text(tok.encode("hello world"))
    merged = interleave(sp, [(0.2, "audio", ids), (0.1, "text", text), (0.2, "pixel", px)])
    assert merged[0] == sp.marker("<|text|>") and sp.marker("<|audio|>") in merged and sp.marker("<|video|>") in merged
    assert merged.index(sp.marker("<|audio|>")) < merged.index(sp.marker("<|video|>"))      # equal times keep the order given
    back = sp.split(merged)
    assert tok.decode(back["text"]) == "hello world" and back["audio"] == ids and back["pixel"] == px
    try:
        interleave(sp, [(0.0, "smell", [])])
        raise AssertionError("unknown modality")
    except ValueError:
        pass


# =========================================================================== sliding-window LM
def _lm(ctx=16, vocab=40, seed=0, **kw):
    cfg = GPTConfig(vocab=vocab, d_model=16, n_layer=2, n_head=2, ffn=24, ctx=ctx, tie=True, dtype="float64", **kw)
    return GPT(cfg, seed=seed, init_std=0.3)


def _fresh(m, ids):
    cache = KVCache(m.cfg)
    lg = forward_cached(m, cache, ids)
    return lg, cache


def test_the_reprefill_window_is_exactly_a_fresh_pass_over_the_current_window_however_long_the_stream():
    m = _lm()
    rng = np.random.default_rng(0)
    stream = rng.integers(1, 40, 130).tolist()
    w = WindowLM(m, window=16, stride=4, mode="reprefill")
    last = None
    for a, b in chunks(len(stream), rng, 1, 9):
        last = w.feed(stream[a:b])
        assert w.cache.n <= 16 and len(w.tokens) == w.cache.n
        lg, cache = _fresh(m, w.tokens)
        assert np.allclose(last, lg, atol=1e-9)
        assert np.allclose(w.cache.K[:, :, :w.cache.n], cache.K[:, :, :cache.n], atol=1e-9)
    assert w.seen == len(stream) and w.slides > 5 and w.tokens == stream[-len(w.tokens):]
    assert 0 < w.recomputed and w.describe()["exact"]
    big = WindowLM(m, window=16, stride=4)
    big.feed(stream)                                                      # one call with far more than the window: fed in pieces, no error
    assert np.allclose(big.feed([]), _fresh(m, big.tokens)[0], atol=1e-9)
    s = WindowLM(m, window=16, stride=4, sink=3)
    s.feed(stream)
    assert s.tokens[:3] == stream[:3] and len(s.tokens) <= 16                       # the attention sink: the first tokens stay for ever
    assert np.allclose(s.feed([]), _fresh(m, s.tokens)[0], atol=1e-9)
    for bad in (dict(window=16, stride=16), dict(window=16, stride=4, sink=12), dict(mode="nope")):
        try:
            WindowLM(m, **bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_the_rotate_window_turns_cached_keys_back_exactly_at_the_first_layer_and_is_only_approximate_deeper():
    from zerowhite.chat.model import _rope
    m = _lm()
    cos, sin = m.rope_tables(40)
    k = np.random.default_rng(0).normal(size=(2, 5, 8))                      # (heads, positions, head_dim) as the cache stores them (already rotated by position)
    for j0, d in ((7, 3), (12, 12)):
        at_j = _rope(k, cos[j0:j0 + 5], sin[j0:j0 + 5])
        at_shifted = _rope(k, cos[j0 - d:j0 - d + 5], sin[j0 - d:j0 - d + 5])
        from zerowhite.chat.model import _rope_back
        assert np.allclose(_rope_back(at_j, cos[d], sin[d]), at_shifted, atol=1e-12)           # rotating by -d turns position j into j-d
    rng = np.random.default_rng(1)
    stream = rng.integers(1, 40, 90).tolist()
    w = WindowLM(m, window=16, stride=4, mode="rotate")
    w.feed(stream)
    _, fresh = _fresh(m, w.tokens)
    n = w.cache.n
    assert n == len(w.tokens) and w.slides > 5 and w.recomputed == 0
    assert np.allclose(w.cache.K[0, :, :n], fresh.K[0, :, :n], atol=1e-8) and np.allclose(w.cache.V[0, :, :n], fresh.V[0, :, :n], atol=1e-8)     # layer 0: exact
    deeper = float(np.abs(w.cache.K[1, :, :n] - fresh.K[1, :, :n]).max())
    assert deeper > 1e-6                                                     # layer 1 kept context from tokens that are gone: NOT a fresh pass (and the description says so)
    assert not w.describe()["exact"] and "close to" in w.describe()["note"]
    lg = w.feed([5])
    ref = WindowLM(m, window=16, stride=4, mode="reprefill")
    ref.feed(stream + [5])
    assert np.isfinite(lg).all() and lg.shape == ref.feed([]).shape
    s = WindowLM(m, window=16, stride=4, mode="rotate", sink=2)
    s.feed(stream)
    assert s.tokens[:2] == stream[:2] and s.cache.n == len(s.tokens) <= 16
    _, f2 = _fresh(m, s.tokens)
    assert np.allclose(s.cache.K[0, :, :s.cache.n], f2.K[0, :, :s.cache.n], atol=1e-8)


def test_a_window_lm_generates_past_its_context_and_scores_a_long_stream():
    m = _lm()
    w = WindowLM(m, window=16, stride=4)
    out = list(w.generate([3, 4, 5], 60, rng=np.random.default_rng(0), temperature=0.9))
    assert len(out) == 60 and all(0 <= t < 40 for t in out) and w.seen == 63 and w.slides > 5       # 63 tokens through a 16-token window
    w2 = WindowLM(m, window=16, stride=4)
    assert out == list(w2.generate([3, 4, 5], 60, rng=np.random.default_rng(0), temperature=0.9))      # deterministic given the rng
    w3 = WindowLM(m, window=16, stride=4)
    assert list(w3.generate([3], 40, temperature=0.0, end_id=None)) == list(WindowLM(m, window=16, stride=4).generate([3], 40, temperature=0.0))
    first = WindowLM(m, window=16, stride=4)
    lp = first.logprobs([1, 2, 3, 4, 5, 6, 7, 8, 9] * 5)
    assert np.isnan(lp[0]) and np.isfinite(lp[1:]).all() and (lp[1:] <= 1e-12).all()
    assert np.isfinite(WindowLM(m, window=16, stride=4).perplexity(list(range(1, 30)) * 3))
    ended = list(WindowLM(m, window=16).generate([1], 30, temperature=0.0, end_id=-1))
    assert len(ended) == 30
    try:
        list(WindowLM(m).generate([], 3))
        raise AssertionError("empty prompt on a fresh stream")
    except ValueError:
        pass


# =========================================================================== pipeline
def test_the_pipeline_turns_live_text_audio_and_video_into_one_framed_token_stream_and_reports_latency_and_loss():
    tok, cd = tokenizer(), small_codec()
    frames = [_scene(t) for t in range(12)]
    pc = PixelCodec(patch=4).fit(frames, stages=1, codewords=8)
    sp = TokenSpace.for_models(tok, cd, pc)
    now = [0.0]
    got = []
    pipe = StreamPipeline(sp, clock=lambda: now[0], on_tokens=got.append)
    pipe.add_text("chat", TextStream(tok))
    pipe.add_audio("mic", AudioStream(cd), capacity=4000)
    pipe.add_video("cam", VideoStream(pc), shape=(16, 16), capacity=4)
    x = np.random.default_rng(0).normal(size=1600) * 0.3
    pipe.push("chat", "hello there, how are you today?")
    pipe.push("mic", x[:800])
    pipe.push("cam", np.stack(frames[:2]))
    now[0] = 0.25
    pipe.push("mic", x[800:])
    assert pipe.pump() > 0
    now[0] = 0.5
    pipe.flush()
    flat = [i for chunk in got for i in chunk]
    parts = sp.split(flat)
    assert tok.decode(parts["text"]) == "hello there, how are you today?"
    assert np.array_equal(sp.audio_back(parts["audio"]), cd.encode(x)) and len(parts["pixel"]) == 2 * 16
    assert all(c[0] in (sp.marker("<|text|>"), sp.marker("<|audio|>"), sp.marker("<|video|>")) and c[-1] == sp.marker("<|end_chunk|>") for c in got)
    st = pipe.stats()
    assert st["mic"]["latency_max"] == 0.25 and st["chat"]["latency_mean"] == 0.25 and st["cam"]["dropped"] == 0 and st["mic"]["ids"] == 3 * len(cd.encode(x))
    now[0] = 1.0
    pipe.push("chat", "again")
    now[0] = 1.1
    pipe.pump()
    assert abs(pipe.stats()["chat"]["latency_max"] - 0.25) < 1e-9 and abs(pipe.stats()["chat"]["latency_mean"] - 0.175) < 1e-9          # the mean follows the newer, shorter wait
    pipe.push("cam", np.stack(frames[:10]))                               # 10 frames into a ring of 4: the 6 oldest are lost, and the report says so
    assert pipe.stats()["cam"]["dropped"] == 6
    assert StreamPipeline(sp).pump() == 0


def test_the_pipeline_thread_feeds_a_window_lm_and_a_failing_consumer_is_reported_not_swallowed():
    tok, cd = tokenizer(), small_codec()
    sp = TokenSpace.for_models(tok, cd)
    cfg = GPTConfig(vocab=sp.vocab_size, d_model=16, n_layer=1, n_head=2, ffn=24, ctx=32, dtype="float64")
    lm = WindowLM(GPT(cfg, seed=0, init_std=0.2), window=32, stride=8)
    pipe = StreamPipeline(sp, on_tokens=lm.feed)
    pipe.add_text("chat", TextStream(tok))
    pipe.add_audio("mic", AudioStream(cd), capacity=1 << 15)
    pipe.start(poll=0.001)
    x = np.random.default_rng(0).normal(size=8000) * 0.3
    for i in range(0, 8000, 400):
        pipe.push("mic", x[i:i + 400])
        pipe.push("chat", "stream " * 3)
    pipe.stop(flush=True)
    assert lm.seen > 100 and lm.slides > 0 and len(lm.tokens) <= 32 and np.isfinite(lm.feed([])).all()
    assert pipe.stats()["mic"]["dropped"] == 0

    def boom(ids):
        raise ValueError("consumer exploded")

    bad = StreamPipeline(sp, on_tokens=boom)
    bad.add_text("chat", TextStream(tok))
    bad.start(poll=0.001)
    bad.push("chat", "hi there")
    for _ in range(200):
        if bad.error is not None:
            break
        threading.Event().wait(0.01)
    try:
        bad.stop()
        raise AssertionError("the failure must surface")
    except RuntimeError as e:
        assert "consumer exploded" in str(e)


# =========================================================================== frame stacking
def test_frame_stacking_gives_a_memoryless_policy_the_velocity_a_partially_observable_task_hides():
    from zerowhite.agents import make_env

    def run(name, policy, n=12):
        rets = []
        for s in range(n):
            env, rng = make_env(name), np.random.default_rng(s)
            o, done, tot = env.reset(rng), False, 0.0
            while not done:
                o, r, done, _ = env.step(policy(o))
                tot += r
            rets.append(tot)
        return float(np.mean(rets))

    single = run("pomdp:cartpole", lambda o: int(o[1] > 0))                      # angle only: cannot tell a falling pole from a rising one
    stacked = run("stack2:pomdp:cartpole", lambda o: int(o[3] + 10.0 * (o[3] - o[1]) > 0))      # two angles: their difference IS the angular velocity
    assert single < 80 and stacked >= 190
    e = make_env("stack3:pomdp:cartpole")
    assert (e.obs_dim, e.n_actions, e.name) == (6, 2, "stack3:pomdp:cartpole")
    o = e.reset(np.random.default_rng(0))
    assert np.allclose(o[:2], o[2:4]) and np.allclose(o[2:4], o[4:])                  # the first observation fills the whole stack: no invented zeros
    o2, _, _, _ = e.step(1)
    assert np.allclose(o2[:4], o[2:]) and not np.allclose(o2[4:], o[4:])               # oldest first, newest last
    k1 = FrameStackEnv(make_env("cartpole"), 1)
    assert k1.obs_dim == 4 and np.allclose(k1.reset(np.random.default_rng(0)), make_env("cartpole").reset(np.random.default_rng(0)))
    for bad in ("pomdp:gridworld5", "stackx:cartpole"):
        try:
            make_env(bad)
            raise AssertionError(bad)
        except KeyError:
            pass
    for bad in (dict(keep=()), dict(keep=(9,))):
        try:
            MaskObsEnv(make_env("cartpole"), **bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    try:
        FrameStackEnv(make_env("cartpole"), 0)
        raise AssertionError("k")
    except ValueError:
        pass


def test_the_policy_gradient_family_and_the_task_machinery_accept_a_stacked_environment():
    from zerowhite.forge import task as T
    from zerowhite.forge.families import FAMILIES, families_for
    t = T.from_env_name("stack2:pomdp:cartpole")
    assert t.obs_dim == 4 and t.n_actions == 2 and t.signature()["dim"] == 4
    assert "actor_critic" in [f.name for f in families_for(t)]
    g = dict(hidden=12, depth=1, lr=0.01, gamma=0.97, entropy=0.02, curiosity=0.0)
    a = FAMILIES["actor_critic"].build(t, g, 0)
    a.train(t, 0.3, np.random.default_rng(0))
    skill, met = a.val_score(t)
    assert np.isfinite(skill) and met["mean_return"] > 0
    assert t.random_baseline()[0] > 0 and 0.0 < t.reference() <= 200.0
