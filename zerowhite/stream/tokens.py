"""tokens.py - tokenisers that work on a LIVE stream: bytes/text, audio samples, pixel frames. The promise that makes them usable is one sentence:

    feeding the stream in ANY chunking gives exactly the tokens that tokenising the whole thing at once gives.

* TextStream     the chat model's byte-level BPE, fed bytes or text in arbitrary pieces. A token must not straddle a chunk boundary, so the tail of the stream that
                 could still change (an unfinished word, a half UTF-8 character, an apostrophe that may become "'ll") is held back until the next chunk or `finish()`.
* AudioStream    the voice codec (voice/codec.py): a frame is emitted the moment its 320 samples exist (latency = one hop, 10 ms), `AudioDecoder` overlap-adds frames back
                 into samples as soon as they are final.
* PixelCodec     a patch codec for images / video frames: non-overlapping p x p patches, residual vector quantisation (k-means codebooks), and `VideoStream`, which
                 tokenises each frame as it arrives and reports which patches changed since the last frame.
* TokenSpace     ONE integer id space for text, audio and pixel tokens plus modality markers (so a single transformer can read an interleaved stream), and `interleave`
                 to merge several timestamped modality streams into it. It is a layout, not a trained model: a chat model has to be trained on this vocabulary before it
                 does anything with the audio and pixel ids (see docs/STREAM.md).
"""
from __future__ import annotations

import codecs
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..chat.tokenizer import _PRETOKEN, Tokenizer
from ..voice import codec as _vc


# =========================================================================== text
class TextStream:
    """Chunk-invariant streaming wrapper of a `Tokenizer`.

    The BPE tokeniser works piece by piece (a piece is a regex 'pre-token': a word with its leading space, a run of digits, a run of punctuation, a run of spaces); pieces
    never merge across their edges, so the stream can be cut at any piece edge. A piece is only FINAL when text that follows it exists, because the pattern looks ahead:
    `'ll` needs three characters, and a run of spaces gives its last space to the next word. So the held-back tail is everything from the last piece edge that has at
    least three characters after it.

    `max_hold` bounds the tail for a pathological stream (one enormous word): past it the tail is cut where the tokeniser itself cuts a long piece (every
    `max_piece_bytes` bytes), which is again identical to the batch result."""

    def __init__(self, tokenizer: Tokenizer, max_hold: int = 1 << 16):
        self.tok = tokenizer
        self.max_hold = int(max_hold)
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending = ""
        self.fed_chars = 0
        self.emitted = 0
        self.forced = 0                                                   # times the max_hold cut was used

    def feed(self, data) -> List[int]:
        if isinstance(data, (bytes, bytearray, memoryview)):
            data = self._dec.decode(bytes(data))
        self._pending += data
        self.fed_chars += len(data)
        return self._drain(False)

    def finish(self) -> List[int]:
        self._pending += self._dec.decode(b"", final=True)
        return self._drain(True)

    @property
    def held(self) -> int:
        """Characters buffered but not yet turned into tokens (the latency of the text stream, in characters)."""
        return len(self._pending)

    def _drain(self, final: bool) -> List[int]:
        out: List[int] = []
        while self._pending:
            t = self._pending
            if final:
                cut = len(t)
            else:
                starts = [m.start() for m in _PRETOKEN.finditer(t)]
                limit = min(starts[-1], len(t) - 3)
                cut = max([s for s in starts if s <= limit] or [0])
                if cut == 0 and len(t) > self.max_hold and len(starts) == 1:
                    cut = self._aligned_cut(t)
                    if cut:
                        self.forced += 1
            if cut == 0:
                break
            out.extend(self.tok.encode(t[:cut]))
            self._pending = t[cut:]
            if final:
                break
        self.emitted += len(out)
        return out

    def _aligned_cut(self, t: str) -> int:
        """The largest char index at which the UTF-8 length is a multiple of the tokeniser's piece-split size (so cutting there is what the tokeniser does anyway)."""
        step, b = self.tok.max_piece_bytes, 0
        best = 0
        for i, ch in enumerate(t):
            b += len(ch.encode("utf-8"))
            if b % step == 0 and i + 1 < len(t):
                best = i + 1
        return best


# =========================================================================== audio
class AudioStream:
    """Live samples -> codec tokens. `feed(samples)` returns an int array (frames, stages) of the frames that became complete; `finish()` flushes the tail so that the
    concatenation of everything returned equals `codec.encode(whole signal)`."""

    def __init__(self, codec: _vc.Codec, stages: Optional[int] = None):
        self.codec, self.stages = codec, stages
        self._buf = np.zeros(_vc.HOP)                                     # the batch encoder pads HOP zeros in front
        self._base = 0                                                    # absolute position (in the padded signal) of _buf[0]
        self._i = 0                                                       # next frame to emit
        self.n_samples = 0
        self._done = False

    latency_samples = _vc.FRAME - _vc.HOP                                 # samples that must arrive before the first token exists

    def feed(self, samples) -> np.ndarray:
        if self._done:
            raise RuntimeError("stream already finished")
        x = np.asarray(samples, dtype=np.float64).ravel()
        self.n_samples += len(x)
        self._buf = np.concatenate([self._buf, x * self.codec.gain])
        return self._emit()

    def finish(self) -> np.ndarray:
        if self._done:
            return np.zeros((0, self.stages or self.codec.stages), dtype=np.int32)
        self._buf = np.concatenate([self._buf, np.zeros(_vc.HOP + _vc.FRAME)])
        self._done = True
        return self._emit()

    def _emit(self) -> np.ndarray:
        have = self._base + len(self._buf)
        n = (have - _vc.FRAME) // _vc.HOP + 1 - self._i if have >= _vc.FRAME else 0
        S = self.stages or self.codec.stages
        if n <= 0:
            return np.zeros((0, S), dtype=np.int32)
        idx = (self._i * _vc.HOP - self._base) + _vc.HOP * np.arange(n)[:, None] + np.arange(_vc.FRAME)[None, :]
        fr = self._buf[idx] * _vc.WIN[None, :]
        toks, _ = self.codec.quantise(self.codec._enc(fr), self.stages)
        self._i += n
        keep_from = self._i * _vc.HOP - self._base
        self._buf = self._buf[keep_from:]
        self._base += keep_from
        return toks.astype(np.int32)


class AudioDecoder:
    """Codec tokens -> samples, emitting every sample as soon as no later frame can change it. Latency: FRAME - HOP samples behind the newest frame."""

    def __init__(self, codec: _vc.Codec, n_samples: Optional[int] = None):
        """`n_samples`: the original length when it is known in advance (a file); output is then capped at it, as `Codec.decode(tokens, n_samples)` does. A live stream has no
        length: the codec's last frames carry a little padding past the end, which the consumer may trim."""
        self.codec = codec
        self.limit = None if n_samples is None else int(n_samples)
        self._acc = np.zeros(0)                                           # raw overlap-add positions [self._pos, self._pos + len)
        self._pos = 0
        self._frames = 0
        self._emitted = 0                                                 # output samples returned so far

    def _out(self, upto: int) -> np.ndarray:
        """Return raw positions [self._pos, upto) as output samples (raw positions < HOP are the batch decoder's discarded padding)."""
        k = max(0, upto - self._pos)
        seg = self._acc[:k]
        start = self._pos
        self._acc = self._acc[k:]
        self._pos += k
        if start < _vc.HOP:
            seg = seg[min(len(seg), _vc.HOP - start):]
        if self.limit is not None:
            seg = seg[:max(0, self.limit - self._emitted)]
        self._emitted += len(seg)
        return seg / self.codec.gain

    def feed(self, tokens) -> np.ndarray:
        toks = np.asarray(tokens, dtype=np.int64)
        if toks.ndim == 1:
            toks = toks[None]
        if len(toks) == 0:
            return np.zeros(0)
        fr = self.codec._dec(self.codec.dequantise(toks)) * _vc.WIN[None, :]
        for f in fr:
            at = self._frames * _vc.HOP - self._pos
            need = at + _vc.FRAME
            if len(self._acc) < need:
                self._acc = np.concatenate([self._acc, np.zeros(need - len(self._acc))])
            self._acc[at:at + _vc.FRAME] += f
            self._frames += 1
        return self._out(self._frames * _vc.HOP)

    def finish(self, n_samples: Optional[int] = None) -> np.ndarray:
        """Flush. `n_samples` is the original length if known (as for `Codec.decode`); by default HOP * (frames - 1), the batch decoder's default.
        Samples already returned by `feed` cannot be taken back, so an `n_samples` below what was streamed out just returns nothing more."""
        n = n_samples if n_samples is not None else self.limit if self.limit is not None else _vc.HOP * max(0, self._frames - 1)
        target = _vc.HOP + n                                              # raw end position
        end = self._pos + len(self._acc)
        return self._out(min(target, end))


# =========================================================================== pixels
class PixelCodec:
    """Patch codec for frames. `fit(frames)` learns `stages` residual k-means codebooks over p x p patches; `encode(frame)` -> (H/p, W/p, stages) ints; `decode` inverts.

    Frames are float arrays in [0, 1] of shape (H, W) or (H, W, C); H and W must be multiples of `patch`. It is a vector-quantiser, not a neural codec: honest, small,
    and exact enough to test (the reconstruction error falls as stages are added)."""

    def __init__(self, patch: int = 4, books: Optional[Sequence[np.ndarray]] = None, mean: Optional[np.ndarray] = None, channels: int = 1):
        self.patch, self.channels = int(patch), int(channels)
        self.books = [np.asarray(b, dtype=np.float64) for b in (books or [])]
        self.mean = np.zeros(self.patch * self.patch * self.channels) if mean is None else np.asarray(mean, dtype=np.float64)

    @property
    def stages(self) -> int:
        return len(self.books)

    @property
    def codewords(self) -> int:
        return int(self.books[0].shape[0]) if self.books else 0

    def _patches(self, frame) -> Tuple[np.ndarray, Tuple[int, int]]:
        f = np.asarray(frame, dtype=np.float64)
        if f.ndim == 2:
            f = f[:, :, None]
        H, W, C = f.shape
        p = self.patch
        if H % p or W % p or C != self.channels:
            raise ValueError(f"frame {f.shape} does not fit patch {p} with {self.channels} channel(s)")
        P = f.reshape(H // p, p, W // p, p, C).transpose(0, 2, 1, 3, 4).reshape(-1, p * p * C)
        return P, (H // p, W // p)

    def fit(self, frames: Iterable, stages: int = 2, codewords: int = 32, seed: int = 0, max_patches: int = 20000) -> "PixelCodec":
        rng = np.random.default_rng(seed)
        parts = []
        for fr in frames:
            P, _ = self._patches(fr)
            parts.append(P)
        X = np.concatenate(parts)
        if len(X) > max_patches:
            X = X[rng.choice(len(X), max_patches, replace=False)]
        self.mean = X.mean(0)
        res, books = X - self.mean, []
        for _ in range(int(stages)):
            C = _vc._kmeans(res, int(codewords), rng)
            d = (res ** 2).sum(1)[:, None] - 2 * res @ C.T + (C ** 2).sum(1)[None, :]
            res = res - C[d.argmin(1)]
            books.append(C)
        self.books = books
        return self

    def encode(self, frame, stages: Optional[int] = None) -> np.ndarray:
        P, (gh, gw) = self._patches(frame)
        res, toks = P - self.mean, []
        for C in self.books[:stages or self.stages]:
            d = (res ** 2).sum(1)[:, None] - 2 * res @ C.T + (C ** 2).sum(1)[None, :]
            a = d.argmin(1)
            toks.append(a)
            res = res - C[a]
        return np.stack(toks, axis=1).reshape(gh, gw, -1).astype(np.int32)

    def decode(self, tokens) -> np.ndarray:
        t = np.asarray(tokens, dtype=np.int64)
        gh, gw, S = t.shape
        P = np.zeros((gh * gw, self.mean.size)) + self.mean
        for s in range(S):
            P += self.books[s][t[:, :, s].reshape(-1)]
        p, C = self.patch, self.channels
        f = P.reshape(gh, gw, p, p, C).transpose(0, 2, 1, 3, 4).reshape(gh * p, gw * p, C)
        f = np.clip(f, 0.0, 1.0)
        return f[:, :, 0] if C == 1 else f

    def mse(self, frames: Iterable, stages: Optional[int] = None) -> float:
        errs = []
        for fr in frames:
            rec = self.decode(self.encode(fr, stages))
            errs.append(float(np.mean((np.asarray(fr, float).reshape(rec.shape) - rec) ** 2)))
        return float(np.mean(errs))

    def bits_per_frame(self, shape: Tuple[int, int], stages: Optional[int] = None) -> float:
        return (shape[0] // self.patch) * (shape[1] // self.patch) * (stages or self.stages) * float(np.log2(max(2, self.codewords)))

    def save(self, path: str):
        d = {"mean": self.mean, "patch": np.array([self.patch]), "channels": np.array([self.channels])}
        for i, b in enumerate(self.books):
            d[f"book{i}"] = b
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path: str) -> "PixelCodec":
        z = np.load(path)
        books, i = [], 0
        while f"book{i}" in z:
            books.append(z[f"book{i}"])
            i += 1
        return cls(int(z["patch"][0]), books, z["mean"], int(z["channels"][0]))


class VideoStream:
    """Frame-at-a-time tokenising with change detection. `push(frame)` -> (tokens, changed) where `changed` is a boolean (H/p, W/p) mask of patches whose
    token differs from the previous frame's (all True for the first frame). A consumer that only needs the changes (a static camera: most patches repeat) reads `changes()`.
    A bounded ring of the most recent token grids is kept so a late reader can catch up (`recent(n)`)."""

    def __init__(self, codec: PixelCodec, keep: int = 8):
        self.codec = codec
        self.keep = int(keep)
        self._recent: List[np.ndarray] = []
        self.frames = 0
        self.patches_total = 0
        self.patches_changed = 0

    def push(self, frame) -> Tuple[np.ndarray, np.ndarray]:
        tok = self.codec.encode(frame)
        prev = self._recent[-1] if self._recent else None
        changed = np.ones(tok.shape[:2], dtype=bool) if prev is None else (tok != prev).any(axis=2)
        self._recent.append(tok)
        del self._recent[:-self.keep]
        self.frames += 1
        self.patches_total += changed.size
        self.patches_changed += int(changed.sum())
        return tok, changed

    def recent(self, n: int) -> List[np.ndarray]:
        return list(self._recent[-int(n):])

    def changes(self, tok: np.ndarray, changed: np.ndarray) -> List[Tuple[int, int, Tuple[int, ...]]]:
        return [(int(i), int(j), tuple(int(v) for v in tok[i, j])) for i, j in zip(*np.nonzero(changed))]

    @property
    def static_fraction(self) -> float:
        return 0.0 if not self.patches_total else 1.0 - self.patches_changed / self.patches_total


# =========================================================================== one id space
class TokenSpace:
    """Layout of one integer vocabulary for several modalities:

        [0, text_vocab)                       text ids, exactly the chat tokenizer's
        text_vocab + 0..M-1                   modality markers  (<|audio|>, <|video|>, <|end|>, ... in `MARKERS`)
        then audio: audio_stages * audio_codewords ids          (stage s, code c -> base + s * K + c)
        then pixels: pixel_stages * pixel_codewords ids

    Two different stages (or modalities) never share an id, so the model can tell a stage-1 code from a stage-2 one without extra context."""

    MARKERS = ("<|audio|>", "<|video|>", "<|text|>", "<|sep|>", "<|end_chunk|>")

    def __init__(self, text_vocab: int, audio_stages: int = 0, audio_codewords: int = 0, pixel_stages: int = 0, pixel_codewords: int = 0):
        self.text_vocab = int(text_vocab)
        self.marker_base = self.text_vocab
        self.audio_base = self.marker_base + len(self.MARKERS)
        self.audio_stages, self.audio_codewords = int(audio_stages), int(audio_codewords)
        self.pixel_base = self.audio_base + self.audio_stages * self.audio_codewords
        self.pixel_stages, self.pixel_codewords = int(pixel_stages), int(pixel_codewords)
        self.vocab_size = self.pixel_base + self.pixel_stages * self.pixel_codewords

    @classmethod
    def for_models(cls, tokenizer: Tokenizer, codec: Optional[_vc.Codec] = None, pixels: Optional[PixelCodec] = None) -> "TokenSpace":
        return cls(tokenizer.vocab_size, codec.stages if codec else 0, codec.codewords if codec else 0, pixels.stages if pixels else 0, pixels.codewords if pixels else 0)

    def marker(self, name: str) -> int:
        return self.marker_base + self.MARKERS.index(name)

    def text(self, ids: Sequence[int]) -> List[int]:
        return [int(i) for i in ids]

    def audio(self, tokens: np.ndarray) -> List[int]:
        t = np.asarray(tokens, dtype=np.int64)
        if t.ndim == 1:
            t = t[None]
        if t.shape[1] > self.audio_stages or (t.size and int(t.max()) >= self.audio_codewords):
            raise ValueError("audio tokens do not fit this TokenSpace")
        return [int(self.audio_base + s * self.audio_codewords + c) for row in t for s, c in enumerate(row)]

    def pixels(self, tokens: np.ndarray) -> List[int]:
        t = np.asarray(tokens, dtype=np.int64)
        if t.ndim == 2:
            t = t[:, :, None]
        if t.shape[2] > self.pixel_stages or (t.size and int(t.max()) >= self.pixel_codewords):
            raise ValueError("pixel tokens do not fit this TokenSpace")
        return [int(self.pixel_base + s * self.pixel_codewords + c) for cell in t.reshape(-1, t.shape[2]) for s, c in enumerate(cell)]

    def classify(self, tid: int) -> Tuple[str, int, int]:
        """(modality, stage, code) of an id; stage / code are 0 for text and markers (code carries the text id or marker index)."""
        tid = int(tid)
        if tid < 0 or tid >= self.vocab_size:
            raise ValueError(f"id {tid} outside the vocabulary of {self.vocab_size}")
        if tid < self.marker_base:
            return "text", 0, tid
        if tid < self.audio_base:
            return "marker", 0, tid - self.marker_base
        if tid < self.pixel_base:
            r = tid - self.audio_base
            return "audio", r // self.audio_codewords, r % self.audio_codewords
        r = tid - self.pixel_base
        return "pixel", r // self.pixel_codewords, r % self.pixel_codewords

    def split(self, ids: Sequence[int]) -> Dict[str, List[int]]:
        """Undo `interleave` by modality: returns {'text': [...ids], 'audio': [flat ids], 'pixel': [flat ids]} (markers dropped), each in stream order."""
        out: Dict[str, List[int]] = {"text": [], "audio": [], "pixel": []}
        for t in ids:
            mod, _, _ = self.classify(t)
            if mod in out:
                out[mod].append(int(t))
        return out

    def audio_back(self, ids: Sequence[int]) -> np.ndarray:
        """Flat audio ids -> (frames, stages) codec tokens."""
        rows = np.array([self.classify(i)[2] for i in ids], dtype=np.int64)
        return rows.reshape(-1, self.audio_stages) if len(rows) else np.zeros((0, self.audio_stages), dtype=np.int64)


def interleave(space: TokenSpace, events: Sequence[Tuple[float, str, Sequence[int]]]) -> List[int]:
    """Merge timestamped chunks of several modalities into one token stream: events are (time, modality, ids) with ids already in `space` (use space.text / .audio / .pixels);
    they are ordered by time (ties keep their given order) and each is wrapped as `<|modality|> ids <|end_chunk|>`. Chunks from one source stay contiguous."""
    marks = {"audio": "<|audio|>", "pixel": "<|video|>", "video": "<|video|>", "text": "<|text|>"}
    out: List[int] = []
    for _, mod, ids in sorted(events, key=lambda e: e[0]):                 # sorted() is stable: equal times keep their given order
        if mod not in marks:
            raise ValueError(f"unknown modality '{mod}'")
        out.append(space.marker(marks[mod]))
        out.extend(int(i) for i in ids)
        out.append(space.marker("<|end_chunk|>"))
    return out
