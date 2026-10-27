"""codec.py - a small learned audio codec: waveform -> a few integer tokens per 10 ms -> waveform, in plain numpy.

Pipeline (the structure of a neural codec such as SoundStream / EnCodec, scaled down to what trains on a CPU in a minute):

  frames    320-sample frames (20 ms at 16 kHz), hop 160, sqrt-Hann window on analysis and synthesis (so overlap-add of the windowed frames reconstructs exactly);
  encoder   MLP 320 -> 160 (tanh) -> 24, trained jointly with
  decoder   MLP 24 -> 160 (tanh) -> 320, by Adam on the reconstruction error of the windowed frames (an autoencoder);
  RVQ       RESIDUAL vector quantisation of the 24-d latent: stage 1 is a k-means codebook of the latents, stage 2 a k-means codebook of what stage 1 left over, and so on
            (S stages x K codewords each; the token for a frame is S integers < K); the decoder is then fine-tuned on QUANTISED latents so it learns to undo the
            quantisation noise. More stages = more bits = higher SNR; any prefix of the stages is a valid, coarser code (scalable bitrate).

`encode(wave) -> int array (frames, S)` and `decode(tokens) -> wave` are the 'audio tokens'; `tokens_to_text` / `text_to_tokens` serialise them as plain text so they can be
put in a corpus next to words (`chat/modal.py`). What it is NOT: it is trained and tested on SYNTHETIC speech from the project's own formant synthesiser (the only speech this
repository can generate). It has never seen a real voice, real room noise or music; real speech has aperiodicity, prosody and background the synthetic voices lack, and a
codec trained on one distribution loses quality on another. Treat the SNR figures as 'it works on this distribution', not as codec quality; retrain on your own recordings with
`zerowhite voice codec-train --wav-dir ...`. A comparison against a plain PCA transform at equal bitrate is in the tests and docs.
"""
from __future__ import annotations

import numpy as np

FRAME, HOP = 320, 160
WIN = np.sqrt(0.5 * (1.0 - np.cos(2.0 * np.pi * (np.arange(FRAME) + 0.5) / FRAME)))      # sqrt-Hann: WIN[t]^2 + WIN[t+HOP]^2 == 1


def frames_of(x: np.ndarray):
    """(n_frames, FRAME) windowed frames of a mono signal (padded with HOP zeros at both ends so every sample is covered twice) and the original length."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    pad = np.concatenate([np.zeros(HOP), x, np.zeros(HOP + FRAME)])
    nf = 1 + (len(pad) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(nf)[:, None]
    return pad[idx] * WIN[None, :], n


def overlap_add(fr: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(HOP * (len(fr) - 1) + FRAME)
    for i, f in enumerate(fr):
        out[i * HOP:i * HOP + FRAME] += f * WIN
    return out[HOP:HOP + n]


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    ref, est = np.asarray(ref, float), np.asarray(est, float)[: len(ref)]
    return float(10 * np.log10((np.sum(ref ** 2) + 1e-12) / (np.sum((ref - est) ** 2) + 1e-12)))


def _kmeans(X, k, rng, iters=20):
    k = min(k, len(X))
    C = X[rng.choice(len(X), size=k, replace=False)].copy()
    for _ in range(iters):
        d = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = X[m].mean(0)
            else:
                C[j] = X[rng.integers(len(X))]                              # re-seed an empty cell
    return C


class Codec:
    def __init__(self, W1, b1, W2, b2, W3, b3, W4, b4, books, gain):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4 = W1, b1, W2, b2, W3, b3, W4, b4
        self.books, self.gain = books, float(gain)                          # books: list of (K, latent) arrays, one per RVQ stage

    # ---- structure
    @property
    def stages(self) -> int:
        return len(self.books)

    @property
    def codewords(self) -> int:
        return int(self.books[0].shape[0])

    def bitrate(self, stages=None) -> float:
        """Bits per second at 16 kHz for the first `stages` stages."""
        return (stages or self.stages) * float(np.log2(self.codewords)) * (16000.0 / HOP)

    # ---- network
    def _enc(self, U):
        return np.tanh(U @ self.W1 + self.b1) @ self.W2 + self.b2

    def _dec(self, Z):
        return np.tanh(Z @ self.W3 + self.b3) @ self.W4 + self.b4

    # ---- tokens
    def quantise(self, Z, stages=None):
        S = stages or self.stages
        res, toks, q = Z.copy(), [], np.zeros_like(Z)
        for book in self.books[:S]:
            d = (res ** 2).sum(1)[:, None] - 2 * res @ book.T + (book ** 2).sum(1)[None, :]
            a = d.argmin(1)
            toks.append(a)
            q += book[a]
            res = Z - q
        return np.stack(toks, axis=1), q

    def encode(self, x: np.ndarray, stages=None) -> np.ndarray:
        fr, _ = frames_of(np.asarray(x, dtype=np.float64) * self.gain)
        toks, _ = self.quantise(self._enc(fr), stages)
        return toks.astype(np.int32)

    def dequantise(self, toks):
        toks = np.asarray(toks)
        q = np.zeros((len(toks), self.books[0].shape[1]))
        for s in range(toks.shape[1]):
            q += self.books[s][toks[:, s]]
        return q

    def decode(self, toks: np.ndarray, n_samples: int = None) -> np.ndarray:
        toks = np.asarray(toks, dtype=np.int64)
        fr = self._dec(self.dequantise(toks))
        n = n_samples if n_samples is not None else HOP * (len(toks) - 1)
        return overlap_add(fr, n) / self.gain

    def roundtrip(self, x, stages=None):
        return self.decode(self.encode(x, stages), len(x))

    # ---- persistence
    def save(self, path):
        arrays = dict(W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3, W4=self.W4, b4=self.b4, gain=np.array([self.gain]))
        for i, b in enumerate(self.books):
            arrays[f"book{i}"] = b
        np.savez_compressed(path, **{k: v.astype(np.float32) for k, v in arrays.items()})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        books, i = [], 0
        while f"book{i}" in z:
            books.append(z[f"book{i}"].astype(np.float64))
            i += 1
        g = lambda k: z[k].astype(np.float64)                                                  # noqa: E731
        return cls(g("W1"), g("b1"), g("W2"), g("b2"), g("W3"), g("b3"), g("W4"), g("b4"), books, float(z["gain"][0]))


def _adam(params, grads, m, v, step, lr):
    for j, (p, g) in enumerate(zip(params, grads)):
        m[j] = 0.9 * m[j] + 0.1 * g
        v[j] = 0.999 * v[j] + 0.001 * g * g
        p -= lr * (m[j] / (1 - 0.9 ** step)) / (np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)


def train(waves, stages: int = 4, codewords: int = 64, hidden: int = 160, latent: int = 24, epochs: int = 40, finetune_epochs: int = 10, seed: int = 0,
          lr: float = 2e-3, verbose: bool = False, quantiser_dropout: bool = True) -> Codec:
    """Train on a list of mono float waveforms (16 kHz). Deterministic given the seed."""
    rng = np.random.default_rng(seed)
    allx = np.concatenate([np.asarray(w, float).ravel() for w in waves])
    gain = 1.0 / max(np.sqrt(np.mean(allx ** 2)), 1e-6) * 0.25                  # a fixed gain so that the network sees signals of a stable scale
    U = np.concatenate([frames_of(np.asarray(w, float) * gain)[0] for w in waves])
    W1 = rng.normal(0, np.sqrt(1.0 / FRAME), (FRAME, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, np.sqrt(1.0 / hidden), (hidden, latent))
    b2 = np.zeros(latent)
    W3 = rng.normal(0, np.sqrt(1.0 / latent), (latent, hidden))
    b3 = np.zeros(hidden)
    W4 = rng.normal(0, np.sqrt(1.0 / hidden), (hidden, FRAME))
    b4 = np.zeros(FRAME)
    P = [W1, b1, W2, b2, W3, b3, W4, b4]
    m, v = [np.zeros_like(p) for p in P], [np.zeros_like(p) for p in P]
    step = 0

    def epoch(Xin, quantised=None, decoder_only=False):
        nonlocal step
        order = rng.permutation(len(Xin))
        tot = 0.0
        for i in range(0, len(order), 128):
            idx = order[i:i + 128]
            X = Xin[idx]
            h1 = np.tanh(X @ W1 + b1)
            Z = quantised[idx] if quantised is not None else h1 @ W2 + b2
            h2 = np.tanh(Z @ W3 + b3)
            Y = h2 @ W4 + b4
            dY = 2.0 * (Y - X) / X.size
            tot += float(((Y - X) ** 2).sum())
            gW4, gb4 = h2.T @ dY, dY.sum(0)
            dh2 = (dY @ W4.T) * (1 - h2 ** 2)
            gW3, gb3 = Z.T @ dh2, dh2.sum(0)
            if decoder_only:
                grads = [np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2), gW3, gb3, gW4, gb4]
            else:
                dZ = dh2 @ W3.T
                gW2, gb2 = h1.T @ dZ, dZ.sum(0)
                dh1 = (dZ @ W2.T) * (1 - h1 ** 2)
                grads = [X.T @ dh1, dh1.sum(0), gW2, gb2, gW3, gb3, gW4, gb4]
            step += 1
            _adam(P, grads, m, v, step, lr if not decoder_only else lr * 0.5)
        return tot / Xin.size
    for ep in range(epochs):
        loss = epoch(U)
        if verbose and ep % 10 == 9:
            print(f"autoencoder epoch {ep + 1}/{epochs}  mse {loss:.5f}")
    Z = np.tanh(U @ W1 + b1) @ W2 + b2
    books, res = [], Z.copy()
    for s in range(stages):
        C = _kmeans(res, codewords, rng)
        d = (res ** 2).sum(1)[:, None] - 2 * res @ C.T + (C ** 2).sum(1)[None, :]
        res = res - C[d.argmin(1)]
        books.append(C)
    codec = Codec(*P, books, gain)
    Qs = [codec.quantise(Z, s)[1] for s in range(1, stages + 1)]                  # the latent as seen after 1, 2, ... S stages
    for ep in range(finetune_epochs):                                           # the decoder learns to undo the quantisation noise at EVERY depth
        for s in (rng.permutation(stages) if quantiser_dropout else [stages - 1]):   # ('quantiser dropout', as in SoundStream: without it a decoder tuned on the full
            loss = epoch(U, quantised=Qs[int(s)], decoder_only=True)             # code does worse on a prefix of it than on the full code, and quality is not monotone)
        if verbose and ep % 5 == 4:
            print(f"decoder fine-tune epoch {ep + 1}/{finetune_epochs}  mse {loss:.5f}")
    return Codec(W1, b1, W2, b2, W3, b3, W4, b4, books, gain)


def pca_baseline(waves, bits_per_frame_stages=4, codewords=64, latent=24, seed=0):
    """The classical alternative at the SAME bit budget: a linear PCA transform (no learning beyond an SVD) in place of the MLP encoder / decoder, same RVQ."""
    rng = np.random.default_rng(seed)
    allx = np.concatenate([np.asarray(w, float).ravel() for w in waves])
    gain = 1.0 / max(np.sqrt(np.mean(allx ** 2)), 1e-6) * 0.25
    U = np.concatenate([frames_of(np.asarray(w, float) * gain)[0] for w in waves])
    mu = U.mean(0)
    _, _, vt = np.linalg.svd(U - mu, full_matrices=False)
    B = vt[:latent]
    Z = (U - mu) @ B.T
    books, res = [], Z.copy()
    for _ in range(bits_per_frame_stages):
        C = _kmeans(res, codewords, rng)
        d = (res ** 2).sum(1)[:, None] - 2 * res @ C.T + (C ** 2).sum(1)[None, :]
        res = res - C[d.argmin(1)]
        books.append(C)

    class PCA:
        def roundtrip(self, x, stages=None):
            fr, n = frames_of(np.asarray(x, float) * gain)
            Zx = (fr - mu) @ B.T
            q, rr = np.zeros_like(Zx), Zx.copy()
            for book in books[:stages or len(books)]:
                d = (rr ** 2).sum(1)[:, None] - 2 * rr @ book.T + (book ** 2).sum(1)[None, :]
                q += book[d.argmin(1)]
                rr = Zx - q
            return overlap_add(q @ B + mu, n) / gain
    return PCA()


def tokens_to_text(toks: np.ndarray) -> str:
    """One word per frame, e.g. `a12.7.55.3`, so audio can sit in a text corpus; separated by spaces."""
    return " ".join("a" + ".".join(str(int(t)) for t in row) for row in np.asarray(toks))


def text_to_tokens(text: str) -> np.ndarray:
    rows = [[int(x) for x in w[1:].split(".")] for w in text.split() if w.startswith("a") and w[1:].replace(".", "").isdigit()]
    return np.array(rows, dtype=np.int32).reshape(len(rows), -1) if rows else np.zeros((0, 0), dtype=np.int32)
