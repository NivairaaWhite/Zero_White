"""train_stt.py - train the built-in recogniser on the speech datasets in the data folder.

    zerowhite chat voice train-stt            reads <data>/<voice.audio_folder>/ (LJ Speech, VCTK, Common Voice, your own recordings ...)

What happens: find (audio, transcript) pairs (datasets.py) -> log-mel features, cached on disk -> alphabet from the transcripts ->
stable-hash train/validation split -> CTC training with SpecAugment-style masking, cosine learning rate, wall-clock budget and
early stopping -> the best validation checkpoint (lowest character error rate) is saved to <models>/voice/stt.npz.
Sizes (hidden width, depth) are derived from how much speech there is unless you set them; everything is recorded in the result.
Continues from the existing model when its alphabet and shape still fit.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from collections import Counter
from typing import Callable, List, Optional, Tuple

import numpy as np

from .. import fileio
from ..chat import hardware as hw
from ..chat.optim import AdamW, cosine_lr
from . import audio as A
from . import ctc as C
from . import datasets as D
from .stt import BuiltinModel, SR, normalize, pad_silence, raw_features

MIN_SECONDS = 0.3


class TrainSTTError(RuntimeError):
    pass


def _hash(s: str, mod: int = 1_000_003) -> float:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def _stable_key(path: str, root: str) -> str:
    """Name a clip by where it sits *inside* the data folder. The training order and the held-out clips are chosen from this, so
    moving or re-mounting the folder (or a temp directory with a random name) does not reshuffle them."""
    try:
        rel = os.path.relpath(path, root)
    except ValueError:                                           # a different drive on Windows
        rel = path
    return rel.replace(os.sep, "/")


def derive_size(hours: float) -> Tuple[int, int, str]:
    for limit, h, l in ((0.5, 96, 3), (3.0, 128, 4), (20.0, 192, 5)):
        if hours < limit:
            return h, l, f"{hours:.2f} h of speech < {limit} h"
    return 256, 6, f"{hours:.1f} h of speech"


def _feature_cache_dir(lay) -> str:
    d = os.path.join(lay.voice_state, "features")
    os.makedirs(d, exist_ok=True)
    return d


def _load_features(path: str, cfg, lay, params: tuple) -> Optional[np.ndarray]:
    st = os.stat(path)
    key = hashlib.sha1(repr((os.path.abspath(path), st.st_mtime_ns, st.st_size, params)).encode()).hexdigest()[:24]
    cp = os.path.join(_feature_cache_dir(lay), key + ".npy")
    if os.path.exists(cp):
        try:
            return np.load(cp)
        except (OSError, ValueError):
            pass
    try:
        x, sr = A.load_audio(path, SR, True, cfg)
    except A.AudioError:
        return None
    if len(x) < MIN_SECONDS * SR:
        return None
    f = raw_features(x, SR, *params[1:]).astype(np.float16)
    tmp = cp + ".tmp.npy"
    np.save(tmp, f)
    fileio.replace(tmp, cp)
    return f


def _prepare(f_raw: np.ndarray, rng: Optional[np.random.Generator], pad_frames: int, hop_ms: float) -> np.ndarray:
    """Training / evaluation input from cached raw features. With `rng` (training): random silence either side, random speed change,
    frequency and time masks. Without: the fixed silence padding used at inference."""
    f = f_raw.astype(np.float32)
    if rng is None:
        return normalize(pad_silence(f, pad_frames, pad_frames))
    T = len(f)
    stretch = float(rng.uniform(0.88, 1.12))                                   # speaking-rate perturbation
    if abs(stretch - 1.0) > 0.01 and T > 8:
        src = np.linspace(0, T - 1, max(8, int(round(T * stretch))))
        f = np.stack([np.interp(src, np.arange(T), f[:, m]) for m in range(f.shape[1])], axis=1).astype(np.float32)
    lead = int(rng.integers(0, int(2 * pad_frames) + 1))
    tail = int(rng.integers(0, int(2 * pad_frames) + 1))
    sil = silence_of(f)
    for _ in range(2):                                                         # frequency masks
        w = int(rng.integers(0, max(2, f.shape[1] // 8)))
        s = int(rng.integers(0, max(1, f.shape[1] - w)))
        f[:, s:s + w] = sil
    for _ in range(max(1, len(f) // 150)):                                     # time masks
        w = int(rng.integers(0, min(20, max(2, len(f) // 8))))
        s = int(rng.integers(0, max(1, len(f) - w)))
        f[s:s + w, :] = sil
    return normalize(pad_silence(f, lead, tail)) + float(rng.normal(0, 0.02))


def silence_of(f: np.ndarray) -> float:
    return float(f.min())


def _pad(feats: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    lens = np.array([len(f) for f in feats])
    X = np.zeros((len(feats), int(lens.max()), feats[0].shape[1]), np.float32)
    for i, f in enumerate(feats):
        X[i, : len(f)] = f
    return X, lens


def evaluate(model: C.CTCModel, bm_alphabet: List[str], items: List[Tuple[np.ndarray, str]], pad_frames: int, hop_ms: float, batch: int = 16) -> Tuple[float, float]:
    """(character error rate, word error rate) of greedy decoding over (features, reference text) pairs."""
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    refs, hyps = [], []
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        X, lens = _pad([_prepare(items[i][0], None, pad_frames, hop_ms) for i in idx])
        logits, lo, _ = model.forward(X, lens)
        for j, i in enumerate(idx):
            ids = C.greedy_decode(logits[j, : lo[j]])
            hyps.append(" ".join("".join(bm_alphabet[k - 1] for k in ids if 1 <= k <= len(bm_alphabet)).split()))
            refs.append(items[i][1])
    return C.error_rate(refs, hyps), C.error_rate(refs, hyps, words=True)


def train_stt(cfg, lay, progress: Optional[Callable[[dict], None]] = None, should_stop: Optional[Callable[[], bool]] = None,
              source_dir: Optional[str] = None, pairs: Optional[List[Tuple[str, str]]] = None, seed: int = 0) -> dict:
    g = cfg.get
    say = progress or (lambda ev: None)
    t_start = time.time()
    stop = should_stop or (lambda: False)
    rng = np.random.default_rng(seed)

    # ---- data -----------------------------------------------------------------------------------------------------
    root = source_dir or lay.audio_data
    if pairs is None:
        if not os.path.isdir(root):
            raise TrainSTTError(f"there is no speech data folder at {root}. Create it and put a speech dataset there "
                                f"(audio files + transcripts; see docs/VOICE.md), then run this again.")
        pairs, rep = D.discover(root, g("voice.builtin_stt.max_utterances", None))
        say(dict(event="found", pairs=len(pairs), **rep))
        if not pairs:
            raise TrainSTTError(f"no (audio, transcript) pairs found in {root}. Found {rep['audio_files']} audio file(s); "
                                f"{rep['missing_audio']} metadata rows pointed at missing audio. Layouts understood: a metadata table with "
                                f"an audio column and a text column, or a .txt/.lab next to each clip (docs/VOICE.md). "
                                f"A dataset.json next to the table can name the columns.")
    n_mels = int(g("voice.builtin_stt.n_mels"))
    win_ms, hop_ms = float(g("voice.builtin_stt.win_ms")), float(g("voice.builtin_stt.hop_ms"))
    max_s = float(g("voice.builtin_stt.max_seconds"))
    floor_db = float(g("voice.builtin_stt.floor_db"))
    params = ("raw2", n_mels, win_ms, hop_ms, floor_db)
    pairs = sorted(pairs, key=lambda p: _hash(_stable_key(p[0], root)))
    cap = g("voice.builtin_stt.max_utterances", None)
    if cap:
        pairs = pairs[: int(cap)]

    feats: List[np.ndarray] = []
    texts: List[str] = []
    paths: List[str] = []
    skipped = Counter()
    frames_ms = hop_ms
    ram_budget = 0.4 * hw.ram_available_bytes()
    used = 0
    for i, (p, t) in enumerate(pairs):
        if stop():
            raise TrainSTTError("stopped while reading the data")
        f = _load_features(p, cfg, lay, params)
        if f is None:
            skipped["unreadable or too short"] += 1
            continue
        if len(f) * frames_ms / 1000.0 > max_s:
            skipped["longer than voice.builtin_stt.max_seconds"] += 1
            continue
        if used + f.nbytes > ram_budget:
            skipped["not enough memory (fewer clips used)"] += 1
            continue
        used += f.nbytes
        feats.append(f)
        texts.append(t)
        paths.append(p)
        if (i + 1) % 200 == 0:
            say(dict(event="features", done=i + 1, total=len(pairs)))
    if len(feats) < 4:
        raise TrainSTTError(f"only {len(feats)} usable clips; need at least a few dozen to learn anything ({dict(skipped)})")
    hours = sum(len(f) for f in feats) * frames_ms / 1000.0 / 3600.0
    say(dict(event="data", clips=len(feats), hours=round(hours, 3), skipped=dict(skipped)))

    # ---- alphabet / split -------------------------------------------------------------------------------------------
    counts = Counter("".join(texts))
    min_c = int(g("voice.builtin_stt.min_char_count"))
    alphabet = [" "] + [c for c, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if c != " " and n >= min_c]
    index = {c: i + 1 for i, c in enumerate(alphabet)}
    targets = [[index[c] for c in t if c in index] for t in texts]
    vf, vmin = float(g("voice.builtin_stt.val_fraction")), int(g("voice.builtin_stt.val_min"))
    order = sorted(range(len(feats)), key=lambda i: _hash("val:" + _stable_key(paths[i], root)))
    n_val = min(len(feats) // 2, max(vmin, int(round(vf * len(feats)))))
    val_idx, tr_idx = order[:n_val], order[n_val:]
    val_items = [(feats[i], "".join(c for c in texts[i] if c in index)) for i in val_idx]

    # ---- model ----------------------------------------------------------------------------------------------------
    hidden, layers = g("voice.builtin_stt.hidden", None), g("voice.builtin_stt.layers", None)
    why = "set explicitly"
    if hidden is None or layers is None:
        dh, dl, why = derive_size(hours)
        hidden, layers = hidden or dh, layers or dl
    cc = C.CTCConfig(n_mels, hidden, layers, len(alphabet) + 1, int(g("voice.builtin_stt.kernel")), g("voice.builtin_stt.dilations"))
    model = C.CTCModel(cc, rng)
    out_path = os.path.join(lay.voice_models, "stt.npz")
    resumed = False
    if os.path.exists(out_path):
        try:
            old = BuiltinModel.load(out_path)
            if old.alphabet == alphabet and old.model.cfg.to_dict() == cc.to_dict():
                model.p = {k: v.copy() for k, v in old.model.p.items()}
                resumed = True
        except Exception:                                                  # noqa: BLE001
            pass
    say(dict(event="model", params=model.num_params(), hidden=hidden, layers=layers, alphabet=len(alphabet), reason=why, resumed=resumed))

    # ---- training -------------------------------------------------------------------------------------------------
    batch = max(1, int(g("voice.builtin_stt.batch")))
    epochs = int(g("voice.builtin_stt.epochs"))
    budget = g("voice.builtin_stt.budget_minutes", None)
    budget_s = float(budget) * 60.0 if budget else None
    peak = float(g("voice.builtin_stt.lr"))
    clip = float(g("voice.builtin_stt.clip_norm"))
    patience = int(g("voice.builtin_stt.patience"))
    augment = bool(g("voice.builtin_stt.augment"))
    by_len = sorted(tr_idx, key=lambda i: len(feats[i]))
    batches_per_epoch = max(1, math.ceil(len(by_len) / batch))
    planned = epochs * batches_per_epoch
    opt = AdamW(model.p, beta1=0.9, beta2=0.98, weight_decay=0.01)
    warm = max(10, planned // 30)
    step = 0
    best = (float("inf"), float("inf"), None)
    bad = 0
    last_loss = float("nan")
    reason = "epochs completed"
    t_train = time.time()
    pad_s = float(g("voice.builtin_stt.pad_s"))
    pad_fr = int(round(pad_s * 1000.0 / hop_ms))
    meta_base = dict(n_mels=n_mels, win_ms=win_ms, hop_ms=hop_ms, floor_db=floor_db, pad_s=pad_s, max_seconds=max_s, sr=SR)
    if val_items:
        cer0, wer0 = evaluate(model, alphabet, val_items, pad_fr, hop_ms)
        say(dict(event="eval", epoch=0, cer=cer0, wer=wer0))
        if resumed:
            best = (cer0, wer0, {k: v.copy() for k, v in model.p.items()})
    try:
        for epoch in range(1, epochs + 1):
            chunks = [by_len[i:i + batch] for i in range(0, len(by_len), batch)]
            rng.shuffle(chunks)
            losses, nch = 0.0, 0
            for idx in chunks:
                if stop():
                    reason = "stopped"
                    raise KeyboardInterrupt
                el = time.time() - t_train
                if budget_s and el > budget_s:
                    reason = "time budget reached"
                    raise KeyboardInterrupt
                frac = max(step / max(1, planned), (el / budget_s) if budget_s else 0.0)
                lr = cosine_lr(peak, 0.05, warm, step, frac)
                fl = [_prepare(feats[i], rng if augment else None, pad_fr, hop_ms) for i in idx]
                X, lens = _pad(fl)
                logits, lo, cache = model.forward(X, lens, train=True)
                tg = [targets[i] for i in idx]
                loss, dl, ok, sk = C.batch_loss_grad(logits, lo, tg)
                step += 1
                if ok == 0:
                    continue
                nchars = max(1, sum(len(t) for t, d in zip(tg, dl) if np.any(d)))
                grads = model.backward((dl / nchars).astype(np.float32), cache)
                info = opt.step(model.p, grads, lr, clip)
                last_loss = loss / nchars
                if info["applied"]:
                    losses += last_loss
                    nch += 1
                if step % 20 == 0:
                    say(dict(event="step", step=step, planned=planned, epoch=epoch, loss=last_loss, lr=lr, seconds=time.time() - t_train))
            ep_loss = losses / max(1, nch)
            if val_items:
                cer, wer = evaluate(model, alphabet, val_items, pad_fr, hop_ms)
            else:
                cer, wer = ep_loss, ep_loss
            improved = cer < best[0] - 1e-4
            say(dict(event="eval", epoch=epoch, loss=ep_loss, cer=cer, wer=wer, improved=improved, seconds=time.time() - t_train))
            if improved:
                best, bad = (cer, wer, {k: v.copy() for k, v in model.p.items()}), 0
                _save(out_path, model, alphabet, meta_base, hours, len(feats), cer, wer, step)
            else:
                bad += 1
                if bad >= patience:
                    reason = f"no improvement for {patience} epochs"
                    break
    except KeyboardInterrupt:
        pass
    if best[2] is None:                                                   # never evaluated an improvement (e.g. stopped in epoch 1): keep what we have
        cer, wer = evaluate(model, alphabet, val_items, pad_fr, hop_ms) if val_items else (float("nan"), float("nan"))
        best = (cer, wer, {k: v.copy() for k, v in model.p.items()})
        _save(out_path, model, alphabet, meta_base, hours, len(feats), cer, wer, step)
    model.p = best[2]
    res = dict(path=out_path, clips=len(feats), val_clips=len(val_items), hours=round(hours, 3), params=model.num_params(), hidden=hidden, layers=layers,
               alphabet="".join(alphabet), steps=step, epochs_planned=epochs, reason=reason, val_cer=best[0], val_wer=best[1],
               seconds=round(time.time() - t_start, 1), resumed=resumed, skipped=dict(skipped), size_reason=why)
    fileio.write_json(os.path.join(lay.voice_state, "stt_training.json"), res, indent=2)
    say(dict(event="done", **{k: v for k, v in res.items() if k != "alphabet"}))
    return res


def _save(path: str, model: C.CTCModel, alphabet: List[str], meta_base: dict, hours: float, clips: int, cer: float, wer: float, step: int):
    meta = dict(meta_base, hours=round(hours, 3), clips=clips, val_cer=cer, val_wer=wer, steps=step, trained=time.time())
    BuiltinModel(model, alphabet, meta).save(path)
