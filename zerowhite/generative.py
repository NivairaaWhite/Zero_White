"""
generative.py - generative models Zero White builds, trains and evolves on its
own autograd engine (no external ML library).

  CVAE      class-conditional variational autoencoder for images
            (digits / fashion / quickdraw / shapes): draw a "7", a "sneaker"...
  CharLM    character-level neural language model (embedding -> MLP over a
            context window): learns a text and writes more of it.

Both are scored on held-out data, and both are created by evolve.py's
factory: `evolve_generator` and `evolve_lm` search the model's own genome
(widths, depth, latent size, KL weight, learning rate...), let children
inherit trained weights when only training genes changed, and keep the best.

The CVAE is judged three ways, all on held-out data: ELBO (how well it models
the data), fidelity (Zero White's OWN classifier labels 40 samples of every
class; how often does the label match the request?) and diversity (do samples
vary as much as real data does, or has the latent collapsed to one blurry
average per class?).
"""

import copy
import json

import numpy as np

from . import fileio
from .engine import Tensor, embedding, bce_with_logits, softmax_cross_entropy, sigmoid_np
from .evolve import Evolver, Gene
from .nn import Params, Adam, init_mlp, mlp, mlp_raw


# =========================================================================== CVAE
class CVAE:
    kind = "cvae"
    WARMUP = 300          # KL weight ramps 0 -> beta over this many steps (avoids posterior collapse)

    def __init__(self, in_dim, n_classes, latent=8, hidden=96, depth=1, beta=1.0, lr=2e-3, seed=0):
        self.cfg = dict(in_dim=in_dim, n_classes=n_classes, latent=latent, hidden=hidden,
                        depth=depth, beta=beta, lr=lr)
        rng = np.random.default_rng(seed)
        self.rng = rng
        D, C, L, H = in_dim, n_classes, latent, hidden
        he = lambda i, o, s=1.0: rng.standard_normal((i, o)) * np.sqrt(2.0 / i) * s   # noqa: E731
        p = Params()
        p.add("eWx", he(D, H)).add("eWc", he(C, H, 0.5)).add("eb", np.zeros(H))
        p.add("dWz", he(L, H)).add("dWc", he(C, H, 0.5)).add("db", np.zeros(H))
        for i in range(1, depth):
            for side in ("e", "d"):
                p.add(f"{side}{i}W", he(H, H)).add(f"{side}{i}b", np.zeros(H))
        p.add("muW", he(H, L, 0.1)).add("mub", np.zeros(L))
        p.add("lvW", he(H, L, 0.1)).add("lvb", np.zeros(L))
        p.add("oW", he(H, D, 0.5)).add("ob", np.zeros(D))
        self.p, self.opt, self.steps_trained = p, Adam(lr=lr), 0

    # ---- model ---------------------------------------------------------------
    def _encode(self, l, x, oh):
        h = (x.matmul(l["eWx"]) + oh.matmul(l["eWc"]) + l["eb"]).relu()
        for i in range(1, self.cfg["depth"]):
            h = (h.matmul(l[f"e{i}W"]) + l[f"e{i}b"]).relu()
        return h.matmul(l["muW"]) + l["mub"], (h.matmul(l["lvW"]) + l["lvb"]).clip(-8.0, 6.0)

    def _decode(self, l, z, oh):
        h = (z.matmul(l["dWz"]) + oh.matmul(l["dWc"]) + l["db"]).relu()
        for i in range(1, self.cfg["depth"]):
            h = (h.matmul(l[f"d{i}W"]) + l[f"d{i}b"]).relu()
        return h.matmul(l["oW"]) + l["ob"]

    def _losses(self, l, X, y, eps):
        oh = Tensor(np.eye(self.cfg["n_classes"])[y])
        mu, lv = self._encode(l, Tensor(X), oh)
        z = mu + (lv * 0.5).exp() * Tensor(eps)
        rec = bce_with_logits(self._decode(l, z, oh), X)                    # nats / image
        kl = ((1.0 + lv - mu * mu - lv.exp()).sum(axis=1)).mean() * -0.5    # nats / image
        return rec, kl

    # ---- training / evaluation -------------------------------------------------
    def fit(self, X, y, steps, batch=64, rng=None):
        rng = rng or self.rng
        X = np.clip(X, 0.0, 1.0)
        last = None
        for _ in range(steps):
            idx = rng.integers(0, len(X), size=min(batch, len(X)))
            xb, yb = X[idx], y[idx]
            l = self.p.leaves()
            rec, kl = self._losses(l, xb, yb, rng.standard_normal((len(idx), self.cfg["latent"])))
            warm = min(1.0, (self.steps_trained + 1) / self.WARMUP)
            loss = rec + kl * (self.cfg["beta"] * warm)
            loss.backward()
            self.opt.step(self.p, l)
            self.steps_trained += 1
            last = float(loss.data)
        return last

    def elbo(self, X, y, seed=0):
        """True negative ELBO in nats per image (beta = 1, so it compares across genomes)."""
        X = np.clip(X, 0.0, 1.0)
        rng = np.random.default_rng(seed)
        with np.errstate(all="ignore"):
            rec, kl = self._losses(self.p.leaves(), X, y, rng.standard_normal((len(X), self.cfg["latent"])))
        return dict(rec=float(rec.data), kl=float(kl.data), elbo=float(rec.data + kl.data))

    def sample(self, y, rng=None, temperature=1.0):
        """Images (N, D) in [0,1] for the requested class ids."""
        rng = rng or self.rng
        y = np.asarray(y, dtype=np.int64)
        z = rng.standard_normal((len(y), self.cfg["latent"])) * temperature
        oh = np.eye(self.cfg["n_classes"])[y]
        a = self.p.a
        h = np.maximum(0.0, z @ a["dWz"] + oh @ a["dWc"] + a["db"])
        for i in range(1, self.cfg["depth"]):
            h = np.maximum(0.0, h @ a[f"d{i}W"] + a[f"d{i}b"])
        return sigmoid_np(h @ a["oW"] + a["ob"])

    # ---- lifecycle -------------------------------------------------------------
    def clone(self):
        c = copy.deepcopy(self)
        c.rng = np.random.default_rng(self.rng.integers(0, 2**31 - 1))
        return c

    def set_training(self, beta, lr):
        self.cfg["beta"], self.cfg["lr"], self.opt.lr = beta, lr, lr

    def genome(self):
        return {k: self.cfg[k] for k in ("latent", "hidden", "depth", "beta", "lr")}

    def save(self, path):
        arrays = {f"p.{k}": v for k, v in self.p.a.items()}
        arrays["__cfg__"] = np.frombuffer(json.dumps(dict(cfg=self.cfg, steps=self.steps_trained)).encode(), np.uint8)
        fileio.save_npz(path, arrays)

    @classmethod
    def load(cls, path):
        w = fileio.load_npz(path)
        meta = json.loads(bytes(w["__cfg__"]).decode("utf-8"))
        m = cls(**meta["cfg"])
        m.p = Params({k[2:]: w[k] for k in w if k.startswith("p.")})
        m.steps_trained = meta["steps"]
        return m


def fidelity(gen, predict, per_class=40, seed=0):
    """Fraction of class-conditioned samples that Zero White's classifier labels as the requested class."""
    y = np.repeat(np.arange(gen.cfg["n_classes"]), per_class)
    return float((predict(gen.sample(y, np.random.default_rng(seed))) == y).mean())


CVAE_SPACE = {"latent": Gene("int", 2, 32, shape=True), "hidden": Gene("int", 16, 256, shape=True),
              "depth": Gene("int", 1, 2, shape=True), "beta": Gene("log", 0.1, 2.0),
              "lr": Gene("log", 3e-4, 1e-2)}


def diversity(gen, Xval, yval, per_class=40, seed=0):
    """Generated within-class pixel variance / real within-class variance, capped at 1.
    ELBO alone rewards a partly collapsed latent (a VAE that ignores z makes one
    blurry 'average' per class); this term is what keeps the factory honest."""
    C = gen.cfg["n_classes"]
    real = [Xval[yval == c].var(axis=0).mean() for c in range(C) if (yval == c).sum() > 2]
    rng = np.random.default_rng(seed)
    fake = [gen.sample(np.full(per_class, c), rng).var(axis=0).mean() for c in range(C) if (yval == c).sum() > 2]
    return float(min(1.0, np.mean(fake) / (np.mean(real) + 1e-12)))


def score_generator(model, task, predict, val_cap=1000):
    """fitness = fidelity/4 + diversity/4 - ELBO per pixel, all measured on held-out data."""
    Xv, yv = np.clip(task.Xval[:val_cap], 0, 1), task.yval[:val_cap]
    e = model.elbo(Xv, yv)
    fid, div = fidelity(model, predict), diversity(model, Xv, yv)
    return 0.25 * fid + 0.25 * div - e["elbo"] / model.cfg["in_dim"], dict(
        elbo=e["elbo"], rec=e["rec"], kl=e["kl"], fidelity=fid, diversity=div,
        params=model.p.count(), steps=model.steps_trained)


def evolve_generator(task, predict, generations=3, population=4, steps=400, seed=0,
                     incumbent=None, verbose=True):
    """Create (or improve) a class-conditional image generator for `task`.
    `predict` is Zero White's classifier for this task (X -> class ids).
    `incumbent` = an existing CVAE to keep improving. Returns the Evolver."""
    rng_i = [0]

    def evaluate(genome, parent):
        if parent is not None:                       # training-genes-only child: inherit weights
            m = parent.clone()
            m.set_training(genome["beta"], genome["lr"])
        else:
            m = CVAE(task.in_dim, task.n_classes, seed=seed + rng_i[0], **genome)
        rng_i[0] += 1
        m.fit(task.Xtr, task.ytr, steps, rng=np.random.default_rng(seed * 1000 + rng_i[0]))
        fit, info = score_generator(m, task, predict)
        return fit, m, info

    ev = Evolver(CVAE_SPACE, evaluate, population=population, mu=2, seed=seed)
    if incumbent is not None:
        fit, info = score_generator(incumbent, task, predict)
        ev.inject(incumbent.genome(), fit, incumbent, info)
    # no incumbent: no default generator either, generation 1 is random genomes
    ev.run(max(1, generations), verbose, label=f"cvae/{task.name}")
    return ev


# =========================================================================== text
class CharVocab:
    UNK = 0

    def __init__(self, chars):
        self.chars = list(chars)
        self.index = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text, max_vocab=128):
        from collections import Counter
        common = [c for c, _ in Counter(text).most_common(max_vocab - 1)]
        return cls(["\x00"] + sorted(common))

    def __len__(self):
        return len(self.chars)

    def encode(self, text):
        return np.array([self.index.get(c, self.UNK) for c in text], dtype=np.int64)

    def decode(self, ids):
        return "".join("?" if i == self.UNK else self.chars[i] for i in ids)


class CharLM:
    kind = "charlm"

    def __init__(self, vocab_size, ctx=8, emb=16, hidden=128, depth=1, lr=3e-3, seed=0):
        self.cfg = dict(vocab_size=vocab_size, ctx=ctx, emb=emb, hidden=hidden, depth=depth, lr=lr)
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.p = Params()
        self.p.add("E", rng.standard_normal((vocab_size, emb)) * 0.1)
        init_mlp(self.p, "lm", [ctx * emb] + [hidden] * depth + [vocab_size], rng, out_scale=0.5)
        self.opt, self.steps_trained = Adam(lr=lr), 0

    def _batch(self, tokens, rng, batch):
        c = self.cfg["ctx"]
        pos = rng.integers(c, len(tokens), size=batch)
        return np.stack([tokens[p - c:p] for p in pos]), tokens[pos]

    def fit(self, tokens, steps, batch=64, rng=None):
        rng = rng or self.rng
        last = None
        for _ in range(steps):
            xb, yb = self._batch(tokens, rng, batch)
            l = self.p.leaves()
            h = embedding(l["E"], xb).reshape(len(xb), self.cfg["ctx"] * self.cfg["emb"])
            loss, _ = softmax_cross_entropy(mlp(l, "lm", h), yb)
            loss.backward()
            self.opt.step(self.p, l)
            self.steps_trained += 1
            last = float(loss.data)
        return last

    def _logits_raw(self, xb):
        e = self.p.a["E"][xb].reshape(len(xb), -1)
        return mlp_raw(self.p.a, "lm", e)

    def bits_per_char(self, tokens, n=4096, seed=0):
        rng = np.random.default_rng(seed)
        xb, yb = self._batch(tokens, rng, min(n, len(tokens) - self.cfg["ctx"] - 1))
        with np.errstate(all="ignore"):
            z = self._logits_raw(xb)
            z = z - z.max(axis=1, keepdims=True)
            nll = -(z[np.arange(len(yb)), yb] - np.log(np.exp(z).sum(axis=1)))
        return float(nll.mean() / np.log(2.0))

    def generate(self, vocab, prompt="", n=200, temperature=0.8, rng=None):
        rng = rng or self.rng
        c = self.cfg["ctx"]
        pad = vocab.index.get(" ", 1)
        ids = list(vocab.encode(prompt)) if prompt else []
        window = ([pad] * c + ids)[-c:]
        out = []
        for _ in range(n):
            z = self._logits_raw(np.array([window]))[0] / max(temperature, 1e-3)
            z[vocab.UNK] = -1e9                                  # never emit <unk>
            p = np.exp(z - z.max())
            nxt = int(rng.choice(len(p), p=p / p.sum()))
            out.append(nxt)
            window = window[1:] + [nxt]
        return prompt + vocab.decode(out)

    def clone(self):
        c = copy.deepcopy(self)
        c.rng = np.random.default_rng(self.rng.integers(0, 2**31 - 1))
        return c

    def set_training(self, lr):
        self.cfg["lr"] = self.opt.lr = lr

    def genome(self):
        return {k: self.cfg[k] for k in ("ctx", "emb", "hidden", "depth", "lr")}

    def save(self, path, vocab):
        arrays = {f"p.{k}": v for k, v in self.p.a.items()}
        arrays["__cfg__"] = np.frombuffer(json.dumps(dict(cfg=self.cfg, steps=self.steps_trained,
                                                          vocab=vocab.chars)).encode("utf-8"), np.uint8)
        fileio.save_npz(path, arrays)

    @classmethod
    def load(cls, path):
        w = fileio.load_npz(path)
        meta = json.loads(bytes(w["__cfg__"]).decode("utf-8"))
        m = cls(**meta["cfg"])
        m.p = Params({k[2:]: w[k] for k in w if k.startswith("p.")})
        m.steps_trained = meta["steps"]
        return m, CharVocab(meta["vocab"])


LM_SPACE = {"ctx": Gene("int", 4, 24, shape=True), "emb": Gene("int", 8, 48, shape=True),
            "hidden": Gene("int", 32, 256, shape=True), "depth": Gene("int", 1, 2, shape=True),
            "lr": Gene("log", 5e-4, 1e-2)}


def split_text_tokens(tokens, val_frac=0.1, block=400):
    """Train/validation split by BLOCK, assigned by a fixed hash of the block index.
    Appending new text later never moves an old block from one side to the other, so a
    model that kept training on yesterday's text is never scored on text it has seen."""
    tokens = np.asarray(tokens)
    n_blocks = len(tokens) // block
    if n_blocks >= 10:
        every = max(2, int(round(1 / val_frac)))
        is_val = np.array([(i * 2654435761) % every == 0 for i in range(n_blocks)])
        parts = [tokens[i * block:(i + 1) * block] for i in range(n_blocks)]
        parts.append(tokens[n_blocks * block:])                      # remainder trains
        val = [p for p, v in zip(parts, list(is_val) + [False]) if v]
        tr = [p for p, v in zip(parts, list(is_val) + [False]) if not v]
        if val and tr:
            return np.concatenate(tr), np.concatenate(val)
    cut = int(len(tokens) * (1 - val_frac))
    return tokens[:cut], tokens[cut:]


def evolve_lm(train_tokens, val_tokens, vocab, generations=3, population=4, steps=600, seed=0,
              incumbent=None, verbose=True):
    """Create (or improve) a character language model. Fitness = - held-out bits per character."""
    n = [0]

    def evaluate(genome, parent):
        if parent is not None:
            m = parent.clone()
            m.set_training(genome["lr"])
        else:
            m = CharLM(len(vocab), seed=seed + n[0], **genome)
        n[0] += 1
        m.fit(train_tokens, steps, rng=np.random.default_rng(seed * 1000 + n[0]))
        bpc = m.bits_per_char(val_tokens)
        return -bpc, m, dict(bpc=bpc, params=m.p.count(), steps=m.steps_trained)

    ev = Evolver(LM_SPACE, evaluate, population=population, mu=2, seed=seed)
    if incumbent is not None:
        bpc = incumbent.bits_per_char(val_tokens)
        ev.inject(incumbent.genome(), -bpc, incumbent, dict(bpc=bpc, params=incumbent.p.count(),
                                                            steps=incumbent.steps_trained))
    ev.run(max(1, generations), verbose, label="charlm")
    return ev
