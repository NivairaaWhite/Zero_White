"""task.py - the interface between the outside world and the Generalist.

Four kinds of task, each carrying everything needed to *score* an agent honestly:

  SupervisedTask   tabular / image data with train / val / test splits  (classification | regression)
  EnvTask          an environment: reset(rng) -> obs, step(a) -> (obs, reward, done, success)
  TextTask         a corpus to learn to continue
  ImageGenTask     an image dataset to learn to draw from (judged by a classifier it also builds)

Selection uses validation data / selection episodes; reported numbers use test data / fresh episodes.
"""

import csv
import importlib.util
import os
from dataclasses import dataclass

import numpy as np

from .. import datasets


# --------------------------------------------------------------------------- supervised
@dataclass
class SupervisedTask(datasets.Task):
    kind: str = "classification"          # or "regression"

    @property
    def family_kind(self):
        return self.kind

    def signature(self):
        return dict(kind=self.kind, dim=int(self.in_dim), classes=int(self.n_classes), n=int(len(self.Xtr)),
                    image=bool(self.is_image))

    def chance(self):
        if self.kind == "classification":
            return float(np.bincount(self.ytr, minlength=self.n_classes).max() / len(self.ytr))
        return float(np.var(self.yte))


def from_dataset_task(t):
    return SupervisedTask(**{**t.__dict__, "kind": "classification"})


def from_arrays(name, X, y, kind, seed=0, standardize=True):
    """Split arrays 60/20/20 and wrap them. Classification labels must be 0..C-1 integers."""
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if kind == "classification":
        t = datasets.make_task(name, X, np.asarray(y).astype(np.int64), seed=seed, standardize=standardize,
                               source="user arrays")
        return SupervisedTask(**{**t.__dict__, "kind": "classification"})
    y = np.asarray(y, dtype=np.float64)
    idx = rng.permutation(len(X))
    n = len(X)
    a, b = int(0.6 * n), int(0.8 * n)
    tr, va, te = idx[:a], idx[a:b], idx[b:]
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    if standardize:
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd < 1e-8] = 1.0
        Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    return SupervisedTask(name, Xtr, y[tr], Xva, y[va], Xte, y[te], 0, datasets.best_chunks(X.shape[1]), None, (),
                          "user arrays", {}, "regression")


def from_csv(path, target=None, kind=None, seed=0):
    """Numeric CSV with a header row. `target` = column name (default: the last column).
    kind=None -> classification when the target has <= 20 distinct integer values, else regression."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], [r for r in rows[1:] if any(c.strip() for c in r)]
    ti = head.index(target) if target else len(head) - 1
    feats = [i for i in range(len(head)) if i != ti]
    X = np.array([[float(r[i]) for i in feats] for r in body])
    ycol = np.array([r[ti] for r in body])
    try:
        y = ycol.astype(float)
        distinct = np.unique(y)
        is_int = np.allclose(y, np.round(y))
        kind = kind or ("classification" if is_int and len(distinct) <= 20 else "regression")
        if kind == "classification":
            _, y = np.unique(y, return_inverse=True)
    except ValueError:                                           # non-numeric labels
        kind = kind or "classification"
        names, y = np.unique(ycol, return_inverse=True)
    name = os.path.splitext(os.path.basename(path))[0]
    return from_arrays(name, X, y, kind, seed=seed)


# --------------------------------------------------------------------------- environments
class EnvTask:
    kind = "env"

    def __init__(self, env, name=None, max_return=None):
        self.env = env
        self.name = name or getattr(env, "name", type(env).__name__.lower())
        self.max_return = max_return if max_return is not None else getattr(env, "max_return", None)
        self.obs_dim, self.n_actions = int(env.obs_dim), int(env.n_actions)
        self._rand = None

    def signature(self):
        return dict(kind="env", dim=self.obs_dim, classes=self.n_actions, n=0, image=False)

    def rollout(self, act_fn, seed):
        rng = np.random.default_rng(seed)
        o, done, total, won = self.env.reset(rng), False, 0.0, False
        while not done:
            o, r, done, won = self.env.step(act_fn(o))
            total += r
        return total, bool(won)

    def random_baseline(self, episodes=30):
        if self._rand is None:
            rng = np.random.default_rng(7)
            rets = [self.rollout(lambda o: int(rng.integers(self.n_actions)), 5000 + i)[0] for i in range(episodes)]
            self._rand = (float(np.mean(rets)), float(np.std(rets)))
        return self._rand

    def reference(self):
        """Return that counts as 'solved' for skill normalisation. Used to compare candidates on one task and as a
        rough cross-task signal for the prior; if the env does not declare max_return it is a documented guess."""
        if self.max_return is not None:
            return float(self.max_return)
        m, s = self.random_baseline()
        return m + max(10.0 * s, 0.5 * abs(m) + 1.0)

    def skill(self, mean_return):
        m, _ = self.random_baseline()
        ref = self.reference()
        return float(np.clip((mean_return - m) / max(1e-9, ref - m), -1.0, 1.0))

    def evaluate_fn(self, act_fn, episodes=20, seed0=12345):
        rets, wins = zip(*[self.rollout(act_fn, seed0 + i) for i in range(episodes)])
        return float(np.mean(rets)), float(np.mean(wins))


def load_env_file(path):
    """Load a user environment from a .py file that defines `class Env` with obs_dim, n_actions,
    reset(rng) -> obs and step(a) -> (obs, reward, done, success). NOTE: this executes your file,
    in this process, with your permissions; it is YOUR code, not evolved code."""
    spec = importlib.util.spec_from_file_location("zw_user_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = mod.Env()
    return EnvTask(env, name=os.path.splitext(os.path.basename(path))[0], max_return=getattr(mod, "MAX_RETURN", None))


def from_env_name(name):
    from ..agents import make_env
    return EnvTask(make_env(name), name=name)


# --------------------------------------------------------------------------- text / image generation
class TextTask:
    kind = "text"

    def __init__(self, name, text):
        from ..generative import CharVocab, split_text_tokens
        self.name, self.text = name, text
        self.vocab = CharVocab.from_text(text)
        toks = self.vocab.encode(text)
        if len(toks) < 500:
            raise ValueError(f"need at least 500 characters of text, got {len(toks)}")
        self.train, self.val = split_text_tokens(toks)

    def signature(self):
        return dict(kind="text", dim=len(self.vocab), classes=0, n=int(len(self.train)), image=False)

    def chance(self):
        return float(np.log2(len(self.vocab)))              # uniform bits per char


class ImageGenTask:
    kind = "image_gen"

    def __init__(self, supervised):
        if not supervised.is_image:
            raise ValueError("image generation needs an image dataset")
        self.sup, self.name = supervised, supervised.name
        self.classifier = None                                 # set by the Generalist (it builds one first)

    def signature(self):
        s = self.sup.signature()
        return dict(s, kind="image_gen")
