"""
tasks.py — datasets of increasing difficulty + a curriculum controller.

v1 trained on one easy synthetic task and hit 100% by generation 57, so growth
had no reason to continue. Here difficulty is a dial:

  L0  4 clean-ish shape classes
  L1  more noise + random occlusion + 1px shifts
  L2  6 classes (adds `frame`, `checker`), heavier noise/occlusion/shifts
  L3  same, plus ADVERSARIAL examples (FGSM crafted with Zero White's own
      autograd against its current network) mixed into train and val.

The curriculum promotes the system only after it holds `promote_acc` for
`patience` generations, so capacity pressure appears when (and only when) the
current level is actually solved. Pure numpy, no external data.
"""

import numpy as np

CLASSES = ["cross", "ring", "stripe", "blob", "frame", "checker"]
N_CLASSES = len(CLASSES)
SIZE = 8

LEVELS = [
    dict(name="L0 clean",       n_classes=4, noise=0.15, occlude=0.0, shift=0, adv_eps=0.0, promote=0.95),
    dict(name="L1 noisy",       n_classes=4, noise=0.25, occlude=0.2, shift=1, adv_eps=0.0, promote=0.88),
    dict(name="L2 six-class",   n_classes=6, noise=0.30, occlude=0.2, shift=1, adv_eps=0.0, promote=0.72),
    dict(name="L3 adversarial", n_classes=6, noise=0.35, occlude=0.3, shift=1, adv_eps=0.20, promote=0.62),
]


def _draw(cls, rng, size=SIZE):
    g = np.zeros((size, size))
    ii, jj = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    if cls == "cross":
        r, c = rng.integers(2, size - 2), rng.integers(2, size - 2)
        g[r, :] = 1.0; g[:, c] = 1.0
    elif cls == "ring":
        rad = rng.uniform(2.0, 3.3)
        d = np.sqrt((ii - (size / 2 - 0.5)) ** 2 + (jj - (size / 2 - 0.5)) ** 2)
        g[np.abs(d - rad) < 0.7] = 1.0
    elif cls == "stripe":
        off = rng.integers(-2, 3)
        for i in range(size):
            for j in (i + off, i + off + 1):
                if 0 <= j < size:
                    g[i, j] = 1.0
    elif cls == "blob":
        cy, cx = rng.integers(1, size - 1), rng.integers(1, size - 1)
        g[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] = 1.0
    elif cls == "frame":
        k = rng.integers(0, 2)
        g[k, k:size - k] = 1.0; g[size - 1 - k, k:size - k] = 1.0
        g[k:size - k, k] = 1.0; g[k:size - k, size - 1 - k] = 1.0
    else:  # checker
        s = rng.integers(0, 2)
        g[((ii + jj + s) % 2 == 0) & (ii >= 2) & (ii < 6) & (jj >= 2) & (jj < 6)] = 1.0
    return g


def make_shapes(level=0, n_per_class=70, seed=0):
    cfg = LEVELS[level] if isinstance(level, int) else level
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label in range(cfg["n_classes"]):
        for _ in range(n_per_class):
            g = _draw(CLASSES[label], rng)
            if cfg["shift"]:
                g = np.roll(g, (rng.integers(-cfg["shift"], cfg["shift"] + 1),
                                rng.integers(-cfg["shift"], cfg["shift"] + 1)), axis=(0, 1))
            if rng.random() < cfg["occlude"]:
                r, c = rng.integers(0, SIZE - 2, 2)
                g[r:r + 2, c:c + 2] = 0.0
            g = g + rng.normal(0, cfg["noise"], g.shape)
            X.append(g.ravel()); y.append(label)
    X, y = np.array(X), np.array(y)
    p = rng.permutation(len(X))
    return X[p], y[p]


def fgsm(net, X, y, eps):
    """Fast Gradient Sign Method using the net's own autograd."""
    return X + eps * np.sign(net.input_gradient(X, y))


def split(X, y, frac=0.8):
    n = int(len(X) * frac)
    return X[:n], y[:n], X[n:], y[n:]


class Curriculum:
    def __init__(self, levels=LEVELS, promote_acc=None, patience=2,
                 n_per_class=250, seed=0, adv_refresh=8):
        self.levels, self.promote_acc, self.patience = levels, promote_acc, patience  # promote_acc overrides per-level thresholds
        self.n_per_class, self.seed, self.adv_refresh = n_per_class, seed, adv_refresh
        self.level, self.streak, self.gens_at_level = 0, 0, 0
        self._build()

    @property
    def cfg(self):
        return self.levels[self.level]

    @property
    def threshold(self):
        return self.promote_acc if self.promote_acc is not None else self.cfg["promote"]

    @property
    def finished(self):
        return self.level == len(self.levels) - 1 and self.streak >= self.patience

    def pin(self, level):
        """Hold the curriculum at one level for good (no promotion, never 'finished')."""
        self.level, self.streak, self.gens_at_level = level, 0, 0
        self.promote_acc = 2.0                  # unreachable accuracy -> never promotes
        self._build()

    def _build(self):
        X, y = make_shapes(self.level, self.n_per_class, seed=self.seed + 101 * self.level)
        self.Xtr, self.ytr, self.Xval, self.yval = split(X, y)
        self._clean = (self.Xtr, self.ytr, self.Xval, self.yval)

    def data(self):
        return self.Xtr, self.ytr, self.Xval, self.yval

    def refresh_adversarial(self, net):
        """Re-craft adversarial half of the data against the CURRENT network."""
        eps = self.cfg["adv_eps"]
        if not eps:
            return
        Xtr, ytr, Xval, yval = self._clean
        self.Xtr = np.vstack([Xtr, fgsm(net, Xtr, ytr, eps)]); self.ytr = np.concatenate([ytr, ytr])
        self.Xval = np.vstack([Xval, fgsm(net, Xval, yval, eps)]); self.yval = np.concatenate([yval, yval])

    def step(self, val_acc, net):
        """Report this generation's val accuracy. Returns True if promoted."""
        self.gens_at_level += 1
        self.streak = self.streak + 1 if (val_acc is not None and val_acc >= self.threshold) else 0
        if self.streak >= self.patience and self.level < len(self.levels) - 1:
            self.level += 1; self.streak = 0; self.gens_at_level = 0
            self._build(); self.refresh_adversarial(net)
            return True
        if self.cfg["adv_eps"] and self.gens_at_level % self.adv_refresh == 0:
            self.refresh_adversarial(net)
        return False

    def to_dict(self):
        return dict(level=self.level, streak=self.streak, gens_at_level=self.gens_at_level,
                    promote_acc=self.promote_acc, patience=self.patience,
                    n_per_class=self.n_per_class, seed=self.seed, adv_refresh=self.adv_refresh)

    @classmethod
    def from_dict(cls, d, net=None):
        c = cls(promote_acc=d["promote_acc"], patience=d["patience"], n_per_class=d["n_per_class"],
                seed=d["seed"], adv_refresh=d["adv_refresh"])
        c.level, c.streak, c.gens_at_level = d["level"], d["streak"], d["gens_at_level"]
        c._build()
        if net is not None:
            c.refresh_adversarial(net)
        return c
