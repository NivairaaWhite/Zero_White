"""
datasets.py - real datasets for Zero White, loaded with the standard library
and numpy only (urllib, zipfile, gzip, struct).

  digits     UCI Optical Recognition of Handwritten Digits (8x8, 10 classes)
  sonar      UCI Sonar, Mines vs. Rocks (60 features, 2 classes)
  fashion    Fashion-MNIST (28x28, 10 classes)
  quickdraw  Quick, Draw! numpy bitmaps (28x28, any class names you choose)
  shapes     the built-in synthetic shape curriculum levels (tasks.py)

Every loader returns a Task with THREE splits: train (learning), val (what
self-modification selection looks at) and test (never looked at by anything
that selects or tunes; only reported).

Downloads are cached under $ZERO_WHITE_DATA (default ~/.zero_white/data).
Each dataset lists several sources, tried in order. Nothing here is fetched
unless you call a loader or `python3 app.py data fetch <name>`.
"""

import gzip
import io
import os
import struct
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field

import numpy as np

from . import fileio

HERE = os.path.dirname(os.path.abspath(__file__))
_FASHION_FILES = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                  "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")

# Source lists are module-level so they can be overridden (tests point them at file:// URLs).
SOURCES = {
    "digits_zip": ["https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip"],
    "sonar": ["https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",
              "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"],
    "fashion": ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
                "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"],
    "quickdraw": ["https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"],
}
QUICKDRAW_DEFAULT = ("circle", "square", "star", "zigzag")


class DatasetUnavailable(RuntimeError):
    pass


def data_dir():
    d = os.environ.get("ZERO_WHITE_DATA") or os.path.join(os.path.expanduser("~"), ".zero_white", "data")
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- Task
@dataclass
class Task:
    name: str
    Xtr: np.ndarray
    ytr: np.ndarray
    Xval: np.ndarray
    yval: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    n_classes: int
    num_chunks: int
    image_shape: tuple = None            # (h, w) for image tasks, else None
    class_names: tuple = ()
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def in_dim(self):
        return self.Xtr.shape[1]

    @property
    def is_image(self):
        return self.image_shape is not None

    def summary(self):
        return (f"{self.name}: train={len(self.Xtr)} val={len(self.Xval)} test={len(self.Xte)} "
                f"dim={self.in_dim} classes={self.n_classes} source={self.source}")


def best_chunks(dim):
    """Divisor of `dim` closest to sqrt(dim): image rows for squares, sensible groups otherwise."""
    target = np.sqrt(dim)
    divs = [d for d in range(2, dim) if dim % d == 0] or [1]
    return int(min(divs, key=lambda d: abs(d - target)))


def _stratified_indices(y, fracs, rng):
    """Split indices per class into len(fracs) parts (fractions sum to 1)."""
    parts = [[] for _ in fracs]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        cuts = (np.cumsum(fracs)[:-1] * len(idx)).astype(int)
        for p, chunk in zip(parts, np.split(idx, cuts)):
            p.extend(chunk.tolist())
    return [rng.permutation(np.array(p, dtype=int)) for p in parts]


def make_task(name, X, y, *, seed=0, test=None, val_frac=0.2, test_frac=0.2, standardize=False,
              image_shape=None, class_names=(), source="", max_train=None):
    """Build a Task. `test=(Xte, yte)` supplies an official test split; otherwise one is carved out."""
    rng = np.random.default_rng(seed)
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)
    if test is not None:
        Xte, yte = np.asarray(test[0], dtype=np.float64), np.asarray(test[1], dtype=np.int64)
        tr_i, va_i = _stratified_indices(y, [1 - val_frac, val_frac], rng)
    else:
        tr_i, va_i, te_i = _stratified_indices(y, [1 - val_frac - test_frac, val_frac, test_frac], rng)
        Xte, yte = X[te_i], y[te_i]
    if max_train and len(tr_i) > max_train:
        tr_i = tr_i[:max_train]
    Xtr, ytr, Xval, yval = X[tr_i], y[tr_i], X[va_i], y[va_i]
    if standardize:
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd[sd < 1e-8] = 1.0
        Xtr, Xval, Xte = (Xtr - mu) / sd, (Xval - mu) / sd, (Xte - mu) / sd
    return Task(name, Xtr, ytr, Xval, yval, Xte, yte, int(max(y.max(), yte.max())) + 1,
                best_chunks(Xtr.shape[1]), image_shape, tuple(class_names), source)


# --------------------------------------------------------------------------- downloading
def _open(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "zero-white/3.0"})
    return urllib.request.urlopen(req, timeout=timeout)


def _download(urls, dest, timeout=60):
    """Try each URL in order; cache bytes at `dest`."""
    if os.path.exists(dest):
        return dest
    errors = []
    for u in urls:
        try:
            with _open(u, timeout) as r:
                fileio.write_bytes(dest, r.read())
            return dest
        except Exception as e:      # noqa: BLE001 - report every source's failure together
            errors.append(f"  {u}\n    -> {type(e).__name__}: {e}")
    raise DatasetUnavailable(
        f"could not download {os.path.basename(dest)}:\n" + "\n".join(errors) +
        f"\nIf you are offline or behind a firewall, download it by hand and place it at:\n  {dest}")


def _join(base, name):
    return base + name if base.endswith("/") else base.rstrip("/") + "/" + name


# --------------------------------------------------------------------------- parsers
def parse_idx(raw: bytes) -> np.ndarray:
    """Parse the IDX format used by MNIST / Fashion-MNIST (uint8 payloads)."""
    if len(raw) < 4 or raw[0] != 0 or raw[1] != 0:
        raise ValueError("not an IDX file (bad magic)")
    dtype_code, ndim = raw[2], raw[3]
    if dtype_code != 0x08:
        raise ValueError(f"unsupported IDX dtype 0x{dtype_code:02x} (only uint8)")
    dims = struct.unpack(">" + "I" * ndim, raw[4:4 + 4 * ndim])
    payload = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    if payload.size != int(np.prod(dims)):
        raise ValueError(f"IDX size mismatch: header says {dims}, payload has {payload.size} bytes")
    return payload.reshape(dims)


def parse_csv_rows(text, label_map=None):
    """Rows of comma-separated numbers with the label in the LAST column
    (digits: integer; sonar: 'R'/'M' text -> 0/1 via label_map)."""
    X, y = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        lab = parts[-1].strip()
        try:
            X.append([float(v) for v in parts[:-1]])
        except ValueError:
            continue                                   # header line
        y.append(label_map[lab] if label_map else int(float(lab)))
    if not X:
        raise ValueError("no data rows found")
    return np.array(X), np.array(y)


def _npy_prefix(stream, rows, width=784):
    """Read only the first `rows` rows of an .npy stream (Quick, Draw! files are ~100 MB each)."""
    head = stream.read(10)
    if head[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    hlen = struct.unpack("<H", head[8:10])[0] if head[6] == 1 else struct.unpack("<I", head[8:12])[0]
    rest = stream.read(hlen + (0 if head[6] == 1 else 2))     # remainder of the header
    buf = io.BytesIO(head + rest)
    np.lib.format.read_magic(buf)
    shape, _, dtype = (np.lib.format.read_array_header_1_0(buf) if head[6] == 1
                             else np.lib.format.read_array_header_2_0(buf))
    n = min(rows, shape[0])
    data = stream.read(n * width * dtype.itemsize)
    arr = np.frombuffer(data, dtype=dtype)
    n = arr.size // width
    return arr[:n * width].reshape(n, width)


# --------------------------------------------------------------------------- loaders
def load_digits(seed=0, download=True, allow_bundled=True):
    """Full UCI optdigits if reachable (official train/test split), else the bundled
    1,797-sample real subset."""
    path = os.path.join(data_dir(), "optdigits.zip")
    try:
        if download or os.path.exists(path):
            _download(SOURCES["digits_zip"], path)
            with zipfile.ZipFile(path) as z:
                names = {os.path.basename(n): n for n in z.namelist()}
                tra = next(n for b, n in names.items() if b.endswith(".tra"))
                tes = next(n for b, n in names.items() if b.endswith(".tes"))
                X, y = parse_csv_rows(z.read(tra).decode("utf-8", "replace"))
                Xt, yt = parse_csv_rows(z.read(tes).decode("utf-8", "replace"))
            return make_task("digits", X / 16.0, y, seed=seed, test=(Xt / 16.0, yt), image_shape=(8, 8),
                             class_names=tuple(map(str, range(10))), source="UCI optdigits (full, official split)")
    except (DatasetUnavailable, zipfile.BadZipFile, StopIteration, ValueError) as e:
        if not allow_bundled:
            raise
        if os.path.exists(path) and not isinstance(e, DatasetUnavailable):
            os.remove(path)                              # corrupt cache: drop it
        note = str(e).splitlines()[0]
        print(f"[datasets] full digits unavailable ({note}); using the bundled real 1,797-sample subset")
    with gzip.open(os.path.join(HERE, "data", "optdigits_uci_sample.csv.gz"), "rt", encoding="utf-8") as f:
        X, y = parse_csv_rows(f.read())
    return make_task("digits", X / 16.0, y, seed=seed, image_shape=(8, 8),
                     class_names=tuple(map(str, range(10))), source="UCI optdigits (bundled 1,797-sample subset)")


def load_sonar(seed=0):
    path = _download(SOURCES["sonar"], os.path.join(data_dir(), "sonar.all-data"))
    X, y = parse_csv_rows(fileio.read_text(path), label_map={"R": 0, "M": 1})
    if X.shape[1] != 60:
        raise DatasetUnavailable(f"sonar file has {X.shape[1]} features, expected 60: {path}")
    return make_task("sonar", X, y, seed=seed, standardize=True, class_names=("rock", "mine"),
                     source="UCI Sonar, Mines vs. Rocks")


def load_fashion(seed=0, max_train=None):
    arrays = {}
    for fn in _FASHION_FILES:
        path = os.path.join(data_dir(), fn)
        _download([_join(b, fn) for b in SOURCES["fashion"]], path)
        try:
            with gzip.open(path, "rb") as f:
                arrays[fn] = parse_idx(f.read())
        except (OSError, ValueError) as e:
            os.remove(path)
            raise DatasetUnavailable(f"{fn} was corrupt ({e}); deleted - run again to re-download") from e
    Xtr = arrays["train-images-idx3-ubyte.gz"].reshape(-1, 784) / 255.0
    Xte = arrays["t10k-images-idx3-ubyte.gz"].reshape(-1, 784) / 255.0
    names = ("t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot")
    return make_task("fashion", Xtr, arrays["train-labels-idx1-ubyte.gz"], seed=seed,
                     test=(Xte, arrays["t10k-labels-idx1-ubyte.gz"]), val_frac=0.1, image_shape=(28, 28),
                     class_names=names, source="Fashion-MNIST (Zalando Research)", max_train=max_train)


def load_quickdraw(classes=QUICKDRAW_DEFAULT, per_class=3000, seed=0):
    """Download only the first `per_class` drawings of each class (not the ~100 MB files)."""
    Xs, ys = [], []
    for label, name in enumerate(classes):
        cache = os.path.join(data_dir(), f"quickdraw_{name.replace(' ', '_')}_{per_class}.npy")
        if not os.path.exists(cache):
            errors = []
            for base in SOURCES["quickdraw"]:
                url = _join(base, urllib.parse.quote(name) + ".npy")
                try:
                    with _open(url, 120) as r:
                        arr = _npy_prefix(r, per_class)
                    buf = io.BytesIO()
                    np.save(buf, arr)
                    fileio.write_bytes(cache, buf.getvalue())
                    break
                except Exception as e:    # noqa: BLE001
                    errors.append(f"  {url}\n    -> {type(e).__name__}: {e}")
            else:
                raise DatasetUnavailable(f"could not fetch Quick, Draw! class '{name}':\n" + "\n".join(errors))
        arr = np.load(cache)
        Xs.append(arr / 255.0)
        ys.append(np.full(len(arr), label))
    return make_task("quickdraw", np.vstack(Xs), np.concatenate(ys), seed=seed, image_shape=(28, 28),
                     class_names=tuple(classes), source="Quick, Draw! (Google Creative Lab)")


def load_shapes(level=0, n_per_class=250, seed=0):
    """The synthetic shape curriculum as a 3-split Task."""
    from .tasks import make_shapes, LEVELS
    X, y = make_shapes(level, n_per_class, seed=seed)
    t = make_task(f"shapes{level}", X, y, seed=seed, image_shape=(8, 8), source=f"synthetic {LEVELS[level]['name']}")
    t.num_chunks = 8
    return t


LOADERS = {"digits": load_digits, "sonar": load_sonar, "fashion": load_fashion,
           "quickdraw": load_quickdraw, "shapes": load_shapes}


def load(name, **kw):
    if name not in LOADERS:
        raise KeyError(f"unknown dataset '{name}' (choose from {', '.join(LOADERS)})")
    return LOADERS[name](**kw)
