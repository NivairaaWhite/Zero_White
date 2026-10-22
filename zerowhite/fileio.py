"""fileio.py - every disk write in Zero White goes through here.

* always UTF-8 (the Windows `cp1252` UnicodeEncodeError came from relying on
  the platform default encoding);
* atomic: data is written to a temp file and os.replace()d into place, so a
  crash or Ctrl-C mid-save can never leave a half-written checkpoint.
"""

import json
import os
import tempfile
import time

import numpy as np


def _tmp_beside(path):
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=os.path.splitext(path)[1])
    os.close(fd)
    return tmp


def _replace_with_retry(src, dst, tries=8):
    """os.replace, retried briefly: on Windows an antivirus scanner or the search indexer can hold the destination
    open for a few milliseconds, which surfaces as PermissionError (WinError 5/32). POSIX never needs the retry."""
    for k in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if os.name != "nt" or k == tries - 1:
                raise
            time.sleep(0.05 * (k + 1))


replace = _replace_with_retry            # public name: every module that swaps a file into place uses this one, not a bare os.replace


def write_bytes(path, data: bytes):
    tmp = _tmp_beside(path)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        _replace_with_retry(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_text(path, text: str):
    write_bytes(path, text.encode("utf-8"))


def read_text(path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path, obj, indent=None):
    write_text(path, json.dumps(obj, indent=indent, default=_json_default))


def read_json(path):
    return json.loads(read_text(path))


def save_npz(path, arrays: dict):
    tmp = _tmp_beside(path)
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        _replace_with_retry(tmp, path)               # was a bare os.replace: the checkpoint writer was the one place Windows could still raise PermissionError
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_npz(path):
    """Read every array from an .npz into a plain dict and close the file at once."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def rng_state(rng):
    return json.loads(json.dumps(rng.bit_generator.state, default=_json_default))


def rng_from_state(state):
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return rng


def write_png(path, img):
    """Write a 2-D uint8 grayscale array as a PNG (pure zlib/struct, no imaging library)."""
    import struct
    import zlib
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    write_bytes(path, png)


def image_grid(imgs, cols, pad=1, scale=4):
    """Tile images (N, h, w) in [0,1] into one uint8 grid, upscaled by `scale`."""
    imgs = np.clip(np.asarray(imgs), 0, 1)
    n, h, w = imgs.shape
    rows = int(np.ceil(n / cols))
    canvas = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad), 0.5)
    for i in range(n):
        r, c = divmod(i, cols)
        canvas[pad + r * (h + pad): pad + r * (h + pad) + h, pad + c * (w + pad): pad + c * (w + pad) + w] = imgs[i]
    return np.kron((canvas * 255).astype(np.uint8), np.ones((scale, scale), dtype=np.uint8))
