"""synth.py - a generator of messy delimited files WITH their true format, for training and testing the learned sniffer (learned.py).

Every file comes with its labels: delimiter, header present, decimal separator, text encoding, plus which of those the text actually determines
(`ident`): a table with no fractional number does not reveal its decimal separator, a table with no numeric column does not reveal whether its
first row is a header, a pure-ASCII file does not reveal whether it was saved as UTF-8 or Latin-1. Scoring only on what is identifiable keeps a
model (or a rule) from being blamed for a question the file cannot answer.

The families deliberately differ from each other (clean vs `hard`: a preamble, ragged rows, a cut-off tail, CRLF) and are drawn from the vocabulary
below: this is a SYNTHETIC distribution written by the same author as the model, so accuracy here says the model has learned this distribution,
not that it will meet real-world files as well. See docs/OMNIVORE.md.
"""
from __future__ import annotations

import numpy as np

WORDS = ["alpha", "beta", "gamma", "delta", "red", "green", "blue", "north", "south", "east", "west", "yes", "no", "low", "high", "cat", "dog", "mouse",
         "café", "naïve", "São", "Zürich", "señor", "Ångström", "über", "jalapeño", "Müller", "Çelik", "Łódź", "crème"]
NAMES = ["id", "age", "income", "score", "weight", "height", "price", "qty", "region", "status", "label", "city", "temp", "speed", "rating", "name",
         "size", "count", "dist", "amount"]
DELIMS = [",", ";", "\t", "|", " "]
CLASSES_ENC = ["utf-8/ascii", "single-byte", "utf-16-le", "utf-16-be"]


def _num(v, decimal, thousands, rng):
    s = f"{v:.{int(rng.integers(1, 4))}f}" if rng.random() < 0.6 else f"{v:.2f}"
    if thousands and abs(v) >= 1000:
        ip, _, fp = s.partition(".")
        ip = f"{int(ip):,}"
        sep_t, sep_d = ("." if decimal == "," else ","), decimal
        return ip.replace(",", sep_t) + (sep_d + fp if fp else "")
    return s.replace(".", decimal)


_NAME_STYLES = (lambda w: w, lambda w: w.title(), lambda w: w.upper(), lambda w: w + "_id", lambda w: w.replace("a", "A", 1), lambda w: f"{w} (x)",
                lambda w: f"{w}Count", lambda w: f"{w}_{w[:2]}")


def _cell(kind, decimal, rng, delim):
    if kind == "int":
        return str(int(rng.integers(0, 5000)))
    if kind == "neg":
        return _num(float(rng.normal(0, 40)), decimal, False, rng)
    if kind == "float":
        return _num(float(rng.normal(50, 30)), decimal, False, rng)
    if kind == "money":
        return _num(float(abs(rng.normal(20000, 15000))), decimal, True, rng)
    if kind == "pct":
        return _num(float(rng.uniform(0, 100)), decimal, False, rng) + "%"
    if kind == "sci":
        return f"{float(rng.normal(0, 1)):.3e}".replace(".", decimal)
    if kind == "date":
        y, m, d = int(rng.integers(1990, 2025)), int(rng.integers(1, 13)), int(rng.integers(1, 29))
        return rng.choice([f"{y}-{m:02d}-{d:02d}", f"{d:02d}.{m:02d}.{y}", f"{m}/{d}/{y}"])
    if kind == "bool":
        return str(rng.choice(["yes", "no", "true", "false"]))
    w = WORDS[int(rng.integers(len(WORDS)))]
    if rng.random() < 0.15:
        w = w + " " + WORDS[int(rng.integers(len(WORDS)))]            # a two-word value
    if delim == " ":
        w = w.replace(" ", "_")
    elif rng.random() < 0.1 and delim in (",", ";", "|", "\t"):
        w = f'"{w}{delim}{rng.choice(WORDS)}"'                            # a quoted value containing the delimiter
    return w


def make(rng: np.random.Generator, hard: bool = False):
    """(bytes, labels) with labels = dict(delim, decimal, header, enc, ident=dict(decimal, header, enc), text)."""
    ncol = int(rng.integers(2, 13))
    nrow = int(rng.integers(4, 150))
    delim = DELIMS[int(rng.integers(len(DELIMS)))]
    decimal = "," if rng.random() < 0.35 else "."
    if delim == "," and decimal == ",":
        decimal = "."                                              # a comma cannot be both the delimiter and the decimal mark
    header = bool(rng.random() < 0.7)
    enc = ["utf-8", "latin-1", "cp1252", "utf-16-le", "utf-16-be"][int(rng.choice(5, p=[.45, .2, .1, .125, .125]))]
    kinds = [str(rng.choice(["int", "float", "cat", "money", "neg", "pct", "sci", "date", "bool", "cat"])) for _ in range(ncol)]
    style = _NAME_STYLES[int(rng.integers(len(_NAME_STYLES)))]
    names = [style(str(x)) for x in rng.choice(NAMES, size=ncol, replace=ncol > len(NAMES))]
    if delim == " ":
        names = [n.replace(" ", "_") for n in names]
    has_frac = any(k in ("float", "money", "neg", "pct", "sci") for k in kinds)
    has_num = any(k not in ("cat", "bool", "date") for k in kinds)
    pad = delim == "," and rng.random() < 0.1                          # "a, b, c"
    miss = float(rng.choice([0.0, 0.0, 0.03, 0.1]))
    rows = []
    for _ in range(nrow):
        r = [(_cell(k, decimal, rng, delim) if rng.random() >= miss else "") for k in kinds]
        rows.append(r)
    join = (delim + " ") if pad else delim
    lines = ([join.join(names)] if header else []) + [join.join(r) for r in rows]
    if hard:
        if rng.random() < 0.5:
            lines = ["# exported report", "generated by tool"] + lines
        if rng.random() < 0.4 and len(lines) > 6:
            for _ in range(2):
                lines[int(rng.integers(4, len(lines)))] = join.join(rows[0][:max(1, ncol - 2)])
    text = "\n".join(lines) + ("\n" if (not hard or rng.random() < 0.7) else "")
    if hard and rng.random() < 0.3:
        text = text[: int(len(text) * rng.uniform(0.6, 0.95))]
    if rng.random() < 0.2:
        text = text.replace("\n", "\r\n")
    try:
        data = text.encode(enc)
    except UnicodeEncodeError:
        enc, data = "utf-8", text.encode("utf-8")
    ascii_only = all(ord(c) < 128 for c in text)
    if enc in ("utf-8", "latin-1", "cp1252"):
        cls = "utf-8/ascii" if (enc == "utf-8" or ascii_only) else "single-byte"
    else:
        cls = enc
    return data, dict(delim=delim, decimal=decimal, header=header, enc=cls, text=text,
                      ident=dict(decimal=bool(has_frac), header=bool(has_num), enc=True))


def dataset(n: int, seed: int, hard_share: float = 0.5):
    rng = np.random.default_rng(seed)
    return [make(rng, bool(rng.random() < hard_share)) for _ in range(n)]
