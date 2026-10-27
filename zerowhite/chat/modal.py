"""modal.py - put tables, audio codes and reinforcement-learning episodes into the SAME token stream as text, and read them back.

The unified-brain idea of the architecture review is one model over every kind of data. A transformer already is one: it reads token ids. What stops the chat model from
seeing a table or a sound is only that nothing writes them as tokens. This module defines a plain-text serialisation with a typed marker at the start of every line, so the
existing tokenizer, corpus and trainer carry all modalities with no code change:

    [table sales] region | units | price           one header line, then one `[row]` line per row
    [row] north | 120 | 9.5
    [audio] a12.7.55.3 a40.1.9.22 ...               codec tokens (voice/codec.py), one word per 10 ms frame
    [rl] o=0.1,-0.3 a=1 r=1.0                       one environment step: observation, action, reward

Each writer has an exact reader (numbers are kept to 6 significant digits). `modality_of(line)` says which a line is. NOT done: a shared learned EMBEDDING space (the model sees
these as ordinary text tokens, so it learns them as it learns any text), tokenizer changes (BPE will split `a12.7.55.3` into pieces; a dedicated vocabulary would be tighter),
and any evidence that a chat model trained on this text performs better on tables or audio: that needs real training runs on real data.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import numpy as np

_NUM = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")


def _cell(x) -> str:
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.6g}"
    return " ".join(str(x).replace("|", "/").split())


def _parse(c: str):
    c = c.strip()
    if _NUM.match(c):
        return int(c) if re.fullmatch(r"-?\d+", c) else float(c)
    return c


def table_to_text(name: str, columns: Sequence[str], rows: Sequence[Sequence], max_rows: int = 200) -> str:
    head = f"[table {re.sub(r'[^A-Za-z0-9_.-]', '_', name) or 'table'}] " + " | ".join(_cell(c) for c in columns)
    return "\n".join([head] + ["[row] " + " | ".join(_cell(c) for c in r) for r in list(rows)[:max_rows]])


def text_to_table(text: str) -> Tuple[str, List[str], List[list]]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    m = re.match(r"\[table ([^\]]*)\] ?(.*)$", lines[0]) if lines else None
    if not m:
        raise ValueError("not a serialised table: the first line must start with [table NAME]")
    cols = [c.strip() for c in m.group(2).split("|")]
    rows = [[_parse(c) for c in ln[len("[row]"):].split("|")] for ln in lines[1:] if ln.startswith("[row]")]
    return m.group(1), cols, rows


def audio_to_text(tokens: np.ndarray) -> str:
    from ..voice import codec
    return "[audio] " + codec.tokens_to_text(tokens)


def text_to_audio(text: str) -> np.ndarray:
    from ..voice import codec
    return codec.text_to_tokens(text[len("[audio]"):] if text.startswith("[audio]") else text)


def episode_to_text(obs: Sequence, acts: Sequence, rews: Sequence) -> str:
    return "\n".join(f"[rl] o={','.join(_cell(v) for v in np.ravel(o))} a={_cell(a)} r={_cell(r)}" for o, a, r in zip(obs, acts, rews))


def text_to_episode(text: str):
    O, A, R = [], [], []
    for ln in text.splitlines():
        m = re.match(r"\[rl\] o=(\S*) a=(\S+) r=(\S+)$", ln.strip())
        if m:
            O.append([float(v) for v in m.group(1).split(",") if v != ""])
            A.append(_parse(m.group(2)))
            R.append(float(m.group(3)))
    return np.array(O, dtype=float), np.array(A), np.array(R, dtype=float)


def modality_of(line: str) -> str:
    for tag in ("[table", "[row]", "[audio]", "[rl]"):
        if line.startswith(tag):
            return tag.strip("[]")
    return "text"
