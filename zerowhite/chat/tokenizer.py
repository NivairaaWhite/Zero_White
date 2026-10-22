"""tokenizer.py - a byte-level BPE tokenizer trained from scratch on YOUR data, plus the chat template.

* Byte level: every possible text round-trips exactly, there is no <unk>; non-English text and emoji simply cost more tokens.
* Special tokens live at ids 0..S-1 (pad, system, user, assistant, end + anything in `template.extra_tokens`), then the 256
  raw bytes, then the learned merges. Special-token *strings* inside ordinary text are NEVER interpreted: a user (or a
  dataset) typing "<|end|>" gets plain characters, so nobody can inject a role or an end-of-message.
* Training is the classic incremental pair-count algorithm with a lazy max-heap, so it scales to tens of MB in Python.

The file format is plain JSON (merges + specials), readable and diffable; its hash names the token shards, so changing the
tokenizer automatically invalidates (and rebuilds) everything tokenised with the old one.
"""
from __future__ import annotations

import codecs
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .. import fileio

# GPT-2 style pre-tokenisation without \p{..} classes (python's re has none): contractions, letters, digits, punctuation, spaces.
_PRETOKEN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""", re.UNICODE)
PRETOKENIZER_NAME = "gpt2-like-v1"


class TokenizerError(ValueError):
    pass


def _merge_word(ids: List[int], a: int, b: int, new: int) -> List[int]:
    out, i, n = [], 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, merges: Sequence[Tuple[int, int]], specials: Sequence[str], max_piece_bytes: int = 64):
        self.specials: List[str] = list(specials)
        if len(set(self.specials)) != len(self.specials):
            raise TokenizerError("duplicate special tokens")
        self.offset = len(self.specials)
        self.merges: List[Tuple[int, int]] = [tuple(m) for m in merges]            # in internal ids (bytes 0..255, merges 256..)
        self.max_piece_bytes = int(max_piece_bytes)
        self.ranks: Dict[Tuple[int, int], int] = {m: i for i, m in enumerate(self.merges)}
        self._special_ids = {s: i for i, s in enumerate(self.specials)}
        vb: List[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            vb.append(vb[a] + vb[b])
        self._vb = vb
        self._cache: Dict[str, List[int]] = {}

    # ---- sizes / ids -------------------------------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self.offset + 256 + len(self.merges)

    def token_id(self, special: str) -> int:
        try:
            return self._special_ids[special]
        except KeyError:
            raise TokenizerError(f"'{special}' is not a special token of this tokenizer (has: {self.specials}); "
                                 f"retrain the tokenizer or remove it from template.*") from None

    def has_special(self, s: str) -> bool:
        return s in self._special_ids

    def is_special(self, tid: int) -> bool:
        return 0 <= tid < self.offset

    # ---- encode ------------------------------------------------------------------------------------------------
    def _bpe_piece(self, piece: str) -> List[int]:
        hit = self._cache.get(piece)
        if hit is not None:
            return hit
        raw = piece.encode("utf-8")
        out: List[int] = []
        step = self.max_piece_bytes
        for s in range(0, len(raw), step):
            ids = list(raw[s:s + step])
            ranks = self.ranks
            while len(ids) > 1:
                best, best_rank = None, None
                for p in zip(ids, ids[1:]):
                    r = ranks.get(p)
                    if r is not None and (best_rank is None or r < best_rank):
                        best, best_rank = p, r
                if best is None:
                    break
                ids = _merge_word(ids, best[0], best[1], 256 + best_rank)
            out.extend(ids)
        out = [i + self.offset for i in out]
        if len(self._cache) > 300_000:
            self._cache.clear()
        self._cache[piece] = out
        return out

    def encode(self, text: str) -> List[int]:
        """Plain text -> ids. Special-token strings are treated as ordinary characters."""
        out: List[int] = []
        for piece in _PRETOKEN.findall(text):
            out.extend(self._bpe_piece(piece))
        return out

    # ---- decode ------------------------------------------------------------------------------------------------
    def token_bytes(self, tid: int) -> bytes:
        if tid < self.offset:
            return b""
        return self._vb[tid - self.offset]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        parts: List[bytes] = []
        for t in ids:
            t = int(t)
            if t < self.offset:
                if not skip_special and 0 <= t:
                    parts.append(self.specials[t].encode("utf-8"))
            elif t - self.offset < len(self._vb):
                parts.append(self._vb[t - self.offset])
        return b"".join(parts).decode("utf-8", errors="replace")

    def stream_decoder(self):
        """Incremental decoder for token streams: feed(ids) -> text that is safe to show now (never a half character)."""
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

        class _S:
            def feed(_s, ids):
                data = b"".join(self.token_bytes(int(t)) for t in ids)
                return dec.decode(data)

            def flush(_s):
                return dec.decode(b"", final=True)

        return _S()

    # ---- persistence -------------------------------------------------------------------------------------------
    def to_json(self) -> dict:
        return dict(format="zerowhite-bpe", version=1, pretokenizer=PRETOKENIZER_NAME, specials=self.specials,
                    max_piece_bytes=self.max_piece_bytes, merges=[list(m) for m in self.merges])

    @classmethod
    def from_json(cls, d: dict) -> "Tokenizer":
        if d.get("format") != "zerowhite-bpe":
            raise TokenizerError("not a Zero White tokenizer file")
        if d.get("pretokenizer") != PRETOKENIZER_NAME:
            raise TokenizerError(f"tokenizer was built with pre-tokenizer '{d.get('pretokenizer')}', this build has "
                                 f"'{PRETOKENIZER_NAME}'")
        return cls([tuple(m) for m in d["merges"]], d["specials"], d.get("max_piece_bytes", 64))

    def save(self, path: str):
        fileio.write_json(path, self.to_json())

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        return cls.from_json(fileio.read_json(path))

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:10]

    # ---- training ----------------------------------------------------------------------------------------------
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, specials: Sequence[str], min_frequency: int = 2,
              max_piece_bytes: int = 64, progress: Optional[Callable[[int, int], None]] = None) -> "Tokenizer":
        n_merges = int(vocab_size) - len(specials) - 256
        if n_merges < 0:
            raise TokenizerError(f"vocab_size {vocab_size} is smaller than specials+256 ({len(specials) + 256})")
        piece_counts: Counter = Counter()
        for t in texts:
            piece_counts.update(_PRETOKEN.findall(t))
        chunk_counts: Counter = Counter()
        for piece, c in piece_counts.items():
            raw = piece.encode("utf-8")
            for s in range(0, len(raw), max_piece_bytes):
                chunk_counts[raw[s:s + max_piece_bytes]] += c
        del piece_counts
        words: List[List[int]] = [list(b) for b in chunk_counts]
        freqs: List[int] = list(chunk_counts.values())
        del chunk_counts

        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        pair_words: Dict[Tuple[int, int], set] = defaultdict(set)
        for wi, ids in enumerate(words):
            f = freqs[wi]
            for p in zip(ids, ids[1:]):
                pair_counts[p] += f
                pair_words[p].add(wi)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)
        merges: List[Tuple[int, int]] = []
        next_id = 256
        while len(merges) < n_merges and heap:
            negc, pair = heapq.heappop(heap)
            c = pair_counts.get(pair, 0)
            if c != -negc:
                if c > 0:
                    heapq.heappush(heap, (-c, pair))
                continue
            if c < max(1, min_frequency):
                break
            a, b = pair
            new_id = next_id
            next_id += 1
            merges.append(pair)
            touched = []
            for wi in list(pair_words.get(pair, ())):
                ids = words[wi]
                if len(ids) < 2:
                    continue
                f = freqs[wi]
                new_ids = _merge_word(ids, a, b, new_id)
                if len(new_ids) == len(ids):
                    continue
                for p in zip(ids, ids[1:]):
                    pair_counts[p] -= f
                for p in zip(new_ids, new_ids[1:]):
                    pair_counts[p] += f
                    pair_words[p].add(wi)
                    touched.append(p)
                words[wi] = new_ids
            pair_counts[pair] = 0
            pair_words.pop(pair, None)
            seen = set()
            for p in touched:
                if p not in seen:
                    seen.add(p)
                    cc = pair_counts[p]
                    if cc > 0:
                        heapq.heappush(heap, (-cc, p))
            if progress and len(merges) % 200 == 0:
                progress(len(merges), n_merges)
        if progress:
            progress(len(merges), n_merges)
        return cls(merges, specials, max_piece_bytes)


# ====================================================================================================================
class ChatTemplate:
    """Turns (system, turns) into token ids and a loss mask.  Layout (no extra newlines, nothing hidden):

        <|system|> system text <|end|> <|user|> user text <|end|> <|assistant|> reply text <|end|> ...

    mask[i] = 1 means "token i is a training target" (assistant text and its <|end|>); role tokens are never targets
    because the prompt supplies them. With template.train_on = "all" every non-pad token is a target.
    """

    def __init__(self, tok: Tokenizer, cfg):
        t = cfg.template
        self.tok = tok
        self.sys_id = tok.token_id(t.system_token)
        self.user_id = tok.token_id(t.user_token)
        self.asst_id = tok.token_id(t.assistant_token)
        self.end_id = tok.token_id(t.end_token)
        self.pad_id = tok.token_id(t.pad_token)
        self.train_on = t.train_on
        self.role_id = {"system": self.sys_id, "user": self.user_id, "assistant": self.asst_id}

    @staticmethod
    def special_tokens(cfg) -> List[str]:
        t = cfg.template
        base = [t.pad_token, t.system_token, t.user_token, t.assistant_token, t.end_token]
        for x in t.extra_tokens:
            if x not in base:
                base.append(x)
        return base

    # ---- pieces ------------------------------------------------------------------------------------------------
    def turn(self, role: str, content: str) -> Tuple[List[int], List[int]]:
        body = self.tok.encode(content)
        ids = [self.role_id[role]] + body + [self.end_id]
        if self.train_on == "all" or role == "assistant":
            mask = [0] + [1] * (len(body) + 1)
            if self.train_on == "all":
                mask[0] = 1 if role != "system" else 0
        else:
            mask = [0] * len(ids)
        return ids, mask

    def header(self, system: Optional[str], limit: int) -> Tuple[List[int], List[int]]:
        if not system:
            return [], []
        body = self.tok.encode(system)[:max(0, limit - 2)]
        ids = [self.sys_id] + body + [self.end_id]
        return ids, [0] * len(ids)

    # ---- whole conversations -----------------------------------------------------------------------------------
    def prompt_ids(self, system: Optional[str], turns: Sequence[dict], max_len: int) -> List[int]:
        """Ids for generation: system + as many of the most recent turns as fit + the assistant tag.
        The system header is always kept; the oldest turns are dropped first."""
        head, _ = self.header(system, max_len // 2)
        pieces = [self.turn(t["role"], t["content"])[0] for t in turns if t.get("content")]
        budget = max_len - len(head) - 1
        kept: List[List[int]] = []
        used = 0
        for p in reversed(pieces):
            if used + len(p) > budget:
                if not kept:                        # even the newest turn is too long: keep its tail
                    tail = p[-(budget - 1):] if budget > 1 else []
                    kept.append([p[0]] + tail if tail else [])
                break
            kept.append(p)
            used += len(p)
        out = list(head)
        for p in reversed(kept):
            out.extend(p)
        out.append(self.asst_id)
        return out

    def encode_text(self, text: str, max_len: int) -> List[Tuple[List[int], List[int]]]:
        ids = self.tok.encode(text)
        out = []
        for s in range(0, len(ids), max_len - 1):
            part = ids[s:s + max_len - 1] + [self.end_id]
            out.append((part, [1] * len(part)))
        return out

    def chunk_chat(self, system: Optional[str], turns: Sequence[dict], max_len: int) -> List[Tuple[List[int], List[int]]]:
        """Split one conversation into windows of at most max_len tokens. Every window starts with the system header and
        contains whole turns (a turn longer than the room left is truncated); windows without a training target are dropped."""
        head_ids, head_mask = self.header(system, max_len // 2)
        room = max_len - len(head_ids)
        windows: List[Tuple[List[int], List[int]]] = []
        pending: List[Tuple[List[int], List[int], str]] = []          # pieces in current window (for carry-over)
        for t in turns:
            role, content = t["role"], t["content"]
            if role not in self.role_id or not content:
                continue
            ids, mask = self.turn(role, content)
            if len(ids) > room:                                       # truncate an over-long turn, keep its closing tag
                ids = ids[:room - 1] + [self.end_id]
                mask = mask[:room - 1] + [mask[-1]]
            if sum(len(p[0]) for p in pending) + len(ids) > room and pending:
                carry = []
                if pending[-1][2] == "user":                          # do not strand a question from its answer
                    carry = [pending.pop()]
                if any(sum(p[1]) for p in pending):
                    windows.append((head_ids + [i for p in pending for i in p[0]],
                                    head_mask + [m for p in pending for m in p[1]]))
                pending = carry if carry and len(carry[0][0]) + len(ids) <= room else []
            pending.append((ids, mask, role))
        if pending and any(sum(p[1]) for p in pending):
            windows.append((head_ids + [i for p in pending for i in p[0]], head_mask + [m for p in pending for m in p[1]]))
        return windows

    def encode_pair(self, system: Optional[str], context: Sequence[dict], answer: str, max_len: int):
        """(prompt_ids, answer_ids) for preference learning; answer_ids include the closing <|end|>."""
        prompt = self.prompt_ids(system, context, max_len - 2)
        ans = self.tok.encode(answer)[:max(1, max_len - len(prompt) - 1)] + [self.end_id]
        return prompt, ans


# ====================================================================================================================
def derive_vocab_size(corpus_bytes: int, n_specials: int) -> Tuple[int, str]:
    """vocab ~ 2*sqrt(bytes) rounded to a power of two, clamped to [1024, 32768] (+ specials).
    Small corpora cannot support big vocabularies (rare merges never get trained); large ones waste compute on a tiny one."""
    import math
    if corpus_bytes <= 0:
        return 1024 + n_specials, "empty corpus: minimum vocabulary"
    raw = 2.0 * math.sqrt(corpus_bytes)
    v = 2 ** int(round(math.log2(max(raw, 1.0))))
    v = int(min(32768, max(1024, v)))
    return v + n_specials, f"2*sqrt({corpus_bytes:,} bytes) = {raw:,.0f} -> {v} (+{n_specials} special)"
