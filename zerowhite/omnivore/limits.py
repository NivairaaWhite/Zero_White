"""limits.py - every cap the ingestion layer obeys, in one place.

A file that exceeds a cap is never an error: it is read in part (and the report says so), or set aside with the reason.
`Limits.for_machine()` derives the size caps from what `substrate.probe()` measured; the plain constructor holds the
values used when nothing could be measured (and in tests).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

MiB = 1 << 20


@dataclass
class Limits:
    # reading
    max_file_bytes: int = 64 * MiB          # a larger file is read from its start only
    max_member_bytes: int = 64 * MiB        # one file unpacked from an archive
    max_total_bytes: int = 256 * MiB        # everything unpacked from one archive (zip-bomb guard)
    max_members: int = 2000                 # files unpacked from one archive
    max_depth: int = 3                      # archives inside archives
    max_files: int = 20000                  # files visited under one folder
    settle_seconds: float = 1.0             # a file modified more recently than this is "still being written": try later
    # tables
    max_rows: int = 50_000
    max_features: int = 256
    max_levels: int = 30                    # categories turned into columns; the rest share one "other" column
    min_rows: int = 30                      # fewer rows than this cannot be split into train / validation / test
    min_class_count: int = 5                # a class rarer than this cannot be learned or tested and is set aside
    max_classes: int = 50
    # text
    min_text_chars: int = 500               # the char model's own minimum
    max_text_chars: int = 2_000_000

    @classmethod
    def for_machine(cls, plan: dict) -> "Limits":
        """Size caps from `substrate.plan()`; everything else keeps its value."""
        names = {f.name for f in fields(cls)}
        kw = {k: int(v) for k, v in plan.items() if k in names and isinstance(v, (int, float))}
        lim = cls(**kw)
        lim.max_member_bytes = min(lim.max_member_bytes, lim.max_file_bytes)
        return lim

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
