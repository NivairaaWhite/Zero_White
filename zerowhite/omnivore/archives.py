"""archives.py - open containers safely, and get what can be got out of damaged ones.

  expand(data, name, kind, limits)   yields Member(name, data, notes, error) for the files inside a gzip / bz2 / xz / zip / tar

Guarantees:
  * nothing is ever written to disk under a name taken from the archive (no path traversal: names are only labels)
  * every limit is enforced on what is actually decompressed, not on what the header claims (zip bombs); an over-size member
    becomes a Member with an `error`, it does not abort its siblings
  * damage is survivable: a truncated gzip / bz2 / xz stream yields the bytes before the damage; a zip whose directory is missing
    (a download that stopped) is scanned member by member; a tar that ends early yields the files before the end
"""
from __future__ import annotations

import io
import struct
import tarfile
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import PurePosixPath


@dataclass
class Member:
    name: str
    data: bytes = b""
    notes: list = field(default_factory=list)
    error: str = ""


def safe_label(name: str) -> str:
    """A display name from an archive member: no drive letters, no '..', no leading slashes."""
    parts = [p for p in PurePosixPath(str(name).replace("\\", "/")).parts if p not in ("/", "..", ".") and ":" not in p]
    return "/".join(parts) or "member"


class _Budget:
    def __init__(self, limits):
        self.limits, self.members, self.total = limits, 0, 0

    def take(self, n: int) -> str:
        """'' if `n` more bytes / one more member are allowed, else the reason they are not."""
        if self.members + 1 > self.limits.max_members:
            return f"archive holds more than {self.limits.max_members} files; the rest were not opened"
        if self.total + n > self.limits.max_total_bytes:
            return f"unpacking would exceed {self.limits.max_total_bytes // (1 << 20)} MiB in total (zip-bomb guard)"
        self.members += 1
        self.total += n
        return ""


# ---------------------------------------------------------------------------------------------- single-stream formats
def _inflate(data: bytes, kind: str, cap: int) -> tuple:
    """-> (bytes, notes, error). Decompress a whole stream, keeping what came out before any damage, never more than `cap` bytes."""
    notes, out, error = [], bytearray(), ""
    try:
        if kind == "gzip":
            rest = data
            while rest:
                d = zlib.decompressobj(zlib.MAX_WBITS | 16)
                try:
                    chunk = d.decompress(rest, cap - len(out) + 1)
                except zlib.error as e:
                    out += _partial(d, rest)
                    notes.append(f"corrupt gzip data ({e}); recovered {len(out):,} bytes before the damage")
                    break
                out += chunk
                if len(out) > cap:
                    error = f"larger than {cap // (1 << 20)} MiB once decompressed"
                    break
                if not d.eof:
                    notes.append(f"truncated gzip stream; recovered {len(out):,} bytes")
                    break
                rest = d.unused_data
                if rest and not rest.startswith(b"\x1f\x8b"):
                    break
        elif kind == "bz2":
            import bz2
            d = bz2.BZ2Decompressor()
            try:
                chunk = d.decompress(data, max_length=cap + 1)
                out += chunk
                if not d.eof and len(out) <= cap:
                    notes.append(f"truncated bzip2 stream; recovered {len(out):,} bytes")
            except (OSError, ValueError, EOFError) as e:
                notes.append(f"corrupt bzip2 data ({e}); recovered {len(out):,} bytes")
            if len(out) > cap:
                error = f"larger than {cap // (1 << 20)} MiB once decompressed"
        elif kind == "xz":
            import lzma
            d = lzma.LZMADecompressor()
            try:
                out += d.decompress(data, max_length=cap + 1)
                if not d.eof and len(out) <= cap:
                    notes.append(f"truncated xz stream; recovered {len(out):,} bytes")
            except (lzma.LZMAError, EOFError) as e:
                notes.append(f"corrupt xz data ({e}); recovered {len(out):,} bytes")
            if len(out) > cap:
                error = f"larger than {cap // (1 << 20)} MiB once decompressed"
    except ImportError as e:
        error = f"this Python cannot read {kind} ({e})"
    except Exception as e:                                # noqa: BLE001
        error = f"{kind} could not be read: {type(e).__name__}: {e}"
    if error:
        return b"", notes, error
    return bytes(out), notes, ""


def _partial(d, data: bytes) -> bytes:
    """Whatever a deflate stream produces before it breaks (zlib returns nothing once it raises, so re-feed in small pieces)."""
    out = bytearray()
    fresh = zlib.decompressobj(zlib.MAX_WBITS | 16 if data[:2] == b"\x1f\x8b" else -zlib.MAX_WBITS)
    step = 256
    for i in range(0, len(data), step):
        try:
            out += fresh.decompress(data[i:i + step])
        except zlib.error:
            break
    return bytes(out)


# ---------------------------------------------------------------------------------------------- zip
_LOCAL = struct.Struct("<4sHHHHHIIIHH")


def salvage_zip(data: bytes, limits, budget: _Budget):
    """Members of a zip whose central directory is missing or damaged, found by scanning for local file headers."""
    pos, found = 0, 0
    cap = limits.max_member_bytes
    while True:
        i = data.find(b"PK\x03\x04", pos)
        if i < 0 or i + _LOCAL.size > len(data):
            break
        try:
            _, _ver, flag, method, _t, _d, _crc, csize, usize, nlen, elen = _LOCAL.unpack_from(data, i)
        except struct.error:
            break
        pos = i + 4
        if not (0 < nlen < 1024) or flag & 1:             # implausible name, or encrypted
            continue
        raw_name = data[i + 30:i + 30 + nlen]
        try:
            name = raw_name.decode("utf-8" if flag & 0x800 else "cp437")
        except UnicodeDecodeError:
            continue
        start = i + 30 + nlen + elen
        if name.endswith("/"):
            continue
        known = csize if (csize and not flag & 8) else None
        end = start + known if known is not None else len(data)
        payload = data[start:end]
        notes = ["recovered from a zip with no usable directory (a stopped download?)"]
        body, err = b"", ""
        if method == 0:
            body = payload if known is None else payload[:known]
            if known is not None and len(payload) < known:
                notes.append(f"truncated: {len(payload):,} of {known:,} bytes present")
        elif method == 8:
            d = zlib.decompressobj(-zlib.MAX_WBITS)
            try:
                body = d.decompress(payload, cap + 1)
                if not d.eof:
                    notes.append(f"truncated member: recovered {len(body):,} bytes")
                else:
                    pos = start + (len(payload) - len(d.unused_data))
            except zlib.error:
                body = b""
                fresh = zlib.decompressobj(-zlib.MAX_WBITS)
                for k in range(0, len(payload), 256):
                    try:
                        body += fresh.decompress(payload[k:k + 256])
                    except zlib.error:
                        break
                if body:
                    notes.append(f"corrupt member: recovered {len(body):,} bytes before the damage")
                else:
                    err = "corrupt member: nothing recoverable"
        else:
            err = f"compression method {method} is not supported"
        if len(body) > cap:
            body, err = b"", f"larger than {cap // (1 << 20)} MiB once decompressed"
        if not err and not body:
            continue
        reason = "" if err else budget.take(len(body))
        if reason:
            yield Member(safe_label(name), b"", notes, reason)
            return
        found += 1
        yield Member(safe_label(name), body, notes, err)


def _zip_members(data: bytes, limits, budget: _Budget):
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        infos = zf.infolist()
    except (zipfile.BadZipFile, OSError, ValueError, EOFError, NotImplementedError):
        yield from salvage_zip(data, limits, budget)
        return
    cap = limits.max_member_bytes
    for info in infos:
        if info.is_dir():
            continue
        name = safe_label(info.filename)
        if info.flag_bits & 1:
            yield Member(name, b"", [], "encrypted member (no password is ever guessed)")
            continue
        try:
            with zf.open(info) as f:
                body = f.read(cap + 1)
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError, OSError, ValueError, zlib.error) as e:
            yield Member(name, b"", [], f"unreadable member ({type(e).__name__}: {e})")
            continue
        if len(body) > cap:
            yield Member(name, b"", [], f"larger than {cap // (1 << 20)} MiB once decompressed")
            continue
        reason = budget.take(len(body))
        if reason:
            yield Member(name, b"", [], reason)
            return
        yield Member(name, body, [])


# ---------------------------------------------------------------------------------------------- tar
def _tar_members(data: bytes, limits, budget: _Budget):
    cap = limits.max_member_bytes
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:")
    except (tarfile.TarError, OSError, EOFError) as e:
        yield Member("(tar)", b"", [], f"tar could not be opened ({type(e).__name__}: {e})")
        return
    try:
        while True:
            try:
                ti = tf.next()
            except (tarfile.TarError, EOFError, OSError) as e:
                yield Member("(rest of tar)", b"", [f"tar ends early or is damaged ({type(e).__name__}); files before this point were kept"], "")
                return
            if ti is None:
                return
            if not ti.isreg():
                continue                                   # links, devices, directories: never followed
            name = safe_label(ti.name)
            if ti.size > cap:
                yield Member(name, b"", [], f"larger than {cap // (1 << 20)} MiB")
                continue
            try:
                body = tf.extractfile(ti).read(cap + 1)    # type: ignore[union-attr]
            except (tarfile.TarError, EOFError, OSError):
                body = data[ti.offset_data:ti.offset_data + ti.size]          # the file ends inside this member: keep what is there
                if not body:
                    yield Member(name, b"", [], "unreadable member (the tar ends before its data)")
                    return
            reason = budget.take(len(body))
            if reason:
                yield Member(name, b"", [], reason)
                return
            yield Member(name, body, [] if len(body) == ti.size else [f"truncated: {len(body):,} of {ti.size:,} bytes present"])
    finally:
        tf.close()


def expand(data: bytes, name: str, kind: str, limits):
    """Yield the Members of one archive. `kind` is what `sniff.identify` said: gzip | bz2 | xz | zip | tar."""
    budget = _Budget(limits)
    if kind in ("gzip", "bz2", "xz"):
        body, notes, err = _inflate(data, kind, limits.max_member_bytes)
        inner = name
        for suf in (".gz", ".gzip", ".bz2", ".xz", ".tgz"):
            if inner.lower().endswith(suf):
                inner = inner[:-len(suf)] + (".tar" if suf == ".tgz" else "")
                break
        else:
            inner = inner + ".out"
        if err:
            yield Member(safe_label(inner), b"", notes, err)
        elif body:
            budget.take(len(body))
            yield Member(safe_label(inner), body, notes)
        else:
            yield Member(safe_label(inner), b"", notes, "empty after decompression")
    elif kind == "zip":
        yield from _zip_members(data, limits, budget)
    elif kind == "tar":
        yield from _tar_members(data, limits, budget)
