"""watchdog.py - the part of self-modification that must not be rewritten by the thing being modified.

The source patcher already judges every change in a throw-away copy (sandbox.SandboxCopy). What it did NOT have was a safe way to put a
winning change into the real tree: files were written one by one, so a crash (power loss, the process killed, an exception) between two
writes left the repository half patched with nothing to repair it. This module is that missing step, as a transaction:

    prepare   copy every original and every new file into <journal>/<txid>/, record their SHA-256, write the journal (state "prepared")
    apply     swap each file into place atomically (temp file in the same folder + os.replace), journal state "applying"
    verify    a caller-supplied check on the REAL tree: the test suite, and that the sealed constitution files are byte-for-byte unchanged
    commit    journal state "committed"            |   rollback  restore every original, journal state "rolled_back"
    recover   at every start: any journal that is not committed or rolled back is rolled back. A half-applied change cannot survive a restart.

What it refuses: a path outside `allowed`, an absolute or `..` path, a file that does not exist yet, and any change that leaves a file listed
in CONSTITUTION.sha256 different from its seal (checked after applying, before committing).

Standalone ON PURPOSE: standard library only, no zerowhite imports, so that nothing the self-modifier can edit sits between a decision and the
disk. It is part of the CONSTITUTION (sealed). It can run in-process (`run(...)`, what patcher.search uses) or as a separate process:

    python -m zerowhite.selfmod.watchdog promote --repo . --changeset changes.json [--verify-cmd '["python","tests/run_all.py"]']
    python -m zerowhite.selfmod.watchdog recover --repo .

HONEST LIMITS: the swap is atomic per FILE, not across files, so a crash mid-transaction is repaired by rolling back, not by finishing;
the journal is fsync'd but a disk that lies about fsync can defeat any journal; and `allowed` / the verify command are only as good as
the caller (the CLI derives `allowed` from patcher.MUTABLE in the repo, which is sealed).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import posixpath
import subprocess
import sys
import tempfile
import time
import uuid

JOURNAL_DIR = ".zw_promotion"
OPEN_STATES = ("prepared", "applying", "verifying")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _seal_sha(data: bytes) -> str:                       # the same CRLF-insensitive hash doctor.py uses for the seal
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def _swap(src, dst, tries=8):
    for k in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:                          # Windows: a scanner / indexer holds the file for a moment
            if os.name != "nt" or k == tries - 1:
                raise
            time.sleep(0.05 * (k + 1))


def _atomic_write(path, data: bytes, fsync=True):
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        _swap(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _save(tx_dir, j):
    _atomic_write(os.path.join(tx_dir, "journal.json"), json.dumps(j, indent=1).encode("utf-8"))


def _safe_rel(rel, allowed):
    r = posixpath.normpath(str(rel).replace("\\", "/"))
    if r.startswith("/") or r.startswith("../") or r == ".." or (len(r) > 1 and r[1] == ":"):
        raise PermissionError(f"{rel!r} is outside the repository")
    if r not in allowed:
        raise PermissionError(f"{rel!r} is not a file the self-modifier may change")
    return r


def seal_problems(repo):
    """Files listed in CONSTITUTION.sha256 that no longer match (and the seal itself if unreadable). Empty list = intact."""
    p = os.path.join(repo, "zerowhite", "CONSTITUTION.sha256")
    if not os.path.exists(p):
        return []                                         # an unsealed checkout: nothing to compare with (doctor warns about that)
    try:
        want = json.loads(_read(p).decode("utf-8"))
    except (ValueError, OSError):
        return ["CONSTITUTION.sha256 is unreadable"]
    bad = []
    for rel, h in want.items():
        fp = os.path.join(repo, "zerowhite", *rel.split("/"))
        if not os.path.exists(fp) or _seal_sha(_read(fp)) != h:
            bad.append(rel)
    return bad


def rollback(repo, tx_dir):
    j = json.loads(_read(os.path.join(tx_dir, "journal.json")).decode("utf-8"))
    for f in j["files"]:
        live = os.path.join(repo, *f["rel"].split("/"))
        orig = _read(os.path.join(tx_dir, "orig", *f["rel"].split("/")))
        if not os.path.exists(live) or _sha(_read(live)) != f["old_sha"]:
            _atomic_write(live, orig)
    j["state"], j["finished"] = "rolled_back", time.time()
    _save(tx_dir, j)
    return j


def recover(repo, journal_dir=None):
    """Roll back every transaction that was interrupted. Returns [{txid, state_found}]. Safe to call at any time, any number of times."""
    jd = journal_dir or os.path.join(repo, JOURNAL_DIR)
    done = []
    if not os.path.isdir(jd):
        return done
    for name in sorted(os.listdir(jd)):
        jp = os.path.join(jd, name, "journal.json")
        if not os.path.exists(jp):
            continue
        try:
            j = json.loads(_read(jp).decode("utf-8"))
        except ValueError:
            continue
        if j.get("state") in OPEN_STATES:
            rollback(repo, os.path.join(jd, name))
            done.append(dict(txid=j.get("txid", name), state_found=j["state"]))
    return done


def run(repo, changes, allowed, verify=None, journal_dir=None, check_seal=True, keep=20):
    """Apply {repo-relative path: new text/bytes} as one journaled transaction.

    `verify()` runs on the REAL tree after the swap and returns True/False or (ok, detail). Returns
    dict(ok, state, reason, txid, files=[rel...]); never leaves the tree half-changed (but see HONEST LIMITS)."""
    repo = os.path.abspath(repo)
    jd = journal_dir or os.path.join(repo, JOURNAL_DIR)
    recovered = recover(repo, jd)                                          # first, repair whatever an earlier crash left behind
    allowed = {posixpath.normpath(a.replace("\\", "/")) for a in allowed}
    plan = []
    for rel, new in changes.items():
        r = _safe_rel(rel, allowed)
        live = os.path.join(repo, *r.split("/"))
        if not os.path.exists(live):
            raise PermissionError(f"{r!r} does not exist: only existing files can be changed")
        nb = new.encode("utf-8") if isinstance(new, str) else bytes(new)
        ob = _read(live)
        if ob != nb:
            plan.append((r, live, ob, nb))
    if not plan:
        return dict(ok=True, state="unchanged", reason="nothing differs", txid=None, files=[], recovered=recovered)
    if check_seal:
        pre = seal_problems(repo)
        if pre:                                           # not this change's doing, and not something to build on: stop before touching anything
            return dict(ok=False, state="refused", txid=None, files=[r for r, *_ in plan], recovered=recovered,
                        reason="the constitution does not match its seal before the change (" + ", ".join(pre)
                               + "); if you edited it on purpose run `python -m zerowhite doctor --seal`, otherwise restore it")
    txid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    tx_dir = os.path.join(jd, txid)
    j = dict(txid=txid, state="prepared", started=time.time(),
             files=[dict(rel=r, old_sha=_sha(ob), new_sha=_sha(nb)) for r, _, ob, nb in plan])
    for r, _, ob, nb in plan:
        _atomic_write(os.path.join(tx_dir, "orig", *r.split("/")), ob)
        _atomic_write(os.path.join(tx_dir, "new", *r.split("/")), nb)
    _save(tx_dir, j)
    result = dict(ok=False, state="rolled_back", reason="", txid=txid, files=[r for r, *_ in plan], recovered=recovered)
    try:
        j["state"] = "applying"
        _save(tx_dir, j)
        for r, live, _, nb in plan:
            _atomic_write(live, nb)
        j["state"] = "verifying"
        _save(tx_dir, j)
        if verify is not None:
            v = verify()
            ok, detail = (v if isinstance(v, tuple) else (bool(v), ""))
            if not ok:
                result["reason"] = f"verification failed on the real tree ({detail})" if detail else "verification failed on the real tree"
        if not result["reason"] and check_seal:             # LAST, after verification: nothing that ran before the commit may have touched the seal
            problems = seal_problems(repo)
            if problems:
                result["reason"] = "the change altered sealed constitution files: " + ", ".join(problems)
        if not result["reason"]:
            j["state"], j["finished"] = "committed", time.time()
            _save(tx_dir, j)
            result.update(ok=True, state="committed")
            _prune(jd, keep)
            return result
    except BaseException as ex:                                            # incl. KeyboardInterrupt: undo, then let it propagate
        rollback(repo, tx_dir)
        if not isinstance(ex, Exception):
            raise
        result["reason"] = f"{type(ex).__name__}: {ex}"
        return result
    rollback(repo, tx_dir)
    return result


def _prune(jd, keep):
    try:
        names = sorted(n for n in os.listdir(jd) if os.path.exists(os.path.join(jd, n, "journal.json")))
        for n in names[:-keep]:
            import shutil
            shutil.rmtree(os.path.join(jd, n), ignore_errors=True)
    except OSError:
        pass


def mutable_files(repo):
    """The allow-list, read from the sealed patcher.py in the repository (never from the changeset)."""
    tree = ast.parse(_read(os.path.join(repo, "zerowhite", "selfmod", "patcher.py")).decode("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") == "MUTABLE":
            return ["zerowhite/" + f for f in ast.literal_eval(node.value)]
    raise RuntimeError("patcher.MUTABLE not found")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="watchdog", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("promote")
    p.add_argument("--repo", default=".")
    p.add_argument("--changeset", required=True, help='JSON {"zerowhite/evolve.py": "<new text>", ...}')
    p.add_argument("--verify-cmd", default=None, help='JSON list, run in the repo; exit code 0 = accept (e.g. \'["python","tests/run_all.py"]\')')
    p.add_argument("--timeout", type=float, default=900.0)
    r = sub.add_parser("recover")
    r.add_argument("--repo", default=".")
    a = ap.parse_args(argv)
    repo = os.path.abspath(a.repo)
    if a.cmd == "recover":
        print(json.dumps(recover(repo)))
        return 0
    changes = json.loads(_read(a.changeset).decode("utf-8"))
    verify = None
    if a.verify_cmd:
        cmd = json.loads(a.verify_cmd)
        cmd = [sys.executable if c in ("python", "python3") else c for c in cmd]

        def verify():
            q = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=a.timeout)
            return q.returncode == 0, (q.stdout.strip().splitlines() or [""])[-1][:200]
    res = run(repo, changes, mutable_files(repo), verify=verify)
    print(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
