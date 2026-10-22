"""doctor.py - troubleshooting: find what is wrong, say exactly why, and (with --fix) repair what is safe to repair.

    python -m zerowhite doctor --workspace runs/zw            # diagnose
    python -m zerowhite doctor --workspace runs/zw --fix      # also apply the safe repairs
    python -m zerowhite doctor --seal                         # developers: re-seal the constitution after editing it

Checks: Python / NumPy / platform / encodings; that every module imports; the CONSTITUTION seal (the files the
self-modification machinery must never be able to change); the sandbox's real guarantees on this machine; a
file:// URL round trip (the exact bug class behind the Windows test failures); dataset cache; the workspace
(journal, stale temp files, checkpoints, zoo); and the self-written code (history store hashes, live hooks vs the
genesis hooks, with automatic rollback of a hook that has made the search worse).
Repairs never delete evidence: corrupt files are renamed `*.corrupt-<time>`, and every rollback is logged.
"""

import hashlib
import importlib
import json
import os
import pathlib
import platform
import sys
import tempfile
import time
from dataclasses import dataclass

import numpy as np

from . import fileio

HERE = os.path.dirname(os.path.abspath(__file__))
SEAL = os.path.join(HERE, "CONSTITUTION.sha256")
CONSTITUTION = ("selfmod/safety.py", "selfmod/history.py", "selfmod/sandbox.py", "selfmod/patcher.py", "selfmod/winjob.py",
                "selfmod/stats.py", "selfmod/watchdog.py", "selfmod/engine_tier.py", "forage/fetch.py",
                "selfmod/astmut.py", "fileio.py", "doctor.py", "evolved/__init__.py")
MODULES = ("agents", "architecture", "checkpoint", "curiosity", "datasets", "engine", "evolve", "evolved", "factory",
           "fileio", "generative", "meta_optimizer", "nn", "report", "self_model", "tasks", "zero_white",
           "selfmod.safety", "selfmod.astmut", "selfmod.islands", "selfmod.history", "selfmod.sandbox",
           "selfmod.hooks", "selfmod.patcher", "selfmod.bench", "forge.task", "forge.envs", "forge.families",
           "forge.prior", "forge.zoo", "forge.generalist", "forge.poet",
           "forecast", "selfmod.winjob", "selfmod.stats", "selfmod.watchdog", "forge.reduce", "forage", "chat.grow",
           "metric", "selfmod.egraph", "selfmod.engine_tier", "omnivore.learned", "omnivore.synth", "kernels", "kernels.graph", "kernels.cjit", "kernels.cuda",
           "voice.codec", "forge.latent", "forge.system2", "forage.curiosity", "chat.sleep", "chat.modal", "chat.shared")


@dataclass
class Finding:
    level: str            # ok | warn | fail | fixed
    area: str
    msg: str
    hint: str = ""


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()      # CRLF checkouts hash the same


def seal():
    d = {rel: _sha(os.path.join(HERE, *rel.split("/"))) for rel in CONSTITUTION}
    fileio.write_json(SEAL, d, indent=2)
    return d


def check_environment(out):
    v = sys.version_info
    out.append(Finding("ok" if v >= (3, 9) else "fail", "python", f"Python {platform.python_version()} on {platform.platform()}",
                       "" if v >= (3, 9) else "Zero White needs Python 3.9 or newer (it uses ast.unparse)"))
    nv = tuple(int(x) for x in np.__version__.split(".")[:2])
    out.append(Finding("ok" if nv >= (1, 24) else "warn", "numpy", f"NumPy {np.__version__}",
                       "" if nv >= (1, 24) else "pip install -U numpy"))
    enc = getattr(sys.stdout, "encoding", None) or "unknown"
    out.append(Finding("ok" if enc.lower().replace("-", "") in ("utf8", "utf8sig") else "warn", "encoding",
                       f"stdout encoding is {enc}",
                       "" if enc.lower().replace("-", "") in ("utf8", "utf8sig") else
                       "all files Zero White writes are UTF-8 regardless; set PYTHONUTF8=1 to make console output UTF-8 too"))
    out.append(Finding("ok", "files", f"default file encoding for text mode: {os.device_encoding(0) or 'n/a'}; "
                       "Zero White always passes encoding='utf-8' explicitly"))


def check_imports(out):
    bad = []
    for m in MODULES:
        try:
            importlib.import_module(f"{__package__}.{m}")
        except Exception as e:                                                       # noqa: BLE001
            bad.append(f"{m}: {type(e).__name__}: {e}")
    out.append(Finding("ok" if not bad else "fail", "imports", f"{len(MODULES) - len(bad)}/{len(MODULES)} modules import",
                       "; ".join(bad)))


def check_constitution(out, fix=False):
    if not os.path.exists(SEAL):
        out.append(Finding("warn", "constitution", "no CONSTITUTION.sha256 seal present",
                           "run `python -m zerowhite doctor --seal` once after a clean install or checkout"))
        return
    want, bad = fileio.read_json(SEAL), []
    for rel, h in want.items():
        p = os.path.join(HERE, *rel.split("/"))
        if not os.path.exists(p):
            bad.append(f"{rel} (missing)")
        elif _sha(p) != h:
            bad.append(rel)
    out.append(Finding("ok" if not bad else "fail", "constitution",
                       "protected files match their seal" if not bad else "protected files changed: " + ", ".join(bad),
                       "" if not bad else "the self-modifier is not allowed to edit these. If YOU edited them on purpose, "
                                          "re-seal with `doctor --seal`; otherwise restore them from git"))


def check_sandbox(out):
    from .selfmod import sandbox
    rep = sandbox.sandbox_report()
    r = sandbox.run_code("print(40 + 2)", timeout=20)
    t = sandbox.run_code("while True: pass", timeout=2)
    ok = r.ok and r.stdout.strip() == "42" and t.timed_out
    weak = [k for k in ("rlimits", "network_namespace") if not rep[k]]
    if rep["rlimits"]:                                           # 6.3: the limit must actually bite, not merely be configured
        ctl = sandbox.run_code("b = bytearray(16 * 2**20); print('alive')", timeout=20, mem_mb=256)
        hog = sandbox.run_code("b = bytearray(900 * 2**20); b[::4096] = bytes(len(b[::4096])); print('alive')", timeout=30, mem_mb=256)
        if ctl.stdout.strip() == "alive" and hog.stdout.strip() == "alive":
            out.append(Finding("warn", "sandbox", "a child allocated 900 MB under a 256 MB limit: the memory limit is not enforced here",
                               "the sandbox is containment for accidents only; run Zero White in a container or VM for untrusted code"))
        elif ctl.stdout.strip() != "alive":
            out.append(Finding("warn", "sandbox", "a small child could not run under a 256 MB limit; the memory limit could not be verified", ""))
    out.append(Finding("ok" if ok else "fail", "sandbox",
                       f"subprocess run OK, runaway loop killed after {t.seconds:.1f}s" if ok else "sandbox self-test failed",
                       ""))
    if weak:
        out.append(Finding("warn", "sandbox", "not enforced on this OS: " + ", ".join(weak),
                           "expected on Windows. The sandbox is containment for accidents, not a security boundary; "
                           "for untrusted code run Zero White inside a container or VM"))


def check_file_urls(out):
    """The Windows failure in your log: 'file://' + 'C:/path' puts 'C:' in the URL's host slot."""
    from . import datasets
    d = tempfile.mkdtemp(prefix="zw_doc_")
    try:
        src = os.path.join(d, "probe.bin")
        with open(src, "wb") as f:
            f.write(b"zero-white")
        url = pathlib.Path(src).resolve().as_uri()
        dest = os.path.join(d, "copy.bin")
        datasets._download([url], dest, timeout=10)
        with open(dest, "rb") as f:
            ok = f.read() == b"zero-white"
        out.append(Finding("ok" if ok else "fail", "file-url", f"file:// round trip works ({url[:60]}...)" if ok else "file:// round trip returned wrong bytes"))
    except Exception as e:                                                           # noqa: BLE001
        out.append(Finding("fail", "file-url", f"file:// round trip failed: {type(e).__name__}: {e}",
                           "build file URLs with pathlib.Path(p).resolve().as_uri(), never 'file://' + path"))
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def check_data(out):
    from . import datasets
    d = datasets.data_dir()
    bundled = os.path.join(HERE, "data", "optdigits_uci_sample.csv.gz")
    out.append(Finding("ok" if os.path.exists(bundled) else "fail", "data", "bundled digits sample present" if os.path.exists(bundled)
                       else "bundled digits sample is missing", "" if os.path.exists(bundled) else "reinstall the package"))
    out.append(Finding("ok", "data", f"dataset cache directory: {d} (override with ZERO_WHITE_DATA)"))


def _quarantine(path):
    dst = f"{path}.corrupt-{int(time.time())}"
    fileio.replace(path, dst)
    return dst


def check_workspace(out, ws, fix=False):
    if not os.path.isdir(ws):
        out.append(Finding("warn", "workspace", f"{ws} does not exist yet (nothing to check)"))
        return
    # stale temp files from a hard kill
    stale = []
    for root, _dirs, files in os.walk(ws):
        for fn in files:
            if fn.startswith(".tmp_"):
                p = os.path.join(root, fn)
                if time.time() - os.path.getmtime(p) > 600:
                    stale.append(p)
    if stale and fix:
        for p in stale:
            os.remove(p)
        out.append(Finding("fixed", "workspace", f"removed {len(stale)} stale temp file(s) left by an interrupted save"))
    elif stale:
        out.append(Finding("warn", "workspace", f"{len(stale)} stale temp file(s) from an interrupted save", "doctor --fix removes them"))
    else:
        out.append(Finding("ok", "workspace", "no stale temp files"))
    # journal
    jp = os.path.join(ws, "journal.jsonl")
    if os.path.exists(jp):
        lines = fileio.read_text(jp).splitlines()
        torn = 0
        for ln in lines:
            try:
                json.loads(ln)
            except ValueError:
                torn += 1
        out.append(Finding("ok" if torn <= 1 else "warn", "journal", f"{len(lines)} events, {torn} unreadable line(s)",
                           "" if torn <= 1 else "more than one torn line suggests repeated hard kills or disk errors"))
    # legacy checkpoints (classifier / generator / lm / agent dirs)
    from . import checkpoint
    bad = []
    for root, _dirs, files in os.walk(ws):
        if "weights.npz" in files and "forge" not in root.replace("\\", "/").split("/"):
            try:
                checkpoint.load(root)
            except Exception as e:                                                   # noqa: BLE001
                bad.append((root, f"{type(e).__name__}: {e}"))
    for root, why in bad:
        if fix:
            for fn in ("weights.npz",):
                if os.path.exists(os.path.join(root, fn)):
                    _quarantine(os.path.join(root, fn))
            out.append(Finding("fixed", "checkpoint", f"quarantined unreadable checkpoint in {root} ({why})"))
        else:
            out.append(Finding("fail", "checkpoint", f"unreadable checkpoint in {root}: {why}", "doctor --fix quarantines it"))
    if not bad:
        out.append(Finding("ok", "checkpoint", "all classifier checkpoints load"))
    check_zoo(out, ws, fix)
    check_selfmod(out, ws, fix)


def check_zoo(out, ws, fix):
    zroot = os.path.join(ws, "forge", "zoo")
    if not os.path.isdir(zroot):
        return
    from .forge import task as T
    from .forge.zoo import Zoo
    z, problems = Zoo(ws), []
    for t in z.tasks():
        bp = os.path.join(z.task_dir(t), "best.json")
        if os.path.exists(bp) and z.best(t) is None:
            problems.append(t)
            if fix:
                _quarantine(bp)
    if problems and fix:
        out.append(Finding("fixed", "zoo", f"quarantined dangling champion pointer for: {', '.join(problems)}"))
    elif problems:
        out.append(Finding("fail", "zoo", f"champion pointer broken for: {', '.join(problems)}", "doctor --fix quarantines it; the next solve rebuilds"))
    else:
        out.append(Finding("ok", "zoo", f"{len(z.tasks())} task(s) in the zoo, all champion pointers valid"))


def check_selfmod(out, ws, fix):
    from . import evolved
    from .selfmod import hooks
    from .selfmod.history import HistoryStore
    live = os.path.join(ws, "selfmod", "live")
    hist = os.path.join(ws, "selfmod", "history")
    if os.path.isdir(hist):
        probs = HistoryStore(hist).verify()
        out.append(Finding("ok" if not probs else "fail", "history", "history store hashes verify" if not probs else "; ".join(probs[:3]),
                           "" if not probs else "objects are append-only; a mismatch means disk corruption or tampering"))
    if not os.path.isdir(live):
        out.append(Finding("ok", "hooks", "no evolved hooks installed (genesis behaviour)"))
        return
    mgr = hooks.HookManager(ws)
    gen = hooks.genesis_policy()
    n_live = 0
    for name in evolved.HOOKS:
        src = mgr.live_source(name)
        if src is None:
            continue
        n_live += 1
        try:
            evolved.compile_hook(name, src)
            pol = mgr.policy("live")
            tr_live = hooks.suite_score(pol, hooks.TRAIN_FUNCS, (0, 1), 6)
            tr_gen = hooks.suite_score(gen, hooks.TRAIN_FUNCS, (0, 1), 6)
            faults = pol.faults[name]
            if faults or tr_live < tr_gen - 0.03:
                raise RuntimeError(f"live hook scores {tr_live:.3f} vs genesis {tr_gen:.3f}; {faults} fault(s)")
            out.append(Finding("ok", "hooks", f"live hook '{name}' compiles and scores {tr_live:.3f} (genesis {tr_gen:.3f})"))
        except Exception as e:                                                       # noqa: BLE001
            if fix:
                r = mgr.rollback(name)
                out.append(Finding("fixed", "hooks", f"hook '{name}' was faulty ({e}); rolled back to " +
                                   (f"version {r['version']}" if r else "genesis")))
            else:
                out.append(Finding("fail", "hooks", f"hook '{name}' is faulty: {e}", "doctor --fix rolls it back"))
    if n_live == 0 and not any(f.area == "hooks" for f in out):
        out.append(Finding("ok", "hooks", "no evolved hooks installed (genesis behaviour)"))


def run(workspace=None, fix=False, quick=False):
    out = []
    check_environment(out)
    check_imports(out)
    check_constitution(out)
    check_file_urls(out)
    check_data(out)
    if not quick:
        check_sandbox(out)
    if workspace:
        check_workspace(out, os.path.abspath(workspace), fix)
    return out


def report(findings):
    tag = dict(ok="[ OK ]", warn="[WARN]", fail="[FAIL]", fixed="[FIXD]")
    for f in findings:
        print(f"{tag[f.level]} {f.area:<13} {f.msg}")
        if f.hint:
            print(f"        -> {f.hint}")
    n = {k: sum(1 for f in findings if f.level == k) for k in tag}
    print(f"\n{n['ok']} ok, {n['warn']} warning(s), {n['fail']} failure(s), {n['fixed']} repaired")
    return 1 if n["fail"] else 0
