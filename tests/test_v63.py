"""Tests for 6.3: cost forecasting, Windows parity, the statistical gate and transactional promotion, model growth,
high-dimensional program inputs, and the opt-in forager. Each test is written so that it fails when the property it names breaks."""
import ast
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("ZERO_WHITE_DATA", tempfile.mkdtemp(prefix="zw_test_data_"))

from zerowhite import fileio, forecast                                # noqa: E402
from zerowhite import organism as O                                   # noqa: E402


def tmp():
    return tempfile.mkdtemp(prefix="zw_t63_")


# =============================================================================================================== forecast
def test_the_cost_curve_recovers_superlinear_scaling_and_one_outlier_does_not_move_it():
    pts = [[u, 0.4 * u ** 1.4] for u in (1.75, 7.0, 7.0, 14.0, 28.0, 56.0)]
    a, g = forecast.fit_power(pts)
    assert abs(g - 1.4) < 1e-6 and abs(a - 0.4) < 1e-6                        # exact data: exact recovery
    pts[3][1] *= 6.0                                                          # one throttled / interrupted attempt
    a2, g2 = forecast.fit_power(pts)
    assert abs(g2 - 1.4) < 0.15, g2                                           # the median of pairwise slopes shrugs it off
    assert abs(forecast.predict(pts, 112.0) / (0.4 * 112.0 ** 1.4) - 1.0) < 0.3
    assert forecast.fit_power(pts[:3]) is None                                # too little history: the caller keeps its linear probe
    assert forecast.fit_power([[7.0, 3.0]] * 6) is None                       # six attempts at ONE size say nothing about scaling


def test_damaged_history_is_ignored_not_fatal():
    junk = [None, "x", [1], [0, 5], [-3, 2], [float("nan"), 1], [2, float("inf")], [4.0, 2.0], [8.0, 5.0]]
    assert forecast.clean(junk) == [[4.0, 2.0], [8.0, 5.0]]
    assert forecast.predict(junk, 10.0) is None
    assert forecast.overrun_margin([None, "a", -1, 0, float("nan")], 0.9, 1.5)[:2] == (1.5, "fixed")


def test_the_overrun_margin_is_the_conformal_quantile_and_is_honest_when_history_is_short():
    ratios = [round(0.8 + 0.02 * i, 2) for i in range(19)]                    # 19 ratios, q=0.9 -> rank ceil(20 * 0.9) = 18
    pad, basis, n = forecast.overrun_margin(ratios, 0.9, 1.5)
    assert basis == "calibrated" and n == 19 and pad == sorted(ratios)[17]
    pad, basis, n = forecast.overrun_margin(ratios[:8], 0.9, 1.5)             # 8 ratios cannot support a 90% claim: rank 9 > 8
    assert basis == "fixed" and pad == 1.5
    pad, basis, _ = forecast.overrun_margin([3.0, 1.0, 1.1], 0.9, 1.5)        # ...and an overrun already seen is never ignored
    assert basis == "fixed" and pad == 3.0
    assert forecast.overrun_margin([40.0] * 30, 0.9, 1.5)[0] == forecast.PAD_HI           # bounded
    assert forecast.overrun_margin([0.1] * 30, 0.9, 1.5)[0] == forecast.PAD_LO            # never below the prediction itself


def test_affordable_units_solves_the_fitted_curve_not_a_straight_line():
    cost = lambda u: 0.4 * u ** 1.5                                           # noqa: E731
    u = forecast.affordable_units(cost, 60.0, 1.5, 100.0)
    assert abs(cost(u) * 1.5 - 60.0) < 0.05 and u < 100.0
    assert forecast.affordable_units(cost, 1e9, 1.5, 100.0) == 100.0          # everything fits: the request is returned untouched
    assert forecast.affordable_units(cost, 0.0, 1.5, 100.0) == 0.0            # nothing fits


def _regime(world, seed, calibrate):
    """One organism's life in a synthetic cost world; returns (fitted inside prediction x pad?, pad / actual) per decision."""
    rng, cands = random.Random(seed), 6
    budgets = [0.25, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 8.0, 8.0, 16.0, 16.0, 16.0, 8.0, 4.0, 4.0, 2.0, 2.0, 2.0, 1.0, 1.0]
    spu, obs, ratios, rows = None, [], [], []
    for t, b in enumerate(budgets):
        units = b * (cands + 1)
        linear = None if spu is None else spu * units
        fitted = forecast.predict(obs, units) if calibrate else None
        point = fitted if fitted is not None else linear
        pad = forecast.overrun_margin(ratios, 0.9, 1.5)[0] if calibrate else 1.5
        actual = world(units, t, rng)
        if point is not None:
            rows.append((actual <= point * pad, point * pad / actual))
            if fitted is not None:                                              # as in the organism: only the fitted curve's overruns calibrate the pad
                ratios = forecast.push(ratios, actual / point)
        spu = actual / units if spu is None else 0.5 * (actual / units) + 0.5 * spu
        obs = forecast.push(obs, [units, actual])
    return rows


def test_calibrated_predictions_hold_their_coverage_where_the_fixed_margin_does_not():
    worlds = {
        "linear": lambda u, t, rng: (1.5 + 0.8 * u) * math.exp(rng.gauss(0, .15)),
        "superlinear": lambda u, t, rng: (0.05 + 0.12 * u ** 1.4) * math.exp(rng.gauss(0, .15)),
        "drift": lambda u, t, rng: (0.05 + 0.12 * u ** 1.3) * (1 + 0.04 * t) * math.exp(rng.gauss(0, .15)),
        "spikes": lambda u, t, rng: (0.5 + 0.3 * u) * math.exp(rng.gauss(0, .12)) * (2.2 if rng.random() < .08 else 1.0),
    }
    cov = {}
    for name, w in worlds.items():
        for cal in (False, True):
            rows = [r for s in range(150) for r in _regime(w, s, cal)]
            cov[name, cal] = (sum(r[0] for r in rows) / len(rows), sum(r[1] for r in rows) / len(rows))
    for name in worlds:                                                       # the target is 0.90; allow sampling noise either way
        assert cov[name, True][0] > 0.86, (name, cov[name, True])
        assert cov[name, True][1] < 2.2, (name, cov[name, True])              # ...without hoarding the time budget
    assert cov["superlinear", True][0] > cov["superlinear", False][0] + 0.03, cov     # where the old model is optimistic, the new one is not
    assert cov["drift", True][0] > cov["drift", False][0] - 0.03, cov                 # under drift about as good, with a narrower pad (exchangeability is violated there)
    assert cov["linear", True][1] < cov["linear", False][1]                            # and where the old pad was needlessly wide, the new one is narrower


# a scripted organism, as in test_v6
class Scripted(O.Organism):
    script = {}
    calls = []

    def _act_solve(self, dec, t0):
        seq = self.script[dec["task"]]
        skill, secs = seq.pop(0) if len(seq) > 1 else seq[0]
        self.calls.append((dec["task"], round(dec["budget"], 3)))
        prev = self.state["tasks"][dec["task"]]["best_skill"]
        return dict(ok=True, skill=skill, improved=prev is None or skill > prev + 1e-9, family="mlp", seconds=secs)


def _scripted_organism(script, **kw):
    import test_v6 as T6
    inbox, ws = tmp(), tmp()
    for i, name in enumerate(script):
        T6.put(inbox, f"{name}.csv", T6.csv_bytes(100, seed=100 + i))
    settings = dict(inbox=[inbox], quiet=True, probe_budget=0.25, standard_budget=1.0, patience=50, min_gain=0.001, doctor_every=0)
    settings.update(kw)
    cls = type("S63", (Scripted,), dict(script={k: list(v) for k, v in script.items()}, calls=[]))
    return cls(ws, O.Settings(**settings)), cls


def test_the_organism_records_what_it_predicted_and_switches_to_the_fitted_curve_and_a_measured_pad():
    # cost grows like budget**1.5: a linear extrapolation from the small first attempt is optimistic by construction
    secs = [0.02 * (7 * b) ** 1.5 for b in (0.25,) + (1.0,) * 14]
    script = {"t": [(0.30 + 0.01 * i, s) for i, s in enumerate(secs)]}
    org, cls = _scripted_organism(script, time_margin=1.5)
    for _ in range(len(secs)):
        org.step()
    ts = org.state["tasks"]["t"]
    assert len(ts["cost_obs"]) == len(secs) and all(len(o) == 2 for o in ts["cost_obs"])
    assert 0 < len(ts["cost_ratios"]) <= len(secs) - 1                        # the first attempt had no prediction to be wrong about
    fitted = forecast.predict(ts["cost_obs"], 7.0)
    assert fitted is not None and org.cost(ts, 1.0) == fitted                 # decisions use the fitted curve once the history supports one
    pad, basis, n = org.pad(ts)
    assert basis == "calibrated" and n >= 10 and 1.0 <= pad <= forecast.PAD_HI
    org.s.calibrate = False                                                   # the old behaviour is one switch away, exactly
    assert org.cost(ts, 1.0) == ts["sec_per_unit"] * 7.0 and org.pad(ts) == (1.5, "fixed", 0)


def test_with_too_little_history_the_organism_behaves_exactly_as_before():
    org, cls = _scripted_organism({"t": [(0.5, 100.0), (0.55, 100.0)]}, minutes=1.0, patience=2, min_gain=0.01)
    org.step()
    ts = org.state["tasks"]["t"]
    assert org.cost(ts, 1.0) == ts["sec_per_unit"] * 7.0 and org.pad(ts) == (1.5, "fixed", 0)
    dec = org.decide(dict(free_gb=8.0, new_tasks=[], counts={}))
    assert dec["budget"] < 1.0 and "cut" in dec["reason"] and "safety" in dec["reason"]


# ============================================================================================================ windows parity
def test_job_object_structures_have_the_documented_windows_layout_on_any_os():
    import ctypes
    from zerowhite.selfmod import winjob as W
    p = ctypes.sizeof(ctypes.c_void_p)
    ext, basic = W.JOBOBJECT_EXTENDED_LIMIT_INFORMATION, W.JOBOBJECT_BASIC_LIMIT_INFORMATION
    assert ctypes.sizeof(W.IO_COUNTERS) == 48
    assert ctypes.sizeof(basic) == (64 if p == 8 else 48)                    # winnt.h, x64 / x86
    assert ctypes.sizeof(ext) == (144 if p == 8 else 112)
    if p == 8:                                                               # the offsets SetInformationJobObject reads
        assert basic.LimitFlags.offset == 16 and basic.ActiveProcessLimit.offset == 40
        assert ext.IoInfo.offset == 64 and ext.ProcessMemoryLimit.offset == 112 and ext.JobMemoryLimit.offset == 120


class FakeKernel32:
    def __init__(self, fail=None):
        self.calls, self.fail, self.info = [], fail, None

    def CreateJobObjectW(self, attrs, name):
        self.calls.append("create")
        return None if self.fail == "create" else 0xBEEF

    def SetInformationJobObject(self, h, cls, ref, size):
        import ctypes
        self.calls.append(("set", cls, size))
        self.info = ctypes.cast(ref, ctypes.POINTER(__import__("zerowhite.selfmod.winjob", fromlist=["x"]).JOBOBJECT_EXTENDED_LIMIT_INFORMATION)).contents
        self.snapshot = (self.info.BasicLimitInformation.LimitFlags, self.info.JobMemoryLimit, self.info.BasicLimitInformation.PerProcessUserTimeLimit,
                         self.info.BasicLimitInformation.ActiveProcessLimit)
        return 0 if self.fail == "set" else 1

    def AssignProcessToJobObject(self, h, ph):
        self.calls.append(("assign", h, ph))
        return 0 if self.fail == "assign" else 1

    def TerminateJobObject(self, h, code):
        self.calls.append(("terminate", h, code))
        return 1

    def CloseHandle(self, h):
        self.calls.append(("close", h))
        return 1


def test_a_job_object_is_created_limited_assigned_terminated_and_closed_in_that_order_and_never_raises():
    from zerowhite.selfmod import winjob as W
    k = FakeKernel32()
    job = W.JobObject(mem_mb=512, cpu_s=30, active_processes=4, api=k)
    assert job.ok and k.calls[:2] == ["create", ("set", W.JobObjectExtendedLimitInformation, __import__("ctypes").sizeof(W.JOBOBJECT_EXTENDED_LIMIT_INFORMATION))]
    flags, mem, cpu, active = k.snapshot
    want = (W.JOB_OBJECT_LIMIT_JOB_MEMORY | W.JOB_OBJECT_LIMIT_PROCESS_TIME | W.JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | W.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | W.JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION)
    assert flags == want and mem == 512 * 2 ** 20 and cpu == 30 * 10_000_000 and active == 4
    popen = type("P", (), {"_handle": 1234})()
    assert job.assign(popen) and k.calls[-1] == ("assign", 0xBEEF, 1234)
    assert job.assign(object()) is False and "handle" in job.error           # a child without a handle: reported, not raised
    assert job.terminate(7) and k.calls[-1] == ("terminate", 0xBEEF, 7)
    job.close()
    assert k.calls[-1] == ("close", 0xBEEF) and not job.ok and job.assign(popen) is False and job.terminate() is False
    for stage in ("create", "set"):                                          # each way the OS can refuse: a quiet False, no exception
        j = W.JobObject(mem_mb=64, api=FakeKernel32(fail=stage))
        assert not j.ok and j.error and j.assign(popen) is False
    j = W.JobObject(mem_mb=64, api=FakeKernel32(fail="assign"))
    assert j.ok and j.assign(popen) is False and "AssignProcessToJobObject" in j.error
    assert W.JobObject(mem_mb=64).ok is (os.name == "nt" and W.available())  # on this OS: honest about not being on Windows


def test_the_checkpoint_writer_and_every_file_swap_retry_on_windows_and_nothing_bypasses_them():
    class FakeOS:                                                            # delegates everything to os, but is "nt" and fails replace twice
        name = "nt"

        def __init__(self):
            self.n = 0

        def __getattr__(self, k):
            return getattr(os, k)

        def replace(self, a, b):
            self.n += 1
            if self.n <= 2:
                raise PermissionError(13, "sharing violation")
            return os.replace(a, b)

    d = tmp()
    real_os, real_sleep = fileio.os, fileio.time.sleep
    fake = FakeOS()
    fileio.os, fileio.time.sleep = fake, (lambda s: None)
    try:
        fileio.save_npz(os.path.join(d, "w.npz"), {"a": np.arange(5)})       # used to be a bare os.replace: PermissionError on Windows
    finally:
        fileio.os, fileio.time.sleep = real_os, real_sleep
    assert fake.n == 3 and fileio.load_npz(os.path.join(d, "w.npz"))["a"].tolist() == [0, 1, 2, 3, 4]
    assert not [f for f in os.listdir(d) if f.startswith(".tmp_")]
    boom = {"n": 0}

    def always(a, b):
        boom["n"] += 1
        raise PermissionError(13, "denied")
    real_replace = os.replace
    os.replace = always
    try:
        try:
            fileio.replace("x", "y")                                         # POSIX: no retry, the error is real
            raise AssertionError("expected PermissionError")
        except PermissionError:
            pass
    finally:
        os.replace = real_replace
    assert boom["n"] == 1 if os.name != "nt" else boom["n"] == 8
    pkg = os.path.join(ROOT, "zerowhite")                                    # the guard: no module outside fileio.py calls os.replace directly
    for path in glob.glob(os.path.join(pkg, "**", "*.py"), recursive=True):
        rel = os.path.relpath(path, pkg).replace(os.sep, "/")
        if rel in ("fileio.py", "selfmod/watchdog.py"):      # watchdog is standalone by design (stdlib only) and carries its own retrying swap
            continue
        for node in ast.walk(ast.parse(fileio.read_text(path))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "replace" \
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                raise AssertionError(f"{rel}:{node.lineno} calls os.replace directly; use fileio.replace (Windows retries)")


def test_the_memory_limit_is_enforced_and_the_report_and_doctor_say_so():
    from zerowhite import doctor
    from zerowhite.selfmod import sandbox
    rep = sandbox.sandbox_report()
    assert rep["rlimits"] is True and rep["job_object"] is (os.name == "nt" and rep["job_object"])
    ok = sandbox.run_code("b = bytearray(16 * 2**20); print('alive')", mem_mb=256)
    hog = sandbox.run_code("b = bytearray(900 * 2**20); b[::4096] = bytes(len(b[::4096])); print('alive')", mem_mb=256, timeout=30)
    assert ok.stdout.strip() == "alive" and hog.stdout.strip() != "alive" and not hog.ok      # the limit really bites
    out = []
    doctor.check_sandbox(out)
    assert not [f for f in out if f.level == "fail"], [f.msg for f in out]
    assert not [f for f in out if "memory limit" in f.msg], [f.msg for f in out]


def test_on_windows_a_real_job_object_kills_a_memory_hog():                  # runs on the windows-latest CI leg; a no-op elsewhere
    if os.name != "nt":
        return
    from zerowhite.selfmod import sandbox, winjob
    assert winjob.available() and sandbox.sandbox_report()["job_object"]
    hog = sandbox.run_code("b = bytearray(900 * 2**20); b[::4096] = bytes(len(b[::4096])); print('alive')", mem_mb=256, timeout=60)
    assert hog.stdout.strip() != "alive"


# ================================================================================================== statistical gate + promotion
def test_the_significance_gate_rejects_one_lucky_seed_and_accepts_a_consistent_gain():
    from zerowhite.selfmod import stats
    base = [0.50] * 6
    lucky = [0.70, 0.49, 0.49, 0.495, 0.49, 0.49]                              # mean +0.0275: clears a 0.004 margin on one seed alone
    v = stats.gate(base, lucky, margin=0.004)
    assert np.mean(lucky) - 0.5 > 0.004 and not v["accept"] and v["positive"] == 1 and "not significant" in v["reason"]
    steady = [0.51, 0.512, 0.509, 0.511, 0.513, 0.510]
    v = stats.gate(base, steady, margin=0.004)
    assert v["accept"] and abs(v["p"] - 1 / 64) < 1e-12 and v["ci"][0] > 0
    assert not stats.gate(base, steady, margin=0.02)["accept"]                  # significant but too small: the margin still applies
    assert "fewer than 5" in stats.gate([0, 0, 0, 0], [1, 1, 1, 1], margin=0.1)["reason"]    # 4 pairs can never reach p < 0.05
    assert stats.gate(base, lucky, margin=0.004, alpha=None)["accept"]          # alpha=None is exactly the pre-6.3 rule
    try:
        stats.gate([1, 2], [1], margin=0)
        raise AssertionError("unpaired scores must be refused")
    except ValueError:
        pass


def test_signflip_p_is_exact_for_small_n_and_calibrated_under_the_null_for_large_n():
    from zerowhite.selfmod import stats
    assert abs(stats.signflip_p([1, 2, 3]) - 1 / 8) < 1e-12 and stats.signflip_p([-1, -2, -3]) == 1.0
    assert abs(stats.signflip_p([1, -1]) - 0.75) < 1e-12                        # {+,+}, {+,-}, {-,+} all reach mean >= 0
    rng = np.random.default_rng(1)
    ps = [stats.signflip_p(rng.normal(size=24), n_resamples=2000, seed=k) for k in range(300)]
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.09, np.mean(np.array(ps) < 0.05)   # false positives ~5% when nothing is there
    pw = [stats.signflip_p(rng.normal(0.8, 1.0, size=24), n_resamples=2000, seed=k) for k in range(100)]
    assert np.mean(np.array(pw) < 0.05) > 0.85                                  # and it does notice a real effect


def _mini_repo():
    """A tiny repo with a sealed constitution and one mutable file, laid out like the real one."""
    import hashlib
    repo = tmp()
    pkg = os.path.join(repo, "zerowhite")
    os.makedirs(os.path.join(pkg, "selfmod"))
    fileio.write_text(os.path.join(pkg, "selfmod", "patcher.py"), 'MUTABLE = ("evolve.py", "agents.py")\n')
    fileio.write_text(os.path.join(pkg, "evolve.py"), "SIGMA = 0.5\n")
    fileio.write_text(os.path.join(pkg, "agents.py"), "LR = 0.1\n")
    seal = {"selfmod/patcher.py": hashlib.sha256(open(os.path.join(pkg, "selfmod", "patcher.py"), "rb").read()).hexdigest()}
    fileio.write_text(os.path.join(pkg, "CONSTITUTION.sha256"), json.dumps(seal))
    return repo


def test_a_promotion_commits_atomically_and_every_failure_mode_restores_the_originals():
    from zerowhite.selfmod import watchdog as W
    repo = _mini_repo()
    ev, ag = os.path.join(repo, "zerowhite", "evolve.py"), os.path.join(repo, "zerowhite", "agents.py")
    allowed = ["zerowhite/evolve.py", "zerowhite/agents.py"]
    new = {"zerowhite/evolve.py": "SIGMA = 0.9\n", "zerowhite/agents.py": "LR = 0.3\n"}
    r = W.run(repo, new, allowed, verify=lambda: (True, "ok"))
    assert r["ok"] and r["state"] == "committed" and fileio.read_text(ev) == "SIGMA = 0.9\n" and fileio.read_text(ag) == "LR = 0.3\n"
    assert W.run(repo, new, allowed)["state"] == "unchanged"                    # same content again: no transaction at all
    back = {"zerowhite/evolve.py": "SIGMA = 0.5\n", "zerowhite/agents.py": "LR = 0.1\n"}
    r = W.run(repo, back, allowed, verify=lambda: (False, "2 tests failed"))     # verification fails on the real tree
    assert not r["ok"] and "2 tests failed" in r["reason"] and fileio.read_text(ev) == "SIGMA = 0.9\n"   # nothing changed
    boom = lambda: (_ for _ in ()).throw(RuntimeError("kaboom"))                # noqa: E731
    r = W.run(repo, back, allowed, verify=boom)
    assert not r["ok"] and "kaboom" in r["reason"] and fileio.read_text(ag) == "LR = 0.3\n"
    for bad in ("zerowhite/selfmod/patcher.py", "../escape.py", "/etc/passwd", "zerowhite/missing.py", "C:/x.py"):
        try:
            W.run(repo, {bad: "x"}, allowed + ["zerowhite/missing.py"])
            raise AssertionError(f"{bad} must be refused")
        except PermissionError:
            pass
    assert not os.path.exists(os.path.join(repo, "escape.py"))
    js = [json.loads(open(os.path.join(repo, W.JOURNAL_DIR, n, "journal.json")).read()) for n in os.listdir(os.path.join(repo, W.JOURNAL_DIR))]
    assert [j["state"] for j in sorted(js, key=lambda j: j["started"])] == ["committed", "rolled_back", "rolled_back"]   # failures are on record too


def test_a_crash_between_two_file_writes_is_repaired_at_the_next_start():
    from zerowhite.selfmod import watchdog as W
    repo = _mini_repo()
    ev, ag = os.path.join(repo, "zerowhite", "evolve.py"), os.path.join(repo, "zerowhite", "agents.py")
    allowed = ["zerowhite/evolve.py", "zerowhite/agents.py"]
    changes = {"zerowhite/evolve.py": "SIGMA = 0.9\n", "zerowhite/agents.py": "LR = 0.3\n"}
    real_write, calls = W._atomic_write, {"live": 0}

    def dying(path, data, fsync=True):                                          # the process dies after the first live file is swapped in
        if path in (ev, ag):
            calls["live"] += 1
            if calls["live"] == 2:
                raise SystemExit("power cut")
        return real_write(path, data, fsync)
    W._atomic_write = dying
    try:
        try:
            W.run(repo, changes, allowed)
        except SystemExit:
            pass
    finally:
        W._atomic_write = real_write
    # simulate "the process really died": rollback inside run() would have saved us, so undo ITS repair and recreate the crashed state
    j = [n for n in os.listdir(os.path.join(repo, W.JOURNAL_DIR))][0]
    tx = os.path.join(repo, W.JOURNAL_DIR, j)
    fileio.write_text(ev, "SIGMA = 0.9\n")                                       # half applied: first file new, second file old
    fileio.write_text(ag, "LR = 0.1\n")
    jj = json.loads(open(os.path.join(tx, "journal.json")).read())
    jj["state"] = "applying"
    fileio.write_text(os.path.join(tx, "journal.json"), json.dumps(jj))
    found = W.recover(repo)
    assert found and found[0]["state_found"] == "applying"
    assert fileio.read_text(ev) == "SIGMA = 0.5\n" and fileio.read_text(ag) == "LR = 0.1\n"      # both originals are back
    assert W.recover(repo) == []                                                # idempotent


def test_a_promotion_that_would_touch_the_sealed_constitution_is_refused_and_a_broken_seal_blocks_promotion():
    from zerowhite.selfmod import watchdog as W
    repo = _mini_repo()
    ev = os.path.join(repo, "zerowhite", "evolve.py")
    patcher_path = os.path.join(repo, "zerowhite", "selfmod", "patcher.py")

    def sneaky():                                                               # verification code that tampers with a sealed file
        fileio.write_text(patcher_path, 'MUTABLE = ("evolve.py", "selfmod/safety.py")\n')
        return True
    r = W.run(repo, {"zerowhite/evolve.py": "SIGMA = 0.7\n"}, ["zerowhite/evolve.py"], verify=sneaky)
    assert False if r["ok"] else "sealed" in r["reason"]
    fileio.write_text(patcher_path, 'MUTABLE = ("evolve.py", "agents.py")\n')    # restore the seal's file and check the original survived
    assert fileio.read_text(ev) == "SIGMA = 0.5\n"
    fileio.write_text(patcher_path, "# someone edited the constitution\nMUTABLE = ()\n")
    r = W.run(repo, {"zerowhite/evolve.py": "SIGMA = 0.7\n"}, ["zerowhite/evolve.py"])
    assert r["state"] == "refused" and "doctor --seal" in r["reason"] and fileio.read_text(ev) == "SIGMA = 0.5\n"


def test_the_watchdog_runs_as_its_own_process_and_reads_its_allow_list_from_the_sealed_patcher():
    repo = _mini_repo()
    ev = os.path.join(repo, "zerowhite", "evolve.py")
    cs = os.path.join(repo, "cs.json")
    fileio.write_json(cs, {"zerowhite/evolve.py": "SIGMA = 0.8\n"})
    script = os.path.join(ROOT, "zerowhite", "selfmod", "watchdog.py")
    r = subprocess.run([sys.executable, script, "promote", "--repo", repo, "--changeset", cs, "--verify-cmd", '["python", "-c", "import sys; sys.exit(0)"]'],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and json.loads(r.stdout)["state"] == "committed" and fileio.read_text(ev) == "SIGMA = 0.8\n", r.stdout + r.stderr
    fileio.write_json(cs, {"zerowhite/evolve.py": "SIGMA = 0.1\n"})
    r = subprocess.run([sys.executable, script, "promote", "--repo", repo, "--changeset", cs, "--verify-cmd", '["python", "-c", "import sys; sys.exit(3)"]'],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and json.loads(r.stdout)["state"] == "rolled_back" and fileio.read_text(ev) == "SIGMA = 0.8\n"
    fileio.write_json(cs, {"zerowhite/selfmod/patcher.py": "MUTABLE = ()\n"})    # the changeset cannot widen what may change
    r = subprocess.run([sys.executable, script, "promote", "--repo", repo, "--changeset", cs], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "PermissionError" in r.stderr


def test_the_patcher_applies_through_the_journal_and_reports_the_significance_gate():
    from zerowhite.selfmod import patcher
    from zerowhite.selfmod.history import HistoryStore
    import re
    repo = os.path.join(tmp(), "repo")
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns("__pycache__", ".git", "runs", "*.egg-info", ".zw_promotion"))
    store = HistoryStore(tmp())

    def fake_bench(tree_root, seeds, timeout=0):
        txt = fileio.read_text(os.path.join(tree_root, "zerowhite", "evolve.py"))
        s = float(re.search(r"sigma=([0-9.e-]+)[,)]", txt.split("class Evolver")[1].split("def step")[0]).group(1))
        return 1.0 - abs(s - 0.9) + 0.001 * (seeds[0] % 3)                       # per-seed variation, identical for both trees
    real_b, real_t = patcher.bench_in, patcher.tests_pass
    patcher.bench_in, patcher.tests_pass = fake_bench, (lambda root, timeout=0: (True, "stub ok"))
    try:
        rep = patcher.search(repo, store, evals=60, seed=1, apply=True, verbose=False, files=("evolve.py",), margin=0.0)
    finally:
        patcher.bench_in, patcher.tests_pass = real_b, real_t
    assert rep.applied and rep.gate["accept"] and rep.gate["n"] == 6 and rep.gate["p"] <= 1 / 64 + 1e-12
    states = [json.loads(open(os.path.join(repo, ".zw_promotion", n, "journal.json")).read())["state"] for n in os.listdir(os.path.join(repo, ".zw_promotion"))]
    assert states == ["committed"]
    assert fileio.read_text(os.path.join(repo, "zerowhite", "evolve.py")) != fileio.read_text(os.path.join(ROOT, "zerowhite", "evolve.py"))


def test_the_patcher_final_gate_rejects_a_gain_that_comes_from_one_seed():
    from zerowhite.selfmod import patcher
    from zerowhite.selfmod.history import HistoryStore
    repo = os.path.join(tmp(), "repo")
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns("__pycache__", ".git", "runs", "*.egg-info", ".zw_promotion"))
    before = fileio.read_text(os.path.join(repo, "zerowhite", "evolve.py"))
    real_b, real_t = patcher.bench_in, patcher.tests_pass
    patched = {"flag": False}

    def bench(tree_root, seeds, timeout=0):
        txt = fileio.read_text(os.path.join(tree_root, "zerowhite", "evolve.py"))
        changed = txt != before
        if seeds[0] >= 100:                                                      # the FINAL seeds: a big gain on seed 100 only
            return 0.5 + (0.2 if (changed and seeds[0] == 100) else (-0.002 if changed else 0.0))
        return 0.5 + (0.01 if changed else 0.0)                                  # select / confirm seeds look fine, so a patch is proposed
    patcher.bench_in, patcher.tests_pass = bench, (lambda root, timeout=0: (True, "stub ok"))
    try:
        rep = patcher.search(repo, HistoryStore(tmp()), evals=6, seed=0, apply=True, verbose=False, files=("evolve.py",), margin=0.004)
        legacy = patcher.search(repo, HistoryStore(tmp()), evals=6, seed=0, apply=False, verbose=False, files=("evolve.py",), margin=0.004, alpha=None)
    finally:
        patcher.bench_in, patcher.tests_pass = real_b, real_t
    assert rep.final_patched > rep.final_base + 0.004                            # the old rule (mean + margin) WOULD have accepted it ...
    assert not rep.applied and rep.gate["positive"] == 1 and "significance gate" in rep.note       # ... the paired test does not
    assert fileio.read_text(os.path.join(repo, "zerowhite", "evolve.py")) == before
    assert legacy.gate["accept"] and "NOT applied" in legacy.note                # alpha=None keeps the margin-only behaviour


# ======================================================================================================================= forager
def _pages(table):
    """A fake opener over {url: (status, content_type, body, location)}; records every URL it is asked for."""
    seen = []

    class H(dict):
        def get(self, k, d=None):
            return dict.get(self, k, d)

    def opener(url, timeout, max_bytes):
        seen.append(url)
        status, ctype, body, loc = table.get(url, (404, "text/plain", b"", None))
        h = H({"Content-Type": ctype})
        if loc:
            h["Location"] = loc
        return status, h, body if isinstance(body, bytes) else body.encode("utf-8"), loc
    return opener, seen


def _forager(table, hosts=("docs.example.org",), resolver=None, **pol):
    from zerowhite.forage import Forager, Policy
    opener, seen = _pages(table)
    d = tmp()
    clock = {"t": 0.0}
    sleeps = []
    f = Forager(Policy(allow_hosts=hosts, delay=pol.pop("delay", 2.0), **pol), os.path.join(d, "inbox"), os.path.join(d, "ledger.jsonl"), opener=opener,
                resolver=resolver or (lambda h: ["93.184.216.34"]), sleep=lambda s: (sleeps.append(s), clock.__setitem__("t", clock["t"] + s)),
                clock=lambda: clock["t"])
    return f, seen, sleeps, d


PAGE = "<html><head><title>t</title><script>steal()</script></head><body><h1>Title</h1><p>" + ("A useful sentence about numpy broadcasting. " * 8) + "</p></body></html>"


def test_the_forager_fetches_nothing_unless_a_host_is_allowed_and_every_refusal_never_touches_the_network():
    base = "https://docs.example.org/a"
    table = {base: (200, "text/html", PAGE, None)}
    f, seen, _, d = _forager(table, hosts=())
    assert "no host is allowed" in f.fetch(base)["reason"]
    f, seen, _, d = _forager(table)
    cases = {"https://evil.example.com/a": "not on the allow-list", "http://docs.example.org/a": "plain http is refused",
             "ftp://docs.example.org/a": "scheme", "file:///etc/passwd": "scheme", "https://user:pw@docs.example.org/a": "credentials",
             "https://docs.example.org:99999/a": "not a valid URL"}
    for url, why in cases.items():
        r = f.fetch(url)
        assert not r["ok"] and why in r["reason"], (url, r["reason"])
    assert seen == []                                                          # not one request was made for any of them
    f2, seen2, _, _ = _forager(table, resolver=lambda h: ["10.0.0.5"])         # an allow-listed name that points inside the network
    assert "private/loopback" in f2.fetch(base)["reason"] and seen2 == []
    f3, seen3, _, _ = _forager(table, resolver=lambda h: ["10.0.0.5"], allow_local=True, respect_robots=False)
    assert f3.fetch(base)["ok"] and seen3 == [base]                            # ... unless you said that is what you mean
    lines = [json.loads(x) for x in open(os.path.join(d, "ledger.jsonl"), encoding="utf-8")]
    assert len(lines) == len(cases) and all(not x["ok"] for x in lines)       # refusals are on the record


def test_the_forager_follows_redirects_by_hand_and_never_leaves_the_allow_list():
    a = "https://docs.example.org/a"
    table = {a: (302, "text/html", b"", "https://evil.example.com/steal"), "https://evil.example.com/steal": (200, "text/html", PAGE, None)}
    f, seen, _, _ = _forager(table, respect_robots=False)
    r = f.fetch(a)
    assert not r["ok"] and "not on the allow-list" in r["reason"] and seen == [a]       # the second hop was refused BEFORE it was requested
    table = {a: (301, "text/html", b"", "/b"), "https://docs.example.org/b": (200, "text/html", PAGE, None)}
    f, seen, _, _ = _forager(table, respect_robots=False)
    assert f.fetch(a)["ok"] and seen == [a, "https://docs.example.org/b"]                # a relative redirect inside the host is fine
    loop = {"https://docs.example.org/x": (302, "text/html", b"", "/x")}
    f, seen, _, _ = _forager(loop, respect_robots=False, redirects=3)
    assert "redirects" in f.fetch("https://docs.example.org/x")["reason"] and len(seen) == 4


def test_the_forager_honours_robots_txt_and_treats_an_unreadable_one_as_no():
    page = "https://docs.example.org/private/x"
    robots = "https://docs.example.org/robots.txt"
    f, seen, _, _ = _forager({page: (200, "text/html", PAGE, None), robots: (200, "text/plain", "User-agent: *\nDisallow: /private/\n", None)})
    r = f.fetch(page)
    assert not r["ok"] and "robots.txt" in r["reason"] and page not in seen
    ok_page = "https://docs.example.org/public/x"
    assert _forager({ok_page: (200, "text/html", PAGE, None), robots: (200, "text/plain", "User-agent: *\nDisallow: /private/\n", None)})[0].fetch(ok_page)["ok"]
    assert _forager({ok_page: (200, "text/html", PAGE, None), robots: (500, "text/plain", "", None)})[0].fetch(ok_page)["ok"] is False   # 5xx: do not fetch
    assert _forager({ok_page: (200, "text/html", PAGE, None)})[0].fetch(ok_page)["ok"]                                              # no robots.txt: allowed


def test_the_forager_keeps_only_text_within_the_limits_and_reduces_html_to_its_words():
    ok, big, binary, tiny = ("https://docs.example.org/" + n for n in ("ok", "big", "bin", "tiny"))
    table = {ok: (200, "text/html; charset=utf-8", PAGE, None), big: (200, "text/plain", "x" * 5000, None),
             binary: (200, "application/octet-stream", b"\x00\x01", None), tiny: (200, "text/plain", "hi", None)}
    f, seen, sleeps, d = _forager(table, respect_robots=False, max_bytes=4096, delay=2.0)
    r = f.fetch(ok)
    text = open(r["path"], encoding="utf-8").read()
    assert r["ok"] and "numpy broadcasting" in text and "steal()" not in text and "<h1>" not in text and text.startswith("Title")
    assert not f.fetch(big)["ok"] and "limit" in f.fetch(big)["reason"]
    assert "not text" in f.fetch(binary)["reason"] and "characters of text" in f.fetch(tiny)["reason"]
    again = f.fetch(ok)
    assert again["ok"] and "already" in again["reason"] and len(os.listdir(os.path.join(d, "inbox"))) == 1      # identical text is stored once
    assert sleeps and all(abs(s - 2.0) < 1e-9 or s < 2.0 for s in sleeps)                                       # paced: never faster than `delay`


def test_the_forager_stops_at_its_url_and_byte_limits():
    table = {f"https://docs.example.org/{i}": (200, "text/plain", ("line %d of text. " % i) * 20, None) for i in range(6)}
    f, seen, _, _ = _forager(table, respect_robots=False, max_urls=3, delay=0.0)
    rows = f.fetch_all(list(table))
    assert [r["ok"] for r in rows] == [True, True, True, False, False, False] and "per-run limit" in rows[3]["reason"]
    f, seen, _, _ = _forager(table, respect_robots=False, max_total_bytes=500, delay=0.0)
    rows = f.fetch_all(list(table))
    assert sum(r["ok"] for r in rows) == 2 and "byte limit" in rows[-1]["reason"]


def _serve(routes):
    """A real HTTP server on 127.0.0.1 (random port) answering from {path: (status, content_type, body, location)}."""
    import http.server

    class H(http.server.BaseHTTPRequestHandler):
        hits = []

        def do_GET(self):
            H.hits.append(self.path)
            status, ctype, body, loc = routes.get(self.path, (404, "text/plain", b"", None))
            body = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if loc:
                self.send_header("Location", loc)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, H


def test_the_forager_reads_a_real_page_end_to_end_and_the_cli_refuses_without_an_allow_list():
    from zerowhite.forage import Forager, Policy
    srv, H = _serve({"/doc": (200, "text/html; charset=utf-8", PAGE, None), "/go": (302, "text/html", b"", "http://example.invalid/x")})
    try:
        d = tmp()
        f = Forager(Policy(allow_hosts=("127.0.0.1",), allow_local=True, delay=0.0), os.path.join(d, "in"), os.path.join(d, "led.jsonl"))
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        r = f.fetch(base + "/doc")
        assert r["ok"] and "numpy broadcasting" in open(r["path"], encoding="utf-8").read()
        assert "/robots.txt" in H.hits and "/doc" in H.hits                                   # it asked robots.txt first, as promised
        r = f.fetch(base + "/go")
        assert not r["ok"] and "not on the allow-list" in r["reason"] and "/x" not in H.hits   # the redirect target was never contacted
        denied = Forager(Policy(allow_hosts=("127.0.0.1",), delay=0.0), os.path.join(d, "in2"), os.path.join(d, "led2.jsonl"))
        assert "private/loopback" in denied.fetch(base + "/doc")["reason"]                    # loopback needs allow_local even when listed
        ws = tmp()
        env = dict(os.environ, ZERO_WHITE_DATA=os.environ["ZERO_WHITE_DATA"])
        none = subprocess.run([sys.executable, "-m", "zerowhite", "forage", base + "/doc", "--workspace", ws], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        assert none.returncode == 2 and "--allow-host" in none.stderr and not os.path.exists(os.path.join(ws, "forage"))
        ok = subprocess.run([sys.executable, "-m", "zerowhite", "forage", base + "/doc", "--workspace", ws, "--allow-host", "127.0.0.1", "--allow-local",
                             "--delay", "0"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        assert ok.returncode == 0 and os.listdir(os.path.join(ws, "forage", "inbox")) and "ledger.jsonl" in ok.stdout, ok.stdout + ok.stderr
    finally:
        srv.shutdown()


def test_only_the_forager_and_its_cli_command_know_about_the_forager_and_it_is_sealed():
    pkg = os.path.join(ROOT, "zerowhite")
    for path in glob.glob(os.path.join(pkg, "**", "*.py"), recursive=True):
        rel = os.path.relpath(path, pkg).replace(os.sep, "/")
        if rel.startswith("forage/") or rel in ("cli_v6.py", "organism.py"):             # 6.4: the organism may use it, lazily inside _act_forage (checked below), opt-in only
            continue
        for node in ast.walk(ast.parse(fileio.read_text(path))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module or ""] + [a.name for a in node.names]) if isinstance(node, ast.ImportFrom) else []
            assert not any("forage" in n for n in names), f"{rel} imports the forager: nothing autonomous may reach the network through it"
    org = fileio.read_text(os.path.join(pkg, "organism.py"))                          # ... and ONLY lazily, and only behind the opt-in
    top = [n for n in ast.parse(org).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("forage" in (getattr(n, "module", "") or "") + " ".join(a.name for a in n.names) for n in top), "organism.py must import the forager inside _act_forage only"
    assert "def _forage_due" in org and "self.s.forage_hosts and self.s.forage_wishlist" in org.replace("s.forage_hosts and s.forage_wishlist", "self.s.forage_hosts and self.s.forage_wishlist")
    from zerowhite import doctor
    assert "forage/fetch.py" in doctor.CONSTITUTION and "selfmod/watchdog.py" in doctor.CONSTITUTION and "selfmod/stats.py" in doctor.CONSTITUTION
    from zerowhite.selfmod import patcher
    assert not any(f.startswith(("forage", "selfmod")) for f in patcher.MUTABLE)


# ============================================================================================================= model growth
def _gpt(dtype="float64", tie=True, seed=0, layers=2, heads=2, d=16, ffn=24):
    from zerowhite.chat.model import GPT, GPTConfig
    return GPT(GPTConfig(vocab=23, d_model=d, n_layer=layers, n_head=heads, ffn=ffn, ctx=16, tie=tie, dtype=dtype), seed=seed, init_std=0.3)


def _gbatch(vocab=23, B=3, T=10, seed=1):
    r = np.random.default_rng(seed)
    ids, tgt = r.integers(1, vocab, (B, T)), r.integers(1, vocab, (B, T))
    w = (r.random((B, T)) > 0.3).astype(np.float32)
    pos = np.tile(np.arange(T), (B, 1))
    seg = np.ones((B, T), dtype=np.int64)
    seg[:, T // 2:] = 2
    pos[:, T // 2:] -= T // 2
    return dict(ids=ids, tgt=tgt, w=w, pos=pos, seg=seg)


def _logits(m, b):
    xf, _ = m._forward(b, keep=False)
    E = m.p["tok_emb"] if m.cfg.tie else m.p["lm_head"]
    return np.asarray(xf).reshape(-1, m.cfg.d_model) @ np.asarray(E).T


def test_growing_layers_or_ffn_units_leaves_the_model_function_exactly_unchanged():
    from zerowhite.chat.grow import grow
    b = _gbatch()
    for tie in (True, False):
        m = _gpt(tie=tie)
        base = _logits(m, b)
        before = {k: v.copy() for k, v in m.p.items()}
        cases = [dict(layers=1), dict(layers=2, at=0), dict(layers=1, at=1, init="random"), dict(ffn=16), dict(layers=1, ffn=8, at=1)]
        for kw in cases:
            g = grow(m, **kw)
            assert np.max(np.abs(_logits(g, b) - base)) < 1e-10, (tie, kw)               # float64: identical to rounding
            assert abs(g.loss_and_grads(b)[0] - m.loss_and_grads(b)[0]) < 1e-10
            assert g.num_params() > m.num_params()
        assert all(np.array_equal(before[k], m.p[k]) for k in before)                       # the original model was not touched
    g = grow(_gpt(), layers=2, ffn=8)
    assert (g.cfg.n_layer, g.cfg.ffn, g.cfg.d_model, g.cfg.n_head) == (4, 32, 16, 2)


def test_growing_attention_heads_preserves_the_function_exactly():
    from zerowhite.chat.grow import grow
    b = _gbatch()
    for tie in (True, False):
        m = _gpt(tie=tie)
        base = _logits(m, b)
        g = grow(m, heads=2)                                                                # d_model 16 -> 32, 2 -> 4 heads, head_dim unchanged
        assert (g.cfg.d_model, g.cfg.n_head, g.cfg.head_dim) == (32, 4, m.cfg.head_dim)
        err = np.max(np.abs(_logits(g, b) - base)) / np.max(np.abs(base))
        assert err < 1e-10, (tie, err)                                                      # the gain AND epsilon corrections make it exact
        assert abs(g.loss_and_grads(b)[0] - m.loss_and_grads(b)[0]) < 1e-10
        assert abs(g.cfg.eps - m.cfg.eps * 16 / 32) < 1e-15
        both = grow(m, heads=1, ffn=8, layers=1, at=1)
        assert np.max(np.abs(_logits(both, b) - base)) / np.max(np.abs(base)) < 1e-10
        twice = grow(grow(m, heads=1), heads=1)                                             # growth composes
        assert np.max(np.abs(_logits(twice, b) - base)) / np.max(np.abs(base)) < 1e-10
    # the gain correction is the reason it works: without it the logits move a lot
    from zerowhite.chat import grow as G
    plain = G._grow_heads(_gpt().host_params(), _gpt().cfg, 2, np.random.default_rng(0))
    for name in plain[0]:
        if name == "nf" or name.endswith((".n1", ".n2")):
            plain[0][name][: _gpt().cfg.d_model] /= np.sqrt(_gpt().cfg.d_model / plain[1].d_model)
    from zerowhite.chat.model import GPT
    broken = GPT(plain[1], params=plain[0])
    assert np.max(np.abs(_logits(broken, b) - base)) / np.max(np.abs(base)) > 0.05


def test_grown_capacity_starts_idle_and_then_learns():
    from zerowhite.chat.grow import grow
    from zerowhite.chat.optim import AdamW
    m = _gpt(dtype="float32")
    g = grow(m, layers=1, ffn=8, heads=1)
    b = _gbatch()
    loss0, grads, _ = g.loss_and_grads(b)
    assert abs(loss0 - m.loss_and_grads(b)[0]) < 1e-3
    new = f"b{g.cfg.n_layer - 1}."
    assert np.abs(grads[new + "wo"]).max() > 0 and np.abs(grads[new + "wd"]).max() > 0     # the zero output projections get a gradient at once
    assert np.abs(grads[new + "wq"]).max() == 0                                            # ... the rest wakes up once those have moved
    opt = AdamW(g.p, 0.9, 0.99, weight_decay=0.0)
    for _ in range(30):
        loss, grads, info = g.loss_and_grads(b)
        opt.step(g.p, {k: v / max(1, info["n"]) for k, v in grads.items()}, 0.01, 1.0)
    assert np.abs(g.p[new + "wo"]).max() > 0 and np.abs(g.p[new + "wq"]).max() > 0
    assert g.loss_and_grads(b)[0] < 0.6 * loss0                                             # it is a model that trains, not a dead copy


def test_growing_rejects_nonsense_and_a_workspace_grows_into_a_new_version_the_trainer_continues_from():
    from zerowhite.chat import grow as G
    import test_chat as TC
    from zerowhite.chat import prepare as PR, train as TR
    from zerowhite.chat.model import GPT, read_meta
    m = _gpt()
    for bad in (dict(layers=-1), dict(layers=1, at=9), dict(layers=1, init="zeros")):
        try:
            G.grow(m, **bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert G.grow(m).num_params() == m.num_params()                                        # nothing asked: an exact copy
    cfg, lay = TC.copy_workspace()
    old_file = lay.current()["file"]
    old_cfg = read_meta(lay.current_weights())["config"]
    out = G.grow_current(lay, layers=1, ffn=16, heads=1)
    cur = lay.current()
    assert cur["file"].startswith("grow-") and cur["grown_from"] == old_file and os.path.exists(os.path.join(lay.weights_dir, old_file))
    new_cfg = read_meta(lay.current_weights())["config"]
    assert new_cfg["n_layer"] == old_cfg["n_layer"] + 1 and new_cfg["ffn"] == old_cfg["ffn"] + 16 and cur["params"] > cur["params_before"]
    assert new_cfg["n_head"] == old_cfg["n_head"] + 1 and new_cfg["d_model"] == old_cfg["d_model"] + old_cfg["d_model"] // old_cfg["n_head"]
    assert "->" in out["summary"]
    P = PR.prepare(cfg, lay, do_bench=False)                                               # the plan honours the grown shape ...
    assert P.model_cfg is not None and P.model_cfg.n_layer == new_cfg["n_layer"] and P.plan.info.get("notice", "").find("fresh") < 0
    res = TR.Trainer(P).run()                                                              # ... and training continues from the grown weights
    model, meta = GPT.load(lay.current_weights())
    assert model.cfg.n_layer == new_cfg["n_layer"] and model.cfg.ffn == new_cfg["ffn"] and model.cfg.d_model == new_cfg["d_model"] and res["steps"] > 0
    assert abs(model.cfg.eps - new_cfg["eps"]) < 1e-15                                      # the corrected epsilon survives training and saving
    from zerowhite.chat.util import FileLock
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])      # another live process holds the training lock
    try:
        fileio.write_text(lay.train_lock, f"{sleeper.pid} {time.time():.0f}\n")
        try:
            G.grow_current(lay, layers=1)
            raise AssertionError("must refuse while training holds the lock")
        except RuntimeError as ex:
            assert "training run is active" in str(ex)
    finally:
        sleeper.kill()
        sleeper.wait()
        if os.path.exists(lay.train_lock):
            os.remove(lay.train_lock)


# ================================================================================================ high-dimensional program inputs
class HiDimBandit:
    """obs = 3*x*v + noise in 40 dimensions; the signal lives ONLY in columns 26..39, the first 24 are pure noise. Reward 1 for action == (x > 0)."""
    obs_dim, n_actions, max_steps, max_return, name = 40, 2, 20, 20.0, "hidim_bandit"

    def __init__(self):
        r = np.random.default_rng(5)
        self.v = np.zeros(40)
        self.v[26:] = r.normal(size=14)
        self.v /= np.linalg.norm(self.v)

    def reset(self, rng):
        self.rng, self.t = rng, 0
        return self._next()

    def _next(self):
        self.x = self.rng.normal()
        return self.x * self.v * 3 + 0.3 * self.rng.normal(size=40)

    def step(self, a):
        r = 1.0 if int(a) == int(self.x > 0) else 0.0
        self.t += 1
        return self._next(), r, self.t >= self.max_steps, False


def test_the_projector_keeps_the_variance_of_all_columns_and_survives_a_round_trip():
    from zerowhite.forge.reduce import Projector, needed
    rng = np.random.default_rng(0)
    z = rng.normal(size=(300, 3))
    X = z @ rng.normal(size=(3, 40)) + 0.01 * rng.normal(size=(300, 40))              # 40 columns, really 3 directions
    P = Projector.fit(X, 24)
    assert P.k <= 24 and P(X[0]).shape == (P.k,)
    out = np.array([P(x) for x in X])
    assert np.all(np.abs(out[:, :3].std(axis=0) - 1.0) < 0.05)                         # whitened: the top components have unit variance
    assert np.abs(np.corrcoef(out[:, 0], out[:, 1])[0, 1]) < 0.05                      # ... and are uncorrelated
    P2 = Projector.from_dict(json.loads(json.dumps(P.to_dict())))
    assert np.allclose(P2(X[7]), P(X[7]))
    const = np.hstack([X, np.ones((300, 1))])                                          # a constant column must not produce NaN or blow up
    assert np.all(np.isfinite(Projector.fit(const, 24)(const[3])))
    assert not needed(24) and needed(25)
    try:
        Projector.fit(X[:3], 24)
        raise AssertionError("too few observations must be refused")
    except ValueError:
        pass


def test_program_agents_solve_a_task_whose_signal_is_beyond_the_first_24_observation_columns():
    from zerowhite.forge import families as F, task as T
    genome = dict(n_islands=3, island_size=8, tune_steps=8, max_nodes=80, n_mut=2)
    res = {}
    try:
        for flag in (True, False):
            F.ProgramPolicyAgent.project_high_dim = flag
            for seed in range(4):
                task = T.EnvTask(HiDimBandit())
                a = F.ProgramPolicyAgent(task, genome, seed)
                a.train(task, 2.0, np.random.default_rng(seed))
                res[flag, seed] = a.val_score(task)[0]
                assert (a.proj is not None) == flag
    finally:
        F.ProgramPolicyAgent.project_high_dim = True
    proj, trunc = [res[True, s] for s in range(4)], [res[False, s] for s in range(4)]
    assert max(trunc) < 0.3, trunc                                                      # truncation sees only noise: chance level, every seed
    assert sum(s > 0.6 for s in proj) >= 2 and np.mean(proj) > np.mean(trunc) + 0.3, (proj, trunc)   # the projected agent finds the signal (not every seed: the search is stochastic)


def test_a_reloaded_program_agent_sees_the_same_projection_and_narrow_tasks_are_untouched():
    from zerowhite.forge import families as F, task as T
    genome = dict(n_islands=2, island_size=6, tune_steps=4, max_nodes=60, n_mut=2)
    task = T.EnvTask(HiDimBandit())
    a = F.ProgramPolicyAgent(task, genome, 0)
    a.train(task, 1.0, np.random.default_rng(0))
    d = tmp()
    a.save(d)
    assert "projector" in json.loads(fileio.read_text(os.path.join(d, "state.json"))) and "PCA projection" in fileio.read_text(os.path.join(d, "program.py"))
    b = F.ProgramPolicyFamily().load(d, task, genome)                                   # built with a DIFFERENT seed, so a refit would differ
    assert np.allclose(b.proj.comps, a.proj.comps)
    xs = [HiDimBandit().reset(np.random.default_rng(i)) for i in range(20)]
    assert [a._act_fn()(x) for x in xs] == [b._act_fn()(x) for x in xs]
    narrow = F.ProgramPolicyAgent(T.from_env_name("cartpole"), genome, 0)               # 4 observations: no projector, the same 4 columns as before
    assert narrow.proj is None and narrow.vocab.n_cols == 4


def test_a_patch_that_passes_in_the_sandbox_but_fails_on_the_real_tree_is_rolled_back_by_the_journal():
    from zerowhite.selfmod import patcher
    from zerowhite.selfmod.history import HistoryStore
    import re
    repo = os.path.join(tmp(), "repo")
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns("__pycache__", ".git", "runs", "*.egg-info", ".zw_promotion"))
    ev = os.path.join(repo, "zerowhite", "evolve.py")
    before = fileio.read_text(ev)
    store = HistoryStore(tmp())

    def fake_bench(tree_root, seeds, timeout=0):
        txt = fileio.read_text(os.path.join(tree_root, "zerowhite", "evolve.py"))
        s = float(re.search(r"sigma=([0-9.e-]+)[,)]", txt.split("class Evolver")[1].split("def step")[0]).group(1))
        return 1.0 - abs(s - 0.9) + 0.001 * (seeds[0] % 3)
    real_b, real_t = patcher.bench_in, patcher.tests_pass
    patcher.bench_in = fake_bench
    patcher.tests_pass = lambda root, timeout=0: ((False, "real tree disagrees") if os.path.abspath(root) == os.path.abspath(repo) else (True, "sandbox ok"))
    try:
        rep = patcher.search(repo, store, evals=60, seed=1, apply=True, verbose=False, files=("evolve.py",), margin=0.0)
    finally:
        patcher.bench_in, patcher.tests_pass = real_b, real_t
    assert not rep.applied and not rep.accepted and "real tree disagrees" in rep.note and "originals restored" in rep.note
    assert fileio.read_text(ev) == before                                                 # byte-for-byte the original
    assert len(store.lineage("src:evolve.py")) >= 3                                       # original, attempted patch, auto-rollback: every attempt is on record
    states = [json.loads(open(os.path.join(repo, ".zw_promotion", n, "journal.json")).read())["state"] for n in os.listdir(os.path.join(repo, ".zw_promotion"))]
    assert states == ["rolled_back"]


def test_recording_the_same_text_twice_no_longer_makes_lineage_loop_forever():
    """6.2 bug: original -> patch -> original-again (the patcher's auto-rollback) made lineage() cycle; it is run in a subprocess so a regression
    fails in 30 s instead of hanging the whole suite."""
    code = (
        "import sys, tempfile; sys.path.insert(0, %r)\n"
        "from zerowhite.selfmod.history import HistoryStore\n"
        "s = HistoryStore(tempfile.mkdtemp()); n = 'src:x.py'\n"
        "s.put(n, 'A', dict(kind='original')); s.put(n, 'B', dict(kind='self-patch')); s.put(n, 'A', dict(kind='auto-rollback'))\n"
        "ch = s.lineage(n); print([(e['version'], e['meta']['kind']) for e in ch])\n"
        "s.rollback(n, 1); print([e['version'] for e in s.lineage(n)], s.text(n))\n"
        "s.put(n, 'C', {}); print([e['version'] for e in s.lineage(n)], s.text(n))\n"
        "s.put(n, 'C', {}); print([e['version'] for e in s.lineage(n)])\n" % ROOT)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert lines[0] == "[(1, 'original'), (2, 'self-patch'), (3, 'auto-rollback')]"
    assert lines[1] == "[1, 2] B"                                                  # rolling back one step from 'A' (v3) lands on the patch, as before
    assert lines[2] == "[1, 2, 3] C"
    assert lines[3] == "[1, 2, 3, 4]"                                              # the same text twice in a row is still two versions
