"""Tests for the GPU backend (6.2). There is no GPU in the test environment, so the device path runs on tests/fakedevice.py: device
memory that refuses host arrays, offers only the CuPy functions this project relies on, and counts every transfer and synchronisation.
What these tests prove: the CPU path is untouched, the device path is the same maths with explicit transfers and few syncs, checkpoints
are ordinary host files, and a failure is reported with the fix. What they cannot prove: that a real card is fast (`zerowhite chat gpu`)."""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import fakedevice as F                                           # noqa: E402
import test_v6 as T6                                             # noqa: E402  (helpers only)
import test_v61 as T61                                           # noqa: E402  (helpers only)
from zerowhite import chat_task as CT                            # noqa: E402
from zerowhite.chat import autoscale as AS                       # noqa: E402
from zerowhite.chat import backend as bk                         # noqa: E402
from zerowhite.chat import gpucheck                              # noqa: E402
from zerowhite.chat.config import Config                        # noqa: E402
from zerowhite.chat.model import GPT, GPTConfig                  # noqa: E402
from zerowhite.chat.optim import AdamW                           # noqa: E402
from zerowhite.chat.paths import Layout                          # noqa: E402
from zerowhite.chat.prepare import prepare                       # noqa: E402
from zerowhite.chat.train import GpuOutOfMemory, Trainer         # noqa: E402

TINY = T61.TINY


def _batch(seed=1, B=3, T=40, V=97):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, V // 4, (B, T)).astype(np.int64)                   # few distinct ids: repeats exercise the embedding scatter
    seg = np.ones((B, T), np.int64)
    seg[:, T // 2:] = 2
    pos = np.concatenate([np.arange(T // 2), np.arange(T - T // 2)])[None].repeat(B, 0).astype(np.int64)
    w = (rng.random((B, T)) > 0.3).astype(np.float32)
    w[1, 5] = -0.5
    return dict(ids=ids, tgt=np.roll(ids, -1, 1), w=w, pos=pos, seg=seg)


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-12))


# ------------------------------------------------------------------------------------------------- the fake must be strict
def test_the_fake_device_refuses_host_arrays_and_apis_cupy_lacks():
    be = F.FakeGPU()
    d = be.asarray(np.ones((2, 3), np.float32))
    try:
        d + np.ones((2, 3), np.float32)
        raise AssertionError("a host array was silently mixed with device memory")
    except F.HostLeak:
        pass
    try:
        d[np.array([0, 1])]
        raise AssertionError("a host index array was accepted")
    except F.HostLeak:
        pass
    for bad in (lambda: be.xp.add, lambda: be.xp.einsum, lambda: np.add.reduceat(d, np.array([0]), axis=0)):
        try:
            bad()
            raise AssertionError("an API outside the CuPy surface was allowed")
        except F.NotOnDevice:
            pass
    F.STATS.reset()
    float(d.sum())
    assert F.STATS.scalar_reads == 1                                         # reading a device scalar is counted as a synchronisation
    assert (d + d)[0, 0] == 2.0 and isinstance(be.to_host(d), np.ndarray) and not isinstance(be.to_host(d), F.DevArray)


# ------------------------------------------------------------------------------------------------- the model and the optimiser
def test_device_model_matches_cpu_with_no_host_leaks_and_a_handful_of_syncs():
    be = F.FakeGPU()
    for tie in (True, False):
        cfg = GPTConfig(vocab=97, d_model=32, n_layer=2, n_head=4, ffn=64, ctx=64, tie=tie)
        cpu, dev = GPT(cfg, seed=3), GPT(cfg, seed=3, backend=be)
        assert all(isinstance(v, F.DevArray) for v in dev.p.values())
        assert all(np.array_equal(cpu.p[k], be.to_host(dev.p[k])) for k in cpu.p)           # initialised on the host: the same weights everywhere
        b = _batch()
        lc, gc, ic = cpu.loss_and_grads(b, 0.2, np.random.default_rng(9))                   # dropout on: the host RNG makes the masks identical
        F.STATS.reset()
        ld, gd, idev = dev.loss_and_grads(b, 0.2, np.random.default_rng(9))
        st = F.STATS.snapshot()
        assert abs(ld - lc) / abs(lc) < 1e-6 and ic["n"] == idev["n"]
        assert all(isinstance(g, F.DevArray) for g in gd.values())                          # gradients stay on the device
        assert max(_rel(gc[k], be.to_host(gd[k])) for k in gc) < 1e-5
        assert st["d2h_calls"] == 0 and st["scalar_reads"] <= 4                             # nothing but the two loss scalars comes back
        assert abs(cpu.eval_nll(b)[0] - dev.eval_nll(b)[0]) < 1e-3 * abs(cpu.eval_nll(b)[0])
        lp = dev.token_logprobs(b)
        assert isinstance(lp, np.ndarray) and not isinstance(lp, F.DevArray) and np.allclose(lp, cpu.token_logprobs(b), atol=1e-5)


def test_device_optimizer_matches_cpu_with_one_transfer_per_step_and_host_state():
    be = F.FakeGPU()
    cfg = GPTConfig(vocab=97, d_model=32, n_layer=3, n_head=4, ffn=64, ctx=64)
    cpu, dev = GPT(cfg, seed=3), GPT(cfg, seed=3, backend=be)
    _, gc, _ = cpu.loss_and_grads(_batch())
    _, gd, _ = dev.loss_and_grads(_batch())
    oc, od = AdamW(cpu.p), AdamW(dev.p, backend=be)
    F.STATS.reset()
    for _ in range(4):
        rc, rd = oc.step(cpu.p, gc, 1e-3, 1.0), od.step(dev.p, gd, 1e-3, 1.0)
        assert abs(rc["grad_norm"] - rd["grad_norm"]) < 1e-4 * rc["grad_norm"] and rd["applied"] == 1.0
    assert F.STATS.d2h_calls == 4, f"{F.STATS.d2h_calls} transfers for 4 steps over {len(dev.p)} tensors: the norm must be ONE transfer per step"
    assert F.STATS.scalar_reads == 0, "a float() per tensor is one synchronisation each; the norm must be stacked on the device first"
    assert max(float(np.max(np.abs(cpu.p[k] - be.to_host(dev.p[k])))) for k in cpu.p) < 1e-6
    st = od.state_arrays()
    assert all(isinstance(v, np.ndarray) and not isinstance(v, F.DevArray) for v in st.values())     # what gets written to disk is host data
    fresh = AdamW(dev.p, backend=be)
    assert fresh.load_state(st) and fresh.t == od.t and np.array_equal(be.to_host(fresh.m["tok_emb"]), be.to_host(od.m["tok_emb"]))
    gd["tok_emb"][...] = be.asarray(np.full(gd["tok_emb"].shape, np.nan, np.float32))               # a poisoned batch must still skip the step
    before = {k: be.to_host(v).copy() for k, v in dev.p.items()}
    assert od.step(dev.p, gd, 1e-3, 1.0)["applied"] == 0.0 and all(np.array_equal(before[k], be.to_host(dev.p[k])) for k in before)


# ------------------------------------------------------------------------------------------------- whole training runs
def _cfg(chat, **extra):
    return Config.load(chat, overrides=TINY).replace(**{"train.budget_minutes": None}, **extra)      # no time budget: the schedule is by steps only


def _prepared(root, backend, **extra):
    chat = T61._chat_ws(root)
    cfg = _cfg(chat, **extra)
    return prepare(cfg, Layout(cfg).ensure(), do_bench=False, backend=backend), chat


def test_a_training_run_on_the_device_matches_the_cpu_and_leaves_ordinary_files():
    be = F.FakeGPU()
    from zerowhite import fileio
    Pc, chat_c = _prepared(T6.tmp(), bk.CPU)
    Pd, chat_d = _prepared(T6.tmp(), be)
    rc = Trainer(Pc).run()
    F.STATS.reset()
    written = []
    real_save = fileio.save_npz

    def spy(path, arrays, *a, **k):                                           # everything the run writes passes through here
        written.append({k2: type(v) for k2, v in arrays.items()})
        return real_save(path, arrays, *a, **k)
    fileio.save_npz = spy
    try:
        rd = Trainer(Pd, backend=be).run()
    finally:
        fileio.save_npz = real_save
    assert written and all(t is not F.DevArray for w in written for t in w.values()), "a device array was handed to a file writer"
    assert F.STATS.h2d_calls > 100 and F.STATS.scalar_reads > 10, "the run never really used the device (a CPU model passes every other check)"
    assert rc["promoted"] and rd["promoted"] and rc["steps"] == rd["steps"]
    assert abs(rc["val_nll"] - rd["val_nll"]) < 2e-3 * rc["val_nll"], (rc["val_nll"], rd["val_nll"])       # same maths, same schedule
    assert F.STATS.releases >= 1                                                                        # the card's cache was handed back
    live = Layout(Pd.cfg).current_weights()
    served, _ = GPT.load(live)                                                                           # what the CPU server and exporter read
    assert all(type(v) is np.ndarray and v.dtype == np.float32 for v in served.p.values())              # plain float32 host arrays
    plan = json.load(open(Layout(Pd.cfg).resolved_config_path, encoding="utf-8"))
    assert "Fake GPU" in plan["compute"]["name"] and "Fake GPU" in plan["info"]["backend"]


def test_a_run_stopped_on_the_device_resumes_to_the_same_result():
    be = F.FakeGPU()
    Pa, _ = _prepared(T6.tmp(), be)
    full = Trainer(Pa, backend=be).run()
    Pb, chat_b = _prepared(T6.tmp(), be)
    calls = {"n": 0}

    def stop_after_40():
        calls["n"] += 1
        return calls["n"] > 40
    part = Trainer(Pb, should_stop=stop_after_40, backend=be).run()
    assert part["reason"] == "stopped" and not part["promoted"]
    cfg_b = _cfg(chat_b)
    again = Trainer(prepare(cfg_b, Layout(cfg_b).ensure(), do_bench=False, backend=be), backend=be).run()   # the checkpoint (host arrays) goes back to the device
    assert again["promoted"] and again["steps"] == full["steps"]
    assert abs(again["val_nll"] - full["val_nll"]) < 5e-3 * full["val_nll"], (again["val_nll"], full["val_nll"])


def test_the_organism_trains_the_chat_model_on_a_device_and_says_so():
    be = F.FakeGPU()
    chat = T61._chat_ws(T6.tmp())
    ct = CT.ChatTask(chat, TINY, backend=be)
    r = ct.attempt(0.4)
    assert r["ok"] and r["promoted"] and r["skill"] > 0.1 and "Fake GPU" in r["compute"]
    ws = T6.tmp()
    T61._chat_ws(ws)                                                          # <ws>/chat: adopted by --chat auto
    org = T6.O.Organism(ws, T6.O.Settings(max_cycles=2, quiet=True, chat="auto", chat_set=TINY, chat_first_minutes=0.4, doctor_every=0, poll_seconds=0.05))
    org._chat = CT.ChatTask(os.path.join(ws, "chat"), TINY, backend=be)       # the organism's own chat task, pointed at the fake card
    org.run()
    assert org.state["tasks"]["chat"]["attempts"] >= 1
    assert "trained on gpu:0 Fake GPU" in T6.O.status(ws)                      # `organism status` tells which device the language model was trained on


# ------------------------------------------------------------------------------------------------- planning
def test_autoscale_plans_gpu_memory_from_vram_and_lets_explicit_settings_win():
    cfg = Config.load(T6.tmp())
    assert AS._default_unless_set(cfg, "autoscale.max_params", "compute.max_params", False) == 60_000_000          # CPU: the CPU-sized cap
    assert AS._default_unless_set(cfg, "autoscale.max_params", "compute.max_params", True) == 200_000_000          # GPU: its own cap
    pinned = cfg.replace(**{"autoscale.max_params": 5_000_000})
    assert AS._default_unless_set(pinned, "autoscale.max_params", "compute.max_params", True) == 5_000_000         # something the user set always wins
    big, _ = _prepared(T6.tmp(), F.FakeGPU(free_gb=24, total_gb=24, label="Big"))
    small, _ = _prepared(T6.tmp(), F.FakeGPU(free_gb=0.001, total_gb=0.001, label="Tiny"), **{"compute.vram_fraction": 0.5})
    assert "warning" not in big.plan.info and "Big" in big.plan.info["backend"]
    assert "VRAM" in small.plan.info["warning"] and "compute.vram_fraction" in small.plan.info["warning"]          # says which memory, and which knob
    cpu, _ = _prepared(T6.tmp(), bk.CPU)
    assert cpu.plan.info["backend"] == "cpu" and "RAM" not in cpu.plan.info.get("warning", "")


def test_the_benchmark_synchronises_the_device_around_its_timer():
    be = F.FakeGPU()
    F.STATS.reset()
    tps = AS.bench(GPTConfig(vocab=97, d_model=32, n_layer=2, n_head=4, ffn=64, ctx=64), 2, 32, 2, be)
    assert tps > 0 and F.STATS.syncs >= 2 and F.STATS.releases >= 1          # before the clock starts and before it stops: kernels are asynchronous
    c = GPTConfig(vocab=97, d_model=32, n_layer=2, n_head=4, ffn=64, ctx=64)
    want = be.matmul_gflops() * 1e9 * AS._GPU_EFFICIENCY / AS.flops_per_token(c, 64)         # measured matmul speed x the (stated, unmeasured) GPU efficiency
    assert abs(AS.estimate_tps(c, 64, be) - want) < 1e-6 * want


def test_gpu_out_of_memory_stops_with_the_fix_and_keeps_the_checkpoint_path_open():
    be = F.FakeGPU(free_gb=1.5)
    P, _ = _prepared(T6.tmp(), be)
    real = GPT.loss_and_grads
    calls = {"n": 0}

    def boom(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 5:
            raise F.FakeOOM("out of memory while trying to allocate 4.0 GiB")
        return real(self, *a, **k)
    GPT.loss_and_grads = boom
    try:
        try:
            Trainer(P, backend=be).run()
            raise AssertionError("the out-of-memory error was swallowed")
        except GpuOutOfMemory as e:
            msg = str(e)
    finally:
        GPT.loss_and_grads = real
    assert "train.batch_tokens" in msg and "compute.vram_fraction" in msg and "Fake GPU" in msg
    from zerowhite.chat.util import FileLock
    lock = FileLock(P.lay.train_lock)
    lock.acquire(wait_s=0)                                                                              # the failed run released it: the next one can start
    lock.release()


# ------------------------------------------------------------------------------------------------- selection and the self-check
def test_backend_selection_falls_back_quietly_demands_loudly_and_never_guesses():
    assert bk.get("cpu") is bk.CPU
    try:
        bk.get("tpu")
        raise AssertionError("an unknown backend was accepted")
    except bk.BackendError as e:
        assert "cpu, gpu or auto" in str(e)
    cfg = Config.load(T6.tmp())
    assert cfg.get("compute.backend") == "auto"                                # a GPU is used when there is one; nothing to configure
    assert bk.from_config(cfg.replace(**{"compute.backend": "cpu"})) is bk.CPU
    if importlib.util.find_spec("cupy") is None:
        auto = bk.get("auto")
        assert not auto.is_gpu and "CuPy is not importable" in auto.note and "pip install cupy" in auto.note      # quiet fallback, with the reason
        try:
            bk.get("gpu")
            raise AssertionError("a demanded GPU silently became the CPU")
        except bk.BackendError as e:
            assert "pip install cupy-cuda12x" in str(e) and "nvidia-smi" in str(e)                                # a demand is never downgraded
    else:
        assert bk.get("auto").is_gpu or bk.get("auto").note                    # a real GPU machine: either it works, or we are told why not


def test_the_gpu_self_check_passes_a_good_device_and_catches_a_wrong_one():
    good = gpucheck.run(F.FakeGPU(), speed=False)
    assert good["ok"] and [c["name"] for c in good["checks"]] == ["kernels", "model", "optimiser"] and all(c["ok"] for c in good["checks"])
    assert "The GPU is ready" in gpucheck.format_report(good)

    class WrongScatter(F.FakeGPU):                                             # a device whose embedding-gradient scatter drops repeated ids
        def scatter_rows(self, dE, ids_flat, rows):
            dE.view(np.ndarray)[ids_flat.view(np.ndarray)] += rows.view(np.ndarray)           # fancy += loses duplicates: a classic GPU-port bug
    bad = gpucheck.run(WrongScatter(), speed=False)
    assert not bad["ok"] and any(c["name"] == "model" and not c["ok"] for c in bad["checks"])
    assert "do not train on this setup" in gpucheck.format_report(bad)
    none = gpucheck.run(bk.Backend(note="CuPy is not importable"), speed=False)
    assert not none["ok"] and "pip install cupy" in gpucheck.format_report(none)
    fast = gpucheck.run(F.FakeGPU(), speed=True, speed_rows=4, speed_ctx=64)
    assert fast["ok"] is True or fast["checks"][-1]["name"] == "speed"          # the speed check runs and is reported either way
    assert "tokens_per_s_gpu" in fast and "tokens_per_s_cpu" in fast
