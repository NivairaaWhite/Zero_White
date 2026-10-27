"""cli_v6.py - the commands of the autonomy layer: devour, organism, substrate."""
from __future__ import annotations

import argparse
import json
import os
import sys


def cmd_substrate(a):
    from . import substrate
    p = substrate.cached(a.workspace, force=a.refresh) if a.workspace else substrate.probe()
    pl = substrate.plan(p)
    if a.json:
        print(json.dumps(dict(probe=p, plan=pl), indent=2, default=str))
    else:
        print(substrate.describe(p, pl))
        if a.workspace:
            print(f"(stored in {os.path.join(a.workspace, substrate.CACHE_NAME)}; a workspace copied to another machine is re-measured automatically)")
    return 0


def cmd_engine_tier(a):
    """6.4: OPT-IN second tier of self-modification: single-edit mutations of the numerical core (nn.py, engine.py) under the full judge. Dry run unless --apply."""
    from .cli_v4 import REPO_ROOT, _store
    from .selfmod import engine_tier
    print(f"engine tier: {', '.join(engine_tier.ENGINE)} may be edited; the judge (tests, significance gate, journal, sealed constitution) is unchanged"
          + ("" if a.apply else "  [DRY RUN: nothing is written to the repository; add --apply to promote an accepted change]"))
    res = engine_tier.search(REPO_ROOT, _store(a), evals=a.evals, seed=a.seed, apply=a.apply)
    print(json.dumps({k: v for k, v in res.items() if k != "gate"}, indent=1, default=str))
    if res.get("gate"):
        g = res["gate"]
        print(f"final seeds: gain {g.get('mean_gain', 0):+.4f}, p = {g.get('p', 1):.3f}, {g.get('positive')}/{g.get('n')} improved")
    return 0


def cmd_share(a):
    """6.4: write what the forge learned (the zoo) and the system's own source into the chat model's data folder as plain text (chat/shared.py)."""
    from .chat import shared
    data = a.data or os.path.join(a.workspace, "chat", "data")
    out = {}
    if not a.no_zoo:
        out["zoo"] = shared.export_zoo(a.workspace, os.path.join(data, "_zoo"))
    if not a.no_self:
        out["self"] = shared.ingest_self_source(os.path.join(data, "_self"))
    print(json.dumps(out), "->", data)
    print("next: `zerowhite chat prepare` / `chat train` read these files like any others")
    return 0


def cmd_codec(a):
    """6.4: the learned audio codec (voice/codec.py): train on synthetic speech or your own wav files, encode a wav to tokens, decode tokens to a wav."""
    import numpy as np
    from .voice import audio, codec as C
    if a.action == "train":
        waves = []
        if a.wav_dir:
            for f in sorted(os.listdir(a.wav_dir)):
                if f.lower().endswith(".wav"):
                    waves.append(audio.load_audio(os.path.join(a.wav_dir, f), 16000)[0])
        if not waves:
            from .voice.formant import synthesize
            texts = ["the quick brown fox jumps over the lazy dog", "hello world how are you today", "she sells sea shells by the sea shore", "please call me back tomorrow morning",
                     "a stitch in time saves nine", "we hold these truths to be self evident", "open the door and turn on the light", "numbers one two three four five six seven"]
            waves = [synthesize(t, f0_hz=float(f0), rate=float(r)) for t in texts for f0, r in [(105, 1.0), (125, 0.9), (150, 1.1), (180, 1.0)]]
            print("no wav files given: training on SYNTHETIC speech from the built-in formant synthesiser (it has never seen a real voice)")
        print(f"{sum(len(w) for w in waves) / 16000:.0f} s of audio")
        cd = C.train(waves, stages=a.stages, codewords=a.codewords, hidden=a.hidden, latent=a.latent, epochs=a.epochs, verbose=True)
        cd.save(a.model)
        pca = C.pca_baseline(waves, a.stages, a.codewords, a.latent)
        probe = waves[:: max(1, len(waves) // 4)]
        for st in sorted({1, max(1, a.stages // 2), a.stages}):
            print(f"  {st} stage(s) = {cd.bitrate(st) / 1000:.2f} kbps: SNR {np.mean([C.snr_db(w, cd.roundtrip(w, st)) for w in probe]):.1f} dB "
                  f"(PCA at the same bitrate {np.mean([C.snr_db(w, pca.roundtrip(w, st)) for w in probe]):.1f} dB)   [on training clips: an optimistic figure]")
        print("saved", a.model)
        return 0
    cd = C.Codec.load(a.model)
    if a.action == "encode":
        w, _ = audio.load_audio(a.input, 16000)
        toks = cd.encode(w, a.use_stages)
        out = C.tokens_to_text(toks)
        with open(a.out or a.input + ".tok", "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print(f"{len(toks)} frames x {toks.shape[1]} stages = {cd.bitrate(toks.shape[1]) / 1000:.2f} kbps -> {a.out or a.input + '.tok'}")
        return 0
    with open(a.input, "r", encoding="utf-8") as f:
        toks = C.text_to_tokens(f.read())
    y = cd.decode(toks)
    audio.save_audio(a.out or a.input + ".wav", y, 16000)
    print(f"{len(y) / 16000:.2f} s -> {a.out or a.input + '.wav'}")
    return 0


def cmd_kernels(a):
    """6.4: check the lazy-graph kernel backends on THIS machine: the C JIT (compiler found? same numbers as numpy? how much faster?) and CUDA (driver, NVRTC, a device?)."""
    import time
    import numpy as np
    from .kernels import cjit, cuda, graph as G
    r = np.random.default_rng(0)
    n = int(a.size)
    x, y, z = r.normal(size=(n // 1000, 1000)), r.normal(size=(n // 1000, 1000)), r.normal(size=(1000,))

    graphs = {"cheap ops (mul add mul relu sub abs)": lambda: (((G.lazy(x) * G.lazy(y) + G.lazy(z)) * 0.5).relu() - 0.25).abs(),
              "with transcendentals (tanh sqrt exp)": lambda: (((G.lazy(x) * G.lazy(y) + G.lazy(z).tanh()) * 0.5).relu() - 0.25).abs().sqrt().exp()}
    build = list(graphs.values())[-1]
    rc = 0
    print(f"graph over {n:,} elements, best of 3")

    def best(fn):
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            out = fn()
            ts.append(time.perf_counter() - t)
        return min(ts) * 1000, out
    for name, mk in graphs.items():
        t_np, ref = best(lambda: G.evaluate(mk(), "numpy"))
        if cjit.available():
            G.evaluate(mk(), "c")                                       # warm: compile once, cached by source hash
            st = {}
            t_c, got = best(lambda: G.evaluate(mk(), "c", stats=st))
            err = float(np.max(np.abs(got - ref)))
            print(f"  {name:40s} numpy {t_np:7.1f} ms | generated C {t_c:7.1f} ms ({t_np / t_c:.2f}x, {st['kernels']} fused kernel, max |diff| {err:.1e})")
            rc = rc or (0 if err < 1e-9 else 1)
        else:
            print(f"  {name:40s} numpy {t_np:7.1f} ms | no C compiler found (cc / gcc / clang): numpy is used")
    print("  (fusion saves memory passes, but numpy has SIMD exp/tanh and libm does not: expect a gain on cheap chains and possibly a loss with transcendentals)")
    ref = G.evaluate(build(), "numpy")
    info = cuda.probe()
    print(f"CUDA: driver {info['driver'] or 'not found'}, NVRTC {info['nvrtc'] or 'not found'}, devices {info['devices'] if info['devices'] is not None else 'unknown'}")
    if info["driver"] and info["nvrtc"] and info["devices"]:
        try:
            be = cuda.CudaBackend()
            print(f"  {be.describe()}  (architectures this NVRTC can target: {be.supported if be.supported else 'it cannot say'})")
            got = G.evaluate(build(), "cuda", cuda=be)
            err = float(np.max(np.abs(got - ref)))
            print(f"  generated CUDA via NVRTC + driver API: target {be.target}, max |difference| {err:.2e}")
            if err >= 1e-9:
                print("  MISMATCH with numpy: the generated kernel computed different numbers on this device")
                rc = 1
            for t, why in be.attempts:
                if why != "ok":
                    print(f"  (tried {t} first: {why[:160]})")
            if a.size >= 100000:
                mk = list(graphs.values())[0]
                ref0 = G.evaluate(mk(), "numpy")
                G.evaluate(mk(), "cuda", cuda=be)                      # warm: the compile happens once, the buffers move every time
                t_cu, got0 = best(lambda: G.evaluate(mk(), "cuda", cuda=be))
                t_np0, _ = best(lambda: G.evaluate(mk(), "numpy"))
                print(f"  cheap chain, whole call incl. host<->device copies: numpy {t_np0:.1f} ms | generated CUDA {t_cu:.1f} ms ({t_np0 / t_cu:.2f}x), max |diff| {float(np.max(np.abs(got0 - ref0))):.1e}")
        except Exception as ex:                                       # noqa: BLE001
            print(f"  generated CUDA FAILED: {type(ex).__name__}: {ex}")
            print("  this is reported, not fatal: numpy and the generated-C path are unaffected" + ("" if a.strict else " (use --strict to make it an error)"))
            rc = rc or (1 if a.strict else 0)
    else:
        print("  no usable NVIDIA driver + NVRTC here: the CUDA path was verified only against a CPU emulation of the generated source (docs/ARCHITECTURE.md)")
    return rc


def cmd_forage(a):
    """6.3: read named pages from hosts YOU allow into an inbox folder. Nothing runs without --allow-host (see forage/fetch.py)."""
    from .forage import Forager, Policy
    urls = list(a.urls or [])
    if a.urls_file:
        with open(a.urls_file, "r", encoding="utf-8") as f:
            urls += [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not a.allow_host:
        print("forage fetches nothing unless you name the hosts it may contact: --allow-host docs.python.org (repeatable). There are no defaults.", file=sys.stderr)
        return 2
    if not urls:
        print("forage needs at least one URL (or --urls-file)", file=sys.stderr)
        return 2
    pol = Policy(allow_hosts=tuple(a.allow_host), allow_http=a.allow_http, allow_local=a.allow_local, respect_robots=not a.no_robots,
                 max_bytes=int(a.max_mb * (1 << 20)), max_urls=a.max_urls, delay=a.delay)
    base = os.path.join(a.workspace, "forage")
    out_dir = a.out or os.path.join(base, "inbox")
    f = Forager(pol, out_dir, os.path.join(base, "ledger.jsonl"))
    rows = f.fetch_all(urls)
    for r in rows:
        print(f"[{'ok' if r['ok'] else 'no'}] {r['url']}  {r['reason']}" + (f"  -> {r['path']}" if r.get("path") and r["ok"] else ""))
    n_ok = sum(1 for r in rows if r["ok"])
    print(f"{n_ok}/{len(rows)} saved under {out_dir}; every attempt (refusals too) is in {os.path.join(base, 'ledger.jsonl')}")
    if n_ok:
        print(f"to let the organism read them: zerowhite organism --inbox {out_dir} ...")
    return 0 if n_ok or not rows else 1


def cmd_devour(a):
    from .omnivore import Library, Limits, default_limits, devour, load_tasks
    if a.list:
        lib = Library(a.workspace)
        names = lib.names()
        print(f"{len(names)} task(s) in {lib.tasks_dir}")
        for n in names:
            sp = lib.spec(n) or {}
            extra = f"{sp.get('rows', '')} rows, target '{sp.get('target')}'" if sp.get("kind") == "table" else (
                f"{sp.get('bytes', 0):,} bytes of text" if sp.get("kind") == "text" else sp.get("kind", ""))
            print(f"  {n:<28} {sp.get('task_kind', ''):<15} {extra}")
        q = lib.quarantined()
        print(f"{len(q)} quarantine record(s):")
        for r in q[-a.limit:]:
            print(f"  {r.get('label', '')[-60:]:<60} {r.get('reason', '')[:100]}")
        return 0
    if not a.paths:
        print("devour needs at least one file or folder (or --list to see what was digested earlier)", file=sys.stderr)
        return 2
    limits = default_limits(a.workspace)
    if a.max_mb:
        limits.max_file_bytes = int(a.max_mb * (1 << 20))
        limits.max_member_bytes = min(limits.max_member_bytes, limits.max_file_bytes)
    if a.max_rows:
        limits.max_rows = a.max_rows
    d = devour(a.paths, a.workspace, limits=limits, allow_code=a.allow_code, target=a.target, pool=a.pool, retry=a.retry, learned=a.learned_sniffer,
               log=None if (a.quiet or a.json) else (lambda m: None), seed=a.seed)
    if a.json:
        print(json.dumps(d.to_dict(), indent=2, default=str))
    else:
        print(d.summary(max_lines=a.limit))
    if a.solve:
        from .forge.generalist import Generalist
        tasks = load_tasks(a.workspace, names=d.tasks, allow_code=a.allow_code, limits=limits, quiet=False)
        if not tasks:
            print("\nnothing new to solve" + (" (every task was already digested; run `zerowhite solve` on the library tasks or add --retry)" if d.counts().get("unchanged") else ""))
            return 0
        gen = Generalist(a.workspace, seed=a.seed, verbose=True)
        for t in tasks:
            print(f"\n=== solving '{t.name}' ===")
            try:
                gen.solve(t, budget=a.budget, n_new=a.candidates, rungs=a.rungs)
            except Exception as ex:                          # noqa: BLE001 - one bad task must not stop the others
                print(f"[devour] '{t.name}' failed: {type(ex).__name__}: {ex}")
    return 0


def cmd_organism(a):
    from . import organism as O
    if a.action == "status":
        print(O.status(a.workspace))
        return 0
    if a.action == "stop":
        d = os.path.join(os.path.abspath(a.workspace), "organism")
        os.makedirs(d, exist_ok=True)
        from . import fileio
        fileio.write_text(os.path.join(d, "STOP"), "stop requested\n")
        print(f"stop requested: the organism in {a.workspace} will finish its current action and exit")
        return 0
    s = O.Settings(inbox=a.inbox or [], minutes=a.minutes, max_cycles=a.cycles, forever=a.forever, self_modify=a.self_modify,
                   allow_code=a.allow_code, pool=a.pool, probe_budget=a.probe_budget, candidates=a.candidates, rungs=a.rungs,
                   min_gain=a.min_gain, patience=a.patience, hook_evals=a.hook_evals, tools=a.tools, architect=a.architect, time_margin=a.time_margin, calibrate=not a.fixed_margin, coverage=a.coverage, metric=a.metric, bits_per_skill=a.bits_per_skill, sleep=a.sleep, sleep_cooldown_s=a.sleep_cooldown, forage_hosts=tuple(a.forage_host or ()), forage_wishlist=a.forage_wishlist or "", forage_max_urls=a.forage_max_urls, forage_cooldown_s=a.forage_cooldown, forage_min_novelty=a.forage_min_novelty, forage_allow_local=a.forage_allow_local, learned_sniffer=a.learned_sniffer, poll_seconds=a.poll, seed=a.seed, quiet=a.quiet,
                   chat=a.chat, chat_set=a.chat_set or [], chat_feed=not a.no_chat_feed, chat_first_minutes=a.chat_first_minutes,
                   chat_probe_minutes=a.chat_probe_minutes, chat_standard_minutes=a.chat_minutes, chat_max_minutes=a.chat_max_minutes)
    if s.minutes is None and s.max_cycles is None and not s.forever:
        s.max_cycles = 10                                    # a bare `organism` is a short, finite run; --forever is explicit
        print("[organism] no --minutes / --cycles / --forever given: running 10 decisions", file=sys.stderr)
    try:
        out = O.Organism(a.workspace, s).run()
    except RuntimeError as ex:
        print(f"[organism] {ex}", file=sys.stderr)
        return 1
    print("\n" + O.status(a.workspace))
    return 0


def register(sub, common):
    s = sub.add_parser("substrate", help="measure this machine and show the limits that follow from it")
    s.add_argument("--workspace"); s.add_argument("--refresh", action="store_true", help="measure again even if a fresh measurement is stored")
    s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_substrate)

    et = sub.add_parser("engine-tier", help="6.4: OPT-IN: let the self-modifier try single edits of nn.py / engine.py under the full judge (dry run unless --apply)")
    et.add_argument("--workspace", default="zw"); et.add_argument("--evals", type=int, default=12); et.add_argument("--seed", type=int, default=0)
    et.add_argument("--apply", action="store_true")
    et.set_defaults(fn=cmd_engine_tier)

    sh = sub.add_parser("share", help="6.4: write the zoo's discoveries and the system's own source into the chat data folder as text")
    sh.add_argument("--workspace", default="zw"); sh.add_argument("--data", help="chat data folder (default: <workspace>/chat/data)")
    sh.add_argument("--no-zoo", action="store_true"); sh.add_argument("--no-self", action="store_true")
    sh.set_defaults(fn=cmd_share)

    cp = sub.add_parser("codec", help="6.4: learned audio codec (RVQ tokens): train / encode / decode")
    cp.add_argument("action", choices=["train", "encode", "decode"]); cp.add_argument("input", nargs="?")
    cp.add_argument("--model", default="codec.npz"); cp.add_argument("--out"); cp.add_argument("--wav-dir", help="train on the .wav files in this folder (default: synthetic speech)")
    cp.add_argument("--stages", type=int, default=6); cp.add_argument("--codewords", type=int, default=128); cp.add_argument("--hidden", type=int, default=256)
    cp.add_argument("--latent", type=int, default=48); cp.add_argument("--epochs", type=int, default=100); cp.add_argument("--use-stages", type=int, help="encode with only the first N stages (a coarser code)")
    cp.set_defaults(fn=cmd_codec)

    kp = sub.add_parser("kernels", help="6.4: check the lazy-graph kernel backends (generated C, generated CUDA) on this machine")
    kp.add_argument("--size", type=int, default=2_000_000, help="elements in the benchmark graph")
    kp.add_argument("--strict", action="store_true", help="6.5: exit non-zero when the generated CUDA fails on a machine that has a GPU (default: report it and exit 0, because numpy and the C path still work)")
    kp.set_defaults(fn=cmd_kernels)

    fg = sub.add_parser("forage", help="6.3: read specific web pages from hosts you allow into an inbox folder (off unless you name the hosts)")
    fg.add_argument("urls", nargs="*"); fg.add_argument("--urls-file", help="one URL per line; # starts a comment")
    fg.add_argument("--workspace", default="zw"); fg.add_argument("--out", help="output folder (default: <workspace>/forage/inbox)")
    fg.add_argument("--allow-host", action="append", metavar="HOST", help="a host that may be contacted (repeatable; required: there are no defaults)")
    fg.add_argument("--allow-http", action="store_true", help="also allow plain http:// for the allowed hosts (https only otherwise)")
    fg.add_argument("--allow-local", action="store_true", help="allow hosts that resolve to private / loopback addresses")
    fg.add_argument("--no-robots", action="store_true", help="do not consult robots.txt (not recommended)")
    fg.add_argument("--max-mb", type=float, default=2.0, help="largest page kept"); fg.add_argument("--max-urls", type=int, default=20)
    fg.add_argument("--delay", type=float, default=2.0, help="seconds between two requests to the same host")
    fg.set_defaults(fn=cmd_forage)

    d = sub.add_parser("devour", help="digest ANY files / folders / archives into tasks (repairs damage, sets aside the rest with a reason)")
    d.add_argument("paths", nargs="*", help="files or folders; archives, tables, JSON, SQLite, numpy, text and your env files are understood")
    d.add_argument("--workspace", default="zw"); d.add_argument("--solve", action="store_true", help="then give every new task to the general model")
    d.add_argument("--target", help="column to predict in tables (default: a column named target/label/class/y/outcome/response, else the last)")
    d.add_argument("--allow-code", action="store_true", help="let environment files (class Env) run: it is YOUR code, executed after a sandboxed smoke test")
    d.add_argument("--retry", action="store_true", help="look again at files that were set aside earlier even if they have not changed")
    d.add_argument("--learned-sniffer", choices=["off", "rescue", "arbitrate"], default="rescue", help="6.4: the byte-level model that reads a file's delimiter / header / decimal mark: rescue = only files the rules gave up on (default), arbitrate = may overrule the rules, off")
    d.add_argument("--pool", default="notes", help="name of the shared text corpus that prose is added to")
    d.add_argument("--max-mb", type=float, help="read at most this many MB of one file (default: derived from free memory)")
    d.add_argument("--max-rows", type=int, help="keep at most this many rows of a table (default: derived from machine speed)")
    d.add_argument("--list", action="store_true", help="show the tasks and quarantine records from earlier runs")
    d.add_argument("--json", action="store_true"); d.add_argument("--quiet", action="store_true"); d.add_argument("--limit", type=int, default=60)
    d.add_argument("--budget", type=float, default=1.0); d.add_argument("--candidates", type=int, default=8); d.add_argument("--rungs", type=int, default=3)
    common(d); d.set_defaults(fn=cmd_devour)

    o = sub.add_parser("organism", help="the autonomous loop: watches an inbox, decides what to improve next, and changes how it searches when stuck")
    o.add_argument("action", nargs="?", choices=["run", "status", "stop"], default="run")
    o.add_argument("--workspace", default="zw"); o.add_argument("--inbox", nargs="*", help="folders / files digested every cycle")
    o.add_argument("--minutes", type=float, help="stop after this long"); o.add_argument("--cycles", type=int, help="stop after this many decisions")
    o.add_argument("--forever", action="store_true", help="keep watching the inbox when there is nothing to do (stop with `organism stop`)")
    o.add_argument("--self-modify", choices=["off", "hooks"], default="off",
                   help="'hooks': when every task has plateaued, rewrite its own search rules under the test-gated hook machinery")
    o.add_argument("--allow-code", action="store_true", help="run environment files found in the inbox (your code)")
    o.add_argument("--pool", default="notes"); o.add_argument("--probe-budget", type=float, default=0.25)
    o.add_argument("--candidates", type=int, default=6); o.add_argument("--rungs", type=int, default=3)
    o.add_argument("--min-gain", type=float, default=0.005); o.add_argument("--patience", type=int, default=3)
    o.add_argument("--hook-evals", type=int, default=200); o.add_argument("--poll", type=float, default=2.0)
    o.add_argument("--tools", action="store_true", help="6.5: answer tool requests left in <workspace>/tools/requests/ (see `tools request`)")
    o.add_argument("--architect", action="store_true", help="6.5: when a task's race plateaus, let the architect propose, test and keep edits to the winner")
    o.add_argument("--time-margin", type=float, default=1.5, help="safety factor on predicted attempt cost when fitting attempts to --minutes")
    o.add_argument("--fixed-margin", action="store_true", help="6.3: always use --time-margin (the pre-6.3 behaviour) instead of a margin measured from each task's own past overruns")
    o.add_argument("--learned-sniffer", choices=["off", "rescue", "arbitrate"], default="rescue", help="6.4: see `devour --help`")
    o.add_argument("--forage-host", action="append", metavar="HOST", help="6.4: a host the organism may read from when every task has plateaued (repeatable; none = it never touches the network; needs --forage-wishlist)")
    o.add_argument("--forage-wishlist", help="6.4: a file of URLs, one per line: YOUR list; the organism decides when to read the next ones and keeps only pages that are new to it")
    o.add_argument("--forage-max-urls", type=int, default=5); o.add_argument("--forage-cooldown", type=float, default=3600.0); o.add_argument("--forage-min-novelty", type=float, default=0.4)
    o.add_argument("--forage-allow-local", action="store_true", help=argparse.SUPPRESS)
    o.add_argument("--sleep", action="store_true", help="6.4: when every task has plateaued, consolidate the chat model's pending feedback in a long, gentle, worst-age-band-gated pass")
    o.add_argument("--sleep-cooldown", type=float, default=900.0, help="seconds between two sleep passes")
    o.add_argument("--metric", choices=["skill", "bits"], default="skill", help="6.4: what the scheduler maximises per second: per-task skill (default) or bits of information gained (metric.py)")
    o.add_argument("--bits-per-skill", type=float, default=50.0, help="6.4: with --metric bits, the assumed worth of a unit of skill on tasks with no information model (control tasks)")
    o.add_argument("--coverage", type=float, default=0.9, help="6.3: with the measured margin, the share of attempts that should fit inside the prediction (0.5-0.99)")
    o.add_argument("--chat", default="auto", metavar="off|auto|on|PATH",
                   help="the chat model as one of the tasks: 'auto' uses <workspace>/chat when it exists, 'on' creates it, or give a chat workspace path")
    o.add_argument("--chat-minutes", type=float, default=10.0, help="minutes of training in a standard chat attempt (a no-gain attempt doubles it)")
    o.add_argument("--chat-first-minutes", type=float, default=10.0,
                   help="minutes for a NEW chat model's first run (this budget also sets the model's size, so it is not a cheap probe)")
    o.add_argument("--chat-probe-minutes", type=float, default=2.0, help="minutes for the first attempt on an already trained chat model")
    o.add_argument("--chat-max-minutes", type=float, default=120.0, help="the longest a single chat attempt is allowed to be")
    o.add_argument("--chat-set", action="append", metavar="KEY=VALUE", help="override a chat config value (repeatable), e.g. --chat-set train.lr=0.002")
    o.add_argument("--no-chat-feed", action="store_true", help="do not mirror prose digested from the inbox into the chat data folder")
    o.add_argument("--quiet", action="store_true"); common(o); o.set_defaults(fn=cmd_organism)
