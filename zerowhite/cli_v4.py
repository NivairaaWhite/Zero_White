"""cli_v4.py - commands added in v4: solve, selfmod, evolve, poet, zoo, doctor."""

import json
import os
import time

import numpy as np

from . import datasets, doctor, factory, fileio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _task_from_args(a):
    from .forge import task as T
    tasks = []
    for p in a.csv or []:
        tasks.append(T.from_csv(p, target=a.target, kind=a.kind, seed=a.seed))
    for n in a.env or []:
        tasks.append(T.from_env_name(n))
    for p in a.env_file or []:
        tasks.append(T.load_env_file(p))
    for n in a.dataset or []:
        tasks.append(T.from_dataset_task(datasets.load(n, seed=a.seed)))
    for p in a.text or []:
        tasks.append(T.TextTask(os.path.splitext(os.path.basename(p))[0], factory.read_corpus(p)))
    for n in a.images or []:
        tasks.append(T.ImageGenTask(T.from_dataset_task(datasets.load(n, seed=a.seed))))
    return tasks


def cmd_solve(a):
    from .forge.generalist import Generalist
    tasks = _task_from_args(a)
    if not tasks:
        raise SystemExit("solve needs a task: --csv FILE | --env NAME | --env-file FILE | --dataset NAME | --text FILE | --images DATASET")
    g = Generalist(a.workspace, seed=a.seed)
    for t in tasks:
        r = g.solve(t, budget=a.budget, n_new=a.candidates, rungs=a.rungs)
        if t.kind == "text" and a.sample:
            print("\n--- sample ---\n" + r["_agent"].sample(t, prompt=a.prompt, n=a.sample, seed=a.seed) + "\n--------------")
        if t.kind == "image_gen":
            out = os.path.join(a.workspace, "samples", f"{t.name}.png")
            h, w = t.sup.image_shape
            classes = list(range(min(t.sup.n_classes, 10)))
            y = np.repeat(classes, 8)
            imgs = r["_agent"].model.sample(y, np.random.default_rng(a.seed), 0.8).reshape(-1, h, w)
            fileio.write_png(out, fileio.image_grid(imgs, cols=8, scale=max(1, 112 // max(h, w))))
            print(f"samples written to {out}")


def cmd_zoo(a):
    from .forge.zoo import Zoo
    z = Zoo(a.workspace)
    if not z.tasks():
        print("the zoo is empty: no agents have been built yet (run `solve`)")
        return
    for t in z.tasks():
        b = z.best(t)
        if not b:
            continue
        sp = z.spec(t, b["id"])
        print(f"{t:<28} {sp['family']:<16} skill={sp['skill']:.3f}  kind={sp['kind']}  from={sp.get('origin', '?')}")
        if a.show:
            print("    " + sp["description"].replace("\n", "\n    "))


def cmd_poet(a):
    from .forge.poet import poet
    r = poet(a.workspace, rounds=a.rounds, seed=a.seed, units=a.units)
    fileio.write_json(os.path.join(a.workspace, "forge", "poet.json"), r, indent=2)
    print(f"\n{len(r['environments'])} environments in the archive (details: {a.workspace}/forge/poet.json)")


def cmd_doctor(a):
    if a.seal:
        d = doctor.seal()
        print(f"sealed {len(d)} constitution files -> {doctor.SEAL}")
        return 0
    return doctor.report(doctor.run(a.workspace, fix=a.fix, quick=a.quick))


def _store(a):
    from .selfmod.history import HistoryStore
    return HistoryStore(os.path.join(a.workspace, "selfmod", "history"))


def cmd_selfmod(a):
    from .selfmod import bench, hooks, patcher
    if a.action == "bench":
        bench.main(a.seeds)
        return
    if a.action == "status":
        mgr = hooks.HookManager(a.workspace)
        print("hooks (level 1, evolved functions):")
        for h, s in mgr.status().items():
            print(f"  {h:<15} {'LIVE v' + str(s['version']) if s['live'] else 'genesis (shipped trivial version)'}")
        names = _store(a).names()
        print("source patches (level 2):", [n for n in names if n.startswith('src:')] or "none")
        for h in mgr.status():
            src = mgr.live_source(h)
            if src:
                print(f"\n--- live {h} ---\n{src.strip()}")
        return
    if a.action == "history":
        for ev in _store(a).log(a.name):
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev["t"])), ev["event"], ev["name"], "v%s" % ev.get("version"),
                  ev["sha"][:10], json.dumps(ev.get("meta", {}))[:100])
        return
    if a.action == "hooks":
        mgr = hooks.HookManager(a.workspace)
        names = list(hooks.evolved.HOOKS) if a.hook == "all" else [a.hook]
        for h in names:
            hooks.evolve_hook(mgr, h, evals=a.evals, seed=a.seed)
        return
    if a.action == "rollback":
        if a.hook:
            r = hooks.HookManager(a.workspace).rollback(a.hook, a.steps)
            print(f"hook {a.hook} rolled back to " + (f"version {r['version']}" if r else "genesis"))
        elif a.file:
            r = patcher.rollback_file(REPO_ROOT, _store(a), a.file, a.steps)
            print(f"{a.file} restored to version {r['version']} (sha {r['sha'][:10]})")
        else:
            raise SystemExit("rollback needs --hook NAME or --file NAME.py")
        return
    if a.action == "patch":
        if not os.path.isdir(os.path.join(REPO_ROOT, "tests")):
            raise SystemExit("source patching needs a source checkout that includes the tests/ directory "
                             "(the test suite is the gate); an installed package cannot patch itself")
        rep = patcher.search(REPO_ROOT, _store(a), evals=a.evals, seed=a.seed, apply=a.apply)
        base = patcher.save_report(a.workspace, rep)
        print(f"\n{rep.note}\nreport: {base}.json" + (f"\ndiff:   {base}.diff" if rep.diff else ""))
        for p in rep.accepted:
            print(f"  {p['file']}:{p['line']}  {p['name']}: {p['old']} -> {p['new']}")
        return


def cmd_evolve(a):
    """The unattended loop: keep solving the given tasks AND keep improving Zero White itself."""
    from .forge.generalist import Generalist
    from .selfmod import hooks, patcher
    tasks = _task_from_args(a)
    g = Generalist(a.workspace, seed=a.seed, verbose=not a.quiet)
    mgr = hooks.HookManager(a.workspace).activate()
    t_end = time.time() + a.minutes * 60 if a.minutes else None
    rd = 0
    while True:
        rd += 1
        print(f"\n===== evolve round {rd} =====", flush=True)
        for t in tasks:
            g.solve(t, budget=a.budget, n_new=a.candidates)
        if a.self_modify in ("hooks", "patch") and rd % a.every == 0:
            h = ("sigma_update", "select_parent")[(rd // a.every) % 2]
            hooks.evolve_hook(mgr, h, evals=a.hook_evals, seed=a.seed + rd)
            g = Generalist(a.workspace, seed=a.seed + rd, verbose=not a.quiet)       # pick up the new live hooks
        if a.self_modify == "patch" and rd % (a.every * 3) == 0 and os.path.isdir(os.path.join(REPO_ROOT, "tests")):
            rep = patcher.search(REPO_ROOT, _store(a), evals=a.patch_evals, seed=a.seed + rd, apply=True)
            patcher.save_report(a.workspace, rep)
            print(f"[evolve] source patching: {rep.note}", flush=True)
        if (a.rounds and rd >= a.rounds) or (t_end and time.time() >= t_end):
            break
        if not a.rounds and not t_end and not a.forever:
            break
    print("\nevolve finished; `zoo` lists what was built, `selfmod status` shows what Zero White rewrote in itself")


def register(sub, common):
    def task_args(p):
        p.add_argument("--csv", nargs="*", help="numeric CSV with a header row (target = last column unless --target)")
        p.add_argument("--target"); p.add_argument("--kind", choices=["classification", "regression"])
        p.add_argument("--env", nargs="*", help="built-in env: gridworld5, cartpole, mountaincar, cartpole:g=15,len=1")
        p.add_argument("--env-file", nargs="*", help="your own environment: a .py file defining class Env (see docs)")
        p.add_argument("--dataset", nargs="*", choices=sorted(datasets.LOADERS))
        p.add_argument("--text", nargs="*", help="text file(s) to learn to continue")
        p.add_argument("--images", nargs="*", choices=sorted(datasets.LOADERS), help="image dataset(s) to learn to draw")

    s = sub.add_parser("solve", help="give the general model a task; it builds the agent(s) for it")
    s.add_argument("--workspace", default="zw"); task_args(s)
    s.add_argument("--budget", type=float, default=1.0, help="training budget multiplier")
    s.add_argument("--candidates", type=int, default=8, help="new agents proposed per task")
    s.add_argument("--rungs", type=int, default=3)
    s.add_argument("--sample", type=int, default=0, help="for text tasks: print this many characters")
    s.add_argument("--prompt", default="")
    common(s); s.set_defaults(fn=cmd_solve)

    z = sub.add_parser("zoo", help="list the agents the general model has built")
    z.add_argument("--workspace", default="zw"); z.add_argument("--show", action="store_true", help="print each agent's description / source")
    z.set_defaults(fn=cmd_zoo)

    p = sub.add_parser("poet", help="open-ended co-evolution: invent new environments and agents for them")
    p.add_argument("--workspace", default="zw"); p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--units", type=float, default=0.5); common(p); p.set_defaults(fn=cmd_poet)

    d = sub.add_parser("doctor", help="diagnose (and with --fix repair) the install, workspace and self-written code")
    d.add_argument("--workspace"); d.add_argument("--fix", action="store_true"); d.add_argument("--quick", action="store_true")
    d.add_argument("--seal", action="store_true", help="re-seal the constitution files (developers)")
    d.set_defaults(fn=cmd_doctor)

    m = sub.add_parser("selfmod", help="inspect / drive Zero White's rewriting of its own code")
    m.add_argument("action", choices=["status", "hooks", "patch", "bench", "rollback", "history"])
    m.add_argument("--workspace", default="zw"); m.add_argument("--hook", default="all")
    m.add_argument("--file"); m.add_argument("--name"); m.add_argument("--steps", type=int, default=1)
    m.add_argument("--evals", type=int, default=400); m.add_argument("--seeds", nargs="*", type=int, default=[0])
    m.add_argument("--apply", action="store_true", help="patch: write accepted patches into the real source tree")
    common(m); m.set_defaults(fn=cmd_selfmod)

    e = sub.add_parser("evolve", help="unattended: keep solving tasks and keep improving Zero White itself")
    e.add_argument("--workspace", default="zw"); task_args(e)
    e.add_argument("--rounds", type=int, default=0); e.add_argument("--minutes", type=float)
    e.add_argument("--forever", action="store_true"); e.add_argument("--quiet", action="store_true")
    e.add_argument("--budget", type=float, default=1.0); e.add_argument("--candidates", type=int, default=6)
    e.add_argument("--self-modify", choices=["off", "hooks", "patch"], default="off",
                   help="hooks: rewrite its own search rules; patch: also patch its own source (needs a checkout + tests)")
    e.add_argument("--every", type=int, default=2, help="self-modify every N rounds")
    e.add_argument("--hook-evals", type=int, default=300); e.add_argument("--patch-evals", type=int, default=12)
    common(e); e.set_defaults(fn=cmd_evolve)
