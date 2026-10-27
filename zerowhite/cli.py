"""
Zero White command-line application.   python -m zerowhite <command> ...   (or the `zerowhite` script after pip install)

  v5: the chat model - trains on the files you put in the workspace's data folder, learns online, talks and listens
    zerowhite chat init  -w ws                               create the workspace (data/ folder, config, state)
    zerowhite chat fetch --list | <dataset-id> ...           download public datasets into data/ (optional; any files work)
    zerowhite chat train -w ws [--minutes 60]                real training on everything in data/ (resumable, continues)
    zerowhite chat talk  -w ws [--voice]                     chat (typed or spoken); /good /bad /fix teach it online
    zerowhite chat start | stop | status | logs              the background service (web UI, API, auto-training, online learning)
    zerowhite chat convert | export                          dataset format conversion  |  model export (npz / int8 / safetensors)
    zerowhite chat voice status | train-stt | convert | ...  speech input/output, audio conversion, recogniser training
    zerowhite chat config --explain                          every setting, where its value came from, and why

  v4: the general model (no default agent or model; it builds what each task needs)
    zerowhite solve --csv data.csv --target price            any numeric table (classification or regression is detected)
    zerowhite solve --env cartpole mountaincar               environments; --env-file my_env.py for your own
    zerowhite solve --dataset digits | --text book.txt --sample 300 | --images digits
    zerowhite zoo --show                                     every agent it built, with its readable description / source
    zerowhite poet --rounds 8                                invent new environments + agents for them (open-ended)
    zerowhite evolve --csv a.csv --env cartpole --minutes 30 --self-modify hooks
    zerowhite selfmod status | hooks | patch [--apply] | rollback --hook sigma_update | history
    zerowhite doctor --workspace zw [--fix]                  troubleshoot and repair

  v3 commands (still available; nothing trains on built-in data unless you name it)

  Train the self-modifying classifier
    zerowhite train --out runs/a --task curriculum --generations 150   the synthetic shape curriculum (explicit opt-in)
    zerowhite train --out runs/a --task curriculum --resume --generations 100   ...continue it
    zerowhite train --out runs/d --dataset digits                a real dataset (digits|sonar|fashion|quickdraw|shapes)
    zerowhite eval  --run runs/a [--level 3]                     clean + FGSM accuracy
    zerowhite eval  --run runs/d --dataset digits                accuracy on the real held-out TEST split

  Let it create and improve its own models (every command continues from what exists)
    zerowhite autopilot --workspace zw --datasets digits sonar --text book.txt --envs cartpole gridworld5 --rounds 5
    zerowhite autopilot --workspace zw --datasets digits --minutes 30      keep going for 30 minutes
    zerowhite autopilot --workspace zw --datasets digits --forever         until Ctrl-C
    zerowhite make generator --workspace zw --dataset digits
    zerowhite make lm        --workspace zw --text book.txt
    zerowhite make agent     --workspace zw --env cartpole

  Use what it made
    zerowhite generate image --workspace zw --dataset digits --classes 3 7 --n 20 --out out.png
    zerowhite generate text  --workspace zw --name book --prompt "Once upon" --chars 400
    zerowhite play --workspace zw --env cartpole --episodes 5

  Inspect
    zerowhite status --workspace zw
    zerowhite report --run zw                                    writes zw/report.html
    zerowhite data list | zerowhite data fetch digits sonar fashion quickdraw
"""

import argparse
import os
import sys

import numpy as np

from . import checkpoint  # noqa: E402
from . import datasets  # noqa: E402
from . import factory  # noqa: E402
from . import fileio  # noqa: E402
from . import report  # noqa: E402
from .agents import PGAgent, make_env  # noqa: E402
from .generative import CVAE, CharLM  # noqa: E402
from .tasks import LEVELS, N_CLASSES, Curriculum, fgsm, make_shapes  # noqa: E402
from .zero_white import ZeroWhite, run_curriculum  # noqa: E402
from . import cli_v4  # noqa: E402
from . import cli_v6  # noqa: E402
from . import cli_v65  # noqa: E402
from .chat import cli as chat_cli  # noqa: E402


def _ds_kwargs(a):
    kw = {}
    if getattr(a, "qd_classes", None):
        kw["quickdraw"] = dict(classes=tuple(a.qd_classes), per_class=a.per_class)
    if getattr(a, "max_train", None):
        kw["fashion"] = dict(max_train=a.max_train)
    return kw


def _load_task(a, name=None):
    name = name or a.dataset
    return datasets.load(name, seed=a.seed, **_ds_kwargs(a).get(name, {}))


# --------------------------------------------------------------------------- train / eval
def cmd_train(a):
    if a.dataset:                                           # real data, workspace layout
        ws = factory.Workspace(a.out)
        task = _load_task(a)
        print(task.summary())
        zw, rec = factory.improve_classifier(ws, task, a.generations, a.probation, a.seed, fresh=not a.resume)
        print(f"\nFinal test acc {rec['test_acc']:.3f} (majority baseline {rec['majority_baseline']:.3f}) "
              f"hidden={rec['hidden']} patch={rec['patch']} params={rec['params']}")
        print(f"Saved to {a.out}  (zerowhite report --run {a.out})")
        return

    if a.resume:
        if not checkpoint.exists(a.out):
            sys.exit(f"nothing to resume: no checkpoint in {a.out}")
        zw, cur = checkpoint.load(a.out)
        print(f"Resumed at generation {zw.self_model.generation}")
        cur = cur or Curriculum(seed=a.seed)
    else:
        hidden = a.hidden or int(np.random.default_rng(a.seed).integers(4, 13))
        zw = ZeroWhite(64, N_CLASSES, hidden=(hidden,), num_chunks=8, seed=a.seed, candidates_per_gen=a.candidates)
        cur = Curriculum(seed=a.seed, promote_acc=a.promote_acc)
    if a.task == "shapes":
        cur.pin(a.level)
    every = max(1, a.checkpoint_every)

    def on_gen(z, c):
        if z.self_model.generation % every == 0:
            checkpoint.save(z, a.out, c)

    run_curriculum(zw, cur, a.generations, a.probation, on_generation=on_gen)
    checkpoint.save(zw, a.out, cur)
    zw.self_model.save(os.path.join(a.out, "run_journal.json"))
    loss, acc = zw.evaluate(zw.net, *cur.data()[2:])
    print(f"\nFinal [{cur.cfg['name']}] val_loss={loss:.4f} val_acc={acc:.3f} hidden={zw.net.hidden_sizes()} "
          f"layers={len(zw.net.layers)} patch={zw.net.use_patch} params={zw.net.num_params()}")
    print(f"Saved to {a.out}  (run `zerowhite report --run {a.out}`)")


def cmd_eval(a):
    if a.dataset:
        zw = factory.load_classifier(factory.Workspace(a.run), a.dataset)
        if zw is None:
            sys.exit(f"no classifier for '{a.dataset}' in {a.run}; train one first")
        task = _load_task(a)
        loss, acc = zw.evaluate(zw.net, task.Xte, task.yte)
        print(f"{task.name}: TEST acc={acc:.3f} loss={loss:.4f}  ({len(task.Xte)} held-out rows, never used for selection)")
        return
    if not checkpoint.exists(a.run):
        sys.exit(f"no checkpoint in {a.run}")
    zw, cur = checkpoint.load(a.run)
    lv = a.level if a.level is not None else (cur.level if cur else 0)
    X, y = make_shapes(lv, 70, seed=999)
    loss, acc = zw.evaluate(zw.net, X, y)
    print(f"{LEVELS[lv]['name']}: clean acc={acc:.3f} loss={loss:.4f}")
    eps = LEVELS[lv]["adv_eps"]
    if eps:
        l2, a2 = zw.evaluate(zw.net, fgsm(zw.net, X, y, eps), y)
        print(f"{LEVELS[lv]['name']}: FGSM(eps={eps}) acc={a2:.3f} loss={l2:.4f}")


# --------------------------------------------------------------------------- create / improve
def cmd_autopilot(a):
    ws = factory.Workspace(a.workspace)
    if not (a.datasets or a.text or a.envs):
        sys.exit("nothing to do: give --datasets, --text and/or --envs")
    rounds = 10 ** 9 if a.forever else a.rounds
    seconds = a.minutes * 60 if a.minutes else None
    recs = factory.autopilot(ws, a.datasets or (), a.text or (), a.envs or (), rounds=rounds, seconds=seconds,
                             seed=a.seed, class_generations=a.class_generations, dataset_kwargs=_ds_kwargs(a),
                             scale=a.scale)
    print(f"\n{len(recs)} jobs finished. Report: zerowhite report --run {a.workspace}")


def cmd_make(a):
    ws = factory.Workspace(a.workspace)
    if a.what == "generator":
        task = _load_task(a)
        zw = factory.load_classifier(ws, task.name)
        if zw is None:
            print(f"no classifier for '{task.name}' yet - training one first (the generator is judged by it)")
            zw, _ = factory.improve_classifier(ws, task, a.class_generations, seed=a.seed, verbose=False)
        _, rec = factory.improve_generator(ws, task, zw, a.generations, seed=a.seed)
        print(f"\ngenerator '{task.name}': fidelity {rec['fidelity']:.3f} diversity {rec['diversity']:.3f} "
              f"elbo {rec['elbo']:.1f} improved={rec['improved']}")
    elif a.what == "lm":
        name = a.name or os.path.splitext(os.path.basename(a.text))[0]
        _, _, rec = factory.improve_lm(ws, name, factory.read_corpus(a.text), a.generations, seed=a.seed, fresh=a.fresh)
        print(f"\nlanguage model '{name}': {rec['bpc']:.3f} bits/char (uniform {rec['uniform_bpc']:.2f}) improved={rec['improved']}")
        print(rec["sample"])
    else:
        _, rec = factory.improve_agent(ws, a.env, a.generations, seed=a.seed, scale=a.scale)
        print(f"\nagent '{a.env}': return {rec['eval_return']:.2f} success {rec['success']:.2f} improved={rec['improved']}")


def cmd_generate(a):
    ws = factory.Workspace(a.workspace)
    if a.kind == "image":
        path = os.path.join(ws.root, "generators", a.dataset, "cvae.npz")
        if not os.path.exists(path):
            sys.exit(f"no generator for '{a.dataset}'; run: zerowhite make generator --workspace {a.workspace} --dataset {a.dataset}")
        g = CVAE.load(path)
        task = _load_task(a)
        classes = a.classes or list(range(task.n_classes))
        y = np.repeat(classes, a.n)
        h, w = task.image_shape
        img = fileio.image_grid(g.sample(y, np.random.default_rng(a.seed), a.temperature).reshape(-1, h, w),
                                cols=a.n, scale=max(1, 112 // max(h, w)))
        fileio.write_png(a.out, img)
        print(f"wrote {a.out}: {len(classes)} rows x {a.n} samples, classes {[task.class_names[c] for c in classes]}")
    else:
        path = os.path.join(ws.root, "lm", a.name, "lm.npz")
        if not os.path.exists(path):
            sys.exit(f"no language model '{a.name}' in {a.workspace}")
        lm, vocab = CharLM.load(path)
        print(lm.generate(vocab, a.prompt, a.chars, a.temperature, np.random.default_rng(a.seed)))


def cmd_play(a):
    path = os.path.join(a.workspace, "agents", a.env, "agent.npz")
    if not os.path.exists(path):
        sys.exit(f"no agent for '{a.env}'; run: zerowhite make agent --workspace {a.workspace} --env {a.env}")
    agent, env = PGAgent.load(path), make_env(a.env)
    for i in range(a.episodes):
        _, _, r, _, won = agent._episode(env, np.random.default_rng(a.seed + i))
        print(f"episode {i + 1}: steps={len(r)} return={r.sum():.2f} {'reached goal/limit' if won else 'failed'}")


# --------------------------------------------------------------------------- inspect
def cmd_status(a):
    ws = factory.Workspace(a.workspace)
    jobs = ws.journal()
    print(f"workspace {ws.root}: {len(jobs)} jobs recorded")
    for kind, keys in (("classifier", ("test_acc", "majority_baseline", "params")),
                       ("generators", ("fidelity", "diversity", "elbo")),
                       ("lm", ("bpc", "uniform_bpc", "chars")),
                       ("agents", ("eval_return", "success", "updates"))):
        base = os.path.join(ws.root, kind)
        for name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            b = ws.best(kind, name)
            if b:
                print(f"  {kind:10s} {name:12s} " + "  ".join(f"{k}={b[k]:.3f}" if isinstance(b[k], float) else f"{k}={b[k]}"
                                                           for k in keys if k in b))


def cmd_report(a):
    print("wrote", report.write(a.run, a.out))


def cmd_data(a):
    if a.action == "list":
        print("digits    UCI Optical Recognition of Handwritten Digits (bundled subset works offline)")
        print("sonar     UCI Sonar, Mines vs. Rocks")
        print("fashion   Fashion-MNIST")
        print("quickdraw Quick, Draw! bitmaps (default classes: " + ", ".join(datasets.QUICKDRAW_DEFAULT) + ")")
        print("shapes    built-in synthetic shapes")
        print(f"cache: {datasets.data_dir()}")
        return
    for name in a.names:
        try:
            print(datasets.load(name, seed=0, **_ds_kwargs(a).get(name, {})).summary())
        except datasets.DatasetUnavailable as e:
            print(f"{name}: UNAVAILABLE\n{e}")


# --------------------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(description="Zero White", formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, dataset=False):
        p.add_argument("--seed", type=int, default=0)
        if dataset:
            p.add_argument("--qd-classes", nargs="+", help="Quick, Draw! classes (default circle square star zigzag)")
            p.add_argument("--per-class", type=int, default=3000, help="Quick, Draw! drawings per class")
            p.add_argument("--max-train", type=int, help="cap Fashion-MNIST training rows (faster)")

    t = sub.add_parser("train", help="train the self-modifying classifier")
    t.add_argument("--out", default="runs/default"); t.add_argument("--generations", type=int, default=150)
    t.add_argument("--probation", type=int, default=20); t.add_argument("--hidden", type=int, default=None, help="starting width (default: random)")
    t.add_argument("--candidates", type=int, default=5)
    t.add_argument("--dataset", choices=sorted(datasets.LOADERS), help="train on a real dataset")
    t.add_argument("--task", choices=["curriculum", "shapes"], default=None,
                   help="the built-in synthetic shape task; only used when you ask for it (otherwise pass --dataset)")
    t.add_argument("--level", type=int, default=0)
    t.add_argument("--promote-acc", type=float, default=None, help="override per-level thresholds")
    t.add_argument("--resume", action="store_true"); t.add_argument("--checkpoint-every", type=int, default=10)
    common(t, True); t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="evaluate a saved classifier")
    e.add_argument("--run", required=True); e.add_argument("--level", type=int)
    e.add_argument("--dataset", choices=sorted(datasets.LOADERS))
    common(e, True); e.set_defaults(fn=cmd_eval)

    r = sub.add_parser("report", help="write report.html")
    r.add_argument("--run", required=True); r.add_argument("--out"); r.set_defaults(fn=cmd_report)

    s = sub.add_parser("status", help="summarise a workspace")
    s.add_argument("--workspace", required=True); s.set_defaults(fn=cmd_status)

    d = sub.add_parser("data", help="list / download datasets")
    d.add_argument("action", choices=["list", "fetch"]); d.add_argument("names", nargs="*", default=[])
    common(d, True); d.set_defaults(fn=cmd_data)

    ap_ = sub.add_parser("autopilot", help="keep creating and improving every model, unattended")
    ap_.add_argument("--workspace", required=True)
    ap_.add_argument("--datasets", nargs="*", choices=sorted(datasets.LOADERS))
    ap_.add_argument("--text", nargs="*", help="text file(s) to learn to write like")
    ap_.add_argument("--envs", nargs="*", help="gridworld5, cartpole")
    ap_.add_argument("--rounds", type=int, default=1); ap_.add_argument("--minutes", type=float)
    ap_.add_argument("--forever", action="store_true"); ap_.add_argument("--class-generations", type=int, default=15)
    ap_.add_argument("--scale", type=float, default=1.0, help="multiply every training budget (0.2 quick, 3 deep)")
    common(ap_, True); ap_.set_defaults(fn=cmd_autopilot)

    m = sub.add_parser("make", help="create / improve one model")
    m.add_argument("what", choices=["generator", "lm", "agent"]); m.add_argument("--workspace", required=True)
    m.add_argument("--dataset", choices=sorted(datasets.LOADERS)); m.add_argument("--text"); m.add_argument("--name")
    m.add_argument("--env"); m.add_argument("--generations", type=int, default=3)
    m.add_argument("--class-generations", type=int, default=15)
    m.add_argument("--fresh", action="store_true", help="ignore the existing incumbent")
    m.add_argument("--scale", type=float, default=1.0, help="multiply the agent training budget")
    common(m, True); m.set_defaults(fn=cmd_make)

    g = sub.add_parser("generate", help="sample from a model Zero White made")
    g.add_argument("kind", choices=["image", "text"]); g.add_argument("--workspace", required=True)
    g.add_argument("--dataset", default="digits"); g.add_argument("--classes", nargs="*", type=int)
    g.add_argument("--n", type=int, default=10); g.add_argument("--out", default="samples.png")
    g.add_argument("--name"); g.add_argument("--prompt", default=""); g.add_argument("--chars", type=int, default=300)
    g.add_argument("--temperature", type=float, default=0.8)
    common(g, True); g.set_defaults(fn=cmd_generate)

    p = sub.add_parser("play", help="watch a trained agent")
    p.add_argument("--workspace", required=True); p.add_argument("--env", required=True)
    p.add_argument("--episodes", type=int, default=5); common(p); p.set_defaults(fn=cmd_play)
    cli_v4.register(sub, common)
    cli_v6.register(sub, common)
    cli_v65.register(sub, common)
    chat_cli.register(sub, common)
    return ap


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):                # a cp1252 console must never crash a print
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.cmd == "make":
        need = {"generator": "dataset", "lm": "text", "agent": "env"}[a.what]
        if not getattr(a, need):
            ap.error(f"make {a.what} needs --{need}")
    if a.cmd == "train":
        if a.dataset is None and a.task is None and not a.resume:
            ap.error("train needs something to train on: --dataset NAME (digits|sonar|fashion|quickdraw), or --task curriculum|shapes "
                     "for the built-in synthetic shapes. (For the chat model use `zerowhite chat train`, which reads your data folder.)")
        if a.dataset is None and a.task == "shapes" and not 0 <= a.level < len(LEVELS):
            ap.error(f"--level must be 0..{len(LEVELS) - 1}")
    code = a.fn(a)
    if isinstance(code, int):
        sys.exit(code)


if __name__ == "__main__":
    main()
