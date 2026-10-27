"""cli_v65.py - the commands added in 6.5: tools, memory, stream, act, architect.

    zerowhite tools   list | show NAME | forge NAME --examples FILE | run NAME --input JSON | chain A B --input JSON | rm NAME | request NAME --examples FILE | process
    zerowhite memory  status | fit | recall <task>
    zerowhite stream  demo | check FILE
    zerowhite act     demo | serve-mock | request METHOD PATH --base-url URL --allow-host HOST | ledger | kill | unkill
    zerowhite architect  run <task> | report | export | ask

`act` is the only way anything in Zero White sends data to a web server, it is never started by the organism or by self-modification, and with no --allow-host it refuses everything.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from . import fileio
from .cli_v4 import REPO_ROOT, _task_from_args, task_args


def _json_arg(text):
    """A JSON value, or @FILE to read one, or (failing both) the raw string."""
    if text is None:
        return None
    if text.startswith("@"):
        text = fileio.read_text(text[1:])
    try:
        return json.loads(text)
    except ValueError:
        return text


def _examples(path):
    ex = json.loads(fileio.read_text(path))
    if isinstance(ex, dict):
        ex = ex.get("examples", [])
    if not isinstance(ex, list) or not ex:
        raise SystemExit(f"{path}: expected a JSON list of {{\"input\": ..., \"output\": ...}} objects (or a pair list)")
    return ex


# =========================================================================== tools
def cmd_tools(a):
    from .forge import tools as TL
    tb = TL.Toolbox(a.workspace)
    act, names = a.action, a.name or []
    if act == "list":
        rows = tb.list()
        if not rows:
            print("no tools yet (`tools forge NAME --examples FILE` makes one)")
        for r in rows:
            print(f"  {r['name']:<22} {str(r['in_type']) + ' -> ' + str(r['out_type']):<24} used {r['uses']}x, failed {r['failures']}x{'  validated' if r['validated'] else ''}  {r['description'][:60]}")
        return 0
    if act in ("show", "run", "rm", "forge", "request") and len(names) != 1:
        print(f"tools {act} needs exactly one tool name", file=sys.stderr)
        return 2
    if act == "show":
        print(json.dumps(tb.spec(names[0]), indent=2, default=str))
        print("\n" + tb.source(names[0]))
        return 0
    if act == "forge":
        if not a.examples:
            print("tools forge needs --examples FILE (a JSON list of {\"input\":..., \"output\":...})", file=sys.stderr)
            return 2
        r = tb.forge(names[0], _examples(a.examples), a.description or "", seconds=a.seconds, replace=a.replace)
        print(json.dumps(r.to_dict(), indent=2, default=str) if a.json else (f"forged '{r.name}': {r.expr}  ({r.train_n} training + {r.holdout_n} held-out examples, "
              f"{'held-out validated' if r.validated else 'NOT validated on held-out examples'}, {r.explored} programs explored, {r.seconds:.1f}s)" if r.ok else f"could not forge '{r.name}': {r.reason}"))
        return 0 if r.ok else 1
    if act == "run":
        print(json.dumps(tb.run(names[0], _json_arg(a.input)), default=str))
        return 0
    if act == "chain":
        if not names:
            print("tools chain needs the tool names in order", file=sys.stderr)
            return 2
        print(json.dumps(tb.chain(names, _json_arg(a.input)), default=str))
        return 0
    if act == "rm":
        tb.remove(names[0])
        print(f"removed '{names[0]}'")
        return 0
    if act == "request":
        d = TL.requests_dir(a.workspace)
        os.makedirs(d, exist_ok=True)
        TL.Toolbox(a.workspace).dir(names[0])                                      # validates the name
        fileio.write_json(os.path.join(d, names[0] + ".json"), dict(name=names[0], description=a.description or "", examples=_examples(a.examples), seconds=a.seconds), indent=2)
        print(f"request written to {os.path.join(d, names[0] + '.json')}; `tools process` (or `organism --tools`) answers it")
        return 0
    if act == "process":
        res = TL.process_requests(a.workspace, limit=a.limit, seconds=a.seconds)
        if not res:
            print("no pending requests")
        for r in res:
            print(f"  {'forged' if r.ok else 'FAILED'} {r.name}: {r.expr if r.ok else r.reason}")
        return 0
    return 2


# =========================================================================== memory
def cmd_memory(a):
    from .forge.episodic import EpisodicMemory
    mem = EpisodicMemory(a.workspace, seed=a.seed)
    if a.action == "status":
        print(mem.describe())
        print(f"nearest-neighbour retrieval accuracy on its own records: {mem.retrieval_accuracy():.3f}" if len(mem) > 4 else "(too few records to measure retrieval yet)")
        return 0
    if a.action == "fit":
        print(json.dumps(mem.fit(steps=a.steps, seed=a.seed), default=str))
        mem.save()
        return 0
    tasks = _task_from_args(a)
    if not tasks:
        print("memory recall needs a task: --csv FILE | --env NAME | --dataset NAME | --text FILE", file=sys.stderr)
        return 2
    for t in tasks:
        rec = mem.recall(t, k=a.k, exclude_task=False)
        print(f"{t.name}: {len(rec)} similar earlier champion(s)")
        for r in rec:
            print(f"  cosine {r['score']:.2f}  '{r['task']}'  {r.get('family')}  skill {r.get('skill')}")
    return 0


# =========================================================================== stream
def _demo_models(text, seed=0):
    from .chat.tokenizer import Tokenizer
    from .stream import PixelCodec
    from .voice import codec as C
    tok = Tokenizer.train([text] if text else ["the quick brown fox jumps over the lazy dog. " * 40], 320, ["<|pad|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"], min_frequency=2)
    r = np.random.default_rng(seed)
    waves = [np.sin(2 * np.pi * f * np.arange(8000) / 16000.0) * r.uniform(0.3, 0.8) + 0.05 * r.normal(size=8000) for f in (110, 220, 330, 440)]
    cd = C.train(waves, stages=3, codewords=16, hidden=32, latent=8, epochs=4, finetune_epochs=2, seed=seed)
    frames = [_scene(t) for t in range(12)]
    pc = PixelCodec(patch=4).fit(frames, stages=2, codewords=8, seed=seed)
    return tok, cd, pc


def _scene(t, size=16):
    f = np.zeros((size, size))
    x = 2 + (t % 10)
    f[4:8, x:x + 4] = 1.0
    f[12:14, 0:3] = 0.5
    return f


def cmd_stream(a):
    from .chat.model import GPT, GPTConfig
    from .stream import AudioStream, StreamPipeline, TextStream, TokenSpace, VideoStream, WindowLM
    if a.action == "check":
        if not a.file:
            print("stream check needs a text FILE", file=sys.stderr)
            return 2
        data = open(a.file, "rb").read()
        text = data.decode("utf-8", errors="replace")
        tok = _demo_models(text[:200000])[0]
        want = tok.encode(text)
        rng = np.random.default_rng(a.seed)
        bad = 0
        for trial in range(a.trials):
            ts, out, i = TextStream(tok), [], 0
            while i < len(data):
                k = int(rng.integers(1, a.chunk + 1))
                out.extend(ts.feed(data[i:i + k]))
                i += k
            out.extend(ts.finish())
            bad += int(out != want)
        print(f"{len(data):,} bytes -> {len(want):,} tokens; {a.trials} random chunkings (cuts of 1..{a.chunk} bytes, inside multi-byte characters too): "
              f"{'all identical to the batch tokeniser' if not bad else str(bad) + ' DIFFERED'}")
        return 0 if not bad else 1
    tok, cd, pc = _demo_models("", a.seed)
    sp = TokenSpace.for_models(tok, cd, pc)
    cfg = GPTConfig(vocab=sp.vocab_size, d_model=24, n_layer=2, n_head=2, ffn=48, ctx=96, dtype="float64")
    lm = WindowLM(GPT(cfg, seed=a.seed, init_std=0.2), window=96, stride=16)
    now = [0.0]
    pipe = StreamPipeline(sp, clock=lambda: now[0], on_tokens=lm.feed)
    pipe.add_text("chat", TextStream(tok))
    pipe.add_audio("mic", AudioStream(cd), capacity=1 << 14)
    pipe.add_video("cam", VideoStream(pc), shape=(16, 16), capacity=4)
    r = np.random.default_rng(a.seed)
    for step in range(a.steps):
        now[0] = step * 0.1
        pipe.push("chat", "hello stream " * 2)
        pipe.push("mic", np.sin(np.arange(1600) * 0.05 + step) * 0.4 + 0.05 * r.normal(size=1600))
        pipe.push("cam", np.stack([_scene(step * 2), _scene(step * 2 + 1)]))
        now[0] += 0.05
        pipe.pump()
    pipe.flush()
    print(f"streamed {a.steps} steps of text + audio + video into one {sp.vocab_size}-id token space; the model's window slid {lm.slides} time(s) over {lm.seen} tokens")
    for name, st in pipe.stats().items():
        print(f"  {name:<5} {st['ids']:>6} ids, {st['dropped']} rows dropped, latency mean {st['latency_mean'] * 1000:.0f} ms / max {st['latency_max'] * 1000:.0f} ms")
    print("(random untrained model: this shows the plumbing, not understanding; see docs/STREAM.md)")
    return 0


# =========================================================================== act
def _act_dir(ws):
    d = os.path.join(os.path.abspath(ws), "act")
    os.makedirs(d, exist_ok=True)
    return d


def cmd_act(a):
    from .forage import act as ACT
    d = _act_dir(a.workspace)
    ledger, kill = os.path.join(d, "ledger.jsonl"), os.path.join(d, "KILL")
    if a.action == "kill":
        fileio.write_text(kill, "stop\n")
        print(f"kill switch set: {kill} (every request is refused while it exists; `act unkill` removes it)")
        return 0
    if a.action == "unkill":
        if os.path.exists(kill):
            os.remove(kill)
        print("kill switch cleared")
        return 0
    if a.action == "ledger":
        if not os.path.exists(ledger):
            print("no requests have been made (the ledger is empty)")
            return 0
        rows = [json.loads(ln) for ln in open(ledger, encoding="utf-8") if ln.strip()][-a.tail:]
        for r in rows:
            print(f"  {time.strftime('%H:%M:%S', time.localtime(r.get('t', 0)))} {r.get('verdict', ''):<8} {r.get('method', ''):<6} {r.get('host', '')}{r.get('path', '')} -> {r.get('status')} {r.get('reason', '')}")
        return 0
    if a.action == "serve-mock":
        from .forage.mockapi import MockServer
        with MockServer() as srv:
            print(f"mock API on {srv.url} (loopback only; user alice / password 'correct horse battery'). Ctrl-C stops it.")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\nstopped")
        return 0
    if a.action == "demo":
        from .forge import task as T
        from .forge.families import FAMILIES
        from .forage.mockapi import MockServer
        from .forage.webenv import mock_env
        with MockServer(lock_seconds=0.05) as srv:
            pol = ACT.ActPolicy(write_hosts=("127.0.0.1",), max_requests=20000, max_requests_per_episode=20, kill_file=kill)
            env = mock_env(srv.url, srv.state.reset, policy=pol, ledger_path=ledger)
            task = T.EnvTask(env)
            print(f"practice task: log in (the password is a secret the learner can only NAME), fetch the CSRF value, create a 'widget', set its quantity to 5; actions: {', '.join(env.action_names())}")
            print(f"random policy: mean return {task.random_baseline()[0]:+.2f}")
            ag = FAMILIES["actor_critic"].build(task, dict(hidden=24, depth=1, lr=0.01, gamma=0.97, entropy=0.02, curiosity=0.0), a.seed)
            for k in range(a.rounds):
                ag.train(task, 1.0, np.random.default_rng(a.seed + k))
                skill, met = ag.val_score(task)
                print(f"  round {k + 1}/{a.rounds}: success {met['success']:.0%}, mean return {met['mean_return']:+.2f}, skill {skill:.2f}")
            # One episode of the policy AS TRAINED AND SCORED above (it samples its action; the argmax of a policy trained with an entropy bonus can loop on one action,
            # so showing greedy play here would describe a different policy from the one whose success rate was just printed).
            rng = np.random.default_rng(a.seed)
            o, done, won = env.reset(rng), False, False
            while not done:
                o, _, done, won = env.step(ag.agent.act(o, rng))
            print(f"one sampled episode ({'goal reached' if won else 'goal NOT reached'}): " + " -> ".join(f"{n}[{s}]" for n, s in env.history))
            st = env.session.actor.stats()
            print(f"{st['requests']} requests sent, {st['refused']} refused by the policy; every one is in {ledger} (secrets and tokens redacted)")
        return 0
    if a.action == "request":
        if not a.base_url or not a.method or not a.path:
            print("act request METHOD PATH --base-url URL --allow-host HOST", file=sys.stderr)
            return 2
        secrets = {}
        for item in a.secret_env or []:
            name, _, var = item.partition("=")
            if not var or var not in os.environ:
                print(f"--secret-env {item}: environment variable '{var}' is not set", file=sys.stderr)
                return 2
            secrets[name] = os.environ[var]
        pol = ACT.ActPolicy(write_hosts=tuple(a.allow_host or ()), allow_remote=a.allow_remote, allow_delete=a.allow_delete, max_requests=a.max_requests, kill_file=kill)
        actor = ACT.Actor(a.base_url, pol, ledger_path=ledger, secrets=secrets, dry_run=a.dry_run)
        body = ACT.ref_from_json(_json_arg(a.json)) if a.json else None
        headers = {}
        for h in a.header or []:
            k, _, v = h.partition("=")
            headers[k] = ACT.ref_from_json(_json_arg(v)) if v.startswith("{") else v
        resp = actor.request(a.method, a.path, json_body=body, headers=headers or None)
        print(json.dumps(dict(status=resp.status, sent=resp.sent, refused=resp.refused, reason=resp.reason, json=resp.json, text=resp.text[:500] if resp.json is None else ""), default=str, indent=2))
        return 0 if (resp.ok or a.dry_run) else 1
    return 2


# =========================================================================== architect
def _chat_engine(ws, verbose=True):
    path = os.path.join(os.path.abspath(ws), "chat")
    if not os.path.isdir(path):
        return None
    try:
        from . import chat_task
        from .chat.generate import Engine
        ct = chat_task.ChatTask(path)
        cfg = ct.config()
        eng = Engine(cfg, ct.layout(cfg))
        return eng if eng.load() else None
    except Exception as ex:                                       # noqa: BLE001 - no usable chat model means the built-in proposer carries on
        if verbose:
            print(f"[architect] the chat model could not be loaded ({type(ex).__name__}: {ex}); using the built-in proposer only", file=sys.stderr)
        return None


def _architect(a):
    from .forge.architect import Architect, ChatProposer
    eng = None if getattr(a, "no_chat", False) else _chat_engine(a.workspace)
    return Architect(a.workspace, seed=a.seed, chat=ChatProposer(eng, seed=a.seed) if eng is not None else None, allow_hooks=getattr(a, "hooks", False),
                     allow_patch=getattr(a, "patch_dry_run", False), repo_root=REPO_ROOT, hook_evals=getattr(a, "hook_evals", 60), verbose=True), eng


def cmd_architect(a):
    arch, eng = _architect(a)
    if a.action == "report":
        print(arch.report())
        return 0
    if a.action == "export":
        out = a.out or os.path.join(a.workspace, "chat", "data", "_architect")
        print(json.dumps(arch.export_for_chat(out)), "->", out)
        print("next: `zerowhite chat prepare` / `chat train` read these files like any others (they teach the chat model the proposals that survived testing)")
        return 0
    if a.action == "ask":
        from .forge import architect as R
        if eng is None:
            print("no trained chat model in this workspace: the architect uses its built-in proposer (a bandit, not a language model)")
            return 1
        led = arch.ledger()
        st = dict(task="demo", kind="regression", family="program_formula", skill=0.4, plateau=True, src="def predict(x):\n    return x[:, 0] * 0.5\n", genome=None, space=None)
        if led:
            last = led[-1]
            st.update(task=last["task"], kind=last.get("task_kind", "?"), family=last.get("family") or "?", skill=last.get("skill"), src=(last.get("state") or {}).get("src"), genome=None)
        cp = arch.chat
        hs = cp.propose(st, arch.arms, np.random.default_rng(a.seed), a.n, spaces={n: f.space for n, f in __import__("zerowhite.forge.families", fromlist=["FAMILIES"]).FAMILIES.items()})
        print("prompt:\n" + R.build_prompt(st, led[-4:], a.n) + "\n\nthe chat model said:\n" + (cp.last_reply or "(nothing)") + f"\n\nvalid hypotheses: {len(hs)}; lines that failed the grammar: {cp.invalid}")
        for h in hs:
            print("  " + h.text)
        return 0
    tasks = _task_from_args(a)                                    # run
    if not tasks:
        raise SystemExit("architect run needs a task: --csv FILE | --env NAME | --dataset NAME")
    from .forge.generalist import Generalist
    g = Generalist(a.workspace, seed=a.seed, architect=arch)
    for t in tasks:
        g.solve(t, budget=a.budget, n_new=a.candidates, rungs=a.rungs)
    print("\n" + arch.report())
    return 0


# =========================================================================== registration
def register(sub, common):
    t = sub.add_parser("tools", help="6.5: tool forging: write, test, keep and chain small scripts from input/output examples")
    t.add_argument("action", choices=["list", "show", "forge", "run", "chain", "rm", "request", "process"]); t.add_argument("name", nargs="*")
    t.add_argument("--workspace", default="zw"); t.add_argument("--examples", help="JSON file: a list of {\"input\": ..., \"output\": ...}")
    t.add_argument("--description", default=""); t.add_argument("--input", help="JSON value, or @FILE"); t.add_argument("--seconds", type=float, default=20.0)
    t.add_argument("--replace", action="store_true"); t.add_argument("--limit", type=int, default=3); t.add_argument("--json", action="store_true")
    t.set_defaults(fn=cmd_tools)

    m = sub.add_parser("memory", help="6.5: the episodic memory of tasks the system has met (a learned encoder over task descriptions)")
    m.add_argument("action", choices=["status", "fit", "recall"]); m.add_argument("--workspace", default="zw"); task_args(m)
    m.add_argument("--k", type=int, default=5); m.add_argument("--steps", type=int, default=300); common(m); m.set_defaults(fn=cmd_memory)

    s = sub.add_parser("stream", help="6.5: real-time multimodal streaming: chunk-exact tokenising of text / audio / video into one token stream")
    s.add_argument("action", choices=["demo", "check"]); s.add_argument("file", nargs="?")
    s.add_argument("--steps", type=int, default=20); s.add_argument("--trials", type=int, default=5); s.add_argument("--chunk", type=int, default=64); common(s); s.set_defaults(fn=cmd_stream)

    c = sub.add_parser("act", help="6.5: stateful web actuation under a written policy (CLI only; refuses everything unless you name a host)")
    c.add_argument("action", choices=["demo", "serve-mock", "request", "ledger", "kill", "unkill"]); c.add_argument("method", nargs="?"); c.add_argument("path", nargs="?")
    c.add_argument("--workspace", default="zw"); c.add_argument("--base-url"); c.add_argument("--allow-host", action="append", metavar="HOST")
    c.add_argument("--allow-remote", action="store_true", help="hosts other than loopback (then https, a public address and a ledger are required)")
    c.add_argument("--allow-delete", action="store_true"); c.add_argument("--dry-run", action="store_true", help="check and ledger the request, send nothing")
    c.add_argument("--json", help="request body as JSON or @FILE; {\"$secret\": NAME} placeholders are filled from --secret-env"); c.add_argument("--header", action="append", metavar="NAME=VALUE")
    c.add_argument("--secret-env", action="append", metavar="NAME=ENVVAR", help="a secret, read from this environment variable (never from the command line)")
    c.add_argument("--max-requests", type=int, default=50); c.add_argument("--tail", type=int, default=20); c.add_argument("--rounds", type=int, default=5)
    common(c); c.set_defaults(fn=cmd_act)

    r = sub.add_parser("architect", help="6.5: the chat model (or a built-in stand-in) proposes edits after a plateau; only those that survive a paired test are kept")
    r.add_argument("action", choices=["run", "report", "export", "ask"]); r.add_argument("--workspace", default="zw"); task_args(r)
    r.add_argument("--budget", type=float, default=1.0); r.add_argument("--candidates", type=int, default=8); r.add_argument("--rungs", type=int, default=3)
    r.add_argument("--no-chat", action="store_true", help="use only the built-in proposer"); r.add_argument("--hooks", action="store_true", help="also let it ask for a search-rule search (selfmod/hooks.py)")
    r.add_argument("--patch-dry-run", action="store_true", help="also let it ask for a DRY RUN of the source patcher (never applied)"); r.add_argument("--hook-evals", type=int, default=60)
    r.add_argument("--out", help="export: folder for the training text (default: <workspace>/chat/data/_architect)"); r.add_argument("-n", type=int, default=3)
    common(r); r.set_defaults(fn=cmd_architect)
