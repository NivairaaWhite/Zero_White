"""cli.py - `zerowhite chat ...`: every command of the chat stack.

    zerowhite chat init                      create a workspace (folders + a commented config.json)
    zerowhite chat status                    data / model / service at a glance
    zerowhite chat config [--explain|--reference|--json]
    zerowhite chat fetch --list | NAME...    download datasets into the data folder
    zerowhite chat ingest | prepare          read the data folder / also tokenise and plan the training run
    zerowhite chat convert SRC DST           convert a dataset between formats (jsonl/json/csv/txt/chatml/alpaca/sharegpt ...)
    zerowhite chat train                     real training (resumable, validated, promoted only if better)
    zerowhite chat talk | say TEXT           talk to the model (uses the background service when it is running)
    zerowhite chat start | stop | restart    the background service (web UI + API + auto-ingest + auto-train + online learning)
    zerowhite chat serve                     run the service in this terminal
    zerowhite chat feedback | learn | rollback | versions    online learning
    zerowhite chat export                    convert the model (npz / int8 / safetensors)
    zerowhite chat autostart show|install|remove
    zerowhite chat voice ...                 speech in / speech out
    zerowhite chat logs [train|daemon|supervisor]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

from .. import fileio
from . import hardware as hw
from .cli_common import add_common, fail, human_bytes, load_cfg, load_layout
from .config import Config, reference_markdown
from .paths import Layout


# ---------------------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------------------
def _note(msg: str):
    print(msg, flush=True)


def _need_workspace_data(lay: Layout):
    from .corpus.store import Ingestor
    if not Ingestor(lay.cfg, lay).shards():
        from .prepare import no_data_message
        fail(no_data_message(lay), 2)


def _client(cfg: Config, lay: Layout):
    from .service.client import Client
    return Client.discover(cfg, lay)


def _local_engine(cfg: Config, lay: Layout):
    from .generate import Engine
    from .prepare import resolve_light
    eng = Engine(cfg, lay)
    if not eng.load():
        fail("There is no trained model yet. Put data in the data folder and run `zerowhite chat train`:\n  data folder: " + lay.data, 2)
    eng.cfg = resolve_light(cfg, lay)
    return eng


# ---------------------------------------------------------------------------------------------------------------------
# init / config / status
# ---------------------------------------------------------------------------------------------------------------------
def cmd_init(a):
    cfg, lay = load_layout(a)
    path = os.path.join(cfg.workspace, "config.json")
    if not os.path.exists(path):
        fileio.write_json(path, {"_comment": "Only list what you want to change. Every option is documented in docs/CONFIG.md "
                                             "(`zerowhite chat config --reference`).",
                                 "persona": {"name": cfg.get("persona.name")}}, indent=2)
    print(f"workspace ready: {cfg.workspace}\n"
          f"  data folder  : {lay.data}   <- put your datasets here (any sub-folders; json, jsonl, csv, parquet, txt, md, py ... .gz/.zip ok)\n"
          f"  config file  : {path}\n"
          f"Next:  zerowhite chat train      (or `zerowhite chat start` to let the background service do it for you)")


def cmd_config(a):
    if a.reference:
        sys.stdout.write(reference_markdown())
        return
    cfg = load_cfg(a)
    if a.json:
        print(json.dumps(cfg.to_dict(), indent=2, default=str))
        return
    if a.write:
        cfg.dump(a.write, only_changed=not a.full)
        print(f"wrote {a.write}")
        return
    from .prepare import resolve_light
    lay = Layout(cfg)
    try:
        cfg = resolve_light(cfg, lay)
    except Exception:                                            # noqa: BLE001
        pass
    print(cfg.explain() if a.explain else "\n".join(f"{k} = {json.dumps(v, default=str)}" for k, v in sorted(cfg.changed().items()))
          or "(no settings changed from the defaults; use --explain to list every option)")


def cmd_status(a):
    cfg, lay = load_layout(a)
    from .corpus.store import Ingestor
    print(f"workspace : {cfg.workspace}")
    print(f"data      : {lay.data}")
    try:
        st = Ingestor(cfg, lay).stats()
        files = sum(1 for _ in _walk(lay.data))
        kinds = ", ".join(f"{k} {v:,}" for k, v in sorted(st["kinds"].items()))
        print(f"            {files} file(s) in the folder, {st['files']} ingested, {st['records']:,} usable records" + (f" ({kinds})" if kinds else ""))
        if st["errors"]:
            print(f"            {len(st['errors'])} file(s) could not be read: " + "; ".join(f"{k}: {v}" for k, v in list(st['errors'].items())[:3]))
    except Exception as e:                                       # noqa: BLE001
        print(f"            (no ingest information yet: {e})")
    cur = lay.current()
    if cur:
        print(f"model     : {cur['file']}   step {cur.get('step')}   val loss {cur.get('val_nll')}   mode {cur.get('mode')}   online updates {cur.get('online_updates', 0)}")
    else:
        print("model     : none yet (run `zerowhite chat train`)")
    if os.path.exists(lay.train_status_path):
        try:
            ts = fileio.read_json(lay.train_status_path)
            if hw.pid_alive(int(ts.get("pid", 0))):
                print("training  : " + ", ".join(f"{k}={v}" for k, v in ts.items() if k not in ("updated", "pid")))
        except (OSError, ValueError):
            pass
    from .online import ExperienceLog
    ex = ExperienceLog(lay).stats()
    print(f"feedback  : {ex.get('replies', 0)} replies logged, {ex.get('feedback', 0)} feedback events" if ex else "feedback  : none")
    from .service import daemon
    d = daemon.running(lay)
    if d:
        print(f"service   : running (pid {d['pid']}) at {d.get('url')}   token file: {lay.token_file}")
    else:
        print("service   : not running (`zerowhite chat start`)")


def _walk(root: str):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if not d.startswith(".")]
        for f in fn:
            if not f.startswith(".") and not f.upper().startswith("README") and f != "dataset.json":
                yield os.path.join(dp, f)


# ---------------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------------
def cmd_fetch(a):
    from .corpus import fetch as F
    cfg, lay = load_layout(a)
    try:
        cat = F.load_catalog(cfg)
        if a.list or not a.names:
            print(f"{'name':<24} {'repo':<48} {'kind':<8} notes")
            for k, v in sorted(cat.items()):
                flag = "" if v.get("usable", True) else " [not fetchable here]"
                print(f"{k:<24} {v.get('repo', ''):<48} {v.get('kind', 'chat'):<8} {v.get('note', '')}{flag}")
            print("\nDownload:  zerowhite chat fetch NAME [--limit N]     (also accepts any owner/dataset id)")
            return
        for name in a.names:
            def show(ev, name=name):
                if ev.get("event") == "page":
                    print(f"\r[fetch] {name}: {ev['offset']:,}/{ev['limit']:,} rows", end="", flush=True)
                elif ev.get("event") in ("note", "resume"):
                    print(f"\n[fetch] {ev.get('message') or 'resuming at row ' + str(ev.get('offset'))}", flush=True)
            path = F.fetch_dataset(cfg, lay, name, limit=a.limit, config=a.config, split=a.split, repo=a.repo, progress=show)
            print(f"\n[fetch] {name} -> {path}")
    except F.FetchError as e:
        fail(f"fetch failed: {e}")
    print("Done. The service ingests new files automatically; otherwise run `zerowhite chat train`.")


def cmd_ingest(a):
    from .corpus.store import Ingestor
    cfg, lay = load_layout(a)
    ing = Ingestor(cfg, lay)
    rep = ing.ingest(progress=lambda ev: _note(_fmt_ev(ev)) if isinstance(ev, dict) else None, force=a.force)
    print(rep.summary())
    for rel, err in rep.errors:
        print(f"  could not use {rel}: {err}")
    st = ing.stats()
    print(f"{st['records']:,} usable records from {st['files']} file(s): " + ", ".join(f"{k} {v:,}" for k, v in sorted(st["kinds"].items())))
    if a.verbose:
        for rel, s in sorted(st["by_source"].items()):
            print(f"  {rel}: {s['records']:,}/{s['read']:,} records via adapter '{s['adapter']}'")
        if st["dropped"]:
            print("dropped by filters: " + ", ".join(f"{k}={v:,}" for k, v in sorted(st["dropped"].items())))
    if not ing.shards():
        from .prepare import no_data_message
        fail(no_data_message(lay, rep), 2)


def _fmt_ev(ev: dict) -> str:
    from .prepare import _fmt_ingest_event
    return _fmt_ingest_event(ev)


def _apply_device_flags(a, cfg):
    if getattr(a, "gpu", False):
        cfg = cfg.replace(**{"compute.backend": "gpu"})
    elif getattr(a, "cpu", False):
        cfg = cfg.replace(**{"compute.backend": "cpu"})
    return cfg


def cmd_prepare(a):
    from .backend import BackendError
    from .prepare import NoDataError, prepare
    cfg, lay = load_layout(a)
    cfg = _apply_device_flags(a, cfg)
    try:
        P = prepare(cfg, lay, progress=_note, do_bench=not a.no_bench, force_ingest=a.force)
    except NoDataError as e:
        fail(str(e), 2)
    except BackendError as e:
        fail(str(e), 5)
    cnt = P.corpus.counts()
    print(f"corpus    : {cnt['train_docs']:,} training / {cnt['val_docs']:,} validation conversations, {cnt['train_tokens']:,} / {cnt['val_tokens']:,} tokens")
    print(f"tokenizer : {P.tok.vocab_size} tokens   context window: {P.plan.model.ctx}")
    print(f"compute   : {P.backend.label()}" + (f"  ({P.backend.note})" if P.backend.note else ""))
    info = P.plan.info
    print(f"model plan: {info.get('params', 0):,} parameters ({P.plan.model.d_model}d x {P.plan.model.n_layer}L), {info.get('planned_steps')} steps, "
          f"~{info.get('tokens_per_s', 0):,.0f} tokens/s")
    for k in ("notice", "warning"):
        if info.get(k):
            print(f"{k}: {info[k]}")
    print(f"details   : {lay.resolved_config_path}")


def cmd_convert(a):
    from .corpus import convert as C
    cfg = load_cfg(a)
    try:
        r = C.convert(a.src, a.dst, a.to, cfg, a.adapter, a.limit, apply_filters=not a.no_filters, progress=lambda n: print(f"\r{n:,} records", end="", flush=True),
                      overwrite=a.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        fail(str(e), 2)
    print(f"\nwrote {r['written']:,} records to {r['output']} (format: {r['target']}); read {r['read']:,}"
          + (f", dropped by filters {sum(r['dropped'].values()):,}" if r["dropped"] else "")
          + (f", skipped {r['skipped_wrong_kind']:,} of a kind this format cannot hold" if r["skipped_wrong_kind"] else ""))
    for rel, err in r["errors"]:
        print(f"  could not read {rel}: {err}")


# ---------------------------------------------------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------------------------------------------------
def cmd_train(a):
    from .backend import BackendError
    from .prepare import NoDataError, prepare
    from .train import GpuOutOfMemory, Trainer, TrainingBusy, TrainingDiverged, console_progress
    cfg, lay = load_layout(a)
    cfg = _apply_device_flags(a, cfg)
    if a.minutes is not None:
        cfg = cfg.replace(**{"train.budget_minutes": float(a.minutes)})
    if a.fresh:
        cfg = cfg.replace(**{"train.resume": False})
    if a.background:
        if cfg.get("daemon.low_priority_training"):
            hw.lower_priority()
        if cfg.get("daemon.train_budget_minutes") is not None and a.minutes is None:
            cfg = cfg.replace(**{"train.budget_minutes": float(cfg.get("daemon.train_budget_minutes"))})
    quiet = a.background
    log = (lambda m: None) if quiet else _note
    try:
        P = prepare(cfg, lay, progress=log, do_bench=not a.no_bench, force_ingest=a.force)
    except NoDataError as e:
        fail(str(e), 2)
    except BackendError as e:
        fail(str(e), 5)
    if a.plan:
        info = P.plan.info
        print(f"plan: {info.get('params', 0):,} parameters, {info.get('planned_steps')} steps; details in {lay.resolved_config_path}")
        return 0
    for k in ("notice", "warning"):
        if P.plan.info.get(k):
            _note(f"{k}: {P.plan.info[k]}")
    try:
        res = Trainer(P, progress=None if quiet else console_progress, background=a.background, wait_s=a.wait).run()
    except TrainingBusy as e:
        fail(str(e), 3)
    except TrainingDiverged as e:
        fail(f"training diverged: {e}", 4)
    except GpuOutOfMemory as e:
        fail(str(e), 6)
    if not quiet:
        if res.get("promoted"):
            print("The new model is live. Talk to it:  zerowhite chat talk     (or `zerowhite chat start` for the web UI)")
        else:
            print("The live model was left unchanged.")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# talking
# ---------------------------------------------------------------------------------------------------------------------
class _Remote:
    def __init__(self, client, session: Optional[str]):
        self.c, self.session, self.last_id = client, session, None
        self.where = f"service at {client.url}"

    def send(self, text: str, **gen):
        for ev in self.c.chat(text, self.session, **gen):
            if ev.get("done"):
                self.session, self.last_id = ev.get("session", self.session), ev.get("reply_id")
            yield ev

    def regenerate(self, **gen):
        for ev in self.c.regenerate(self.session, **gen):
            if ev.get("done"):
                self.last_id = ev.get("reply_id")
            yield ev

    def feedback(self, rating: int = 0, correction: Optional[str] = None):
        return self.c.feedback(self.last_id, rating, correction)

    def new(self):
        self.session, self.last_id = None, None


class _Local:
    def __init__(self, cfg: Config, lay: Layout, session: Optional[str], system: Optional[str]):
        from .session import ChatService
        self.eng = _local_engine(cfg, lay)
        self.svc = ChatService(self.eng.cfg, lay, self.eng)
        self.system, self.last_id = system, None
        self.session = self.svc.get(session, system).id
        self.where = f"local model {self.eng.version}"

    def send(self, text: str, **gen):
        for ev in self.svc.send(self.session, text, **gen):
            if ev.get("done"):
                self.last_id = ev.get("reply_id")
            yield ev

    def regenerate(self, **gen):
        for ev in self.svc.regenerate(self.session, **gen):
            if ev.get("done"):
                self.last_id = ev.get("reply_id")
            yield ev

    def feedback(self, rating: int = 0, correction: Optional[str] = None):
        return self.svc.feedback(self.last_id, rating, correction)

    def new(self):
        self.session, self.last_id = self.svc.reset(self.session).id, None


def _open_chat(a, cfg: Config, lay: Layout):
    if not getattr(a, "local", False):
        c = _client(cfg, lay)
        if c is not None:
            return _Remote(c, getattr(a, "session", None))
    return _Local(cfg, lay, getattr(a, "session", None), getattr(a, "system", None))


def _gen_opts(a) -> dict:
    return {k: v for k, v in dict(temperature=getattr(a, "temperature", None), max_new_tokens=getattr(a, "max_tokens", None)).items() if v is not None}


def _print_reply(events) -> str:
    text = ""
    for ev in events:
        if "delta" in ev:
            print(ev["delta"], end="", flush=True)
        elif ev.get("done"):
            text = ev.get("text", "")
    print()
    return text


HELP = """commands:  /good  /bad  /fix <better answer>  /again (regenerate)  /new  /learn  /help  /quit
  /good and /fix teach the model (online learning); /bad alone changes nothing; /again + /good teaches "this answer beat that one"."""


def cmd_gpu(a):
    """Check that training on the GPU works here, and how much faster it is."""
    from . import backend as bk
    from . import gpucheck
    try:
        be = bk.get("gpu", int(a.device))
    except bk.BackendError as e:
        print(gpucheck.format_report(gpucheck.run(bk.Backend(note=str(e)))))
        return 1
    report = gpucheck.run(be, speed=not a.no_speed, say=_note)
    print(gpucheck.format_report(report))
    return 0 if report["ok"] else 1


def cmd_grow(a):
    """Add layers / FFN units / attention heads to the live model without changing what it computes; the next `chat train` continues from it."""
    from . import grow as G
    cfg, lay = load_layout(a)
    if not (a.layers or a.ffn or a.heads or a.experts):
        fail("say what to add: --layers N, --ffn N (hidden units per layer), --heads N and/or --experts N")
    try:
        r = G.grow_current(lay, layers=a.layers, ffn=a.ffn, heads=a.heads, at=a.at, init=a.init, seed=a.seed, experts=a.experts, top_k=a.top_k)
    except (RuntimeError, ValueError) as ex:
        fail(str(ex))
    print("grown: " + r["summary"])
    print(f"new live version: {os.path.basename(r['file'])}  (same function as before; `zerowhite chat rollback --to FILE` returns to the old one)")
    print("next: `zerowhite chat train` continues from it. A running service keeps serving the old model until `zerowhite chat restart`.")


def cmd_talk(a):
    cfg, lay = load_layout(a)
    if getattr(a, "voice", False):
        try:
            from ..voice.talk import run_talk
        except ImportError as e:
            fail(f"voice is not available: {e}")
        return run_talk(cfg, lay, a)
    chat = _open_chat(a, cfg, lay)
    name = cfg.get("persona.name") or "Zero White"
    greeting = cfg.get("persona.greeting")
    print(f"{name} ({chat.where}).  Type /help for commands.")
    if greeting:
        print(f"{name}: {greeting}")
    opts = _gen_opts(a)
    while True:
        try:
            line = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, rest = line[1:].partition(" ")
            cmd = cmd.lower()
            try:
                if cmd in ("quit", "exit", "q"):
                    break
                elif cmd == "help":
                    print(HELP)
                elif cmd == "new":
                    chat.new()
                    print("(new conversation)")
                elif cmd == "again":
                    print(f"{name}: ", end="", flush=True)
                    _print_reply(chat.regenerate(**opts))
                elif cmd == "good":
                    chat.feedback(1)
                    print("(noted: thumbs up)")
                elif cmd == "bad":
                    chat.feedback(-1)
                    print("(noted: thumbs down; nothing is learned from this alone, give a better answer with /fix)")
                elif cmd == "fix":
                    if not rest.strip():
                        print("usage: /fix <what the answer should have been>")
                    else:
                        chat.feedback(0, rest.strip())
                        print("(noted: your correction will be learned)")
                elif cmd == "learn":
                    r = _learn_now(cfg, lay, force=True)
                    print("(" + r + ")")
                else:
                    print("unknown command; /help lists them")
            except Exception as e:                               # noqa: BLE001
                print(f"({e})")
            continue
        print(f"{name}: ", end="", flush=True)
        try:
            _print_reply(chat.send(line, **opts))
        except Exception as e:                                   # noqa: BLE001
            print(f"\n({e})")


def cmd_say(a):
    cfg, lay = load_layout(a)
    chat = _open_chat(a, cfg, lay)
    text = " ".join(a.text) if a.text else sys.stdin.read()
    if not text.strip():
        fail("say what? e.g.  zerowhite chat say \"hello\"")
    out = _print_reply(chat.send(text.strip(), **_gen_opts(a)))
    if a.show_id and chat.last_id:
        print(f"[reply id: {chat.last_id}  session: {chat.session}]", file=sys.stderr)
    return 0 if out is not None else 1


# ---------------------------------------------------------------------------------------------------------------------
# online learning
# ---------------------------------------------------------------------------------------------------------------------
def _learn_now(cfg: Config, lay: Layout, force: bool) -> str:
    c = _client(cfg, lay)
    if c is not None:
        r = c.learn()
    else:
        from .online import OnlineLearner
        from .prepare import load_corpus, resolve_light
        eng = _local_engine(cfg, lay)
        cfg2 = resolve_light(cfg, lay)
        corpus = load_corpus(cfg2, lay, eng.tok, eng.model.cfg.ctx)
        r = OnlineLearner(cfg2, lay, eng, corpus).update(force=force)
    if r.get("updated"):
        return f"updated the model: {r.get('summary') or json.dumps({k: v for k, v in r.items() if k != 'updated'}, default=str)[:300]}"
    return f"no update: {r.get('reason') or r}"


def cmd_learn(a):
    cfg, lay = load_layout(a)
    print(_learn_now(cfg, lay, force=a.force))


def cmd_feedback(a):
    cfg, lay = load_layout(a)
    rating = 1 if a.up else -1 if a.down else 0
    if not rating and not a.fix:
        fail("give --up, --down or --fix \"better answer\"")
    c = _client(cfg, lay)
    if c is not None:
        c.feedback(a.reply_id, rating, a.fix)
    else:
        from .online import ExperienceLog
        ExperienceLog(lay).log_feedback(a.reply_id, rating, a.fix)
    print("feedback recorded" + ("" if c else " (it is learned the next time the service or `chat learn` runs)"))


def cmd_rollback(a):
    cfg, lay = load_layout(a)
    c = _client(cfg, lay)
    if c is not None:
        r = c.rollback(a.to)
        now = r.get("version") or r.get("now")
    else:
        from .online import rollback
        now = rollback(lay, a.to)
    if not now:
        fail("there is nothing to roll back to (the live model has no parent version); see `zerowhite chat versions`")
    print(f"the live model is now {now}")


def cmd_versions(a):
    cfg, lay = load_layout(a)
    live = (lay.current() or {}).get("file")
    rows = lay.versions(a.kind)
    if not rows:
        print("no model versions yet")
        return
    for p, kind, mt in rows[: a.n]:
        print(f"{'*' if os.path.basename(p) == live else ' '} {os.path.basename(p):<46} {kind:<7} {time.strftime('%Y-%m-%d %H:%M', time.localtime(mt))}  {human_bytes(os.path.getsize(p))}")
    print("* = live. Switch with:  zerowhite chat rollback --to FILE")


# ---------------------------------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------------------------------
def cmd_export(a):
    from .export import export_model
    cfg, lay = load_layout(a)
    try:
        out = export_model(lay, a.format, a.out, a.version, cfg)
    except (FileNotFoundError, ValueError) as e:
        fail(str(e), 2)
    files = sorted(os.listdir(out))
    print(f"exported ({a.format}) to {out}\n  " + "\n  ".join(f"{f}  {human_bytes(os.path.getsize(os.path.join(out, f)))}" for f in files))
    if a.format == "safetensors":
        print("Layout: Llama-style tensors + config.json. The tokenizer is Zero White's own byte-level BPE (tokenizer.json), "
              "so other runtimes need that tokenizer ported; the weights themselves follow the standard names.")


# ---------------------------------------------------------------------------------------------------------------------
# service
# ---------------------------------------------------------------------------------------------------------------------
def cmd_serve(a):
    from .service import daemon
    cfg, lay = load_layout(a)
    if a.supervise:
        return daemon.supervise(cfg, lay)
    return daemon.Daemon(cfg, lay, console=not a.worker).run()


def cmd_start(a):
    from .service import daemon
    cfg, lay = load_layout(a)
    r = daemon.start(cfg, lay, a.wait)
    if r.get("error"):
        fail(r["error"])
    print(("already running" if r.get("already") else "started") + f" (pid {r['pid']}): {r.get('url')}\n"
          f"Open the URL in a browser, or use `zerowhite chat talk`. Stop with `zerowhite chat stop`.\n"
          f"Access token (for API clients): {lay.token_file}")


def cmd_stop(a):
    from .service import daemon
    cfg, lay = load_layout(a)
    r = daemon.stop(cfg, lay, a.wait)
    print(r["message"])
    return 0 if r.get("stopped") or "not running" in r["message"] else 1


def cmd_restart(a):
    cmd_stop(a)
    cmd_start(a)


def cmd_logs(a):
    cfg, lay = load_layout(a)
    name = {"train": "train.jsonl", "daemon": "daemon.log", "supervisor": "supervisor.log", "run": "train_run.log", "out": "daemon.out"}[a.which]
    p = lay.log_path(name)
    if not os.path.exists(p):
        fail(f"no {name} yet in {lay.logs}")
    with open(p, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 64 * 1024))
        lines = f.read().decode("utf-8", "replace").splitlines()[-a.n:]
    print("\n".join(lines))


def cmd_autostart(a):
    from .service import autostart
    cfg, lay = load_layout(a)
    print({"show": autostart.show, "install": autostart.install, "remove": autostart.remove}[a.action](cfg, lay))


# ---------------------------------------------------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------------------------------------------------
def register(sub, _common=None):
    chat = sub.add_parser("chat", help="the conversational model: data folder, real training, online learning, voice, background service",
                          description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cs = chat.add_subparsers(dest="chat_cmd", required=True, metavar="COMMAND")

    def mk(name, fn, help, **kw):
        p = cs.add_parser(name, help=help, **kw)
        add_common(p)
        p.set_defaults(fn=fn)
        return p

    mk("init", cmd_init, "create a workspace with a data folder and a starter config")

    p = mk("config", cmd_config, "show the settings in effect")
    p.add_argument("--explain", action="store_true", help="every option with its value, where it came from, and derived values")
    p.add_argument("--reference", action="store_true", help="print the full option reference (markdown)")
    p.add_argument("--json", action="store_true", help="print the effective configuration as JSON")
    p.add_argument("--write", metavar="FILE", help="write the effective configuration to FILE")
    p.add_argument("--full", action="store_true", help="with --write: include defaults too")

    mk("status", cmd_status, "data, model, training and service at a glance")

    p = mk("fetch", cmd_fetch, "download datasets from Hugging Face into the data folder")
    p.add_argument("names", nargs="*", help="catalog name or owner/dataset (omit or --list to see the catalog)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--limit", type=int, help="rows to download (default: fetch.default_limit)")
    p.add_argument("--subset", dest="hf_config", metavar="NAME", help="dataset config / subset name on the hub")
    p.add_argument("--split", help="split (default: train)")
    p.add_argument("--repo", help="override the Hugging Face repo id")

    p = mk("ingest", cmd_ingest, "read new / changed files from the data folder")
    p.add_argument("--force", action="store_true", help="re-read everything")
    p.add_argument("-v", "--verbose", action="store_true")

    p = mk("prepare", cmd_prepare, "ingest, tokenise and print the training plan (changes nothing else)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-bench", action="store_true", help="skip the throughput benchmark (plan from the last measurement)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gpu", action="store_true", help="train on the GPU (compute.backend=gpu); stops with the fix if it cannot")
    g.add_argument("--cpu", action="store_true", help="train on the CPU even if a GPU is available (compute.backend=cpu)")

    p = cs.add_parser("convert", help="convert a dataset file/folder between formats")
    add_common(p)
    p.add_argument("src")
    p.add_argument("dst", help="output file; the extension picks the container (.jsonl .json .csv .txt, optional .gz)")
    p.add_argument("--to", help="record format: normalized | chatml | sharegpt | alpaca | openai | plain | csv ...")
    p.add_argument("--adapter", help="force a dataset adapter instead of auto-detecting")
    p.add_argument("--limit", type=int)
    p.add_argument("--no-filters", action="store_true", help="convert without quality/safety filtering")
    p.add_argument("--overwrite", action="store_true", help="replace the output file if it already exists")
    p.set_defaults(fn=cmd_convert)

    p = mk("train", cmd_train, "train the model on the data folder (resumes, validates, promotes only if better)")
    p.add_argument("--minutes", type=float, help="wall-clock budget for this run (default: train.budget_minutes)")
    p.add_argument("--fresh", action="store_true", help="start a new model instead of continuing the live one")
    p.add_argument("--force", action="store_true", help="re-ingest everything first")
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--plan", action="store_true", help="only print the plan")
    p.add_argument("--background", action="store_true", help="quiet, low priority (used by the service)")
    p.add_argument("--wait", type=float, default=0.0, help="wait up to this many seconds for another training run to finish")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gpu", action="store_true", help="train on the GPU (compute.backend=gpu); stops with the fix if it cannot")
    g.add_argument("--cpu", action="store_true", help="train on the CPU even if a GPU is available (compute.backend=cpu)")

    p = mk("grow", cmd_grow, "make the live model bigger (layers / FFN units / attention heads) without changing what it computes, then keep training")
    p.add_argument("--layers", type=int, default=0, help="residual blocks to add (default position: the top)")
    p.add_argument("--ffn", type=int, default=0, help="hidden units to add to every block's feed-forward layer")
    p.add_argument("--heads", type=int, default=0, help="attention heads to add (widens d_model by head_dim per head)")
    p.add_argument("--experts", type=int, default=0, help="6.4: turn every feed-forward block into a mixture of this many experts (same outputs at first), top-k routed")
    p.add_argument("--top-k", type=int, default=2, help="with --experts: experts used per token")
    p.add_argument("--at", type=int, help="with --layers: insert before this block index (0 = the bottom; default: the top)")
    p.add_argument("--init", choices=["copy", "random"], default="copy", help="new blocks start as a copy of the block below (default) or random")
    p.add_argument("--seed", type=int, default=0)

    p = mk("gpu", cmd_gpu, "check that training on the GPU works here (kernels, model and optimiser parity with the CPU) and measure the speed-up")
    p.add_argument("--device", type=int, default=0, help="which GPU (default 0)")
    p.add_argument("--no-speed", action="store_true", help="skip the throughput measurement")

    for name, fn, h in (("talk", cmd_talk, "chat in the terminal"), ("say", cmd_say, "send one message and print the reply")):
        p = mk(name, fn, h)
        p.add_argument("--session", help="conversation id (continue an earlier one)")
        p.add_argument("--system", help="system prompt for a new conversation")
        p.add_argument("--local", action="store_true", help="use the model directly even if the service is running")
        p.add_argument("--temperature", type=float)
        p.add_argument("--max-tokens", type=int, dest="max_tokens")
        if name == "talk":
            p.add_argument("--voice", action="store_true", help="speak and listen (see `zerowhite chat voice`)")
            p.add_argument("--no-speak", action="store_true", help="with --voice: listen but do not speak the answers")
        else:
            p.add_argument("text", nargs="*")
            p.add_argument("--show-id", action="store_true", help="print the reply id (for `chat feedback`) on stderr")

    p = mk("serve", cmd_serve, "run the service in this terminal (Ctrl-C stops it)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--worker", action="store_true", help="(internal) quiet worker process")
    g.add_argument("--supervise", action="store_true", help="(internal) keep a worker alive")

    p = mk("start", cmd_start, "start the background service")
    p.add_argument("--wait", type=float, default=30.0)
    p = mk("stop", cmd_stop, "stop the background service (a running training job checkpoints first)")
    p.add_argument("--wait", type=float, default=60.0)
    p = mk("restart", cmd_restart, "stop and start the background service")
    p.add_argument("--wait", type=float, default=60.0)

    p = mk("learn", cmd_learn, "apply pending feedback to the model now")
    p.add_argument("--force", action="store_true", help="update even if there is less feedback than online.min_examples")

    p = mk("feedback", cmd_feedback, "rate or correct a reply")
    p.add_argument("reply_id")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--up", action="store_true")
    g.add_argument("--down", action="store_true")
    p.add_argument("--fix", metavar="TEXT", help="what the reply should have said")

    p = mk("rollback", cmd_rollback, "go back to the previous model version")
    p.add_argument("--to", metavar="FILE", help="a specific file from `chat versions`")
    p = mk("versions", cmd_versions, "list saved model versions")
    p.add_argument("--kind", choices=["train", "online", "best"])
    p.add_argument("-n", type=int, default=20)

    p = mk("export", cmd_export, "convert the model for other tools")
    p.add_argument("--format", choices=["npz", "int8", "safetensors"], default="safetensors")
    p.add_argument("--out", help="output folder (default: <workspace>/exports/...)")
    p.add_argument("--version", help="export a specific saved version instead of the live model")

    p = mk("logs", cmd_logs, "show the end of a log file")
    p.add_argument("which", nargs="?", default="daemon", choices=["train", "daemon", "supervisor", "run", "out"])
    p.add_argument("-n", type=int, default=40)

    p = mk("autostart", cmd_autostart, "start the service automatically when you log in")
    p.add_argument("action", choices=["show", "install", "remove"])

    try:
        from ..voice.cli import register as register_voice
        register_voice(cs, add_common)
    except ImportError:
        pass
    return chat
