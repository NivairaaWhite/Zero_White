"""
factory.py - the autonomous loop: Zero White improving itself, over and over,
across sessions.

A Workspace is a folder holding every model Zero White has built:

  classifier/<dataset>/weights.npz    the self-modifying classifier (checkpoint.py)
  generators/<dataset>/cvae.npz       class-conditional image generator + samples.png
  lm/<name>/lm.npz                    character language model
  agents/<env>/agent.npz              actor-critic agent
  <each>/best.json                    genome, score and history of the incumbent
  journal.jsonl                       one line per job, forever

Every job loads the INCUMBENT (if there is one), re-scores it with the current
evaluation, lets evolution propose better children (children inherit trained
weights when only training genes changed), and writes back whichever is best.
Incumbents are always in the pool, so a model never gets worse; run the same
command tomorrow, or with new text, or with a fresh classifier, and it carries on
from where it stopped instead of starting over. That persistence is the
"self-updating" part. Everything is written atomically (fileio.py), so Ctrl-C at
any moment is safe.
"""

import json
import os
import time

import numpy as np

from . import checkpoint
from . import datasets
from . import fileio
from .agents import ENV_BUDGET, PGAgent, evolve_agent
from .generative import (CVAE, CharLM, CharVocab, evolve_generator, evolve_lm,
                        split_text_tokens)
from .zero_white import ZeroWhite

MAX_SELECTION_VAL = 2000          # selection never needs more than this many validation rows


class Workspace:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def dir(self, *parts):
        d = os.path.join(self.root, *parts)
        os.makedirs(d, exist_ok=True)
        return d

    def best(self, *parts):
        p = os.path.join(self.root, *parts, "best.json")
        return fileio.read_json(p) if os.path.exists(p) else None

    def save_best(self, parts, record):
        fileio.write_json(os.path.join(self.dir(*parts), "best.json"), record, indent=2)

    def log(self, event):
        event = dict(event, t=time.time())
        line = json.dumps(event, default=fileio._json_default) + "\n"
        with open(os.path.join(self.root, "journal.jsonl"), "a", encoding="utf-8") as f:
            f.write(line)

    def journal(self):
        p = os.path.join(self.root, "journal.jsonl")
        if not os.path.exists(p):
            return []
        out = []
        for line in fileio.read_text(p).splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass                                   # a torn last line from a hard kill; skip it
        return out


# --------------------------------------------------------------------------- classifier
def make_classifier(task, seed=0, hidden=None, candidates=5):
    """Create a fresh self-modifying classifier. With hidden=None the starting width is drawn at
    random (there is no default architecture); self-modification then grows/prunes it."""
    if hidden is None:
        hidden = int(np.random.default_rng(seed).integers(6, 25))
    big = len(task.Xtr) > 500
    return ZeroWhite(task.in_dim, task.n_classes, hidden=(hidden,), num_chunks=task.num_chunks, seed=seed,
                     candidates_per_gen=candidates, batch_size=64 if big else None,
                     consolidate_steps=40 if big else 0, patch_dims=(8, 24))


def improve_classifier(ws, task, generations=20, probation=20, seed=0, verbose=True, fresh=False):
    """Continue (or start) the self-modifying classifier for `task`. Selection sees val only;
    test accuracy is reported, never used to choose anything."""
    d = ws.dir("classifier", task.name)
    resumed = checkpoint.exists(d) and not fresh
    if resumed:
        zw, _ = checkpoint.load(d)
        if zw.net.in_dim != task.in_dim or zw.net.out_dim != task.n_classes:
            raise ValueError(f"checkpoint in {d} is for {zw.net.in_dim}->{zw.net.out_dim}, "
                             f"task '{task.name}' is {task.in_dim}->{task.n_classes}; use --fresh")
    else:
        zw = make_classifier(task, seed=seed)
    Xv, yv = task.Xval[:MAX_SELECTION_VAL], task.yval[:MAX_SELECTION_VAL]
    before = zw.evaluate(zw.net, task.Xte, task.yte)[1] if resumed else None
    zw.run(task.Xtr, task.ytr, Xv, yv, generations=generations, probation_steps=probation, verbose=verbose)
    test_loss, test_acc = zw.evaluate(zw.net, task.Xte, task.yte)
    majority = float(np.bincount(task.yte).max() / len(task.yte))
    checkpoint.save(zw, d, extra=dict(dataset=task.name))
    zw.self_model.save(os.path.join(d, "run_journal.json"))
    rec = dict(kind="classifier", name=task.name, test_acc=test_acc, test_loss=test_loss, majority_baseline=majority,
               test_acc_before=before, generations=zw.self_model.generation, hidden=zw.net.hidden_sizes(),
               patch=zw.net.use_patch, params=zw.net.num_params(), source=task.source)
    ws.save_best(("classifier", task.name), rec)
    ws.log(rec)
    return zw, rec


def load_classifier(ws, name):
    d = os.path.join(ws.root, "classifier", name)
    return checkpoint.load(d)[0] if checkpoint.exists(d) else None


# --------------------------------------------------------------------------- generators
def improve_generator(ws, task, zw, generations=3, population=4, steps=500, seed=0, verbose=True):
    if not task.is_image:
        raise ValueError(f"'{task.name}' is not an image dataset; the CVAE generator needs images")
    parts = ("generators", task.name)
    path = os.path.join(ws.dir(*parts), "cvae.npz")
    incumbent = CVAE.load(path) if os.path.exists(path) else None
    if incumbent is not None and incumbent.cfg["in_dim"] != task.in_dim:
        incumbent = None
    ev = evolve_generator(task, zw.predict, generations, population, steps, seed, incumbent, verbose)
    best = ev.best
    best.artifact.save(path)
    h, w = task.image_shape
    y = np.tile(np.arange(task.n_classes), 6)
    fileio.write_png(os.path.join(ws.dir(*parts), "samples.png"),
                     fileio.image_grid(best.artifact.sample(y, np.random.default_rng(seed)).reshape(-1, h, w),
                                       cols=task.n_classes, scale=max(1, 112 // max(h, w))))
    rec = dict(kind="generator", name=task.name, fitness=best.fitness, genome=best.genome, **best.info,
               improved=ev.improved)
    ws.save_best(parts, rec)
    ws.log(rec)
    return best.artifact, rec


# --------------------------------------------------------------------------- language model
def read_corpus(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def improve_lm(ws, name, text, generations=3, population=4, steps=600, seed=0, verbose=True, fresh=False):
    parts = ("lm", name)
    path = os.path.join(ws.dir(*parts), "lm.npz")
    incumbent, vocab = (CharLM.load(path) if os.path.exists(path) and not fresh else (None, None))
    if vocab is None:
        vocab = CharVocab.from_text(text)
    tokens = vocab.encode(text)
    if len(tokens) < 500:
        raise ValueError(f"need at least 500 characters of text, got {len(tokens)}")
    tr, va = split_text_tokens(tokens)
    ev = evolve_lm(tr, va, vocab, generations, population, steps, seed, incumbent, verbose)
    best = ev.best
    best.artifact.save(path, vocab)
    sample = best.artifact.generate(vocab, text[:20], 240, 0.8, np.random.default_rng(seed))
    rec = dict(kind="lm", name=name, fitness=best.fitness, genome=best.genome, sample=sample, **best.info,
               uniform_bpc=float(np.log2(len(vocab))), chars=len(tokens), improved=ev.improved)
    ws.save_best(parts, rec)
    ws.log(rec)
    return best.artifact, vocab, rec


# --------------------------------------------------------------------------- agents
def improve_agent(ws, env_name, generations=3, population=4, seed=0, verbose=True, scale=1.0):
    parts = ("agents", env_name)
    path = os.path.join(ws.dir(*parts), "agent.npz")
    incumbent = PGAgent.load(path) if os.path.exists(path) else None
    updates, episodes = ENV_BUDGET.get(env_name, (60, 8))
    updates = max(4, int(updates * scale))
    ev = evolve_agent(env_name, generations, population, updates, episodes, seed, incumbent, verbose)
    best = ev.best
    best.artifact.save(path)
    rec = dict(kind="agent", name=env_name, fitness=best.fitness, genome=best.genome, **best.info,
               improved=ev.improved)
    ws.save_best(parts, rec)
    ws.log(rec)
    return best.artifact, rec


# --------------------------------------------------------------------------- autopilot
def autopilot(ws, datasets_=(), texts=(), envs=(), rounds=1, seconds=None, seed=0, verbose=True,
              class_generations=15, gen_generations=2, lm_generations=2, agent_generations=2,
              dataset_kwargs=None, scale=1.0):
    """Run improvement rounds over every requested target. `seconds` is a wall-clock budget
    checked between jobs; `scale` multiplies every training budget (0.2 = quick look, 3 = deeper).
    Returns the list of job records."""
    t0, records = time.time(), []
    out_of_time = lambda: seconds is not None and time.time() - t0 >= seconds        # noqa: E731
    dataset_kwargs = dataset_kwargs or {}
    try:
        for r in range(1, rounds + 1):
            for name in datasets_:
                if out_of_time():
                    return records
                task = datasets.load(name, seed=seed, **dataset_kwargs.get(name, {}))
                if verbose:
                    print(f"\n=== round {r}: classifier on {task.summary()}")
                zw, rec = improve_classifier(ws, task, class_generations, seed=seed + r, verbose=False)
                records.append(rec)
                if verbose:
                    print(f"    test acc {rec['test_acc']:.3f} (majority baseline {rec['majority_baseline']:.3f}, "
                          f"before {rec['test_acc_before']}) hidden={rec['hidden']} patch={rec['patch']}")
                if task.is_image and not out_of_time():
                    _, rec = improve_generator(ws, task, zw, gen_generations, steps=max(50, int(500 * scale)),
                                              seed=seed + r, verbose=False)
                    records.append(rec)
                    if verbose:
                        print(f"    generator fidelity {rec['fidelity']:.3f} diversity {rec['diversity']:.3f} "
                              f"elbo {rec['elbo']:.1f} genome={rec['genome']}")
            for path in texts:
                if out_of_time():
                    return records
                name = os.path.splitext(os.path.basename(path))[0]
                _, _, rec = improve_lm(ws, name, read_corpus(path), lm_generations, steps=max(50, int(600 * scale)),
                                     seed=seed + r, verbose=False)
                records.append(rec)
                if verbose:
                    print(f"\n=== round {r}: language model '{name}' bpc {rec['bpc']:.3f} "
                          f"(uniform {rec['uniform_bpc']:.2f})\n    {rec['sample'][:120]!r}")
            for env in envs:
                if out_of_time():
                    return records
                _, rec = improve_agent(ws, env, agent_generations, seed=seed + r, verbose=False, scale=scale)
                records.append(rec)
                if verbose:
                    print(f"\n=== round {r}: agent on {env}: return {rec['eval_return']:.2f} success {rec['success']:.2f}")
    except KeyboardInterrupt:
        ws.log(dict(kind="interrupted"))
        if verbose:
            print("\ninterrupted - everything saved so far is intact")
    return records
