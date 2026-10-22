"""bench.py - a small, deterministic benchmark used to judge self-written SOURCE patches.

Given a seed it trains the self-modifying classifier on the bundled real digits subset, evolves a tiny
cartpole agent, and runs the Evolver on a synthetic landscape. Same code + same seed = same number, so
a difference between two code versions is the code's doing (on that seed), not noise. Different seeds
are used to *select* and to *confirm* a patch, which is what keeps benchmark-overfitting honest.

Run as:  python -m zerowhite selfmod bench --seeds 0 1 2   (prints one JSON line)
"""

import json
import os
import tempfile

import numpy as np


def run_seed(seed):
    os.environ.setdefault("ZERO_WHITE_DATA", tempfile.mkdtemp(prefix="zw_bench_"))
    from .. import datasets, factory
    from ..agents import evolve_agent
    from . import hooks

    task = datasets.load_digits(allow_bundled=True, download=False)
    zw = factory.make_classifier(task, seed=seed, hidden=12)
    Xv, yv = task.Xval[:500], task.yval[:500]
    zw.run(task.Xtr, task.ytr, Xv, yv, generations=5, probation_steps=12, verbose=False)
    acc = zw.evaluate(zw.net, Xv, yv)[1]
    ev = evolve_agent("cartpole", generations=1, population=3, updates=12, episodes=6, seed=seed, verbose=False)
    cart = float(np.clip(ev.best.info["eval_return"] / 200.0, 0.0, 1.0))
    land = hooks.suite_score(hooks.genesis_policy(), ("rastrigin", "ellipsoid"), (seed,), gens=6)
    return dict(seed=seed, digits_val_acc=float(acc), cartpole=cart, landscape=float(land),
                score=float((acc + cart + land) / 3.0))


def run(seeds):
    rows = [run_seed(int(s)) for s in seeds]
    return dict(rows=rows, score=float(np.mean([r["score"] for r in rows])))


def main(seeds):
    print(json.dumps(run(seeds)))
