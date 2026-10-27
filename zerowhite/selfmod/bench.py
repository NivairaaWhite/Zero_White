"""bench.py - a small, deterministic benchmark used to judge self-written SOURCE patches.

Given a seed it trains the self-modifying classifier on the bundled real digits subset, evolves a tiny
cartpole agent, and runs the Evolver on a synthetic landscape. Same code + same seed = same number, so
a difference between two code versions is the code's doing (on that seed), not noise. Different seeds
are used to *select* and to *confirm* a patch, which is what keeps benchmark-overfitting honest.

6.5 widened the reach. Until 6.4 the benchmark only ran code in zero_white, self_model, meta_optimizer, evolve, agents (and, through them, architecture); a patch to generative.py or
curiosity.py could never change the number, so the patcher could neither help nor hurt there and had no reason to propose it. The benchmark now ALSO trains a class-conditional VAE (judged by
the fidelity of what it draws, as classified by an independent nearest-centroid rule, and by its diversity - not by its own ELBO), a character LM (bits per character on held-out text) and
the forward-dynamics curiosity model (held-out next-observation R2 on random cartpole transitions), so every module in patcher.MUTABLE runs under it (tests assert that). Scores from earlier
versions are not comparable with these: the patcher always compares two trees under the SAME benchmark, which is all it needs.

Run as:  python -m zerowhite selfmod bench --seeds 0 1 2   (prints one JSON line)
"""

import json
import os
import tempfile

import numpy as np


_WORDS = ("zero", "white", "learns", "grows", "tests", "keeps", "small", "model", "code", "seed", "paired", "gate", "island", "agent", "reads", "writes")


def _corpus(seed, n_words=1400):
    """Deterministic pseudo-text with real structure (a word is usually followed by a few particular words) for the character LM."""
    r = np.random.default_rng(1000 + seed)
    nxt = {w: list(r.choice(len(_WORDS), size=3, replace=False)) for w in range(len(_WORDS))}
    cur, out = 0, []
    for _ in range(n_words):
        out.append(_WORDS[cur])
        cur = int(nxt[cur][int(r.integers(3))]) if r.random() < 0.9 else int(r.integers(len(_WORDS)))
    return " ".join(out) + "\n"


def _centroid_classifier(X, y, n_classes):
    mu = np.stack([X[y == c].mean(axis=0) if (y == c).any() else np.zeros(X.shape[1]) for c in range(n_classes)])
    return lambda Z: ((Z[:, None, :] - mu[None]) ** 2).sum(axis=2).argmin(axis=1)


def _generative_scores(task, seed):
    """(vae, charlm): held-out quality of the two generative models, each in [0, 1]."""
    from .. import generative as G
    Xt, yt = np.clip(task.Xtr, 0, 1), task.ytr
    vae = G.CVAE(task.in_dim, task.n_classes, seed=seed)                 # the module's own defaults: those are what a patch to generative.py changes
    vae.fit(Xt, yt, 300, rng=np.random.default_rng(seed))
    Xv, yv = np.clip(task.Xval[:400], 0, 1), task.yval[:400]
    predict = _centroid_classifier(Xt, yt, task.n_classes)
    # the measures are computed HERE, not by generative.py's own fidelity()/diversity()/bits_per_char(): a patch must not be able to raise its score by changing how it is measured
    yq = np.repeat(np.arange(task.n_classes), 20)
    drawn = vae.sample(yq, np.random.default_rng(seed))
    fid = float((predict(drawn) == yq).mean())
    real = [Xv[yv == c].var(axis=0).mean() for c in range(task.n_classes) if (yv == c).sum() > 2]
    rng_d = np.random.default_rng(seed)
    fake = [vae.sample(np.full(20, c), rng_d).var(axis=0).mean() for c in range(task.n_classes) if (yv == c).sum() > 2]
    div = float(min(1.0, np.mean(fake) / (np.mean(real) + 1e-12)))
    text = _corpus(seed)
    vocab = G.CharVocab.from_text(text)
    toks = vocab.encode(text)
    tr, va = G.split_text_tokens(toks)
    lm = G.CharLM(len(vocab), seed=seed)
    lm.fit(tr, 80, rng=np.random.default_rng(seed))
    from ..nn import mlp_raw
    c = lm.cfg["ctx"]
    pos = np.arange(c, min(len(va), c + 1500))
    xb, yb = np.stack([va[q - c:q] for q in pos]), va[pos]
    with np.errstate(all="ignore"):
        z = mlp_raw(lm.p.a, "lm", lm.p.a["E"][xb].reshape(len(xb), -1))
        z = z - z.max(axis=1, keepdims=True)
        bpc = float((-(z[np.arange(len(yb)), yb] - np.log(np.exp(z).sum(axis=1)))).mean() / np.log(2.0))
    return float(np.clip(0.5 * fid + 0.5 * div, 0.0, 1.0)), float(np.clip(1.0 - bpc / np.log2(len(vocab)), 0.0, 1.0))


def _curiosity_score(seed):
    """Held-out R2 of the forward-dynamics curiosity model's prediction of the next cartpole observation, from random-policy transitions."""
    from ..agents import make_env
    from ..curiosity import ForwardDynamicsCuriosity
    env, rng = make_env("cartpole"), np.random.default_rng(seed)
    O, A_, N = [], [], []
    o = env.reset(rng)
    while len(O) < 700:
        a = int(rng.integers(env.n_actions))
        o2, _, done, _ = env.step(a)
        O.append(o), A_.append(a), N.append(o2)
        o = env.reset(rng) if done else o2
    O, A_, N = np.array(O), np.array(A_), np.array(N)
    cut = 500
    icm = ForwardDynamicsCuriosity(env.obs_dim, env.n_actions, seed=seed)
    icm.update(O[:cut], A_[:cut], N[:cut], epochs=40)
    from ..nn import mlp_raw
    pred = mlp_raw(icm.params.a, "fwd", np.concatenate([O[cut:], np.eye(env.n_actions)[A_[cut:]]], axis=1))
    mse = float(((pred - N[cut:]) ** 2).mean())
    return float(np.clip(1.0 - mse / (float(N[cut:].var(axis=0).mean()) + 1e-12), 0.0, 1.0))


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
    vae, charlm = _generative_scores(task, seed)
    curio = _curiosity_score(seed)
    return dict(seed=seed, digits_val_acc=float(acc), cartpole=cart, landscape=float(land), vae=vae, charlm=charlm, curiosity=curio,
                score=float((acc + cart + land + vae + charlm + curio) / 6.0))


def run(seeds):
    rows = [run_seed(int(s)) for s in seeds]
    return dict(rows=rows, score=float(np.mean([r["score"] for r in rows])))


def main(seeds):
    print(json.dumps(run(seeds)))
