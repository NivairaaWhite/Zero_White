"""Tests for 6.5 non-linear observation compression for program agents (forge/obsenc.py) and its use by ProgramPolicyAgent."""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ZERO_WHITE_DATA", tempfile.mkdtemp(prefix="zw_test_data_"))

from zerowhite import fileio                                            # noqa: E402
from zerowhite.forge import families as F                               # noqa: E402
from zerowhite.forge import obsenc, task as T                           # noqa: E402
from zerowhite.forge.latent import LatentEncoder, pca_recon_mse         # noqa: E402
from zerowhite.forge.obsenc import VQEncoder, choose_encoder, collect, task_score       # noqa: E402
from zerowhite.forge.reduce import Projector                            # noqa: E402

_DIRS = []


def tmp():
    d = tempfile.mkdtemp(prefix="zw_t65o_")
    _DIRS.append(d)
    return d


import atexit                                                           # noqa: E402
atexit.register(lambda: [shutil.rmtree(d, ignore_errors=True) for d in _DIRS])


def blobs(n, seed, size=12, sigma=1.3):
    r = np.random.default_rng(seed)
    xy = r.uniform(1.5, size - 2.5, size=(n, 2))
    g = np.arange(size)
    img = np.exp(-(((g[None, :, None] - xy[:, 0, None, None]) ** 2 + (g[None, None, :] - xy[:, 1, None, None]) ** 2) / (2 * sigma ** 2)))
    return img.reshape(n, -1) + 0.01 * r.normal(size=(n, size * size)), xy


class Blob:
    obs_dim, n_actions, max_steps, max_return, name = 144, 2, 10, 10.0, "blob"

    def reset(self, rng):
        self.rng, self.t = rng, 0
        return self._n()

    def _n(self):
        self.x = self.rng.uniform(1.5, 9.5)
        g = np.arange(12)
        return (np.exp(-(((g[:, None] - self.x) ** 2 + (g[None, :] - 5.5) ** 2) / (2 * 1.3 ** 2))) + 0.01 * self.rng.normal(size=(12, 12))).ravel()

    def step(self, a):
        self.t += 1
        return self._n(), float(int(a) == int(self.x > 5.5)), self.t >= 10, False


class Mixed:
    """20 numbers that are a fixed random mixing of ONE slow variable r that decides the reward and THREE large, irrelevant slow processes (std ~5): a reconstruction-optimal code with
    3 components keeps the walks and drops r."""
    obs_dim, n_actions, max_steps, max_return, name = 20, 2, 200, 200.0, "mixed"
    M = np.random.default_rng(5).normal(size=(20, 4))

    def reset(self, rng):
        self.rng, self.t = rng, 0
        self.r, self.d = float(rng.normal()), rng.normal(size=3) * 5
        return self._o()

    def _o(self):
        return self.M @ np.concatenate([[self.r], self.d]) + 0.05 * self.rng.normal(size=20)

    def step(self, a):
        self.t += 1
        self.r = 0.9 * self.r + 0.3 * (2 * int(a) - 1) + 0.05 * self.rng.normal()                # stationary: the held-out part of a probe stays inside the range of the fitted part
        self.d = 0.95 * self.d + 1.5 * self.rng.normal(size=3)
        return self._o(), float(self.r), self.t >= 200, False


def best_threshold_accuracy(Z, y):
    best = 0.0
    for j in range(Z.shape[1]):
        for t in np.quantile(Z[:, j], np.linspace(0.05, 0.95, 37)):
            acc = float(np.mean((Z[:, j] > t) == y))
            best = max(best, acc, 1 - acc)
    return best


# =========================================================================== the encoder
def test_the_vq_encoders_hand_written_gradients_match_finite_differences_for_both_training_modes():
    rng = np.random.default_rng(0)
    D, h, k = 9, 6, 3
    P = [rng.normal(0, 0.5, (D, h)), rng.normal(0, 0.1, h), rng.normal(0, 0.5, (h, k)), rng.normal(0, 0.1, k), rng.normal(0, 0.5, (k, h)), rng.normal(0, 0.1, h),
         rng.normal(0, 0.5, (h, D)), rng.normal(0, 0.1, D)]
    X = rng.normal(size=(7, D))
    Zq = rng.normal(size=(7, k))
    for override in (None, Zq):                                          # joint autoencoder training, then decoder-only on quantised latents
        loss, G = VQEncoder.loss_grads(P, X, override)
        checked = 0
        for j, p in enumerate(P):
            for _ in range(5):
                ix = tuple(int(rng.integers(s)) for s in p.shape)
                old, e = p[ix], 1e-6
                p[ix] = old + e
                lp = VQEncoder.loss_grads(P, X, override)[0]
                p[ix] = old - e
                lm = VQEncoder.loss_grads(P, X, override)[0]
                p[ix] = old
                num = (lp - lm) / (2 * e)
                if abs(num) > 1e-9 or abs(G[j][ix]) > 1e-9:
                    checked += 1
                    assert abs(num - G[j][ix]) <= 1e-5 * max(1e-3, abs(num), abs(G[j][ix])), (override is not None, j, ix, num, G[j][ix])
        assert checked >= (25 if override is None else 12)
        if override is not None:
            assert all(not G[j].any() for j in range(4))                  # the encoder is frozen in that mode


def test_the_vq_code_describes_a_curved_set_far_better_than_pca_at_a_few_bits_and_every_prefix_of_the_stages_is_usable():
    """Measured (800 training pictures, k=3): test MSE 0.085 from 10 bits (2 stages x 32 codewords) against 0.58 for PCA with 3 CONTINUOUS components."""
    Xtr, _ = blobs(800, 0)
    Xte, _ = blobs(300, 9)
    enc = VQEncoder.fit(Xtr, k=3, stages=2, codewords=32, epochs=70, seed=0)
    pca = pca_recon_mse(np.vstack([Xtr, Xte]), 3)
    full, first = enc.recon_mse(Xte), enc.recon_mse(Xte, 1)
    assert full < 0.3 * pca and full < first < pca, (full, first, pca)                          # a coarser prefix is worse than the whole code, and still better than PCA
    assert enc.bits() == 10.0 and (enc.k, enc.stages, enc.codewords) == (3, 2, 32)
    t = enc.tokens(Xte[0])
    assert t.shape == (2,) and t.dtype == np.int64 and t.min() >= 0 and t.max() < 32 and np.array_equal(enc.tokens(Xte[0]), t)                 # deterministic integer tokens
    assert enc.tokens(Xte[0], stages=1).shape == (1,)
    Z = np.array([enc(x) for x in Xte[:150]])
    assert Z.shape == (150, 3) and np.isfinite(Z).all() and np.abs(Z.mean(0)).max() < 0.5 and 0.5 < Z.std(0).mean() < 1.5                       # standardised numbers for the programs
    assert np.allclose(enc.dequantise(t), enc.dequantise(t[None])) and len(np.unique([tuple(enc.tokens(x)) for x in Xte[:150]], axis=0)) > 5
    enc2 = VQEncoder.from_dict(json.loads(json.dumps(enc.to_dict())))
    assert np.allclose(enc2(Xte[3]), enc(Xte[3])) and np.array_equal(enc2.tokens(Xte[3]), enc.tokens(Xte[3])) and abs(enc2.recon_mse(Xte) - full) < 1e-9
    for bad in (Xtr[:5], np.zeros((20, 3, 2))):
        try:
            VQEncoder.fit(bad, k=2)
            raise AssertionError("must refuse")
        except ValueError:
            pass
    const = VQEncoder.fit(np.ones((40, 6)) + 0.0, k=2, epochs=2, finetune_epochs=1, codewords=4)             # a constant observation does not crash the fit
    assert np.isfinite(const(np.ones(6))).all()


def test_one_number_of_the_vq_code_carries_what_a_program_needs_more_often_than_a_pca_component_does():
    """The point for a program agent that reads single numbers: is there one code number that, thresholded, says which side of the picture the blob is on? PCA 0.79, attention encoder 0.85, VQ 0.90 (k=8)."""
    Xtr, _ = blobs(800, 0)
    Xte, xy = blobs(400, 9)
    y = xy[:, 0] > 5.5
    vq = VQEncoder.fit(Xtr, k=8, seed=0)
    pca = Projector.fit(Xtr, 8)
    a_vq = best_threshold_accuracy(np.array([vq(x) for x in Xte]), y)
    a_pca = best_threshold_accuracy(np.array([pca(x) for x in Xte]), y)
    assert a_vq >= 0.85 and a_vq >= a_pca + 0.05, (a_vq, a_pca)


# =========================================================================== the task-aware score and the choice
def test_the_task_score_tells_a_code_that_keeps_the_rewarding_variable_from_one_that_keeps_the_biggest_variance():
    task = T.EnvTask(Mixed())
    pr = collect(task, n=900, seed=0)
    assert pr.obs.shape == pr.nxt.shape and len(pr.act) == len(pr.ret) == len(pr.obs) and pr.episodes >= 3 and pr.n_actions == 2
    Mx = Mixed.M
    pinv = np.linalg.pinv(Mx)
    oracle = lambda o: (pinv @ o)[:1]                                                                    # noqa: E731 - recovers r exactly (up to noise)
    cut = int(len(pr.obs) * 0.7)
    s_oracle = task_score(oracle, pr)
    s_pca3 = task_score(Projector.fit(pr.obs[:cut], 3), pr)
    s_pca4 = task_score(Projector.fit(pr.obs[:cut], 4), pr)
    noise = np.random.default_rng(1)
    s_noise = task_score(lambda o: noise.normal(size=3), pr)
    assert s_oracle["reward"] > 0.9 and s_pca4["reward"] > 0.9                                          # a 4-component PCA has room for r
    assert s_pca3["reward"] < 0.3 and s_noise["reward"] < 0.3                                           # a 3-component PCA spends all its components on the three loud processes
    assert s_pca3["dynamics"] > 0.5 > s_oracle["dynamics"]                                               # ... and predicts the next picture well (the loud processes ARE the picture): why that is not scored
    assert s_oracle["score"] > s_pca3["score"] + 0.3 and s_oracle["score"] > s_noise["score"] + 0.3 and abs(s_pca4["score"] - s_oracle["score"]) < 0.25
    assert all(-1.0 <= v <= 1.0 for s in (s_oracle, s_pca3, s_pca4, s_noise) for v in s.values())
    flat = obsenc.Probe(pr.obs, pr.nxt, pr.act, np.zeros(len(pr.obs)), np.zeros(len(pr.obs)), pr.n_actions, pr.episodes)       # nothing to predict about the reward: only the dynamics are left
    sc = task_score(Projector.fit(pr.obs[:cut], 4), flat)
    assert sc["value"] == 0.0 and sc["reward"] == 0.0 and sc["score"] == sc["dynamics"] > 0.5


def test_choose_encoder_reports_every_score_skips_a_candidate_that_fails_and_prefers_the_cheaper_one_on_a_tie():
    task = T.EnvTask(Mixed())
    quick = dict(latent=dict(epochs=4), vq=dict(epochs=10, finetune_epochs=2))
    ch = choose_encoder(task, k=4, seed=0, n=400, fit_kwargs=quick)
    assert ch.name in ("pca", "latent", "vq") and set(ch.scores) == {"pca", "latent", "vq"} and ch.errors == {}
    best = max(ch.scores, key=lambda n: (ch.scores[n]["score"], -["pca", "latent", "vq"].index(n)))
    assert ch.name == best and ch.encoder.k == 4 and np.isfinite(ch.encoder(task.env.reset(np.random.default_rng(0)))).all()
    rep = ch.report()
    assert rep["chosen"] == ch.name and set(rep["scores"]["pca"]) == {"value", "reward", "dynamics", "score"}
    json.dumps(rep)                                                                                      # it goes into state.json
    bad = choose_encoder(task, candidates=("pca", "nope"), k=3, n=300)
    assert bad.name == "pca" and "nope" in bad.errors
    try:
        choose_encoder(task, candidates=("nope",), n=300)
        raise AssertionError("nothing fitted")
    except ValueError:
        pass
    ties = choose_encoder(task, candidates=("pca", "pca"), k=3, n=300)                                    # equal scores: the first one stands
    assert ties.name == "pca"


# =========================================================================== in the program agent
GENOME = dict(n_islands=2, island_size=6, tune_steps=4, max_nodes=50, n_mut=2)


def _quick_fits():
    """These tests are about plumbing, not quality: fewer epochs."""
    real_vq, real_lat, real_choose = VQEncoder.fit.__func__, LatentEncoder.fit.__func__, F.choose_encoder
    VQEncoder.fit = classmethod(lambda cls, obs, **kw: real_vq(cls, obs, **dict(kw, epochs=8, finetune_epochs=2)))
    LatentEncoder.fit = classmethod(lambda cls, obs, **kw: real_lat(cls, obs, **dict(kw, epochs=4)))
    F.choose_encoder = lambda task, **kw: real_choose(task, **dict(kw, n=300, fit_kwargs=dict(latent=dict(epochs=3), vq=dict(epochs=8, finetune_epochs=2))))

    def restore():
        VQEncoder.fit, LatentEncoder.fit, F.choose_encoder = classmethod(real_vq), classmethod(real_lat), real_choose
    return restore


def test_a_program_agent_can_use_the_vq_or_the_auto_encoder_and_a_reloaded_agent_sees_the_same_code():
    assert F.ProgramPolicyAgent.encoder == "pca"
    restore = _quick_fits()
    try:
        for mode in ("vq", "auto"):
            F.ProgramPolicyAgent.encoder = mode
            task = T.EnvTask(Blob())
            a = F.ProgramPolicyAgent(task, GENOME, 0)
            assert a.proj is not None and a.vocab.n_cols == a.proj.k == 8
            if mode == "vq":
                assert isinstance(a.proj, VQEncoder) and a.encoder_choice is None
            else:
                assert a.encoder_choice["chosen"] in ("pca", "latent", "vq") and set(a.encoder_choice["scores"]) == {"pca", "latent", "vq"}
            a.train(task, 0.5, np.random.default_rng(0))
            d = tmp()
            a.save(d)
            st = json.loads(fileio.read_text(os.path.join(d, "state.json")))
            key = {VQEncoder: "vq", LatentEncoder: "latent", Projector: "projector"}[type(a.proj)]
            assert key in st and sum(k in st for k in ("vq", "latent", "projector")) == 1
            assert (st.get("encoder_choice") is not None) == (mode == "auto")
            prog = fileio.read_text(os.path.join(d, "program.py"))
            assert {"vq": "vector-quantised", "latent": "attention-encoder", "projector": "PCA"}[key] in prog
            b = F.ProgramPolicyFamily().load(d, task, GENOME)
            xs = [Blob().reset(np.random.default_rng(i)) for i in range(10)]
            assert type(b.proj) is type(a.proj) and np.allclose(b.proj(xs[0]), a.proj(xs[0])) and np.allclose(b.proj(xs[4]), a.proj(xs[4]))        # the SAME code, not a refit
            assert all(b._act_fn()(x) in (0, 1) for x in xs) and np.isfinite(b.val_score(task)[0])                                                   # (the restored population is re-scored, so which program is best may differ)
            assert b.encoder_choice == (st.get("encoder_choice"))
    finally:
        restore()
        F.ProgramPolicyAgent.encoder = "pca"
    narrow = T.EnvTask(Mixed())                                                                          # 20 columns <= 24: no projection is needed, so none is used whatever `encoder` says
    F.ProgramPolicyAgent.encoder = "vq"
    try:
        n = F.ProgramPolicyAgent(narrow, GENOME, 0)
        assert n.proj is None and n.vocab.n_cols == 20
    finally:
        F.ProgramPolicyAgent.encoder = "pca"
