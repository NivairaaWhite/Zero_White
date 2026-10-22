"""latent.py - a Perceiver-style attention bottleneck that compresses a high-dimensional observation into a few numbers for a program agent.

`reduce.py` (6.3) hands program agents a whitened PCA of the observation: exact, linear, cheap. PCA cannot straighten a curved set of observations: pictures of one
blob at 2-D positions form a 2-D SURFACE that PCA needs dozens of directions to describe. This encoder is the nonlinear alternative named in the architecture review:

  tokens     every input number becomes a token  x_i * v_i + p_i  (a learned value embedding and a learned position embedding, so the model knows WHERE each number is);
  attention  M learned latent queries cross-attend over all D tokens (softmax over the D positions): each latent gathers what it needs from the whole input, and the cost
             grows with D * M, not D^2 (the point of the Perceiver);
  code       the M attended vectors are mixed by one tanh layer into k numbers (the only thing the program agent sees), then a small MLP decoder reconstructs the observation;
  training   reconstruction error by Adam, all gradients written out by hand in numpy (the engine's autograd has no batched matmul) and checked against finite differences.

Unsupervised, so it keeps whatever varies most, relevant to the task or not. It is fitted on observations from a RANDOM policy; it needs a few hundred to a few thousand of them.
Only used when `ProgramPolicyAgent.encoder == "latent"` (default stays PCA).
"""
from __future__ import annotations

import numpy as np


def _softmax(S):
    e = np.exp(S - S.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class LatentEncoder:
    NAMES = ("Vv", "P", "Wk", "Wv", "Q", "Wz", "bz", "D1", "d1", "D2", "d2")

    def __init__(self, params, mean, scale, zmean, zscale, d, M, k):
        self.p, self.mean, self.scale, self.zmean, self.zscale = params, mean, scale, zmean, zscale
        self.d, self.M, self.k = d, M, k

    # ---- model -------------------------------------------------------------------------------------------------
    @staticmethod
    def init_params(D, d, M, k, hidden, rng):
        s = lambda *shape, f=1.0: rng.standard_normal(shape) * f                                   # noqa: E731
        return dict(Vv=s(D, d, f=0.5), P=s(D, d, f=0.5), Wk=s(d, d, f=1 / np.sqrt(d)), Wv=s(d, d, f=1 / np.sqrt(d)), Q=s(M, d, f=1.0),
                    Wz=s(M * d, k, f=1 / np.sqrt(M * d)), bz=np.zeros(k), D1=s(k, hidden, f=np.sqrt(1 / k)), d1=np.zeros(hidden),
                    D2=s(hidden, D, f=np.sqrt(1 / hidden)), d2=np.zeros(D))

    @staticmethod
    def encode_raw(p, X, d):
        """X (B, D) standardised -> (z (B,k), cache)."""
        T = X[:, :, None] * p["Vv"][None] + p["P"][None]                  # (B, D, d)
        K, V = T @ p["Wk"], T @ p["Wv"]
        S = np.einsum("md,bid->bmi", p["Q"], K) / np.sqrt(d)               # (B, M, D)
        A = _softmax(S)
        O = np.einsum("bmi,bid->bmd", A, V)                                # (B, M, d)
        Zp = O.reshape(len(X), -1)
        z = np.tanh(Zp @ p["Wz"] + p["bz"])
        return z, dict(T=T, K=K, V=V, A=A, O=O, Zp=Zp, X=X)

    @staticmethod
    def decode_raw(p, z):
        h = np.tanh(z @ p["D1"] + p["d1"])
        return h @ p["D2"] + p["d2"], h

    @staticmethod
    def loss_grads(p, X, d):
        B = len(X)
        z, c = LatentEncoder.encode_raw(p, X, d)
        Y, h = LatentEncoder.decode_raw(p, z)
        loss = float(np.mean((Y - X) ** 2))
        dY = 2.0 * (Y - X) / X.size
        g = {n: np.zeros_like(v) for n, v in p.items()}
        g["D2"], g["d2"] = h.T @ dY, dY.sum(0)
        dh = (dY @ p["D2"].T) * (1 - h ** 2)
        g["D1"], g["d1"] = z.T @ dh, dh.sum(0)
        dz = (dh @ p["D1"].T) * (1 - z ** 2)
        g["Wz"], g["bz"] = c["Zp"].T @ dz, dz.sum(0)
        dO = (dz @ p["Wz"].T).reshape(B, -1, d)                            # (B, M, d)
        dA = np.einsum("bmd,bid->bmi", dO, c["V"])
        dV = np.einsum("bmi,bmd->bid", c["A"], dO)
        dS = c["A"] * (dA - (c["A"] * dA).sum(-1, keepdims=True)) / np.sqrt(d)
        g["Q"] = np.einsum("bmi,bid->md", dS, c["K"])
        dK = np.einsum("bmi,md->bid", dS, p["Q"])
        g["Wk"], g["Wv"] = np.einsum("bid,bie->de", c["T"], dK), np.einsum("bid,bie->de", c["T"], dV)
        dT = dK @ p["Wk"].T + dV @ p["Wv"].T
        g["Vv"] = np.einsum("bi,bid->id", X, dT)
        g["P"] = dT.sum(0)
        return loss, g

    # ---- fitting + use -----------------------------------------------------------------------------------------
    @classmethod
    def fit(cls, obs, k: int = 8, d: int = 16, M: int = 4, hidden: int = 64, epochs: int = 120, seed: int = 0, lr: float = 3e-3):
        X = np.nan_to_num(np.asarray(obs, dtype=float))
        if X.ndim != 2 or len(X) < 16:
            raise ValueError("need at least 16 observations of a fixed width to fit an encoder")
        mean, scale = X.mean(0), X.std(0)
        scale = np.where(scale > 1e-9, scale, 1.0)
        Z = (X - mean) / scale
        rng = np.random.default_rng(seed)
        p = cls.init_params(Z.shape[1], d, M, k, hidden, rng)
        m = {n: np.zeros_like(v) for n, v in p.items()}
        v2 = {n: np.zeros_like(v) for n, v in p.items()}
        step = 0
        for _ in range(epochs):
            order = rng.permutation(len(Z))
            for i in range(0, len(Z), 64):
                loss, g = cls.loss_grads(p, Z[order[i:i + 64]], d)
                step += 1
                for n in p:
                    m[n] = 0.9 * m[n] + 0.1 * g[n]
                    v2[n] = 0.999 * v2[n] + 0.001 * g[n] ** 2
                    p[n] -= lr * (m[n] / (1 - 0.9 ** step)) / (np.sqrt(v2[n] / (1 - 0.999 ** step)) + 1e-8)
        z, _ = cls.encode_raw(p, Z, d)
        return cls(p, mean, scale, z.mean(0), np.maximum(z.std(0), 1e-3), d, M, k)

    def __call__(self, o) -> np.ndarray:
        x = (np.nan_to_num(np.asarray(o, dtype=float)) - self.mean) / self.scale
        z, _ = self.encode_raw(self.p, x[None, :], self.d)
        return (z[0] - self.zmean) / self.zscale

    def recon_mse(self, obs) -> float:
        Z = (np.asarray(obs, float) - self.mean) / self.scale
        z, _ = self.encode_raw(self.p, Z, self.d)
        return float(np.mean((self.decode_raw(self.p, z)[0] - Z) ** 2))

    def to_dict(self) -> dict:
        return dict(p={n: v.tolist() for n, v in self.p.items()}, mean=self.mean.tolist(), scale=self.scale.tolist(), zmean=self.zmean.tolist(), zscale=self.zscale.tolist(),
                    d=self.d, M=self.M, k=self.k)

    @classmethod
    def from_dict(cls, dct: dict) -> "LatentEncoder":
        return cls({n: np.asarray(v, float) for n, v in dct["p"].items()}, np.asarray(dct["mean"]), np.asarray(dct["scale"]), np.asarray(dct["zmean"]),
                   np.asarray(dct["zscale"]), int(dct["d"]), int(dct["M"]), int(dct["k"]))


def pca_recon_mse(obs, k: int) -> float:
    """Reconstruction error (same standardised units) of the best LINEAR k-dimensional code: the fair baseline for the encoder."""
    X = np.asarray(obs, float)
    scale = X.std(0)
    Z = (X - X.mean(0)) / np.where(scale > 1e-9, scale, 1.0)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    B = vt[:k]
    return float(np.mean((Z - (Z @ B.T) @ B) ** 2))
