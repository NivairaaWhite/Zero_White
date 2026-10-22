"""model.py - a decoder-only transformer in plain numpy with a hand-written backward pass.

Architecture (the Llama recipe, which also makes the export to other tools a straight copy):
    token embedding -> N x [ RMSNorm -> multi-head causal attention with rotary positions -> residual
                             RMSNorm -> SwiGLU feed-forward                                -> residual ] -> RMSNorm -> tied output

Why hand-written backward instead of the project's autograd engine (engine.py)?  This model is the one that really trains,
so speed matters: no graph objects, float32, batched matmuls through BLAS, attention probabilities kept for the backward
pass, and a cross-entropy head that is chunked so the (tokens x vocab) logit matrix never exists in full.  Correctness is
guarded by a finite-difference gradient check in the test-suite (tests/test_chat.py).

Batches are dictionaries of (B, T) arrays:
    ids  input tokens        tgt  next tokens (what the model must predict)
    w    loss weight per target position (0 = not trained on; may be negative for preference/unlikelihood updates)
    pos  position of each token inside its own document (RoPE restarts at every document)
    seg  document number (> 0), tokens only attend inside their own document, so several conversations can be packed
         into one row without leaking context between them.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .. import fileio
from . import backend as bk

NEG = -1e30


@dataclass
class GPTConfig:
    vocab: int
    d_model: int
    n_layer: int
    n_head: int
    ffn: int
    ctx: int
    rope_theta: float = 10000.0
    eps: float = 1e-5
    tie: bool = True
    dtype: str = "float32"
    n_experts: int = 1            # 6.4: 1 = the dense SwiGLU feed-forward (unchanged). >1: a mixture of that many SwiGLU experts per block, `top_k` of them used per token
    top_k: int = 1
    moe_aux: float = 0.01         # weight of the load-balancing loss (Switch Transformer's: routed share x mean router probability) that keeps the router from sending everything to one expert

    def __post_init__(self):
        if self.n_experts < 1 or not 1 <= self.top_k <= self.n_experts:
            raise ValueError(f"need n_experts >= 1 and 1 <= top_k <= n_experts (got {self.n_experts}, {self.top_k})")
        if self.d_model % self.n_head:
            raise ValueError(f"d_model {self.d_model} must be divisible by n_head {self.n_head}")
        if (self.d_model // self.n_head) % 2:
            raise ValueError("head_dim must be even (rotary embeddings rotate pairs)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def round_up(x: float, m: int) -> int:
    return int(m * math.ceil(x / m))


def param_shapes(c: GPTConfig) -> Dict[str, Tuple[int, ...]]:
    D, F = c.d_model, c.ffn
    s: Dict[str, Tuple[int, ...]] = {"tok_emb": (c.vocab, D)}
    for l in range(c.n_layer):
        s.update({f"b{l}.n1": (D,), f"b{l}.wq": (D, D), f"b{l}.wk": (D, D), f"b{l}.wv": (D, D), f"b{l}.wo": (D, D), f"b{l}.n2": (D,)})
        if c.n_experts > 1:
            s[f"b{l}.rt"] = (D, c.n_experts)
            for j in range(c.n_experts):
                s.update({f"b{l}.e{j}.wg": (D, F), f"b{l}.e{j}.wu": (D, F), f"b{l}.e{j}.wd": (F, D)})
        else:
            s.update({f"b{l}.wg": (D, F), f"b{l}.wu": (D, F), f"b{l}.wd": (F, D)})
    s["nf"] = (D,)
    if not c.tie:
        s["lm_head"] = (c.vocab, D)
    return s


def count_params(c: GPTConfig) -> Tuple[int, int]:
    """(total, non_embedding)"""
    sh = param_shapes(c)
    tot = sum(int(np.prod(v)) for v in sh.values())
    emb = int(np.prod(sh["tok_emb"])) + (int(np.prod(sh["lm_head"])) if "lm_head" in sh else 0)
    return tot, tot - emb


def _silu_parts(a, xp=np):
    sg = 1.0 / (1.0 + xp.exp(-a))
    return sg, a * sg


def _moe_forward(n2f, p, pre, c, xp=np, keep=False):
    """Mixture-of-experts feed-forward for the rows of n2f (N, D). Each row is sent to its top_k experts (by router logit), the k logits are soft-maxed into gates that
    sum to 1, and the output is the gate-weighted sum of the chosen SwiGLU experts' outputs. Returns (output (N, D), cache). Identical experts therefore give exactly the
    dense feed-forward's output whatever the router says: that is what makes `grow.to_moe` function-preserving."""
    N, E, K = n2f.shape[0], c.n_experts, c.top_k
    g = n2f @ p[pre + "rt"]
    idx = xp.argsort(-g, axis=1)[:, :K] if K < E else xp.broadcast_to(xp.arange(E), (N, E)).copy()
    sel = xp.take_along_axis(g, idx, axis=1)
    w = xp.exp(sel - sel.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    out = xp.zeros_like(n2f)
    experts = []
    for j in range(E):
        rows, slot = xp.where(idx == j)
        if rows.size == 0:
            experts.append(None)
            continue
        xj = n2f[rows]
        a = xj @ p[f"{pre}e{j}.wg"]
        b = xj @ p[f"{pre}e{j}.wu"]
        _, silu = _silu_parts(a, xp)
        h = silu * b
        y = h @ p[f"{pre}e{j}.wd"]
        out[rows] += w[rows, slot][:, None] * y
        if keep:
            experts.append((rows, slot, xj, a, b, h, y))
    aux = 0.0
    cache = None
    if keep:
        P = xp.exp(g - g.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        m = P.mean(axis=0)
        f = xp.bincount(idx.reshape(-1), minlength=E).astype(P.dtype) / float(N * K)         # the share of routed slots each expert actually got (a constant for the gradient)
        aux = float(c.moe_aux * E * (f.astype(np.float64) * m.astype(np.float64)).sum()) if c.moe_aux else 0.0
        cache = dict(idx=idx, w=w, experts=experts, P=P, m=m, f=f, N=N)
    return out, cache, aux


def _moe_backward(dmf, n2f, cache, p, pre, grads, c, xp=np):
    """Gradient of _moe_forward (and of its load-balancing loss) w.r.t. its input and parameters; adds into `grads`. Returns dn2f."""
    N, E, K = n2f.shape[0], c.n_experts, c.top_k
    idx, w = cache["idx"], cache["w"]
    dn2 = xp.zeros_like(n2f)
    dw = xp.zeros_like(w)
    for j, ex in enumerate(cache["experts"]):
        if ex is None:
            continue
        rows, slot, xj, a, b, h, y = ex
        wj = w[rows, slot][:, None]
        dmj = dmf[rows]
        dy = wj * dmj
        dw[rows, slot] = (dmj * y).sum(axis=1)
        grads[f"{pre}e{j}.wd"] += h.T @ dy
        dh = dy @ p[f"{pre}e{j}.wd"].T
        sg, silu = _silu_parts(a, xp)
        db = dh * silu
        da = dh * b * (sg * (1.0 + a * (1.0 - sg)))
        grads[f"{pre}e{j}.wg"] += xj.T @ da
        grads[f"{pre}e{j}.wu"] += xj.T @ db
        dn2[rows] += da @ p[f"{pre}e{j}.wg"].T + db @ p[f"{pre}e{j}.wu"].T
    dsel = w * (dw - (w * dw).sum(axis=1, keepdims=True))                       # softmax Jacobian over the selected logits
    dg = xp.zeros((N, E), dtype=n2f.dtype)
    rr = xp.arange(N)[:, None]
    dg[rr, idx] += dsel
    if c.moe_aux:                                                                  # d/dg of  moe_aux * E * sum_j f_j * mean_n(P_nj)   (Switch Transformer's balancing loss)
        P, f = cache["P"], cache["f"]
        dP = (c.moe_aux * E / N) * f[None, :] * xp.ones_like(P)
        dg += P * (dP - (P * dP).sum(axis=1, keepdims=True))
    grads[pre + "rt"] += n2f.T @ dg
    dn2 += dg @ p[pre + "rt"].T
    return dn2


# The four helpers take an optional array module (default numpy) so inference (generate.py, numpy + KV cache) calls them unchanged
# while training can run them on a GPU through the same lines.
def _rope(x, cos, sin, xp=np):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return xp.concatenate((x1 * cos - x2 * sin, x1 * sin + x2 * cos), axis=-1)


def _rope_back(g, cos, sin, xp=np):
    h = g.shape[-1] // 2
    g1, g2 = g[..., :h], g[..., h:]
    return xp.concatenate((g1 * cos + g2 * sin, g2 * cos - g1 * sin), axis=-1)


def _rms(x, g, eps, xp=np):
    inv = 1.0 / xp.sqrt(xp.mean(x * x, axis=-1, keepdims=True) + eps)
    xh = x * inv
    return xh * g, xh, inv


def _rms_back(dy, g, xh, inv, xp=np):
    dg = (dy * xh).reshape(-1, xh.shape[-1]).sum(axis=0)
    dxh = dy * g
    dx = inv * (dxh - xh * xp.mean(dxh * xh, axis=-1, keepdims=True))
    return dx, dg


def _scatter_rows(dE, ids_flat, rows):
    """Kept for callers of the old name: the CPU implementation now lives on the backend (`Backend.scatter_rows`)."""
    bk.CPU.scatter_rows(dE, ids_flat, rows)


class GPT:
    def __init__(self, cfg: GPTConfig, seed: int = 0, params: Optional[Dict[str, np.ndarray]] = None,
                 init_std: float = 0.02, backend: Optional[bk.Backend] = None):
        self.cfg = cfg
        self.dt = np.dtype(cfg.dtype)
        self.be = backend or bk.CPU
        self.xp = self.be.xp
        xp = self.xp
        if params is not None:
            # host arrays (a file) or arrays already on this device (a clone): either way the weights end up contiguous on the device
            self.p = {k: xp.ascontiguousarray(self.be.asarray(v, dtype=self.dt)) for k, v in params.items()}
            self._check_shapes()
        else:
            rng = np.random.default_rng(seed)                  # initialised on the HOST: the same weights on every backend
            self.p = {}
            res_scale = 1.0 / math.sqrt(2.0 * cfg.n_layer)
            for name, shp in param_shapes(cfg).items():
                if name.endswith((".n1", ".n2")) or name == "nf":
                    self.p[name] = self.be.asarray(np.ones(shp, dtype=self.dt))
                else:
                    std = init_std * (res_scale if name.endswith((".wo", ".wd")) else 1.0)
                    self.p[name] = self.be.asarray((rng.standard_normal(shp) * std).astype(self.dt))
        self._rope_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def _check_shapes(self):
        want = param_shapes(self.cfg)
        for k, shp in want.items():
            if k not in self.p or tuple(self.p[k].shape) != tuple(shp):
                raise ValueError(f"parameter '{k}' has shape {self.p.get(k).shape if k in self.p else None}, expected {shp}")

    # ---- sizes / copies ----------------------------------------------------------------------------------------
    def num_params(self) -> int:
        return int(sum(v.size for v in self.p.values()))

    def clone(self) -> "GPT":
        return GPT(self.cfg, params={k: v.copy() for k, v in self.p.items()}, backend=self.be)

    def load_params(self, params: Dict[str, np.ndarray]):
        for k, v in params.items():
            self.be.assign(self.p[k], v)

    def host_params(self) -> Dict[str, np.ndarray]:
        """The weights as NumPy arrays on the host (what checkpoints, exports and the CPU server read). On the CPU these ARE the arrays."""
        return {k: self.be.to_host(v) for k, v in self.p.items()}

    # ---- rotary tables -----------------------------------------------------------------------------------------
    def rope_tables(self, max_pos: int):
        n = max(int(max_pos), self.cfg.ctx + 1)
        t = self._rope_cache.get("t")
        if t is None or t[0].shape[0] < n:
            hd = self.cfg.head_dim
            inv = self.cfg.rope_theta ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
            ang = np.outer(np.arange(n, dtype=np.float64), inv)
            t = (self.be.asarray(np.cos(ang).astype(self.dt)), self.be.asarray(np.sin(ang).astype(self.dt)))
            self._rope_cache["t"] = t
        return t

    # ---- forward -----------------------------------------------------------------------------------------------
    def _forward(self, batch, keep: bool, dropout: float = 0.0, rng=None):
        c, p, xp, be = self.cfg, self.p, self.xp, self.be
        max_pos = max(int(np.max(be.to_host(batch["pos"]))) + 1, 1)             # decided on the host, before anything moves
        ids, pos, seg = be.asarray(batch["ids"]), be.asarray(batch["pos"]), be.asarray(batch["seg"])
        B, T = ids.shape
        D, H, hd = c.d_model, c.n_head, c.head_dim
        cos_t, sin_t = self.rope_tables(max_pos)
        cos, sin = cos_t[pos][:, None], sin_t[pos][:, None]                       # (B,1,T,hd/2)
        allowed = (seg[:, :, None] == seg[:, None, :]) & xp.tri(T, dtype=bool)[None]
        bias = xp.where(allowed, xp.asarray(0.0, dtype=self.dt), xp.asarray(NEG, dtype=self.dt))[:, None]   # (B,1,T,T)
        scale = 1.0 / math.sqrt(hd)
        x = p["tok_emb"][ids]
        layers = []
        aux_total = 0.0
        for l in range(c.n_layer):
            pre = f"b{l}."
            n1, xh1, inv1 = _rms(x, p[pre + "n1"], c.eps, xp)
            n1f = n1.reshape(-1, D)
            q = (n1f @ p[pre + "wq"]).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            k = (n1f @ p[pre + "wk"]).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            v = (n1f @ p[pre + "wv"]).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            q, k = _rope(q, cos, sin, xp), _rope(k, cos, sin, xp)
            s = xp.matmul(q, k.transpose(0, 1, 3, 2))
            s *= scale
            s += bias
            s -= s.max(axis=-1, keepdims=True)
            xp.exp(s, out=s)
            s /= s.sum(axis=-1, keepdims=True)
            P = s
            ao = xp.matmul(P, v)
            aof = ao.transpose(0, 2, 1, 3).reshape(B * T, D)
            att = (aof @ p[pre + "wo"]).reshape(B, T, D)
            m1 = None
            if dropout > 0.0:
                m1 = ((be.rand(rng, att.shape) >= dropout) / (1.0 - dropout)).astype(self.dt)
                att = att * m1
            x_mid = x + att
            n2, xh2, inv2 = _rms(x_mid, p[pre + "n2"], c.eps, xp)
            n2f = n2.reshape(-1, D)
            moe = None
            if c.n_experts > 1:
                mlp_f, moe, aux_l = _moe_forward(n2f, p, pre, c, xp, keep)
                aux_total += aux_l
                a = b = h = None
                mlp = mlp_f.reshape(B, T, D)
            else:
                a = n2f @ p[pre + "wg"]
                b = n2f @ p[pre + "wu"]
                sg = 1.0 / (1.0 + xp.exp(-a))
                h = (a * sg) * b
                mlp = (h @ p[pre + "wd"]).reshape(B, T, D)
            m2 = None
            if dropout > 0.0:
                m2 = ((be.rand(rng, mlp.shape) >= dropout) / (1.0 - dropout)).astype(self.dt)
                mlp = mlp * m2
            x_new = x_mid + mlp
            if keep:
                layers.append(dict(xh1=xh1, inv1=inv1, n1f=n1f, q=q, k=k, v=v, P=P, aof=aof, m1=m1,
                                   xh2=xh2, inv2=inv2, n2f=n2f, a=a, b=b, h=h, m2=m2, moe=moe))
            x = x_new
        xf, xhf, invf = _rms(x, p["nf"], c.eps, xp)
        ctx = dict(cos=cos, sin=sin, layers=layers, xhf=xhf, invf=invf, B=B, T=T, ids=ids, aux=aux_total)
        return xf, ctx

    # ---- loss head -----------------------------------------------------------------------------------------------
    def _head(self, xf, tgt, w, want_grad: bool):
        """Weighted NLL over the rows with w != 0, computed in chunks. Returns (sum w*nll, sum nll, n, dxf, dE_out)."""
        c, xp, be = self.cfg, self.xp, self.be
        D = c.d_model
        E = self.p["tok_emb"] if c.tie else self.p["lm_head"]
        xff = xf.reshape(-1, D)
        wf = be.asarray(w).reshape(-1)
        sel = xp.flatnonzero(wf != 0)
        dxf = xp.zeros_like(xff) if want_grad else None
        dE = xp.zeros_like(E) if want_grad else None
        if sel.size == 0:
            return 0.0, 0.0, 0, dxf, dE
        tg = be.asarray(tgt).reshape(-1)[sel]
        ww = wf[sel].astype(self.dt)
        xs = xff[sel]
        chunk = max(128, int((1 << 24) // c.vocab))
        loss_w = loss_u = 0.0
        for s in range(0, sel.size, chunk):
            xc = xs[s:s + chunk]
            lg = xc @ E.T
            m = lg.max(axis=-1, keepdims=True)
            lg -= m
            ex = xp.exp(lg)
            z = ex.sum(axis=-1, keepdims=True)
            idx = xp.arange(xc.shape[0])
            lp = lg[idx, tg[s:s + chunk]] - xp.log(z[:, 0])
            wc = ww[s:s + chunk]
            loss_w += float(-(wc.astype(np.float64) * lp).sum())
            loss_u += float(-lp.astype(np.float64).sum())
            if want_grad:
                ex /= z
                ex[idx, tg[s:s + chunk]] -= 1.0
                ex *= wc[:, None]
                dxf[sel[s:s + chunk]] = ex @ E
                dE += ex.T @ xc
        return loss_w, loss_u, int(sel.size), dxf, dE

    # ---- public: loss + gradients ----------------------------------------------------------------------------------
    def loss_and_grads(self, batch, dropout: float = 0.0, rng=None):
        """Returns (weighted loss, gradients, info). The gradients live on this model's device, ready for the optimiser."""
        c, p, xp = self.cfg, self.p, self.xp
        xf, ctx = self._forward(batch, keep=True, dropout=dropout, rng=rng)
        B, T = ctx["B"], ctx["T"]
        D, H, hd = c.d_model, c.n_head, c.head_dim
        loss_w, loss_u, n, dxf, dE_head = self._head(xf, batch["tgt"], batch["w"], True)
        loss_w += float(ctx["aux"])                                             # 6.4: the MoE load-balancing term (0.0 for a dense model); eval_nll never includes it
        grads = {k: xp.zeros_like(v) for k, v in p.items()}
        dx, grads["nf"] = _rms_back(dxf.reshape(B, T, D), p["nf"], ctx["xhf"], ctx["invf"], xp)
        scale = 1.0 / math.sqrt(hd)
        cos, sin = ctx["cos"], ctx["sin"]
        for l in reversed(range(c.n_layer)):
            pre = f"b{l}."
            ly = ctx["layers"][l]
            # ---- feed-forward branch
            dmlp = dx if ly["m2"] is None else dx * ly["m2"]
            dmf = dmlp.reshape(-1, D)
            n2f = ly["n2f"]
            if ly.get("moe") is not None:
                dn2 = _moe_backward(dmf, n2f, ly["moe"], p, pre, grads, c, xp)
            else:
                a, b, h = ly["a"], ly["b"], ly["h"]
                grads[pre + "wd"] += h.T @ dmf
                dh = dmf @ p[pre + "wd"].T
                sg = 1.0 / (1.0 + xp.exp(-a))
                silu = a * sg
                db = dh * silu
                da = dh * b * (sg * (1.0 + a * (1.0 - sg)))
                grads[pre + "wg"] += n2f.T @ da
                grads[pre + "wu"] += n2f.T @ db
                dn2 = da @ p[pre + "wg"].T + db @ p[pre + "wu"].T
            dxm, grads[pre + "n2"] = _rms_back(dn2.reshape(B, T, D), p[pre + "n2"], ly["xh2"], ly["inv2"], xp)
            dx_mid = dx + dxm
            # ---- attention branch
            datt = dx_mid if ly["m1"] is None else dx_mid * ly["m1"]
            dof = datt.reshape(-1, D)
            grads[pre + "wo"] += ly["aof"].T @ dof
            dao = (dof @ p[pre + "wo"].T).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            P, q, k, v = ly["P"], ly["q"], ly["k"], ly["v"]
            dP = xp.matmul(dao, v.transpose(0, 1, 3, 2))
            dv = xp.matmul(P.transpose(0, 1, 3, 2), dao)
            dS = P * (dP - (dP * P).sum(axis=-1, keepdims=True))
            dS *= scale
            dq = _rope_back(xp.matmul(dS, k), cos, sin, xp)
            dk = _rope_back(xp.matmul(dS.transpose(0, 1, 3, 2), q), cos, sin, xp)
            dqf = dq.transpose(0, 2, 1, 3).reshape(B * T, D)
            dkf = dk.transpose(0, 2, 1, 3).reshape(B * T, D)
            dvf = dv.transpose(0, 2, 1, 3).reshape(B * T, D)
            n1f = ly["n1f"]
            grads[pre + "wq"] += n1f.T @ dqf
            grads[pre + "wk"] += n1f.T @ dkf
            grads[pre + "wv"] += n1f.T @ dvf
            dn1 = dqf @ p[pre + "wq"].T + dkf @ p[pre + "wk"].T + dvf @ p[pre + "wv"].T
            dxa, grads[pre + "n1"] = _rms_back(dn1.reshape(B, T, D), p[pre + "n1"], ly["xh1"], ly["inv1"], xp)
            dx = dx_mid + dxa
        if c.tie:
            grads["tok_emb"] += dE_head
        else:
            grads["lm_head"] += dE_head
        self.be.scatter_rows(grads["tok_emb"], ctx["ids"].reshape(-1), dx.reshape(-1, D))
        return loss_w, grads, dict(nll_sum=loss_u, n=n)

    # ---- evaluation (no caches kept) ---------------------------------------------------------------------------------
    def eval_nll(self, batch) -> Tuple[float, int]:
        xf, _ = self._forward(batch, keep=False)
        _, loss_u, n, _, _ = self._head(xf, batch["tgt"], batch["w"], False)
        return loss_u, n

    def token_logprobs(self, batch) -> np.ndarray:
        """(B,T) log p(tgt) at positions with w != 0, else 0. Always a host array."""
        c, xp, be = self.cfg, self.xp, self.be
        xf, _ = self._forward(batch, keep=False)
        E = self.p["tok_emb"] if c.tie else self.p["lm_head"]
        B, T = batch["ids"].shape
        out = np.zeros(B * T, dtype=np.float64)
        wf = be.asarray(batch["w"]).reshape(-1)
        sel = xp.flatnonzero(wf != 0)
        xs = xf.reshape(-1, c.d_model)[sel]
        tg = be.asarray(batch["tgt"]).reshape(-1)[sel]
        chunk = max(128, int((1 << 24) // c.vocab))
        for s in range(0, sel.size, chunk):
            lg = xs[s:s + chunk] @ E.T
            lg -= lg.max(axis=-1, keepdims=True)
            lp = lg[xp.arange(lg.shape[0]), tg[s:s + chunk]] - xp.log(xp.exp(lg).sum(axis=-1))
            out[be.to_host(sel[s:s + chunk])] = be.to_host(lp)
        return out.reshape(B, T)

    # ---- persistence --------------------------------------------------------------------------------------------------
    def save(self, path: str, meta: Optional[dict] = None):
        arrays = {f"p.{k}": v for k, v in self.host_params().items()}
        info = dict(config=self.cfg.to_dict(), meta=meta or {})
        arrays["__meta__"] = np.frombuffer(json.dumps(info, default=fileio._json_default).encode("utf-8"), np.uint8)
        fileio.save_npz(path, arrays)

    @classmethod
    def load(cls, path: str, dtype: Optional[str] = None, backend: Optional[bk.Backend] = None) -> Tuple["GPT", dict]:
        w = fileio.load_npz(path)
        info = json.loads(bytes(w["__meta__"]).decode("utf-8"))
        cfg = GPTConfig.from_dict(info["config"])
        if dtype:
            cfg.dtype = dtype
        params = {k[2:]: v for k, v in w.items() if k.startswith("p.")}
        return cls(cfg, params=params, backend=backend), info.get("meta", {})


def read_meta(path: str) -> dict:
    """Config + meta of a weights file without building the model."""
    w = fileio.load_npz(path)
    return json.loads(bytes(w["__meta__"]).decode("utf-8"))


def activation_bytes_per_token(c: GPTConfig, T: int) -> float:
    """Rough training-memory cost per token: saved activations of every layer + attention probabilities + head chunk."""
    bpe = np.dtype(c.dtype).itemsize
    per_layer = (14 * c.d_model + 3 * c.ffn) * bpe + c.n_head * T * bpe * 1.5
    return per_layer * c.n_layer + 6 * c.d_model * bpe
