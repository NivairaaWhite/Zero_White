"""autoscale.py - turn every "auto" in the configuration into a number, from THIS machine and THIS data.

Nothing here is a fixed model size. The decisions, in order (every one is written with its reason to
state/resolved_config.json, and explicit values in your config always win):

  1. tokenizer vocabulary   from the amount of text (derive_vocab_size)
  2. context window         from the length distribution of your conversations (percentile, rounded, clamped)
  3. throughput             measured by running the real model for a couple of steps
  4. model shape            the largest model whose *compute-optimal* token count fits both the time budget and the data
                            (tokens_per_param, max_epochs): small data / short budget -> small model, more data / time -> bigger
  5. batch / schedule       micro-batch from free RAM, effective batch from the step target, LR from width, warm-up, eval cadence
When a model already exists its shape is read from the checkpoint instead; only the schedule is re-derived.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .. import fileio
from . import backend as bk
from . import hardware as hw
from .config import AUTO, Config
from .model import GPT, GPTConfig, activation_bytes_per_token, count_params, round_up
from .paths import Layout
from .tokenizer import ChatTemplate, Tokenizer


# ---------------------------------------------------------------------------------------------------------------------
# sampling the corpus
# ---------------------------------------------------------------------------------------------------------------------
def record_text(rec: dict) -> str:
    """All the text of a normalised record (used for tokenizer training and size estimates)."""
    parts: List[str] = []
    if rec.get("system"):
        parts.append(rec["system"])
    if rec["kind"] == "text":
        parts.append(rec["text"])
    else:
        parts.extend(t["content"] for t in rec["turns"])
        if rec["kind"] == "preference":
            parts.extend((rec["chosen"], rec["rejected"]))
    return "\n".join(parts)


def estimate_corpus(ing, sample_per_shard: int) -> dict:
    """Records and UTF-8 bytes of text, estimated from the first `sample_per_shard` records of every shard."""
    total_rec, total_bytes = 0, 0.0
    for s in ing.shards():
        n, b = 0, 0
        for rec in ing.iter_shard(s):
            b += len(record_text(rec).encode("utf-8"))
            n += 1
            if n >= sample_per_shard:
                break
        if n:
            total_rec += s.records
            total_bytes += b / n * s.records
    return dict(records=total_rec, bytes=int(total_bytes))


def iter_tokenizer_text(ing, cfg: Config, est_bytes: int) -> Iterator[str]:
    """Text to train the BPE on: a deterministic hash-sample of the corpus capped at tokenizer.train_sample_mb."""
    import hashlib
    cap = float(cfg.get("tokenizer.train_sample_mb")) * 1e6
    p = 1.0 if est_bytes <= cap else cap / max(1, est_bytes)
    sp = cfg.system_prompt()
    if sp:
        yield sp
    for greet in (cfg.get("persona.greeting"), cfg.get("persona.name"), cfg.get("persona.user_name")):
        if greet:
            yield greet
    for rec in ing.iter_records():
        if p < 1.0:
            u = int.from_bytes(hashlib.blake2b(rec["id"].encode("utf-8"), digest_size=8).digest(), "big") / 2.0 ** 64
            if u >= p:
                continue
        yield record_text(rec)


def sample_lengths(ing, tok: Tokenizer, tpl: ChatTemplate, n: int) -> dict:
    """Token length of whole conversations (before any window splitting) from an evenly spread sample."""
    shards = ing.shards()
    total = sum(max(0, s.records) for s in shards)
    if not total:
        return dict(n=0, lengths=[], mean=0.0, total_tokens=0, total_records=0)
    stride = max(1, total // max(1, n))
    lengths: List[int] = []
    i = 0
    for s in shards:
        for rec in ing.iter_shard(s):
            if i % stride == 0:
                if rec["kind"] == "text":
                    lengths.append(len(tok.encode(rec["text"])) + 1)
                else:
                    w = tpl.chunk_chat(rec.get("system"), rec["turns"], 1 << 30)
                    if w:
                        lengths.append(len(w[0][0]))
            i += 1
    mean = float(np.mean(lengths)) if lengths else 0.0
    return dict(n=len(lengths), lengths=lengths, mean=mean, total_tokens=int(mean * total), total_records=total)


# ---------------------------------------------------------------------------------------------------------------------
# model shape + measurements
# ---------------------------------------------------------------------------------------------------------------------
_NUMPY_EFFICIENCY = 0.25        # only used when benchmarking is switched off (tests); real runs measure the model itself
_GPU_EFFICIENCY = 0.2           # the same, for a GPU: an unmeasured GUESS (the model is not fused, so it reaches a smaller share of the card's matmul peak)
def shape_for(cfg: Config, n_nonemb: float, vocab: int, ctx: int) -> GPTConfig:
    """Shape with about n_nonemb non-embedding parameters (12*L*d^2 for a SwiGLU block), honouring explicit model.* values."""
    hd = int(cfg.get("model.head_dim"))
    ar = max(1, int(cfg.get("model.aspect_ratio")))
    d, L, H = cfg.raw("model.d_model"), cfg.raw("model.n_layer"), cfg.raw("model.n_head")          # AUTO unless the user set them
    if d == AUTO:
        if L != AUTO:
            d = round_up(math.sqrt(n_nonemb / (12.0 * int(L))), hd)
        else:
            d = round_up((n_nonemb * ar / 12.0) ** (1.0 / 3.0), hd)
        d = max(2 * hd, d)
    d = int(d)
    if L == AUTO:
        L = max(2, int(round(d / ar)))
    if H == AUTO:
        H = max(1, d // hd)
        while d % H or (d // H) % 2:
            H -= 1
    ffn = round_up(float(cfg.get("model.ffn_mult")) * d, 8)
    return GPTConfig(vocab=int(vocab), d_model=d, n_layer=int(L), n_head=int(H), ffn=ffn, ctx=int(ctx),
                     rope_theta=float(cfg.get("model.rope_theta")), eps=float(cfg.get("model.norm_eps")),
                     tie=bool(cfg.get("model.tie_embeddings")), dtype=str(cfg.get("model.dtype")))


def flops_per_token(c: GPTConfig, T: int) -> float:
    """Training FLOPs per token: 6 x (block + output-head parameters) + attention (QK^T and PV, forward+backward)."""
    _, nonemb = count_params(c)
    return 6.0 * (nonemb + c.vocab * c.d_model) + 12.0 * c.n_layer * T * c.d_model


def bench(c: GPTConfig, rows: int, T: int, steps: int = 2, backend: Optional[bk.Backend] = None) -> float:
    """Measured training tokens/second of the real model at this shape (random tokens; one warm-up step first).
    On a GPU the device is synchronised before the clock starts and again before it stops: kernels are queued asynchronously, so an
    unsynchronised timer would report how fast Python can enqueue work, not how fast the card computes it."""
    be = backend or bk.CPU
    rng = np.random.default_rng(0)
    m = GPT(c, seed=0, backend=be)
    ids = rng.integers(0, c.vocab, size=(rows, T)).astype(np.int64)
    batch = dict(ids=ids, tgt=np.roll(ids, -1, axis=1), w=np.ones((rows, T), np.float32),
                 pos=np.tile(np.arange(T), (rows, 1)).astype(np.int64), seg=np.ones((rows, T), np.int64))
    m.loss_and_grads(batch)
    be.sync()
    t = time.perf_counter()
    for _ in range(max(1, steps)):
        m.loss_and_grads(batch)
    be.sync()
    tps = rows * T * max(1, steps) / max(time.perf_counter() - t, 1e-9)
    del m
    be.release()
    return tps


def estimate_tps(c: GPTConfig, T: int, backend: Optional[bk.Backend] = None) -> float:
    """Throughput estimate without running the model: measured matmul speed x a typical share of it this implementation reaches."""
    be = backend or bk.CPU
    eff = _GPU_EFFICIENCY if be.is_gpu else _NUMPY_EFFICIENCY
    return be.matmul_gflops() * 1e9 * eff / flops_per_token(c, T)


def train_bytes(c: GPTConfig, rows: int, T: int) -> float:
    total, _ = count_params(c)
    bpe = np.dtype(c.dtype).itemsize
    return total * bpe * 4 + rows * T * activation_bytes_per_token(c, T)          # weights + grads + Adam m,v + activations


# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Plan:
    values: Dict[str, Tuple[object, str]] = field(default_factory=dict)
    model: Optional[GPTConfig] = None
    info: Dict[str, object] = field(default_factory=dict)

    def set(self, key: str, value, why: str):
        self.values[key] = (value, why)

    def to_json(self) -> dict:
        return dict(derived={k: dict(value=v, reason=w) for k, (v, w) in self.values.items()},
                    model=self.model.to_dict() if self.model else None, info=self.info)


def _explicit(cfg: Config, key: str) -> bool:
    return not cfg.is_auto(key) and cfg.raw(key) != AUTO


def pick_ctx(cfg: Config, stats: dict, model_cfg: Optional[GPTConfig] = None) -> Tuple[int, str]:
    """Context window and the reason for it."""
    if model_cfg is not None:
        return model_cfg.ctx, "taken from the existing model"
    if _explicit(cfg, "model.ctx"):
        return int(cfg.get("model.ctx")), "set in config"
    lo, hi = int(cfg.get("autoscale.ctx_min")), int(cfg.get("autoscale.ctx_max"))
    pct = float(cfg.get("autoscale.ctx_percentile"))
    if stats["lengths"]:
        p = float(np.percentile(stats["lengths"], pct))
        return (int(min(hi, max(lo, round_up(p, 64)))),
                f"p{pct:g} of {stats['n']:,} sampled conversation lengths = {p:,.0f} tokens, rounded to 64 and clamped to [{lo}, {hi}]")
    return lo, "no data yet: autoscale.ctx_min"


def _default_unless_set(cfg: Config, key: str, gpu_key: str, gpu: bool):
    """The value of `key`; on a GPU, the GPU counterpart `gpu_key` instead, but only while `key` still has its CPU-sized default:
    something the user set explicitly always wins."""
    if gpu and cfg.source_of(key) == "default":
        return cfg.get(gpu_key)
    return cfg.get(key)


def resolve(cfg: Config, lay: Layout, tok: Tokenizer, stats: dict, model_cfg: Optional[GPTConfig] = None,
            val_tokens: Optional[int] = None, do_bench: bool = True, backend: Optional[bk.Backend] = None) -> Tuple[Config, Plan]:
    """Fill in every "auto" and return (config with derived values, plan). `stats` comes from sample_lengths().
    `model_cfg` is given when a model already exists (its shape is then fixed). `backend` is where training will run (default CPU):
    on a GPU the benchmark runs there, memory is planned from the card's free VRAM, and the CPU-sized caps give way to compute.*."""
    plan = Plan()
    be = backend or bk.CPU
    gpu = be.is_gpu
    plan.info["backend"] = be.label()
    if be.note:
        plan.info["backend_note"] = be.note
    ncpu = hw.cpu_count()

    ctx, why = pick_ctx(cfg, stats, model_cfg)
    plan.set("model.ctx", ctx, why)
    mean_len = max(1.0, stats["mean"] or float(ctx) / 2)

    # ---- data size in training tokens (windows add the system header again, hence the small inflation)
    data_tokens = int(stats["total_tokens"] * 1.05) if stats["total_tokens"] else 0
    train_tokens = int(data_tokens * (1.0 - float(cfg.get("data.val_fraction")))) if data_tokens else 0
    budget_min = cfg.get("train.budget_minutes")
    budget_s = float(budget_min) * 60.0 if budget_min else None

    # ---- model shape
    tpp = float(cfg.get("autoscale.tokens_per_param"))
    max_ep = float(cfg.get("autoscale.max_epochs"))
    if model_cfg is not None:
        mc = GPTConfig.from_dict(dict(model_cfg.to_dict(), ctx=ctx))
        why_shape = "existing model"
        tps_probe = None
    else:
        # measure the machine with a reference model, then sweep sizes analytically
        ref = shape_for(cfg.replace(**{"model.d_model": 128, "model.n_layer": 3, "model.n_head": 4}), 0, tok.vocab_size, ctx)
        rows_ref = max(1, min(8, 4096 // ctx)) if not gpu else max(1, min(32, 16384 // ctx))        # a card needs a bigger batch to show its speed
        tps_probe = bench(ref, rows_ref, ctx, int(cfg.get("autoscale.bench_steps")), be) if do_bench else estimate_tps(ref, ctx, be)
        F = tps_probe * flops_per_token(ref, ctx)                                   # achieved FLOP/s of this machine
        plan.info["probe_tokens_per_s"] = round(tps_probe, 1)
        plan.info["probe_gflops"] = round(F / 1e9, 2)
        lo, hi = float(cfg.get("autoscale.min_params")), float(_default_unless_set(cfg, "autoscale.max_params", "compute.max_params", gpu))
        data_cap = train_tokens * max_ep if train_tokens else float("inf")
        best, n = None, lo
        if all(_explicit(cfg, k) for k in ("model.d_model", "model.n_layer")) or _explicit(cfg, "model.target_params"):
            n = float(cfg.get("model.target_params")) if _explicit(cfg, "model.target_params") else lo
            best, grid = shape_for(cfg, n, tok.vocab_size, ctx), []
            why_shape = "model.* set in config"
        else:
            grid = []
            while n <= hi:
                grid.append(n)
                n *= 1.15
            why_shape = ""
        for n in grid:
            c = shape_for(cfg, n, tok.vocab_size, ctx)
            _, nonemb = count_params(c)
            afford = (F * budget_s / flops_per_token(c, ctx)) if budget_s else float("inf")
            if tpp * nonemb <= min(afford, data_cap):
                best = c
                how = ("data" if data_cap < afford else "time budget")
        if best is None:
            best = shape_for(cfg, lo, tok.vocab_size, ctx)
            why_shape = f"autoscale.min_params: even the smallest model wants {tpp:g} tokens/param and the budget/data cannot supply that"
        elif not why_shape:
            _, nonemb = count_params(best)
            why_shape = (f"largest model whose {tpp:g} tokens/param fits the {how}: {nonemb:,} non-embedding params; "
                         f"~{F / 1e9:.1f} GFLOP/s measured")
        mc = best
        # the real throughput at the chosen shape (the sweep used the reference shape's efficiency)
    total_p, nonemb_p = count_params(mc)
    plan.model = mc
    if model_cfg is None:
        for key, val in (("model.d_model", mc.d_model), ("model.n_layer", mc.n_layer), ("model.n_head", mc.n_head)):
            if cfg.is_auto(key):
                plan.set(key, val, why_shape)
        if cfg.is_auto("model.target_params"):
            plan.set("model.target_params", nonemb_p, why_shape)
    else:
        for key, val in (("model.d_model", mc.d_model), ("model.n_layer", mc.n_layer), ("model.n_head", mc.n_head),
                         ("model.target_params", nonemb_p)):
            if cfg.is_auto(key):
                plan.set(key, val, why_shape)
    plan.info.update(params_total=total_p, params_non_embedding=nonemb_p, vocab=mc.vocab, ctx=mc.ctx,
                     data_tokens=data_tokens, train_tokens=train_tokens, mean_conversation_tokens=round(mean_len, 1))

    # ---- micro batch from memory (RAM, or the card's free VRAM) and a measured speed at the chosen shape
    if gpu:
        avail = be.free_memory() or be.total_memory() or (4 << 30)
        ram_budget = float(cfg.get("compute.vram_fraction")) * avail
        mem_name, mem_knob = "VRAM", "compute.vram_fraction x free VRAM"
        bpe = np.dtype(mc.dtype).itemsize

        def need(r: int) -> float:                  # one more copy of the weights (gradient accumulation) and 15% for allocator fragmentation
            return train_bytes(mc, r, ctx) * 1.15 + total_p * bpe
        rows_cap = 256
    else:
        avail = hw.ram_available_bytes() or hw.ram_bytes() or (4 << 30)
        ram_budget = float(cfg.get("train.ram_fraction")) * avail
        mem_name, mem_knob = "RAM", "train.ram_fraction x free RAM"

        def need(r: int) -> float:
            return train_bytes(mc, r, ctx)
        rows_cap = 64
    max_micro = int(_default_unless_set(cfg, "autoscale.max_micro_tokens", "compute.max_micro_tokens", gpu))
    if train_tokens:
        max_micro = min(max_micro, max(ctx, train_tokens // max(1, int(cfg.get("autoscale.min_batches_per_epoch")))))
    rows = max(1, min(max_micro // ctx, rows_cap))
    while rows > 1 and need(rows) > ram_budget:
        rows -= 1
    if need(rows) > ram_budget:
        plan.info["warning"] = (f"a {total_p:,}-parameter model needs about {need(rows) / 2 ** 30:.1f} GB to train but only "
                                f"{ram_budget / 2 ** 30:.1f} GB of {mem_name} is allowed ({mem_knob}); lower model.target_params or free memory")
    micro = rows * ctx
    if cfg.is_auto("train.batch_tokens"):
        plan.set("train.batch_tokens", micro, f"{rows} rows x {ctx}: the most that fits {ram_budget / 2 ** 30:.1f} GB ({mem_knob}), capped by the largest micro-batch setting and by the data size")
    else:
        micro = int(cfg.get("train.batch_tokens"))
        rows = max(1, micro // ctx)
        micro = rows * ctx

    if do_bench:
        tps = bench(mc, rows, ctx, int(cfg.get("autoscale.bench_steps")), be)
    else:
        tps = estimate_tps(mc, ctx, be)
    plan.info["tokens_per_s"] = round(tps, 1)

    # ---- how many tokens this run will see
    max_steps, epochs = cfg.get("train.max_steps"), cfg.get("train.epochs")
    inf = float("inf")
    if epochs and train_tokens:
        data_run = float(epochs) * train_tokens                         # asked for explicitly
    elif max_steps:
        data_run = inf                                                    # an explicit step count overrides the epoch guard
    else:
        data_run = max_ep * train_tokens if train_tokens else inf         # never repeat the corpus more than max_epochs times
    time_run = tps * budget_s if budget_s else inf
    run_tokens = min(data_run, time_run)
    if run_tokens == inf:
        run_tokens = float(micro) * int(max_steps or cfg.get("autoscale.target_steps"))
    run_tokens = max(run_tokens, float(micro))

    eff_cfg = cfg.get("train.effective_batch_tokens", AUTO)
    if eff_cfg == AUTO:
        eff = int(min(int(cfg.get("autoscale.max_effective_tokens")), max(micro, run_tokens / max(1, int(cfg.get("autoscale.target_steps"))))))
        eff = max(micro, int(round(eff / micro)) * micro)
        plan.set("train.effective_batch_tokens", eff,
                 f"{run_tokens / 1e6:.2f}M tokens in this run / autoscale.target_steps={cfg.get('autoscale.target_steps')} steps, in whole micro-batches of {micro}")
    else:
        eff = max(micro, int(round(int(eff_cfg) / micro)) * micro)
    steps = int(max(1, run_tokens // eff))
    if max_steps:
        steps = min(steps, int(max_steps))
    plan.info["planned_steps"] = steps
    plan.info["accumulation"] = eff // micro
    if steps < 50 and not max_steps and not epochs:
        plan.info.setdefault("notice", f"only {train_tokens:,} training tokens: the run will have {steps} step(s). Add more data to the data folder, "
                                      f"or set train.epochs / train.max_steps to train longer on what you have.")

    if cfg.is_auto("train.lr"):
        ref_w = int(cfg.get("autoscale.lr_ref_width"))
        lr = float(cfg.get("autoscale.lr_ref")) * math.sqrt(ref_w / mc.d_model)
        lr = float(min(6e-3, max(2e-4, lr)))
        plan.set("train.lr", round(lr, 6), f"autoscale.lr_ref {cfg.get('autoscale.lr_ref')} x sqrt({ref_w}/{mc.d_model})")
    if cfg.is_auto("train.warmup_steps"):
        wf = float(cfg.get("autoscale.warmup_fraction"))
        plan.set("train.warmup_steps", max(1, int(round(wf * steps))), f"{wf:.0%} of {steps} planned steps")
    passes = max(1, int(cfg.get("autoscale.eval_passes")))
    if cfg.is_auto("train.eval_every"):
        plan.set("train.eval_every", max(1, steps // passes), f"about {passes} validation passes over {steps} planned steps")
    if cfg.is_auto("train.eval_tokens"):
        want = int(max(2 * micro, float(cfg.get("autoscale.eval_overhead")) * run_tokens / passes * 3.0))
        vt = val_tokens if val_tokens else want
        et = int(max(1, min(vt, want)))
        plan.set("train.eval_tokens", et, f"validation may take about {float(cfg.get('autoscale.eval_overhead')):.0%} of the run (capped by the {vt:,} validation tokens)")
    if cfg.is_auto("train.threads"):
        plan.set("train.threads", ncpu, f"all {ncpu} CPUs")

    # ---- the rest
    if cfg.is_auto("tokenizer.vocab_size"):
        plan.set("tokenizer.vocab_size", tok.vocab_size, "from the existing tokenizer")
    if cfg.is_auto("generation.max_new_tokens"):
        plan.set("generation.max_new_tokens", max(16, ctx // 4), "a quarter of the context window")
    if cfg.is_auto("data.workers"):
        plan.set("data.workers", max(1, ncpu - 1), f"CPUs - 1 = {max(1, ncpu - 1)}")
    if cfg.is_auto("online.threads"):
        plan.set("online.threads", max(1, ncpu // 2), f"half of {ncpu} CPUs")
    if cfg.is_auto("online.lr"):
        peak = plan.values.get("train.lr", (cfg.get("train.lr", None) if not cfg.is_auto("train.lr") else None, ""))[0]
        if peak:
            plan.set("online.lr", round(float(peak) * float(cfg.get("online.lr_ratio")), 7), f"online.lr_ratio {cfg.get('online.lr_ratio')} x training LR {peak}")
    if cfg.is_auto("daemon.train_min_new_tokens"):
        v = max(20000, int(0.02 * data_tokens))
        plan.set("daemon.train_min_new_tokens", v, "2% of the corpus, at least 20,000 tokens")
    return cfg.with_derived(plan.values), plan


def save_plan(lay: Layout, plan: Plan, extra: Optional[dict] = None):
    d = plan.to_json()
    d["hardware"] = hw.describe()
    d["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if extra:
        d.update(extra)
    fileio.write_json(lay.resolved_config_path, d, indent=2)
