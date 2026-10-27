"""prepare.py - everything between "files in the data folder" and "ready to train", as one idempotent step:

    ingest data folder -> (train tokenizer if there is none) -> choose context -> tokenise shards -> derive config

Every caller (CLI, trainer, daemon, online learner) goes through here, so they all see the same corpus and the same
derived numbers. Running it twice in a row does no work the second time.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

from . import autoscale as AS
from . import backend as bk
from .config import Config
from .corpus.store import Ingestor, IngestReport
from .model import GPTConfig, read_meta
from .paths import Layout
from .tokenizer import ChatTemplate, Tokenizer, TokenizerError, derive_vocab_size
from .tokenstore import Corpus, TokenStore


class NoDataError(RuntimeError):
    """The data folder has nothing usable in it (yet)."""


@dataclass
class Prepared:
    cfg: Config                    # with derived values
    lay: Layout
    ing: Ingestor
    tok: Tokenizer
    tpl: ChatTemplate
    store: TokenStore
    corpus: Corpus
    plan: AS.Plan
    ingest: IngestReport
    model_cfg: Optional[GPTConfig]   # config of the existing live model, if any
    backend: Optional[bk.Backend] = None   # where training will run (None = the CPU)


def no_data_message(lay: Layout, rep: Optional[IngestReport] = None) -> str:
    msg = [f"There is no usable training data in {lay.data}."]
    if rep is not None and rep.files_unsettled:
        msg.append(f"{rep.files_unsettled} file(s) there changed a moment ago and are still being written or copied. Wait for the copy to finish and run the command "
                   f"again (`data.settle_wait_s` sets how long it waits by itself; `data.settle_s` how long a file must be unchanged).")
    if rep is not None and rep.errors:
        msg.append("Files that could not be read: " + "; ".join(f"{r}: {e}" for r, e in rep.errors[:5]))
    if rep is not None and rep.dropped:
        msg.append("Records dropped by filters: " + ", ".join(f"{k}={v:,}" for k, v in rep.dropped.items()))
    msg.append("Put dataset files there (json / jsonl / csv / tsv / parquet / txt / md / py ..., optionally .gz or .zip, in any sub-folders), "
               "or download one:  zerowhite chat fetch alpaca   (see `zerowhite chat fetch --list`). Then run the command again.")
    return "\n".join(msg)


def build_tokenizer(cfg: Config, lay: Layout, ing: Ingestor, progress: Optional[Callable[[str], None]] = None) -> Tokenizer:
    est = AS.estimate_corpus(ing, int(cfg.get("autoscale.sample_records")))
    specials = ChatTemplate.special_tokens(cfg)
    if cfg.is_auto("tokenizer.vocab_size") or cfg.raw("tokenizer.vocab_size") == "auto":
        vocab, why = derive_vocab_size(est["bytes"], len(specials))
    else:
        vocab, why = int(cfg.get("tokenizer.vocab_size")), "set in config"
    if progress:
        progress(f"training tokenizer: vocab {vocab} ({why}) on ~{est['bytes'] / 1e6:.1f} MB of text")
    last = {"n": -1}

    def tick(done: int, total: int):
        if progress and done != last["n"] and (done == total or done % 1000 == 0):
            last["n"] = done
            progress(f"  tokenizer merges {done}/{total}")

    tok = Tokenizer.train(AS.iter_tokenizer_text(ing, cfg, est["bytes"]), vocab, specials,
                          min_frequency=int(cfg.get("tokenizer.min_frequency")), progress=tick)
    os.makedirs(os.path.dirname(lay.tokenizer_path), exist_ok=True)
    tok.save(lay.tokenizer_path)
    return tok


def load_tokenizer(cfg: Config, lay: Layout) -> Optional[Tokenizer]:
    if not os.path.exists(lay.tokenizer_path):
        return None
    try:
        return Tokenizer.load(lay.tokenizer_path)
    except (OSError, ValueError, KeyError) as e:
        raise TokenizerError(f"cannot read the tokenizer {lay.tokenizer_path}: {e}. Delete it to train a new one "
                             f"(this also means retraining the model, whose embeddings belong to it).") from e


def existing_model_config(lay: Layout) -> Optional[GPTConfig]:
    p = lay.current_weights()
    if not p:
        return None
    return GPTConfig.from_dict(read_meta(p)["config"])


def prepare(cfg: Config, lay: Optional[Layout] = None, progress: Optional[Callable[[str], None]] = None,
            do_bench: bool = True, ingest: bool = True, force_ingest: bool = False, tokenize: bool = True,
            backend: Optional[bk.Backend] = None) -> Prepared:
    """Ingest, tokenise and plan a training run. `backend` is where it will run; by default `compute.backend` decides (cpu / gpu / auto)."""
    from .tokenstore import TokenStore
    lay = (lay or Layout(cfg)).ensure()
    say = progress or (lambda m: None)
    ing = Ingestor(cfg, lay)
    # 6.5: the data folder is read BEFORE the compute device is touched. Starting CuPy / CUDA takes seconds on a GPU machine; done first (as in 6.4) it ate the
    # settle window, so on Colab a file "copied a moment ago" had already settled by the time it was looked at.
    if ingest:
        rep = ing.ingest(progress=lambda ev: say(_fmt_ingest_event(ev)) if isinstance(ev, dict) else None, force=force_ingest)
        say("data folder: " + rep.summary())
    else:
        rep = IngestReport()
    budget, waited = float(cfg.get("data.settle_wait_s")), 0.0
    while not ing.shards() and ingest and rep.files_unsettled and waited < budget:        # files copied a moment ago: wait exactly as long as they need, up to the budget
        pause = min(budget - waited, max(0.1, rep.unsettled_wait_s) + 0.25)
        say(f"waiting {pause:.1f}s for {rep.files_unsettled} file(s) that are still being copied ...")
        time.sleep(pause)
        waited += pause
        rep = ing.ingest(progress=lambda ev: say(_fmt_ingest_event(ev)) if isinstance(ev, dict) else None)
        say("data folder: " + rep.summary())
    if not ing.shards():
        raise NoDataError(no_data_message(lay, rep))
    be = backend or bk.from_config(cfg)

    tok = load_tokenizer(cfg, lay)
    if tok is None:
        tok = build_tokenizer(cfg, lay, ing, say)
    try:
        tpl = ChatTemplate(tok, cfg)
    except KeyError as e:
        raise TokenizerError(f"the tokenizer does not contain the special token {e}; the template.* settings changed after it was "
                             f"trained. Restore them, or delete {lay.tokenizer_path} and retrain.") from None

    model_cfg = existing_model_config(lay) if cfg.get("train.resume") else None
    if model_cfg is not None and model_cfg.vocab != tok.vocab_size:
        raise TokenizerError(f"the live model has vocab {model_cfg.vocab} but the tokenizer has {tok.vocab_size}; they do not belong "
                             f"together. Use the tokenizer the model was trained with, or set train.resume=false to start a new model.")

    stats = AS.sample_lengths(ing, tok, tpl, int(cfg.get("autoscale.sample_records")))
    ctx, _ = AS.pick_ctx(cfg, stats, model_cfg)
    store = TokenStore(cfg, lay, tok, ctx)
    workers = cfg.get("data.workers", 1)
    if tokenize:
        r = store.sync(ing.shards(), workers=int(workers), progress=lambda m: say(f"tokenised {os.path.basename(m['dst'])}: {m['tokens']:,} tokens"))
        if r["built"]:
            say(f"token store: {r['built']} shard(s) built, {r['tokens']:,} tokens")
    last = _last_trained(lay)
    corpus = store.load(ing.shards(), since=last, new_boost=float(cfg.get("train.continual.new_data_weight")) if last else 1.0)
    cnt = corpus.counts()
    cfg2, plan = AS.resolve(cfg, lay, tok, stats, model_cfg, val_tokens=cnt["val_tokens"], do_bench=do_bench, backend=be)
    AS.save_plan(lay, plan, dict(corpus=cnt, ingest=rep.as_dict() if rep.changed_anything else None, compute=be.describe()))
    return Prepared(cfg2, lay, ing, tok, tpl, store, corpus, plan, rep, model_cfg, be)


def resolve_light(cfg: Config, lay: Layout, model_cfg: Optional[GPTConfig] = None) -> Config:
    """Fill the "auto" values using the existing tokenizer/model, without benchmarking, tokenising or ingesting anything.
    Used by the daemon, the chat commands and the online learner; returns `cfg` unchanged when there is no tokenizer yet."""
    tok = load_tokenizer(cfg, lay)
    if tok is None:
        return cfg
    ing = Ingestor(cfg, lay)
    try:
        tpl = ChatTemplate(tok, cfg)
    except KeyError:
        return cfg
    stats = AS.sample_lengths(ing, tok, tpl, int(cfg.get("autoscale.sample_records")))
    if model_cfg is None:
        model_cfg = existing_model_config(lay)
    cfg2, _ = AS.resolve(cfg, lay, tok, stats, model_cfg, do_bench=False)
    return cfg2


def load_corpus(cfg: Config, lay: Layout, tok: Tokenizer, ctx: int) -> Optional[Corpus]:
    """The already-tokenised corpus for a model with context `ctx` (no ingestion, no work); None when there is none."""
    shards = Ingestor(cfg, lay).shards()
    if not shards:
        return None
    corpus = TokenStore(cfg, lay, tok, ctx).load(shards)
    return corpus if len(corpus.train_idx) else None


def _last_trained(lay: Layout) -> Optional[float]:
    cur = lay.current()
    return cur.get("data_cutoff") if cur else None


def _fmt_ingest_event(ev: dict) -> str:
    kind = ev.get("event")
    if kind == "file":
        return f"ingested {ev.get('path')} ({ev.get('status')}): {ev.get('records', 0):,} records"
    if kind == "removed":
        return f"forgot removed file {ev.get('path')}"
    return " ".join(f"{k}={v}" for k, v in ev.items())
