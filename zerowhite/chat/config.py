"""config.py - one place where every tunable lives.

Nothing in the chat stack carries a magic number or a path of its own: every knob is declared once in SCHEMA (with a
type, a documented default and, where it makes sense, the value "auto" meaning *derive it from this machine and this
data*, see autoscale.py). Values are layered, later layers win:

    schema defaults  <  config file (JSON with // comments, TOML or YAML)  <  environment (ZW_TRAIN__LR=0.002)
                     <  command line (--set train.lr=0.002)  <  per-folder dataset.json (data/filter keys only)

`zerowhite chat config --reference` writes the full documented list of options straight from the schema, so the
documentation can never drift from the code.
"""
from __future__ import annotations

import copy
import difflib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .. import fileio

AUTO = "auto"


class ConfigError(ValueError):
    pass


class NotResolved(ConfigError):
    pass


class Opt:
    """One option: dotted key, default, kind ('int', 'float', 'bool', 'str', 'path', 'list', 'dict', optionally
    suffixed |auto and/or |null), a one-line doc and optional allowed choices."""
    __slots__ = ("key", "default", "kind", "doc", "choices")

    def __init__(self, key, default, kind, doc, choices=()):
        self.key, self.default, self.kind, self.doc, self.choices = key, default, kind, doc, tuple(choices)

    @property
    def base(self):
        return self.kind.split("|")[0]


def _o(key, default, kind, doc, choices=()):
    return Opt(key, default, kind, doc, choices)


# --------------------------------------------------------------------------------------------------------------------
# THE SCHEMA. Order = order in the generated reference.
# --------------------------------------------------------------------------------------------------------------------
SCHEMA: List[Opt] = [
    # ---- where things live ------------------------------------------------------------------------------------
    _o("paths.data", None, "path|null", "Folder you drop datasets into (default <workspace>/data). Scanned recursively; new files are picked up automatically."),
    _o("paths.models", None, "path|null", "Checkpoints, tokenizer and the live model pointer (default <workspace>/models)."),
    _o("paths.state", None, "path|null", "Manifests, experience log, corpus shards and token shards (default <workspace>/state)."),
    _o("paths.logs", None, "path|null", "Log files (default <workspace>/logs)."),
    _o("paths.run", None, "path|null", "Runtime files: pid file, server address, stop file (default <workspace>/run)."),
    _o("paths.exports", None, "path|null", "Where `export` and `convert` write by default (default <workspace>/exports)."),

    # ---- who the assistant is -----------------------------------------------------------------------------------
    _o("persona.name", None, "str|null", "The assistant's name; available to the system prompt as {name}."),
    _o("persona.user_name", None, "str|null", "What the assistant calls you; available to the system prompt as {user}."),
    _o("persona.system_prompt", None, "str|null", "System prompt text placed at the start of every conversation ({name} and {user} are substituted)."),
    _o("persona.system_prompt_file", None, "path|null", "Read the system prompt from this file instead (relative paths are resolved against the workspace)."),
    _o("persona.greeting", None, "str|null", "Optional first message the assistant shows/speaks when a session starts."),

    # ---- the chat template ------------------------------------------------------------------------------------
    _o("template.system_token", "<|system|>", "str", "Special token that opens a system message."),
    _o("template.user_token", "<|user|>", "str", "Special token that opens a user message."),
    _o("template.assistant_token", "<|assistant|>", "str", "Special token that opens an assistant message."),
    _o("template.end_token", "<|end|>", "str", "Special token that closes every message (the model learns to emit it to stop)."),
    _o("template.pad_token", "<|pad|>", "str", "Padding token (never trained on)."),
    _o("template.extra_tokens", [], "list", "Extra special tokens to reserve in the vocabulary (e.g. tool-call markers)."),
    _o("template.train_on", "assistant", "str", "Which tokens the loss covers.", ("assistant", "all")),

    # ---- the data folder pipeline ------------------------------------------------------------------------------
    _o("data.extensions", [], "list", "Only ingest these file extensions (empty = every supported one)."),
    _o("data.recursive", True, "bool", "Scan sub-folders of the data folder."),
    _o("data.ignore", ["*.tmp", ".*", "~*", "*.part", "*.crdownload", "*.swp", "Thumbs.db", "desktop.ini", "dataset.json", "README*"], "list", "Glob patterns for files that are never ingested."),
    _o("data.watch_interval_s", 5.0, "float", "How often the daemon re-scans the data folder."),
    _o("data.settle_s", 2.0, "float", "A file must be unchanged for this long before it is ingested (never reads half-copied files)."),
    _o("data.max_records_per_file", None, "int|null", "Cap on records read from any single file (null = all)."),
    _o("data.schema_map", {}, "dict", "Column mapping for datasets with unusual field names, e.g. {\"user\": \"question\", \"assistant\": \"answer\", \"system\": \"persona\"}."),
    _o("data.adapter", AUTO, "str", "Force one adapter for every file (auto = detect per file). A folder's dataset.json can override per folder."),
    _o("data.val_fraction", 0.02, "float", "Share of documents held out for validation (assigned by a stable hash, so appending data never moves a document between train and validation)."),
    _o("data.val_min_docs", 20, "int", "Hold out at least this many documents when the corpus is small."),
    _o("data.dedupe", True, "bool", "Drop exact duplicate records (across all files, across runs)."),
    _o("data.min_chars", 2, "int", "Drop records whose longest message is shorter than this."),
    _o("data.max_chars", None, "int|null", "Drop records longer than this many characters (null = keep; long chats are split to fit the context window instead)."),
    _o("data.workers", AUTO, "int|auto", "Processes used for ingestion/tokenisation (auto = CPUs - 1)."),
    _o("data.text_chunk_chars", 4000, "int", "Plain-text files are cut into documents of about this many characters at paragraph boundaries."),
    _o("data.preference_as_sft", True, "bool", "Preference datasets (chosen/rejected) also contribute their chosen answer as an ordinary chat example."),
    _o("data.max_system_chars", None, "int|null", "Truncate system / persona text longer than this many characters (null = keep; huge character cards otherwise eat the context window)."),
    _o("data.user_alias", "User", "str", "What {{user}} / <USER> placeholders in roleplay datasets are replaced with (persona.user_name wins when set)."),
    _o("data.sniff_rows", 50, "int", "Rows inspected to auto-detect a file's schema."),
    _o("data.weights", {}, "dict", "Sampling weight per top-level data sub-folder, e.g. {\"persona\": 3.0}. Missing = 1.0."),

    _o("filter.persona.enabled", False, "bool", "Keep only records whose persona matches the keywords (the PIPPA \"sister\" filter). Usually set per folder in dataset.json."),
    _o("filter.persona.keywords", ["sister", "sibling", "big sis", "sis", "imouto", "onee-chan", "oneechan", "oneesan", "onee-san"], "list", "Words/phrases that mark a matching persona."),
    _o("filter.persona.match", "word", "str", "How keywords are matched: whole words (the old 'sis ' trick matched 'analysis'), raw substrings, or regular expressions.", ("word", "substring", "regex")),
    _o("filter.persona.where", ["system", "meta"], "list", "Where to look: the system text, record metadata (bot name/description/definitions...), and/or the conversation turns.", ),
    _o("filter.persona.require", "any", "str", "Record matches when any / all keywords are found.", ("any", "all")),
    _o("filter.safety.drop_sexual", True, "bool", "Drop sexually explicit records (lexicon based; imperfect, so review your data). Sexual content that involves a minor, or a family member, is always dropped regardless of this switch."),
    _o("filter.safety.redact", ["email", "phone", "ip", "card"], "list", "PII kinds replaced by placeholders before training.", ),
    _o("filter.safety.drop_terms", [], "list", "Additional words/phrases; a record containing any is dropped."),
    _o("filter.safety.drop_regex", [], "list", "Additional regular expressions; a record matching any is dropped."),
    _o("filter.quality.require_assistant", True, "bool", "Chat records need at least one assistant turn."),
    _o("filter.quality.max_repeat_ratio", 0.6, "float", "Drop a record when more than this share of its words is the same word repeated (degenerate text)."),
    _o("filter.quality.min_assistant_chars", 1, "int", "Drop chat records whose assistant text is shorter than this."),

    # ---- tokenizer -----------------------------------------------------------------------------------------------
    _o("tokenizer.path", None, "path|null", "Use / save the tokenizer here (default <models>/tokenizer.json). An existing file is reused, never retrained."),
    _o("tokenizer.vocab_size", AUTO, "int|auto", "Vocabulary size including the 256 byte tokens and the special tokens."),
    _o("tokenizer.train_sample_mb", 64.0, "float", "Train the BPE merges on at most this many MB of text sampled from the corpus."),
    _o("tokenizer.min_frequency", 2, "int", "A pair must occur at least this often to be merged."),

    # ---- the model -------------------------------------------------------------------------------------------
    _o("model.target_params", AUTO, "int|auto", "Non-embedding parameter budget used to derive the shape (auto = from the compute you will spend and how much data you have)."),
    _o("model.d_model", AUTO, "int|auto", "Width."),
    _o("model.n_layer", AUTO, "int|auto", "Depth."),
    _o("model.n_head", AUTO, "int|auto", "Attention heads."),
    _o("model.head_dim", 32, "int", "Per-head width used when n_head is derived."),
    _o("model.ffn_mult", 2.6667, "float", "SwiGLU hidden width as a multiple of d_model."),
    _o("model.aspect_ratio", 48, "int", "d_model / n_layer used when the shape is derived from target_params."),
    _o("model.ctx", AUTO, "int|auto", "Context window in tokens (auto = from the length of your conversations)."),
    _o("model.rope_theta", 10000.0, "float", "RoPE base frequency."),
    _o("model.norm_eps", 1e-5, "float", "RMSNorm epsilon."),
    _o("model.tie_embeddings", True, "bool", "Share the input embedding and the output projection."),
    _o("model.init_std", 0.02, "float", "Weight init standard deviation (residual projections are scaled by 1/sqrt(2*n_layer))."),
    _o("model.dtype", "float32", "str", "Compute precision (float64 is only for gradient checks).", ("float32", "float64")),
    _o("model.max_params", 120_000_000, "int", "Safety cap on total parameters, so a typo cannot allocate gigabytes."),

    # ---- training --------------------------------------------------------------------------------------------
    _o("train.budget_minutes", 30.0, "float|null", "Wall-clock limit of one training run (null = no limit; use max_steps / epochs)."),
    _o("train.max_steps", None, "int|null", "Stop after this many optimiser steps."),
    _o("train.epochs", None, "float|null", "Stop after this many passes over the training data."),
    _o("train.batch_tokens", AUTO, "int|auto", "Tokens per forward/backward pass (auto = what fits in memory)."),
    _o("train.effective_batch_tokens", AUTO, "int|auto", "Tokens per optimiser step (gradient accumulation fills the gap)."),
    _o("train.lr", AUTO, "float|auto", "Peak learning rate (auto = scaled by width)."),
    _o("train.min_lr_ratio", 0.1, "float", "Cosine decay floor as a fraction of the peak."),
    _o("train.warmup_steps", AUTO, "int|auto", "Linear warm-up steps (auto = autoscale.warmup_fraction of the run)."),
    _o("train.weight_decay", 0.1, "float", "AdamW weight decay (matrices only)."),
    _o("train.beta1", 0.9, "float", "Adam beta1."),
    _o("train.beta2", 0.95, "float", "Adam beta2."),
    _o("train.grad_clip", 1.0, "float", "Global gradient-norm clip."),
    _o("train.dropout", 0.0, "float", "Residual dropout (0 = off; most small-data chat runs do better with early stopping than dropout)."),
    _o("train.eval_every", AUTO, "int|auto", "Optimiser steps between validation passes."),
    _o("train.eval_tokens", AUTO, "int|auto", "Validation tokens per pass."),
    _o("train.patience", 10, "int", "Stop after this many validation passes without improvement (0 = never)."),
    _o("train.save_every_minutes", 5.0, "float", "Checkpoint at least this often."),
    _o("train.keep_checkpoints", 3, "int", "Trained model versions kept in models/weights besides the live one (older ones are deleted)."),
    _o("train.max_skipped_steps", 5, "int", "Abort when this many optimiser steps in a row are skipped for non-finite gradients (divergence)."),
    _o("train.preference_max_regression", 0.05, "float", "Discard the preference phase when it raises validation loss by more than this fraction."),
    _o("train.log_every", 10, "int", "Steps between console/log lines."),
    _o("train.seed", 0, "int", "Seed for initialisation and data order (resume restores the exact stream)."),
    _o("train.threads", AUTO, "int|auto", "BLAS threads for training (auto = every CPU)."),
    _o("train.ram_fraction", 0.35, "float", "Largest share of free RAM a training batch may use."),
    _o("train.continual.lr_scale", 0.3, "float", "When training continues on new data, the peak LR is the previous peak times this."),
    _o("train.continual.new_data_weight", 2.0, "float", "Sampling weight of documents added since the last run (old data is still replayed)."),

    _o("train.preference_steps", AUTO, "int|auto", "Optimiser steps of the preference (chosen vs rejected) phase after supervised training (auto = 10% of the run when the corpus has preference pairs, else 0)."),
    _o("train.preference_batch", 8, "int", "Preference pairs per step."),
    _o("train.preference_beta", 0.1, "float", "Strength of the preference loss in offline training."),
    _o("train.preference_lr_ratio", 0.1, "float", "Preference-phase LR as a fraction of the supervised peak LR."),
    _o("train.resume", True, "bool", "Continue from the last checkpoint / current model when one exists (false = start over)."),

    # ---- compute: CPU or GPU ------------------------------------------------------------------------------------------
    _o("compute.backend", "auto", "str", "Where TRAINING runs: cpu; gpu (CuPy, fails with the fix if it cannot be used); or auto (a GPU when CuPy and a working device are present, else the CPU). Serving, export and online learning always use the CPU.", ("auto", "cpu", "gpu")),
    _o("compute.device", 0, "int", "Which GPU to use when there are several."),
    _o("compute.vram_fraction", 0.8, "float", "Largest share of the GPU's FREE memory a training run may plan for (the rest is headroom for the driver, fragmentation and other programs)."),
    _o("compute.max_params", 200_000_000, "int", "Largest non-embedding parameter count autoscale will pick on a GPU (used in place of autoscale.max_params unless that was set explicitly)."),
    _o("compute.max_micro_tokens", 32768, "int", "Largest forward/backward batch on a GPU (used in place of autoscale.max_micro_tokens unless that was set explicitly)."),

    # ---- how auto values are derived (policy knobs of autoscale.py) -------------------------------------------------
    _o("autoscale.ctx_min", 128, "int", "Smallest context window autoscale will pick."),
    _o("autoscale.ctx_max", 512, "int", "Largest context window autoscale will pick (attention cost grows with its square on CPU)."),
    _o("autoscale.ctx_percentile", 90.0, "float", "The window covers this percentile of conversation lengths."),
    _o("autoscale.tokens_per_param", 20.0, "float", "Training tokens per non-embedding parameter the model should be able to see (compute-optimal ratio)."),
    _o("autoscale.min_params", 300_000, "int", "Smallest non-embedding parameter count."),
    _o("autoscale.max_params", 60_000_000, "int", "Largest non-embedding parameter count autoscale will pick."),
    _o("autoscale.max_epochs", 6.0, "float", "Never size the model so that the corpus would have to be repeated more than this many times."),
    _o("autoscale.target_steps", 1500, "int", "Optimiser steps a run should have; sets the effective batch size."),
    _o("autoscale.max_micro_tokens", 8192, "int", "Largest forward/backward batch (bigger gains nothing on CPU BLAS)."),
    _o("autoscale.max_effective_tokens", 65536, "int", "Largest effective batch (tokens per optimiser step)."),
    _o("autoscale.lr_ref", 0.003, "float", "Peak LR at the reference width."),
    _o("autoscale.lr_ref_width", 128, "int", "Reference width for lr_ref; the LR scales with sqrt(ref/d_model)."),
    _o("autoscale.eval_overhead", 0.08, "float", "Share of a run's time that validation may take."),
    _o("autoscale.min_batches_per_epoch", 8, "int", "A micro-batch never exceeds 1/this of the training data (tiny corpora get small batches)."),
    _o("autoscale.warmup_fraction", 0.03, "float", "Share of the planned steps used for LR warm-up when train.warmup_steps is auto."),
    _o("autoscale.eval_passes", 10, "int", "Validation passes per run when train.eval_every is auto."),
    _o("autoscale.sample_records", 2000, "int", "Records sampled to measure conversation lengths and text size."),
    _o("autoscale.bench_steps", 2, "int", "Timed steps of the throughput benchmark."),

    # ---- online learning ------------------------------------------------------------------------------------
    _o("online.enabled", True, "bool", "Learn from live feedback while chatting."),
    _o("online.lr", AUTO, "float|auto", "Learning rate of online updates (auto = online.lr_ratio x the last training LR)."),
    _o("online.lr_ratio", 0.1, "float", "See online.lr."),
    _o("online.steps_per_update", 6, "int", "Gradient steps per online update."),
    _o("online.batch_examples", 8, "int", "New examples per online step."),
    _o("online.replay_ratio", 3.0, "float", "Replay documents (from the original corpus) mixed in per new example, to prevent forgetting."),
    _o("online.min_events", 4, "int", "Feedback events needed to trigger a batch update."),
    _o("online.immediate_corrections", True, "bool", "A typed correction triggers an update right away instead of waiting for min_events."),
    _o("online.interval_s", 30.0, "float", "How often the learner checks for pending feedback."),
    _o("online.max_regression", 0.03, "float", "Reject (roll back) an update when validation loss on replay documents rises by more than this fraction."),
    _o("online.probe_docs", 48, "int", "Validation documents used to gate each update."),
    _o("online.dpo_beta", 0.1, "float", "Strength of the preference loss for (chosen, rejected) pairs."),
    _o("online.preference_weight", 0.5, "float", "Weight of preference updates relative to supervised ones."),
    _o("online.implicit_accept_s", None, "float|null", "Treat an unrated reply as accepted after this many idle seconds (null = only explicit feedback counts)."),
    _o("online.implicit_weight", 0.5, "float", "Training weight of implicitly accepted replies (see implicit_accept_s); explicit thumbs-up and corrections weigh 1."),
    _o("online.save_to_data", True, "bool", "Write learned feedback examples into the data folder so later training runs keep them."),
    _o("online.data_folder", "_online", "str", "Sub-folder of the data folder that receives them."),
    _o("online.buffer_max", 50000, "int", "Maximum replies kept in the experience log (older ones are compacted away)."),
    _o("online.keep_versions", 5, "int", "Online model versions kept for rollback."),
    _o("online.threads", AUTO, "int|auto", "BLAS threads the learner may use (auto = half of the CPUs)."),

    # ---- generation ------------------------------------------------------------------------------------------
    _o("generation.max_new_tokens", AUTO, "int|auto", "Longest reply in tokens (auto = a quarter of the context window)."),
    _o("generation.temperature", 0.8, "float", "Sampling temperature (0 = greedy)."),
    _o("generation.top_k", 50, "int", "Keep only the k most likely tokens (0 = off)."),
    _o("generation.top_p", 0.95, "float", "Nucleus sampling mass (1 = off)."),
    _o("generation.repetition_penalty", 1.1, "float", "Penalty on recently used tokens (1 = off)."),
    _o("generation.repetition_window", 128, "int", "How many recent tokens the penalty looks at."),
    _o("generation.stop", [], "list", "Extra strings that end a reply."),
    _o("generation.seed", None, "int|null", "Fix the sampling seed (null = fresh randomness)."),
    _o("generation.history_turns", AUTO, "int|auto", "Most recent conversation turns kept in the prompt (auto = as many as fit)."),

    # ---- server / daemon -------------------------------------------------------------------------------------
    _o("server.host", "127.0.0.1", "str", "Bind address. Anything other than loopback requires the token."),
    _o("server.port", 8787, "int", "Preferred port."),
    _o("server.port_fallback", True, "bool", "If the port is taken, use the next free one (the real address is in run/server.json)."),
    _o("server.require_token", True, "bool", "Require the access token on every state-changing request (protects against other web pages talking to your local server)."),
    _o("server.token", AUTO, "str|auto", "Access token (auto = generated once and stored in state/token)."),
    _o("server.cors_origins", [], "list", "Origins allowed to call the API from a browser."),
    _o("server.max_body_mb", 64.0, "float", "Largest request body (voice uploads)."),
    _o("server.ui", True, "bool", "Serve the built-in web chat UI at /."),
    _o("server.session_ttl_s", 86400.0, "float", "Idle chat sessions are dropped from memory after this long (their log stays on disk)."),
    _o("daemon.supervise", True, "bool", "Run a tiny supervisor that restarts the worker if it crashes."),
    _o("daemon.restart_backoff_s", [1, 2, 5, 10, 30, 60], "list", "Seconds to wait before successive restarts."),
    _o("daemon.max_restarts_per_hour", 20, "int", "Give up after this many crashes in an hour."),
    _o("daemon.autotrain", True, "bool", "Train automatically when enough new data has arrived."),
    _o("daemon.train_min_new_tokens", AUTO, "int|auto", "New tokens that trigger a continued-training run (auto = 2% of the corpus, at least 20000)."),
    _o("daemon.train_cooldown_minutes", 60.0, "float", "Minimum gap between automatic training runs."),
    _o("daemon.train_budget_minutes", None, "float|null", "Wall-clock limit of automatic runs (null = train.budget_minutes)."),
    _o("daemon.low_priority_training", True, "bool", "Run background training at low CPU priority."),
    _o("daemon.hot_swap", True, "bool", "Load a newly trained / validated model into the running server without restarting."),
    _o("daemon.log_max_mb", 10.0, "float", "Rotate log files at this size."),
    _o("daemon.log_backups", 5, "int", "Rotated log files kept."),

    # ---- dataset download ------------------------------------------------------------------------------------
    _o("fetch.endpoint", "https://datasets-server.huggingface.co", "str", "Rows API used to download Hugging Face datasets without extra libraries."),
    _o("fetch.hub", "https://huggingface.co", "str", "Hub URL (used for file downloads)."),
    _o("fetch.token_env", "HF_TOKEN", "str", "Environment variable holding a Hugging Face token (needed for gated datasets)."),
    _o("fetch.default_limit", 20000, "int", "Rows fetched per dataset unless --limit says otherwise."),
    _o("fetch.timeout_s", 60.0, "float", "Per-request timeout."),
    _o("fetch.retries", 5, "int", "Retries with exponential backoff."),
    _o("fetch.page_size", 100, "int", "Rows per request (the API allows at most 100)."),
    _o("fetch.catalog", None, "path|null", "Your own catalog JSON (merged over the shipped one): name -> {repo, config, split, adapter, note}."),

    # ---- voice -----------------------------------------------------------------------------------------------
    _o("voice.sample_rate", 16000, "int", "Sample rate used for speech recognition input."),
    _o("voice.input_device", None, "str|null", "Microphone (name fragment or index; null = system default)."),
    _o("voice.output_device", None, "str|null", "Speaker (name fragment or index; null = system default)."),
    _o("voice.audio_folder", "audio", "str", "Sub-folder of the data folder that holds speech datasets (audio + transcripts). The text pipeline skips it; `voice train-stt` reads it."),
    _o("voice.vad.threshold_db", AUTO, "float|auto", "Absolute speech threshold in dBFS (auto = the measured noise floor + margin_db, tracked continuously)."),
    _o("voice.vad.margin_db", 9.0, "float", "How far above the measured noise floor speech must be when the threshold is auto."),
    _o("voice.vad.calibration_s", 0.6, "float", "Seconds of room noise measured before listening."),
    _o("voice.vad.min_speech_ms", 200, "int", "Shorter bursts are ignored."),
    _o("voice.vad.end_silence_ms", 700, "int", "Silence that ends an utterance."),
    _o("voice.vad.pre_roll_ms", 300, "int", "Audio kept from before speech was detected."),
    _o("voice.vad.max_utterance_s", 30.0, "float", "Hard cap on one utterance."),
    _o("voice.stt.engine", AUTO, "str", "Speech recognition backend (auto = best available).", (AUTO, "builtin", "vosk", "faster_whisper", "whisper_cpp", "windows_sapi", "command")),
    _o("voice.stt.model", None, "str|null", "Model path or name for the chosen backend (builtin: the model directory trained with `voice train-stt`)."),
    _o("voice.stt.language", "en", "str", "Language code."),
    _o("voice.stt.command", None, "str|null", "For engine=command: shell template, {wav} is replaced by a 16 kHz mono WAV path; stdout is the transcript."),
    _o("voice.tts.engine", AUTO, "str", "Speech synthesis backend (auto = best available, the built-in synthesizer is the last resort).", (AUTO, "builtin", "espeak", "say", "windows_sapi", "piper", "pyttsx3", "command")),
    _o("voice.tts.voice", None, "str|null", "Voice name / id for the backend (null = backend default)."),
    _o("voice.tts.model", None, "str|null", "Model file for engines that need one (piper)."),
    _o("voice.tts.rate", 1.0, "float", "Speaking speed multiplier."),
    _o("voice.tts.pitch", 1.0, "float", "Pitch multiplier."),
    _o("voice.tts.volume", 1.0, "float", "Volume multiplier."),
    _o("voice.tts.command", None, "str|null", "For engine=command: shell template with {text} and {wav}."),
    _o("voice.tts.cache", True, "bool", "Cache synthesised phrases on disk."),
    _o("voice.tts.cache_max_mb", 200.0, "float", "Oldest cached phrases are deleted beyond this size."),
    _o("voice.tts.max_chars", 1500, "int", "Longer answers are spoken only up to this many characters."),
    _o("voice.convert.ffmpeg", None, "path|null", "Path to ffmpeg (null = look it up on PATH). Needed only for mp3/flac/ogg/m4a input."),
    _o("voice.builtin_stt.n_mels", 40, "int", "Mel bands of the built-in recogniser."),
    _o("voice.builtin_stt.hidden", AUTO, "int|auto", "Width of the built-in recogniser."),
    _o("voice.builtin_stt.layers", AUTO, "int|auto", "Depth of the built-in recogniser."),
    _o("voice.builtin_stt.epochs", 30, "int", "Training epochs of `voice train-stt`."),
    _o("voice.builtin_stt.batch", 8, "int", "Utterances per batch."),
    _o("voice.builtin_stt.lr", 2e-3, "float", "Learning rate."),
    _o("voice.builtin_stt.max_seconds", 12.0, "float", "Longer clips are skipped."),
    _o("voice.builtin_stt.budget_minutes", 15.0, "float|null", "Wall-clock limit of `voice train-stt` (null = until `epochs` are done)."),
    _o("voice.builtin_stt.max_utterances", None, "int|null", "Use at most this many clips (null = all found)."),
    _o("voice.builtin_stt.val_fraction", 0.05, "float", "Share of clips held out to measure the character error rate."),
    _o("voice.builtin_stt.val_min", 8, "int", "Hold out at least this many clips."),
    _o("voice.builtin_stt.augment", True, "bool", "Randomly mask time / frequency bands and jitter the gain while training."),
    _o("voice.builtin_stt.kernel", 5, "int", "Convolution width (frames)."),
    _o("voice.builtin_stt.dilations", [1, 2, 4, 8], "list", "Dilations, cycled over the layers (they set how much context each frame sees)."),
    _o("voice.builtin_stt.min_char_count", 3, "int", "Characters seen fewer times than this in the transcripts are left out of the alphabet."),
    _o("voice.builtin_stt.win_ms", 25.0, "float", "Analysis window."),
    _o("voice.builtin_stt.hop_ms", 10.0, "float", "Hop between analysis windows."),
    _o("voice.builtin_stt.floor_db", 50.0, "float", "Dynamic range of the features: anything quieter than this below the loudest frame is treated as silence."),
    _o("voice.builtin_stt.pad_s", 0.25, "float", "Silence added either side of every utterance when recognising (and, randomly up to twice this, when training)."),
    _o("voice.builtin_stt.clip_norm", 5.0, "float", "Gradient clipping."),
    _o("voice.builtin_stt.patience", 6, "int", "Stop after this many epochs without a better validation error."),
    _o("voice.tts.builtin_f0_hz", None, "float|null", "Base pitch of the built-in synthesizer (null = the measured profile from `voice profile`, else 118)."),
    _o("voice.tts.builtin_tempo", None, "float|null", "Base speed of the built-in synthesizer in phonemes per second (null = profile, else 12)."),
    _o("voice.talk.speak", True, "bool", "Speak the answers aloud in `chat talk --voice`."),
    _o("voice.talk.stop_phrases", ["stop listening", "goodbye", "quit voice chat"], "list", "Saying one of these ends a voice conversation."),
    _o("voice.talk.listen_timeout_s", None, "float|null", "Give up waiting for speech after this long (null = wait forever)."),
    _o("voice.talk.beep", True, "bool", "Play a short tone when the assistant starts listening."),
]

_BY_KEY: Dict[str, Opt] = {o.key: o for o in SCHEMA}


def _derivable(key: str) -> bool:
    """True when 'auto' means 'derive this' (kind has |auto); otherwise 'auto' is just an ordinary string value."""
    return "auto" in _BY_KEY[key].kind.split("|")
_DICT_KEYS = {o.key for o in SCHEMA if o.base == "dict"}


# --------------------------------------------------------------------------------------------------------------------
def _strip_json_comments(text: str) -> str:
    """Remove // and # line comments and /* */ blocks that are outside string literals, and trailing commas."""
    out, i, n, in_str, esc = [], 0, len(text), False, False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif c == "/" and text[i:i + 2] == "//" or c == "#":
            while i < n and text[i] != "\n":
                i += 1
        elif c == "/" and text[i:i + 2] == "/*":
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def read_config_file(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    raw = fileio.read_text(path)
    if raw.startswith("﻿"):
        raw = raw[1:]
    if ext in (".json", ".jsonc", ""):
        try:
            return json.loads(_strip_json_comments(raw) or "{}")
        except ValueError as e:
            raise ConfigError(f"{path}: not valid JSON ({e})") from e
    if ext == ".toml":
        try:
            import tomllib                                                   # Python 3.11+
        except ImportError as e:
            raise ConfigError(f"{path}: TOML needs Python 3.11+; use a .json config instead") from e
        return tomllib.loads(raw)
    if ext in (".yaml", ".yml"):
        try:
            import yaml                                                      # optional
        except ImportError as e:
            raise ConfigError(f"{path}: YAML needs `pip install pyyaml`; use a .json config instead") from e
        return yaml.safe_load(raw) or {}
    raise ConfigError(f"{path}: unsupported config format '{ext}' (use .json, .toml or .yaml)")


def _flatten(d: dict, prefix: str = "") -> Dict[str, Any]:
    flat = {}
    for k, v in d.items():
        if str(k).startswith("_") or str(k).startswith("//"):          # "_comment": ... is documentation, not an option
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict) and key not in _DICT_KEYS:
            flat.update(_flatten(v, key + "."))
        else:
            flat[key] = v
    return flat


def _parse_scalar(s: str):
    try:
        return json.loads(s)
    except ValueError:
        return s


def coerce(opt: Opt, v: Any):
    kinds = opt.kind.split("|")
    if v is None:
        if "null" in kinds:
            return None
        raise ConfigError(f"{opt.key}: null is not allowed")
    if isinstance(v, str) and v.strip().lower() == AUTO:
        if "auto" in kinds:
            return AUTO
        if opt.default == AUTO:
            return AUTO
        raise ConfigError(f"{opt.key}: 'auto' is not allowed here")
    if isinstance(v, str) and v.strip().lower() in ("null", "none") and "null" in kinds:
        return None
    base = kinds[0]
    try:
        if base == "int":
            if isinstance(v, bool):
                raise ValueError
            f = float(v)
            if f != int(f):
                raise ValueError
            return int(f)
        if base == "float":
            if isinstance(v, bool):
                raise ValueError
            return float(v)
        if base == "bool":
            if isinstance(v, bool):
                return v
            if isinstance(v, str) and v.strip().lower() in ("1", "true", "yes", "on"):
                return True
            if isinstance(v, str) and v.strip().lower() in ("0", "false", "no", "off"):
                return False
            raise ValueError
        if base in ("str", "path"):
            v = str(v)
            if base == "path":
                v = os.path.expanduser(os.path.expandvars(v))
        elif base == "list":
            if isinstance(v, str):
                s = v.strip()
                v = json.loads(s) if s.startswith("[") else [x.strip() for x in s.split(",") if x.strip()]
            if not isinstance(v, (list, tuple)):
                raise ValueError
            v = list(v)
        elif base == "dict":
            if isinstance(v, str):
                v = json.loads(v)
            if not isinstance(v, dict):
                raise ValueError
    except (ValueError, TypeError) as e:
        raise ConfigError(f"{opt.key}: expected {opt.kind}, got {v!r}") from e
    if opt.choices and v not in opt.choices:
        raise ConfigError(f"{opt.key}: must be one of {list(opt.choices)}, got {v!r}")
    return v


def _unknown(key: str, where: str) -> ConfigError:
    near = difflib.get_close_matches(key, list(_BY_KEY), n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(near)}?" if near else ""
    return ConfigError(f"unknown option '{key}' ({where}).{hint} `zerowhite chat config --reference` lists every option.")


class Section:
    """Read-only attribute view of one dotted prefix: cfg.train.lr, cfg.filter.persona.enabled."""
    __slots__ = ("_cfg", "_prefix")

    def __init__(self, cfg: "Config", prefix: str):
        object.__setattr__(self, "_cfg", cfg)
        object.__setattr__(self, "_prefix", prefix)

    def __getattr__(self, name: str):
        key = f"{self._prefix}.{name}"
        if key in _BY_KEY:
            return self._cfg.get(key)
        if any(k.startswith(key + ".") for k in _BY_KEY):
            return Section(self._cfg, key)
        raise AttributeError(key)

    def __setattr__(self, name, value):
        raise AttributeError("configuration is read-only; use cfg.replace(...)")

    def keys(self):
        p = self._prefix + "."
        return [k[len(p):] for k in _BY_KEY if k.startswith(p)]


class Config:
    """Immutable-ish layered configuration bound to a workspace directory."""

    def __init__(self, workspace: str, values: Optional[Dict[str, Any]] = None,
                 derived: Optional[Dict[str, Tuple[Any, str]]] = None, sources: Optional[Dict[str, str]] = None,
                 origin: Optional[Dict[str, Any]] = None):
        self.workspace = os.path.abspath(workspace)
        self.origin: Dict[str, Any] = dict(origin or {})              # how this config was loaded: config_file + CLI overrides
        self._v: Dict[str, Any] = {o.key: copy.deepcopy(o.default) for o in SCHEMA}
        self._src: Dict[str, str] = {o.key: "default" for o in SCHEMA}
        self.derived: Dict[str, Tuple[Any, str]] = dict(derived or {})
        if values:
            for k, v in values.items():
                self._v[k] = v
        if sources:
            self._src.update(sources)

    # ---- construction --------------------------------------------------------------------------------------
    @classmethod
    def load(cls, workspace: Optional[str] = None, config_file: Optional[str] = None,
             overrides: Iterable[str] = (), env: Optional[dict] = None) -> "Config":
        env = os.environ if env is None else env
        ws = workspace or env.get("ZW_WORKSPACE") or "zw-chat"
        cfg = cls(ws)
        path = config_file or env.get("ZW_CONFIG")
        if not path:
            for name in ("config.json", "config.jsonc", "config.toml", "config.yaml", "config.yml"):
                cand = os.path.join(cfg.workspace, name)
                if os.path.isfile(cand):
                    path = cand
                    break
        if path:
            if not os.path.isfile(path):
                raise ConfigError(f"config file not found: {path}")
            cfg = cfg.merge(read_config_file(path), f"file:{path}")
        envd = {}
        for k, v in env.items():
            if k.startswith("ZW_") and "__" in k:
                key = ".".join(p.lower() for p in k[3:].split("__"))
                if key in _BY_KEY:
                    envd[key] = _parse_scalar(v) if _BY_KEY[key].base in ("int", "float", "bool", "list", "dict") else v
        if envd:
            cfg = cfg.merge(envd, "env")
        sets = {}
        for item in overrides or ():
            if "=" not in item:
                raise ConfigError(f"--set expects key=value, got '{item}'")
            k, v = item.split("=", 1)
            k = k.strip()
            if k not in _BY_KEY:
                raise _unknown(k, "--set")
            sets[k] = _parse_scalar(v) if _BY_KEY[k].base != "str" and _BY_KEY[k].base != "path" else v
        if sets:
            cfg = cfg.merge(sets, "cli")
        cfg.origin = dict(config_file=os.path.abspath(path) if path else None, overrides=list(overrides or ()))
        return cfg

    def cli_args(self) -> List[str]:
        """Command-line arguments that make a child process load the same configuration."""
        out = ["--workspace", self.workspace]
        if self.origin.get("config_file"):
            out += ["--config", self.origin["config_file"]]
        for o in self.origin.get("overrides") or ():
            out += ["--set", o]
        return out

    def merge(self, layer: dict, source: str) -> "Config":
        new_v, new_src = dict(self._v), dict(self._src)
        for k, v in _flatten(layer).items():
            if k not in _BY_KEY:
                raise _unknown(k, source)
            new_v[k] = coerce(_BY_KEY[k], v)
            new_src[k] = source
        return Config(self.workspace, new_v, self.derived, new_src, self.origin)

    def replace(self, **kv) -> "Config":
        """cfg.replace(**{'train.lr': 0.001}) -> new Config."""
        return self.merge({k: v for k, v in kv.items()}, "code")

    def with_derived(self, derived: Dict[str, Tuple[Any, str]]) -> "Config":
        d = dict(self.derived)
        d.update(derived)
        return Config(self.workspace, self._v, d, self._src, self.origin)

    # ---- access --------------------------------------------------------------------------------------------
    def raw(self, key: str):
        if key not in _BY_KEY:
            raise _unknown(key, "lookup")
        return self._v[key]

    def get(self, key: str, fallback: Any = NotImplemented):
        v = self.raw(key)
        if _derivable(key) and isinstance(v, str) and v == AUTO:
            if key in self.derived:
                return self.derived[key][0]
            if fallback is not NotImplemented:
                return fallback
            raise NotResolved(f"{key} is 'auto' and has not been derived yet (run autoscale.resolve, or set it explicitly)")
        return v

    def is_auto(self, key: str) -> bool:
        v = self.raw(key)
        return _derivable(key) and isinstance(v, str) and v == AUTO and key not in self.derived

    def __getattr__(self, name: str):
        if name.startswith("_") or name in ("workspace", "derived", "origin"):
            raise AttributeError(name)
        if any(k.startswith(name + ".") for k in _BY_KEY):
            return Section(self, name)
        raise AttributeError(name)

    # ---- paths ---------------------------------------------------------------------------------------------
    def path(self, which: str) -> str:
        """Resolved absolute directory for paths.<which>; created on demand by callers via ensure()."""
        v = self.get(f"paths.{which}")
        p = v if v else os.path.join(self.workspace, {"data": "data", "models": "models", "state": "state", "logs": "logs",
                                                      "run": "run", "exports": "exports"}[which])
        return p if os.path.isabs(p) else os.path.abspath(os.path.join(self.workspace, p))

    def resolve_path(self, p: str) -> str:
        p = os.path.expanduser(os.path.expandvars(p))
        return p if os.path.isabs(p) else os.path.abspath(os.path.join(self.workspace, p))

    def system_prompt(self) -> Optional[str]:
        """The persona's system prompt with {name}/{user} substituted, or None when none is configured."""
        text = self.get("persona.system_prompt")
        f = self.get("persona.system_prompt_file")
        if f:
            fp = self.resolve_path(f)
            if not os.path.isfile(fp):
                raise ConfigError(f"persona.system_prompt_file not found: {fp}")
            text = fileio.read_text(fp)
        if not text:
            return None
        subs = {"name": self.get("persona.name") or "", "user": self.get("persona.user_name") or ""}
        return re.sub(r"\{(name|user)\}", lambda m: subs[m.group(1)], text).strip() or None

    # ---- output --------------------------------------------------------------------------------------------
    def to_dict(self, nested: bool = True, resolved: bool = True) -> dict:
        flat = {}
        for o in SCHEMA:
            v = self._v[o.key]
            if resolved and o.key in self.derived and _derivable(o.key) and isinstance(v, str) and v == AUTO:
                v = self.derived[o.key][0]
            flat[o.key] = v
        if not nested:
            return flat
        out: dict = {}
        for k, v in flat.items():
            cur = out
            parts = k.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = v
        return out

    def changed(self) -> Dict[str, Any]:
        return {k: self._v[k] for k, s in self._src.items() if s != "default"}

    def source_of(self, key: str) -> str:
        return self._src.get(key, "default")

    def dump(self, path: str, only_changed: bool = False):
        data = _nest(self.changed()) if only_changed else self.to_dict()
        fileio.write_json(path, data, indent=2)

    def explain(self) -> str:
        lines = []
        for o in SCHEMA:
            v = self._v[o.key]
            note = ""
            if _derivable(o.key) and isinstance(v, str) and v == AUTO and o.key in self.derived:
                v, why = self.derived[o.key]
                note = f"   <- derived: {why}"
            elif self._src[o.key] != "default":
                note = f"   <- {self._src[o.key]}"
            lines.append(f"{o.key:<36} {json.dumps(v, default=str)}{note}")
        return "\n".join(lines)


def _nest(flat: dict) -> dict:
    out: dict = {}
    for k, v in flat.items():
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out


def reference_markdown() -> str:
    """Every option, generated from SCHEMA (docs/CONFIG.md is this output)."""
    out = ["# Configuration reference", "",
           "Generated from `zerowhite/chat/config.py`; do not edit by hand (`zerowhite chat config --reference`).", "",
           "Layers, later wins: schema default < `<workspace>/config.json` (or `--config`, `$ZW_CONFIG`) < environment "
           "(`ZW_TRAIN__LR=0.002`) < `--set train.lr=0.002` < a folder's `dataset.json` (data/filter keys).",
           "`auto` means the value is derived from your machine and data at run time and recorded in "
           "`<workspace>/state/resolved_config.json` with the reason.", ""]
    section = None
    for o in SCHEMA:
        sec = o.key.split(".")[0]
        if sec != section:
            section = sec
            out += ["", f"## {sec}", "", "| option | type | default | description |", "|---|---|---|---|"]
        cell = lambda t: str(t).replace("|", "\\|").replace("\n", " ")                      # noqa: E731  (a | would end a table cell)
        d = json.dumps(o.default)
        ch = f" One of: {', '.join(map(str, o.choices))}." if o.choices else ""
        out.append(f"| `{o.key}` | {cell(o.kind)} | {cell(d)} | {cell(o.doc + ch)} |")
    return "\n".join(out) + "\n"
