# Configuration reference

Generated from `zerowhite/chat/config.py`; do not edit by hand (`zerowhite chat config --reference`).

Layers, later wins: schema default < `<workspace>/config.json` (or `--config`, `$ZW_CONFIG`) < environment (`ZW_TRAIN__LR=0.002`) < `--set train.lr=0.002` < a folder's `dataset.json` (data/filter keys).
`auto` means the value is derived from your machine and data at run time and recorded in `<workspace>/state/resolved_config.json` with the reason.


## paths

| option | type | default | description |
|---|---|---|---|
| `paths.data` | path\|null | null | Folder you drop datasets into (default <workspace>/data). Scanned recursively; new files are picked up automatically. |
| `paths.models` | path\|null | null | Checkpoints, tokenizer and the live model pointer (default <workspace>/models). |
| `paths.state` | path\|null | null | Manifests, experience log, corpus shards and token shards (default <workspace>/state). |
| `paths.logs` | path\|null | null | Log files (default <workspace>/logs). |
| `paths.run` | path\|null | null | Runtime files: pid file, server address, stop file (default <workspace>/run). |
| `paths.exports` | path\|null | null | Where `export` and `convert` write by default (default <workspace>/exports). |

## persona

| option | type | default | description |
|---|---|---|---|
| `persona.name` | str\|null | null | The assistant's name; available to the system prompt as {name}. |
| `persona.user_name` | str\|null | null | What the assistant calls you; available to the system prompt as {user}. |
| `persona.system_prompt` | str\|null | null | System prompt text placed at the start of every conversation ({name} and {user} are substituted). |
| `persona.system_prompt_file` | path\|null | null | Read the system prompt from this file instead (relative paths are resolved against the workspace). |
| `persona.greeting` | str\|null | null | Optional first message the assistant shows/speaks when a session starts. |

## template

| option | type | default | description |
|---|---|---|---|
| `template.system_token` | str | "<\|system\|>" | Special token that opens a system message. |
| `template.user_token` | str | "<\|user\|>" | Special token that opens a user message. |
| `template.assistant_token` | str | "<\|assistant\|>" | Special token that opens an assistant message. |
| `template.end_token` | str | "<\|end\|>" | Special token that closes every message (the model learns to emit it to stop). |
| `template.pad_token` | str | "<\|pad\|>" | Padding token (never trained on). |
| `template.extra_tokens` | list | [] | Extra special tokens to reserve in the vocabulary (e.g. tool-call markers). |
| `template.train_on` | str | "assistant" | Which tokens the loss covers. One of: assistant, all. |

## data

| option | type | default | description |
|---|---|---|---|
| `data.extensions` | list | [] | Only ingest these file extensions (empty = every supported one). |
| `data.recursive` | bool | true | Scan sub-folders of the data folder. |
| `data.ignore` | list | ["*.tmp", ".*", "~*", "*.part", "*.crdownload", "*.swp", "Thumbs.db", "desktop.ini", "dataset.json", "README*"] | Glob patterns for files that are never ingested. |
| `data.watch_interval_s` | float | 5.0 | How often the daemon re-scans the data folder. |
| `data.settle_s` | float | 2.0 | A file must be unchanged for this long before it is ingested (never reads half-copied files). |
| `data.settle_wait_s` | float | 120.0 | How long `chat prepare` / `chat train` waits by itself for files that are still being copied, when nothing else has been ingested yet (0 = do not wait). |
| `data.max_records_per_file` | int\|null | null | Cap on records read from any single file (null = all). |
| `data.schema_map` | dict | {} | Column mapping for datasets with unusual field names, e.g. {"user": "question", "assistant": "answer", "system": "persona"}. |
| `data.adapter` | str | "auto" | Force one adapter for every file (auto = detect per file). A folder's dataset.json can override per folder. |
| `data.val_fraction` | float | 0.02 | Share of documents held out for validation (assigned by a stable hash, so appending data never moves a document between train and validation). |
| `data.val_min_docs` | int | 20 | Hold out at least this many documents when the corpus is small. |
| `data.dedupe` | bool | true | Drop exact duplicate records (across all files, across runs). |
| `data.min_chars` | int | 2 | Drop records whose longest message is shorter than this. |
| `data.max_chars` | int\|null | null | Drop records longer than this many characters (null = keep; long chats are split to fit the context window instead). |
| `data.workers` | int\|auto | "auto" | Processes used for ingestion/tokenisation (auto = CPUs - 1). |
| `data.text_chunk_chars` | int | 4000 | Plain-text files are cut into documents of about this many characters at paragraph boundaries. |
| `data.preference_as_sft` | bool | true | Preference datasets (chosen/rejected) also contribute their chosen answer as an ordinary chat example. |
| `data.max_system_chars` | int\|null | null | Truncate system / persona text longer than this many characters (null = keep; huge character cards otherwise eat the context window). |
| `data.user_alias` | str | "User" | What {{user}} / <USER> placeholders in roleplay datasets are replaced with (persona.user_name wins when set). |
| `data.sniff_rows` | int | 50 | Rows inspected to auto-detect a file's schema. |
| `data.weights` | dict | {} | Sampling weight per top-level data sub-folder, e.g. {"persona": 3.0}. Missing = 1.0. |

## filter

| option | type | default | description |
|---|---|---|---|
| `filter.persona.enabled` | bool | false | Keep only records whose persona matches the keywords (the PIPPA "sister" filter). Usually set per folder in dataset.json. |
| `filter.persona.keywords` | list | ["sister", "sibling", "big sis", "sis", "imouto", "onee-chan", "oneechan", "oneesan", "onee-san"] | Words/phrases that mark a matching persona. |
| `filter.persona.match` | str | "word" | How keywords are matched: whole words (the old 'sis ' trick matched 'analysis'), raw substrings, or regular expressions. One of: word, substring, regex. |
| `filter.persona.where` | list | ["system", "meta"] | Where to look: the system text, record metadata (bot name/description/definitions...), and/or the conversation turns. |
| `filter.persona.require` | str | "any" | Record matches when any / all keywords are found. One of: any, all. |
| `filter.safety.drop_sexual` | bool | true | Drop sexually explicit records (lexicon based; imperfect, so review your data). Sexual content that involves a minor, or a family member, is always dropped regardless of this switch. |
| `filter.safety.redact` | list | ["email", "phone", "ip", "card"] | PII kinds replaced by placeholders before training. |
| `filter.safety.drop_terms` | list | [] | Additional words/phrases; a record containing any is dropped. |
| `filter.safety.drop_regex` | list | [] | Additional regular expressions; a record matching any is dropped. |
| `filter.quality.require_assistant` | bool | true | Chat records need at least one assistant turn. |
| `filter.quality.max_repeat_ratio` | float | 0.6 | Drop a record when more than this share of its words is the same word repeated (degenerate text). |
| `filter.quality.min_assistant_chars` | int | 1 | Drop chat records whose assistant text is shorter than this. |

## tokenizer

| option | type | default | description |
|---|---|---|---|
| `tokenizer.path` | path\|null | null | Use / save the tokenizer here (default <models>/tokenizer.json). An existing file is reused, never retrained. |
| `tokenizer.vocab_size` | int\|auto | "auto" | Vocabulary size including the 256 byte tokens and the special tokens. |
| `tokenizer.train_sample_mb` | float | 64.0 | Train the BPE merges on at most this many MB of text sampled from the corpus. |
| `tokenizer.min_frequency` | int | 2 | A pair must occur at least this often to be merged. |

## model

| option | type | default | description |
|---|---|---|---|
| `model.target_params` | int\|auto | "auto" | Non-embedding parameter budget used to derive the shape (auto = from the compute you will spend and how much data you have). |
| `model.d_model` | int\|auto | "auto" | Width. |
| `model.n_layer` | int\|auto | "auto" | Depth. |
| `model.n_head` | int\|auto | "auto" | Attention heads. |
| `model.head_dim` | int | 32 | Per-head width used when n_head is derived. |
| `model.ffn_mult` | float | 2.6667 | SwiGLU hidden width as a multiple of d_model. |
| `model.aspect_ratio` | int | 48 | d_model / n_layer used when the shape is derived from target_params. |
| `model.ctx` | int\|auto | "auto" | Context window in tokens (auto = from the length of your conversations). |
| `model.rope_theta` | float | 10000.0 | RoPE base frequency. |
| `model.norm_eps` | float | 1e-05 | RMSNorm epsilon. |
| `model.tie_embeddings` | bool | true | Share the input embedding and the output projection. |
| `model.init_std` | float | 0.02 | Weight init standard deviation (residual projections are scaled by 1/sqrt(2*n_layer)). |
| `model.dtype` | str | "float32" | Compute precision (float64 is only for gradient checks). One of: float32, float64. |
| `model.max_params` | int | 120000000 | Safety cap on total parameters, so a typo cannot allocate gigabytes. |

## train

| option | type | default | description |
|---|---|---|---|
| `train.budget_minutes` | float\|null | 30.0 | Wall-clock limit of one training run (null = no limit; use max_steps / epochs). |
| `train.max_steps` | int\|null | null | Stop after this many optimiser steps. |
| `train.epochs` | float\|null | null | Stop after this many passes over the training data. |
| `train.batch_tokens` | int\|auto | "auto" | Tokens per forward/backward pass (auto = what fits in memory). |
| `train.effective_batch_tokens` | int\|auto | "auto" | Tokens per optimiser step (gradient accumulation fills the gap). |
| `train.lr` | float\|auto | "auto" | Peak learning rate (auto = scaled by width). |
| `train.min_lr_ratio` | float | 0.1 | Cosine decay floor as a fraction of the peak. |
| `train.warmup_steps` | int\|auto | "auto" | Linear warm-up steps (auto = autoscale.warmup_fraction of the run). |
| `train.weight_decay` | float | 0.1 | AdamW weight decay (matrices only). |
| `train.beta1` | float | 0.9 | Adam beta1. |
| `train.beta2` | float | 0.95 | Adam beta2. |
| `train.grad_clip` | float | 1.0 | Global gradient-norm clip. |
| `train.dropout` | float | 0.0 | Residual dropout (0 = off; most small-data chat runs do better with early stopping than dropout). |
| `train.eval_every` | int\|auto | "auto" | Optimiser steps between validation passes. |
| `train.eval_tokens` | int\|auto | "auto" | Validation tokens per pass. |
| `train.patience` | int | 10 | Stop after this many validation passes without improvement (0 = never). |
| `train.save_every_minutes` | float | 5.0 | Checkpoint at least this often. |
| `train.keep_checkpoints` | int | 3 | Trained model versions kept in models/weights besides the live one (older ones are deleted). |
| `train.max_skipped_steps` | int | 5 | Abort when this many optimiser steps in a row are skipped for non-finite gradients (divergence). |
| `train.preference_max_regression` | float | 0.05 | Discard the preference phase when it raises validation loss by more than this fraction. |
| `train.log_every` | int | 10 | Steps between console/log lines. |
| `train.seed` | int | 0 | Seed for initialisation and data order (resume restores the exact stream). |
| `train.threads` | int\|auto | "auto" | BLAS threads for training (auto = every CPU). |
| `train.ram_fraction` | float | 0.35 | Largest share of free RAM a training batch may use. |
| `train.continual.lr_scale` | float | 0.3 | When training continues on new data, the peak LR is the previous peak times this. |
| `train.continual.new_data_weight` | float | 2.0 | Sampling weight of documents added since the last run (old data is still replayed). |
| `train.preference_steps` | int\|auto | "auto" | Optimiser steps of the preference (chosen vs rejected) phase after supervised training (auto = 10% of the run when the corpus has preference pairs, else 0). |
| `train.preference_batch` | int | 8 | Preference pairs per step. |
| `train.preference_beta` | float | 0.1 | Strength of the preference loss in offline training. |
| `train.preference_lr_ratio` | float | 0.1 | Preference-phase LR as a fraction of the supervised peak LR. |
| `train.resume` | bool | true | Continue from the last checkpoint / current model when one exists (false = start over). |

## compute

| option | type | default | description |
|---|---|---|---|
| `compute.backend` | str | "auto" | Where TRAINING runs: cpu; gpu (CuPy, fails with the fix if it cannot be used); or auto (a GPU when CuPy and a working device are present, else the CPU). Serving, export and online learning always use the CPU. One of: auto, cpu, gpu. |
| `compute.device` | int | 0 | Which GPU to use when there are several. |
| `compute.vram_fraction` | float | 0.8 | Largest share of the GPU's FREE memory a training run may plan for (the rest is headroom for the driver, fragmentation and other programs). |
| `compute.max_params` | int | 200000000 | Largest non-embedding parameter count autoscale will pick on a GPU (used in place of autoscale.max_params unless that was set explicitly). |
| `compute.max_micro_tokens` | int | 32768 | Largest forward/backward batch on a GPU (used in place of autoscale.max_micro_tokens unless that was set explicitly). |

## autoscale

| option | type | default | description |
|---|---|---|---|
| `autoscale.ctx_min` | int | 128 | Smallest context window autoscale will pick. |
| `autoscale.ctx_max` | int | 512 | Largest context window autoscale will pick (attention cost grows with its square on CPU). |
| `autoscale.ctx_percentile` | float | 90.0 | The window covers this percentile of conversation lengths. |
| `autoscale.tokens_per_param` | float | 20.0 | Training tokens per non-embedding parameter the model should be able to see (compute-optimal ratio). |
| `autoscale.min_params` | int | 300000 | Smallest non-embedding parameter count. |
| `autoscale.max_params` | int | 60000000 | Largest non-embedding parameter count autoscale will pick. |
| `autoscale.max_epochs` | float | 6.0 | Never size the model so that the corpus would have to be repeated more than this many times. |
| `autoscale.target_steps` | int | 1500 | Optimiser steps a run should have; sets the effective batch size. |
| `autoscale.max_micro_tokens` | int | 8192 | Largest forward/backward batch (bigger gains nothing on CPU BLAS). |
| `autoscale.max_effective_tokens` | int | 65536 | Largest effective batch (tokens per optimiser step). |
| `autoscale.lr_ref` | float | 0.003 | Peak LR at the reference width. |
| `autoscale.lr_ref_width` | int | 128 | Reference width for lr_ref; the LR scales with sqrt(ref/d_model). |
| `autoscale.eval_overhead` | float | 0.08 | Share of a run's time that validation may take. |
| `autoscale.min_batches_per_epoch` | int | 8 | A micro-batch never exceeds 1/this of the training data (tiny corpora get small batches). |
| `autoscale.warmup_fraction` | float | 0.03 | Share of the planned steps used for LR warm-up when train.warmup_steps is auto. |
| `autoscale.eval_passes` | int | 10 | Validation passes per run when train.eval_every is auto. |
| `autoscale.sample_records` | int | 2000 | Records sampled to measure conversation lengths and text size. |
| `autoscale.bench_steps` | int | 2 | Timed steps of the throughput benchmark. |

## online

| option | type | default | description |
|---|---|---|---|
| `online.enabled` | bool | true | Learn from live feedback while chatting. |
| `online.lr` | float\|auto | "auto" | Learning rate of online updates (auto = online.lr_ratio x the last training LR). |
| `online.lr_ratio` | float | 0.1 | See online.lr. |
| `online.steps_per_update` | int | 6 | Gradient steps per online update. |
| `online.batch_examples` | int | 8 | New examples per online step. |
| `online.replay_ratio` | float | 3.0 | Replay documents (from the original corpus) mixed in per new example, to prevent forgetting. |
| `online.min_events` | int | 4 | Feedback events needed to trigger a batch update. |
| `online.immediate_corrections` | bool | true | A typed correction triggers an update right away instead of waiting for min_events. |
| `online.interval_s` | float | 30.0 | How often the learner checks for pending feedback. |
| `online.max_regression` | float | 0.03 | Reject (roll back) an update when validation loss on replay documents rises by more than this fraction. |
| `online.probe_docs` | int | 48 | Validation documents used to gate each update. |
| `online.dpo_beta` | float | 0.1 | Strength of the preference loss for (chosen, rejected) pairs. |
| `online.preference_weight` | float | 0.5 | Weight of preference updates relative to supervised ones. |
| `online.implicit_accept_s` | float\|null | null | Treat an unrated reply as accepted after this many idle seconds (null = only explicit feedback counts). |
| `online.implicit_weight` | float | 0.5 | Training weight of implicitly accepted replies (see implicit_accept_s); explicit thumbs-up and corrections weigh 1. |
| `online.save_to_data` | bool | true | Write learned feedback examples into the data folder so later training runs keep them. |
| `online.data_folder` | str | "_online" | Sub-folder of the data folder that receives them. |
| `online.buffer_max` | int | 50000 | Maximum replies kept in the experience log (older ones are compacted away). |
| `online.keep_versions` | int | 5 | Online model versions kept for rollback. |
| `online.threads` | int\|auto | "auto" | BLAS threads the learner may use (auto = half of the CPUs). |

## generation

| option | type | default | description |
|---|---|---|---|
| `generation.max_new_tokens` | int\|auto | "auto" | Longest reply in tokens (auto = a quarter of the context window). |
| `generation.temperature` | float | 0.8 | Sampling temperature (0 = greedy). |
| `generation.top_k` | int | 50 | Keep only the k most likely tokens (0 = off). |
| `generation.top_p` | float | 0.95 | Nucleus sampling mass (1 = off). |
| `generation.repetition_penalty` | float | 1.1 | Penalty on recently used tokens (1 = off). |
| `generation.repetition_window` | int | 128 | How many recent tokens the penalty looks at. |
| `generation.stop` | list | [] | Extra strings that end a reply. |
| `generation.seed` | int\|null | null | Fix the sampling seed (null = fresh randomness). |
| `generation.history_turns` | int\|auto | "auto" | Most recent conversation turns kept in the prompt (auto = as many as fit). |

## server

| option | type | default | description |
|---|---|---|---|
| `server.host` | str | "127.0.0.1" | Bind address. Anything other than loopback requires the token. |
| `server.port` | int | 8787 | Preferred port. |
| `server.port_fallback` | bool | true | If the port is taken, use the next free one (the real address is in run/server.json). |
| `server.require_token` | bool | true | Require the access token on every state-changing request (protects against other web pages talking to your local server). |
| `server.token` | str\|auto | "auto" | Access token (auto = generated once and stored in state/token). |
| `server.cors_origins` | list | [] | Origins allowed to call the API from a browser. |
| `server.max_body_mb` | float | 64.0 | Largest request body (voice uploads). |
| `server.ui` | bool | true | Serve the built-in web chat UI at /. |
| `server.session_ttl_s` | float | 86400.0 | Idle chat sessions are dropped from memory after this long (their log stays on disk). |

## daemon

| option | type | default | description |
|---|---|---|---|
| `daemon.supervise` | bool | true | Run a tiny supervisor that restarts the worker if it crashes. |
| `daemon.restart_backoff_s` | list | [1, 2, 5, 10, 30, 60] | Seconds to wait before successive restarts. |
| `daemon.max_restarts_per_hour` | int | 20 | Give up after this many crashes in an hour. |
| `daemon.autotrain` | bool | true | Train automatically when enough new data has arrived. |
| `daemon.train_min_new_tokens` | int\|auto | "auto" | New tokens that trigger a continued-training run (auto = 2% of the corpus, at least 20000). |
| `daemon.train_cooldown_minutes` | float | 60.0 | Minimum gap between automatic training runs. |
| `daemon.train_budget_minutes` | float\|null | null | Wall-clock limit of automatic runs (null = train.budget_minutes). |
| `daemon.low_priority_training` | bool | true | Run background training at low CPU priority. |
| `daemon.hot_swap` | bool | true | Load a newly trained / validated model into the running server without restarting. |
| `daemon.log_max_mb` | float | 10.0 | Rotate log files at this size. |
| `daemon.log_backups` | int | 5 | Rotated log files kept. |

## fetch

| option | type | default | description |
|---|---|---|---|
| `fetch.endpoint` | str | "https://datasets-server.huggingface.co" | Rows API used to download Hugging Face datasets without extra libraries. |
| `fetch.hub` | str | "https://huggingface.co" | Hub URL (used for file downloads). |
| `fetch.token_env` | str | "HF_TOKEN" | Environment variable holding a Hugging Face token (needed for gated datasets). |
| `fetch.default_limit` | int | 20000 | Rows fetched per dataset unless --limit says otherwise. |
| `fetch.timeout_s` | float | 60.0 | Per-request timeout. |
| `fetch.retries` | int | 5 | Retries with exponential backoff. |
| `fetch.page_size` | int | 100 | Rows per request (the API allows at most 100). |
| `fetch.catalog` | path\|null | null | Your own catalog JSON (merged over the shipped one): name -> {repo, config, split, adapter, note}. |

## voice

| option | type | default | description |
|---|---|---|---|
| `voice.sample_rate` | int | 16000 | Sample rate used for speech recognition input. |
| `voice.input_device` | str\|null | null | Microphone (name fragment or index; null = system default). |
| `voice.output_device` | str\|null | null | Speaker (name fragment or index; null = system default). |
| `voice.audio_folder` | str | "audio" | Sub-folder of the data folder that holds speech datasets (audio + transcripts). The text pipeline skips it; `voice train-stt` reads it. |
| `voice.vad.threshold_db` | float\|auto | "auto" | Absolute speech threshold in dBFS (auto = the measured noise floor + margin_db, tracked continuously). |
| `voice.vad.margin_db` | float | 9.0 | How far above the measured noise floor speech must be when the threshold is auto. |
| `voice.vad.calibration_s` | float | 0.6 | Seconds of room noise measured before listening. |
| `voice.vad.min_speech_ms` | int | 200 | Shorter bursts are ignored. |
| `voice.vad.end_silence_ms` | int | 700 | Silence that ends an utterance. |
| `voice.vad.pre_roll_ms` | int | 300 | Audio kept from before speech was detected. |
| `voice.vad.max_utterance_s` | float | 30.0 | Hard cap on one utterance. |
| `voice.stt.engine` | str | "auto" | Speech recognition backend (auto = best available). One of: auto, builtin, vosk, faster_whisper, whisper_cpp, windows_sapi, command. |
| `voice.stt.model` | str\|null | null | Model path or name for the chosen backend (builtin: the model directory trained with `voice train-stt`). |
| `voice.stt.language` | str | "en" | Language code. |
| `voice.stt.command` | str\|null | null | For engine=command: shell template, {wav} is replaced by a 16 kHz mono WAV path; stdout is the transcript. |
| `voice.tts.engine` | str | "auto" | Speech synthesis backend (auto = best available, the built-in synthesizer is the last resort). One of: auto, builtin, espeak, say, windows_sapi, piper, pyttsx3, command. |
| `voice.tts.voice` | str\|null | null | Voice name / id for the backend (null = backend default). |
| `voice.tts.model` | str\|null | null | Model file for engines that need one (piper). |
| `voice.tts.rate` | float | 1.0 | Speaking speed multiplier. |
| `voice.tts.pitch` | float | 1.0 | Pitch multiplier. |
| `voice.tts.volume` | float | 1.0 | Volume multiplier. |
| `voice.tts.command` | str\|null | null | For engine=command: shell template with {text} and {wav}. |
| `voice.tts.cache` | bool | true | Cache synthesised phrases on disk. |
| `voice.tts.cache_max_mb` | float | 200.0 | Oldest cached phrases are deleted beyond this size. |
| `voice.tts.max_chars` | int | 1500 | Longer answers are spoken only up to this many characters. |
| `voice.convert.ffmpeg` | path\|null | null | Path to ffmpeg (null = look it up on PATH). Needed only for mp3/flac/ogg/m4a input. |
| `voice.builtin_stt.n_mels` | int | 40 | Mel bands of the built-in recogniser. |
| `voice.builtin_stt.hidden` | int\|auto | "auto" | Width of the built-in recogniser. |
| `voice.builtin_stt.layers` | int\|auto | "auto" | Depth of the built-in recogniser. |
| `voice.builtin_stt.epochs` | int | 30 | Training epochs of `voice train-stt`. |
| `voice.builtin_stt.batch` | int | 8 | Utterances per batch. |
| `voice.builtin_stt.lr` | float | 0.002 | Learning rate. |
| `voice.builtin_stt.max_seconds` | float | 12.0 | Longer clips are skipped. |
| `voice.builtin_stt.budget_minutes` | float\|null | 15.0 | Wall-clock limit of `voice train-stt` (null = until `epochs` are done). |
| `voice.builtin_stt.max_utterances` | int\|null | null | Use at most this many clips (null = all found). |
| `voice.builtin_stt.val_fraction` | float | 0.05 | Share of clips held out to measure the character error rate. |
| `voice.builtin_stt.val_min` | int | 8 | Hold out at least this many clips. |
| `voice.builtin_stt.augment` | bool | true | Randomly mask time / frequency bands and jitter the gain while training. |
| `voice.builtin_stt.kernel` | int | 5 | Convolution width (frames). |
| `voice.builtin_stt.dilations` | list | [1, 2, 4, 8] | Dilations, cycled over the layers (they set how much context each frame sees). |
| `voice.builtin_stt.min_char_count` | int | 3 | Characters seen fewer times than this in the transcripts are left out of the alphabet. |
| `voice.builtin_stt.win_ms` | float | 25.0 | Analysis window. |
| `voice.builtin_stt.hop_ms` | float | 10.0 | Hop between analysis windows. |
| `voice.builtin_stt.floor_db` | float | 50.0 | Dynamic range of the features: anything quieter than this below the loudest frame is treated as silence. |
| `voice.builtin_stt.pad_s` | float | 0.25 | Silence added either side of every utterance when recognising (and, randomly up to twice this, when training). |
| `voice.builtin_stt.clip_norm` | float | 5.0 | Gradient clipping. |
| `voice.builtin_stt.patience` | int | 6 | Stop after this many epochs without a better validation error. |
| `voice.tts.builtin_f0_hz` | float\|null | null | Base pitch of the built-in synthesizer (null = the measured profile from `voice profile`, else 118). |
| `voice.tts.builtin_tempo` | float\|null | null | Base speed of the built-in synthesizer in phonemes per second (null = profile, else 12). |
| `voice.talk.speak` | bool | true | Speak the answers aloud in `chat talk --voice`. |
| `voice.talk.stop_phrases` | list | ["stop listening", "goodbye", "quit voice chat"] | Saying one of these ends a voice conversation. |
| `voice.talk.listen_timeout_s` | float\|null | null | Give up waiting for speech after this long (null = wait forever). |
| `voice.talk.beep` | bool | true | Play a short tone when the assistant starts listening. |
