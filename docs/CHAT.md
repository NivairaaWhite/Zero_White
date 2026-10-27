# The chat model

A decoder-only transformer written in NumPy — tokenizer, forward pass, **hand-written backward pass**, optimiser, KV-cache generation — trained
on the files in your data folder. No PyTorch, no pre-trained weights, no calls to any language-model service. (Optional speech engines are
described in [VOICE.md](VOICE.md); none is required.)

```bash
pip install -e .                                  # numpy is the only dependency
python -m zerowhite chat init   -w my-bot         # creates my-bot/data, my-bot/config.json
cp my_dataset.jsonl my-bot/data/                  # or: python -m zerowhite chat fetch alpaca -w my-bot
python -m zerowhite chat train  -w my-bot --minutes 60
python -m zerowhite chat talk   -w my-bot         # type to it; /good /bad /fix teach it
python -m zerowhite chat start  -w my-bot         # background service: web UI at the printed address, keeps learning
```

Windows PowerShell is identical. All files are UTF-8; paths work with `\`.

## Nothing is hard-coded

Every setting lives in one schema ([CONFIG.md](CONFIG.md), 200+ options, each with a type, default and description). Layers, later wins:
default < `<workspace>/config.json` < environment (`ZW_TRAIN__LR=0.002`) < `--set train.lr=0.002` < a folder's `dataset.json`.
Many options are `auto`: the model size, context length, vocabulary size, batch size, learning rate, warm-up, evaluation cadence, thread count and
the point at which the background service retrains are **derived from your machine (cores, RAM, measured speed) and your data** and written to
`state/resolved_config.json` with the reason for each. `zerowhite chat config --explain` shows every value and where it came from;
`chat train --plan` prints the plan without training.

## How training works (`chat train`)

1. **Ingest** the data folder incrementally ([DATA.md](DATA.md)): normalise, filter, de-duplicate, shard.
2. **Tokenizer**: byte-level BPE trained on *your* text (so any language or code works, and nothing can be "unknown"). Vocabulary size is derived from the corpus size.
   User text can never produce the special role tokens, so a message cannot impersonate the system or the assistant.
3. **Plan**: model width/depth from the data size and the measured throughput of this machine (CPU, or the GPU when one is used: [Training on a GPU](#training-on-a-gpu)) within `train.budget_minutes`; capped by `model.max_params`.
4. **Train**: next-token loss on **assistant text only** (the prompt is context, not target), documents packed without attending across boundaries,
   AdamW, warm-up + cosine decay, gradient clipping, a held-out validation split that is never trained on, early stopping, a divergence guard that skips bad steps.
5. **Preference phase** (if the data has chosen/rejected pairs): short DPO phase against a frozen copy, gated on not hurting language modelling.
6. **Promote only if better.** The result becomes the live model only when its validation loss is no worse than the live model's on the same data. A worse run is kept
   as a version but never goes live. A first model is always promoted.

Training is **resumable and continual**: the next `chat train` continues from the live model; new files get extra weight (`train.continual.new_data_weight`) and a
lower learning rate (`train.continual.lr_scale`) so existing skills are not overwritten. Ctrl-C (or `chat stop`) checkpoints cleanly.
`chat train --fresh` starts a new model. Versions are in `models/weights/` (`chat versions`); `chat rollback` points the live model back at the previous one.

## Online learning (it improves while you talk to it)

* Feedback is collected from the terminal (`/good`, `/bad`, `/fix TEXT`, `/again`), the web UI, the API, or `chat feedback REPLY_ID --fix "..."`.
* What each signal teaches: a **correction** becomes a supervised example *and* a preference pair (corrected answer > what it said); a **thumbs-up** becomes a supervised
  example (and, if the reply was a regeneration, a preference pair over the answer it replaced); a thumbs-down on its own teaches nothing (the model is not told what
  *would* have been right) — pair it with `/fix`, or press `/again` and rate the better answer. Optionally (`online.implicit_accept_s`, off by default) a reply you simply
  continued the conversation after counts as a weak thumbs-up (`online.implicit_weight`).
* An update runs a few small optimiser steps on a **copy** of the live model, mixing in replayed examples from the training corpus so it does not forget (`online.replay_ratio`).
* **Regression gate**: the copy goes live only if its loss on a probe set of old training data has not grown by more than `online.max_regression`.
  Otherwise the update is rejected (kept in `state/online/rejected.jsonl`) and the model is untouched. Accepted updates are versions you can roll back.
* Accepted feedback is also written to `data/_online/` as ordinary dataset files, so the next full training run keeps what it learned.
* Feedback passes through the same quality, safety and PII filters as datasets.
* Updates run automatically in the background service every `online.interval_s` once `online.min_events` pieces of feedback exist; `chat learn --force` runs one now.
  Turn it all off with `online.enabled=false`.

## The background service

`chat start` launches a supervised process that detaches from your terminal (Windows: no console window) and survives you closing it:

* HTTP API + web chat UI (below), bound to `127.0.0.1` by default;
* watches the data folder and ingests new files;
* retrains by itself when enough new data arrived (`daemon.train_min_new_tokens`, `daemon.train_cooldown_minutes`) in a separate low-priority process, and **hot-swaps** the new
  model into the running server when it is validated and better — no restart, conversations continue;
* applies online learning;
* a supervisor restarts it with back-off if it crashes (`daemon.restart_backoff_s`, `daemon.max_restarts_per_hour`); logs rotate (`chat logs`).

`chat status` shows everything at a glance; `chat stop` asks it to finish (a running training job checkpoints first and resumes later).
`chat autostart install` registers it to start when you log in (Windows Task Scheduler, systemd user unit, or a macOS LaunchAgent; `show` prints what it would write).
`chat talk` and `chat say` use the running service automatically (same model, same sessions); `--local` loads the model in the terminal process instead.

### HTTP API

Base URL is printed by `chat start` (default `http://127.0.0.1:8787`; if the port is busy the next free one is used). The token is in `<workspace>/state/token`
(or set `server.token`). Send it as `X-ZW-Token: ...` or `Authorization: Bearer ...`.

| endpoint | purpose |
|---|---|
| `GET /` | web chat UI (also speaks and listens if voice is available) |
| `GET /api/health`, `GET /api/status` | liveness; model / data / training / online-learning state |
| `POST /api/chat` `{"message", "session"?}` | reply, streamed as server-sent events (`delta` ... `done` with `reply_id`) |
| `POST /api/regenerate` `{"session"}` | another answer to the last message |
| `POST /api/feedback` `{"reply_id", "rating": 1/-1, "correction"?}` | teach it |
| `GET/POST /api/session` | read / start a conversation |
| `POST /api/learn`, `POST /api/train`, `POST /api/rollback`, `GET /api/versions`, `POST /api/reload` | online update now; request a training run; undo; list; reload |
| `GET /api/voice`, `POST /api/stt`, `POST /api/tts`, `POST /api/voice` | speech: see [VOICE.md](VOICE.md) |

**Security model.** It is a local app, not a hardened internet service. POST requests (and every request if you bind beyond loopback) need the token;
the `Host` header must be a local name (defeats DNS-rebinding); cross-origin requests are refused unless listed in `server.cors_origins`; request bodies are
size-limited; responses carry `nosniff`, `frame-deny` and a strict content-security-policy; there is no file access by path. To expose it on a network set
`server.host`, keep the token, and put it behind a TLS reverse proxy — it speaks plain HTTP.

## Using the model elsewhere (`chat export`)

| `--format` | contents |
|---|---|
| `safetensors` (default) | `model.safetensors` with Hugging Face **Llama** tensor names + `config.json` + `tokenizer.json` (our own BPE format, not a `transformers` tokenizer) |
| `int8` | per-row int8 weights (about 4x smaller) that `zerowhite` loads back |
| `npz` | our native file |

The architecture is exactly Llama-style (RMSNorm, rotary half-split embeddings, SwiGLU, tied embeddings, no biases); a test re-implements the Llama forward pass
independently from the exported tensors and checks that the logits match. To *use* the export in other software you still need a tokenizer for it;
the byte-level BPE in `tokenizer.json` is simple to port but is not loaded by `transformers` out of the box.

## Letting the organism schedule training

`chat train` and the service's auto-training are two ways to train; the [organism](ORGANISM.md#the-chat-model-as-a-task) is a third. With `python -m zerowhite organism --workspace zw --chat on --minutes 60`
the chat workspace `zw/chat` becomes one task among the organism's others: it is trained when its expected learning progress per second beats theirs, the prose in the organism's inbox is mirrored into
`zw/chat/data/_omnivore/` so one dropped file teaches both models, and a promoted model is hot-swapped into a running service. Training takes one lock, so a service and an organism queue rather than collide;
set `daemon.autotrain=false` if the organism should be the only scheduler.

## Training on a GPU

Training can run on a CUDA GPU through [CuPy](https://cupy.dev), a drop-in for NumPy. Only *training* moves: serving, chat, export and online learning load the weights from disk and stay on the CPU, so a model
trained on a GPU is an ordinary model file.

```
pip install cupy-cuda12x            # or cupy-cuda11x; `nvidia-smi` shows which CUDA your driver supports
python -m zerowhite chat gpu        # checks it HERE: kernels, model and optimiser parity with the CPU, and the real speed-up
python -m zerowhite chat train --fresh --minutes 120      # compute.backend=auto (the default) now uses the card
```

* `compute.backend`: `auto` (default: a GPU when CuPy and a working device exist, else the CPU, quietly), `gpu` (a demand: stops with the fix if it cannot be honoured, rather than training for hours on the CPU by accident),
  `cpu`. `--gpu` / `--cpu` on `train` and `prepare` are shortcuts; `compute.device` picks the card.
* **The plan changes with the card**: the throughput benchmark runs on the device (synchronised, since a GPU runs asynchronously), memory is planned from its *free VRAM* (`compute.vram_fraction`, default 0.8) and the CPU-sized caps
  (`autoscale.max_params` 60 M, `autoscale.max_micro_tokens` 8192) give way to `compute.max_params` (200 M) and `compute.max_micro_tokens` (32768), unless you set the CPU ones yourself, which always wins.
* **An existing model keeps its size unless you grow it.** Its shape was fixed by the first run, so a model trained on the CPU stays small after you get a card. Since 6.3 `chat grow` adds capacity without losing what was learned; or use `chat train --fresh` for a bigger one (it trains a new model; the old one stays as a version
  and is only replaced if the new one is better on the same validation data).
* **Out of memory** stops with a message naming the knobs (`train.batch_tokens`, `compute.vram_fraction`, `model.target_params`); the last checkpoint is kept and the next `chat train` resumes from it.
* **What is and is not verified.** The device code path is tested against a strict fake device (it refuses host arrays, offers only CuPy functions, counts every transfer and synchronisation), and the CPU path is bit-identical to before. A real
  card has not been exercised by the author's tests, which is why `chat gpu` exists: run it first; it exits non-zero with the numbers if the card disagrees with the CPU.
* **Honest caveats.** float32 only (CuPy can use TF32 tensor cores with `CUPY_TF32=1`, faster and slightly less precise; `chat gpu` reports the difference). GPU training is **not bit-reproducible** (the embedding gradient uses atomic adds, whose
  order varies), unlike the CPU path. The first steps are slow while CuPy compiles its kernels (cached afterwards). The implementation is not fused (no FlashAttention), so it reaches a modest share of the card's peak and attention memory grows with
  the square of the context. One GPU only. NVIDIA through CuPy's CUDA build is the target; AMD (CuPy's ROCm build) uses the same code but is untested; Apple Silicon has no CuPy and is not supported.

## Growing the model (`chat grow`, 6.3)

An existing model used to keep its size for life. `zerowhite chat grow --layers 1 --ffn 64 --heads 1` now turns the live model into a bigger one that computes **exactly the same function** (same logits, same loss),
saved as a new version; the next `chat train` continues from it, and `chat rollback` returns to the old one. How, per kind of growth (`chat/grow.py`):

* **layers** - a new residual block whose output projections (`wo`, `wd`) are zero adds nothing to the residual stream; its other weights start as a copy of the block below it (`--init random` for random ones).
* **ffn** - new SwiGLU hidden units, with zero rows in `wd`.
* **heads** - new heads need a wider residual stream (`head_dim` is fixed). The new channels hold zeros, the RMSNorm gains are rescaled by sqrt(D_old / D_new) and its epsilon by D_old / D_new so the norm is unchanged for every input.

Tested: logits and loss identical to ~1e-10 in float64 for every combination (tied and untied heads), new layers get a gradient at once, a grown workspace goes through `prepare` and a real `Trainer` run and keeps its shape.
**Not done:** mixture-of-experts blocks, shrinking, a new vocabulary or context length, and any automatic decision of *when* to grow (you run the command). Growth costs nothing in quality at the moment it happens and
buys nothing until training uses the new capacity; whether a bigger model is worth its slower steps on your data is yours to judge (`chat train` reports the plan and the validation loss). It refuses to run while a training run holds the lock.

## Mixture of experts (6.4, `chat grow --experts E --top-k K`)

`chat/grow.py: to_moe` turns every dense feed-forward block into E copies behind a small random router; each token uses its `top_k` experts, whose gates (a softmax over the chosen logits) sum to 1, so identical experts reproduce the dense output exactly (tested to 1e-15). The
experts then diverge because different tokens give them different gradients (tested). Training adds Switch-Transformer's load-balancing loss (`moe_aux`, 0.01): with the loss, the busiest of 4 experts handled 28% of tokens (even = 25%), without it 57% (3-4 seeds, router started biased).
My first version of that loss (on soft probabilities) did **not** balance the hard routing; the measurement caught it and it was replaced. Backpropagation was checked against finite differences (top-1, top-2, all experts, with and without the loss), and the KV-cache inference path against the
training forward. A grown workspace goes through `prepare` and the real `Trainer`. Not done: GPU parity for the MoE path (it uses array-module-neutral code, tested only with the fake device), `chat export` of a MoE model (refused with a message: Llama's MLP is dense), sparse dispatch for speed (experts are looped, not batched), and the memory estimator in `autoscale` still assumes dense blocks.

## One stream for every modality, and shared memory (6.4)

`chat/modal.py` writes tables, audio codes (`voice/codec.py`) and RL episodes as plain text lines with a typed marker (`[table name] a | b`, `[row] ...`, `[audio] a12.7.3.9 ...`, `[rl] o=... a=1 r=1.0`) with exact readers, so the ordinary tokenizer and trainer carry them.
`chat/shared.py` (and `zerowhite share`) writes the forge's zoo (programs, skills) and the system's own source (module docstrings and signatures) into the chat data folder as text; the real ingestion path takes them in (tested: 40+ new documents, no errors).
`forge/system2.py` closes the loop the other way: any text generator can propose policy programs (`chat_proposer` parses code out of free text); each must pass the sandbox's AST safety rules, then successive halving on the real task ranks them, and the winners
seed a program agent's islands. **Not done / not shown:** a shared learned embedding space, any evidence that a small chat model trained on this text does better on tables or audio, any evidence it proposes useful programs (the tests use scripted proposers, one deliberately poor), and feedback in chat updating the forge's priors.

## Honest limits

* **Measured on the build machine (2 CPU cores, no GPU):** Alpaca (52k examples, 4.3 M tokens), a 1.6 M-parameter model reached held-out perplexity 228 (loss 5.43) after 5.5 minutes of training and 140 (loss 4.94) after a further 7 minutes of continued training, at about 5,000 tokens/s. Its answers have the right *shape* (numbered tips, short sentences) but little sense: it has started to learn the format, not to write sensible answers; useful quality needs hours of training, more data for the size you pick, or a faster machine. The pipeline is the same either way.

* It is trained in NumPy on the CPU, or since 6.2 on a CUDA GPU through CuPy (unmeasured by the author: run `chat gpu`). Models are small (the default cap is 120 M parameters; realistic models on a laptop are far smaller) and learn
  what is in *your* data. **Do not expect the fluency or knowledge of large pre-trained models.** The tests prove the machinery learns real tasks end to end
  (a tiny model learns a copy task to near-zero loss in seconds; gradients match finite differences; exports match an independent reference), not that a given dataset
  yields a good assistant. Quality is the data, the time you give it, and the model size your machine allows.
* Online learning makes *small, safe* changes. It will absorb a correction it is shown a few times; it will not turn a weak model into a strong one.
* The safety and PII filters are heuristics (see [DATA.md](DATA.md)).
* The persona/"sister" behaviour is only as good as the persona data and the system prompt you give it.
* Windows, macOS and the autostart installers were written to the platforms' documented interfaces but **were only executed on Linux** in the environment this was built in; CI covers the test suite on all three.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| "There is no usable training data" | the data folder is empty, files are still being copied (wait `data.settle_s`), or every record was filtered (the message lists drop counts) |
| training says another run is active | `chat stop`, or wait; `--wait SECONDS` makes `chat train` queue behind it |
| out of memory | lower `train.ram_fraction` or `model.max_params`; the plan prints the memory estimate. On a GPU the message names the knobs (`train.batch_tokens`, `compute.vram_fraction`) and the run resumes from its checkpoint |
| the GPU is not used | `chat gpu` says why (CuPy missing or not matching your CUDA, no visible device); `compute.backend=gpu` turns the silent CPU fallback into an error |
| a GPU run is slow at first | CuPy compiles its kernels on first use (cached in `~/.cupy`); later runs start fast |
| a candidate "was not promoted" | it was worse than the live model on the same validation data; train longer or add data; nothing was lost |
| a second `chat train` improves the model only slowly | continued runs use a lower learning rate (`train.continual.lr_scale`, default 0.3) to protect what the model knows; for a model that is still far from converged, run with `--set train.continual.lr_scale=1.0` or use a larger `--minutes` |
| `chat talk` says no model | run `chat train` first (a first model is always promoted) |
| web UI cannot connect | the port was busy and the service moved on: `chat status` prints the real address |
| anything else | `chat logs daemon`, `chat logs train`, and `zerowhite doctor --workspace <ws>` |
