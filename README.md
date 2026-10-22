# Zero White

A numpy-only system with **no pretrained weights and no external model or API** (the chat part trains its own small transformer from the files you give it). Three parts share one rule: *nothing is hard-coded and nothing is trained on built-in data.*

| | what it is | start here |
|---|---|---|
| **Chat model** (v5) | a transformer written in NumPy, **really trained on the files you put in a data folder**, that keeps learning from your feedback, runs as a background service with a web UI and an HTTP API, and can listen and speak | [docs/CHAT.md](docs/CHAT.md) |
| **General model** (v4) | you hand it a task (table, environment, text, images); it **builds the agent for that task**, remembers what worked, and can rewrite its own search rules and, under test-gated reversible control, its own source | below |
| **Autonomy layer** (v6) | **`devour`** takes *any* file, folder or archive (repairing damage, setting aside what it cannot use, with reasons); **`organism`** is the loop that watches an inbox, decides what is worth improving, and changes how it searches when stuck; **`substrate`** measures the machine the limits come from | [Autonomy layer](#autonomy-layer-v6-devour-anything-decide-for-yourself) |

## Chat model: quick start

```bash
pip install -e .                                   # numpy is the only dependency
python -m zerowhite chat init  -w my-bot           # creates my-bot/data (put datasets here) and my-bot/config.json
python -m zerowhite chat fetch --list              # optional: public datasets you can download into data/
python -m zerowhite chat fetch alpaca -w my-bot    #   (or copy any json/jsonl/csv/parquet/txt/md/py files in yourself)
python -m zerowhite chat train -w my-bot --minutes 60      # real training; resumable; promotes only if better
python -m zerowhite chat talk  -w my-bot                   # chat; /good /bad /fix teach it online
python -m zerowhite chat start -w my-bot                   # background service: web UI + API, auto-ingest, auto-retrain, online learning
python -m zerowhite chat talk  -w my-bot --voice           # speak to it, hear it answer
python -m zerowhite chat voice train-stt -w my-bot         # train the built-in speech recogniser on data/audio
python -m zerowhite chat export -w my-bot --format safetensors   # Llama-layout weights for other tools
```

* **Folder-driven**: whatever is in `data/` is the training set - Alpaca, ShareGPT, OpenAI messages, preference pairs, PIPPA, code, plain text, `.gz`/`.zip`... New files are picked up automatically.
* **Not hard-coded**: ~230 documented options ([docs/CONFIG.md](docs/CONFIG.md)); model size, context, vocabulary, batch, learning rate, schedules and when to retrain are derived from your machine and data, with the reasons recorded.
* **Online learning**: corrections and thumbs-up become small gated updates (regression-checked, versioned, one-command rollback).
* **Background process**: supervised, detaches from the terminal, hot-swaps new models without restarting, optional start-at-login.
* **Voice**: system voices/espeak/piper if present, otherwise a built-in synthesiser; a recogniser you train yourself, or Whisper/Vosk if you installed one. ([docs/VOICE.md](docs/VOICE.md))
* **Conversion**: datasets between formats (`chat convert`), audio between formats and rates (`chat voice convert`), the model to safetensors / int8 (`chat export`).
* Guides: [CHAT](docs/CHAT.md) · [DATA](docs/DATA.md) (incl. the PIPPA "sister" persona recipe) · [VOICE](docs/VOICE.md) · [CONFIG](docs/CONFIG.md).

**What to expect.** It is trained on CPU in NumPy, so models are small and learn what is in *your* data; it will not match large pre-trained models. The machinery is tested end to end
(see "tests" below); how good a given assistant is depends on your data, time and hardware. Speech recognition in particular is only measured on synthetic speech so far ([docs/VOICE.md](docs/VOICE.md)).

---

# The general model (v4)

A numpy-only, **no-LLM** general model. You hand it a task; it **builds the agent for that task**, remembers what worked,
and can **rewrite its own search rules and (under test-gated, reversible control) its own source code**.

* **No LLM, anywhere.** Nothing here calls, loads or imitates a language-model service. A test fails the build if any LLM client
  library is imported or if network code exists outside the few named modules (dataset downloaders and the local service). (The optional `char_lm` family is a ~100k-parameter
  character model *trained from scratch on your text*; it is something Zero White can build for a text task, never something that drives it.)
* **No default agent or model.** There is no starting network, no preset hyper-parameters, no built-in step-size rule. On day one the
  proposal distribution is uniform-random; everything smarter is learned from results. A test enforces this.
* **Pure Python + NumPy.** `pip install numpy` is the only dependency.

```
task ──► PROFILE ──► INCUMBENT? ──► TRANSFER? ──► PROPOSE (learned prior) ──► RACE (successive halving) ──► KEEP
(table | env | text |   re-score the     try champions from      new (family, genome)      train, drop the weaker      zoo + prior update,
 image generation)      saved champion    similar tasks           pairs; random on day 1    half, train 2x longer…      test set reported once
```

## General model: quick start

```bash
pip install -e .              # or just run from the repo root with:  python -m zerowhite ...

python -m zerowhite solve --csv mytable.csv --target price           # any numeric table (classification / regression detected)
python -m zerowhite solve --env mountaincar cartpole                 # control problems
python -m zerowhite solve --env-file examples/catch_env.py           # YOUR environment, defined in a file
python -m zerowhite solve --dataset digits                           # real data (1,797 digits are bundled; works offline)
python -m zerowhite solve --text mybook.txt --sample 300             # learn to continue text
python -m zerowhite zoo --show                                       # every agent it built, with source code where it wrote some
python -m zerowhite doctor --workspace zw --fix                      # troubleshoot
```

Windows PowerShell is identical (`python -m zerowhite ...`). Everything is written under `--workspace` (default `zw`).

## Autonomy layer (v6): devour anything, decide for yourself

```bash
python -m zerowhite devour ~/Downloads/stuff --workspace zw --solve    # any files / folders / archives -> tasks -> trained agents; the rest set aside with reasons
python -m zerowhite devour --workspace zw --list                       # what was built, and every file that was set aside and why
python -m zerowhite organism --workspace zw --inbox ~/inbox --minutes 30 --self-modify hooks
python -m zerowhite organism --workspace zw --inbox ~/inbox --chat on --minutes 60   # 6.1: the chat model (zw/chat) is scheduled among the tasks too
python -m zerowhite organism status --workspace zw                     # what it decided, and why
python -m zerowhite chat gpu                                           # 6.2: check your GPU (parity with the CPU + real speed-up) before a long run
python -m zerowhite chat grow --layers 1 --ffn 64 --heads 1            # 6.3: make the live chat model bigger WITHOUT changing what it computes, then keep training
python -m zerowhite forage --allow-host docs.python.org URL...         # 6.3: read named pages from hosts YOU allow into an inbox folder (off unless you name hosts)
python -m zerowhite kernels                                            # 6.4: what the lazy-graph backends (generated C, generated CUDA) can do on THIS machine
python -m zerowhite codec train|encode|decode                          # 6.4: a small learned audio codec (residual-VQ tokens), on synthetic speech or your own wavs
python -m zerowhite share                                              # 6.4: write the zoo's discoveries and the system's own source into the chat data folder as text
python -m zerowhite engine-tier                                        # 6.4: OPT-IN single-edit code mutations of nn.py / engine.py under the full judge (dry run unless --apply)
python -m zerowhite organism --metric bits --sleep --forage-host H --forage-wishlist FILE   # 6.4 opt-ins: bits currency, consolidation, curiosity-driven reading
python -m zerowhite chat grow --experts 4 --top-k 2                    # 6.4: turn the feed-forward blocks into a mixture of experts (same outputs at first)
python -m zerowhite substrate --workspace zw                           # the machine measurements that size everything

python examples/messy_inbox.py /tmp/inbox                              # try it: 12 files, each broken differently (truncated gzip, cp1252 + decimal commas, ...)
python -m zerowhite devour /tmp/inbox --workspace zw --solve --allow-code
```

* **Devour** ([docs/OMNIVORE.md](docs/OMNIVORE.md)): file type is decided by content; encodings, ragged / quoted / semicolon / decimal-comma tables, JSON Lines, truncated JSON, SQLite, `.npy`, truncated gzip and
  zip are read or repaired; archives are opened with zip-bomb limits; source files are never touched; your environment files run only with `--allow-code`. The test suite feeds it random bytes and
  truncated / bit-flipped copies of every supported format and asserts: no exception, one recorded outcome per file.
* **Organism** ([docs/ORGANISM.md](docs/ORGANISM.md)): new task -> cheap probe; then the task with the most learning progress per second; budget doubles after an attempt that gains nothing; a task at its
  ceiling or with no progress is left alone; when *everything* has plateaued it changes how it searches (rewrite a search rule under the test-gated hooks, or invent environments) and re-opens the tasks if that wins.
  Every decision is logged with its reason.

### What "agent" means here

| family | task kinds | what the agent *is* |
|---|---|---|
| `mlp` | classification, regression | a small network, size and learning rate chosen by search |
| `selfmod_net` | classification | the original Zero White net that grows/prunes its own architecture |
| `actor_critic` | environments | policy-gradient agent with optional curiosity |
| `program_policy` | environments | **a Python function it wrote** (e.g. CartPole: `-1.59 if -0.73 > obs[3] else obs[2]`) |
| `program_formula` | regression | **a formula it discovered**, over the most informative columns |
| `char_lm` | text | character-level model trained from scratch |
| `cvae` | image generation | class-conditional VAE (judged by a classifier it builds first) |

Which family gets built for a task is *not* hard-coded: the learned prior (`forge/prior.py`) proposes it, and updates from every race.
After a few control tasks it already prefers `program_policy` there (mean skill 0.79 vs 0.60 for `actor_critic` in our run).

## Self-modification: three levels, each with a different safety story

| level | what it rewrites | gate before anything changes | undo |
|---|---|---|---|
| **0 – learned prior** | which agents to propose | none needed (it only reorders proposals) | delete `forge/prior.json` |
| **1 – evolved hooks** (`evolved/*.py`) | its own *search rules* (step-size update, parent selection) as real Python source | static AST safety gate → score on train landscapes → **held-out** landscapes & fresh seeds must beat the live hook by a margin | `selfmod rollback --hook NAME` |
| **2 – source patcher** (`selfmod/patcher.py`) | numeric hyper-parameters in 8 core modules, in the real `.py` files | single-literal AST diff → benchmark on *select* seeds → *confirm* seeds → one-shot *final* seeds with a margin → **whole test suite passes in a sandboxed copy** → (only with `--apply`) written, tests re-run on the real tree, auto-restored if they fail | `selfmod rollback --file NAME.py` |

Rules that hold at every level:

* **A constitution it cannot edit**: `safety.py`, `history.py`, `sandbox.py`, `patcher.py`, `astmut.py`, `fileio.py`, `doctor.py`, `evolved/__init__.py`
  are outside the patcher's reach and sealed by hash (`doctor` checks the seal).
* **It cannot improve its score by changing the measurement**: constants that define how a model is scored (`episodes`, `per_class`, `val_cap`,
  `step_cost`, seeds…), structural dimensions and plain compute budgets are never patch sites.
* **Everything is versioned**: every installed hook and every patch goes into an append-only, hash-verified history; rollbacks are themselves logged.
* **Evolved code is boxed**: no imports, loops, dunders, file or process access (`selfmod/safety.py`); runs in subprocesses with timeouts, a scrubbed
  environment and (POSIX) rlimits (`selfmod/sandbox.py`). *This is containment for accidents, not a security boundary against a hostile actor* — for
  untrusted use, run inside a container or VM. `doctor` prints exactly what is enforced on your OS.

```bash
python -m zerowhite selfmod hooks --workspace zw --evals 600       # level 1: evolve its own search rules
python -m zerowhite selfmod patch --workspace zw --evals 40        # level 2, dry run: nothing is written
python -m zerowhite selfmod patch --workspace zw --evals 40 --apply
python -m zerowhite selfmod status | history | rollback --hook select_parent
python -m zerowhite evolve --csv a.csv --env cartpole --minutes 30 --self-modify hooks   # unattended: solve + improve itself
python -m zerowhite poet --rounds 8                                # invent new environments + agents for them
```

## What it actually achieved (measured, single runs, Linux, Python 3.12, NumPy 2.4)

| task | result on **held-out** data / fresh episodes | what it built | time |
|---|---|---|---|
| digits (real, bundled) | 97.8 % test accuracy | MLP 2×114 | 3 s |
| CSV with hidden formula `3a − sin 2b + 0.5cd` | R² 0.992 (test) | MLP; the formula family reached R² ≈ 0.93 on validation | 24 s |
| CartPole | 199.5 / 200 mean return, 96.7 % success | a **one-line program it wrote** | 76 s |
| MountainCar | 100 % success, return −109 | an evolved program | 33 s |
| *Catch* (an environment it had never seen) | 9.87 / 10 | an evolved program | 13 s |
| POET-lite (8 rounds) | invented 4 new physics variants (parent's agent skill 0.38–0.95 there); child agents recovered to ≈ 0.99 | programs seeded from their parents | 24 s |
| **hook search**, `select_parent` | held-out score 0.720 → 0.771 (+0.051) → **installed** | "always breed from the current best" (the search found this itself) | 48 s |
| **hook search**, `sigma_update` | best found 0.733 vs genesis 0.720 and textbook 1/5-rule 0.734 → **rejected** (below the margin) | — | 33 s |
| **source patcher**, 52 real candidates in 2 runs | **0 accepted.** One looked good on its selection seeds (+0.018) and was *worse* on unseen seeds (0.7314 vs 0.7358); another gained +0.002, below the margin | — | 42 s, 104 s |

Read that last row carefully. The patcher *works* (the full apply → re-test → history → rollback path was exercised on the real benchmark and
test suite, on a copy), but on this benchmark it found nothing worth applying: the shipped hyper-parameters are already near a local optimum for it, and the
held-out gate correctly refused noise-level wins. Do not expect level 2 to find big improvements; its value is that it **can** change source safely when
one exists.

## Honest limits

* The hook benchmark is a **proxy** (cheap synthetic landscapes); a rule that wins there may not win on your real training. Held-out landscapes and fresh seeds
  reduce, not remove, that risk.
* The patch benchmark only exercises the classifier, the actor-critic agent and the Evolver, so constants in other modules (VAE, language model, curiosity) can never
  show a gain and are never accepted.
* Program agents read at most 24 numbers and are scalar-output (action = sign / floor of the value mod #actions). Since 6.3 a wider observation is compressed to 24 whitened PCA components fitted on random-policy probes instead of being
  truncated (on a 40-number task whose signal sat in the last 16 numbers, 4 seeds scored -0.12..0.01 truncated and 0.88, 0.79, 0.34, 0.00 projected: better, not reliable); PCA is linear and unsupervised, so it is a better default than truncation, not an encoder. Great on small control
  problems, still weak on high-dimensional ones like `gridworld`; the prior learns that, which is the point.
* Evolved programs are correct but not always pretty (`np.minimum` chains). `simplify` folds constants and drops dead temporaries, nothing smarter.
* `skill` is normalised per task (accuracy vs majority class, R², return vs random). Comparable between candidates on one task; only roughly comparable across tasks.
* Results above are single seeded runs on one machine, not averages with error bars.
* `--env-file` executes **your** file in-process with your permissions.
* Windows: `file://` URLs are built with `Path.as_uri()` (verified with `PureWindowsPath` and by `doctor`'s URL round-trip self-test). Development and the results above were on Linux;
  the CI matrix includes `windows-latest`.
* v6.2: GPU training (chat model only, through CuPy) is tested against a strict fake device, not on a real card by the author: run `zerowhite chat gpu`. float32, one GPU, not bit-reproducible; the general model still runs on NumPy.
* v6.4 (read CHANGELOG.md for the measured numbers and the negative results): the generated C kernels are NOT faster than numpy on the build machine, and the CUDA path has never run on a GPU; the audio codec was trained and tested only on synthetic
  speech (4 dB SNR at 4.2 kbps: better than PCA at the same rate, not a good codec); the attention encoder compresses 10x better than PCA but did not help evolved programs; the learned delimiter ranker is validated on 16 independent files; MoE, sleep, bits and curiosity
  foraging are verified against scripted organisms, a local server and finite differences, not over long real runs; the engine tier is opt-in and will mostly find nothing.
* v6.3: the Windows Job Object limits, the opt-in forager's real-network path and the calibrated cost forecast are verified against fakes, a local server and simulated cost shapes, not on a real Windows machine, the public internet or a long real
  run (see [CHANGELOG.md](CHANGELOG.md)). Model growth is exact (tested to 1e-10) but says nothing about whether a bigger model is worth it on your data.
* v6: the organism schedules the general model and (since 6.1) the chat model, but they share the task list, clock and inbox prose, not hooks or the zoo; its cost estimates are calibrated from each task's own history since 6.3 but still start from a
  guess; a hook "win" is on a proxy benchmark (and since 6.3 must also be statistically significant); heuristics in `devour`
  are rules, not learned. See the *Honest limits* sections of [docs/ORGANISM.md](docs/ORGANISM.md) and [docs/OMNIVORE.md](docs/OMNIVORE.md).

## Troubleshooting

`python -m zerowhite doctor --workspace zw` checks Python/NumPy/encodings, that every module imports, the constitution seal, the sandbox's real guarantees, a `file://`
round trip, the dataset cache, stale temp files, checkpoints, zoo pointers, history hashes, and whether any self-written hook has made things worse than genesis.
`--fix` applies only safe repairs and never deletes evidence (corrupt files are renamed `*.corrupt-<time>`; hooks are rolled back, not erased).

| symptom | cause / fix |
|---|---|
| `UnicodeEncodeError` on Windows | all files are written as UTF-8; set `PYTHONUTF8=1` for console output too (the CLI also reconfigures stdout) |
| dataset download fails | the message prints the exact path to place the file by hand; the digits set is bundled and needs no network |
| `PermissionError` replacing a file (Windows) | antivirus/indexer holding the file; writes now retry briefly, exclude your workspace from scanning if it persists |
| `doctor`: "protected files changed" | you edited a constitution file: re-seal with `doctor --seal`; otherwise `git checkout` it |
| `selfmod patch` says it needs a checkout | the test suite is the gate; run from a git clone, not an installed package |

## Layout

```
zerowhite/
  forge/      task.py envs.py families.py prior.py zoo.py generalist.py poet.py     the general model
  selfmod/    safety.py astmut.py islands.py history.py sandbox.py hooks.py patcher.py bench.py
  evolved/    sigma_update.py select_parent.py          source files the system may rewrite (genesis = no-ops)
  chat/       config.py corpus/ tokenizer.py model.py train.py online.py generate.py export.py service/ ...   the chat model (v5)
              backend.py gpucheck.py                                                                        v6.2: NumPy or CuPy (GPU) arrays; the on-your-machine GPU check
  voice/      audio.py vad.py g2p.py formant.py tts.py stt.py ctc.py train_stt.py devices.py ...           speech in / out
  omnivore/   sniff.py tables.py archives.py library.py devour.py limits.py                       v6: ingest anything without breaking
  substrate.py organism.py chat_task.py cli_v6.py         v6: machine probe, the autonomous loop, the chat model as one of its tasks, their commands
  cli.py cli_v4.py doctor.py                              interface + troubleshooting
  (v3 core)   zero_white.py self_model.py architecture.py meta_optimizer.py curiosity.py evolve.py engine.py nn.py
              agents.py generative.py factory.py datasets.py tasks.py checkpoint.py fileio.py report.py
tests/        test_basic.py test_v3.py test_v4.py test_chat.py test_voice.py test_v6.py test_v61.py test_gpu.py   ~210 tests, a few minutes:  python tests/run_all.py
examples/     persona_sister/ (config + dataset.json for a "like a sister" companion)   catch_env.py (template for your own environment)
docs/         CHAT.md DATA.md VOICE.md CONFIG.md  OMNIVORE.md ORGANISM.md  ARCHITECTURE.md SELF_MODIFICATION.md FORAGE.md KERNELS.md REFERENCES.md
```

## License / data

No license file is included yet: choose one before publishing. The bundled digits sample is a subset of the UCI Optical Recognition of Handwritten Digits data
(CC BY 4.0); see `zerowhite/data/README.txt`. Background on what was taken from the reference repositories: `docs/REFERENCES.md`.
