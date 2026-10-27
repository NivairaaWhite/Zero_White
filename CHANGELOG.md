# Changelog

## 6.5.0, CI follow-up (first run on GitHub Actions)
Four legs of the matrix failed on the first push. Causes and fixes:

| leg | cause | fix |
|---|---|---|
| ubuntu, 3.9 | `ubuntu-latest` now means Ubuntu 26.04, for which `actions/setup-python` has no Python 3.9 ("version '3.9' ... was not found for Ubuntu 26.04"); nothing of ours ran | the workflow pins `ubuntu-24.04` |
| windows, 3.9 and 3.12 (3-4 tests of `test_v65`) | **a real bug in `memwatch.needed(platform)`**: it also tested the HOST's `os.name`, so on a Windows host `needed("darwin")` answered `False`. `memwatch.py` is sealed, so `CONSTITUTION.sha256` was re-sealed. Three tests also assumed POSIX / a precise clock: forcing `memory="watcher"` is meaningless on Windows (the Job Object is the mechanism there and prints no watcher message), and a 1 ms `Event.wait` takes about 15 ms on Windows, so "more than 5 samples in 50 ms" could not hold | `needed()` looks only at the platform it is asked about (a test now makes the module believe it is on Windows, so Linux CI catches a regression); the three tests are Windows-aware and wait for samples instead of for the clock |
| macos, 3.9 (`Segmentation fault: 11`, exit 139) | **not reproduced** (no Mac and no 3.9 interpreter was available). stdout was block-buffered into a pipe, so the log lost the lines just before the crash and does not say which test died. | `tests/run_all.py` now enables `faulthandler` (a crash prints the Python stack and test name) and flushes every line; the macOS leg runs with `VECLIB_MAXIMUM_THREADS=1`, and test threads get a 16 MB stack (macOS gives secondary threads 512 KB). **These two are mitigations for the most likely causes, not a confirmed fix**: if that leg still crashes, the stack in its log names the test. |

## 6.5.0
**Fixes for the three failures in the 6.4 CI / Colab logs, and the ten abilities on the 6.5 wishlist.** Tests: 436 (303 in 6.4 + 133 new, in `tests/test_v65*.py`), all passing in about 6 minutes on one CPU. Two older tests were updated because what they assumed changed on purpose (see "Test changes" below).

### The three failures
| failure | cause | fix |
|---|---|---|
| macOS CI: memory-limit test | `RLIMIT_AS` is not enforced on macOS, so a 900 MB allocation under a 256 MB limit succeeded | the sandbox now picks a **mechanism** per OS (`rlimit_as`, Windows job object, or the parent's **watcher**, which reads the child's resident size through `libproc` / `/proc` / `ps` and kills it once it passes the limit). `sandbox_report()` names the mechanism (`none` when nothing can measure), and `doctor` runs a real 900 MB allocation and reports `memory limit verified (mechanism: ...)`. **The watcher has never run on a real Mac**; the libproc structure layout is checked against the documented offsets and the logic against fakes, and the Linux CI runs the watcher path by forcing it. |
| Colab: file-copy settle test (301 / 303 passed) | a **test-timing** failure: the 6.4 test used the real clock, and starting CuPy on the GPU machine took longer than its 1 s settle window, so the file counted as settled when it should not have | the test now runs on a **fake clock** (sleeping advances it instantly), so it is deterministic and takes no real time. The ingestion code was not the cause, but the same area was hardened (the "ingestion timing" row below). |
| Colab (T4): CUDA kernels failed to compile | 6.4 hard-coded `--gpu-architecture=compute_70`; NVRTC from CUDA 13 no longer targets anything below sm_75, so a T4 under CUDA 13 got "invalid value for --gpu-architecture" (reproduced here against the real NVRTC 13.4 and 12.9 libraries) | kernels are now **hardware-targeted** ([KERNELS](docs/KERNELS.md)): the layer asks the device for its compute capability and NVRTC for the architectures it supports, prefers a real-architecture CUBIN (a T4 under CUDA 13 gets `sm_75`), falls back through PTX targets, tries the next target if one compiles but will not load (keeping every failure), and tells a person with a device older than the installed compiler what to install. **Verified against the real NVRTC libraries (compiling the generated source for every architecture they support) and a fake driver; the kernels have never run on a GPU.** |

### The wishlist
| item | what was built | result / limit |
|---|---|---|
| tool forging ([TOOLS](docs/TOOLS.md)) | typed bottom-up synthesis over about 250 primitives, held-out examples, safety gate, sandbox, toolbox, chains | `tools forge / run / chain / request`; organism `--tools`. **Synthesis over a fixed DSL, not a code generator**: it finds what about 10 operations can express. |
| episodic memory ([EPISODIC](docs/EPISODIC.md)) | contrastive task encoder (hand-written gradients, finite-difference checked), recall of earlier champions, program splicing and crossover | on by default in the Generalist; a recalled candidate still has to win the race. Does nothing on a fresh workspace. |
| mixture of agents ([MOA](docs/MOA.md)) | `moa`, `moa_policy`, `moa_blend` families with a load-balanced top-k router; frozen evolved programs as experts | usage and entropy are measured. **Routing helped only where regions of the input want different experts**; in one run the blend measured -0.04 against the winner and was correctly not used. |
| real-time streaming ([STREAM](docs/STREAM.md)) | counted ring buffers, chunk-invariant text / audio / video tokenisers, one token space, a sliding-window language model, frame stacking and partial observability | exact where it says exact (`reprefill` equals a fresh pass to 1e-9). **No trained audio or vision understanding; `rotate` is approximate; no hard real-time deadline.** |
| stateful web actuation ([ACT](docs/ACT.md)) | deny-by-default policy, secrets by name only, redacting ledger, budgets, kill file, dry run; a web-API environment for the learners; a loopback mock API | **verified only against the loopback mock server** and fakes; DNS rebinding can defeat the address check (the host allow-list is the control). CLI only. |
| non-linear observation compression ([ARCHITECTURE](docs/ARCHITECTURE.md)) | residual-VQ autoencoder and a task-aware `choose_encoder` (`encoder = "auto"`) | 0.085 vs 0.58 reconstruction MSE against PCA on a blob manifold. **Evolved programs did not solve the blob task at the small budgets tried**; PCA stays the default. |
| ingestion timing ([DATA](docs/DATA.md)) | a timestamp far in the **future** (a zip extracted with future dates, a share whose clock runs fast) used to mean "still being copied" for ever; such a file now settles once its size and timestamp have looked the same on every visit for `data.settle_s` seconds of this process's own clock. A file that never settles ends in a message that names `data.settle_wait_s`, not in "no data". | tested on the fake clock; a real slow network share was not available. |
| chat model as architect ([ARCHITECT](docs/ARCHITECT.md)) | a four-line grammar, paired-control tests with shared alpha, hit-rate back-off with per-task cooldown, export of validated hypotheses as chat training text | a program edit raised a toy regression from R² 0.31 to 0.88 (p of about 0.001). **The built-in proposer is a bandit, not the chat model, and the chat model is not shown to beat it.** On the small gridworld no hypothesis survived the test at the budgets used. |
| wider patcher reach | the benchmark now runs every `patcher.MUTABLE` module (VAE, character LM, curiosity added), measured inside the sealed `bench.py` | **benchmark scores are not comparable with earlier versions.** |
| topology invention ([ARCHITECT](docs/ARCHITECT.md)) | `topology_net`: DAGs with per-node width and activation, skip connections, exactly function-preserving mutations, greedy rounds on an inner holdout; three structural mutators (`gate`, `neuron`, `mgate`) for programs | beats a too-small MLP on XOR by more than 0.1 validation accuracy; once won a race by 0.002 skill. **Greedy, no pruning.** |

**Constitution:** new sealed files `forage/act.py`, `selfmod/bench.py`, `forge/architect.py` (17 files now); `CONSTITUTION.sha256` re-sealed; `doctor` checks the new modules. `act.py` is imported by the CLI only, never by the organism, the Generalist, the patcher or any hook (a test checks each).

**Test changes (not weakened to pass):**
- the 6.4 Generalist test assumed the champion family was `mlp` or `program_formula`; it now accepts any registered family (the new `topology_net` legitimately won);
- the 6.3 doctor test assumed a working memory limit produced no finding mentioning it; a working limit is now an `ok` finding, so the test rejects only non-ok ones;
- the 6.3 forager-isolation test allows `cli_v65.py`, which hosts the person-run `act` commands.
- `tests/test_v4.py` allows `forage/act.py` and `forage/mockapi.py` to speak HTTP, like `fetch.py`.

**Still not done**
- Verification on a GPU, on a real Mac, on Windows, over a real web API, and with a trained chat model proposing hypotheses (nothing here offered them): please run `zerowhite kernels`, `doctor`, the CI legs, and `zerowhite architect run` on a workspace with a trained chat model, and send me what they print.
- A chat model trained on audio and pixel tokens (the token space is ready; the training is not), hard real-time scheduling, exact (non-approximate) cheap window sliding.
- Tool synthesis from a prose description; primitives beyond the hand-written DSL.
- Node pruning in topology search; a population-based topology search; a learned (not bandit) architect that is shown to beat the heuristic.
- Making the judge editable (`astmut.py`, the safety gate, the watchdog, `bench.py`): declined again, with the reason in docs/SELF_MODIFICATION.md.

## 6.4.0
**Everything 6.3 listed under "NOT done", built, except what cannot honestly be built here (no GPU, no Windows machine, no real speech, no public internet, and the judge itself).** Tests: 303 (212 in 6.2 + 36 in 6.3 + 55 new, `tests/test_v64.py`), all passing in about 5 minutes on one CPU.
The new tests were checked the same way as in 6.3: **26 deliberate bugs were injected into the 6.4 code and every one is now caught by a test**; the first run caught 22, and the four it missed exposed one equivalent "bug" (a window-phase change that is the same function) and three real weak spots in my tests (the table-likeness guard,
the CUDA grid size, quantiser dropout), which I strengthened and re-ran.

**The results, including the ones that did not go my way** (the numbers are measured; the detail is in the linked docs):

| item | what was built | result |
|---|---|---|
| e-graph simplifier ([SELF_MODIFICATION](docs/SELF_MODIFICATION.md)) | equality saturation over evolved straight-line programs | `np.minimum(np.minimum(x, 3.0), 2.0)` -> `np.minimum(x, 2.0)` etc.; 300 random programs x 20 unseen inputs agree; deterministic across hash seeds. **My first version had a precedence bug** (an opaque `if/else` lost its parentheses) that my own narrow probes missed and the random-program test found. |
| bits currency ([ORGANISM](docs/ORGANISM.md)) | `--metric bits`: classification / regression / language-model skill as information | two tasks with equal skill gains are ranked by what is at stake; default unchanged. Control tasks and chat use an *assumed* exchange rate, flagged as such. |
| learned parser ([OMNIVORE](docs/OMNIVORE.md)) | a ~600-parameter delimiter ranker | **a byte-level network scored 100% on its own generator but 12/16 on independent files (rules: 14/16) with probability 1.0 on wrong answers: I deleted it.** The ranker: 1.00 vs rules 0.83 on held-out synthetic, **16/16 vs 14/16** on the independent files (small sample; I wrote both). Default mode `rescue`; end to end 0.81 / 0.83 / 0.95 for off / rescue / arbitrate. Prose can never be made a table (tested on the repo's own example essay). |
| kernels ([KERNELS](docs/KERNELS.md)) | lazy graph, fusion, C JIT via ctypes, CUDA via NVRTC + driver API via ctypes | numbers identical to numpy on 60 random graphs, including the generated CUDA source executed on the CPU. **Generated C is not faster than numpy here (1.31x on cheap chains at 2M elements, 0.55x with transcendentals, slower at 1M). CUDA never ran on a GPU.** |
| audio codec ([VOICE](docs/VOICE.md)) | MLP autoencoder + residual VQ, scalable bitrate, tokens as text | held-out synthetic speech: 1.6 / 3.3 / 3.8 dB at 1 / 3 / 6 stages vs PCA 1.1 / 2.0 / 2.5 at the same bitrate. **Low absolute quality; synthetic speech only; the small test-sized codec does not beat PCA clearly.** |
| attention encoder (docs/ARCHITECTURE) | Perceiver-style cross-attention bottleneck, hand-written gradients (finite-difference checked) | reconstruction error 0.067 vs PCA 0.70 on a blob manifold (k=2). **Did not help evolved programs** (ring task: PCA agents 0.81/0.63/0.69, encoder agents 0.58/0.55/0.84, 10x slower), so PCA stays the default. |
| mixture of experts ([CHAT](docs/CHAT.md)) | `chat grow --experts`, router, balancing loss, inference path | dense -> MoE exact to 1e-15; gradients match finite differences; KV-cache path matches training. **My first balancing loss did not balance** (measured, replaced): with the Switch-style loss the busiest of 4 experts takes 28% of tokens, without 57%. |
| sleep ([ORGANISM](docs/ORGANISM.md)) | `--sleep`: long, gentle, replay-heavy pass gated on the WORST age band of the corpus | the test shows a case a mean-based gate would accept and this one refuses; a destructive pass is refused and the live model untouched. |
| curiosity foraging ([FORAGE](docs/FORAGE.md)) | the organism reads from YOUR wishlist and allow-list when everything has plateaued, keeps pages that are lexically new | tested against a real local server end to end. Novelty is lexical, not a gap detector. |
| shared memory / system 2 ([CHAT](docs/CHAT.md)) | table / audio / RL as text, zoo + own source into the chat corpus, a safety-gated successive-halving race for proposed programs | the real ingestion takes the 40+ self-source documents; the race rejects unsafe / broken proposals and ranks real ones. **No evidence that a small chat model proposes useful programs or gains from reading its source.** |
| engine tier ([SELF_MODIFICATION](docs/SELF_MODIFICATION.md)) | opt-in single-edit mutations (and arbitrary replacement text) for `nn.py` / `engine.py` under the full judge | every stage verified with stubbed runners; **expect "no gain" almost always.** `astmut.py` and the judge stay sealed. |

**Fixed (bug in 6.2's table parser):** an all-text table with one short row raised `AttributeError` in the header check (padded cells are `None`). Regression test added.

**Policy change you should know about:** `forage/` could previously be imported only by the CLI (a test enforced it). The organism may now import it, lazily inside `_act_forage`, behind `--forage-host` AND `--forage-wishlist`; the guard test now checks exactly that instead of "nobody".

**Constitution:** new sealed file `selfmod/engine_tier.py`; `doctor.py` and `patcher.py` edited only in their sealed-file lists; re-sealed. Everything else in the sealed set is byte-identical to 6.3. `patcher.MUTABLE` is unchanged.

**Still not done**
- Verification on a GPU, on Windows, against real speech, and over the public internet (nothing in this environment offered them): please run `zerowhite kernels`, the `windows-latest` CI leg, and `doctor`, and send me what they print.
- Making the judge editable ("unsealing" `astmut.py`, the safety gate, the watchdog): declined again, with the reason in docs/SELF_MODIFICATION.md.
- A proposer that writes correct numerical code (the engine tier and System 2 are ready for one; nothing here is); chat-driven refactoring of evolved programs; sharing common sub-expressions back into temporaries after e-graph extraction.
- A byte-level neural parser (tried, failed out of distribution, removed); learned header / decimal / encoding decisions (still rules); repair of corrupted files beyond the existing rules.
- MoE on a real GPU, `chat export` of a MoE model, sparse expert dispatch, a MoE-aware memory estimator; kernels: autograd through the graph, float32, device-side reductions, evolving kernels.
- Feedback in chat updating the forge's priors; a shared learned embedding for audio / tables; any evidence that unified-stream training helps.

## 6.3.0
**Seven improvements taken from an outside architecture review, the ones that could be built and checked here; the rest are listed under "Not done" with the reason.** Tests: 248 (212 before + 36 new, `tests/test_v63.py`), all passing in 211 s on one CPU.

**Verification status, stated plainly.** Nothing here ran on Windows, on a GPU, or against the public internet (the build environment had none of them). What was established, and how:
the cost forecaster against simulated cost shapes **and** a real two-task organism run; model growth to 1e-10 in float64, then through a real `prepare` + `Trainer` run; the transaction and the significance gate against stubbed benchmarks on a real copy of the repository; the Windows Job Object
only as far as struct layouts and a fake `kernel32` allow; the forager against fakes and a real local HTTP server. **24 deliberate bugs were injected into the new code and every one is caught by a test** (the first harness gave a false alarm: a same-size edit restored within one second left a stale `.pyc`; the harness now purges bytecode).

**Added**
- **Cost forecasting** (`forecast.py`, [docs/ORGANISM.md](docs/ORGANISM.md#cost-forecasting-63)). The organism's estimate of an attempt's duration was linear in the budget with a fixed x1.5 pad; the docs recorded 2 s predicted / 8 s taken. Now a robust power-law fit (`seconds ~ a * units^gamma`) replaces the linear probe once a task has 4 attempts,
  and the pad is the split-conformal 90% overrun measured from that task's own fitted predictions (chat: its overhead). Simulated, 150 lives per regime: coverage where the old rule was optimistic (super-linear cost) 0.88 -> 0.92; where it was needlessly wide (linear) the pad shrinks 1.64 -> 1.57 at 0.97; under drift 0.90 -> 0.88 with a narrower pad.
  `--fixed-margin` restores the old behaviour exactly; with fewer than 4 attempts it is identical anyway. *A real run caught a flaw in my first version*: the early linear predictions overran x3.4 and, kept in the window, would have padded every later attempt by x3.4; only fitted-curve overruns now calibrate the pad.
- **Windows parity** (`selfmod/winjob.py`, `fileio.replace`). A Windows Job Object (ctypes) gives sandboxed children a whole-tree memory limit, a CPU-time limit and kill-on-close, where Windows previously had only the wall-clock timeout. **`save_npz` - the checkpoint writer - used a bare `os.replace`
  and could still raise `PermissionError` on Windows; so did six other places.** All now go through the retrying `fileio.replace`, and a test fails if any module calls `os.replace` directly. `doctor` now verifies a memory limit really bites (a 900 MB allocation under a 256 MB limit must die) on every OS that reports limits.
- **Statistical gate** (`selfmod/stats.py`). Source patches and hooks were accepted when a *mean* over a few seeds cleared a margin. They must now also pass a paired sign-flip test on the per-seed scores (exact for six seeds, p < 0.05). A test shows a patch whose mean gain clears the old rule on one lucky seed (+0.0275 mean, 1 of 6 seeds improved) being refused. `alpha=None` restores the old rule.
- **Transactional promotion** (`selfmod/watchdog.py`). Applying a patch wrote files one by one with nothing to repair a crash in between. It is now a journaled transaction: originals and new files copied with hashes, atomic per-file swaps, the test suite and a byte-for-byte check of the sealed constitution on the real tree, then commit; any failure,
  exception or interrupt restores every original, and an interrupted journal is rolled back at the next start. Standard-library only; also runnable as its own process.
- **Model growth** (`chat/grow.py`, `zerowhite chat grow`). Add layers, FFN units and attention heads to the live chat model with **exactly** the same outputs (zero-initialised output projections; for heads, zero channels with RMSNorm gains *and* epsilon rescaled), then keep training. Previously "an existing model keeps its size".
- **Program agents see wide observations** (`forge/reduce.py`). They read only the first 24 numbers; wider observations are now compressed by a whitened PCA fitted on random-policy probes (only when `obs_dim > 24`; every existing task is untouched). On a 40-number task whose signal sat in the last 16 numbers: over 4 seeds truncated agents stayed at chance (skill -0.12 to 0.01) while projected agents reached 0.88, 0.79, 0.34 and 0.00. It helps; it is not reliable, because the program search itself is stochastic.
- **Opt-in forager** (`forage/`, `zerowhite forage`, [docs/FORAGE.md](docs/FORAGE.md)). Reads URLs *you* give from hosts *you* allow: https, allow-list re-checked at every redirect, SSRF guard, robots.txt, size/count/rate limits, text only, a ledger line per attempt including refusals. Nothing calls it autonomously (a test enforces that) and the file is sealed.

**Fixed (a bug in 6.2, found by a test for the above)**: `HistoryStore.lineage` looped forever, growing until the process was killed, once the same text had been recorded twice. The patcher's own auto-rollback does exactly that (original -> patch -> original), so `selfmod history` / `rollback` after an auto-rollback would hang. Now keyed by version and content, with a cycle guard; the regression test runs in a subprocess so a recurrence cannot hang the suite.

**Constitution files I edited, and re-sealed (please review the diff)**: `fileio.py` (`replace`, `save_npz`), `selfmod/sandbox.py` (Job Object, ignore `.zw_promotion`), `selfmod/patcher.py` (significance gate, watchdog apply, recover), `selfmod/history.py` (the lineage fix), `doctor.py` (seal list, live memory check, module list, `fileio.replace`). New sealed files:
`selfmod/winjob.py`, `selfmod/stats.py`, `selfmod/watchdog.py`, `forage/fetch.py`. `patcher.MUTABLE` is unchanged: the self-modifier cannot touch any of them. Run `python -m zerowhite doctor` first.

**Changed**: hooks / patches need significance as well as margin; `organism` gains `--fixed-margin` and `--coverage`; `chat` gains `grow`; a new top-level `forage` command; the `.zw_promotion/` journal folder (git-ignored, excluded from sandbox copies).

**From the review, deliberately NOT done**
- *Unsealing the core* (letting the self-modifier rewrite `astmut.py`, the safety checks and the sandbox). The shadow-fork *transaction* is built; the unsealing is not, because the seal exists so the thing being judged cannot edit the judge. Widening what may change stays a human decision ([docs/SELF_MODIFICATION.md](docs/SELF_MODIFICATION.md)).
- *Autonomous foraging driven by curiosity*: the forager exists, but a person chooses the URLs and the allow-list, and the organism never reaches the network.
- *Lazy tensor graph, ctypes CUDA driver bridge, NVRTC kernels*: unverifiable without a GPU and driver, and a 600-line unverified rewrite of the engine would be a regression risk for the one place the project has measured parity.
- *Neural audio codec / unified audio tokens, multimodal byte backbone with System 1 / System 2, learned neural parser for `devour`, Perceiver encoders, e-graph simplification, MoE growth*: research-scale designs that need real data or training runs to evaluate, not a code change. The speech recogniser is still only measured on synthetic speech.
- *"Sleep / consolidation" cycles*: the online learner already replays corpus examples, gates every update on a regression check and persists accepted corrections into `data/`, which the scheduled training then absorbs; a second mechanism would duplicate it.
- *Information-gain as a universal metric*: per-task skill stays; converting every task to bits needs held-out losses the tasks do not all report.
- Windows enforcement, GPU, public-internet fetching: see Verification status.

## 6.2.0
**The chat model can train on a GPU** (CuPy: NVIDIA/CUDA; AMD ROCm builds use the same code, untested). See [docs/CHAT.md](docs/CHAT.md#training-on-a-gpu).

**Verification status, stated plainly:** there was no GPU, CuPy or network in the environment this was built in, so **no line of it has run on a real card**. What *is* established: the CPU path is bit-identical to 6.1 (old and new model compared
exactly: loss, every gradient, evaluation and log-probs, tied and untied heads, dropout, packed documents, negative preference weights); the device path is the same maths, exercised against a strict fake device; and `zerowhite chat gpu` exists so
the first thing you do on a real machine is check it there.

**Added**
- `chat/backend.py`: one array backend, NumPy (default) or CuPy. Transfers are explicit both ways; the one operation CuPy lacks (`np.add.reduceat`, the embedding-gradient scatter) is a backend method (`cupyx.scatter_add` on a GPU, the exact old code on the CPU);
  `compute.backend` = `auto` (a GPU when CuPy and a working device exist, else the CPU with the reason kept), `gpu` (a demand: fails with the fix) or `cpu`.
- `chat/gpucheck.py` and `zerowhite chat gpu`: on your machine it checks kernels, model parity (loss and gradients against the CPU, with repeated token ids, dropout, packed documents, negative weights), optimiser parity and the real CPU-vs-GPU
  tokens/s; exits non-zero with the numbers if anything disagrees.
- `compute.backend`, `compute.device`, `compute.vram_fraction`, `compute.max_params`, `compute.max_micro_tokens`; `--gpu` / `--cpu` on `chat train` and `chat prepare`; a `compute` line in the plan; the device in `organism status`.
- A GPU out-of-memory becomes a message naming the knobs, with the checkpoint kept (`GpuOutOfMemory`, exit code 6); an unusable demanded GPU exits 5 with the install hint.
- `tests/fakedevice.py` and `tests/test_gpu.py` (11 tests): the fake refuses host arrays mixed into device math, offers only the CuPy functions the project relies on (it rejects `ufunc.reduceat`), and counts every transfer and scalar synchronisation.
  Checked by mutation: 15 deliberate bugs, all caught.

**Changed**
- `GPT`, `AdamW`, `Trainer`, `autoscale` and `prepare` take a backend. Weights are initialised on the host (the same weights on every backend) and checkpoints, exports and the served model are plain host float32 files.
- On a GPU the benchmark is bracketed by device synchronisation (kernels are asynchronous: an unsynchronised timer measures how fast Python enqueues work), memory is planned from free VRAM, and the CPU-sized caps give way to `compute.*` unless set explicitly.
- The optimiser's gradient norm costs one device-to-host transfer per step (stacked on the device) instead of one synchronisation per tensor; pinned by a test.
- `ChatTask` accepts a backend and reports the device; the organism picks the GPU up through `compute.backend=auto` with nothing to configure.

**Caveats (also in the docs)**: float32 only (TF32 via `CUPY_TF32=1` is reported by `chat gpu`, not enabled); GPU training is not bit-reproducible (atomic adds); first steps are slow while CuPy compiles kernels; an existing model keeps its size, so use
`chat train --fresh` for a bigger one; not fused, so a modest share of the card's peak; one GPU; Apple Silicon unsupported.

**Not done**: mixed precision, multi-GPU, fused attention, GPU serving / online learning, a GPU path for the general model, widening the self-rewriting beyond two hooks.

**While testing**: two of my first assertions could not fail (a check on arrays loaded back from disk, and a comparison multiplied by zero) and a CPU model would have passed the end-to-end test unnoticed; all three were replaced by checks that fail when the property breaks.

## 6.1.0
**One scheduler for both models**: the chat model (v5) is now a task of the organism (v6), by the same measure, learning progress per second. See [docs/ORGANISM.md](docs/ORGANISM.md#the-chat-model-as-a-task).

**Added**
- `chat_task.py`: the bridge. Skill = `1 - validation_nll / ln(vocab)`; an attempt is the existing `prepare()` + `Trainer` with a minutes budget; gain is measured with the live model scored before and after on the same
  validation set (new data changes the split, so a number recorded earlier is not comparable); a run the trainer does not promote counts as no gain.
- `organism --chat off|auto|on|PATH`, `--chat-minutes`, `--chat-first-minutes`, `--chat-probe-minutes`, `--chat-max-minutes`, `--chat-set KEY=VALUE`, `--no-chat-feed`.
- A new chat model's first attempt is a real-sized run, not a cheap probe: the first run's time budget fixes the model's size for good, so a one-minute probe would have locked in a tiny model.
- Prose the inbox digests is mirrored (atomically, only when changed) into `<chat>/data/_omnivore/`: one dropped file teaches both models.
- New or changed chat data re-opens a plateaued chat task. A hook win does not (hooks steer the general model's search only).
- Coexistence with the chat service: while another process holds the training lock the organism waits instead of failing or touching shards; a STOP file or Ctrl-C ends a chat attempt at a checkpoint, ends the run,
  and the next attempt resumes it. Neither a deferred nor an interrupted attempt counts as a result.
- `tests/test_v61.py` (17 tests: 14 scripted scheduling / mapping tests and 3 that train a real tiny chat model through the organism). Checked by mutation: 12 deliberate bugs, all caught. (The first pass caught 11: the survivor was a chat cost formula written twice, once in `cost()` and once inline in the decision, so the tested copy was dead code. There is now one formula, and a test pins it.)

**Changed**
- Decisions now try candidates in priority order and take the first that FITS the time left; before, the best candidate not fitting meant idling even when a cheap task would have fitted.
- The first attempt is a baseline by `attempts == 0` rather than by `best_skill is None` (a task whose standing is known up front, such as an existing chat model, still gets a baseline attempt).
- `organism status` shows the chat task's budget in minutes and the model's perplexity and parameter count.
- Version 6.1.0. No sealed (constitution) file changed.

**Fixed**
- **A lone failing task was never retried.** Back-off counted decisions, but idle rounds do not advance the decision counter, so a single failing task (and anything that backs off while nothing else runs, which a busy chat model
  makes likely) idled forever in `--forever` mode. Waiting out a back-off now advances it; the idle message says "backing off" instead of the wrong "no tasks yet".

**Found while testing (and fixed before release)**
- With the chat model being trained by another process at start-up, the organism could have started rewriting its own search rules because "every task had plateaued"; a chat model that merely cannot be touched now is
  waited for, whether or not it has been adopted yet.

**Not done (still true)**
- No GPU backend (all compute is NumPy; the probe still only reports an accelerator stack) and the self-rewriting is still limited to two hooks. Neither can be verified in a CPU-only sandbox, and the chat trainer
  has no evolutionary search for hooks to steer.

## 6.0.0
**An autonomy layer**: `devour` (ingest anything without breaking), `organism` (a loop that decides what to improve next, and changes how it searches when stuck) and `substrate` (what the machine is, and what follows).

**Added**
- `omnivore/` and `zerowhite devour`: content-based file typing; encodings (BOMs, UTF-16/32, Windows-1252, bad bytes, NUL padding); tables (found delimiter, preamble, ragged rows, unbalanced quotes,
  half-written last row, decimal commas, thousands separators, currency, dates, booleans, categories, missing values, identifier / constant / index columns, rare classes); JSON, JSON Lines, truncated JSON;
  SQLite (read-only), `.npy`; archives (gzip / bz2 / xz / zip / tar, nested) with zip-bomb limits, path-traversal safety, and salvage of truncated streams and zips whose directory is missing;
  prose into a shared corpus; environment files only with `--allow-code` after a sandboxed smoke test. A content-hash ledger makes re-runs free, a quarantine record explains every file set aside
  and retries it when its content changes, a corrupt ledger is rebuilt. Source files are never touched.
- `organism.py` and `zerowhite organism [run|status|stop]`: observe / decide / act / learn. New tasks get a cheap probe; the next attempt goes to the highest expected learning progress per second
  (measured gain capped by headroom); an attempt that gains nothing doubles that task's budget; ceiling and patience define a plateau; attempts are fitted to the time left (with a safety margin);
  when everything has plateaued it changes how it searches (`hooks:*` under the existing test gate, `poet` when an environment task exists) chosen by Thompson sampling, and re-opens tasks after a win.
  Failures back off, state is atomic, a lock prevents two organisms on one workspace, every decision is logged with its reason.
- `substrate.py` and `zerowhite substrate`: matmul / disk / memory probe cached per machine (a workspace copied elsewhere is re-measured); size limits derived from it with a stated reason for each.
- `examples/messy_inbox.py` (12 deliberately broken files to try it on); `docs/OMNIVORE.md`, `docs/ORGANISM.md`; `tests/test_v6.py` (36 tests: component tests, a fuzz property test over random and truncated / bit-flipped files of every supported format, scripted
  scheduling tests, a CLI end-to-end test). The new tests were checked by mutation: 18 deliberate bugs, all caught.

**Changed**
- README: the tagline says "no pretrained weights and no external model or API" instead of "no-LLM" (the chat part trains its own transformer); the Windows note no longer reads like a chat reply.
- Version 6.0.0. No sealed (constitution) file changed.

**Found while testing (and fixed before release)**
- A semicolon file with decimal commas was read as a comma file (the wrong delimiter was perfectly "consistent"); the scorer now also rewards the number of columns.
- A column named `y` beside a column named `label` hijacked target selection; ambiguous name hints now fall back to the last column.
- A first attempt counted as "progress from zero", so a task that was already perfect kept winning the scheduler; it is now a baseline, expected gain is capped by headroom, and a task at its ceiling is left alone.
- The self-improvement cool-down was computed before the cycle counter advanced, so the same arm could repeat back to back.
- The cost model under-predicted by up to 4x; the fit-to-time decision now applies a 1.5x margin (and the docs say the estimate is still optimistic).
- `organism stop` created its file without an explicit encoding, which the repository's own lint test rejects (the cause of an earlier Windows crash).

## 5.0.0
**A real, trainable chat model** (`zerowhite chat ...`) and speech (`zerowhite chat voice ...`), on the same numpy-only rules.

**Added**
- `chat/`: byte-level BPE tokenizer, chat template with assistant-only loss, decoder-only transformer with a hand-written backward pass, AdamW, packed-document
  batches, a resumable continual trainer that promotes a model only if its validation loss is no worse than the live one, a DPO preference phase, KV-cache generation, model export (npz / int8 / safetensors in Llama layout).
- A **data folder**: drop json / jsonl / csv / tsv / parquet / txt / md / source files (optionally `.gz`, `.zip`, `.tar`) anywhere under `data/`; they are ingested incrementally and de-duplicated. Adapters for Alpaca,
  ShareGPT, OpenAI/ChatML, preference pairs, PIPPA, MBPP and critique data; `dataset.json` sidecars for anything else.
- Filters: quality, persona keywords (whole-word, so `'sis '` no longer matches "analysis"), safety (sexual content with a minor or family member is always dropped; general sexual-content filter on by default), PII redaction.
- **Nothing hard-coded**: one schema of ~230 options (docs/CONFIG.md, generated); `auto` values are derived from your hardware and data and recorded with reasons.
- **Online learning** from thumbs/corrections with replay, a regression gate, versioning and rollback; accepted feedback is kept in `data/_online/`.
- **Background process**: supervised daemon with HTTP API + web UI, folder watching, automatic retraining and hot swap, autostart installers (Windows / systemd / launchd).
- `chat fetch`: resumable downloads of public datasets (catalogue entries are unverified; see docs/DATA.md). `chat convert`: dataset format conversion.
- **Voice**: WAV/resampling/VAD in NumPy, built-in formant synthesiser, TTS/STT plug-ins (espeak, piper, SAPI, say, pyttsx3, Whisper, Vosk, your own command), a CTC speech recogniser you train on `data/audio`, `chat talk --voice`, web UI microphone, audio conversion.
- Docs: CHAT.md, DATA.md, VOICE.md, CONFIG.md; `examples/persona_sister/`; ~50 new tests (gradient checks, KV-cache and Llama-export references, trainer, online gate, daemon, server security, voice).

**Changed / removed**
- `zerowhite train` no longer defaults to the synthetic shape curriculum: pass `--dataset NAME` or `--task curriculum|shapes` explicitly. The demo scripts and demo tables were removed; tests write their own data.
- The "no network code outside the dataset downloader" test now names its exceptions (chat downloader, local service).

**Fixed (found while testing)**
- A config file with a `_comment` key (the one `chat init` writes) was rejected.
- The voice-activity detector took digital silence inside speech as the noise floor.
- Phone-number redaction swallowed the first digit groups of longer numbers.

## 4.0.0
**Structure**: flat scripts became the `zerowhite` package (`python -m zerowhite ...`, or the `zerowhite` script after `pip install -e .`).

**Added**
- `forge/` the general model: `solve` profiles any task (table, environment, text, image generation), proposes agents from a
  learned prior, races them, keeps the winner in a persistent zoo, and transfers champions between similar tasks.
- Program families: agents whose weights are *Python source* (evolved formulas and controllers), searched with an island model,
  regularized evolution, a functional-equivalence cache and constant tuning (no language model; AST-level mutation).
- `selfmod/`: static safety gate, content-addressed history with rollback, subprocess sandbox, evolvable hook files
  (`evolved/`), and a test-gated source patcher for hyperparameter sites. A sealed "constitution" the machinery cannot edit.
- `poet`: open-ended co-evolution of environments and agents. `evolve`: unattended solve + self-modify loop.
- `doctor`: diagnose and repair (stale temp files, corrupt checkpoints/zoo pointers, faulty self-written hooks, seal check).
- Bring-your-own tasks: `--csv`, `--env-file`.

**Removed**
- Every default model: `AGENT_DEFAULT`, `CVAE_DEFAULT`, `LM_DEFAULT`, the fixed classifier width, the hard-coded 1/5-success rule
  (it is now an evolvable file whose shipped version is a no-op). A test asserts none come back.

**Fixed**
- Windows: 5 test failures in the dataset loaders. The test helper built `file://` + `C:/...`, which makes `C:` the URL's host;
  it now uses `Path.as_uri()` (`file:///C:/...`). `doctor` runs the same round trip as a self-test.
- Windows: atomic file replace is retried briefly when an antivirus scanner or indexer briefly holds the destination.
- Source-patcher preserves CRLF line endings; constitution seal hashes are CRLF-insensitive.
