# Self-modification: what it is allowed to do, and how that is enforced

## Threat model, stated plainly

The risk being managed is **an optimiser that edits itself and optimises the wrong thing**, plus ordinary accidents (runaway loops, bad numerics,
half-written files). It is *not* a defence against a malicious human or a hostile code author. `selfmod/safety.py` is a static allow-list and
`selfmod/sandbox.py` is process-level containment; neither is a security boundary. If you ever feed Zero White code you do not trust, run it in a container or VM.

## Level 1: evolved hooks

* **What**: `evolved/sigma_update.py` and `evolved/select_parent.py` are real Python source that the Evolver calls. Shipped versions: "return sigma unchanged" and
  "pick uniformly at random". `selfmod/hooks.py` evolves replacements with `islands.ProgramEvolver`.
* **Proposer**: grammar-guided AST mutation (`astmut.py`): replace/wrap/hoist subtrees, change an operator, re-index an input, insert/delete a temporary, tune
  constants, subtree crossover. Every result is repaired (no undefined names), safety-checked and size-capped.
* **Fitness (search)**: how well the Evolver optimises 4 synthetic landscapes (sphere, ellipsoid, rastrigin, noisy sphere) with the candidate hook, 6 generations,
  2 population settings, several seeds, minus a parsimony penalty.
* **Acceptance**: the best candidates are re-scored on **4 landscapes the search never saw** (rosenbrock, ackley, plateau, deceptive) with 10 fresh seeds. A hook is installed
  only if it beats the live hook there by `margin` (0.015) **and** is not worse on the training landscapes. The textbook 1/5 rule and genesis are measured alongside and
  reported, never used as a default or a seed.
* **Observed behaviour**: `select_parent` found "always mutate the current best" (+0.051 held-out, installed). Its source is `return <negative number>`, which the
  caller clamps to index 0: a valid but blunt rule, and a reminder that held-out *proxy* gains need checking on the real workload. `sigma_update` found only a different
  constant (0.77 vs the default 0.5), +0.013, below the margin, rejected.
* **Runtime protection**: `evolved.Policy` wraps every call; an exception, NaN or wrong type falls back to genesis behaviour for that call and increments a fault counter;
  the caller clips results (σ to [0.1, 1.0], parent index into range). `doctor` rolls back a hook with faults or one that is worse than genesis.

## Level 2: source patcher

* **Sites**: numeric literals that are named hyper-parameters (signature defaults, keyword arguments, class/module constants, `self.x = const` in `__init__`, and the
  weights in the candidate-scoring formula) in `zero_white.py, self_model.py, meta_optimizer.py, evolve.py, agents.py, generative.py, curiosity.py, architecture.py`.
  58 sites at the time of writing.
* **Excluded on purpose** (`patcher.SKIP_NAMES`): anything that defines *how a model is scored* (episode counts, validation caps, per-class sample sizes, seeds), reward
  definitions (`step_cost`), structural dimensions, and plain compute budgets. A patch that raises the score by changing the ruler, or by spending more compute, is not an improvement.
* **Patch = exactly one literal.** `single_literal_change` re-parses both versions and refuses anything else.
* **Five gates**: structure → SELECT seeds improve → CONFIRM seeds not worse → FINAL seeds (used once, on the accumulated patch set, vs the *original* tree, margin 0.004 **and, since 6.3, a paired
  sign-flip significance test on the six per-seed scores, p < 0.05**: see below) → full test suite passes in the sandboxed copy. Only then, and only with `--apply`, is the real tree changed, through a journaled
  transaction (below); the tests run again there and the originals are restored automatically on failure.
* **The significance gate (6.3, `selfmod/stats.py`).** A mean over six seeds can clear a fixed margin because one seed went well. The final decision now also requires the per-seed gains to be consistently positive:
  each paired difference is as likely to be +d as -d if the patch did nothing, so the observed mean is compared with the means of all 2^6 re-signings (exact, no distributional assumption). Six seeds can reach p = 1/64;
  fewer than five could never reach 0.05 and the gate says so. `patcher.search(..., alpha=None)` restores the margin-only rule. Hook acceptance uses the same test over the (function, seed) pairs of the held-out suite.
  What it does not do: it cannot make a proxy benchmark a real one, and with six seeds it is a filter against luck, not a proof of improvement.
* **Transactional promotion (6.3, `selfmod/watchdog.py`).** Before 6.3 the changed files were written one after another and a crash between two writes left the repository half patched. Now: every original and every new file
  is copied into `.zw_promotion/<txid>/` with its SHA-256 and a journal is written; files are swapped in atomically one by one; the test suite runs on the real tree; the sealed constitution is checked byte for byte; only then
  does the journal say `committed`. Any failure, exception or `KeyboardInterrupt` restores every original, and `watchdog.recover()` (run at the start of every `patcher.search`) rolls back a journal that a crash left open. The watchdog
  refuses paths outside `patcher.MUTABLE`, absolute / `..` paths and files that do not exist, and refuses to start when the constitution already differs from its seal. It is standard-library only, sealed, and can run as its own
  process: `python -m zerowhite.selfmod.watchdog promote --repo . --changeset changes.json --verify-cmd '["python","tests/run_all.py"]'`. Limits: the swap is atomic per file, not across files (recovery is rollback, not roll-forward);
  a disk that lies about `fsync` can defeat any journal; the in-process path (what `patcher.search` uses) shares a process with its caller, the separate-process path does not.
* **Observed behaviour**: 52 real candidates over two runs, none applied. A patch that won on SELECT (+0.018) lost on FINAL (−0.004); another won on FINAL by +0.002, under the margin.
  That is the gate working. Treat level 2 as a capability with a high bar, not a source of free gains.

## The constitution

`selfmod/safety.py, history.py, sandbox.py, patcher.py, winjob.py, stats.py, watchdog.py, astmut.py, fileio.py, doctor.py, evolved/__init__.py, forage/fetch.py` and, since 6.5, `forage/act.py` (the actuator's policy), `selfmod/bench.py` (the judge of patches: it is the number a patch has to move) and `forge/architect.py` (the judge of architect proposals): none is in `patcher.MUTABLE`, `rollback_file` refuses them, and
`CONSTITUTION.sha256` (CRLF-insensitive hashes) lets `doctor` detect any change. If *you* edit one on purpose, run `python -m zerowhite doctor --seal`.

## Sandbox guarantees by OS (what `doctor` reports)

| | Linux / macOS | Windows |
|---|---|---|
| separate process, temp dir, scrubbed env, wall-clock timeout, whole process tree killed | yes | yes (`taskkill /T`) |
| CPU-time / memory / file-size rlimits | yes | memory (whole process tree) and CPU time through a **Job Object** (6.3, `selfmod/winjob.py`, ctypes); file size is not limited |
| network namespace (`unshare -rn`) | Linux with unprivileged user namespaces | no |

The Windows Job Object is **verified here only as far as a Linux machine can verify it**: the structure layouts against the documented sizes and offsets, and the call sequence (create, set limits, assign, terminate, close, every failure path)
against a fake `kernel32`. The enforcement itself has not been run on Windows by the author; the `windows-latest` CI leg runs a real memory-hog test, and `doctor` runs the same check live on any OS where it reports limits (a child that
allocates 900 MB under a 256 MB limit must die). A child that spawns a process in the few milliseconds before it is assigned to the job escapes it.

**Why the constitution stays sealed (a design decision, not an omission).** A review of this project suggested "unsealing the core" so the self-modifier could rewrite everything, `astmut.py` and the safety checks included, behind a
shadow-fork watchdog. 6.3 builds the shadow-fork *transaction* but not the unsealing: the point of the seal is that the thing being judged cannot edit the judge, and a watchdog that lives inside a tree the proposer may rewrite
protects nothing. Widening what may change is a human decision (edit `patcher.MUTABLE`, run `doctor --seal`), made with the diff in front of you.

## The engine tier (6.4, opt-in): code edits in `nn.py` / `engine.py`, same judge

`python -m zerowhite engine-tier` (dry run; `--apply` to promote) lets the machinery try changes to the numerical core, `selfmod/engine_tier.py`. Level 2 only changed numeric literals in eight tuning modules; this tier changes **code**:
single-edit mutations (an operator, a comparison, a constant) proposed by the built-in search, and, through `evaluate_candidate`, any valid replacement text for those files, including new loops and tensor expressions, from a person or from the
System-2 proposers. The judge is the same and none of it is loosened:

* **structure** (`engine_safe`): the file parses; no new import; no new call of `exec` / `eval` / `compile` / `open` / `__import__` and no new use of `os`, `sys`, `subprocess`, `socket`, `ctypes`, `shutil`, `importlib`, `pickle`, ...; no function or
  class removed or renamed;
* the **full test suite** in a sandboxed copy; the **final-seed gain** over the ORIGINAL tree must clear the margin and the paired significance test; promotion through the journaled **watchdog** transaction (tests again on the real tree, constitution
  byte-for-byte);
* what stays sealed: the judge itself and `engine_tier.py`. **`astmut.py` stays sealed too**, although the review asked to free it: it contains `compile_program`, the boundary between evolved text and executed code, so it belongs with the judge. `ENGINE`
  can only be widened by editing the sealed file.

Verified: the structure gate on accepted and refused shapes, mutations of the real `nn.py` / `engine.py` (each is exactly one edit, passes the gate, changes exactly one line), and every stage of the judge with stubbed benchmark / test runners on a copy of
the repository (accepted, applied through the journal, refused as unsafe before any test runs, tests failing in the sandbox, the real tree disagreeing and rolling back, a one-lucky-seed gain refused by the significance gate).
**Not verified, and probably unrewarding:** whether the benchmark exercises `nn.py` / `engine.py` enough for any edit to show a gain. Expect almost every candidate to end as "no gain"; that is the judge working. Unrestricted synthesis of new classes
or autograd rules requires a proposer that can write correct numerical code; none of this project's models has been shown to.

## Equality saturation for evolved programs (6.4)

Programs the forge evolves are now cleaned with an e-graph (`selfmod/egraph.py`) after `astmut.simplify`: sub-expressions are stored once per equivalence class, rewrite rules (identities, constant folding, min / max absorption and associativity,
distribution, double negation, `max(x, 0)` = `relu(x)`, ...) ADD equal forms instead of replacing, and the cheapest form is extracted, e.g. `np.minimum(np.minimum(x, 3.0), 2.0)` becomes `np.minimum(x, 2.0)`. Straight-line functions only; anything else is
returned untouched; a result is used only if it is not larger and gives the same numbers on probes of several scales and shapes (a deliberately unsound rule is caught by that check in the tests). The shown / saved program is the cleaned one; **what runs is
never replaced**. A first version had a real bug (an opaque `if / else` leaf lost its parentheses and changed precedence; the random-program test found it), fixed and kept as a regression. Not done: sharing common sub-expressions back into temporaries, `if` / loops,
and "chat-driven refactoring into modular functions" (that needs a model that writes correct code).

## Auditing

`python -m zerowhite selfmod history --workspace zw` lists every install, rollback and patch with timestamps and hashes. `selfmod status` prints the live source of every
installed hook. `selfmod/patches/*.diff` holds the unified diff of every accepted patch set.

## What the source patcher can see (6.5)

The patcher keeps an edit only if the benchmark improves under the paired test, so an edit in a module the benchmark never runs could neither be rewarded nor caught. Until 6.4 that was true of several `patcher.MUTABLE` modules. 6.5 widens the benchmark to six parts (digits, CartPole, a landscape search, a VAE, a character LM, curiosity), a test profiles a run and fails if any `MUTABLE` module is not executed, and the VAE / character-LM / curiosity measures are computed inside the sealed `bench.py`, not by the module under patch. Scores are **not comparable with earlier versions**. Details in [ARCHITECT.md](ARCHITECT.md#4-the-patchers-reach-selfmodpatcherpy-selfmodbenchpy).
The architect may ask for a source-patch **dry run** (never an application) and for a hook search; both go through their existing gates unchanged.
