# Architecture

```
                         ┌───────────────────────── forge/generalist.py ──────────────────────────┐
 task.py ── Task ──────► │ profile → incumbent → transfer → propose(prior) → race → keep(zoo,prior) │
 (csv/env/text/image)    └──────────┬──────────────────────────┬────────────────────────────────────┘
                                    │ families.py              │ prior.py (learned) / zoo.py (disk)
                       ┌────────────▼───────────┐     ┌────────▼─────────────────────┐
                       │ mlp, selfmod_net,      │     │ selfmod/islands.py            │
                       │ actor_critic, cvae,    │     │ astmut.py  safety.py          │
                       │ char_lm  (v3 models)   │     │ (program_policy, program_formula)
                       └────────────────────────┘     └───────────────────────────────┘
 evolve.py ◄── Policy ◄── evolved/*.py (live overlay) ◄── selfmod/hooks.py ◄── history.py (versions, rollback)
 *.py source ◄── selfmod/patcher.py ◄── sandbox.py (copy of the tree) + bench.py + tests/
```

## Invariants (each one has a test)

1. **A task's champion never gets worse.** It is re-scored at the start of every `solve`, stays in the race, and is replaced only by a strictly higher
   validation skill. When nothing beats it, the stored champion, not the further-trained copy, is returned.
2. **Selection and reporting never share data.** Validation data / selection-episode seeds (12345…) pick winners; test data / seeds (777000…) are only reported.
   Program agents are searched on training data and chosen among the top candidates by validation.
3. **No default agent.** Genomes come from `AgentPrior`: uniform-random until it has observations, then Gaussian around elites. Hook files ship as no-ops.
4. **Self-written code passes the gate before it runs.** `selfmod/safety.py` rejects imports, loops, dunders, file/process access and over-long programs; the
   hook loader falls back to genesis behaviour on any fault and counts it.
5. **Self-modification is reversible and audited.** `history.py` is append-only and hash-verified; `doctor` re-verifies it.
6. **The constitution is sealed.** The files that police self-modification are not patch targets and are hash-checked.

## Data on disk (`<workspace>/`)

```
journal.jsonl                      every solve (v3 events + forge events)
forge/prior.json                   the learned prior (family stats + elite genomes)
forge/zoo/<task>/<agent-id>/       agent files (agent.npz | state.json + program.py | lm.npz ...) + spec.json
forge/zoo/<task>/best.json         current champion pointer
selfmod/live/<hook>.py             installed evolved hooks (the shipped files in zerowhite/evolved/ are never modified)
selfmod/history/                   objects/ (by sha256), log.jsonl, heads.json
selfmod/patches/<time>.{json,diff} every patch-search report
```

## Extending

* **A new agent family**: subclass `Agent` and `Family` in `forge/families.py` (`build`, `train`, `val_score`, `test_metrics`, `save`, `load`, `describe`) and add it to
  `FAMILIES`. The prior and the race pick it up automatically.
* **A new environment**: put a `class Env` in a `.py` file (`examples/catch_env.py` shows the 4-method contract) and `solve --env-file`, or add it to `forge/envs.py`.
* **A new evolvable hook**: add a genesis file in `evolved/`, an entry in `evolved.HOOKS` and a `Policy` method that calls it, a vocabulary in `selfmod/hooks.vocab_for`,
  and (if it is not the step-size/parent hooks) a benchmark that can tell good from bad.
* **A new patchable module**: add it to `patcher.MUTABLE` *and* make sure `selfmod/bench.py` exercises it, otherwise no patch there can ever show a gain.

## The chat model (v5)

```
data/ (any files) --corpus/readers+adapters--> normalised records --filters(quality, persona, safety, PII)--> state/corpus shards (gz, per file, manifest)
   --tokenizer (byte BPE trained on the corpus) + chat template (assistant-only loss mask)--> state/tokens shards (packed documents)
   --autoscale (machine + data -> model shape, ctx, batch, lr, ...; recorded with reasons)--> train.Trainer (AdamW, cosine, val split, early stop, resume)
   --promote only if val loss <= live--> models/weights/*.npz  +  models/current.json
service/daemon: watcher -> ingest -> (enough new tokens) -> low-priority `chat train --background` -> hot swap into the running server
online.OnlineLearner: feedback log -> examples/pairs -> small steps on a copy (replay mixed in) -> regression gate on probe docs -> version + swap (or reject)
```

* `chat/model.py` is the whole transformer with a hand-written backward pass (RMSNorm, RoPE, SwiGLU, tied embeddings, packed-sequence masks); a test checks every gradient against finite differences.
* `chat/generate.py` runs generation with a KV cache and checks it against the full forward pass.
* `chat/config.py` is the single schema (type, default, doc per option). Nothing else holds a tunable value; a test checks that docs/CONFIG.md is its generated output.
* Network code exists only in `datasets.py`, `chat/corpus/fetch.py` and the local service (`chat/service/server.py`, `client.py`) - enforced by a test.
* `voice/` is independent of the chat package except for config and paths: audio I/O, VAD, G2P + formant synthesis, TTS/STT engine plug-ins, and a CTC recogniser trained in NumPy.

## v6: how the autonomy layer sits on top

```
  inbox (any files)                                                    machine
        |                                                                |
   omnivore.devour --sniff by content--> archives (bounded, salvaged)    substrate.probe / plan
        |                  \--> tables / JSON / SQLite / npy -> Table -> build -> task     (limits derive from here)
        |                   \--> prose -> corpus ;  class Env -> sandbox smoke test (--allow-code)
        v
   <ws>/omnivore/  ledger.json  quarantine.jsonl  tasks/<name>/
        |
   organism  observe -> decide -> act -> learn          state.json, log.jsonl (every decision + reason)
        |        |                  |
        |        |                  +--> forge.Generalist.solve(task, budget)  -> zoo (champions, transfer)
        |        +--> stuck? -> selfmod.hooks.evolve_hook (test-gated, versioned)  |  forge.poet
        +--> doctor (periodic and after every self-modification)
```

The new modules call the sealed `selfmod` files only through their public functions (`sandbox.run_code`, `hooks.evolve_hook`) and none of them is itself in the constitution, so the seal is unchanged (`doctor` confirms it).

## Wide observations for program agents (6.3, 6.4)

A program agent reads at most 24 numbers. Wider observations used to be truncated to the FIRST 24 (6.2). 6.3 compresses them with a whitened PCA fitted on random-policy probes (`forge/reduce.py`; only when `obs_dim > 24`): on a 40-number task whose signal sat in the last 16 numbers, truncated agents
stayed at chance (-0.12 to 0.01 over 4 seeds) and projected agents reached 0.88, 0.79, 0.34, 0.00 (better, not reliable: the program search is stochastic). 6.4 adds a nonlinear alternative, `forge/latent.py` (`ProgramPolicyAgent.encoder = "latent"`): a Perceiver-style bottleneck where every
input number becomes a token (value embedding + position embedding), a few learned latent queries cross-attend over all of them (cost ~ D x M, not D^2), and a tanh layer mixes the result into k numbers; gradients are written by hand (the engine has no batched matmul) and checked against finite differences.
On pictures of one blob (a curved 2-D set in 144 dimensions) it reconstructs far better than PCA with the same number of codes (test MSE 0.067 vs 0.70 at k=2, 0.027 vs 0.25 at k=8). **It did not make evolved programs better**: on a ring-shaped decision task PCA agents scored 0.81 / 0.63 / 0.69 and encoder agents 0.58 / 0.55 / 0.84 (3 seeds
each), and fitting it is ~10x slower. So: a better compressor, not shown to be a better input; PCA stays the default.
