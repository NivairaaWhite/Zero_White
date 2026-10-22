# What was merged from `zero.zip`, and what was not

`zero.zip` contained 17 reference repositories. The brief was "merge useful features, no LLM as operator". Each one was judged against
that, and against the constraint that this package depends on NumPy only.

**No code was copied from any of them.** Where an idea was used it was re-implemented from the idea.

## Used (read in detail)

| repo | what was read | what became of it |
|---|---|---|
| `google-research/automl_zero` | `mutator.cc`, `regularized_evolution.cc`, `fec_hashing.cc`, the op list in `definitions.h` | **Regularized evolution** (oldest non-champion dies) and the **FEC cache** (programs with identical behaviour on probe inputs are one program, evaluated once) in `selfmod/islands.py`; insert / delete / alter-constant mutation operators in `selfmod/astmut.py`. In our runs the FEC cache avoided roughly a third to over 95 % of evaluations, depending on the problem. |
| `funsearch` | `implementation/programs_database.py` | The **island model**: sub-populations evolve separately, the weaker half is periodically reset from the stronger islands' champions. FunSearch's proposer is an LLM; here it is grammar-guided AST mutation (`astmut.py`). |
| `poet` | `poet_algo.py`, `reproduce_ops.py` | `forge/poet.py`: mutate environment parameters, **minimal criterion** (not too easy, not too hard for the parent's agent), **novelty** filter, child agent initialised from the parent, **direct transfer** between environments. POET's agents are ES-trained neural policies; here they are evolved programs. |

## Used (concept level only; their code was not read)

| repo | idea | where |
|---|---|---|
| `PySR` | symbolic regression by evolution, then local tuning of constants | `program_formula` family; `astmut.tune_constants` ((1+1)-ES on literals) |
| `deap` | subtree crossover/mutation, estimation-of-distribution search | `astmut.crossover`; `forge/prior.py` (Gaussian around elites, in normalised gene space) |
| `pyglove` | programs as mutable symbolic trees | the AST-as-genome design |
| `nsjail`, `judge0`, `E2B` | run untrusted code in a separate process with wall-clock and resource limits and a scrubbed environment | `selfmod/sandbox.py`, at the level a pure-Python cross-platform package can honestly provide (no namespaces/seccomp except `unshare -n` when available) |

## Not merged, and why

| repo | reason |
|---|---|
| `MetaGPT`, `OpenHands`, `DSPy`, `AI-Scientist-v2` | LLM-driven by design. The brief rules an LLM out as operator, and there is nothing left to merge once the model is removed. |
| `E2B`, `judge0` (as products) | hosted / Docker services for running LLM-generated code; the *principle* was kept (above), the service was not. |
| `gvisor`, `nsjail` (as code) | kernel-level sandboxes in Go / C++ for Linux only; cannot be part of a numpy-only cross-platform package. `doctor` reports whether `unshare` network isolation is available. |
| `brax` | JAX physics engine; would make JAX a dependency. Two small numpy environments (`MountainCar`, parametrised `CartPole`) cover the same ground for this purpose. |
| `crafter`, `nle` | heavyweight environments (rendering stack / compiled NetHack). `--env-file` lets you plug any environment in. |
| `google-research` (rest) | hundreds of unrelated projects; only AutoML-Zero was relevant. |

## Carried over unchanged from `zero_white_3.zip` (v3)

The self-modifying classifier (`zero_white.py`, `self_model.py`, `architecture.py`, `meta_optimizer.py`), curiosity, the autograd engine, CVAE and character LM,
dataset loaders, checkpoints, report generation and the `autopilot` workflow, with all 63 original tests still passing. What changed in v3 code: default
models removed (see CHANGELOG), the Evolver's step-size and parent rules became hook calls, flat imports became package-relative imports.
