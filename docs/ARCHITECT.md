# The chat model as architect, wider patcher reach, topology invention (6.5)

Three linked pieces that let Zero White change *how* it solves a task, not only the numbers inside a fixed solver.

## 1. The architect (`forge/architect.py`)

After a plateau (a Generalist race that stopped improving) the architect is asked for **hypotheses** about how to do better. A hypothesis is one line of a deliberately tiny grammar, so a small model can write it and nothing it writes can do more than the grammar allows:

```
HYPOTHESIS {"kind": "program", "mutator": "neuron", "n": 1}                          one named edit of the champion program (astmut.MUTATOR_KINDS)
HYPOTHESIS {"kind": "genome", "family": "mlp", "changes": {"hidden": 64, "lr": {"x": 0.5}}}      other values for named genes of the champion's family
HYPOTHESIS {"kind": "hook", "hook": "sigma_update"}                                   ask the self-modifier to search for a better search rule (its own gate)
HYPOTHESIS {"kind": "patch", "file": "evolve.py"}                                     a DRY RUN of the source patcher: reports, never applies
```

Nothing is free text that gets executed. A program edit is made by `astmut` (the same safe grammar every evolved program lives in); a genome change is clipped to the `Gene` ranges; a hook or patch is a name from a short list. A line that does not parse is counted and dropped.

**The verdict, which the proposer never sees:**

* **Paired control.** A program edit is scored on the same held-out episodes (or validation blocks) as the incumbent. A genome change trains the incumbent's genome and the new genome from the same seeds for the same budget. Per-pair differences go through `stats.gate`, the sign-flip test that also judges self-written hooks and source patches.
* **Multiple testing.** Every hypothesis tested in one `improve` call shares the significance level (`alpha / number tested`); a call with too few pairs to ever reach it tests fewer.
* **Hit-rate back-off.** Each proposer is a bandit arm with its own measured acceptance rate. A chat model that keeps proposing nonsense falls to a small probing share (0.1 to 0.9 of proposals), and a task whose architect rounds keep failing waits exponentially longer before being asked again (0, 1, 3, 7, 15, 16 rounds).
* **Pinning.** An accepted program agent is pinned as the task's champion until the search finds a strictly better selection score; if the architect's result is not better on full validation, the previous champion is restored.
* **Export.** Accepted hypotheses are written as chat training text in the exact prompt/answer format the `ChatProposer` uses (`zerowhite architect export`), so training the chat model on its own validated successes is a loop a person can run. Rejected ones go only into `architect-lessons.txt`, never in the answer position.

```
zerowhite solve --csv data.csv --architect               # the Generalist asks the architect after a plateau (step 5c)
zerowhite architect run --csv data.csv --budget 1.0      # architect only, on one task
zerowhite architect report | export | ask
zerowhite organism --architect ...                       # the organism may spend a cycle on it
```

Files: `<workspace>/architect/ledger.jsonl` (every hypothesis and verdict), `state.json` (arms, cooldowns).

### What it has and has not shown

* End to end on a toy regression the architect accepted a program edit with p of about 0.001 and moved validation R² from 0.31 to 0.88; the MLP genome path and the back-off and export loops are tested.
* The built-in `HeuristicProposer` is a **Thompson-sampling bandit over the grammar, not a language model**. It runs whenever no chat model is trained, and it is the benchmark the chat proposer has to beat (`report()` prints both acceptance rates). The chat proposer has **not been shown to beat it**. The loop is the mechanism; whether it pays is measured, not assumed.
* On the small gridworld task at the budgets used in testing, no hypothesis survived the test. That is the gate working, not a defect.

## 2. Structural program mutators (`selfmod/astmut.py`)

Programs used to change only operators and constants. Three structural mutators change a program's *topology*:

| mutator | edit |
|---|---|
| `gate` | adds a branch (a conditional / select) |
| `neuron` | adds a new hidden unit `t_k` wired into the return |
| `mgate` | multiplies by a learned sigmoid gate |

They are off unless a `Vocab` asks (`structural=p`, default 0.0) or a caller names one (`apply_mutator`), so every earlier search behaves exactly as before. Every result still goes through `repair` and the safety gate.

## 3. Topology invention (`forge/topology.py`, family `topology_net`)

`mlp` searches width and depth. `topology_net` searches **DAGs**: every hidden node has its own width and activation (tanh / relu / sigmoid) and takes input from any earlier node, the raw input included, so skip connections, parallel branches and mixed activations are reachable.

Every mutation is **exactly function-preserving** when it happens (tested to 1e-12), so wiring alone can never make a network worse; the new capacity is learned from there:

* `widen(i, k)`: k more units, random incoming weights, zero outgoing weights
* `add_node(w, act)`: a new node, random incoming, zero outgoing
* `add_edge(j -> i)`: a new connection, zero weights

Each round tries `cands` mutations, fine-tunes each a little, and keeps the best only if it beats the current network on an inner 20 % holdout of the **training** data (not the validation split the Generalist selects on, so architecture selection cannot leak into the race). The family joins ordinary races for classification and regression and wins only on validation skill.

Limits: greedy, not a population; no node removal (pruning is not function-preserving); on a task a plain MLP already solves it mostly keeps its initial wiring and the report says how many rounds changed anything. The family's 5-round test beats a too-small MLP on XOR by more than 0.1 validation accuracy.

## 4. The patcher's reach (`selfmod/patcher.py`, `selfmod/bench.py`)

The source patcher edits tunable values in `patcher.MUTABLE` modules and keeps an edit only if it beats the benchmark under the paired test. Until 6.4 the benchmark ran only some of those modules, so a change to the others could not be seen and was neither rewarded nor caught. Now:

* the benchmark scores **six parts**: digits validation accuracy, CartPole, a landscape search, a VAE, a character LM, and curiosity; `score` is their mean;
* a test profiles a benchmark run and fails if any `MUTABLE` module is never executed, and another changes a default in each newly covered module and checks the benchmark moves;
* the VAE, character-LM and curiosity measures are computed **inside `bench.py`** (through `nn.mlp_raw`), not by the module under patch, so a patch cannot flatter its own measurement.

`bench.py` is now **sealed** (the constitution has 17 files: new members `forage/act.py`, `selfmod/bench.py`, `forge/architect.py`): it is the judge of patches, so the patcher cannot edit it. **Benchmark scores are not comparable with earlier versions** (a different mix of parts); a workspace's patch history keeps its old numbers and the new ones start a new baseline.
