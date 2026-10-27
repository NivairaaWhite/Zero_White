# Episodic memory (6.5)

`forge/episodic.py`. Until 6.4 the only memory of past work was the zoo (one champion per task, found by *name*) and the learned prior (per-family running means). A task the system had never seen under that name started from random genomes and the trivial program, however much it resembled an earlier one.

```
zerowhite memory status                 # how many records, how well the encoder retrieves
zerowhite memory fit                    # train the encoder on the records so far
zerowhite memory recall --csv new.csv   # which earlier champions does this task resemble?
```

The Generalist uses it by default (`Generalist(..., episodic=True)`); it records every solve and recalls before proposing.

## What is stored and how it is used

* **Records** (`<workspace>/forge/episodic/`): one per task profile, per champion (family, genome, skill, program source), per failed candidate, and per remembered observation window. Records are summaries, not data.
* **Encoder.** A small two-layer numpy network trained with a **contrastive** loss (multi-positive InfoNCE, the SupCon form): two noisy views of a record, and records of the same task (or champions of the same family on same-sized tasks), are pulled together; the rest of the batch is pushed apart. Input: hashed words and character trigrams of the description plus a vector of numbers (log size, kind, simple data statistics, a linear-probe skill). Gradients are written by hand and checked against finite differences in the tests.
* **Recall.** Cosine similarity of the new task against every stored champion vector: the `k` nearest, with genomes and (for program agents) **source code**.
* **Splice.** For program agents the recalled programs are re-indexed to the new observation width, compiled under the same safety gate as any evolved program, and **crossed over** pairwise; the results join the evolver's starting population. For other families the recalled genomes (and a blend of two) replace some of the uniform-random proposals of the race.

## What it does not do

* It does not change the champion rules: a recalled candidate must still **win the race on this task's validation data**.
* It cannot know two tasks are alike in a way their profile does not show (a profile is the data's shape, simple statistics and name).
* With few records the untrained encoder is a random projection of the same features, which is still a usable similarity (the tests compare both). It gets better with the number of tasks seen; on a fresh workspace it does nothing.
