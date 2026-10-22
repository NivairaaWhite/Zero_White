# Organism: the loop that decides what to do next

Until v6 every capability waited for a command: `solve`, `forge`, `evolve`, `selfmod hooks`, `poet`. The **organism** is one loop that wires them together and
decides *for itself*, from what it measures, which to run next, and writes down why.

```bash
python -m zerowhite organism --workspace zw --inbox ~/inbox --minutes 30                  # watch the inbox, improve what is worth improving
python -m zerowhite organism --workspace zw --inbox ~/inbox --forever --self-modify hooks # keep running; rewrite its own search rules when stuck
python -m zerowhite organism --workspace zw --inbox ~/inbox --chat on --minutes 60       # ...and schedule the chat model (zw/chat) among the tasks
python -m zerowhite organism status --workspace zw                                         # what it knows and did (reads files only)
python -m zerowhite organism stop   --workspace zw                                         # graceful stop: it finishes the current attempt and exits
```

A bare `organism` runs 10 decisions and stops; `--forever` is explicit. Everything it decides is in `<workspace>/organism/log.jsonl`; its memory is `state.json`.

## The loop

**Observe.** The inbox (files or folders) is digested by [omnivore](OMNIVORE.md) every cycle: anything new or changed becomes a task or a recorded reason; unchanged files cost
a `stat`. It also reads free memory and the cached [substrate](../zerowhite/substrate.py) probe, and adopts tasks that an earlier `devour` already built. If a chat workspace is part of
the organism (see [The chat model as a task](#the-chat-model-as-a-task)), it feeds that workspace the prose it just digested and reads its data folder the same way.

**Decide.** One rule, in this order:

1. A task that has **never been attempted** is attempted, with a small probe budget (0.25; the chat model is the exception, see below). That first attempt is a *baseline*: it tells the cost of an attempt on this machine and where the task
   stands. It is not counted as progress. Cheap tasks go first, the chat model last.
2. Otherwise the task with the highest **expected learning progress per second**: the exponentially weighted gain in validation skill (or, before any gain has been measured, half the headroom),
   capped by the headroom `1 - skill`, divided by what its last attempt cost. Skill is normalised so 1.0 is perfect, which is what makes the cap meaningful.
3. A task **plateaus** when its skill is within `min_gain` of 1.0 (it is at its ceiling), or when `patience` attempts in a row each gained less than `min_gain`. After every attempt that gains nothing, that
   task's next budget **doubles** (up to `max_budget`): maybe it needed more effort, and if not, the plateau is established at the largest affordable budget. A plateaued task is left alone.
4. Each attempt is **fitted to the time left**: predicted seconds, times a safety pad, must fit `--minutes` and `max_action_minutes`; otherwise the budget is cut. Until a task has 4 attempts the prediction is
   the old one (measured seconds per budget-unit x budget, pad `time_margin` = 1.5). Since 6.3 it is then a fitted curve (`seconds ~ a * units^gamma`, robust Theil-Sen fit, see
   [Cost forecasting](#cost-forecasting-63)) and the pad is measured from the task's own past overruns.
   If the best candidate does not fit at all, the next best that does is taken (a ten-minute language-model run must not block a three-second table when 40 s remain).
5. When **every task has plateaued**, the system is stuck, and that is when it changes *how it searches* (only with `--self-modify hooks`): it picks one of `hooks:select_parent`, `hooks:sigma_update`
   (rewrite a search rule under the test-gated hook machinery of [SELF_MODIFICATION.md](SELF_MODIFICATION.md)) or `poet` (invent harder environments; only offered when you brought an environment task)
   by **Thompson sampling** over what has worked before. The same arm never runs twice in a row. A win re-opens every plateaued task and triggers a self-check (`doctor`).
6. If free memory falls below `min_free_gb` it rests. If there is nothing worth doing, a bounded run ends ("nothing worth doing: ...") and a `--forever` run keeps polling the inbox.

**Act.** Inside `try/except`. A failing task or arm is logged, backs off exponentially (2, 4, 8 ... decisions) and never stops the loop. While everything is backing off, waiting counts as time passing,
so the back-off expires (in 6.0 it did not: a lone failing task was never retried).

**Learn.** Per-task gain, cost and budget; per-arm wins and tries. State is written atomically after every step, so a crash or `kill -9` loses at most the attempt in progress. A state file that cannot
be parsed is renamed `state.json.corrupt-<time>` and the organism starts fresh.

## The chat model as a task

Since 6.1 the [chat model](CHAT.md) is scheduled by the same loop as everything else, by the same measure (learning progress per second). A chat workspace joins the organism when it is
`<workspace>/chat` and exists (`--chat auto`, the default), is created there (`--chat on`), or is given by path (`--chat PATH`); `--chat off` removes it. It becomes a task once its data folder has something in it.

* **Skill** = `1 - validation_nll / ln(vocabulary)`: the share of a uniform guess's uncertainty the model has removed (0 = guessing; 1 is unreachable on real text, so a chat task never "reaches its ceiling";
  it plateaus only by patience).
* **Budget is minutes of training**, which the trainer honours, so the cost of an attempt is known *before* the first one: minutes plus a measured overhead (ingest, tokenise, save; assumed 15 s until measured).
  A no-gain attempt doubles the minutes up to `--chat-max-minutes` (the general model's `max_action_minutes` cap does not apply to it).
* **The first run of a new chat model is not a cheap probe**: the time budget of the first run decides the model's *size*, and every later run keeps that size. A one-minute probe would fix a tiny model for good, so a
  new model starts with `--chat-first-minutes` (10); an already trained one starts with `--chat-probe-minutes` (2) and is judged from there.
* **Gain is measured on one validation set**: the live model is scored before and after on the same documents (new data changes the validation split, and comparing with a number recorded earlier would mix
  "learned more" with "the exam changed"). A run the trainer does not promote counts as no gain.
* **One inbox file teaches both models**: prose that omnivore pools into the text corpus is mirrored (atomically, only when it changed) to `<chat>/data/_omnivore/`, where the chat ingestor reads it like any other file.
  `--no-chat-feed` turns that off.
* **New or changed data re-opens a plateaued chat task** (a plateau on old data says nothing about new data). A hook win does not: hooks steer the general model's search, not the chat trainer.
* **Coexisting with the chat service.** Training takes one lock. While another process (the service's auto-training, a `chat train`) holds it, the organism does not touch the chat data and waits instead of failing; the
  service hot-swaps any model the organism promotes, since it watches the model pointer. Running both trainers works but queues behind one lock; set `daemon.autotrain=false` if the organism should be the only scheduler.
* **Stopping.** `organism stop` (a STOP file) or Ctrl-C ends a chat attempt at a checkpoint, the organism exits, and the next attempt resumes that run. Such an attempt is not counted as a result.

## Options

| flag | meaning |
|---|---|
| `--inbox PATH...` | folders / files digested every cycle |
| `--minutes N` / `--cycles N` / `--forever` | bounds (a bare run = 10 decisions) |
| `--self-modify off\|hooks` | what it may do when stuck. `off` (default): it stops or waits. Source patching is never scheduled by the organism |
| `--allow-code` | run environment files found in the inbox (your code; see OMNIVORE.md) |
| `--probe-budget` 0.25, `--candidates` 6, `--rungs` 3 | size of attempts |
| `--min-gain` 0.005, `--patience` 3 | what counts as "nothing happened", and how many in a row make a plateau |
| `--time-margin` 1.5 | safety factor on predicted attempt cost |
| `--hook-evals` 200 | search effort of one self-modification attempt |
| `--pool NAME` | which shared text corpus prose goes into |
| `--chat off\|auto\|on\|PATH` | whether, and where, the chat model is one of the tasks (default `auto`: `<workspace>/chat` if it exists) |
| `--chat-minutes` 10, `--chat-first-minutes` 10, `--chat-probe-minutes` 2, `--chat-max-minutes` 120 | minutes of training: a standard attempt / a new model's first run / the first attempt on a trained model / the cap |
| `--chat-set KEY=VALUE` | override a chat config value for the organism's runs (repeatable) |
| `--no-chat-feed` | do not mirror inbox prose into the chat data folder |

`min_gain`, `patience` and the budgets are explicit settings, not learned.

## What it will not do

* Edit its own source files. `hooks` rewrites two small files under `evolved/`, behind the safety gate, held-out landscapes, a margin and versioned rollback; the five-gate **source patcher** stays a separate, explicit command.
* Run your code unless `--allow-code`. Move, edit or delete your files. Run two organisms on one workspace (a lock file holding the process id; a stale lock from a dead process is ignored).
* Stop because something failed: every action is isolated.

## Worked examples (real runs, one CPU core, 3.7 GiB free, Linux, Python 3.12, NumPy 2.4)

**A. A hostile inbox, 7 minutes, `--self-modify hooks --patience 2 --min-gain 0.02`.** The inbox held 12 broken files (cp1252 semicolon CSV with decimal commas, UTF-16 CSV, truncated JSON Lines,
truncated gzip, a zip with no directory, an environment file, prose, junk, an image, an empty file). Digestion produced 8 tasks and set 3 files aside with reasons. In 25 decisions (24 attempts, 414 s of
compute) the organism tried every task once, then spent its time where progress per second was highest. Six tasks reached their ceiling (skill >= 0.98); `excel_export` (skill 0.276 after 6 attempts, budget
raised to x4) and `utf16` (0.617 after 4) plateaued by patience. In my test files `excel_export` keeps only the magnitudes of the two inputs the label depends on (the sign was thrown away), so the
label is mostly unrecoverable; `utf16` holds an XOR-style rule plus noise from a column that is not in the file, which about 140 training rows barely support. Giving up on both after doubling the budget is the intended outcome. With all 8 plateaued
it chose `poet` (it had an environment task), which produced 3 environments in 7.6 s and counted as a win. (The text corpus in that run repeats a few sentences, so its 0.98 says nothing about language ability.)

**B. Forced self-modification, 3 cheap tasks.** All three hit their ceiling within about 90 s. The organism then tried six hook rewrites over the next minutes: `hooks:select_parent` **won once** (held-out
benchmark score 0.720 -> 0.771, installed as version 1, self-check clean) and was refused twice afterwards; `hooks:sigma_update` was refused all three times. After the win every task was re-opened and
re-tested, found still at ceiling, and the loop moved on to the other arm. The 5 refusals are the gate doing its job: once the live rule scored 0.771, candidates scoring 0.7707, 0.760, 0.766 and 0.771 did not
beat it by the margin.

## Bits as a currency (6.4, `--metric bits`)

`skill` is normalised per task, so +0.1 on a three-row table and +0.1 on a language model are not the same learning. `--metric bits` (default stays `skill`) converts skill to bits of information (`metric.py`) and schedules by bits gained per second:
classification as a symmetric channel (n * [I(acc) - I(majority rate)]), regression as a Gaussian channel (0.5 * log2(1 / (1 - R^2)) per sample), the language model as bits per token below a uniform guess. Control tasks have no information
reading and are converted at `--bits-per-skill` (50) and flagged 'assumed'; so are chat bits (token count taken from training tokens, because the trainer does not report the validation count). Bits are monotone in skill, so ordering candidates on one task never changes; what changes is
the ranking BETWEEN tasks: in the test, two tasks with equal skill gains are separated by what is at stake (2000 vs 20 validation examples). The exchange rate for control tasks is a judgement, not a measurement.

## Sleep (6.4, `--sleep`, opt-in)

When every task has plateaued the organism may "sleep" (at most once per `--sleep-cooldown`, 900 s): `chat/sleep.py` runs the online learner's own update on the pending /good /bad /fix feedback with a longer, gentler, more replay-heavy schedule (4x the steps,
half the learning rate, twice the replay, half the tolerated regression) and a **stricter gate**: the probe documents are stratified into four age bands of the training corpus and the gate watches the WORST band, not the mean (the test builds a case
where the mean improves by 37% and one band worsens by 30%: the mean-based gate would accept it, this one refuses). Reports the loss per band before and after. Honest scope: bands are age bands, not topics; it learns the same feedback the online learner would,
in one safer pass; there is no generative "dreaming" or distillation.

## Cost forecasting (6.3)

`forecast.py`; the numbers it uses live in each task's state (`cost_obs`: the last 24 `[units, seconds]` pairs, `cost_ratios` / `over_ratios`: actual / predicted).

* **Point estimate.** With at least 4 attempts at 2 or more different sizes, `seconds ~ a * units^gamma` is fitted on the log-log pairs with Theil-Sen (median of pairwise slopes, so one throttled or interrupted attempt barely moves it):
  the exponent `gamma` (clipped to [0.7, 1.6]) from the last 24 attempts, the level `a` from the last 12, so that a long run of same-size attempts does not erase what is known about scaling while slow drift (throttling, other load) is followed.
  Before that, the old linear estimate is used unchanged.
* **Pad.** Every finished attempt whose prediction came from the fitted curve records `actual / predicted` (what the organism said when it decided). The pad is the split-conformal quantile of the last 24 of those ratios at `--coverage` (default 0.9: the
  ceil((n+1) * 0.9)-th smallest). If the next ratio resembles the recent ones, the attempt then fits inside prediction x pad about 90% of the time. With too few ratios to support that claim (fewer than 9) the pad is `--time-margin`,
  or the worst overrun seen if that is larger; it is bounded to [1, 10]. Overruns of the early *linear* predictions are deliberately not mixed in: in a real two-task run they were x3.4 and, kept in the window, would have inflated
  the pad for the next 24 attempts although the fitted curve was predicting within 0.6-1.3x. The chat model gets the same treatment on its *overhead* (tokenising, benchmarking), because the training minutes themselves are honoured by the trainer.
* **Budget cut.** With a fitted curve the budget that fits is found by bisection on `cost(b) * pad <= time left`, not by a straight-line division.
* **`--fixed-margin`** restores the exact pre-6.3 behaviour (linear estimate, fixed pad).
* **What was measured** (synthetic cost models, 150 simulated lives each, target coverage 0.90; `tests/test_v63.py` asserts these properties with margins). Share of attempts that fit inside prediction x pad, and mean pad / actual cost:

  | cost regime | 6.2 rule (linear x 1.5) | 6.3 |
  |---|---|---|
  | linear + overhead | 0.99, 1.64 | 0.97, 1.57 |
  | super-linear (units^1.4) | **0.88**, 1.54 | 0.92, 1.41 |
  | slow drift (+4% per attempt) | 0.90, 1.44 | 0.88, 1.31 |
  | heavy-tailed spikes (8% of attempts x2.2) | 0.93, 1.70 | 0.94, 1.84 |

  So: where the old model was optimistic (super-linear) the new one covers more; where the old pad was needlessly wide (linear) it is narrower; under drift it is slightly *less* covered with a narrower pad (exchangeability, which the guarantee needs, is
  violated by drift); with spikes it buys a little coverage with a wider pad. These are simulations of cost *shapes*, not measurements on real training runs.
* It cannot foresee a change it has no history of (another program starting, thermal throttling that begins mid-run): the ratios adapt after the first overrun, not before it.

## Honest limits

* **The gain it optimises is validation skill**, measured on the same data the champion was chosen on, so it is mildly optimistic. The test metric is logged with every attempt but not used for scheduling.
* **The cost model was optimistic, and is now calibrated, not perfect.** In run A (6.2), attempts were predicted at 2 s and took 8 s, 56 s and took 86 s, because the estimate was linear in the budget and the first attempt is small. Since 6.3 the
  estimate is a fitted curve and the pad is measured (below), but the first few attempts of every task still use the old linear estimate and the fixed 1.5 pad, a task whose cost suddenly changes (other load on the machine) is
  under-covered until a few attempts have passed, and the deadline is still only checked before an action: an attempt that runs long can overrun `--minutes` by up to its own length.
* **A hook win is a win on a proxy benchmark** (cheap synthetic landscapes plus held-out ones). In example B the tasks were already at their ceiling, so it shows the mechanism works, not that the new rule
  helps your real tasks. Re-opening the tasks after a win is the test of that; it can only show a benefit where there is headroom.
* Only **two small hooks** can be rewritten (`select_parent`, `sigma_update`). POET is specific to the CartPole family and helps only environment tasks with a similar signature.
* Thompson sampling over two or three arms with a handful of tries is mostly exploration; do not read the win counts as statistics.
* **What the two models share, and what they do not.** They now share the task list, the scheduler, the clock and the inbox's prose. They do **not** share hooks or the zoo: the evolved hooks steer the general
  model's evolutionary search, which the chat trainer does not use, so the organism never rewrites anything on the chat model's behalf (and a hook win does not re-open it). Serving, chat sessions and online learning from feedback remain the
  chat service's job.
* **The chat overhead is an estimate until measured**: the first attempt on a large new corpus can spend far more than the assumed 15 s on tokenising and benchmarking, and so can overrun a tight `--minutes`. The language model's skill
  is a loss-based proxy for quality (a lower perplexity is not the same as better answers), and attempts are whole training runs of minutes, so a chat task is scheduled far more coarsely than a table.
* One process on one machine. The substrate probe's `accelerators` only *report* what is installed and its GFLOP/s is the CPU's. Since 6.2 the **chat model's training** can use a CUDA GPU (see [CHAT.md](CHAT.md#training-on-a-gpu); the
  organism picks it up through `compute.backend=auto` with nothing to configure, and a first-time CuPy kernel compile lands in that attempt's measured overhead). The general model and everything else still run on NumPy. The speed probe
  itself varies between runs (71 to 116 GFLOP/s on the same machine here).
* Runs of hours or days have not been tested; `log.jsonl` grows by roughly a kilobyte per decision and is never truncated.
