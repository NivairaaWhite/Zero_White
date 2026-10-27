# Omnivore: devour anything without breaking

`zerowhite devour PATH...` takes files, folders and archives of any kind, and for every file it does exactly one of these things:
turns it into a **task** the general model can train on, adds it to a growing text **corpus**, opens it (if it is an archive or a database) and
treats what is inside, recognises it as a **duplicate**, **defers** it (not yet), or **sets it aside with a reason** you can act on.
It does not raise on what a file contains, and it never moves, edits or deletes a source file.

```bash
python -m zerowhite devour ~/Downloads/stuff --workspace zw            # digest; print what happened to every file
python -m zerowhite devour ~/Downloads/stuff --workspace zw --solve     # ... then train the general model on every new task
python -m zerowhite devour --workspace zw --list                        # tasks built so far + every file that was set aside, and why
python -m zerowhite devour data.csv --target price                      # choose the column to predict (default: see "Tables")
python -m zerowhite devour envs/ --allow-code                           # let YOUR environment files (class Env) run, after a sandboxed smoke test
python -m zerowhite substrate --workspace zw                            # the machine measurements the size limits are derived from
```

To try it on something realistic: `python examples/messy_inbox.py /tmp/inbox` writes 12 files, each broken in a different way (a truncated gzip, a zip with no directory, Windows-1252 with decimal commas, JSON Lines with garbage, ...).

Python: `from zerowhite.omnivore import devour, load_tasks, Library, Limits`; `devour(paths, workspace)` returns a `Digest`
(`.entries`, `.counts()`, `.tasks`, `.summary()`, `.to_dict()`); `load_tasks(workspace)` returns task objects for `Generalist.solve`.

## What each kind of input becomes

The file's **content** decides, never its name (`data.txt` may be JSON; `notes.csv` may be prose).

| the bytes are | what happens |
|---|---|
| a delimited table (comma, semicolon, tab, pipe or whitespace) | supervised task (classification or regression), see "Tables" |
| JSON: an array of records, an object holding such an array, a columnar object (`{"a":[..],"b":[..]}`), an object of objects; **JSON Lines**; a **truncated** array | the same table path; nested keys are flattened (`a.b`) |
| a numpy `.npy` (2-D numeric or structured) | table (columns `c0, c1, ...`); a 1-D array is set aside (nothing to predict from) |
| a **SQLite** database | each table of at least 3 rows and 2 columns becomes a task (the file is opened read-only) |
| prose, markdown, source code, HTML/XML (tags removed), anything else readable | appended to the shared corpus `corpus-notes` (name with `--pool`); a char-level text task once it holds >= 500 characters |
| a Python file defining `class Env` | an environment task, **only with `--allow-code`**; otherwise it waits (`deferred`) |
| a Python file without `Env` | text |
| `.gz` `.bz2` `.xz` `.zip` `.tar` (and `.tar.gz`, `.npz`, nested up to 3 deep) | opened safely; every file inside is digested by the same rules |
| image, PDF, audio, parquet, executable | set aside with the specific reason and what to do instead (e.g. audio belongs in the chat model's `data/audio` with transcripts) |
| unknown binary, empty file | set aside: size, entropy, how many readable strings |

## What gets repaired (and reported)

Every repair is listed in the report next to the file, never applied silently.

* **Encoding**: UTF-8 BOM, UTF-16 / UTF-32 with or without a BOM, Windows-1252 / Latin-1, stray bad bytes (replaced and counted), trailing NUL padding
  (what an interrupted write leaves on some file systems).
* **Tables**: the delimiter is *found* (the one that makes rows the same width; decimal commas in a `;` file do not fool it), a preamble before the header
  is skipped, a missing header is detected, ragged rows are padded or trimmed, an unbalanced quote does not swallow the file, a half-written last row is dropped,
  decimal commas (`3,14`), thousands separators (`1,234.5`, `1.234,5`, `1 234,5`), currency, percentages and `(negative)` parentheses are read as numbers,
  `NA`/`null`/`?`/`-` are missing, dates become days since 1970, yes/no/true/false become 0/1, categories become one-hot columns (top 30 + "other"),
  missing numbers are filled with the median plus a "was missing" indicator, constant columns and row-index columns are dropped, identifier / free-text
  columns are dropped, classes with fewer than 5 examples are set aside (they cannot be learned or tested).
* **JSON**: unparsable lines in JSON Lines are skipped; a truncated array keeps every complete record before the damage.
* **Archives**: a truncated gzip / bz2 / xz stream yields the bytes before the damage; a zip whose central directory is missing (a stopped download) is
  scanned member by member; a tar that ends early yields the files before the end; a member cut off inside its data is kept as far as it goes.

## Tables: how the task is chosen

* **Target**: `--target NAME` (or a column number) if you give one; else the single column named `target`, `label`, `class`, `y`, `outcome` or `response`
  (if several match, the last column); else the **last column**. The report says which rule was used.
* **Task kind**: integer-valued with at most 20 distinct values, or non-numeric labels (<= 50 classes) -> classification; otherwise regression.
  Text labels with more than 50 distinct values are treated as an identifier, not a label, and the file is set aside with a message that says so.
* At least 30 usable rows are needed (to split train / validation / test). Tables with more rows than the machine can usefully train on are sampled (the cap comes
  from the measured speed; see `substrate`). The split is deterministic: the same data always gives the same split, so a stored agent is judged on the same
  validation set every time.

## Safety contract

* **Read-only.** Source files are never moved, edited, deleted or added to; "quarantine" is a record in the workspace, not an action on your file.
* **Your code does not run** unless you pass `--allow-code`. Even then an environment file is first executed in the sandbox (a separate process with a timeout, memory and file-size
  limits, and, where `unshare` is permitted on Linux, no network namespace: best effort, detected at runtime) for a smoke test; only if that passes is it loaded
  in-process, which is the same trust level as `solve --env-file`.
* **Bounded.** Every limit is enforced on what is actually decompressed, not on what a header claims: one archive may not exceed the total / per-file / member-count
  caps (zip bombs), nesting stops at 3 levels, and a file larger than the read cap is read from its start only (the report says so). Member names are only labels:
  nothing is ever written under a name taken from an archive. Symbolic links are not followed; the workspace is never digested (so an inbox that contains it does not feed itself).
* **Does not break on damage.** Every file is processed inside its own exception boundary; an unexpected error becomes one quarantine entry with the error text.
  The test suite feeds it random bytes and truncations and bit-flips of every supported format and asserts: no exception, and every file has exactly one recorded outcome.
* **Cheap to repeat.** A ledger (`<workspace>/omnivore/ledger.json`) remembers each file by size and modification time and each content by SHA-256.
  Unchanged files are skipped without being read; identical content under another name is a duplicate; a file that was set aside is looked at again **automatically
  when its content changes** (a download that finishes) or with `--retry`. A file modified in the last second is `deferred` (it may still be being written).
* **Self-healing.** A ledger or state file that cannot be parsed is renamed `*.corrupt-<time>` and rebuilt; every file is simply digested again and keeps its task name.

## Files it writes (all under `<workspace>/omnivore/`)

```
ledger.json         content hash -> what happened; path -> (size, mtime, hash)
quarantine.jsonl    append-only: every file set aside, with reason and what would make it retry
tasks/<name>/       spec.json  +  data.npz (table) | corpus.txt (text) | env.py (a copy of your environment)
```

## A learned delimiter ranker (6.4, `--learned-sniffer off|rescue|arbitrate`)

`tables.find_delimiter` scores candidates with `consistency^2 * log2(width)`, a guess with blind spots (a space-delimited file with decimal commas splits "consistently" on the commas). `omnivore/learned.py` **learns the scoring function**: for each candidate the same dozen pieces of
evidence (consistency, width, how numeric the cells look, spaces inside cells, empty cells, decimal-comma density, the old score itself, ...), a ~600-parameter network shared across candidates, trained on files from `omnivore/synth.py`. The rules always run first; **rescue** (default) lets the
ranker supply the delimiter only when the rules found no table, **arbitrate** also lets it replace the rules' delimiter; in both a hinted table is accepted only if it has strictly more numeric columns than what the rules made (prose scores 0, so it can never be turned into a table; tested).

| delimiter | rules | ranker |
|---|---|---|
| held-out synthetic files (500, seed unseen) | 0.83 | 1.00 |
| 16 files written independently of the generator | 14 / 16 | 16 / 16 |
| end to end on 500 synthetic files: correct table delimiter, `off` / `rescue` / `arbitrate` | 0.81 / 0.83 / 0.95 | |

**What I tried first and removed.** A byte-level network (byte and byte-pair histograms -> delimiter, header, decimal mark, encoding) scored ~100% on held-out synthetic files and **12 of 16 on the independent ones, worse than the rules' 14, with probability 1.0 on its wrong answers**; it was deleted
rather than shipped. Honest scope of what remains: 16 independent files is a small sample and I wrote both the generator and those files; the ranker picks among five delimiters and nothing else: headers, decimal marks, encodings and all repair (ragged rows, quotes, truncation, preambles) stay rule-based. Default `rescue`; use `arbitrate` only after
checking it on your own files.

Two bugs in 6.2's table parser were found on the way: a short row in an all-text table raised `AttributeError` in the header check (fixed), and `HistoryStore.lineage` could loop forever (see CHANGELOG 6.3).

## Honest limits

* The heuristics (delimiter search, header detection, target guess, class / identifier rules) are **rules**, not learned from your data. They are conservative and
  every decision is reported, but a table with an unusual layout can be misread: check the report's `target = ...` line, and use `--target`.
* Text from many files is pooled into **one** corpus task; it is not separated by topic or author. When new text arrives the vocabulary can change, in which case the
  stored language model for that corpus cannot be reused and is rebuilt.
* Images, PDFs and audio are recognised and explained, not read. Parquet needs `pyarrow` (an optional extra). Encrypted archive members are skipped (no password is guessed).
* Only the first part of a file larger than the read cap is used; the report says how much.
* The speed of a task depends on the machine: row and character caps are derived from a short matrix-multiply benchmark that is itself noisy (we measured the same
  machine at 71 and 116 GFLOP/s on different runs), so the caps are approximate.
