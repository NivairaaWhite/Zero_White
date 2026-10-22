# The data folder

Zero White's chat model trains on **whatever you put in the data folder** — `<workspace>/data` by default (`paths.data` moves it).
There is no built-in training data: with an empty folder, `chat train` stops and tells you what to do.

```
zw-chat/
  data/                      <- you put files here (any sub-folders)
    instruct/alpaca.jsonl
    code/magicoder.jsonl
    persona/dataset.json     <- optional: settings for everything beneath this folder
    persona/pippa.jsonl.gz
    audio/                   <- speech datasets for the voice recogniser (not read as text)
    _online/                 <- written by online learning (your accepted corrections), kept as ordinary data
  state/                     <- derived: normalised shards, token shards, manifests (safe to delete; rebuilt)
  models/                    <- tokenizer + versions of the model + which one is live
```

New, changed and deleted files are noticed automatically (`chat train` and the background service both look; a file must be
unchanged for `data.settle_s` seconds so half-copied files are never read). Each file is processed once; one unreadable file is
reported and skipped, never fatal.

## What can be in it

| container | notes |
|---|---|
| `.jsonl` `.ndjson` `.json` (array, or `{"rows": [...]}`) | streamed, so multi-GB files are fine |
| `.csv` `.tsv` | header row; delimiter sniffed |
| `.parquet` | needs `pip install pyarrow` (optional) |
| `.txt` `.md` `.rst` and source code (`.py` `.js` `.ts` `.java` `.c` `.cpp` `.go` `.rs` `.sh` `.sql` `.html` ... ) | plain text; split into chunks of `data.text_chunk_chars` |
| `.gz` `.bz2` `.xz`, `.zip` `.tar` `.tgz` | any of the above, compressed or archived (archives are walked member by member; an oversized member is skipped) |

Anything not understood is listed by `chat status` with the reason; it is skipped, never guessed at.

### Record layouts recognised automatically

A sample of rows is inspected and the best-fitting *adapter* is chosen (`chat ingest` prints which one):

* question/answer columns — `instruction`/`input`/`output` (Alpaca), `prompt`/`response`, `question`/`answer`, `problem`/`solution` ...
* conversation lists — `messages`, `conversations` (ShareGPT `from`/`value`), `conversation`, OpenAI/ChatML role lists
* preference pairs — `chosen`/`rejected` as text or message lists, Anthropic `\n\nHuman: ... \n\nAssistant: ...` strings, UltraFeedback / Nectar styles
* PIPPA — `bot_name`, `bot_description`, `bot_definitions`, `conversation` (`{{char}}` / `{{user}}` placeholders replaced)
* MBPP-style code tasks (`text` + `code` + `test_list`), critique/feedback sets (Prometheus-style judge data)
* plain text

If your columns are unusual, say so once in a `dataset.json` next to the files:

```json
{
  "adapter": "qa",
  "mapping": {"user": "question_text", "assistant": "gold_answer", "system": "persona"},
  "weight": 2.0,
  "tags": ["my-corpus"],
  "filter": {"quality": {"min_assistant_chars": 20}},
  "data": {"val_fraction": 0.05}
}
```

A `dataset.json` applies to everything beneath its folder (the nearest one wins). `weight` makes that folder count more in training;
`filter` and `data` override the config keys of the same name for those files only. `//` comments are allowed.
When nothing fits, the error names the columns it saw and prints a `mapping` you can paste.

## Preference data and "learning from feedback"

Records with a chosen and a rejected answer are used in a second, short **preference phase** (DPO) after normal training, and — if
`data.preference_as_sft` is true — their chosen answers also count as ordinary examples. The phase is gated: a model that gets worse at
its core language modelling by more than `train.preference_max_regression` is not kept.

## Filters

Applied to every record, in this order: **quality → persona → safety → PII redaction.** All of it is configuration
(`filter.*`, see [CONFIG.md](CONFIG.md)); a folder's `dataset.json` can change it for that folder.

* **quality**: too short/long, no assistant turn, degenerate repetition. Exact duplicates are removed across all files and runs.
* **persona** (off by default): keep only records whose persona text matches keywords — see below.
* **safety**: two layers on purpose.
  1. *Always on, no switch:* sexual content together with a minor or a family member is dropped. Public roleplay corpora (PIPPA among them)
     contain sibling/parent roleplay of exactly this kind.
  2. `filter.safety.drop_sexual` (default **true**): drop sexually explicit records. Plus your own `drop_terms` / `drop_regex`.
  The detector is a lexicon with word boundaries and a list of benign phrases ("breast cancer", "sexual harassment policy"). It is
  conservative about false positives, which means **it misses euphemistic or misspelled text**. It lowers risk; it does not remove the need to look at your data.
* **PII**: e-mail addresses, phone numbers, IPv4 addresses and Luhn-valid card numbers are replaced by `<email>`, `<phone>`, `<ip>`, `<card>`
  (`filter.safety.redact`). Digit groups that are not phone numbers (dates, order numbers) are left alone.

`chat ingest` prints how many records each filter dropped, so you can see what it is doing.

### The PIPPA "sister" persona (a wholesome companion)

Your original script kept PIPPA conversations whose persona text contained `sister`, `sibling`, `'sis '`, `step-sister`, `imouto`, `onee-chan`...
Two problems with that, both fixed here: `'sis '` also matches *analysis* and *thesis* (matching is now by whole words, `filter.persona.match = "word"`),
and the corpus itself contains explicit material (the safety layers above run first-class, not as an afterthought).

```bash
zerowhite chat fetch pippa --limit 20000          # or copy the PIPPA files in by hand
```

`data/persona/dataset.json`:

```json
{
  "adapter": "pippa",
  "filter": {
    "persona": {
      "enabled": true,
      "keywords": ["sister", "sibling", "big sis", "sis", "step-sister", "stepsister", "imouto", "onee-chan", "oneesan"],
      "match": "word",
      "where": ["system", "meta"]
    },
    "safety": {"drop_sexual": true}
  }
}
```

`examples/persona_sister/` has a complete workspace config (name, system prompt, greeting) for the "like a sister" companion.
The system prompt matters as much as the data: a model trained on a few thousand roleplay chats mostly learns the *tone*.

## Downloading public datasets (optional)

```bash
zerowhite chat fetch --list
zerowhite chat fetch alpaca magicoder openhermes --limit 20000
zerowhite chat fetch somebody/some-dataset --subset default --split train      # any Hugging Face id
```

Rows come through Hugging Face's public "dataset viewer" API, page by page, **resumable** (a torn last line is repaired and the download continues),
with retries/back-off. Gated datasets need a token in the environment variable named by `fetch.token_env` (default `HF_TOKEN`); the error message says so.
`fetch` is the only command that uses the network besides the local service; nothing else ever does (a test enforces it).

| name | dataset | what fetch does |
|---|---|---|
| `magicoder` | Magicoder-OSS-Instruct-75K | rows -> `data/code/` |
| `mbpp` | MBPP (`full`) | rows -> `data/code/` |
| `stack_v2` | The Stack v2 | **cannot be fetched**: only metadata is hosted; file contents are in Software Heritage's bucket. Put your own `.py` files in `data/code/` |
| `openhermes` | OpenHermes-2.5 | rows -> `data/instruct/` |
| `openorca` | OpenOrca | rows -> `data/instruct/` |
| `aya` | Aya dataset | rows -> `data/instruct/` |
| `alpaca`, `alpaca_cleaned` | tatsu-lab/alpaca, yahma/alpaca-cleaned | rows -> `data/instruct/` |
| `pippa` | PygmalionAI/PIPPA | rows -> `data/persona/` (add the `dataset.json` above) |
| `hh_rlhf` | Anthropic HH-RLHF | preference pairs -> `data/preference/` |
| `ultrafeedback` | UltraFeedback (binarized) | preference pairs |
| `nectar` | Nectar | preference pairs |
| `feedback_collection` | Prometheus Feedback-Collection | critique/judge data -> `data/critique/` |
| `shepherd` | Shepherd | id came from a third-party list; may need `--repo` |
| `toolbench`, `bfcl` | ToolBench, Gorilla BFCL | usually distributed as archives / eval sets; see the note `fetch` prints |
| `lj_speech`, `vctk`, `common_voice` | speech corpora | **audio**: download by hand into `data/audio/` ([VOICE.md](VOICE.md)) |

> **Honest caveat:** every catalogue entry is marked `"verified": false`. The build environment had no access to huggingface.co, so ids,
> configs and splits were written from documentation and could not be checked against the live hub. If a fetch fails, the message shows the real HTTP
> error; correct the entry in your own catalogue file (`fetch.catalog`, same layout as `zerowhite/chat/corpus/catalog.json`) or pass `--repo` / `--subset` / `--split`.
> Downloading by hand and dropping the files in the folder always works.

## Converting datasets

```bash
zerowhite chat convert data/raw/hermes.json out/hermes.sharegpt.jsonl --to sharegpt
zerowhite chat convert out/hermes.sharegpt.jsonl out/hermes.alpaca.csv --to alpaca
zerowhite chat convert big.jsonl.gz clean.jsonl --to chatml --limit 100000
```

Reads anything the folder accepts; writes `.jsonl`, `.json`, `.csv` or `.txt` (optionally `.gz`) in `normalized`, `chatml`, `openai`, `sharegpt`,
`alpaca`, `plain`, `csv`, `preference` or `text` shape (records that cannot be expressed in the target shape are counted as skipped, never mangled). Filters apply unless `--no-filters`. It refuses to overwrite an existing output unless you pass `--overwrite`.
Audio conversion is `chat voice convert` ([VOICE.md](VOICE.md)); model conversion is `chat export` ([CHAT.md](CHAT.md)).
