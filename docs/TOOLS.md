# Tool forging (6.5)

`forge/tools.py`. Zero White writes, runs, keeps and **chains** its own small utility scripts (a parser, a converter, a little solver) in plain Python, with no library.
`program_policy` evolves a *decision* from numbers; a **tool** is a self-contained script `run(x) -> y` over text, lists, tables and numbers that does one job.

```
zerowhite tools forge split_commas --examples ex.json --description "split on commas"
zerowhite tools list
zerowhite tools run split_commas --input '"red,green,blue"'
zerowhite tools chain split_lines parse_row total --input @log.txt
zerowhite tools request NAME --examples ex.json      # leave a request; `tools process` or `organism --tools` answers it
```

`ex.json` is a list of `{"input": ..., "output": ...}`.

## How a tool is made

1. **Synthesis.** A typed, bottom-up enumerative search over about 250 primitives (string / list / table / dict / number operations built only on `re`, `json`, `csv`, `io`, `math`, `statistics`, `collections`) finds the *smallest* program that reproduces every training example. Candidates that behave identically on the examples are merged (observational equivalence), which keeps the search small. A share of the examples is **held out** and never searched on: a program that only memorised the training cases is rejected and the search goes on.
2. **Script.** The program is written as a standalone Python file (`run` plus a stdin/stdout JSON entry point). `check_source` is the gate every script passes before it is stored or run: allow-listed imports only; no `open`, `exec`, `eval`, `__import__`, `getattr`, ...; no dunder attribute; no `global`.
3. **Sandbox.** The script is then *executed* in the sandbox (a subprocess with wall-clock, CPU and memory limits and no network where the OS allows it: `selfmod/sandbox.py`) on every example, training and held-out. A tool is stored only when the subprocess reproduces the expected outputs.
4. **Toolbox.** `<workspace>/forge/tools/<name>/{tool.py, tool.json}`: source, examples, hash, uses, failures. Stored tools are themselves *primitives* of the next synthesis, so "split lines, then parse each, then total" is found as a chain of tools (the composite carries its parts' source: still one file, still no library). This is the library-learning step of DreamCoder in the smallest form that works here.

## Safety

* Scripts this module produced and whose hash is intact run in process under a restricted builtins table; **anything else** (a script a person supplied or the chat model proposed) runs only in the sandbox subprocess.
* A request file is answered once: it becomes `.done.json` or `.failed.json`, so a request that cannot be met is not retried for ever.
* The organism answers requests only with `--tools`, and chooses only *when*; the examples are a person's.

## Honest limits

* This is program synthesis from examples over a fixed, hand-written DSL: it finds what the DSL can express in roughly 10 operations, on examples someone wrote. It is **not** a code generator that understands a prose description, and it cannot invent a primitive the DSL lacks (a tokenizer for a new file format needs a primitive written by a person).
* Examples under-determine a program. A held-out check catches the worst over-fitting, not all of it (`validated` says whether any examples were held out; with fewer than 5 examples nothing is).
* A sandbox is containment for accidents, not a security boundary (see `selfmod/sandbox.py`).
