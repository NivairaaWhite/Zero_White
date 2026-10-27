"""tools.py - tool forging: Zero White writes, runs, keeps and CHAINS its own small utility scripts (a parser, a converter, a little solver), in plain Python, with no library.

`program_policy` evolves a scalar decision from numbers. A *tool* is a different thing: a self-contained script `run(x) -> y` over text, lists, tables and numbers that does one job
(split a log line, turn CSV into JSON, pull links out of HTML, sum the numbers in a sentence). The wishlist asked for a sandbox where the system can write, execute and save such scripts and
then use them as sub-routines of bigger jobs. This module is that, in four parts:

  1. SYNTHESIS    given input -> output EXAMPLES (a "tool request"), a typed bottom-up enumerative search over ~250 primitives (string / list / table / dict / number operations built only on
                  `re`, `json`, `csv`, `io`, `math`, `statistics`, `collections`) finds the SMALLEST program that reproduces every training example. Candidates with the same behaviour on the
                  examples are merged (observational equivalence), which is what keeps the search small. A share of the examples is held out and never searched on: a program that only
                  memorised the training cases is rejected and the search goes on.
  2. SCRIPT       the program is written as a standalone Python file (`run` + a stdin/stdout JSON entry point) that imports nothing but the standard modules above. `check_source` is the gate every
                  script has to pass before it is stored or run: only allow-listed imports, no `open` / `exec` / `eval` / `__import__` / `getattr`..., no dunder attribute, no `global`.
  3. SANDBOX      the script is then EXECUTED in the sandbox (a subprocess with a wall-clock limit, a CPU limit, a memory limit and no network where the OS allows it: selfmod/sandbox.py) on
                  every example, training and held-out. A tool is only stored when the subprocess reproduces the expected outputs.
  4. TOOLBOX      `<workspace>/forge/tools/<name>/{tool.py, tool.json}`: the source, the examples, a hash, how often it was used and how often it failed. Stored tools are themselves PRIMITIVES
                  of the next synthesis, so "split lines, then parse each, then total" is found as a chain of tools (the composite's script carries its parts' source: still one file, still no
                  library) - the library-learning step of DreamCoder, in the smallest form that works here. `Toolbox.chain(names, x)` runs tools one after the other.

HONEST LIMITS. This is program synthesis from examples over a fixed, hand-written DSL: it finds what the DSL can express in roughly 10 operations, on examples the person (or another module) wrote.
It is not a code generator that understands a prose description, and it cannot invent a primitive the DSL lacks (a tokenizer for a new file format needs a new primitive, written by a person).
Examples under-determine a program: a held-out check catches the worst over-fitting, not all of it (it is reported as `validated`; with fewer than 5 examples nothing is held out and the result says so).
In-process execution (`Toolbox.run`) is limited to scripts this module produced (it re-derives the hash first) and to a restricted builtins table; scripts that merely PASSED `check_source`
(supplied by a person, or proposed by the chat model) only ever run in the sandbox subprocess. A sandbox is containment for accidents, not a security boundary (see selfmod/sandbox.py).
"""
from __future__ import annotations

import ast
import collections
import csv
import hashlib
import io
import itertools
import json
import math
import os
import re
import statistics
import time
from dataclasses import dataclass, field

from .. import fileio

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
ALLOWED_IMPORTS = frozenset({"collections", "csv", "io", "itertools", "json", "math", "re", "statistics", "string"})
_MODULES = {"collections": collections, "csv": csv, "io": io, "itertools": itertools, "json": json, "math": math, "re": re, "statistics": statistics}
_FORBIDDEN_NAMES = frozenset({"open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr", "globals", "locals", "vars", "input", "breakpoint", "memoryview",
                              "exit", "quit", "help", "dir", "type", "object", "super", "classmethod", "staticmethod", "property", "__builtins__", "__builtin__"})
MAX_SOURCE_CHARS = 20000
MAX_VALUE_LEN = 20000


# ======================================================================================================================================= the script gate
def check_source(src: str):
    """(ok, reason). The rules every stored or executed tool script must satisfy. Purely syntactic: it keeps accidents and obvious abuse out, it is not a proof of safety."""
    if not isinstance(src, str) or not src.strip():
        return False, "empty source"
    if len(src) > MAX_SOURCE_CHARS:
        return False, f"source is longer than {MAX_SOURCE_CHARS} characters"
    try:
        tree = ast.parse(src)
    except SyntaxError as ex:
        return False, f"syntax error: {ex.msg} (line {ex.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "sys":                                                       # only for the entry point: `sys.stdin` / `sys.stdout`, checked below
                    continue
                if a.name.split(".")[0] not in ALLOWED_IMPORTS or "." in a.name:
                    return False, f"import of '{a.name}' is not allowed (allowed: {', '.join(sorted(ALLOWED_IMPORTS))})"
        elif isinstance(node, ast.ImportFrom):
            return False, "'from ... import' is not allowed: write `import module` and use `module.name`"
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            return False, "global / nonlocal are not allowed"
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "sys" and node.attr not in ("stdin", "stdout"):
            return False, f"'sys.{node.attr}' is not allowed (only sys.stdin and sys.stdout)"
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in ("system", "popen", "remove", "unlink", "rmtree", "environ", "call", "run_path")):
            return False, f"attribute '{node.attr}' is not allowed"
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                return False, f"the name '{node.id}' is not allowed"
            if node.id.startswith("__") and node.id not in ("__name__",):
                return False, f"the name '{node.id}' is not allowed"
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith, ast.Yield, ast.YieldFrom, ast.ClassDef)):
            return False, f"{type(node).__name__} is not allowed in a tool"
    has_run = any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body)
    if not has_run:
        return False, "the script must define a top-level `def run(x)`"
    return True, ""


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):               # noqa: A002
    if name.split(".")[0] not in ALLOWED_IMPORTS or level:
        raise ImportError(f"import of '{name}' is not allowed in a tool")
    return __import__(name, globals, locals, fromlist, level)


_SAFE_BUILTINS = {n: getattr(__import__("builtins"), n) for n in (
    "abs", "all", "any", "bool", "chr", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset", "int", "isinstance", "iter", "len", "list", "map", "max", "min", "next", "ord",
    "pow", "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip", "print", "ValueError", "TypeError", "KeyError", "IndexError", "Exception",
    "ZeroDivisionError", "StopIteration")}
_SAFE_BUILTINS["__import__"] = _safe_import


def load_function(src: str):
    """The `run` function of a script that passed `check_source`, compiled with a restricted builtins table. Used ONLY for scripts this module itself generated (see Toolbox.run)."""
    ok, why = check_source(src)
    if not ok:
        raise ValueError(f"script refused: {why}")
    code = compile(src, "<tool>", "exec")
    ns = {"__builtins__": dict(_SAFE_BUILTINS), "__name__": "zw_tool"}
    exec(code, ns)                                                                          # noqa: S102 - checked above; generated by this module; restricted builtins
    return ns["run"]


# ======================================================================================================================================= types and values
def type_of(v) -> str:
    """The DSL type of a JSON-like value. str num bool dict strs nums rows dicts."""
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "num"
    if isinstance(v, str):
        return "str"
    if isinstance(v, dict):
        return "dict"
    if isinstance(v, list):
        if not v or all(isinstance(e, str) for e in v):
            return "strs"
        if all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in v):
            return "nums"
        if all(isinstance(e, list) and all(isinstance(c, str) for c in e) for e in v):
            return "rows"
        if all(isinstance(e, dict) for e in v):
            return "dicts"
    return "any"


def _pick_type(values) -> str:
    """The type of a column of examples: the first non-empty list decides for lists (an empty list is every list type)."""
    for v in values:
        if not (isinstance(v, list) and not v):
            return type_of(v)
    return "strs"


def _equal(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a != a or b != b:
            return a != a and b != b
        return abs(a - b) <= 1e-9 * max(1.0, abs(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a.keys()) == list(b.keys()) and all(_equal(a[k], b[k]) for k in a)
    return type(a) == type(b) and a == b


def _size_ok(v, budget=MAX_VALUE_LEN) -> bool:
    try:
        return len(repr(v)) <= budget
    except Exception:                                                                      # noqa: BLE001
        return False


# ======================================================================================================================================= primitives
_NS = dict(_MODULES)
_NUM = r"-?\d+(?:\.\d+)?"
PATTERNS = (r"\d+", _NUM, r"[A-Za-z]+", r"\w+", r"https?://[^\s\"'<>)]+", r"<(\w+)", r"href=[\"']([^\"']*)[\"']", r"[A-Z][a-z]+", r"\S+@\S+\.\w+", r"#\w+", r"[A-Z]+", r"[a-z]+",
            r"[^\s,;]+", r">([^<]+)<", r"\b\w{4,}\b", r"[aeiouAEIOU]")


@dataclass(frozen=True)
class Prim:
    name: str
    args: tuple
    ret: str
    code: str                                                  # a Python EXPRESSION over {0}, {1}: the single source of truth for both execution and the emitted script
    fn: object = field(default=None, compare=False, hash=False, repr=False)

    @property
    def arity(self) -> int:
        return len(self.args)


_HOLE = re.compile(r"\{(\d)\}")


def _fill(code: str, subs) -> str:
    """Put the argument expressions into a template in ONE pass (an argument that itself contains '{0}' text is not substituted twice)."""
    return _HOLE.sub(lambda m: subs[int(m.group(1))], code)


def _mk(name, args, ret, code) -> Prim:
    params = ", ".join(f"a{i}" for i in range(len(args)))
    body = _fill(code, [f"a{i}" for i in range(len(args))])
    g = dict(_NS)
    g["__builtins__"] = dict(_SAFE_BUILTINS)
    fn = eval(f"lambda {params}: {body}", g)                                                # noqa: S307 - templates are this file's own text; constants go through repr()
    return Prim(name, tuple(args), ret, code, fn)


_FIXED = (
    # str -> str
    ("strip", ("str",), "str", "{0}.strip()"), ("lstrip", ("str",), "str", "{0}.lstrip()"), ("rstrip", ("str",), "str", "{0}.rstrip()"),
    ("lower", ("str",), "str", "{0}.lower()"), ("upper", ("str",), "str", "{0}.upper()"), ("title", ("str",), "str", "{0}.title()"), ("capitalize", ("str",), "str", "{0}.capitalize()"),
    ("swapcase", ("str",), "str", "{0}.swapcase()"), ("reverse_str", ("str",), "str", "{0}[::-1]"), ("squeeze", ("str",), "str", "' '.join({0}.split())"),
    ("strip_tags", ("str",), "str", "re.sub(r'<[^>]*>', '', {0})"), ("only_digits", ("str",), "str", "re.sub(r'\\D', '', {0})"), ("no_digits", ("str",), "str", "re.sub(r'\\d', '', {0})"),
    ("no_vowels", ("str",), "str", "re.sub(r'[aeiouAEIOU]', '', {0})"), ("strip_quotes", ("str",), "str", "{0}.strip('\"\\'')"), ("sort_chars", ("str",), "str", "''.join(sorted({0}))"),
    ("dedupe_chars", ("str",), "str", "''.join(dict.fromkeys({0}))"), ("acronym", ("str",), "str", "''.join(w[0] for w in {0}.split()).upper()"),
    ("first_line", ("str",), "str", "{0}.splitlines()[0]"), ("last_line", ("str",), "str", "{0}.splitlines()[-1]"),
    ("json_min", ("str",), "str", "json.dumps(json.loads({0}), separators=(',', ':'))"), ("json_pretty", ("str",), "str", "json.dumps(json.loads({0}), indent=2)"),
    ("collapse_nl", ("str",), "str", "re.sub(r'\\n+', '\\n', {0})"), ("snake", ("str",), "str", "re.sub(r'[^a-z0-9]+', '_', {0}.lower()).strip('_')"),
    ("camel_split", ("str",), "str", "re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', {0})"),
    # str -> list / number / table
    ("lines", ("str",), "strs", "{0}.splitlines()"), ("words", ("str",), "strs", "{0}.split()"), ("chars", ("str",), "strs", "list({0})"),
    ("sentences", ("str",), "strs", "[s for s in re.split(r'(?<=[.!?])\\s+', {0}.strip()) if s]"),
    ("csv_row", ("str",), "strs", "next(csv.reader(io.StringIO({0})))"), ("csv_rows", ("str",), "rows", "list(csv.reader(io.StringIO({0})))"),
    ("tsv_rows", ("str",), "rows", "list(csv.reader(io.StringIO({0}), delimiter='\\t'))"), ("nums_in", ("str",), "nums", "[int(v) if re.fullmatch(r'-?\\d+', v) else float(v) for v in re.findall(r'-?\\d+(?:\\.\\d+)?', {0})]"),
    ("md_rows", ("str",), "rows", "[[c.strip() for c in ln.strip().strip('|').split('|')] for ln in {0}.splitlines() if ln.strip().startswith('|') and not re.fullmatch(r'[|\\s:\\-]+', ln.strip())]"),
    ("length", ("str",), "num", "len({0})"), ("to_int", ("str",), "num", "int({0}.strip())"), ("to_float", ("str",), "num", "float({0}.strip())"),
    ("n_words", ("str",), "num", "len({0}.split())"), ("n_lines", ("str",), "num", "len({0}.splitlines())"), ("n_vowels", ("str",), "num", "sum(ch in 'aeiouAEIOU' for ch in {0})"),
    ("json_dict", ("str",), "dict", "(lambda v: v if isinstance(v, dict) else 1 / 0)(json.loads({0}))"),
    ("json_dicts", ("str",), "dicts", "(lambda v: v if isinstance(v, list) and all(isinstance(e, dict) for e in v) else 1 / 0)(json.loads({0}))"),
    ("json_strs", ("str",), "strs", "(lambda v: v if isinstance(v, list) and all(isinstance(e, str) for e in v) else 1 / 0)(json.loads({0}))"),
    ("json_nums", ("str",), "nums", "(lambda v: v if isinstance(v, list) and all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in v) else 1 / 0)(json.loads({0}))"),
    ("is_empty", ("str",), "bool", "len({0}.strip()) == 0"), ("is_palindrome", ("str",), "bool", "(lambda s: s == s[::-1])(re.sub(r'[^a-z0-9]', '', {0}.lower()))"),
    ("is_digits", ("str",), "bool", "{0}.strip().isdigit()"), ("is_upper", ("str",), "bool", "{0}.isupper()"), ("is_alpha", ("str",), "bool", "{0}.isalpha()"),
    # list of strings
    ("sort_s", ("strs",), "strs", "sorted({0})"), ("rev_s", ("strs",), "strs", "{0}[::-1]"), ("uniq_s", ("strs",), "strs", "list(dict.fromkeys({0}))"),
    ("non_empty", ("strs",), "strs", "[e for e in {0} if e.strip()]"), ("map_strip", ("strs",), "strs", "[e.strip() for e in {0}]"), ("map_lower", ("strs",), "strs", "[e.lower() for e in {0}]"),
    ("map_upper", ("strs",), "strs", "[e.upper() for e in {0}]"), ("map_title", ("strs",), "strs", "[e.title() for e in {0}]"), ("map_rev", ("strs",), "strs", "[e[::-1] for e in {0}]"),
    ("map_strip_quotes", ("strs",), "strs", "[e.strip('\"\\'') for e in {0}]"), ("sort_by_len", ("strs",), "strs", "sorted({0}, key=len)"), ("drop_first", ("strs",), "strs", "{0}[1:]"),
    ("drop_last", ("strs",), "strs", "{0}[:-1]"), ("map_len", ("strs",), "nums", "[len(e) for e in {0}]"), ("map_int", ("strs",), "nums", "[int(e.strip()) for e in {0}]"),
    ("map_float", ("strs",), "nums", "[float(e.strip()) for e in {0}]"), ("longest", ("strs",), "str", "max({0}, key=len)"), ("shortest", ("strs",), "str", "min({0}, key=len)"),
    ("most_common", ("strs",), "str", "collections.Counter({0}).most_common(1)[0][0]"), ("count_s", ("strs",), "num", "len({0})"), ("n_unique_s", ("strs",), "num", "len(set({0}))"),
    ("strs_json", ("strs",), "str", "json.dumps({0}, separators=(',', ':'))"), ("has_dups", ("strs",), "bool", "len(set({0})) != len({0})"), ("is_sorted_s", ("strs",), "bool", "{0} == sorted({0})"),
    ("counts", ("strs",), "dict", "dict(collections.Counter({0}))"), ("strs_rows", ("strs",), "rows", "[[e] for e in {0}]"),
    # list of numbers
    ("sum_n", ("nums",), "num", "sum({0})"), ("max_n", ("nums",), "num", "max({0})"), ("min_n", ("nums",), "num", "min({0})"), ("mean_n", ("nums",), "num", "sum({0}) / len({0})"),
    ("median_n", ("nums",), "num", "statistics.median({0})"), ("count_n", ("nums",), "num", "len({0})"), ("prod_n", ("nums",), "num", "math.prod({0})"), ("range_n", ("nums",), "num", "max({0}) - min({0})"),
    ("argmax_n", ("nums",), "num", "{0}.index(max({0}))"), ("argmin_n", ("nums",), "num", "{0}.index(min({0}))"), ("sort_n", ("nums",), "nums", "sorted({0})"), ("rev_n", ("nums",), "nums", "{0}[::-1]"),
    ("sort_desc", ("nums",), "nums", "sorted({0}, reverse=True)"), ("cumsum", ("nums",), "nums", "list(itertools.accumulate({0}))"), ("diffs", ("nums",), "nums", "[b - a for a, b in zip({0}, {0}[1:])]"),
    ("map_sq", ("nums",), "nums", "[v * v for v in {0}]"), ("map_abs", ("nums",), "nums", "[abs(v) for v in {0}]"), ("evens", ("nums",), "nums", "[v for v in {0} if v % 2 == 0]"),
    ("odds", ("nums",), "nums", "[v for v in {0} if v % 2 == 1]"), ("positives", ("nums",), "nums", "[v for v in {0} if v > 0]"), ("uniq_n", ("nums",), "nums", "list(dict.fromkeys({0}))"),
    ("map_str", ("nums",), "strs", "[str(int(v)) if v == int(v) else str(v) for v in {0}]"), ("nums_json", ("nums",), "str", "json.dumps({0}, separators=(',', ':'))"),
    ("is_sorted_n", ("nums",), "bool", "{0} == sorted({0})"),
    # one number
    ("neg", ("num",), "num", "-{0}"), ("abs_n", ("num",), "num", "abs({0})"), ("sq", ("num",), "num", "{0} * {0}"), ("sqrt", ("num",), "num", "math.sqrt({0})"), ("round0", ("num",), "num", "int(round({0}))"),
    ("floor", ("num",), "num", "math.floor({0})"), ("ceil", ("num",), "num", "math.ceil({0})"), ("fact", ("num",), "num", "math.factorial(int({0})) if 0 <= {0} <= 170 else 1 / 0"), ("half", ("num",), "num", "{0} / 2"),
    ("inc", ("num",), "num", "{0} + 1"), ("dec", ("num",), "num", "{0} - 1"), ("to_str", ("num",), "str", "str(int({0})) if {0} == int({0}) else str({0})"), ("fmt2", ("num",), "str", "f'{{0}:.2f}'"),
    ("fmt1", ("num",), "str", "f'{{0}:.1f}'"), ("is_even", ("num",), "bool", "{0} % 2 == 0"), ("is_pos", ("num",), "bool", "{0} > 0"),
    ("is_prime", ("num",), "bool", "{0} > 1 and all({0} % d for d in range(2, int(math.isqrt(int({0}))) + 1))"), ("digits_sum", ("num",), "num", "sum(int(c) for c in str(abs(int({0}))))"),
    ("num_digits", ("num",), "strs", "list(str(abs(int({0}))))"), ("range_to", ("num",), "nums", "list(range(1, int({0}) + 1)) if 0 <= {0} <= 1000 else 1 / 0"),
    # tables / dicts
    ("rows_dicts", ("rows",), "dicts", "[dict(zip({0}[0], r)) for r in {0}[1:]]"), ("transpose", ("rows",), "rows", "[list(c) for c in zip(*{0})]"), ("drop_first_row", ("rows",), "rows", "{0}[1:]"),
    ("rows_csv", ("rows",), "str", "(lambda b: (lambda w: ([w.writerow(r) for r in {0}], b.getvalue().rstrip('\\n'))[1])(csv.writer(b, lineterminator='\\n')))(io.StringIO())"),
    ("rows_tsv", ("rows",), "str", "'\\n'.join('\\t'.join(r) for r in {0})"), ("rows_json", ("rows",), "str", "json.dumps({0}, separators=(',', ':'))"), ("n_rows", ("rows",), "num", "len({0})"),
    ("rows_md", ("rows",), "str", "'\\n'.join(['| ' + ' | '.join({0}[0]) + ' |', '|' + '|'.join(' --- ' for _ in {0}[0]) + '|'] + ['| ' + ' | '.join(r) + ' |' for r in {0}[1:]])"),
    ("flatten", ("rows",), "strs", "[c for r in {0} for c in r]"),
    ("keys", ("dict",), "strs", "list({0}.keys())"), ("values_s", ("dict",), "strs", "[str(v) for v in {0}.values()]"), ("items_rows", ("dict",), "rows", "[[str(k), str(v)] for k, v in {0}.items()]"),
    ("dict_json", ("dict",), "str", "json.dumps({0}, separators=(',', ':'))"), ("dict_pretty", ("dict",), "str", "json.dumps({0}, indent=2)"), ("sort_keys", ("dict",), "dict", "dict(sorted({0}.items()))"),
    ("n_keys", ("dict",), "num", "len({0})"), ("dict_sum", ("dict",), "num", "sum(v for v in {0}.values() if isinstance(v, (int, float)))"),
    ("dicts_json", ("dicts",), "str", "json.dumps({0}, separators=(',', ':'))"), ("n_dicts", ("dicts",), "num", "len({0})"),
    ("dicts_rows", ("dicts",), "rows", "[list({0}[0].keys())] + [[str(d.get(k, '')) for k in {0}[0]] for d in {0}]"),
)

_BINARY = (
    ("add", ("num", "num"), "num", "{0} + {1}"), ("sub", ("num", "num"), "num", "{0} - {1}"), ("mul", ("num", "num"), "num", "{0} * {1}"), ("div", ("num", "num"), "num", "{0} / {1}"),
    ("fdiv", ("num", "num"), "num", "{0} // {1}"), ("mod", ("num", "num"), "num", "{0} % {1}"), ("pow", ("num", "num"), "num", "{0} ** {1} if abs({1}) <= 64 and abs({0}) <= 10 ** 9 else 1 / 0"), ("gt", ("num", "num"), "bool", "{0} > {1}"),
    ("eq_n", ("num", "num"), "bool", "{0} == {1}"), ("concat", ("str", "str"), "str", "{0} + {1}"), ("eq_s", ("str", "str"), "bool", "{0} == {1}"),
    ("gcd", ("num", "num"), "num", "math.gcd(int({0}), int({1}))"),
)

_SEPARATORS = (",", ";", "|", "\t", "\n", " ", ":", "=", "-", "_", "/", ".", ", ", " ", "&", "#")
_QUOTE_FREE = ("", " ", "_", "-", "\n")


def _flatten_strings(v, out: list, depth=0):
    if depth > 4:
        return
    if isinstance(v, str):
        out.append(v)
    elif isinstance(v, dict):
        for k, e in v.items():
            out.append(str(k))
            _flatten_strings(e, out, depth + 1)
    elif isinstance(v, (list, tuple)):
        for e in v[:50]:
            _flatten_strings(e, out, depth + 1)


@dataclass
class Consts:
    seps: tuple
    joiners: tuple
    str_consts: tuple
    nums: tuple
    keys: tuple
    idx: tuple
    pairs: tuple

    @classmethod
    def from_examples(cls, ins, outs):
        in_s, out_s = [], []
        for v in ins:
            _flatten_strings(v, in_s)
        for v in outs:
            _flatten_strings(v, out_s)
        blob_in, blob_out = "\x00".join(in_s), "\x00".join(out_s)
        seps = tuple(dict.fromkeys(s for s in _SEPARATORS if s in blob_in))
        joiners = tuple(dict.fromkeys(["", *[s for s in _SEPARATORS if s in blob_out or s in blob_in]]))
        consts = []
        out_strs = [o for o in outs if isinstance(o, str)]
        if out_strs:
            pre = os.path.commonprefix(out_strs)
            suf = os.path.commonprefix([o[::-1] for o in out_strs])[::-1]
            for c in (pre, suf):
                if 0 < len(c) <= 40 and any(o != c for o in out_strs):
                    consts.append(c)
            if len(set(out_strs)) <= 3 and all(len(o) <= 40 for o in out_strs):
                consts.extend(sorted(set(out_strs)))
            consts = [c for c in consts if c]
        consts = [c for c in consts if "{" not in c and "}" not in c]
        consts.extend(s for s in seps if s not in consts)
        keys = []
        for v in list(ins) + list(outs):
            vv = v[0] if isinstance(v, list) and v and isinstance(v[0], dict) else v
            if isinstance(vv, dict):
                keys.extend(str(k) for k in vv.keys())
        keys = tuple(dict.fromkeys(k for k in keys if len(k) <= 40 and "{" not in k and "}" not in k))[:12]
        nums = [0, 1, 2, 3, 4, 5, 7, 10, 100, 0.5]
        for o in outs:
            if isinstance(o, (int, float)) and not isinstance(o, bool) and float(o).is_integer() and 0 < abs(o) <= 1000 and o not in nums:
                nums.append(int(o))
        pairs = ()
        if len(outs) >= 2 and all(isinstance(o, str) for o in outs):
            distinct = sorted(set(outs))
            if len(distinct) == 2 and all(len(d) <= 40 and "{" not in d and "}" not in d for d in distinct):
                pairs = ((distinct[0], distinct[1]), (distinct[1], distinct[0]))
        return cls(seps, joiners, tuple(dict.fromkeys(consts))[:10], tuple(nums[:14]), keys, (0, 1, 2, 3, -1), pairs)


def build_prims(c: Consts, extra=()):
    ps = [_mk(*p) for p in _FIXED] + [_mk(*p) for p in _BINARY]
    r = repr
    for sep in c.seps:
        ps.append(_mk(f"split{r(sep)}", ("str",), "strs", f"{{0}}.split({r(sep)})"))
        ps.append(_mk(f"split_rows{r(sep)}", ("strs",), "rows", f"[e.split({r(sep)}) for e in {{0}}]"))
        for sep2 in c.seps:
            if sep2 != sep:
                ps.append(_mk(f"kvdict{r(sep)}{r(sep2)}", ("str",), "dict", "{k.strip(): v.strip() for k, v in [it.split(" + r(sep2) + ", 1) for it in {0}.split(" + r(sep) + ") if " + r(sep2) + " in it]}"))
    for j in c.joiners:
        ps.append(_mk(f"join{r(j)}", ("strs",), "str", f"{r(j)}.join({{0}})"))
        ps.append(_mk(f"join_nums{r(j)}", ("nums",), "str", f"{r(j)}.join(str(int(v)) if v == int(v) else str(v) for v in {{0}})"))
        ps.append(_mk(f"join_rows{r(j)}", ("rows",), "str", f"'\\n'.join({r(j)}.join(row) for row in {{0}})"))
        for j2 in c.joiners:
            if j2 and j != j2:
                ps.append(_mk(f"dictkv{r(j)}{r(j2)}", ("dict",), "str", f"{r(j)}.join(str(k) + {r(j2)} + str(v) for k, v in {{0}}.items())"))
    for a in c.seps:
        for b in dict.fromkeys(list(c.joiners) + list(_QUOTE_FREE)):
            if a != b:
                ps.append(_mk(f"replace{r(a)}{r(b)}", ("str",), "str", f"{{0}}.replace({r(a)}, {r(b)})"))
        ps.append(_mk(f"count{r(a)}", ("str",), "num", f"{{0}}.count({r(a)})"))
        ps.append(_mk(f"contains{r(a)}", ("str",), "bool", f"{r(a)} in {{0}}"))
        ps.append(_mk(f"startswith{r(a)}", ("str",), "bool", f"{{0}}.startswith({r(a)})"))
    for i, pat in enumerate(PATTERNS):
        ps.append(_mk(f"findall{i}", ("str",), "strs", f"re.findall({r(pat)}, {{0}})"))
        ps.append(_mk(f"count_re{i}", ("str",), "num", f"len(re.findall({r(pat)}, {{0}}))"))
        for rep in ("", " "):
            ps.append(_mk(f"resub{i}{r(rep)}", ("str",), "str", f"re.sub({r(pat)}, {r(rep)}, {{0}})"))
    for k in (1, 2, 3, 4):
        ps.append(_mk(f"head{k}", ("str",), "str", f"{{0}}[:{k}]"))
        ps.append(_mk(f"tail{k}", ("str",), "str", f"{{0}}[-{k}:]"))
        ps.append(_mk(f"skip{k}", ("str",), "str", f"{{0}}[{k}:]"))
        ps.append(_mk(f"take{k}", ("strs",), "strs", f"{{0}}[:{k}]"))
        ps.append(_mk(f"take{k}_n", ("nums",), "nums", f"{{0}}[:{k}]"))
        ps.append(_mk(f"top{k}_n", ("nums",), "nums", f"sorted({{0}}, reverse=True)[:{k}]"))
    for k in (2, 3):
        ps.append(_mk(f"repeat{k}", ("str",), "str", f"{{0}} * {k}"))
    for i in c.idx:
        ps.append(_mk(f"at{i}", ("strs",), "str", f"{{0}}[{i}]"))
        ps.append(_mk(f"at{i}_n", ("nums",), "num", f"{{0}}[{i}]"))
        ps.append(_mk(f"col{i}", ("rows",), "strs", f"[r[{i}] for r in {{0}}]"))
        ps.append(_mk(f"row{i}", ("rows",), "strs", f"{{0}}[{i}]"))
        ps.append(_mk(f"dict_at{i}", ("dicts",), "dict", f"{{0}}[{i}]"))
    for key in c.keys:
        ps.append(_mk(f"get_s{r(key)}", ("dict",), "str", f"{{0}}[{r(key)}] if isinstance({{0}}[{r(key)}], str) else 1 / 0"))
        ps.append(_mk(f"get_n{r(key)}", ("dict",), "num", f"{{0}}[{r(key)}] if isinstance({{0}}[{r(key)}], (int, float)) and not isinstance({{0}}[{r(key)}], bool) else 1 / 0"))
        ps.append(_mk(f"col_s{r(key)}", ("dicts",), "strs", f"[d[{r(key)}] for d in {{0}}] if all(isinstance(d[{r(key)}], str) for d in {{0}}) else 1 / 0"))
        ps.append(_mk(f"col_n{r(key)}", ("dicts",), "nums", f"[d[{r(key)}] for d in {{0}}] if all(isinstance(d[{r(key)}], (int, float)) for d in {{0}}) else 1 / 0"))
        ps.append(_mk(f"pick{r(key)}", ("dicts",), "dicts", "[{" + r(key) + ": d[" + r(key) + "]} for d in {0}]"))
        ps.append(_mk(f"sort_by{r(key)}", ("dicts",), "dicts", f"sorted({{0}}, key=lambda d: d[{r(key)}])"))
        ps.append(_mk(f"drop{r(key)}", ("dict",), "dict", "{k: v for k, v in {0}.items() if k != " + r(key) + "}"))
    for a, b in c.pairs:
        for pname, arg in (("choose", "bool"),):
            ps.append(_mk(f"{pname}{r(a)}{r(b)}", (arg,), "str", f"{r(a)} if {{0}} else {r(b)}"))
    ps.extend(extra)
    return ps


# ======================================================================================================================================= enumeration
class Expr:
    __slots__ = ("prim", "args", "vals", "size", "type", "const", "is_var")

    def __init__(self, prim, args, vals, size, type_, const=None, is_var=False):
        self.prim, self.args, self.vals, self.size, self.type, self.const, self.is_var = prim, args, vals, size, type_, const, is_var

    def show(self) -> str:
        if self.is_var:
            return "x"
        if self.prim is None:
            return repr(self.const)
        return f"{self.prim.name}({', '.join(a.show() for a in self.args)})"


def _eval_expr(e: Expr, x, memo=None):
    if e.is_var:
        return x
    if e.prim is None:
        return e.const
    memo = {} if memo is None else memo
    if id(e) in memo:
        return memo[id(e)]
    v = e.prim.fn(*[_eval_expr(a, x, memo) for a in e.args])
    memo[id(e)] = v
    return v


def _apply(prim, args, n):
    vals = []
    for i in range(n):
        try:
            v = prim.fn(*[a.vals[i] for a in args])
        except Exception:                                                                  # noqa: BLE001 - a primitive that does not apply to this input makes the expression invalid
            return None
        if not _size_ok(v):
            return None
        vals.append(v)
    return tuple(vals)


def _types_ok(vals, ret) -> bool:
    """A primitive's declared result type must be what it really returned (a `strs` primitive that returned a list of numbers is a bug, not a result)."""
    for v in vals:
        if v == [] and ret in ("strs", "nums", "rows", "dicts"):
            continue
        if type_of(v) != ret:
            return False
    return True


@dataclass
class Candidate:
    expr: Expr
    train_ok: bool
    holdout_ok: bool | None


def synthesize(examples, holdout=(), tools=(), max_size=12, max_exprs=150000, seconds=20.0, want_holdout=True, now=time.monotonic):
    """Bottom-up search. examples / holdout: lists of (input, output). Returns (best_expr or None, info dict).

    The first program (smallest by node count) that matches every TRAINING example AND every held-out example wins; programs that match only the training examples are counted in
    `overfit` and the search goes on."""
    ins = [e[0] for e in examples]
    outs = [e[1] for e in examples]
    n_train = len(ins)
    all_ins = ins + [h[0] for h in holdout]                                              # every expression is evaluated on the training AND the held-out inputs, so two programs
    all_outs = outs + [h[1] for h in holdout]                                             # that differ only on a held-out input are NOT merged as equivalent
    n = len(all_ins)
    in_t, out_t = _pick_type(ins), _pick_type(outs)
    if any(type_of(v) != in_t and v != [] for v in ins):
        return None, dict(reason="the inputs of the examples do not all have the same shape", explored=0)
    if any(type_of(v) != out_t and v != [] for v in outs):
        return None, dict(reason="the outputs of the examples do not all have the same shape", explored=0)
    if in_t == "any" or out_t == "any":
        return None, dict(reason="inputs and outputs must be text, numbers, booleans, lists of those, tables (lists of lists of text), dicts or lists of dicts", explored=0)
    consts = Consts.from_examples(all_ins, outs)
    prims = list(tools) + build_prims(consts)                                            # stored tools first: when a tool and a built-in do the same job, the tool (reuse) wins the tie
    t0 = now()
    pools: dict = {}
    seen: set = set()
    explored = 0
    overfit = 0
    solutions = []
    xvar = Expr(None, (), tuple(all_ins), 1, in_t, is_var=True)

    def add(e: Expr, size: int):
        pools.setdefault((size, e.type), []).append(e)

    def key(e):
        return (e.type, repr(e.vals))
    seen.add(key(xvar))
    add(xvar, 1)
    for c in consts.str_consts:
        e = Expr(None, (), (c,) * n, 1, "str", const=c)
        if key(e) not in seen:
            seen.add(key(e))
            add(e, 1)
    for v in consts.nums:
        e = Expr(None, (), (v,) * n, 1, "num", const=v)
        if key(e) not in seen:
            seen.add(key(e))
            add(e, 1)
    def check(e: Expr):
        nonlocal overfit
        if e.type != out_t:
            return None
        if not all(_equal(a, b) for a, b in zip(e.vals[:n_train], all_outs[:n_train])):
            return None
        if not all(_equal(a, b) for a, b in zip(e.vals[n_train:], all_outs[n_train:])):
            overfit += 1                                                                   # fits every training example, fails a held-out one: memorised, not learned
            return None
        return e
    hit = check(xvar)
    if hit is not None:
        return hit, dict(reason="", explored=1, overfit=0, consts=consts)
    for size in range(2, max_size + 1):
        for p in prims:
            if now() - t0 > seconds or explored > max_exprs:
                return None, dict(reason=f"the search budget ran out at size {size} ({explored} programs tried, {now() - t0:.1f}s): give more examples or a larger budget", explored=explored, overfit=overfit)
            if p.arity == 1:
                combos = ((a,) for a in pools.get((size - 1, p.args[0]), ()))
            else:
                combos = ((a, b) for i in range(1, size - 1) for a in pools.get((i, p.args[0]), ()) for b in pools.get((size - 1 - i, p.args[1]), ()))
            for args in combos:
                explored += 1
                if explored & 1023 == 0 and (now() - t0 > seconds or explored > max_exprs):
                    return None, dict(reason=f"the search budget ran out at size {size} ({explored} programs tried, {now() - t0:.1f}s): give more examples or a larger budget", explored=explored, overfit=overfit)
                if all(a.prim is None and not a.is_var for a in args):                       # a primitive of constants only: a constant, never useful
                    continue
                vals = _apply(p, args, n)
                if vals is None or not _types_ok(vals, p.ret):
                    continue
                e = Expr(p, args, vals, size, p.ret)
                k = key(e)
                if k in seen:
                    continue
                seen.add(k)
                add(e, size)
                hit = check(e)
                if hit is not None:
                    return hit, dict(reason="", explored=explored, overfit=overfit, consts=consts)
        if explored > max_exprs or now() - t0 > seconds:
            return None, dict(reason=f"the search budget ran out at size {size} ({explored} programs tried): give more examples or a larger budget", explored=explored, overfit=overfit)
    why = f"no program of up to {max_size} operations over this vocabulary reproduces the examples ({explored} distinct behaviours tried)"
    if overfit:
        why += f"; {overfit} program(s) fitted the training examples but failed the held-out ones"
    return None, dict(reason=why, explored=explored, overfit=overfit)


# ======================================================================================================================================= script emission
def _topo(e: Expr, order: list, seen: set):
    if id(e) in seen or e.is_var or e.prim is None:
        return
    seen.add(id(e))
    for a in e.args:
        _topo(a, order, seen)
    order.append(e)


def _used_tools(e: Expr, out: dict):
    if e.prim is not None and e.prim.name.startswith("tool:"):
        out[e.prim.name[5:]] = e.prim
    for a in e.args:
        _used_tools(a, out)


def emit_script(expr: Expr, name: str, description: str = "", examples=(), dep_sources: dict | None = None) -> str:
    """The standalone script for an expression. `dep_sources` maps a used tool's name to its `def _t_<name>` source (bundled so the file stands alone)."""
    order: list = []
    _topo(expr, order, set())
    names = {}
    lines = []
    for i, e in enumerate(order):
        var = f"t{i + 1}"
        names[id(e)] = var
        subs = ["x" if a.is_var else (repr(a.const) if a.prim is None else names[id(a)]) for a in e.args]
        lines.append(f"    {var} = {_fill(e.prim.code, subs)}")
    ret = "x" if expr.is_var else (repr(expr.const) if expr.prim is None else names[id(expr)])
    body = "\n".join(lines + [f"    return {ret}"])
    deps = "".join((dep_sources or {}).values())
    mods = {m for m in ("collections", "csv", "io", "itertools", "json", "math", "re", "statistics") if re.search(rf"\b{m}\.", deps + body)} | {"json"}
    desc = re.sub(r"\s+", " ", (description or "a tool forged by Zero White from examples").replace('"', "'"))[:200]
    header = (f'"""tool {name}: {desc}\n\nForged by Zero White (zerowhite/forge/tools.py) from {len(examples)} example(s); standard library only.\n'
              f'Usage: echo \'<json input>\' | python tool.py   ->   json output\n"""\n')
    main = '\n\nif __name__ == "__main__":\n    import sys\n    print(json.dumps(run(json.loads(sys.stdin.read()))))\n'
    return header + "import " + ", ".join(sorted(mods)) + "\n\n\n" + (deps + "\n\n" if deps else "") + "def run(x):\n" + body + "\n" + main


def _dep_function(src: str, name: str) -> str:
    """The source of a stored tool's functions, with `run` renamed `_t_<name>`, ready to be pasted above a composite tool's `run`. Only 'simple' tools (imports + functions, no classes) can be bundled."""
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "run":
                node.name = f"_t_{name}"
            keep.append(node)
        elif isinstance(node, (ast.Import, ast.Expr)):
            continue
        elif isinstance(node, ast.If):
            continue
        else:
            raise ValueError("this tool cannot be bundled into another (it has top-level code besides imports and functions)")
    for node in ast.walk(ast.Module(body=keep, type_ignores=[])):
        if isinstance(node, ast.Name) and node.id == "run":
            node.id = f"_t_{name}"
    return "\n\n".join(ast.unparse(k) for k in keep) + "\n\n\n"


# ======================================================================================================================================= the sandbox
_HARNESS = '''import json, sys
import tool
cases = json.load(open("cases.json", encoding="utf-8"))
out = []
for x in cases:
    try:
        out.append({"ok": True, "v": tool.run(x)})
    except BaseException as e:
        out.append({"ok": False, "err": type(e).__name__ + ": " + str(e)[:120]})
print(json.dumps(out))
'''


def run_in_sandbox(src: str, inputs, timeout: float = 30.0):
    """Run a script's `run` on every input in a SANDBOX subprocess. Returns (ok, results|reason): results = [{'ok': bool, 'v': value | 'err': text}]."""
    import tempfile
    from ..selfmod.sandbox import run_python
    ok, why = check_source(src)
    if not ok:
        return False, f"script refused: {why}"
    with tempfile.TemporaryDirectory(prefix="zw_tool_") as d:
        fileio.write_text(os.path.join(d, "tool.py"), src)
        fileio.write_text(os.path.join(d, "cases.json"), json.dumps(list(inputs)))
        fileio.write_text(os.path.join(d, "harness.py"), _HARNESS)
        r = run_python(["harness.py"], cwd=d, timeout=timeout, mem_mb=1024, fsize_mb=4)
    if not r.ok:
        tail = (r.stderr.strip().splitlines() or ["no output"])[-1][:200]
        return False, f"the sandbox run failed ({'timeout' if r.timed_out else 'rc=' + str(r.returncode)}): {tail}"
    try:
        return True, json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, "the sandbox run printed no result"


# ======================================================================================================================================= the toolbox
@dataclass
class ForgeResult:
    ok: bool
    name: str
    reason: str = ""
    source: str = ""
    expr: str = ""
    in_type: str = ""
    out_type: str = ""
    train_n: int = 0
    holdout_n: int = 0
    validated: bool = False             # True: held-out examples existed and the tool reproduced them
    sandbox_ok: bool = False
    explored: int = 0
    seconds: float = 0.0
    uses: list = field(default_factory=list)

    def to_dict(self):
        return dict(self.__dict__)


class Toolbox:
    """`<workspace>/forge/tools/<name>/{tool.py, tool.json}`. Everything stored here passed `check_source` and was executed in the sandbox before it was kept."""

    def __init__(self, ws_root: str):
        self.root = os.path.join(os.path.abspath(ws_root), "forge", "tools")

    # ---- inventory ---------------------------------------------------------------------------------------------------
    def dir(self, name: str) -> str:
        if not NAME_RE.match(name or ""):
            raise ValueError("a tool name is lowercase letters, digits and underscores, starting with a letter (at most 32 characters)")
        return os.path.join(self.root, name)

    def names(self) -> list:
        if not os.path.isdir(self.root):
            return []
        return sorted(d for d in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, d, "tool.json")) and NAME_RE.match(d))

    def spec(self, name: str) -> dict:
        p = os.path.join(self.dir(name), "tool.json")
        if not os.path.exists(p):
            raise KeyError(f"no tool called '{name}'")
        return fileio.read_json(p)

    def source(self, name: str) -> str:
        return fileio.read_text(os.path.join(self.dir(name), "tool.py"))

    def list(self) -> list:
        out = []
        for n in self.names():
            try:
                s = self.spec(n)
            except (ValueError, OSError):
                continue
            out.append(dict(name=n, description=s.get("description", ""), in_type=s.get("in_type"), out_type=s.get("out_type"), uses=s.get("uses", 0), failures=s.get("failures", 0),
                            parts=s.get("parts", []), validated=s.get("validated", False)))
        return out

    def intact(self, name: str) -> bool:
        """True when the stored source is exactly what was stored (the hash in tool.json)."""
        try:
            return hashlib.sha256(self.source(name).encode("utf-8")).hexdigest() == self.spec(name).get("sha256")
        except (KeyError, ValueError, OSError):
            return False

    def add(self, name: str, source: str, spec: dict):
        ok, why = check_source(source)
        if not ok:
            raise ValueError(f"script refused: {why}")
        d = self.dir(name)
        os.makedirs(d, exist_ok=True)
        fileio.write_text(os.path.join(d, "tool.py"), source)
        rec = dict(spec, name=name, sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(), created=time.time(), uses=spec.get("uses", 0), failures=spec.get("failures", 0))
        fileio.write_json(os.path.join(d, "tool.json"), rec, indent=2)
        return rec

    def remove(self, name: str):
        import shutil
        shutil.rmtree(self.dir(name), ignore_errors=True)

    def record(self, name: str, ok: bool):
        try:
            s = self.spec(name)
            s["uses"] = s.get("uses", 0) + 1
            if not ok:
                s["failures"] = s.get("failures", 0) + 1
            fileio.write_json(os.path.join(self.dir(name), "tool.json"), s, indent=2)
        except (KeyError, ValueError, OSError):
            pass

    # ---- running -----------------------------------------------------------------------------------------------------
    def run(self, name: str, x, sandbox: bool | None = None):
        """Run one tool on one input. Tools this module forged (origin 'forged', hash intact) run in process under restricted builtins; anything else runs in the sandbox subprocess."""
        spec = self.spec(name)
        if sandbox is not True and spec.get("origin") == "forged" and self.intact(name):
            try:
                v = load_function(self.source(name))(x)
                self.record(name, True)
                return v
            except Exception:                                                              # noqa: BLE001
                self.record(name, False)
                raise
        ok, res = run_in_sandbox(self.source(name), [x])
        if not ok:
            self.record(name, False)
            raise RuntimeError(res)
        if not res[0]["ok"]:
            self.record(name, False)
            raise RuntimeError(res[0]["err"])
        self.record(name, True)
        return res[0]["v"]

    def chain(self, names, x, sandbox: bool | None = None):
        """Run tools one after the other: the output of each is the input of the next."""
        for n in names:
            x = self.run(n, x, sandbox=sandbox)
        return x

    def tool_prims(self):
        """Stored tools as primitives for the next synthesis (the library-learning step)."""
        ps = []
        for t in self.list():
            if t["in_type"] in (None, "any") or t["out_type"] in (None, "any") or not self.intact(t["name"]):
                continue
            sp = self.spec(t["name"])
            if sp.get("origin") != "forged":
                continue
            try:
                fn = load_function(self.source(t["name"]))
            except ValueError:
                continue
            ps.append(Prim(f"tool:{t['name']}", (t["in_type"],), t["out_type"], f"_t_{t['name']}({{0}})", fn))
        return ps

    # ---- forging -----------------------------------------------------------------------------------------------------
    def forge(self, name: str, examples, description: str = "", seconds: float = 20.0, max_size: int = 12, use_tools: bool = True, replace: bool = False, sandbox: bool = True,
              seed: int = 0) -> ForgeResult:
        """Synthesize, hold out, write, sandbox-run and store a tool from [(input, output), ...] (or [{'input':..,'output':..}, ...])."""
        t0 = time.time()
        try:
            self.dir(name)
        except ValueError as ex:
            return ForgeResult(False, name, reason=str(ex))
        if name in self.names() and not replace:
            return ForgeResult(False, name, reason=f"a tool called '{name}' already exists (pass replace=True to overwrite it)")
        pairs = []
        for e in examples:
            if isinstance(e, dict) and "input" in e and "output" in e:
                pairs.append((e["input"], e["output"]))
            elif isinstance(e, (list, tuple)) and len(e) == 2:
                pairs.append((e[0], e[1]))
            else:
                return ForgeResult(False, name, reason="each example must be {'input': ..., 'output': ...} or an (input, output) pair")
        if len(pairs) < 3:
            return ForgeResult(False, name, reason="at least 3 examples are needed (more is better: examples under-determine a program)")
        for i, o in pairs:
            if not (_size_ok(i) and _size_ok(o)):
                return ForgeResult(False, name, reason="an example is too large")
        order = list(range(len(pairs)))
        import random
        random.Random(seed).shuffle(order)
        n_hold = 0 if len(pairs) < 5 else max(1, len(pairs) // 4)
        hold = [pairs[i] for i in order[:n_hold]]
        train = [pairs[i] for i in order[n_hold:]]
        tools = self.tool_prims() if use_tools else []
        expr, info = synthesize(train, hold, tools=tools, max_size=max_size, seconds=seconds)
        if expr is None:
            return ForgeResult(False, name, reason=info["reason"], explored=info.get("explored", 0), train_n=len(train), holdout_n=len(hold), seconds=time.time() - t0)
        used: dict = {}
        _used_tools(expr, used)
        try:
            deps = {k: _dep_function(self.source(k), k) for k in used}
        except (ValueError, SyntaxError, KeyError) as ex:
            return ForgeResult(False, name, reason=f"a part of this chain cannot be bundled: {ex}")
        src = emit_script(expr, name, description, pairs, deps)
        ok, why = check_source(src)
        if not ok:
            return ForgeResult(False, name, reason=f"internal error: the generated script was refused: {why}")
        sandbox_ok = False
        if sandbox:
            ok, res = run_in_sandbox(src, [p[0] for p in pairs])
            if not ok:
                return ForgeResult(False, name, reason=str(res), source=src, expr=expr.show(), explored=info.get("explored", 0), seconds=time.time() - t0)
            bad = [i for i, (r, (_, want)) in enumerate(zip(res, pairs)) if not (r["ok"] and _equal(r["v"], want))]
            if bad:
                return ForgeResult(False, name, reason=f"the script did not reproduce example {bad[0] + 1} when run in the sandbox: {res[bad[0]]}", source=src, expr=expr.show(), seconds=time.time() - t0)
            sandbox_ok = True
        in_t, out_t = type_of(pairs[0][0]), type_of(pairs[0][1])
        self.add(name, src, dict(description=description or f"forged from {len(pairs)} examples", in_type=in_t, out_type=out_t, expr=expr.show(), origin="forged", parts=sorted(used),
                                 examples=[dict(input=i, output=o) for i, o in pairs][:40], validated=bool(hold), train_n=len(train), holdout_n=len(hold), sandbox_ok=sandbox_ok))
        return ForgeResult(True, name, source=src, expr=expr.show(), in_type=in_t, out_type=out_t, train_n=len(train), holdout_n=len(hold), validated=bool(hold), sandbox_ok=sandbox_ok,
                           explored=info.get("explored", 0), seconds=time.time() - t0, uses=sorted(used))

    def register(self, name: str, source: str, examples=(), description: str = "") -> dict:
        """Store a script a PERSON (or the chat model) wrote. It must pass check_source and reproduce every example in the sandbox; it is marked origin 'supplied': it only ever runs in the sandbox."""
        ok, why = check_source(source)
        if not ok:
            raise ValueError(f"script refused: {why}")
        pairs = [(e["input"], e["output"]) if isinstance(e, dict) else tuple(e) for e in examples]
        if not pairs:
            raise ValueError("a supplied tool needs at least one example it must reproduce")
        ok, res = run_in_sandbox(source, [p[0] for p in pairs])
        if not ok:
            raise ValueError(str(res))
        bad = [i for i, (r, (_, want)) in enumerate(zip(res, pairs)) if not (r["ok"] and _equal(r["v"], want))]
        if bad:
            raise ValueError(f"the script did not reproduce example {bad[0] + 1}")
        return self.add(name, source, dict(description=description or "supplied", in_type=type_of(pairs[0][0]), out_type=type_of(pairs[0][1]), origin="supplied", parts=[],
                                           examples=[dict(input=i, output=o) for i, o in pairs][:40], validated=False, sandbox_ok=True))


# ======================================================================================================================================= requests (the organism's interface)
def requests_dir(ws_root: str) -> str:
    return os.path.join(os.path.abspath(ws_root), "tools", "requests")


def pending_requests(ws_root: str) -> list:
    d = requests_dir(ws_root)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json") and not f.endswith(".done.json") and not f.endswith(".failed.json"))


def process_requests(ws_root: str, limit: int = 3, seconds: float = 20.0) -> list:
    """Forge a tool for each request file `<ws>/tools/requests/<name>.json` = {"name", "description", "examples": [{"input", "output"}, ...]}. A request is answered once: the file is renamed
    `.done.json` (with the result) or `.failed.json` (with the reason), so a request that cannot be met is not retried forever."""
    tb = Toolbox(ws_root)
    results = []
    for path in pending_requests(ws_root)[:limit]:
        base = path[:-5]
        try:
            req = fileio.read_json(path)
            name = str(req.get("name") or os.path.basename(base))
            res = tb.forge(name, req.get("examples", []), str(req.get("description", "")), seconds=float(req.get("seconds", seconds)), replace=bool(req.get("replace", False)))
        except (ValueError, OSError) as ex:
            res = ForgeResult(False, os.path.basename(base), reason=f"unreadable request: {ex}")
        fileio.write_json(base + (".done.json" if res.ok else ".failed.json"), dict(result=res.to_dict(), request=os.path.basename(path)), indent=2)
        try:
            os.remove(path)
        except OSError:
            pass
        results.append(res)
    return results
