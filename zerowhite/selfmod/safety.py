"""safety.py - the static gate every piece of self-written code must pass.

Zero White evolves and rewrites Python source (hook functions, agent programs,
patches). Before such code is ever compiled it goes through `check_source`.
This module is deliberately standalone (no zerowhite imports) and is part of
the *constitution*: the self-modification machinery is not allowed to patch it.

What it enforces (the default "pure" profile used for evolved functions):
  * no imports, classes, async, global/nonlocal, with/try/raise/delete
  * no loops or comprehensions (so evolved code always terminates)
  * no attribute names starting with '_' (blocks the usual __class__ escapes)
  * no dangerous builtins (eval, exec, open, getattr, __import__ ...)
  * `np.` / `math.` may only call a whitelist of pure numerical functions
  * a node-count cap so programs cannot bloat without bound

HONEST LIMIT: this is defence in depth against evolved-code accidents, not a
security boundary against a determined attacker. For hostile code use a real
sandbox (nsjail, gVisor, a container). See docs/SELF_MODIFICATION.md.
"""

import ast

FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "input", "breakpoint", "exit", "quit", "memoryview", "type", "object",
    "super", "classmethod", "staticmethod", "property", "help", "dir", "id", "print", "callable",
    "isinstance", "issubclass", "hasattr", "bytearray", "bytes", "frozenset", "set", "dict", "list",
    "tuple", "str", "format", "repr", "ascii", "hash", "iter", "next", "slice",
})
FORBIDDEN_ATTRS = frozenset({
    "tofile", "dump", "dumps", "load", "loads", "save", "savez", "fromfile", "memmap", "ctypes",
    "system", "popen", "remove", "unlink", "rmdir", "mkdir", "write", "read", "readlines",
    "environ", "path", "modules", "stdin", "stdout", "stderr", "buffer", "mro", "bit_generator",
    "state",
})
NP_ALLOWED = frozenset({
    "abs", "absolute", "all", "any", "arange", "arccos", "arcsin", "arctan", "arctan2", "argmax",
    "argmin", "argsort", "array", "asarray", "bincount", "ceil", "clip", "concatenate", "cos",
    "cosh", "cumsum", "diff", "dot", "e", "exp", "expm1", "float64", "floor", "full", "hypot",
    "inf", "isfinite", "isnan", "linspace", "log", "log10", "log1p", "log2", "maximum", "mean",
    "median", "minimum", "nan", "nan_to_num", "ndim", "ones", "percentile", "pi", "power",
    "quantile", "round", "searchsorted", "shape", "sign", "sin", "sinh", "size", "sort", "sqrt",
    "square", "stack", "std", "sum", "take", "tan", "tanh", "unique", "var", "where", "zeros",
    "zeros_like", "ones_like", "prod", "min", "max", "ptp", "argpartition", "exp2", "cbrt",
})
MATH_ALLOWED = frozenset({
    "acos", "asin", "atan", "atan2", "ceil", "copysign", "cos", "cosh", "e", "erf", "exp",
    "expm1", "fabs", "floor", "hypot", "inf", "isfinite", "isnan", "log", "log10", "log1p",
    "log2", "pi", "pow", "sin", "sinh", "sqrt", "tan", "tanh",
})
_BANNED_NODES = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef, ast.Await,
                 ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Raise, ast.Delete,
                 ast.Yield, ast.YieldFrom, ast.AsyncFor, ast.Try, ast.Starred)
_LOOP_NODES = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


class UnsafeSource(ValueError):
    pass


def violations(src, *, allow_loops=False, max_nodes=1500, extra_names=(), module_aliases=("np", "math")):
    """Return a list of human-readable problems (empty list = acceptable)."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e.msg} (line {e.lineno})"]
    problems = []
    n_nodes = 0
    allowed_extra = set(extra_names)
    docstrings = set()                       # documentation is inert; it gets its own, larger cap
    for scope in ast.walk(tree):
        if isinstance(scope, (ast.Module, ast.FunctionDef)) and scope.body:
            first = scope.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    for node in ast.walk(tree):
        n_nodes += 1
        if isinstance(node, _BANNED_NODES):
            problems.append(f"forbidden construct: {type(node).__name__}")
        if not allow_loops and isinstance(node, _LOOP_NODES):
            problems.append(f"loops/comprehensions are not allowed ({type(node).__name__})")
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES and node.id not in allowed_extra:
                problems.append(f"forbidden name: {node.id}")
            if node.id.startswith("__"):
                problems.append(f"forbidden dunder name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                problems.append(f"forbidden attribute: {node.attr}")
            if node.attr in FORBIDDEN_ATTRS:
                problems.append(f"forbidden attribute: {node.attr}")
            if isinstance(node.value, ast.Name) and node.value.id in module_aliases:
                allowed = NP_ALLOWED if node.value.id == "np" else MATH_ALLOWED
                if node.attr not in allowed:
                    problems.append(f"{node.value.id}.{node.attr} is not on the allow-list")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            limit = 4000 if id(node) in docstrings else 200
            if len(node.value) > limit:
                problems.append("very long string literal")
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, int) and abs(node.value) > 10 ** 9:
                problems.append("integer literal too large")
    if n_nodes > max_nodes:
        problems.append(f"program too large ({n_nodes} nodes > {max_nodes})")
    # exactly the functions, nothing executable at module level
    for stmt in tree.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.Expr, ast.Assign)):
            problems.append(f"unexpected module-level statement: {type(stmt).__name__}")
        if isinstance(stmt, ast.Expr) and not (isinstance(stmt.value, ast.Constant)
                                               and isinstance(stmt.value.value, str)):
            problems.append("only docstrings may be bare module-level expressions")
    return sorted(set(problems))


def check_source(src, **kw):
    """Raise UnsafeSource if `src` breaks any rule; return the parsed tree otherwise."""
    bad = violations(src, **kw)
    if bad:
        raise UnsafeSource("; ".join(bad[:6]))
    return ast.parse(src)


def is_safe(src, **kw):
    return not violations(src, **kw)


def function_names(src):
    return [n.name for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)]
