"""astmut.py - grammar-guided mutation of Python source (no language model involved).

FunSearch asks an LLM to rewrite a program; Zero White does the same job with the
operators evolutionary computation has used for decades (DEAP-style subtree
crossover/mutation, AutoML-Zero-style insert/delete/alter-parameter, PySR-style
constant tuning), applied to Python's own AST so the result is always real,
readable source code.

A *program* is one function in canonical form:

    def name(args):
        t0 = <expr>
        t1 = <expr>
        return <expr>

Expressions are built from a `Vocab`: input leaves (x[:, i] / obs[i] / named
arguments), constants, unary and binary functions, and conditionals. Every
mutation is followed by `repair`, so offspring never reference undefined names,
and by the safety gate (safety.py), so they can never import, open files, loop
forever, or reach dunder attributes.

6.5: three STRUCTURAL mutators (`gate`: a branch, `neuron`: a new hidden unit wired into the output, `mgate`: a multiplicative sigmoid gate) change the TOPOLOGY of a program,
not just its operators and constants. They are off unless a Vocab asks for them (`structural=p`: the chance that a mutation step uses one) or a caller names one
(`apply_mutator`), so every earlier search behaves exactly as before.
"""

import ast
import copy
import math

import numpy as np

from .safety import violations

# --------------------------------------------------------------------------- helpers visible to programs


def pdiv(a, b):
    b = np.asarray(b, dtype=float)
    return np.divide(a, np.where(np.abs(b) < 1e-6, 1e-6, b))


def psqrt(a):
    return np.sqrt(np.abs(a))


def plog(a):
    return np.log(np.abs(a) + 1e-6)


def pexp(a):
    return np.exp(np.clip(a, -20.0, 20.0))


def sel(c, a, b):
    return np.where(c, a, b)


def relu(a):
    return np.maximum(a, 0.0)


def sig(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a, -30.0, 30.0)))


def sgn(a):
    return np.sign(a)


HELPERS = {"pdiv": pdiv, "psqrt": psqrt, "plog": plog, "pexp": pexp, "sel": sel, "relu": relu,
           "sig": sig, "sgn": sgn}
_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
             for n in ("abs", "min", "max", "sum", "len", "range", "int", "float", "bool",
                       "round", "sorted", "enumerate", "zip", "all", "any", "pow", "divmod")}
_ALWAYS_OK_NAMES = frozenset(HELPERS) | frozenset(_BUILTINS) | {"np", "math", "True", "False", "None"}


def program_namespace():
    ns = {"np": np, "math": math, "__builtins__": dict(_BUILTINS)}
    ns.update(HELPERS)
    return ns


def compile_program(src, fn_name, **safety_kw):
    """Gate, compile and return the function `fn_name` defined in `src` (raises ValueError if unsafe)."""
    bad = violations(src, extra_names=(), **safety_kw)
    if bad:
        raise ValueError("; ".join(bad[:4]))
    ns = program_namespace()
    exec(compile(src, f"<program {fn_name}>", "exec"), ns)       # noqa: S102 - gated just above
    fn = ns.get(fn_name)
    if not callable(fn):
        raise ValueError(f"no function '{fn_name}' in program")
    return fn


# --------------------------------------------------------------------------- vocabulary
DEFAULT_UNARY = ("np.sin({})", "np.cos({})", "np.tanh({})", "np.abs({})", "psqrt({})", "plog({})",
                 "pexp({})", "relu({})", "sig({})", "np.square({})", "(-{})")
DEFAULT_BINARY = ("({a} + {b})", "({a} - {b})", "({a} * {b})", "pdiv({a}, {b})",
                  "np.maximum({a}, {b})", "np.minimum({a}, {b})")
DEFAULT_CONSTS = (-2.0, -1.0, -0.5, 0.1, 0.5, 1.0, 2.0, 3.0)


class Vocab:
    """Everything the generator is allowed to write. One Vocab per kind of program."""

    def __init__(self, fn_name, args, *, col_var=None, n_cols=0, vector=True, leaves=(), cond_leaves=(),
                 unary=DEFAULT_UNARY, binary=DEFAULT_BINARY, consts=DEFAULT_CONSTS, allow_pow=False,
                 conditionals=True, max_nodes=160, allow_loops=False, structural=0.0):
        self.structural = float(structural)                    # 6.5: probability that a mutation step is a structural one (gate / neuron / mgate); 0 = never (the default)
        self.fn_name, self.args = fn_name, tuple(args)
        self.col_var, self.n_cols, self.vector = col_var, n_cols, vector
        self.leaves, self.cond_leaves = tuple(leaves), tuple(cond_leaves)
        self.unary, self.binary, self.consts = tuple(unary), tuple(binary), tuple(consts)
        self.allow_pow, self.conditionals = allow_pow, conditionals
        self.max_nodes, self.allow_loops = max_nodes, allow_loops

    @property
    def safety(self):
        return dict(allow_loops=self.allow_loops, max_nodes=self.max_nodes * 2)


class _FoldNeg(ast.NodeTransformer):
    """`-0.7` parses as USub(0.7); fold it to the constant -0.7 so tuning sees (and keeps) the sign."""

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) \
                and isinstance(node.operand.value, (int, float)) and not isinstance(node.operand.value, bool):
            return ast.copy_location(ast.Constant(-node.operand.value), node)
        return node


def _parse(src):
    return ast.fix_missing_locations(_FoldNeg().visit(ast.parse(src)))


def _parse_expr(src):
    return _FoldNeg().visit(ast.parse(src, mode="eval")).body


def _const_src(rng, vocab):
    r = rng.random()
    if r < 0.5:
        return repr(float(rng.choice(vocab.consts)))
    return repr(round(float(rng.normal(0.0, 1.5)), 3))


def _leaf_src(rng, vocab, scope):
    r = rng.random()
    pools = []
    if vocab.col_var and vocab.n_cols:
        pools.append(("col", 0.55))
    if vocab.leaves:
        pools.append(("leaf", 0.45))
    if scope:
        pools.append(("temp", 0.30))
    pools.append(("const", 0.30 if len(pools) else 1.0))
    total = sum(w for _, w in pools)
    r *= total
    for kind, w in pools:
        if r < w:
            break
        r -= w
    if kind == "col":
        i = int(rng.integers(vocab.n_cols))
        return f"{vocab.col_var}[:, {i}]" if vocab.vector else f"{vocab.col_var}[{i}]"
    if kind == "leaf":
        return str(vocab.leaves[int(rng.integers(len(vocab.leaves)))])
    if kind == "temp":
        return scope[int(rng.integers(len(scope)))]
    return _const_src(rng, vocab)


def rand_expr_src(rng, vocab, scope=(), depth=2):
    """A random expression (as source text) over the vocabulary."""
    if depth <= 0 or rng.random() < 0.28:
        return _leaf_src(rng, vocab, scope)
    r = rng.random()
    sub = lambda: rand_expr_src(rng, vocab, scope, depth - 1)           # noqa: E731
    if r < 0.50:
        t = vocab.binary[int(rng.integers(len(vocab.binary)))]
        return t.format(a=sub(), b=sub())
    if r < 0.78 or not (vocab.conditionals or vocab.allow_pow):
        t = vocab.unary[int(rng.integers(len(vocab.unary)))]
        return t.format(sub())
    if vocab.allow_pow and (not vocab.conditionals or rng.random() < 0.4):
        return f"({sub()} ** {float(rng.choice([2.0, 3.0, 0.5, 1.5]))})"
    if vocab.cond_leaves and rng.random() < 0.6:
        c = vocab.cond_leaves[int(rng.integers(len(vocab.cond_leaves)))]
    else:
        c = f"{sub()} > {sub()}"
    if vocab.vector:
        return f"sel({c}, {sub()}, {sub()})"
    return f"({sub()} if {c} else {sub()})"


# --------------------------------------------------------------------------- AST plumbing
class Slot:
    __slots__ = ("parent", "field", "idx", "node")

    def __init__(self, parent, field, idx, node):
        self.parent, self.field, self.idx, self.node = parent, field, idx, node

    def put(self, new):
        if self.idx is None:
            setattr(self.parent, self.field, new)
        else:
            getattr(self.parent, self.field)[self.idx] = new
        self.node = new


_LEAF_CALLS = {"len"}


def _is_leaf_like(node):
    if isinstance(node, (ast.Constant, ast.Name, ast.Subscript)):
        return True
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _LEAF_CALLS:
            return True
        if isinstance(node.func, ast.Attribute) and not node.args:
            return True                                    # rng.random()
    return False


def _slots(fn):
    out = []

    def visit(node, parent, field, idx):
        if isinstance(node, ast.expr):
            out.append(Slot(parent, field, idx, node))
        if _is_leaf_like(node):
            return
        for f, v in ast.iter_fields(node):
            if isinstance(node, ast.Call) and f in ("func", "keywords"):
                continue
            if isinstance(v, list):
                for i, c in enumerate(v):
                    if isinstance(c, ast.expr):
                        visit(c, node, f, i)
            elif isinstance(v, ast.expr):
                visit(v, node, f, None)

    for st in fn.body:
        if isinstance(st, (ast.Assign, ast.Return)) and st.value is not None:
            visit(st.value, st, "value", None)
    return out


def _fn(tree):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef))


def _assigned(fn):
    return [(i, st.targets[0].id) for i, st in enumerate(fn.body)
            if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name)]


def count_nodes(src):
    try:
        return sum(1 for _ in ast.walk(ast.parse(src)))
    except SyntaxError:
        return 10 ** 6


def normalize(src):
    """Canonical text of a program (used to recognise duplicates)."""
    return ast.unparse(_parse(src))


class _Repair(ast.NodeTransformer):
    def __init__(self, defined, rng, vocab):
        self.defined, self.rng, self.vocab = set(defined), rng, vocab

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id not in self.defined and node.id not in _ALWAYS_OK_NAMES:
            return _parse_expr(_leaf_src(self.rng, self.vocab, sorted(self.defined - set(self.vocab.args))))
        return node


def repair(tree, rng, vocab):
    """Make every name used by the function refer to something defined earlier."""
    fn = _fn(tree)
    defined = set(vocab.args) | set(fn.args.args and [a.arg for a in fn.args.args])
    for st in fn.body:
        if isinstance(st, (ast.Assign, ast.Return)) and st.value is not None:
            st.value = _Repair(defined, rng, vocab).visit(st.value)
            if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name):
                defined.add(st.targets[0].id)
    ast.fix_missing_locations(tree)
    return tree


# --------------------------------------------------------------------------- mutation operators
def _numeric_slots(slots):
    return [s for s in slots if isinstance(s.node, ast.Constant) and isinstance(s.node.value, (int, float))
            and not isinstance(s.node.value, bool)]


def _m_const(tree, rng, vocab, slots):
    cs = _numeric_slots(slots)
    if not cs:
        return False
    s = cs[int(rng.integers(len(cs)))]
    v = float(s.node.value)
    r = rng.random()
    if r < 0.5:
        nv = v * math.exp(rng.normal(0, 0.35)) + rng.normal(0, 0.05)
    elif r < 0.8:
        nv = v + rng.normal(0, 0.5)
    elif r < 0.9:
        nv = -v
    else:
        nv = float(rng.choice(vocab.consts))
    s.put(ast.Constant(round(nv, 6) if isinstance(s.node.value, float) or abs(nv - round(nv)) > 1e-9
                       else int(round(nv))))
    return True


_SWAP_BIN = {ast.Add: (ast.Sub, ast.Mult), ast.Sub: (ast.Add, ast.Mult), ast.Mult: (ast.Add, ast.Sub)}
_SWAP_CMP = {ast.Gt: (ast.Lt, ast.GtE), ast.Lt: (ast.Gt, ast.LtE), ast.GtE: (ast.Gt, ast.LtE),
             ast.LtE: (ast.Lt, ast.GtE)}


def _m_op(tree, rng, vocab, slots):
    cand = [s for s in slots if isinstance(s.node, (ast.BinOp, ast.Compare, ast.Call))]
    if not cand:
        return False
    s = cand[int(rng.integers(len(cand)))]
    n = s.node
    if isinstance(n, ast.BinOp) and type(n.op) in _SWAP_BIN:
        n.op = _SWAP_BIN[type(n.op)][int(rng.integers(2))]()
        return True
    if isinstance(n, ast.Compare) and type(n.ops[0]) in _SWAP_CMP:
        n.ops[0] = _SWAP_CMP[type(n.ops[0])][int(rng.integers(2))]()
        return True
    if isinstance(n, ast.Call) and len(n.args) == 1 and vocab.unary:       # swap one unary function for another
        fname = ast.unparse(n.func)
        if fname in ("int", "float", "len", "abs", "round"):
            return False
        t = vocab.unary[int(rng.integers(len(vocab.unary)))]
        s.put(_parse_expr(t.format(ast.unparse(n.args[0]))))
        return True
    if isinstance(n, ast.Call) and len(n.args) == 2 and vocab.binary:
        t = vocab.binary[int(rng.integers(len(vocab.binary)))]
        s.put(_parse_expr(t.format(a=ast.unparse(n.args[0]), b=ast.unparse(n.args[1]))))
        return True
    return False


def _m_reindex(tree, rng, vocab, slots):
    cand = [s for s in slots if isinstance(s.node, ast.Subscript) and vocab.n_cols > 1]
    if not cand:
        return False
    s = cand[int(rng.integers(len(cand)))]
    i = int(rng.integers(vocab.n_cols))
    s.put(_parse_expr(f"{vocab.col_var}[:, {i}]" if vocab.vector else f"{vocab.col_var}[{i}]"))
    return True


def _scope_at(fn, upto):
    return [st.targets[0].id for st in fn.body[:upto]
            if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name)]


def _m_replace(tree, rng, vocab, slots):
    if not slots:
        return False
    s = slots[int(rng.integers(len(slots)))]
    fn = _fn(tree)
    s.put(_parse_expr(rand_expr_src(rng, vocab, _scope_at(fn, len(fn.body)), depth=int(rng.integers(1, 3)))))
    return True


def _m_wrap(tree, rng, vocab, slots):
    if not slots:
        return False
    s = slots[int(rng.integers(len(slots)))]
    inner = ast.unparse(s.node)
    fn = _fn(tree)
    scope = _scope_at(fn, len(fn.body))
    if rng.random() < 0.5:
        t = vocab.unary[int(rng.integers(len(vocab.unary)))]
        new = t.format(inner)
    else:
        t = vocab.binary[int(rng.integers(len(vocab.binary)))]
        other = rand_expr_src(rng, vocab, scope, depth=1)
        new = t.format(a=inner, b=other) if rng.random() < 0.5 else t.format(a=other, b=inner)
    s.put(_parse_expr(new))
    return True


def _m_hoist(tree, rng, vocab, slots):
    cand = [s for s in slots if isinstance(s.node, (ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Compare))
            or (isinstance(s.node, ast.Call) and s.node.args)]
    if not cand:
        return False
    s = cand[int(rng.integers(len(cand)))]
    n = s.node
    if isinstance(n, ast.BinOp):
        kids = [n.left, n.right]
    elif isinstance(n, ast.UnaryOp):
        kids = [n.operand]
    elif isinstance(n, ast.IfExp):
        kids = [n.body, n.orelse]
    elif isinstance(n, ast.Compare):
        kids = [n.left] + list(n.comparators)
    else:
        kids = [a for a in n.args if isinstance(a, ast.expr)]
    if not kids:
        return False
    s.put(copy.deepcopy(kids[int(rng.integers(len(kids)))]))
    return True


def _m_insert(tree, rng, vocab, slots):
    fn = _fn(tree)
    names = _assigned(fn)
    if len(names) >= 6:
        return False
    k = len(names)
    while f"t{k}" in {n for _, n in names} | set(vocab.args):
        k += 1
    pos = len(fn.body) - 1                                   # just before the final return
    expr = _parse_expr(rand_expr_src(rng, vocab, _scope_at(fn, pos), depth=2))
    fn.body.insert(pos, ast.Assign(targets=[ast.Name(f"t{k}", ast.Store())], value=expr, lineno=0))
    ret = fn.body[-1]
    if isinstance(ret, ast.Return) and rng.random() < 0.8:       # wire the new value into the output
        t = vocab.binary[int(rng.integers(len(vocab.binary)))]
        ret.value = _parse_expr(t.format(a=ast.unparse(ret.value), b=f"t{k}"))
    return True


def _m_delete(tree, rng, vocab, slots):
    fn = _fn(tree)
    names = _assigned(fn)
    if not names:
        return False
    i, name = names[int(rng.integers(len(names)))]
    del fn.body[i]
    return True            # repair() swaps dangling uses for fresh leaves


MUTATORS = ((_m_const, 0.26), (_m_op, 0.18), (_m_replace, 0.18), (_m_wrap, 0.12), (_m_hoist, 0.10),
            (_m_insert, 0.07), (_m_delete, 0.04), (_m_reindex, 0.05))


# --------------------------------------------------------------------------- structural mutators (6.5)
_ACT_TEMPLATES = ("np.tanh({})", "relu({})", "sig({})")


def _m_gate(tree, rng, vocab, slots):
    """A branch: replace e by `sel(c, e, alt)` / `(e if c else alt)` where c compares a leaf with a constant, and `alt` is a small new expression. Either branch may keep the old value."""
    if not slots or not vocab.conditionals:
        return False
    s = slots[int(rng.integers(len(slots)))]
    fn = _fn(tree)
    scope = _scope_at(fn, len(fn.body))
    inner = ast.unparse(s.node)
    cond = f"{_leaf_src(rng, vocab, scope)} {'>' if rng.random() < 0.5 else '<'} {_const_src(rng, vocab)}"
    alt = rand_expr_src(rng, vocab, scope, depth=1)
    a, b = (inner, alt) if rng.random() < 0.5 else (alt, inner)
    s.put(_parse_expr(f"sel({cond}, {a}, {b})" if vocab.vector else f"({a} if {cond} else {b})"))
    return True


def _m_neuron(tree, rng, vocab, slots):
    """A new hidden unit: t_k = act(w1 * a + w2 * b + c) over two leaves, added to the output with a small weight. The weights are ordinary constants, so constant tuning learns them."""
    fn = _fn(tree)
    names = _assigned(fn)
    if len(names) >= 6 or not isinstance(fn.body[-1], ast.Return):
        return False
    k = len(names)
    while f"t{k}" in {n for _, n in names} | set(vocab.args):
        k += 1
    scope = _scope_at(fn, len(fn.body) - 1)
    w = lambda sd: round(float(rng.normal(0.0, sd)), 3)                                   # noqa: E731
    pre = f"{w(0.8)} * {_leaf_src(rng, vocab, scope)} + {w(0.8)} * {_leaf_src(rng, vocab, scope)} + {w(0.5)}"
    act = _ACT_TEMPLATES[int(rng.integers(len(_ACT_TEMPLATES)))]
    fn.body.insert(len(fn.body) - 1, ast.Assign(targets=[ast.Name(f"t{k}", ast.Store())], value=_parse_expr(act.format(pre)), lineno=0))
    ret = fn.body[-1]
    ret.value = _parse_expr(f"({ast.unparse(ret.value)} + {w(0.4) or 0.1} * t{k})")
    return True


def _m_mgate(tree, rng, vocab, slots):
    """A multiplicative gate: replace e by `e * sig(w * leaf + b)` (a GLU-style block: the leaf decides how much of e passes)."""
    if not slots:
        return False
    s = slots[int(rng.integers(len(slots)))]
    fn = _fn(tree)
    scope = _scope_at(fn, len(fn.body))
    gate = f"sig({round(float(rng.normal(0.0, 1.0)), 3)} * {_leaf_src(rng, vocab, scope)} + {round(float(rng.normal(0.0, 0.5)), 3)})"
    s.put(_parse_expr(f"({ast.unparse(s.node)} * {gate})"))
    return True


STRUCTURAL = ((_m_gate, 0.4), (_m_neuron, 0.4), (_m_mgate, 0.2))
MUTATOR_KINDS = {"const": _m_const, "op": _m_op, "replace": _m_replace, "wrap": _m_wrap, "hoist": _m_hoist, "insert": _m_insert, "delete": _m_delete, "reindex": _m_reindex,
                 "gate": _m_gate, "neuron": _m_neuron, "mgate": _m_mgate}


def mutate(src, vocab, rng, n_mut=1, tries=12):
    """Return a mutated, repaired, safety-checked variant of `src`, or None if nothing valid was found."""
    base = normalize(src)
    weights = np.array([w for _, w in MUTATORS])
    weights = weights / weights.sum()
    p_struct = getattr(vocab, "structural", 0.0)
    for _ in range(tries):
        tree = _parse(src)
        changed = False
        for _m in range(n_mut):
            if p_struct > 0.0 and rng.random() < p_struct:                    # 6.5: only draws a number when the vocabulary asks for structural moves
                op = STRUCTURAL[int(rng.choice(len(STRUCTURAL), p=[w for _, w in STRUCTURAL]))][0]
            else:
                op = MUTATORS[int(rng.choice(len(MUTATORS), p=weights))][0]
            changed |= bool(op(tree, rng, vocab, _slots(_fn(tree))))
        if not changed:
            continue
        repair(tree, rng, vocab)
        try:
            out = ast.unparse(tree)
        except Exception:                                    # noqa: BLE001 - malformed tree, try again
            continue
        if out == base or violations(out, **vocab.safety):
            continue
        if count_nodes(out) > vocab.max_nodes:
            continue
        return out
    return None


def apply_mutator(src, vocab, kind, rng, tries=12):
    """Apply ONE named mutator (a key of MUTATOR_KINDS) to `src`, then repair and gate it like `mutate` does. Returns the new source, or None if that mutator found nothing to change in this
    program. The name is the choice a person - or the architect (forge/architect.py) - makes; the draw of WHERE it applies stays with `rng`."""
    if kind not in MUTATOR_KINDS:
        raise KeyError(f"unknown mutator '{kind}' (have: {sorted(MUTATOR_KINDS)})")
    base = normalize(src)
    for _ in range(tries):
        tree = _parse(src)
        if not MUTATOR_KINDS[kind](tree, rng, vocab, _slots(_fn(tree))):
            continue
        repair(tree, rng, vocab)
        try:
            out = ast.unparse(tree)
        except Exception:                                    # noqa: BLE001 - malformed tree, try again
            continue
        if out == base or violations(out, **vocab.safety) or count_nodes(out) > vocab.max_nodes:
            continue
        return out
    return None


def crossover(src_a, src_b, vocab, rng, tries=8):
    """Subtree crossover: graft a random subexpression of B into a random slot of A."""
    base = normalize(src_a)
    for _ in range(tries):
        ta, tb = _parse(src_a), _parse(src_b)
        sa, sb = _slots(_fn(ta)), _slots(_fn(tb))
        if not sa or not sb:
            return None
        a, b = sa[int(rng.integers(len(sa)))], sb[int(rng.integers(len(sb)))]
        a.put(copy.deepcopy(b.node))
        repair(ta, rng, vocab)
        out = ast.unparse(ta)
        if out != base and not violations(out, **vocab.safety) and count_nodes(out) <= vocab.max_nodes:
            return out
    return None


def seed_program(vocab, body_src):
    """Build the (trivial) starting program: `def name(args): return <body_src>`."""
    return f"def {vocab.fn_name}({', '.join(vocab.args)}):\n    return {body_src}\n"


# --------------------------------------------------------------------------- constant tuning (PySR-style)
def constants_of(src):
    tree = _parse(src)
    return [s.node.value for s in _numeric_slots(_slots(_fn(tree)))]


def with_constants(src, values):
    tree = _parse(src)
    cs = _numeric_slots(_slots(_fn(tree)))
    for s, v in zip(cs, values):
        s.put(ast.Constant(float(v) if isinstance(s.node.value, float) else (int(round(v)) if abs(v - round(v)) < 1e-9 else float(v))))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def tune_constants(src, score_fn, rng, steps=24, init_scale=0.3):
    """(1+1)-ES on the numeric constants of `src`. score_fn(src) -> float (higher is better, may raise).
    Returns (best_src, best_score, n_evaluations)."""
    vals = np.array(constants_of(src), dtype=float)
    try:
        best = float(score_fn(src))
    except Exception:                                        # noqa: BLE001
        return src, -np.inf, 1
    if len(vals) == 0:
        return src, best, 1
    scale, evals, best_src = init_scale, 1, src
    for _ in range(steps):
        cand = vals.copy()
        k = int(rng.integers(len(vals)))
        if rng.random() < 0.6:
            cand[k] += rng.normal(0, scale) * (abs(cand[k]) + 0.1)
        else:
            cand += rng.normal(0, scale * 0.3, size=len(cand)) * (np.abs(cand) + 0.1)
        s2 = with_constants(best_src, cand)
        evals += 1
        try:
            sc = float(score_fn(s2))
        except Exception:                                    # noqa: BLE001
            continue
        if sc > best + 1e-12:
            best, vals, best_src = sc, cand, s2
            scale = min(1.0, scale * 1.4)
        else:
            scale = max(0.02, scale * 0.85)
    return best_src, best, evals


# --------------------------------------------------------------------------- readability: simplify
class _Fold(ast.NodeTransformer):
    """Evaluate subtrees whose leaves are all numeric constants (np.abs(3.0) -> 3.0, 2.0 * 0.5 -> 1.0)."""

    def _try(self, node):
        if not all(isinstance(n, (ast.Constant, ast.BinOp, ast.UnaryOp, ast.Call, ast.Attribute, ast.Name, ast.Load,
                                  ast.operator, ast.unaryop)) for n in ast.walk(node)):
            return node
        consts = [n for n in ast.walk(node) if isinstance(n, ast.Constant)]
        names = [n for n in ast.walk(node) if isinstance(n, ast.Name) and n.id not in ("np", "math")]
        if not consts or names:
            return node
        try:
            v = eval(compile(ast.Expression(node), "<fold>", "eval"), program_namespace())      # noqa: S307 - gated source
            v = float(v) if not isinstance(v, bool) else v
        except Exception:                                                                       # noqa: BLE001
            return node
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return node
        return ast.copy_location(ast.Constant(round(v, 9) if isinstance(v, float) else v), node)

    def visit_BinOp(self, node):
        return self._try(self.generic_visit(node))

    def visit_UnaryOp(self, node):
        return self._try(self.generic_visit(node))

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                and node.func.value.id in ("np", "math"):
            return self._try(node)
        return node


def simplify(src, vocab=None, probe=None):
    """Constant-fold and drop unused temporaries. If `probe(fn)` is given, the result is only returned when it
    behaves identically on that probe (otherwise the original source comes back unchanged)."""
    try:
        tree = _parse(src)
        tree = ast.fix_missing_locations(_Fold().visit(tree))
        fn = _fn(tree)
        changed = True
        while changed:                                                     # remove dead temporaries
            changed = False
            used = {n.id for st in fn.body for n in ast.walk(st) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            for st in list(fn.body):
                if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name) and st.targets[0].id not in used:
                    fn.body.remove(st)
                    changed = True
        out = ast.unparse(tree)
        if out == normalize(src):
            return src
        if probe is not None:
            name = fn.name
            a, b = compile_program(src, name), compile_program(out, name)
            if not np.array_equal(np.asarray(probe(a), dtype=float), np.asarray(probe(b), dtype=float), equal_nan=True):
                return src
        return out + "\n"
    except Exception:                                                      # noqa: BLE001 - never lose a program to cosmetics
        return src
