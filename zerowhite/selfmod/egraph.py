"""egraph.py - equality saturation for the expressions evolution writes.

`astmut.simplify` folds constants and drops dead temporaries; it cannot see that `np.minimum(np.minimum(x, 3.0), 2.0)` is `np.minimum(x, 2.0)`, that
`(a * b) + (a * c)` is `a * (b + c)`, or that `np.maximum(a, np.minimum(a, b))` is `a`. This module does, with an e-graph: the expression is stored
once per *equivalence class*, rewrite rules ADD equal forms instead of replacing the old one (so no rule can make things worse or get stuck in a bad
local choice), the graph is saturated up to a node budget, and the cheapest form is extracted.

Scope: straight-line functions (assignments and one `return`, no loops, no `if`). Temporaries are inlined, the returned expression is simplified, and
the result is only used when it (a) is not larger than the original (the e-graph shares sub-expressions, the output does not, so inlining can grow code),
and (b) gives the same numbers as the original on random inputs (a numeric equivalence check on several input shapes; if none runs, nothing changes).
Anything it does not understand becomes an opaque leaf: unknown calls, attributes, slices stay exactly as written.

Rules are sound over the reals; a few assume finite values (`a * 0 = 0`), which the protected helpers in astmut (`pdiv`, `plog`, ...) keep true for evolved
programs, and the equivalence check would reject a result that differed on the probes. Floating-point reassociation can move results by rounding error only
(the check uses a 1e-9 relative tolerance).

NOT done: sharing common sub-expressions back into temporaries (so a result can be larger and is then discarded), rewriting `if` / loops, and the
"chat-driven refactor into modular functions" idea: that needs a model that writes correct code, which this project's models do not.
"""
from __future__ import annotations

import ast
import math

import numpy as np

from . import astmut as A

NODE_LIMIT = 6000
ITERATIONS = 8

_BIN_CALLS = {"np.minimum": "min", "np.maximum": "max", "pdiv": "pdiv"}
_UN_CALLS = {"np.abs": "abs", "abs": "abs", "relu": "relu", "np.square": "sq", "np.tanh": "f:np.tanh", "np.sin": "f:np.sin", "np.cos": "f:np.cos",
             "psqrt": "f:psqrt", "plog": "f:plog", "pexp": "f:pexp", "sig": "f:sig", "sgn": "f:sgn", "np.sign": "f:sgn", "np.exp": "f:np.exp",
             "np.sqrt": "f:np.sqrt", "np.log": "f:np.log"}
_CMP = {ast.Lt: "lt", ast.Gt: "gt", ast.LtE: "le", ast.GtE: "ge"}
_FOLD = {"f:np.tanh": np.tanh, "f:np.sin": np.sin, "f:np.cos": np.cos, "f:psqrt": A.psqrt, "f:plog": A.plog, "f:pexp": A.pexp, "f:sig": A.sig,
         "f:sgn": A.sgn, "f:np.exp": np.exp, "f:np.sqrt": np.sqrt, "f:np.log": np.log}
_COMM = {"add", "mul", "min", "max"}


class Unsupported(Exception):
    pass


class EGraph:
    def __init__(self):
        self.parent, self.nodes, self.memo = [], [], {}          # nodes[cid] = set of (op, kids...)

    def find(self, c):
        while self.parent[c] != c:
            self.parent[c] = self.parent[self.parent[c]]
            c = self.parent[c]
        return c

    def size(self):
        return sum(len(n) for n in self.nodes)

    def add(self, op, *kids):
        node = (op,) + tuple(self.find(k) if isinstance(k, int) else k for k in kids) if op not in ("const", "var") else (op, kids[0])
        if node in self.memo:
            return self.find(self.memo[node])
        c = len(self.parent)
        self.parent.append(c)
        self.nodes.append({node})
        self.memo[node] = c
        return c

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return a
        if len(self.nodes[a]) < len(self.nodes[b]):
            a, b = b, a
        self.parent[b] = a
        self.nodes[a] |= self.nodes[b]
        self.nodes[b] = set()
        return a

    def _canon(self, n):
        return n if n[0] in ("const", "var") else (n[0],) + tuple(self.find(k) for k in n[1:])

    def rebuild(self):
        """Congruence closure: nodes that became identical after merging classes merge their classes too."""
        changed = True
        while changed:
            changed = False
            seen = {}
            for cid in [c for c in range(len(self.parent)) if self.find(c) == c]:
                for node in sorted(self.nodes[self.find(cid)], key=repr):
                    canon = self._canon(node)
                    if canon in seen and self.find(seen[canon]) != self.find(cid):
                        self.union(seen[canon], cid)
                        changed = True
                    seen[canon] = self.find(cid)
        self.memo = {}
        for cid in range(len(self.parent)):
            if self.find(cid) == cid:
                self.nodes[cid] = {self._canon(n) for n in self.nodes[cid]}
                for n in self.nodes[cid]:
                    self.memo[n] = cid

    def _ns(self, c):
        """The nodes of a class in a fixed order: set iteration order depends on the hash seed, and the output must not."""
        return sorted(self.nodes[self.find(c)], key=repr)

    def const(self, c):
        for n in self._ns(c):
            if n[0] == "const":
                return n[1]
        return None

    # ------------------------------------------------------------------------------------------------ rules
    def _proposals(self, cid, node):
        """New (op, kids) forms equal to `node`, as nested tuples with ints (existing classes) and ('const', v) leaves."""
        op = node[0]
        g = self
        out = []
        K = lambda v: ("const", float(v))                                                  # noqa: E731
        if op in _COMM:
            out.append((op, node[2], node[1]))
        if op in ("add", "mul", "min", "max"):
            a, b = node[1], node[2]
            for n in g._ns(b):                                                    # a op (b1 op b2)  ->  (a op b1) op b2
                if n[0] == op:
                    out.append((op, (op, a, n[1]), n[2]))
            ca, cb = g.const(a), g.const(b)
            if ca is not None and cb is not None:
                out.append(K({"add": ca + cb, "mul": ca * cb, "min": min(ca, cb), "max": max(ca, cb)}[op]))
            if a == b and op in ("min", "max"):
                out.append(a)
            if op == "add":
                if cb == 0.0:
                    out.append(a)
                for n in g._ns(b):
                    if n[0] == "neg":
                        out.append(("sub", a, n[1]))
                for p in g._ns(a):                                                # a*b + a*c -> a*(b+c)
                    if p[0] == "mul":
                        for q in g._ns(b):
                            if q[0] == "mul":
                                for x in (p[1], p[2]):
                                    for y in (q[1], q[2]):
                                        if g.find(x) == g.find(y):
                                            out.append(("mul", x, ("add", p[1] if x == p[2] else p[2], q[1] if y == q[2] else q[2])))
            if op == "mul":
                if cb == 1.0:
                    out.append(a)
                if cb == 0.0:
                    out.append(K(0.0))
                if cb == -1.0:
                    out.append(("neg", a))
                if a == b:
                    out.append(("sq", a))
                for n in g._ns(a):
                    if n[0] == "neg":
                        out.append(("neg", ("mul", n[1], b)))
            if op in ("min", "max"):
                other = "max" if op == "min" else "min"
                for n in g._ns(b):                                                # a op (a other x) -> a
                    if n[0] == other and (g.find(n[1]) == g.find(a) or g.find(n[2]) == g.find(a)):
                        out.append(a)
                if op == "max" and cb == 0.0:
                    out.append(("relu", a))
        elif op == "sub":
            a, b = node[1], node[2]
            ca, cb = g.const(a), g.const(b)
            if ca is not None and cb is not None:
                out.append(K(ca - cb))
            if cb == 0.0:
                out.append(a)
            if ca == 0.0:
                out.append(("neg", b))
            if g.find(a) == g.find(b):
                out.append(K(0.0))
            if cb is not None:
                out.append(("add", a, K(-cb)))
            for n in g._ns(b):
                if n[0] == "neg":
                    out.append(("add", a, n[1]))
            out.append(("add", a, ("neg", b)))
        elif op == "neg":
            a = node[1]
            ca = g.const(a)
            if ca is not None:
                out.append(K(-ca))
            for n in g._ns(a):
                if n[0] == "neg":
                    out.append(n[1])
                if n[0] == "sub":
                    out.append(("sub", n[2], n[1]))
        elif op in ("abs", "relu", "sq"):
            a = node[1]
            ca = g.const(a)
            if ca is not None:
                out.append(K(abs(ca) if op == "abs" else max(ca, 0.0) if op == "relu" else ca * ca))
            for n in g._ns(a):
                if op == "abs" and n[0] in ("abs", "neg", "sq", "relu"):
                    out.append(("abs", n[1]) if n[0] == "neg" else a)
                if op == "relu" and n[0] in ("relu", "sq", "abs"):
                    out.append(a)
                if op == "sq" and n[0] in ("neg", "abs"):
                    out.append(("sq", n[1]))
        elif op == "sel":
            if g.find(node[2]) == g.find(node[3]):
                out.append(node[2])
        elif op.startswith("f:"):
            ca = g.const(node[1])
            if ca is not None:
                with np.errstate(all="ignore"):
                    v = float(_FOLD[op](ca))
                if math.isfinite(v):
                    out.append(K(v))
        elif op == "pdiv":
            ca, cb = g.const(node[1]), g.const(node[2])
            if cb == 1.0:
                out.append(node[1])
            if ca == 0.0:
                out.append(K(0.0))
            if ca is not None and cb is not None:
                v = float(A.pdiv(ca, cb))
                if math.isfinite(v):
                    out.append(K(v))
        return [p for p in out if not (isinstance(p, tuple) and p[0] == "const" and not math.isfinite(p[1]))]      # never write inf / nan into source

    def _build(self, t):
        if isinstance(t, int):
            return self.find(t)
        if t[0] == "const":
            return self.add("const", t[1])
        return self.add(t[0], *[self._build(k) for k in t[1:]])

    def saturate(self, iterations=ITERATIONS, limit=NODE_LIMIT):
        for _ in range(iterations):
            self.rebuild()
            if self.size() > limit:
                break
            props = []
            for cid in range(len(self.parent)):
                if self.find(cid) != cid:
                    continue
                for node in self._ns(cid):
                    if node[0] in ("const", "var"):
                        continue
                    for p in self._proposals(cid, node):
                        props.append((cid, p))
            before = self.size(), len({self.find(i) for i in range(len(self.parent))})
            for cid, p in props:
                if self.size() > limit * 1.5:
                    break
                self.union(cid, self._build(p))
            self.rebuild()
            if (self.size(), len({self.find(i) for i in range(len(self.parent))})) == before:
                break

    # ------------------------------------------------------------------------------------------------ extraction
    def extract(self, root):
        INF = float("inf")
        best = {}
        for _ in range(len(self.parent) + 2):
            changed = False
            for cid in range(len(self.parent)):
                if self.find(cid) != cid:
                    continue
                for n in sorted(self.nodes[cid], key=repr):
                    if n[0] in ("const", "var"):
                        cost = 1.0 + (0.001 if n[0] == "const" and n[1] < 0 else 0.0)
                    else:
                        ks = [best.get(self.find(k), (INF, None))[0] for k in n[1:]]
                        cost = 1.0 + sum(ks) + (0.002 if n[0] in ("neg",) else 0.0)
                        if n[0] in _COMM and any(x[0] == "const" for x in self.nodes[self.find(n[1])]) and not any(
                                x[0] == "const" for x in self.nodes[self.find(n[2])]):
                            cost += 0.0005                          # write `np.minimum(x, 2.0)`, not `np.minimum(2.0, x)`
                    if cost < best.get(cid, (INF, None))[0] - 1e-12:
                        best[cid] = (cost, n)
                        changed = True
            if not changed:
                break
        memo = {}

        def build(c):
            c = self.find(c)
            if c in memo:
                return memo[c]
            n = best[c][1]
            memo[c] = None
            t = n if n[0] in ("const", "var") else (n[0],) + tuple(build(k) for k in n[1:])
            memo[c] = t
            return t
        return build(self.find(root)), best[self.find(root)][0]


# ------------------------------------------------------------------------------------------------ source <-> term
def _to_term(node, g):
    """AST expression -> class id. Unknown things become opaque 'var' leaves holding their source text."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return g.add("const", float(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return g.add("neg", _to_term(node.operand, g))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _to_term(node.operand, g)
    if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult):
        op = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}[type(node.op)]
        return g.add(op, _to_term(node.left, g), _to_term(node.right, g))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP:
        return g.add(_CMP[type(node.ops[0])], _to_term(node.left, g), _to_term(node.comparators[0], g))
    if isinstance(node, ast.Call) and not node.keywords:
        name = ast.unparse(node.func)
        if name in _BIN_CALLS and len(node.args) == 2:
            return g.add(_BIN_CALLS[name], _to_term(node.args[0], g), _to_term(node.args[1], g))
        if name in _UN_CALLS and len(node.args) == 1:
            return g.add(_UN_CALLS[name], _to_term(node.args[0], g))
        if name == "sel" and len(node.args) == 3:
            return g.add("sel", *[_to_term(a, g) for a in node.args])
    text = ast.unparse(node)
    if not isinstance(node, (ast.Name, ast.Subscript, ast.Constant, ast.Attribute, ast.Call)):
        text = f"({text})"                       # an opaque leaf is inserted into larger expressions: without parentheses `a if c else b` would swallow its neighbours
    return g.add("var", text)


def _to_src(t):
    op = t[0]
    if op == "const":
        v = t[1]
        s = repr(int(v)) + ".0" if v == int(v) and abs(v) < 1e15 else repr(v)
        return f"({s})" if v < 0 else s
    if op == "var":
        return t[1]
    k = [_to_src(x) for x in t[1:]]
    if op in ("add", "sub", "mul"):
        return f"({k[0]} {'+-*'['add sub mul'.split().index(op)]} {k[1]})"
    if op == "neg":
        return f"(-{k[0]})"
    if op in ("min", "max"):
        return f"np.{'minimum' if op == 'min' else 'maximum'}({k[0]}, {k[1]})"
    if op == "abs":
        return f"np.abs({k[0]})"
    if op == "sq":
        return f"np.square({k[0]})"
    if op == "relu":
        return f"relu({k[0]})"
    if op == "pdiv":
        return f"pdiv({k[0]}, {k[1]})"
    if op == "sel":
        return f"sel({k[0]}, {k[1]}, {k[2]})"
    if op in ("lt", "gt", "le", "ge"):
        return f"({k[0]} {dict(lt='<', gt='>', le='<=', ge='>=')[op]} {k[1]})"
    if op.startswith("f:"):
        return f"{op[2:]}({k[0]})"
    raise Unsupported(op)


def simplify_expr(expr_src: str):
    """Simplify one expression string. Returns (new_src, cost_before, cost_after). Raises Unsupported / SyntaxError when it cannot."""
    g = EGraph()
    root = _to_term(ast.parse(expr_src, mode="eval").body, g)
    before = g.extract(root)[1]
    g.saturate()
    term, after = g.extract(root)
    return _to_src(term), before, after


def _straight_line(fn: ast.FunctionDef):
    """(env, return_expr) for a body of simple assignments and one trailing return, else None."""
    env = {}
    for st in fn.body[:-1]:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
            continue                                                      # docstring
        if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
            return None
        name = st.targets[0].id
        if name in env or name in {a.arg for a in fn.args.args}:
            return None                                                   # a name assigned twice (or shadowing an argument) cannot simply be inlined
        env[name] = st.value
    last = fn.body[-1]
    if not isinstance(last, ast.Return) or last.value is None:
        return None
    return env, last.value


class _Inline(ast.NodeTransformer):
    def __init__(self, env):
        self.env, self.done = env, {}

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.env:
            if node.id not in self.done:
                self.done[node.id] = None
                self.done[node.id] = self.visit(ast.fix_missing_locations(ast.parse(ast.unparse(self.env[node.id]), mode="eval").body))
            return ast.copy_location(ast.parse(ast.unparse(self.done[node.id]), mode="eval").body, node)
        return node


def _probes(fn_src, name, arg_names, rng):
    """Callables inputs for the equivalence check: several plausible shapes of the arguments."""
    def draw(shape):                                    # several scales and heavy tails: branches and clamps that N(0,1) never reaches must be exercised too
        scale = float(rng.choice([0.5, 1.0, 3.0, 10.0]))
        x = rng.normal(0, scale, size=shape)
        if rng.random() < 0.3:
            x = x + rng.choice([-1.0, 1.0], size=shape) * rng.exponential(5.0, size=shape)
        return x
    return (lambda: [draw(40) for _ in arg_names], lambda: [draw((48, 40)) for _ in arg_names],
            lambda: [float(draw(())) for _ in arg_names], lambda: [draw(8) for _ in arg_names])


def simplify(src: str, probe=None, tol: float = 1e-9) -> str:
    """Return `src` simplified by equality saturation, or `src` itself when that is not possible, not smaller, or not provably the same on probes."""
    try:
        tree = ast.parse(src)
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if len(fns) != 1 or len(tree.body) != 1:
            return src
        fn = fns[0]
        sl = _straight_line(fn)
        if sl is None:
            return src
        env, ret = sl
        inlined = _Inline(env).visit(ast.parse(ast.unparse(ret), mode="eval").body)
        new_src, _, _ = simplify_expr(ast.unparse(inlined))
        new_ret = ast.parse(new_src, mode="eval").body
        fn.body = [ast.Return(value=new_ret)]
        out = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
        if A.count_nodes(out) > A.count_nodes(src):
            return src                                                          # inlining shared temporaries can make code bigger: keep the original
        a, b = A.compile_program(src, fn.name), A.compile_program(out, fn.name)
        if probe is not None:
            ra, rb = np.asarray(probe(a), dtype=float), np.asarray(probe(b), dtype=float)
            return out if ra.shape == rb.shape and np.allclose(ra, rb, rtol=tol, atol=tol, equal_nan=True) else src
        rng = np.random.default_rng(12345)
        names = [x.arg for x in fn.args.args]
        ran = 0
        with np.errstate(all="ignore"):
            for mk in _probes(src, fn.name, names, rng):
                for _ in range(12):
                    args = mk()
                    try:
                        ra = np.asarray(a(*args), dtype=float)
                    except Exception:                                           # noqa: BLE001 - this shape does not suit the program
                        break
                    rb = np.asarray(b(*args), dtype=float)
                    if ra.shape != rb.shape or not np.allclose(ra, rb, rtol=tol, atol=tol, equal_nan=True):
                        return src
                    ran += 1
        return out if ran else src
    except Exception:                                                            # noqa: BLE001 - cosmetics must never lose a program
        return src
