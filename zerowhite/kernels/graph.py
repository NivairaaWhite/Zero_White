"""graph.py - a lazy tensor graph with kernel fusion, evaluated by numpy, by generated C (cjit.py), or by generated CUDA (cuda.py).

Operations on `LT` objects only RECORD a graph (a DAG of primitive operations: add sub mul div neg exp log tanh relu sqrt abs maximum minimum, sum / max reductions,
reshape, matmul). `evaluate(t, backend)` then:

  numpy   runs the graph node by node (the reference; every intermediate is materialised).
  c       fuses every maximal chain of elementwise operations whose intermediates have no other consumer into ONE generated C loop, compiles it (cjit), and runs it on the
          numpy buffers through ctypes. `a * b + c.tanh()` becomes a single pass over memory instead of three passes and two temporaries.
  cuda    the same fused regions as CUDA C source, compiled with NVRTC and launched through the driver API (cuda.py). Reductions, reshapes and matmul run on the host.

All three give the same numbers (the test compares them to 1e-9 relative on random graphs); float64 only. Broadcasting follows numpy; a leaf that is broadcast is read through
strides baked into the generated source (so the source is specialised to the shapes, and cached by hash).

NOT done: autograd through this graph, float32/float16, a device-side reduction, and a fast matmul of our own (matmul calls numpy / BLAS on every backend, which beats a naive
generated loop). The engine's eager NumPy path is untouched; this module is an addition, selected by calling it, not a replacement of engine.py.
"""
from __future__ import annotations

import numpy as np

ELEMENTWISE = {"add": "bin", "sub": "bin", "mul": "bin", "div": "bin", "maximum": "bin", "minimum": "bin",
               "neg": "un", "exp": "un", "log": "un", "tanh": "un", "relu": "un", "sqrt": "un", "abs": "un"}
_NP = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide, "maximum": np.maximum, "minimum": np.minimum, "neg": np.negative, "exp": np.exp,
       "log": np.log, "tanh": np.tanh, "relu": lambda x: np.maximum(x, 0.0), "sqrt": np.sqrt, "abs": np.abs}


class LT:
    """A node of the graph. Build with `lazy(array)`; combine with + - * / and the methods below."""
    __slots__ = ("op", "args", "shape", "data", "attrs", "id")
    _n = 0

    def __init__(self, op, args=(), shape=(), data=None, attrs=None):
        self.op, self.args, self.shape, self.data, self.attrs = op, tuple(args), tuple(shape), data, attrs or {}
        LT._n += 1
        self.id = LT._n

    # ---- construction
    @staticmethod
    def _wrap(x):
        return x if isinstance(x, LT) else lazy(np.asarray(x, dtype=np.float64))

    def _bin(self, op, o, swap=False):
        a, b = (LT._wrap(o), self) if swap else (self, LT._wrap(o))
        return LT(op, (a, b), np.broadcast_shapes(a.shape, b.shape))

    def _un(self, op):
        return LT(op, (self,), self.shape)

    __add__ = lambda s, o: s._bin("add", o)                                     # noqa: E731
    __radd__ = lambda s, o: s._bin("add", o, True)                              # noqa: E731
    __sub__ = lambda s, o: s._bin("sub", o)                                     # noqa: E731
    __rsub__ = lambda s, o: s._bin("sub", o, True)                              # noqa: E731
    __mul__ = lambda s, o: s._bin("mul", o)                                     # noqa: E731
    __rmul__ = lambda s, o: s._bin("mul", o, True)                              # noqa: E731
    __truediv__ = lambda s, o: s._bin("div", o)                                 # noqa: E731
    __rtruediv__ = lambda s, o: s._bin("div", o, True)                          # noqa: E731
    __neg__ = lambda s: s._un("neg")                                            # noqa: E731

    def exp(self): return self._un("exp")                                       # noqa: E704
    def log(self): return self._un("log")                                       # noqa: E704
    def tanh(self): return self._un("tanh")                                     # noqa: E704
    def relu(self): return self._un("relu")                                     # noqa: E704
    def sqrt(self): return self._un("sqrt")                                     # noqa: E704
    def abs(self): return self._un("abs")                                       # noqa: E704

    def sum(self, axis=None, keepdims=False): return _reduce("sum", self, axis, keepdims)      # noqa: E704
    def max(self, axis=None, keepdims=False): return _reduce("max", self, axis, keepdims)      # noqa: E704

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return LT("reshape", (self,), np.empty(self.shape, dtype=bool).reshape(shape).shape)

    def __matmul__(self, o):
        o = LT._wrap(o)
        return LT("matmul", (self, o), np.empty((*self.shape[:-1], 1)).shape[:-1] + (o.shape[-1],) if o.ndim == 2 else self.shape[:-1])

    @property
    def ndim(self): return len(self.shape)                                      # noqa: E704


def lazy(a) -> LT:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return LT("input", (), a.shape, data=a)


def maximum(a, b): return LT._wrap(a)._bin("maximum", b)                        # noqa: E704
def minimum(a, b): return LT._wrap(a)._bin("minimum", b)                        # noqa: E704


def _reduce(kind, x, axis, keepdims):
    if axis is None:
        shape = (1,) * x.ndim if keepdims else ()
    else:
        ax = axis % x.ndim
        shape = tuple(1 if i == ax else s for i, s in enumerate(x.shape)) if keepdims else tuple(s for i, s in enumerate(x.shape) if i != ax)
    return LT("reduce_" + kind, (x,), shape, attrs=dict(axis=None if axis is None else axis % x.ndim, keepdims=keepdims))


# ------------------------------------------------------------------------------------------------ planning + code generation
def topo(root):
    order, seen = [], set()

    def visit(n):
        if n.id in seen:
            return
        seen.add(n.id)
        for a in n.args:
            visit(a)
        order.append(n)
    visit(root)
    return order


def count_consumers(order):
    cons = {n.id: 0 for n in order}
    for n in order:
        for a in n.args:
            cons[a.id] += 1
    return cons


class Region:
    """One fused elementwise kernel: the expression tree rooted at `root` over `leaves` (materialised arrays), producing root.shape."""

    def __init__(self, root, leaves, lines):
        self.root, self.leaves, self.lines = root, leaves, lines


def _leaf_offset(shape, out_shape, name, idx_vars):
    """C expression for the flat element offset of a leaf of `shape` broadcast to `out_shape`, using multi-index variables."""
    nd = len(out_shape)
    s = (1,) * (nd - len(shape)) + tuple(shape)
    if tuple(s) == tuple(out_shape):
        return "i"
    if int(np.prod(s)) == 1:
        return "0"
    strides, acc = [0] * nd, 1
    for d in range(nd - 1, -1, -1):
        strides[d] = 0 if s[d] == 1 else acc
        acc *= s[d]
    terms = [f"{idx_vars[d]}*{strides[d]}L" for d in range(nd) if strides[d] != 0]
    return " + ".join(terms) if terms else "0"


def plan_regions(root):
    """Split the graph into fused elementwise regions and 'host' nodes. Returns (order, cons, region_of) with region_of[node.id] = Region for region roots."""
    order = topo(root)
    cons = count_consumers(order)
    is_root = {n.id: n is root for n in order}
    for n in order:
        for a in n.args:
            if n.op not in ELEMENTWISE or cons[a.id] > 1:
                is_root[a.id] = True
    return order, cons, is_root


def build_region(node, cons, is_root, dtype="double"):
    """Emit the SSA body for the elementwise tree under `node`. Returns (leaves, body_lines, result_var, leaf_names)."""
    leaves, body, names, memo = [], [], {}, {}

    def emit(n, top=False):
        if n.id in memo:
            return memo[n.id]
        if n.op not in ELEMENTWISE or (is_root[n.id] and not top):
            if n.id not in names:
                names[n.id] = len(leaves)
                leaves.append(n)
            v = f"L{names[n.id]}"
            memo[n.id] = v
            return v
        args = [emit(a) for a in n.args]
        v = f"t{len(body)}"
        a = args
        expr = {"add": f"{a[0] if len(a) else ''} + {a[1] if len(a) > 1 else ''}", "sub": f"{a[0] if a else ''} - {a[1] if len(a) > 1 else ''}",
                "mul": f"{a[0] if a else ''} * {a[1] if len(a) > 1 else ''}", "div": f"{a[0] if a else ''} / {a[1] if len(a) > 1 else ''}"}.get(n.op)
        if expr is None:
            expr = {"neg": f"-{a[0]}", "exp": f"exp({a[0]})", "log": f"log({a[0]})", "tanh": f"tanh({a[0]})", "sqrt": f"sqrt({a[0]})", "abs": f"fabs({a[0]})",
                    "relu": f"(({a[0]}) > 0.0 ? ({a[0]}) : 0.0)" if a else "",
                    "maximum": f"(({a[0]}) > ({a[1]}) ? ({a[0]}) : ({a[1]}))" if len(a) > 1 else "",
                    "minimum": f"(({a[0]}) < ({a[1]}) ? ({a[0]}) : ({a[1]}))" if len(a) > 1 else ""}[n.op]
        body.append(f"{dtype} {v} = {expr};")
        memo[n.id] = v
        return v
    res = emit(node, top=True)
    return leaves, body, res


def kernel_body(node, cons, is_root, flat_loop: str, dtype="double"):
    """(leaves, c_source_pieces) for one region: parameter list, index code and expression, shared by the C and CUDA emitters."""
    leaves, body, res = build_region(node, cons, is_root, dtype)
    nd = len(node.shape)
    out_shape = node.shape
    idx_vars = [f"j{d}" for d in range(nd)]
    needs_idx = any(_leaf_offset(l.shape, out_shape, "", idx_vars) not in ("i", "0") for l in leaves)
    decomp = []
    if needs_idx:
        decomp.append("long r = i;")
        for d in range(nd - 1, -1, -1):
            decomp.append(f"long {idx_vars[d]} = r % {int(out_shape[d])}L; r /= {int(out_shape[d])}L;")
    loads = [f"{dtype} L{k} = a{k}[{_leaf_offset(l.shape, out_shape, '', idx_vars)}];" for k, l in enumerate(leaves)]
    return leaves, decomp, loads, body, res


def emit_c(node, cons, is_root):
    leaves, decomp, loads, body, res = kernel_body(node, cons, is_root, "")
    n = int(np.prod(node.shape)) if node.shape else 1
    params = ", ".join([f"const double* a{k}" for k in range(len(leaves))] + ["double* out"])
    lines = "\n".join("    " + x for x in decomp + loads + body + [f"out[i] = {res};"])
    src = f"#include <math.h>\nvoid k({params}) {{\n  for (long i = 0; i < {n}L; i++) {{\n{lines}\n  }}\n}}\n"
    return leaves, src


def emit_cuda(node, cons, is_root):
    leaves, decomp, loads, body, res = kernel_body(node, cons, is_root, "")
    n = int(np.prod(node.shape)) if node.shape else 1
    params = ", ".join([f"const double* a{k}" for k in range(len(leaves))] + ["double* out"])
    lines = "\n".join("    " + x for x in decomp + loads + body + [f"out[i] = {res};"])
    src = (f'extern "C" __global__ void k({params}) {{\n  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < {n}L; i += (long)gridDim.x * blockDim.x) {{\n{lines}\n  }}\n}}\n')
    return leaves, src


REDUCE_C = """#include <math.h>
void red(const double* a, double* out, long outer, long mid, long inner) {
  for (long o = 0; o < outer; o++) for (long i = 0; i < inner; i++) {
    double acc = INIT;
    for (long m = 0; m < mid; m++) { double v = a[(o * mid + m) * inner + i]; acc = COMBINE; }
    out[o * inner + i] = acc;
  }
}
"""


def evaluate(root: LT, backend: str = "numpy", cuda=None, stats=None):
    """Run the graph. `stats` (a dict) receives {'kernels': n_generated_kernels, 'fused_ops': n_elementwise_nodes_fused, 'sources': [...]}."""
    if backend == "numpy":
        return _eval_numpy(root)
    if backend not in ("c", "cuda"):
        raise ValueError("backend must be numpy, c or cuda")
    order, cons, is_root = plan_regions(root)
    stats = stats if stats is not None else {}
    stats.update(kernels=0, fused_ops=0, sources=[])
    vals = {}

    def get(n):
        if n.id in vals:
            return vals[n.id]
        if n.op == "input":
            v = n.data
        elif n.op in ELEMENTWISE:
            v = _run_region(n, cons, is_root, get, backend, cuda, stats)
        elif n.op == "reshape":
            v = np.ascontiguousarray(get(n.args[0])).reshape(n.shape)
        elif n.op == "matmul":
            v = np.ascontiguousarray(get(n.args[0]) @ get(n.args[1]))
        elif n.op.startswith("reduce_"):
            v = _run_reduce(n, get(n.args[0]), backend)
        else:
            raise ValueError(n.op)
        vals[n.id] = v
        return v
    return get(root)


def _run_region(node, cons, is_root, get, backend, cuda, stats):
    from . import cjit
    if backend == "c":
        leaves, src = emit_c(node, cons, is_root)
    else:
        leaves, src = emit_cuda(node, cons, is_root)
    stats["kernels"] += 1
    stats["fused_ops"] += sum(1 for n in topo(node) if n.op in ELEMENTWISE and not (is_root[n.id] and n is not node))
    stats["sources"].append(src)
    arrs = [np.ascontiguousarray(get(l), dtype=np.float64) for l in leaves]
    out = np.empty(node.shape, dtype=np.float64)
    if backend == "c":
        import ctypes
        lib = cjit.build(src)
        f = lib.k
        f.restype = None
        f.argtypes = [ctypes.c_void_p] * (len(arrs) + 1)
        f(*[cjit.ptr(a) for a in arrs], cjit.ptr(out))
        return out
    if cuda is None:
        raise RuntimeError("the cuda backend needs a CudaBackend (see kernels/cuda.py)")
    return cuda.run(src, arrs, out)


def _run_reduce(node, x, backend):
    x = np.ascontiguousarray(x, dtype=np.float64)
    ax = node.attrs["axis"]
    kind = node.op.split("_")[1]
    if backend != "c" or x.size == 0:
        f = np.sum if kind == "sum" else np.max
        return np.asarray(f(x, axis=ax, keepdims=node.attrs["keepdims"]), dtype=np.float64).reshape(node.shape)
    import ctypes
    from . import cjit
    if ax is None:
        outer, mid, inner = 1, x.size, 1
    else:
        outer, mid, inner = int(np.prod(x.shape[:ax])), x.shape[ax], int(np.prod(x.shape[ax + 1:]))
    src = REDUCE_C.replace("INIT", "0.0" if kind == "sum" else "-INFINITY").replace("COMBINE", "acc + v" if kind == "sum" else "(v > acc ? v : acc)")
    lib = cjit.build(src)
    out = np.empty(max(1, outer * inner), dtype=np.float64)
    lib.red.restype = None
    lib.red.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_long]
    lib.red(cjit.ptr(x), cjit.ptr(out), outer, mid, inner)
    return out[: outer * inner].reshape(node.shape)


def _eval_numpy(root):
    vals = {}
    for n in topo(root):
        if n.op == "input":
            vals[n.id] = n.data
        elif n.op in ELEMENTWISE:
            vals[n.id] = _NP[n.op](*[vals[a.id] for a in n.args])
        elif n.op == "reshape":
            vals[n.id] = vals[n.args[0].id].reshape(n.shape)
        elif n.op == "matmul":
            vals[n.id] = vals[n.args[0].id] @ vals[n.args[1].id]
        else:
            f = np.sum if n.op == "reduce_sum" else np.max
            vals[n.id] = np.asarray(f(vals[n.args[0].id], axis=n.attrs["axis"], keepdims=n.attrs["keepdims"]), dtype=np.float64)
    return np.asarray(vals[root.id], dtype=np.float64)
