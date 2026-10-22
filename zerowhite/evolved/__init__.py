"""evolved - source files that Zero White is allowed to rewrite about itself (level 1).

Each hook is one small Python function in its own file. The search machinery
(evolve.py) calls these hooks. The versions shipped here are the GENESIS versions: they
are trivial on purpose (constant step size, random parent). Everything smarter has to be
discovered by `selfmod` and is stored as an overlay in the workspace
(<workspace>/selfmod/live/<hook>.py), leaving the shipped files untouched unless you
explicitly promote an overlay into the source tree.

Loading is defensive: a hook that fails the static safety check, does not compile, raises,
or returns garbage is replaced by its genesis behaviour for that call, and the fault is
counted so `doctor` can roll the hook back.
"""

import math
import os

import numpy as np

from ..selfmod.astmut import program_namespace
from ..selfmod.safety import violations

HERE = os.path.dirname(os.path.abspath(__file__))

HOOKS = {
    "sigma_update": ("sigma", "improved", "generation"),
    "select_parent": ("fitnesses", "rng"),
}

_SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                  for n in ("abs", "min", "max", "sum", "len", "range", "int", "float", "bool",
                            "round", "sorted", "enumerate", "zip", "all", "any", "pow", "divmod")}
_ACTIVE = {"dir": os.environ.get("ZERO_WHITE_LIVE") or None}
_CACHE = {}


def use(directory):
    """Make `directory` the overlay that hooks are loaded from (None = shipped genesis files)."""
    _ACTIVE["dir"] = directory


def active_dir():
    return _ACTIVE["dir"]


def hook_path(name, overlay="__active__"):
    d = _ACTIVE["dir"] if overlay == "__active__" else overlay
    if d:
        p = os.path.join(d, name + ".py")
        if os.path.exists(p):
            return p
    return os.path.join(HERE, name + ".py")


def genesis_source(name):
    with open(os.path.join(HERE, name + ".py"), "r", encoding="utf-8") as f:
        return f.read()


def compile_hook(name, src):
    """Check + compile one hook's source into a callable, or raise ValueError."""
    bad = violations(src)
    if bad:
        raise ValueError(f"hook '{name}' failed the safety gate: {'; '.join(bad[:4])}")
    ns = program_namespace()           # np, math, builtins subset + helpers pdiv/psqrt/plog/pexp/sel/relu/sig/sgn
    exec(compile(src, f"<hook {name}>", "exec"), ns)          # noqa: S102 - gated by violations() above
    fn = ns.get(name)
    if not callable(fn):
        raise ValueError(f"hook source does not define a function called '{name}'")
    return fn


def load_hook(name, overlay="__active__"):
    path = hook_path(name, overlay)
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _CACHE:
        with open(path, "r", encoding="utf-8") as f:
            _CACHE[key] = compile_hook(name, f.read())
        if len(_CACHE) > 64:
            _CACHE.pop(next(iter(_CACHE)))
    return _CACHE[key]


_BUILTIN_GENESIS = {"sigma_update": lambda sigma, improved, generation: sigma,
                    "select_parent": lambda fitnesses, rng: int(rng.integers(len(fitnesses)))}


class Policy:
    """The set of live hooks, wrapped so that a faulty hook can never crash or corrupt a search."""

    def __init__(self, sigma_update=None, select_parent=None, label="live"):
        self.label = label
        self._fn = {"sigma_update": sigma_update, "select_parent": select_parent}
        self.faults = {k: 0 for k in HOOKS}

    @classmethod
    def load(cls, overlay="__active__"):
        p = cls(label=f"overlay:{active_dir() if overlay == '__active__' else overlay}")
        for name in HOOKS:
            try:
                p._fn[name] = load_hook(name, overlay)
            except Exception:                                    # noqa: BLE001 - fall back, count the fault
                p.faults[name] += 1
                try:
                    p._fn[name] = load_hook(name, overlay=None)
                except Exception:                                # noqa: BLE001 - even genesis is broken:
                    p._fn[name] = _BUILTIN_GENESIS[name]         # the last line of defence cannot fail
        return p

    def update_sigma(self, sigma, improved, generation):
        try:
            v = float(self._fn["sigma_update"](float(sigma), bool(improved), int(generation)))
            if math.isfinite(v):
                return v
        except Exception:                                        # noqa: BLE001
            pass
        self.faults["sigma_update"] += 1
        return float(sigma)

    def pick_parent(self, fitnesses, rng):
        n = len(fitnesses)
        try:
            return min(max(int(self._fn["select_parent"](np.asarray(fitnesses, dtype=float), rng)), 0), n - 1)
        except Exception:                                        # noqa: BLE001
            self.faults["select_parent"] += 1
            return int(rng.integers(n))


def textbook_policy():
    """The classical 1/5-success-rule policy. NOT used by default: it exists as a reference
    point so benchmarks can say how far the evolved hooks are from the textbook rule."""
    return Policy(sigma_update=lambda s, improved, g: s * (1.3 if improved else 0.85),
                  select_parent=lambda f, r: int(r.integers(len(f))), label="textbook")
