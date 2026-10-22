"""kernels - a lazy tensor graph with fused kernels, run by numpy, by generated C, or by generated CUDA. See graph.py, cjit.py, cuda.py."""
from .graph import LT, evaluate, lazy, maximum, minimum  # noqa: F401
