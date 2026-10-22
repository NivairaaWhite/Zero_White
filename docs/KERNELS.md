# Kernels (6.4): a lazy tensor graph, generated C, generated CUDA

```bash
python -m zerowhite kernels --size 2000000     # what THIS machine can do: C compiler? same numbers as numpy? faster? CUDA driver / NVRTC / a device?
```
```python
from zerowhite.kernels import lazy, evaluate
g = (((lazy(x) * lazy(y) + lazy(z).tanh()) * 0.5).relu() - 0.25).abs().sum(axis=1)
evaluate(g, "numpy")      # node by node (the reference)
evaluate(g, "c")          # one generated C loop for the whole elementwise chain, compiled with the system C compiler, run through ctypes
evaluate(g, "cuda", cuda=CudaBackend())   # the same fused chain as CUDA C, NVRTC + the driver API through ctypes
```

**What exists** (`kernels/graph.py`, `cjit.py`, `cuda.py`). A DAG of primitive operations (add sub mul div neg exp log tanh relu sqrt abs maximum minimum, sum / max reductions, reshape, matmul) is only *recorded* by operator overloading. `evaluate` fuses every maximal chain of elementwise operations
whose intermediates have no other consumer into one loop (a value used twice stays materialised), with broadcasting done by strides baked into the generated source, and runs it on numpy memory through `ctypes`. `cjit` compiles C with whatever compiler the machine has
(`$CC`, `cc`, `gcc`, `clang`; not MSVC) into a shared library cached by the hash of its source. `cuda.py` is the CUDA path written by hand: NVRTC compiles the generated source to PTX, `cuModuleLoadData` loads it, `cuMemAlloc` / `cuMemcpyHtoD` / `cuLaunchKernel` (grid-stride loop) / `cuMemcpyDtoH` run it;
library lookup covers `libcuda.so.1` / `libnvrtc.so*` on Linux and `nvcuda.dll` / `nvrtc64_*.dll` on Windows.

**What was verified** (`tests/test_v64.py`): 60 random graphs (broadcasting, reductions, reshapes, every operator) give the same numbers on numpy, generated C and the generated CUDA source (executed on the CPU by an emulation that compiles it as C with `blockIdx` / `threadIdx` / `gridDim`
defined and calls the kernel once per thread, so the real text, including its index arithmetic and its grid-stride loop, is what runs); fusion actually happens (a six-operation chain is one loop; a value used twice is computed once); sizes that are not a multiple of the launch size; `CudaBackend` issues the right calls in the
right order against a fake driver and frees device memory even when a launch fails; `RawDriver` / `RawNvrtc` pass the right function names and arguments to a fake library and turn error codes into exceptions.

**What was NOT verified, and what the measurements say.**
* **No GPU, no NVRTC, no driver was available to the author.** Whether real NVRTC accepts the generated source, whether `cuLaunchKernel`'s parameter packing is right on a real driver, any speed, and Windows are untested; `zerowhite kernels` runs the real path on your machine and prints what happened. Please send me the output.
* **The generated C is not faster than numpy on the build machine.** Best of 7, 2,000,000 elements: a chain of cheap operations 1.31x faster (19.5 vs 25.5 ms); with tanh / sqrt / exp 0.55x (70.6 vs 38.5 ms); at 1,000,000 elements 0.85x and 0.24x. Fusion saves memory passes, but numpy has SIMD exp / tanh and plain libm in a scalar loop does not. Do not expect a CPU speed-up; the case for the design is the GPU, where fusion matters, and that is the part I could not test.
* float64 only; no autograd through the graph (the engine's eager NumPy path is untouched and is still what trains); reductions run on the host for CUDA; matmul is numpy / BLAS on every backend; one thread per element, no shared memory, no multi-GPU, no operator evolution (the review's "kernels selfmod can evolve" is not implemented: the source is generated from the graph, not mutated).
