"""Run every test module: python3 tests/run_all.py            (or only some: python3 tests/run_all.py test_v65_tools test_v65_moa)"""
import faulthandler
import importlib
import os
import sys
import threading
import time
import traceback

faulthandler.enable()          # a hard crash (segfault, exit 139) prints the Python stack of every thread instead of dying silently

try:                           # threads started by the tests (stream pipeline, mock API server, ring producer) get a 16 MB stack: macOS gives a secondary thread only 512 KB, and NumPy / Accelerate calls made from one can overflow it
    threading.stack_size(16 * 2 ** 20)
except (ValueError, RuntimeError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # the repo root, so `import zerowhite` works without installing

def main():
    passed, failed, t0 = 0, [], time.time()
    modules = ("test_basic", "test_v3", "test_v4", "test_chat", "test_voice", "test_v6", "test_v61", "test_gpu", "test_v63", "test_v64", "test_v65",
               "test_v65_tools", "test_v65_episodic", "test_v65_moa", "test_v65_stream", "test_v65_act", "test_v65_obsenc", "test_v65_architect", "test_v65_reach")
    for modname in (tuple(sys.argv[1:]) or modules):
        mod = importlib.import_module(modname)
        for name in [n for n in dir(mod) if n.startswith("test_")]:
            fn = getattr(mod, name)
            t = time.time()
            try:
                fn()
                passed += 1
                print(f"  PASS  {modname}.{name}  ({time.time() - t:.1f}s)", flush=True)     # flushed: when stdout is a pipe a crash would otherwise lose the last few KB of lines
            except Exception:  # noqa: BLE001
                failed.append(f"{modname}.{name}")
                print(f"  FAIL  {modname}.{name}\n{traceback.format_exc(limit=3)}", flush=True)
    print(f"\n{passed}/{passed + len(failed)} passed in {time.time() - t0:.0f}s", flush=True)
    if failed:
        print("Failed:", ", ".join(failed), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
