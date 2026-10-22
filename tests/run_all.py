"""Run every test module: python3 tests/run_all.py"""
import importlib
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # the repo root, so `import zerowhite` works without installing

def main():
    passed, failed, t0 = 0, [], time.time()
    for modname in ("test_basic", "test_v3", "test_v4", "test_chat", "test_voice", "test_v6", "test_v61", "test_gpu", "test_v63", "test_v64"):
        mod = importlib.import_module(modname)
        for name in [n for n in dir(mod) if n.startswith("test_")]:
            fn = getattr(mod, name)
            t = time.time()
            try:
                fn()
                passed += 1
                print(f"  PASS  {modname}.{name}  ({time.time() - t:.1f}s)")
            except Exception:  # noqa: BLE001
                failed.append(f"{modname}.{name}")
                print(f"  FAIL  {modname}.{name}\n{traceback.format_exc(limit=3)}")
    print(f"\n{passed}/{passed + len(failed)} passed in {time.time() - t0:.0f}s")
    if failed:
        print("Failed:", ", ".join(failed))
        sys.exit(1)

if __name__ == "__main__":
    main()
