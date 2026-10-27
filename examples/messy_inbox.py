"""Make a folder of the kind of files people actually have, each broken in a different way, to try `devour` and `organism` on.

    python examples/messy_inbox.py /tmp/inbox
    python -m zerowhite devour /tmp/inbox --workspace zw --solve --allow-code
    python -m zerowhite organism --workspace zw --inbox /tmp/inbox --allow-code --self-modify hooks --minutes 5

What it writes (12 files):
  clean.csv             a plain table                          copy_of_clean.csv  the same bytes under another name (a duplicate)
  excel_export.csv      Windows-1252, semicolons, decimal commas, accented column names, ragged rows, missing values
  events.jsonl          JSON Lines with a garbage line and a truncated last record, nested fields
  download.csv.gz       a gzip cut off at 80 percent (a stopped download)
  bundle.zip            a zip with its directory missing (a stopped download): a text file and a table inside
  utf16.csv             a UTF-16 table
  essay.md              prose                                   catch.py           an environment (runs only with --allow-code)
  junk.bin  pic.png  empty.txt     things that cannot become tasks: each is set aside with a reason

Files are dated a minute ago so `devour` does not mistake them for downloads in progress.
"""
import gzip
import io
import json
import os
import sys
import time
import zipfile

import numpy as np


def put(folder, name, data):
    path = os.path.join(folder, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    t = time.time() - 60
    os.utime(path, (t, t))


def make(folder, n=240, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] * x[:, 1] + 0.3 * x[:, 2] > 0).astype(int)
    clean = ("a,b,c,label\n" + "\n".join(f"{a:.3f},{b:.3f},{c:.3f},{int(a + b > 0)}" for a, b, c in x) + "\n").encode()
    put(folder, "clean.csv", clean)
    put(folder, "copy_of_clean.csv", clean)
    rows = ["Prix;Qté;Région;Résultat"]
    for i in range(n):
        rows.append(f"{abs(x[i, 0]) * 50:.2f};{'' if i % 13 == 0 else int(abs(x[i, 1]) * 9)};{['Nord', 'Sud', 'Île'][i % 3]};{y[i]}"
                    .replace(".", ",", 1) + (";" if i % 40 == 0 else ""))
    put(folder, "excel_export.csv", ("\n".join(rows) + "\n").encode("cp1252"))
    lines = [json.dumps({"x": float(a), "y": float(b), "z": {"k": float(c)}, "label": int(k)}) for (a, b, c), k in zip(x, y)]
    lines.insert(50, "this line is garbage")
    lines[-1] = lines[-1][:20]
    put(folder, "events.jsonl", "\n".join(lines).encode())
    g = gzip.compress(("u,v,w,out\n" + "\n".join(f"{a:.3f},{b:.3f},{3 * a - b:.3f},{a * b:.3f}" for a, b, _ in x) + "\n").encode())
    put(folder, "download.csv.gz", g[:int(len(g) * 0.8)])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("notes/story.txt", "It was the best of times, it was the worst of times. " * 30)
        z.writestr("data/regr.csv", "p,q,r\n" + "\n".join(f"{a:.3f},{b:.3f},{a + 2 * b:.3f}" for a, b, _ in x) + "\n")
    zb = buf.getvalue()
    put(folder, "bundle.zip", zb[:zb.rfind(b"PK\x01\x02")])
    put(folder, "utf16.csv", ("s,t,class\n" + "\n".join(f"{a:.2f},{b:.2f},{k}" for (a, b, _), k in zip(x, y)) + "\n").encode("utf-16"))
    put(folder, "essay.md", ("# Notes\nThe river does not hurry, yet everything is accomplished. " * 40).encode())
    put(folder, "junk.bin", rng.bytes(4000))
    put(folder, "pic.png", b"\x89PNG\r\n\x1a\n" + rng.bytes(300))
    put(folder, "empty.txt", b"")
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "catch_env.py"), "rb") as f:
        put(folder, "envs/catch.py", f.read())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    target = sys.argv[1]
    if os.path.exists(target) and os.listdir(target):
        sys.exit(f"{target} is not empty; choose a new folder")
    make(target)
    print(f"wrote 12 files under {target}")
