"""Tests for v6: the substrate probe, the omnivore (ingestion that does not break), and the organism (the autonomous loop)."""
import contextlib
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("ZERO_WHITE_DATA", tempfile.mkdtemp(prefix="zw_test_data_"))

from zerowhite import organism as O                                   # noqa: E402
from zerowhite import substrate                                       # noqa: E402
from zerowhite.omnivore import Library, Limits, devour, load_tasks    # noqa: E402
from zerowhite.omnivore import archives, sniff, tables                # noqa: E402

L = Limits()


def tmp():
    return tempfile.mkdtemp(prefix="zw_t6_")


def put(d, name, data, age=60.0):
    """Write a file and make it look `age` seconds old (a file younger than `settle_seconds` is deliberately left alone)."""
    p = os.path.join(d, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)
    t = time.time() - age
    os.utime(p, (t, t))
    return p


def csv_bytes(n=120, seed=0, header="a,b,c,label", noise=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    y = np.where(rng.random(n) < noise, 1 - y, y)                      # label noise keeps some headroom (a perfect task is rightly left alone)
    return (header + "\n" + "\n".join(f"{a:.3f},{b:.3f},{c:.3f},{k}" for (a, b, c), k in zip(x, y)) + "\n").encode()


def tree_hash(d):
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(d)):
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(p.encode())
            h.update(open(p, "rb").read())
    return h.hexdigest()


def hostile_inbox(d, n=120):
    """A folder of things people actually have: every one broken in a different way."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] * x[:, 1] + 0.3 * x[:, 2] > 0).astype(int)
    put(d, "clean.csv", csv_bytes(n))
    rows = ["Prix;Qté;Région;Résultat"]
    for i in range(n):
        rows.append(f"{abs(x[i, 0]) * 50:.2f};{'' if i % 13 == 0 else int(abs(x[i, 1]) * 9)};{['Nord', 'Sud', 'Île'][i % 3]};{y[i]}"
                    .replace(".", ",", 1) + (";" if i % 40 == 0 else ""))
    put(d, "excel_export.csv", ("\n".join(rows) + "\n").encode("cp1252"))
    lines = [json.dumps({"x": float(a), "y": float(b), "z": {"k": float(c)}, "label": int(k)}) for (a, b, c), k in zip(x, y)]
    lines.insert(30, "this line is garbage")
    lines[-1] = lines[-1][:20]
    put(d, "events.jsonl", "\n".join(lines).encode())
    g = gzip.compress(("u,v,w,out\n" + "\n".join(f"{a:.3f},{b:.3f},{3 * a - b:.3f},{a * b:.3f}" for a, b, _ in x) + "\n").encode())
    put(d, "download.csv.gz", g[:int(len(g) * 0.8)])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("notes/story.txt", "It was the best of times, it was the worst of times. " * 30)
        z.writestr("data/regr.csv", "p,q,r\n" + "\n".join(f"{a:.3f},{b:.3f},{a + 2 * b:.3f}" for a, b, _ in x) + "\n")
    zb = buf.getvalue()
    put(d, "bundle.zip", zb[:zb.rfind(b"PK\x01\x02")])
    put(d, "essay.md", ("# Notes\nThe river does not hurry, yet everything is accomplished. " * 40).encode())
    put(d, "junk.bin", np.random.default_rng(2).bytes(3000))
    put(d, "pic.png", b"\x89PNG\r\n\x1a\n" + np.random.default_rng(3).bytes(200))
    put(d, "empty.txt", b"")
    put(d, "utf16.csv", ("s,t,class\n" + "\n".join(f"{a:.2f},{b:.2f},{k}" for (a, b, _), k in zip(x, y)) + "\n").encode("utf-16"))
    put(d, "copy_of_clean.csv", csv_bytes(n))


def by_label(digest):
    return {os.path.basename(e.label): e for e in digest.entries}


# =============================================================================================== substrate
def test_substrate_probe_plan_and_cache():
    p = substrate.probe(bench=False)
    for k in ("cpus", "ram_gb", "ram_available_gb", "platform", "accelerators"):
        assert k in p
    full = substrate.probe(tmp())
    assert full["gflops"] > 0 and full["disk_mb_s"] >= 0
    pl = substrate.plan(full)
    assert pl["max_rows"] >= 4000 and pl["max_file_bytes"] >= 8 << 20
    assert all(k in pl["reasons"] for k in ("max_file_bytes", "max_rows", "max_text_chars"))      # every number says where it came from
    ws = tmp()
    a = substrate.cached(ws)
    assert a["from_cache"] is False
    b = substrate.cached(ws)
    assert b["from_cache"] is True and b["gflops"] == a["gflops"]
    st = json.load(open(os.path.join(ws, substrate.CACHE_NAME)))                                  # the workspace is copied to a bigger machine
    st["cpus"] = st["cpus"] + 7
    json.dump(st, open(os.path.join(ws, substrate.CACHE_NAME), "w"))
    assert substrate.cached(ws)["from_cache"] is False
    assert "accelerators" in substrate.describe(full)


def test_substrate_plan_survives_a_failed_probe():
    pl = substrate.plan(dict(gflops=0.0, ram_available_gb=0.0, ram_gb=0.0))
    assert pl["max_rows"] == 20_000 and pl["max_file_bytes"] > 0 and "could not be measured" in pl["reasons"]["max_rows"]
    lim = Limits.for_machine(pl)
    assert lim.max_rows == 20_000 and lim.max_member_bytes <= lim.max_file_bytes


# =============================================================================================== sniff
def test_identify_uses_content_not_names_and_short_magics_do_not_misfire():
    assert sniff.identify(gzip.compress(b"x")) == "gzip" and sniff.identify(b"") == "empty"
    assert sniff.identify(b"ID3 tags are what this note is about") == "text"
    assert sniff.identify(b"BZh is a prefix") == "text"
    assert sniff.identify(b"\x00\x01\x02\x03") == "binary" and sniff.identify(b"a,b\n1,2\n") == "text"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        ti = tarfile.TarInfo("a")
        ti.size = 3
        t.addfile(ti, io.BytesIO(b"abc"))
    assert sniff.identify(buf.getvalue()) == "tar"


def test_decode_handles_encodings_padding_and_damage_and_reports_it():
    d = sniff.decode(("café naïve\n" * 30).encode("cp1252"))
    assert d.encoding == "cp1252" and "café" in d.text
    d = sniff.decode("a,b\n1,2\n".encode("utf-16"))
    assert d.text.startswith("a,b")
    d = sniff.decode("a,b\n1,2\n".encode("utf-16-le"))                                             # no BOM at all
    assert d.encoding == "utf-16-le" and d.text.startswith("a,b")
    d = sniff.decode(b"a,b\n1,2\n3,4\n" + b"\x00" * 40)
    assert d.text == "a,b\n1,2\n3,4\n" and any("NUL" in r for r in d.repairs)
    d = sniff.decode(("x" * 600 + "é").encode() * 4 + b"\xff")
    assert d.encoding == "utf-8" and any("replaced" in r for r in d.repairs)
    assert sniff.decode(bytes(range(256)) * 8) is None                                              # genuinely binary


def test_json_records_whole_lines_columnar_nested_and_truncated():
    r, rep = sniff.json_records('[{"a":1,"b":2},{"a":3,"b":4}]')
    assert len(r) == 2 and r[0] == {"a": 1, "b": 2}
    r, rep = sniff.json_records('{"a":[1,2,3],"b":[4,5,6]}')
    assert len(r) == 3 and "columnar" in rep[0]
    r, rep = sniff.json_records('{"meta":1,"rows":[{"x":1},{"x":2}]}')
    assert len(r) == 2 and "rows" in rep[0]
    r, rep = sniff.json_records('{"p":{"q":1},"r":2}\n{"p":{"q":3},"r":4}\ngarbage\n{"p":{"q":5},"r":6}\n')
    assert len(r) == 3 and r[0] == {"p.q": 1, "r": 2} and any("unparsable" in x for x in rep)
    r, rep = sniff.json_records('[{"a":1,"b":2},{"a":3,"b":4},{"a":5,"b"')
    assert len(r) == 2 and "truncated" in rep[0]
    assert sniff.json_records("just words") == ([], []) and sniff.json_records("[1,2,3]")[0] == []


def test_env_contract_is_detected_statically_in_every_spelling():
    base = "class Env:\n    {attrs}\n    def reset(self, rng): return [0.0]\n    def step(self, a): return [0.0], 0.0, True, {{}}\n"
    for attrs in ("obs_dim, n_actions = 1, 2", "obs_dim = 1\n    n_actions = 2", "obs_dim: int = 1\n    n_actions: int = 2"):
        assert sniff.classify_text(base.format(attrs=attrs))[1]["has_env"], attrs
    prop = ("class Env:\n    @property\n    def obs_dim(self): return 1\n    @property\n    def n_actions(self): return 2\n"
            "    def reset(self, rng): return [0.0]\n    def step(self, a): return [0.0], 0.0, True, {}\n")
    assert sniff.classify_text(prop)[1]["has_env"]
    assert not sniff.classify_text("class Env:\n    obs_dim = 1\n    n_actions = 2\n    def reset(self, rng): pass\n")[1]["has_env"]
    assert sniff.classify_text("import os\n\ndef f():\n    return 1\n")[0] == "python"


# =============================================================================================== tables
def test_parse_number_reads_the_formats_people_actually_use():
    p = tables.parse_number
    assert p("1,234.5") == 1234.5 and p("$1,000") == 1000.0 and p("(7.5)") == -7.5 and p("45%") == 45.0 and p("1e-3") == 0.001
    assert p("1.234,5", ",") == 1234.5 and p("3,14", ",") == 3.14 and p("1 234,5", ",") == 1234.5
    assert p("nan") is None and p("inf") is None and p("abc") is None and p("") is None and p("1e400") is None


def test_delimiter_is_found_not_assumed_and_decimal_commas_do_not_fool_it():
    # every line has exactly one comma (the decimal comma, and one in the header), so ',' is perfectly *consistent*;
    # one row in nine has a stray extra field, so ';' is not. Only the true delimiter yields more columns.
    rows = ["id;price,net;qty;label"] + [f"{i};{i * 1.5:.2f};{i % 7};{i % 2}".replace(".", ",", 1) + (";x" if i % 9 == 0 else "") for i in range(1, 70)]
    t = tables.parse_delimited("\n".join(rows) + "\n")
    assert t.delimiter == ";" and t.columns == ["id", "price,net", "qty", "label"] and "ragged" in " ".join(t.repairs)
    assert tables.parse_delimited("a\tb\tc\n" + "\n".join(f"{i}\t{i * 2}\t{i % 3}" for i in range(40)) + "\n").delimiter == "\t"
    assert tables.parse_delimited("a|b|c\n" + "\n".join(f"{i}|{i * 2}|{i % 3}" for i in range(40)) + "\n").delimiter == "|"
    assert tables.parse_delimited("1.0  2.0  3\n" * 40).delimiter == " "
    prose = "\n".join(", ".join(" ".join(["word%d" % (i * j) for j in range(1, 7)]) for _ in range(2)) + "." for i in range(1, 60))
    assert tables.parse_delimited(prose) is None                                                   # prose with commas is not a table


def test_table_repairs_are_made_and_reported():
    body = "a,b,label\n" + "\n".join(f"{i * 0.1:.1f},{i % 5},{i % 2}" for i in range(80))
    t = tables.parse_delimited("# exported by a tool\nsource: x\n" + body[:-2])                      # preamble + cut-off last row
    r = " | ".join(t.repairs)
    assert "preamble" in r and "cut off" in r and t.columns == ["a", "b", "label"]
    ragged = "a,b,label\n" + "\n".join((f"{i},{i},{i % 2}" + (",extra" if i % 9 == 0 else "")) if i % 11 else f"{i},{i % 2}" for i in range(1, 80)) + "\n"
    assert "ragged" in " ".join(tables.parse_delimited(ragged).repairs)
    quoted = "x,y,note,label\n" + "\n".join(f'{i},{i},"oops {i},{i % 2}' if i == 5 else f"{i},{i},ok,{i % 2}" for i in range(60)) + "\n"
    t = tables.parse_delimited(quoted)
    assert len(t.rows) >= 50 and "quote" in " ".join(t.repairs)                                      # one bad quote must not swallow the file


def test_build_features_target_and_unusable_reasons():
    n = 100
    rng = np.random.default_rng(0)
    rows = ["id,when,flag,color,amount,const,label"]
    for i in range(n):
        rows.append(f'{i},2023-{i % 12 + 1:02d}-{i % 27 + 1:02d},{"yes" if i % 2 else "no"},{["r", "g", "b"][i % 3]},'
                    f'"{int(rng.integers(1000, 9000)):,}",7,{i % 3}')
    b = tables.build(tables.parse_delimited("\n".join(rows) + "\n"), L)
    assert b.kind == "classification" and b.target_name == "label" and len(b.class_names) == 3
    names = " ".join(b.feature_names)
    assert "id" not in b.feature_names and "const" not in b.feature_names                            # index and constant columns dropped
    assert "when (days)" in names and "flag" in names and "color=r" in names and "amount" in b.feature_names
    assert np.isfinite(b.X).all() and b.X.shape[0] == n
    # missing numbers: median + indicator
    r2 = ["a,b,label"] + [f"{'' if i % 5 == 0 else i},{i % 7},{i % 2}" for i in range(1, 101)]
    b2 = tables.build(tables.parse_delimited("\n".join(r2) + "\n"), L)
    assert "a (was missing)" in b2.feature_names and any("median" in r for r in b2.repairs)
    # an identifier is not a label
    try:
        tables.build(tables.parse_delimited("a,b,name\n" + "\n".join(f"{i},{i * 2},person_{i}" for i in range(100)) + "\n"), L)
        raise AssertionError("an identifier column must not become a target")
    except tables.Unusable as u:
        assert "identifier" in u.reason and "--target" in u.reason
    try:
        tables.build(tables.parse_delimited("a,b,label\n" + "\n".join(f"{i},{i},{i % 2}" for i in range(10)) + "\n"), L)
        raise AssertionError("too few rows")
    except tables.Unusable as u:
        assert "10 usable row" in u.reason
    try:
        tables.build(tables.parse_delimited(csv_bytes().decode()), L, target="nope")
        raise AssertionError("bad target name")
    except tables.Unusable as u:
        assert "does not exist" in u.reason


def test_target_choice_rare_classes_and_regression():
    txt = csv_bytes(100).decode().replace("a,b,c", "y,b,c")
    b = tables.build(tables.parse_delimited(txt), L)
    assert b.target_name == "label"                                  # two hinted names ('y', 'label') are ambiguous: the last column wins
    assert tables.build(tables.parse_delimited(txt), L, target="y").target_name == "y"
    rows = ["f,g,species"] + [f"{i},{i % 4},{['cat', 'dog'][i % 2]}" for i in range(80)] + ["1,1,okapi", "2,2,okapi"]
    b = tables.build(tables.parse_delimited("\n".join(rows) + "\n"), L)
    assert b.class_names == ["cat", "dog"] and any("class(es) with fewer" in r for r in b.repairs)    # a class that cannot be tested is set aside
    rr = "p,q,r\n" + "\n".join(f"{i * 0.37:.3f},{(i * 7) % 11},{i * 0.9 + (i % 5):.3f}" for i in range(90)) + "\n"
    assert tables.build(tables.parse_delimited(rr), L).kind == "regression"


# =============================================================================================== archives
def test_archives_salvage_truncated_streams_and_directories():
    csv = csv_bytes(400)
    g = gzip.compress(csv)
    m = list(archives.expand(g[:len(g) // 2], "t.csv.gz", "gzip", L))[0]
    assert m.name == "t.csv" and 0 < len(m.data) < len(csv) and csv.startswith(m.data[:200]) and any("truncated" in n for n in m.notes)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("a/one.csv", csv)
        z.writestr("two.txt", b"hello " * 200)
    zb = buf.getvalue()
    got = {m.name: m for m in archives.expand(zb[:zb.rfind(b"PK\x01\x02")], "z.zip", "zip", L)}      # no central directory
    assert got["a/one.csv"].data == csv and got["two.txt"].data == b"hello " * 200
    assert "directory" in got["a/one.csv"].notes[0]
    part = {m.name: m for m in archives.expand(zb[:len(zb) // 3], "z.zip", "zip", L)}               # cut inside a member
    assert part["a/one.csv"].data and csv.startswith(part["a/one.csv"].data[:100])


def test_archives_are_bounded_and_never_trust_member_names():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(4):
            z.writestr(f"bomb{i}.txt", b"\0" * (20 << 20))
    lim = Limits(max_member_bytes=64 << 20, max_total_bytes=50 << 20)
    ms = list(archives.expand(buf.getvalue(), "bomb.zip", "zip", lim))
    assert sum(len(m.data) for m in ms) <= 50 << 20 and any("zip-bomb" in m.error for m in ms)
    big = list(archives.expand(buf.getvalue(), "b.zip", "zip", Limits(max_member_bytes=1 << 20)))
    assert all(m.error and "larger than" in m.error for m in big)
    assert archives.safe_label("../../etc/passwd") == "etc/passwd" and archives.safe_label("C:\\x\\..\\y.txt") == "x/y.txt"
    assert ".." not in archives.safe_label("a/../../b") and not archives.safe_label("/abs/path").startswith("/")
    tb = io.BytesIO()
    with tarfile.open(fileobj=tb, mode="w") as t:
        ti = tarfile.TarInfo("ok.txt")
        ti.size = 5
        t.addfile(ti, io.BytesIO(b"hello"))
        ln = tarfile.TarInfo("link")
        ln.type, ln.linkname = tarfile.SYMTYPE, "/etc/passwd"
        t.addfile(ln)
    assert [m.name for m in archives.expand(tb.getvalue(), "a.tar", "tar", L)] == ["ok.txt"]        # links are never followed


# =============================================================================================== devour
def test_devour_hostile_inbox_every_file_gets_the_right_outcome():
    inbox, ws = tmp(), tmp()
    hostile_inbox(inbox)
    before = tree_hash(inbox)
    d = devour(inbox, ws, limits=L)
    e = by_label(d)
    assert e["clean.csv"].status == "task" and e["copy_of_clean.csv"].status == "duplicate"
    assert e["excel_export.csv"].status == "task" and any("cp1252" in r for r in e["excel_export.csv"].repairs)
    assert any("decimal commas" in r for r in e["excel_export.csv"].repairs)
    assert e["events.jsonl"].status == "task" and any("unparsable" in r for r in e["events.jsonl"].repairs)
    assert not any(r.startswith("line:") for r in e["events.jsonl"].repairs)                       # internal markers never reach the report
    assert e["events.jsonl"].kind == "jsonl"
    assert e["download.csv"].status == "task" and any("truncated gzip" in r for r in e["download.csv"].repairs)
    assert e["regr.csv"].status == "task" and any("directory" in r for r in e["regr.csv"].repairs)
    assert e["story.txt"].status == "pooled" and e["essay.md"].status == "pooled"
    assert e["utf16.csv"].status == "task"
    for name, why in (("junk.bin", "binary"), ("pic.png", "image"), ("empty.txt", "empty")):
        assert e[name].status == "quarantined" and why in e[name].reason, (name, e[name].reason)
    assert e["bundle.zip"].status == "expanded" and e["download.csv.gz"].status == "expanded"
    assert tree_hash(inbox) == before                                 # sources are never moved, edited, or added to
    lib = Library(ws)
    assert {"clean", "excel_export", "events", "download", "regr", "utf16", "corpus-notes"} <= set(lib.names())
    q = lib.quarantined()
    assert {os.path.basename(r["label"]) for r in q} >= {"junk.bin", "pic.png", "empty.txt"} and all(r["reason"] for r in q)
    assert "task" in d.summary() and d.to_dict()["counts"]["task"] == d.counts()["task"]


def test_devour_is_idempotent_cheap_and_heals_a_corrupt_ledger():
    inbox, ws = tmp(), tmp()
    hostile_inbox(inbox)
    d1 = devour(inbox, ws, limits=L)
    t = time.time()
    d2 = devour(inbox, ws, limits=L)
    assert d2.tasks == [] and set(d2.counts()) <= {"unchanged", "skipped"} and time.time() - t < 2.0
    open(os.path.join(ws, "omnivore", "ledger.json"), "w").write("{this is not json")
    d3 = devour(inbox, ws, limits=L)
    assert d3.healed and any(n.startswith("ledger.json.corrupt-") for n in os.listdir(os.path.join(ws, "omnivore")))
    assert sorted(d3.tasks) == sorted(d1.tasks)                                                     # same sources keep the same task names
    open(os.path.join(ws, "omnivore", "quarantine.jsonl"), "a").write('{"label": "torn line')       # a hard kill mid-write
    assert len(Library(ws).quarantined()) >= 3


def test_devour_waits_for_files_still_being_written_and_retries_changed_ones():
    inbox, ws = tmp(), tmp()
    fresh = put(inbox, "live.csv", csv_bytes(100), age=0.0)
    d = devour(inbox, ws, limits=Limits(settle_seconds=5.0))
    assert d.entries[0].status == "deferred" and "still be" in d.entries[0].reason and d.tasks == []
    os.utime(fresh, (time.time() + 2, time.time() + 2))                           # a timestamp slightly in the future is "just written" too
    assert devour(inbox, ws, limits=Limits(settle_seconds=5.0)).entries[0].status == "deferred"
    os.utime(fresh, (time.time() - 60, time.time() - 60))
    assert devour(inbox, ws, limits=L).tasks == ["live"]
    p = put(inbox, "half.csv", csv_bytes(8))                                      # too few rows to learn from: set aside
    e = by_label(devour(inbox, ws, limits=L))["half.csv"]
    assert e.status == "quarantined" and "usable row" in e.reason
    again = by_label(devour(inbox, ws, limits=L))["half.csv"]
    assert again.status == "unchanged"                                            # not retried while nothing changed
    open(p, "ab").write(csv_bytes(100, seed=5).split(b"\n", 1)[1])                # the download finishes
    t = time.time() - 60
    os.utime(p, (t, t))
    assert by_label(devour(inbox, ws, limits=L))["half.csv"].status == "task"
    assert by_label(devour(inbox, ws, limits=L, retry=True))["half.csv"].status in ("unchanged", "task")


def test_devour_never_reads_its_own_workspace_or_follows_symlinks():
    inbox = tmp()
    ws = os.path.join(inbox, "zw")
    put(inbox, "a.csv", csv_bytes(100))
    devour(inbox, ws, limits=L)
    d2 = devour(inbox, ws, limits=L)
    assert all(not os.path.abspath(e.label).startswith(ws + os.sep) for e in d2.entries)
    assert d2.tasks == []
    outside = tmp()
    put(outside, "secret.csv", csv_bytes(100))
    try:
        os.symlink(os.path.join(outside, "secret.csv"), os.path.join(inbox, "link.csv"))
    except (OSError, NotImplementedError):
        return
    e = by_label(devour(inbox, ws, limits=L))
    assert e["link.csv"].status == "skipped" and "symbolic link" in e["link.csv"].reason


def test_devour_numpy_sqlite_and_json_sources():
    inbox, ws = tmp(), tmp()
    rng = np.random.default_rng(0)
    arr = np.column_stack([rng.normal(size=(100, 3)), (rng.random(100) > 0.5).astype(float)])
    buf = io.BytesIO()
    np.save(buf, arr)
    put(inbox, "arr.npy", buf.getvalue())
    b1 = io.BytesIO()
    np.save(b1, rng.normal(size=50))
    put(inbox, "vec.npy", b1.getvalue())
    import sqlite3
    db = os.path.join(inbox, "shop.db")
    con = sqlite3.connect(db)
    con.execute("create table sales (a real, b real, y integer)")
    con.executemany("insert into sales values (?,?,?)", [(float(i), float(i % 7), i % 2) for i in range(100)])
    con.execute("create table tiny (x int, y int)")
    con.commit()
    con.close()
    os.utime(db, (time.time() - 60, time.time() - 60))
    put(inbox, "cols.json", json.dumps({"a": list(range(60)), "b": [i % 5 for i in range(60)], "y": [i % 2 for i in range(60)]}).encode())
    e = by_label(devour(inbox, ws, limits=L))
    assert e["arr.npy"].status == "task" and e["vec.npy"].status == "quarantined" and "2-D" in e["vec.npy"].reason
    assert e["sales"].status == "task" and e["shop.db"].status == "expanded"
    assert e["cols.json"].status == "task"
    assert "shop_sales" in Library(ws).names()


def test_devour_environment_files_need_permission_and_pass_a_sandboxed_smoke_test():
    inbox, ws = tmp(), tmp()
    put(inbox, "catch.py", open(os.path.join(ROOT, "examples", "catch_env.py"), "rb").read())
    put(inbox, "helper.py", b"import math\n\ndef f(x):\n    return math.sqrt(abs(x))\n" * 20)
    bad = ("class Env:\n    obs_dim, n_actions = 2, 2\n    def reset(self, rng): return [0.0]\n"
           "    def step(self, a): return [0.0, 0.0], 0.0, False, {}\n")
    put(inbox, "broken_env.py", bad.encode())
    e = by_label(devour(inbox, ws, limits=L))
    assert e["catch.py"].status == "deferred" and "--allow-code" in e["catch.py"].reason
    assert e["helper.py"].status == "pooled"                                      # plain code is just text
    assert by_label(devour(inbox, ws, limits=L))["catch.py"].status == "deferred"  # still waiting, not silently "unchanged"
    e = by_label(devour(inbox, ws, limits=L, allow_code=True))
    assert e["catch.py"].status == "task" and e["catch.py"].task == "catch"
    assert e["broken_env.py"].status == "quarantined" and "smoke test" in e["broken_env.py"].reason
    lib = Library(ws)
    try:
        lib.load_task("catch")
        raise AssertionError("loading a user env must require allow_code")
    except PermissionError:
        pass
    t = lib.load_task("catch", allow_code=True)
    assert t.name == "catch" and t.kind == "env" and t.signature()["dim"] == 3


def test_devour_survives_fuzz_random_bytes_and_every_truncation():
    """The property that matters: whatever the bytes, no exception, and every file gets exactly one recorded outcome."""
    rng = np.random.default_rng(7)
    inbox, ws = tmp(), tmp()
    csv = csv_bytes(60)
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("t.csv", csv)
    seeds = {"csv": csv, "gz": gzip.compress(csv), "zip": zbuf.getvalue(), "json": json.dumps([{"a": i, "b": i % 3} for i in range(40)]).encode(),
             "utf16": csv.decode().encode("utf-16"), "py": b"class Env:\n    obs_dim, n_actions = 1, 1\n    def reset(self, r): return [0]\n    def step(self, a): return [0], 0, True, {}\n"}
    n = 0
    for kind, data in seeds.items():
        for frac in (0.07, 0.31, 0.5, 0.93):
            put(inbox, f"cut_{kind}_{n}.dat", data[:max(1, int(len(data) * frac))])
            n += 1
        flipped = bytearray(data)
        for i in rng.integers(0, len(flipped), size=max(3, len(flipped) // 50)):
            flipped[int(i)] = int(rng.integers(0, 256))
        put(inbox, f"flip_{kind}_{n}.dat", bytes(flipped))
        n += 1
    for i in range(12):
        put(inbox, f"noise_{i}.dat", rng.bytes(int(rng.integers(1, 5000))))
        n += 1
    put(inbox, "plain.zip", zbuf.getvalue())
    d = devour(inbox, ws, limits=L, allow_code=True)                              # must not raise
    allowed = {"task", "pooled", "expanded", "duplicate", "unchanged", "deferred", "quarantined", "skipped", "missing"}
    assert len(d.entries) >= n and all(e.status in allowed for e in d.entries)
    assert all(e.reason for e in d.entries if e.status == "quarantined")           # never a silent drop
    assert not any("internal error" in e.reason for e in d.entries), [e.reason for e in d.entries if "internal error" in e.reason]
    d2 = devour(inbox, ws, limits=L, allow_code=True)
    assert d2.tasks == []                                                          # and the second pass has nothing new to say


def test_devour_respects_size_limits_and_reports_partial_reads():
    inbox, ws = tmp(), tmp()
    put(inbox, "big.csv", csv_bytes(2000))
    d = devour(inbox, ws, limits=Limits(max_file_bytes=20_000, min_rows=10, settle_seconds=0.0))
    e = d.entries[0]
    assert e.status == "task" and any("only the first part" in n for n in e.notes)
    ws2 = tmp()
    d2 = devour(inbox, ws2, limits=Limits(max_rows=300, settle_seconds=0.0))
    assert d2.entries[0].status == "task" and Library(ws2).spec("big")["rows"] == 300


def test_library_tasks_are_deterministic_and_solvable_end_to_end():
    from zerowhite.forge.generalist import Generalist
    inbox, ws = tmp(), tmp()
    put(inbox, "d.csv", csv_bytes(150))
    devour(inbox, ws, limits=L)
    a, b = load_tasks(ws)[0], load_tasks(ws)[0]
    assert np.array_equal(a.Xval, b.Xval) and np.array_equal(a.yval, b.yval)             # same split every time: incumbents are judged fairly
    assert a.kind == "classification" and a.class_names == ("0", "1")
    rec = Generalist(ws, seed=0, verbose=False).solve(a, budget=0.3, n_new=3, rungs=2)
    assert rec["improved"] and rec["skill"] > 0.3


# =============================================================================================== organism
class Scripted(O.Organism):
    """An organism whose attempts follow a script, so the scheduling decisions can be checked exactly."""
    script = {}
    calls = []

    def _act_solve(self, dec, t0):
        seq = self.script[dec["task"]]
        skill, secs = seq.pop(0) if len(seq) > 1 else seq[0]
        self.calls.append((dec["task"], round(dec["budget"], 3)))
        prev = self.state["tasks"][dec["task"]]["best_skill"]
        return dict(ok=True, skill=skill, improved=prev is None or skill > prev + 1e-9, family="mlp", seconds=secs)


def scripted(script, **kw):
    inbox, ws = tmp(), tmp()
    for i, name in enumerate(script):
        put(inbox, f"{name}.csv", csv_bytes(100, seed=100 + i))               # distinct data: identical files would be (rightly) deduplicated
    settings = dict(inbox=[inbox], quiet=True, probe_budget=0.25, standard_budget=1.0, patience=2, min_gain=0.01, doctor_every=0)
    settings.update(kw)
    cls = type("S", (Scripted,), dict(script={k: list(v) for k, v in script.items()}, calls=[]))
    return cls(ws, O.Settings(**settings)), cls


def test_organism_tries_every_new_task_once_before_deepening_any():
    org, cls = scripted({"aaa": [(0.5, 1.0)], "bbb": [(0.4, 1.0)], "ccc": [(0.6, 1.0)]})
    for _ in range(3):
        org.step()
    assert [c[0] for c in cls.calls] == ["aaa", "bbb", "ccc"] and all(c[1] == 0.25 for c in cls.calls)      # a cheap probe each, in order
    ts = org.state["tasks"]["aaa"]
    assert ts["attempts"] == 1 and ts["gain_ewma"] is None and ts["best_skill"] == 0.5 and ts["budget"] == 1.0   # the probe is a baseline, not progress
    ev = [json.loads(line) for line in open(org.log_path)]
    assert all(e["decision"]["reason"] for e in ev if e["kind"] == "decision")                                   # every decision says why


def test_organism_prefers_progress_per_second_and_never_wastes_time_at_the_ceiling():
    org, cls = scripted({"fast": [(0.50, 1.0), (0.60, 1.0), (0.70, 1.0), (0.80, 1.0)],
                         "slow": [(0.50, 50.0), (0.52, 50.0), (0.54, 50.0)],
                         "done": [(1.00, 1.0)]})
    for _ in range(3):
        org.step()                                                    # the three baselines
    assert org.state["tasks"]["done"]["plateaued"] and "ceiling" in org.state["tasks"]["done"]["why_plateaued"]
    org.step()
    org.step()
    picked = [c[0] for c in cls.calls[3:]]
    assert "done" not in [c[0] for c in cls.calls[3:]]                # a perfect task is left alone
    assert picked[0] == "fast"                                        # the same headroom costs 1s there and 50s elsewhere: take the cheap one first
    assert org.priority(org.state["tasks"]["fast"]) > org.priority(org.state["tasks"]["slow"])


def test_expected_gain_is_measured_progress_capped_by_headroom():
    org, _ = scripted({"t": [(0.5, 1.0)]})
    ts = org._new_task_state()
    ts.update(best_skill=0.5, gain_ewma=None)
    assert abs(org.expected_gain(ts) - 0.25) < 1e-9                      # nothing measured yet: assume half the headroom
    ts.update(best_skill=0.97, gain_ewma=0.5)
    assert abs(org.expected_gain(ts) - 0.03) < 1e-9                      # a big past gain cannot promise more than what is left
    ts.update(best_skill=0.60, gain_ewma=0.02)
    assert abs(org.expected_gain(ts) - 0.02) < 1e-9
    ts.update(best_skill=1.0, gain_ewma=0.3)
    assert org.expected_gain(ts) == 0.0 and org.priority(ts) == 0.0


def test_organism_doubles_the_budget_after_no_gain_and_plateaus_after_patience():
    org, cls = scripted({"only": [(0.5, 1.0)] * 4})
    for _ in range(3):
        org.step()
    budgets = [c[1] for c in cls.calls]
    assert budgets == [0.25, 1.0, 2.0]                                # probe, standard, then double after an attempt that gained nothing
    ts = org.state["tasks"]["only"]
    assert ts["plateaued"] and ts["stale"] == 2 and ts["budget"] == 4.0 and "gained less than" in ts["why_plateaued"]
    out = org.step()
    assert out["decision"]["action"] == "idle" and "plateaued" in out["decision"]["reason"] and "off" in out["decision"]["reason"]
    assert len(cls.calls) == 3                                        # a plateaued task is left alone


def test_organism_changes_how_it_searches_only_when_stuck_and_reopens_tasks_on_a_win():
    org, cls = scripted({"t": [(0.5, 1.0)]}, self_modify="hooks", patience=1, min_gain=0.5)
    tried = []

    def fake_modify(dec, t0):
        tried.append(dec["arm"])
        return dict(ok=True, won=len(tried) == 2, seconds=1.0)

    org._act_self_modify = fake_modify
    org._doctor = lambda why: tried.append("doctor:" + why)
    org.step()                                                        # baseline
    assert not tried
    org.step()                                                        # second attempt gains nothing -> plateau
    assert org.state["tasks"]["t"]["plateaued"] and not tried
    d = org.step()["decision"]
    assert d["action"] == "self_modify" and d["arm"] in ("hooks:select_parent", "hooks:sigma_update") and "plateaued" in d["reason"]
    assert org.state["tasks"]["t"]["plateaued"]                       # a loss changes nothing
    org.step()                                                        # a different arm: never the same one twice running
    arms_tried = [x for x in tried if not x.startswith("doctor:")]
    assert len(arms_tried) == 2 and arms_tried[0] != arms_tried[1]
    assert not org.state["tasks"]["t"]["plateaued"] and org.state["tasks"]["t"]["stale"] == 0   # the win re-opened it
    assert any(x.startswith("doctor:") for x in tried)                # and triggered a self-check
    assert org.state["arms"][arms_tried[1]]["wins"] == 1 and org.state["arms"][arms_tried[0]]["wins"] == 0
    assert "poet" not in tried                                        # POET only runs when the user brought an environment task


def test_organism_never_repeats_a_self_improvement_back_to_back():
    org, _ = scripted({"t": [(0.5, 1.0)]}, self_modify="hooks", patience=1, min_gain=0.5, seed=3)
    seen = []
    org._act_self_modify = lambda dec, t0: (seen.append(dec["arm"]), dict(ok=True, won=False, seconds=0.1))[1]
    for _ in range(2):
        org.step()                                                    # baseline, then the attempt that plateaus
    for _ in range(8):
        org.step()
    assert len(seen) == 8 and all(a != b for a, b in zip(seen, seen[1:])), seen
    assert set(seen) == {"hooks:select_parent", "hooks:sigma_update"}


def test_organism_failures_back_off_and_never_stop_the_loop():
    org, cls = scripted({"ok": [(0.5, 1.0), (0.6, 1.0), (0.7, 1.0)], "bad": [(0.5, 1.0)]})
    real = org._act_solve

    def act(dec, t0):
        if dec["task"] == "bad":
            raise RuntimeError("boom")
        return real(dec, t0)

    org._act_solve = act
    outs = [org.step() for _ in range(6)]
    assert any(o["result"].get("ok") is False and "boom" in o["result"]["error"] for o in outs)
    bad = org.state["tasks"]["bad"]
    assert bad["fails"] >= 1 and bad["until"] > org.state["cycle"] - 1                   # it backs off, exponentially
    assert org.state["tasks"]["ok"]["attempts"] >= 2                                      # while the healthy task keeps improving
    assert os.path.exists(org.state_path)


def test_organism_fits_each_attempt_to_the_time_left():
    org, cls = scripted({"t": [(0.5, 100.0), (0.55, 100.0)]}, minutes=1.0)
    org.step()                                                    # baseline: 100 s at budget 0.25 => ~571 s per unit budget
    ts = org.state["tasks"]["t"]
    assert ts["sec_per_unit"] > 0
    dec = org.decide(dict(free_gb=8.0, new_tasks=[], counts={}))
    assert dec["budget"] < 1.0                                           # 1.0 would take ~10 minutes; one minute is left
    assert dec["action"] == "solve"                                      # a minute is enough for a smaller attempt
    assert dec["predicted_s"] * org.s.time_margin <= 61 and "cut" in dec["reason"] and "safety" in dec["reason"]   # and the estimate is padded


def test_organism_rests_when_memory_is_short_and_resumes_from_its_state():
    org, cls = scripted({"a": [(0.5, 1.0), (0.6, 1.0)], "b": [(0.4, 1.0)]})
    assert org.decide(dict(free_gb=0.01, new_tasks=[], counts={}))["action"] == "rest"
    org.step()
    org.step()
    again = type("S2", (Scripted,), dict(script={"a": [(0.6, 1.0)], "b": [(0.4, 1.0)]}, calls=[]))(org.ws, O.Settings(**dict(org.s.__dict__)))
    assert again.state["cycle"] == 2 and again.state["tasks"]["a"]["attempts"] == 1               # it remembers what it learned
    open(org.state_path, "w").write("{broken")
    healed = O.Organism(org.ws, O.Settings(inbox=[], quiet=True))
    assert healed.state["cycle"] == 0 and any(n.startswith("state.json.corrupt-") for n in os.listdir(org.dir))


def test_organism_has_a_lock_a_stop_file_and_honest_bounds():
    org, cls = scripted({"a": [(0.5, 0.1)] * 30}, forever=True, poll_seconds=0.01)
    org.s.max_cycles = None
    org.deadline = time.time() + 8                                    # a regression must fail this test, not hang the suite
    fileio_write = open(org.lock_path, "w")
    fileio_write.write(str(os.getppid()))
    fileio_write.close()
    from zerowhite.chat import hardware as hw
    if hw.pid_alive(os.getppid()):
        try:
            org.run()
            raise AssertionError("a second organism must not start on the same workspace")
        except RuntimeError as ex:
            assert "already running" in str(ex)
    os.remove(org.lock_path)
    open(org.stop_path, "w").close()
    out = org.run()
    assert "stop file" in out["stopped"] and not os.path.exists(org.lock_path) and not os.path.exists(org.stop_path)
    org2, _ = scripted({"a": [(0.5, 0.1)] * 30}, max_cycles=2)
    out = org2.run()
    assert out["stopped"] == "cycle budget used" and out["cycles"] == 2
    org3, _ = scripted({"a": [(0.5, 0.1), (0.5, 0.1), (0.5, 0.1), (0.5, 0.1), (0.5, 0.1)]}, max_cycles=50)
    out = org3.run()
    assert out["stopped"].startswith("nothing worth doing")                 # a bounded run ends when nothing is worth doing
    assert "plateaued" in O.status(org3.ws) and "recent decisions" in O.status(org3.ws)


def test_organism_real_run_digests_solves_and_writes_its_reasoning():
    inbox, ws = tmp(), tmp()
    put(inbox, "small.csv", csv_bytes(160, noise=0.15))
    put(inbox, "broken.dat", np.random.default_rng(0).bytes(500))
    org = O.Organism(ws, O.Settings(inbox=[inbox], max_cycles=3, quiet=True, candidates=3, rungs=2, probe_budget=0.2, standard_budget=0.4,
                                    doctor_every=0))
    out = org.run()
    assert 2 <= out["cycles"] <= 3 and out["tasks"]["small"]["attempts"] == out["cycles"] and out["tasks"]["small"]["best_skill"] > 0.3
    log = [json.loads(line) for line in open(os.path.join(ws, "organism", "log.jsonl"))]
    kinds = [e["kind"] for e in log]
    assert kinds[0] == "start" and "observe" in kinds and kinds.count("decision") >= out["cycles"] and kinds[-1] == "stop"
    assert any(r["reason"] for r in Library(ws).quarantined())          # the unreadable file was recorded, not fatal
    st = O.status(ws)
    assert "small" in st and "skill" in st


def run_cli(*argv):
    from zerowhite import cli
    out, code = io.StringIO(), 0
    with contextlib.redirect_stdout(out):
        try:
            cli.main([str(a) for a in argv])
        except SystemExit as e:
            code = e.code or 0
    return code, out.getvalue()


def test_cli_devour_solve_list_substrate_and_organism_commands():
    inbox, ws = tmp(), tmp()
    put(inbox, "t.csv", csv_bytes(140, noise=0.1))
    put(inbox, "x.bin", np.random.default_rng(0).bytes(700))
    code, out = run_cli("devour", inbox, "--workspace", ws)
    assert code == 0 and "1 task" in out and "1 quarantined" in out and "binary data" in out
    code, out = run_cli("devour", inbox, "--workspace", ws, "--json")                            # nothing changed: nothing to do
    j = json.loads(out)
    assert code == 0 and j["tasks"] == [] and j["counts"] == {"unchanged": 2}
    code, out = run_cli("devour", "--workspace", ws, "--list")
    assert code == 0 and "classification" in out and "x.bin" in out
    assert run_cli("devour", "--workspace", ws)[0] == 2                                           # no paths, no --list: a usage error, not a crash
    code, out = run_cli("devour", os.path.join(inbox, "nope.csv"), "--workspace", ws)
    assert code == 0 and "does not exist" in out                                                  # a missing path is reported, not raised
    ws2 = tmp()
    code, out = run_cli("devour", inbox, "--workspace", ws2, "--solve", "--budget", "0.2", "--candidates", "2", "--rungs", "2")
    assert code == 0 and "solving 't'" in out
    code, out = run_cli("substrate", "--workspace", ws, "--json")
    j = json.loads(out)
    assert code == 0 and j["plan"]["max_rows"] > 0 and "reasons" in j["plan"] and os.path.exists(os.path.join(ws, substrate.CACHE_NAME))
    assert "no organism has run" in run_cli("organism", "status", "--workspace", ws)[1]
    assert run_cli("organism", "stop", "--workspace", ws)[0] == 0
    stop = os.path.join(ws, "organism", "STOP")
    assert open(stop, encoding="utf-8").read().strip() == "stop requested"
    out = O.Organism(ws, O.Settings(inbox=[inbox], forever=True, quiet=True, poll_seconds=0.01)).run()
    assert "stop file" in out["stopped"] and not os.path.exists(stop)                              # `organism stop` really stops a running loop
    code, out = run_cli("organism", "--workspace", ws, "--inbox", inbox, "--cycles", "2", "--quiet", "--candidates", "3", "--rungs", "2",
                        "--probe-budget", "0.2")
    assert code == 0 and "cycles: 2" in out and "t " in out and "recent decisions" in out


def test_the_example_messy_inbox_script_makes_what_the_docs_say():
    import subprocess
    inbox, ws = os.path.join(tmp(), "inbox"), tmp()
    script = os.path.join(ROOT, "examples", "messy_inbox.py")
    r = subprocess.run([sys.executable, script, inbox], capture_output=True, text=True)
    assert r.returncode == 0 and "wrote 12 files" in r.stdout, r.stderr
    assert subprocess.run([sys.executable, script, inbox], capture_output=True, text=True).returncode != 0      # refuses a non-empty folder
    d = devour(inbox, ws, limits=L, allow_code=True)
    c = d.counts()
    assert c["task"] == 7 and c["quarantined"] == 3 and c["duplicate"] == 1 and c["pooled"] == 2 and c["expanded"] == 2, c
    assert sorted(Library(ws).names()) == ["catch", "clean", "corpus-notes", "download", "events", "excel_export", "regr", "utf16"]


def test_new_modules_keep_the_no_llm_no_network_no_default_guarantees():
    import ast
    banned = {"torch", "tensorflow", "jax", "transformers", "openai", "anthropic", "requests", "urllib", "socket", "http", "ftplib",
              "smtplib", "sklearn", "scipy", "pandas"}
    files = [os.path.join(ROOT, "zerowhite", "substrate.py"), os.path.join(ROOT, "zerowhite", "organism.py"), os.path.join(ROOT, "zerowhite", "cli_v6.py")]
    for root, _, fs in os.walk(os.path.join(ROOT, "zerowhite", "omnivore")):
        files += [os.path.join(root, f) for f in fs if f.endswith(".py")]
    assert len(files) >= 9
    for path in files:
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module.split(".")[0]]
            assert not (set(mods) & banned), (path, mods)
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    assert not (isinstance(t, ast.Name) and t.id.upper().endswith("_DEFAULT")), (path, t.id)
