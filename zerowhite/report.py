"""report.py - one self-contained HTML report for a workspace (or a classic run folder).

Everything is written through fileio (UTF-8), which is the fix for the Windows
`UnicodeEncodeError: 'charmap' codec can't encode character '\\u2192'` crash:
the old code used open(path, "w"), which on Windows means cp1252.
"""

import base64
import glob
import html
import os
import time

import numpy as np

from . import fileio

CSS = ("body{font:14px system-ui,sans-serif;max-width:960px;margin:2em auto;color:#222;padding:0 1em}"
       "h1,h2,h3{margin:.9em 0 .3em}.g{display:grid;grid-template-columns:1fr 1fr;gap:1em}"
       "table{border-collapse:collapse}td,th{padding:3px 12px;text-align:left;border-bottom:1px solid #eee}"
       "pre{background:#f6f6f6;padding:.6em;overflow:auto;white-space:pre-wrap}.m{color:#666}"
       "img{image-rendering:pixelated;max-width:100%;border:1px solid #ddd}")


def line_chart(vals, w=420, h=110, color="#3b6ea5", logy=False):
    v = np.array([np.nan if x is None else x for x in vals], dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return '<span class="m">not enough data</span>'
    if logy:
        v = np.log10(np.maximum(v, 1e-6))
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    hi = hi if hi > lo else lo + 1.0
    pts = " ".join(f"{10 + i * (w - 20) / max(1, len(v) - 1):.1f},{h - 10 - (x - lo) / (hi - lo) * (h - 20):.1f}"
                   for i, x in enumerate(v) if np.isfinite(x))
    label = f"{(10 ** lo if logy else lo):.3g} \u2192 {(10 ** hi if logy else hi):.3g}"
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}"><polyline fill="none" stroke="{color}" '
            f'stroke-width="2" points="{pts}"/><text x="10" y="12" font-size="10" fill="#666">{label}</text></svg>')


def _kv(d, keys):
    return " \u00b7 ".join(f"{k} {d[k]:.3f}" if isinstance(d.get(k), float) else f"{k} {d.get(k)}"
                          for k in keys if k in d and d[k] is not None)


def _classifier_section(path, title):
    d = fileio.read_json(path)
    H = d["history"]
    if not H:
        return ""
    tally = d.get("tally")
    if tally:
        moves = sorted(tally.items(), key=lambda kv: -kv[1]["won"])
        rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v['tried']}</td><td>{v['won']}</td></tr>"
                       for k, v in moves if v["tried"])
        note = f"lifetime totals over {d['generation']} generations"
    else:
        from collections import Counter
        wins = Counter(e["action"] for e in H)
        tried = Counter(t for e in H for t in e.get("tried", [e["action"]]))
        rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{tried[k]}</td><td>{wins[k]}</td></tr>"
                       for k in sorted(tried, key=lambda k: -wins[k]))
        note = f"{len(H)} generations"
    promos = [(H[i]["generation"], H[i + 1]["level"]) for i in range(len(H) - 1)
              if H[i].get("level") is not None and H[i + 1].get("level") != H[i].get("level")]
    last = H[-1]
    acc = last.get("val_acc")
    return (f"<h3>{html.escape(title)}</h3><p class=m>generation {d['generation']} \u00b7 final val_acc "
            f"{'n/a' if acc is None else f'{acc:.3f}'} \u00b7 hidden {last['hidden_sizes']} \u00b7 params {last['num_params']} "
            f"\u00b7 lr {last['opt_base_lr']:.3f}" + (f" \u00b7 promotions {promos}" if promos else "") + "</p>"
            f"<div class=g><div><b>val_loss (log)</b><br>{line_chart([e['val_loss'] for e in H], logy=True)}</div>"
            f"<div><b>val_acc</b><br>{line_chart([e['val_acc'] for e in H], color='#2a9d5c')}</div>"
            f"<div><b>parameters</b><br>{line_chart([e['num_params'] for e in H], color='#c4622d')}</div>"
            f"<div><b>learning rate</b><br>{line_chart([e['opt_base_lr'] for e in H], color='#8a4fb5', logy=True)}</div></div>"
            f"<p class=m>self-modification moves ({note})</p>"
            f"<table><tr><th>move</th><th>tried</th><th>won</th></tr>{rows}</table>")


def build(root):
    parts = [f"<!doctype html><meta charset=utf-8><title>Zero White report</title><style>{CSS}</style>",
             f"<h1>Zero White \u2014 report</h1><p class=m>{html.escape(os.path.abspath(root))} \u00b7 "
             f"generated {time.strftime('%Y-%m-%d %H:%M')}</p>"]

    parts.append("<h2>Classifiers (self-modifying)</h2>")
    shown = 0
    classic = os.path.join(root, "run_journal.json")
    if os.path.exists(classic):
        parts.append(_classifier_section(classic, "classic run"))
        shown += 1
    for jp in sorted(glob.glob(os.path.join(root, "classifier", "*", "run_journal.json"))):
        name = os.path.basename(os.path.dirname(jp))
        best = _read_best(os.path.dirname(jp))
        head = f"{name}" + (f" \u2014 test acc {best['test_acc']:.3f} (majority baseline {best['majority_baseline']:.3f})"
                            if best else "")
        parts.append(_classifier_section(jp, head))
        shown += 1
    if not shown:
        parts.append("<p class=m>none yet</p>")

    for kind, title, keys in (("generators", "Image generators (CVAE)", ("fidelity", "diversity", "elbo", "params", "steps")),
                              ("lm", "Language models", ("bpc", "uniform_bpc", "params", "steps", "chars")),
                              ("agents", "Agents", ("eval_return", "success", "updates"))):
        dirs = sorted(glob.glob(os.path.join(root, kind, "*")))
        if not dirs:
            continue
        parts.append(f"<h2>{title}</h2>")
        for d in dirs:
            b = _read_best(d)
            if not b:
                continue
            parts.append(f"<h3>{html.escape(os.path.basename(d))}</h3><p class=m>{html.escape(_kv(b, keys))}"
                         f"<br>genome {html.escape(str(b.get('genome')))}</p>")
            png = os.path.join(d, "samples.png")
            if os.path.exists(png):
                with open(png, "rb") as f:
                    parts.append('<img alt="generated samples" src="data:image/png;base64,'
                                 + base64.b64encode(f.read()).decode("ascii") + '">')
            if b.get("sample"):
                parts.append(f"<pre>{html.escape(b['sample'])}</pre>")

    jl = os.path.join(root, "journal.jsonl")
    if os.path.exists(jl):
        import json
        ev = []
        for line in fileio.read_text(jl).splitlines():
            try:
                ev.append(json.loads(line))
            except ValueError:
                pass
        parts.append(f"<h2>Journal</h2><p class=m>{len(ev)} jobs recorded; most recent first</p><table>"
                     "<tr><th>when</th><th>job</th><th>result</th></tr>")
        for e in reversed(ev[-25:]):
            res = _kv(e, ("test_acc", "fidelity", "diversity", "bpc", "eval_return", "success"))
            parts.append(f"<tr><td>{time.strftime('%m-%d %H:%M', time.localtime(e['t']))}</td>"
                         f"<td>{html.escape(str(e.get('kind')))} {html.escape(str(e.get('name', '')))}</td>"
                         f"<td>{html.escape(res)}</td></tr>")
        parts.append("</table>")
    return "\n".join(parts)


def _read_best(d):
    p = os.path.join(d, "best.json")
    return fileio.read_json(p) if os.path.exists(p) else None


def write(root, out_path=None):
    out_path = out_path or os.path.join(root, "report.html")
    fileio.write_text(out_path, build(root))
    return out_path
