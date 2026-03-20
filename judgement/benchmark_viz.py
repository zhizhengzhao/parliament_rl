"""Science Parliament — benchmark results overview HTML."""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "parliament"))
from visualize import _esc


def generate(bench_dir: str, summary: dict):
    """Read results.jsonl from bench_dir and write index.html."""
    results_path = os.path.join(bench_dir, "results.jsonl")
    records = []
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    records.sort(key=lambda r: r.get("index", 0))

    total = summary.get("total", len(records))
    correct = summary.get("correct", sum(1 for r in records if r.get("is_correct")))
    accuracy = correct / total if total > 0 else 0
    bench_name = summary.get("bench_name", "benchmark")
    timestamp = summary.get("timestamp", "")
    gpu_ids = summary.get("gpu_ids", [])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rounds_list = [r.get("rounds_completed", 0) for r in records if r.get("rounds_completed")]
    avg_rounds = sum(rounds_list) / len(rounds_list) if rounds_list else 0
    early_stopped = sum(1 for r in records if r.get("early_stopped"))
    early_pct = early_stopped / total * 100 if total > 0 else 0
    unanswered = sum(1 for r in records if r.get("parliament_answer") is None)

    rows_html = ""
    for r in records:
        idx = r.get("index", "?")
        q = _esc(r.get("question", "")[:120])
        answer = _esc(r.get("parliament_answer") or "—")
        truth = _esc(str(r.get("ground_truth") or "—"))
        rounds = r.get("rounds_completed", "?")
        early = "yes" if r.get("early_stopped") else "no"
        gpu = r.get("gpu", "?")

        is_correct = r.get("is_correct")
        if is_correct is True:
            status = '<span class="ok">correct</span>'
        elif is_correct is False:
            status = '<span class="wrong">wrong</span>'
        else:
            status = '<span class="na">N/A</span>'

        link = f'{idx}/index.html'
        rows_html += f"""<tr onclick="window.open('{link}','_blank')" style="cursor:pointer">
<td>{idx}</td><td class="q">{q}</td><td>{answer}</td><td>{truth}</td>
<td>{status}</td><td>{rounds}</td><td>{early}</td><td>{gpu}</td></tr>\n"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>{bench_name} — Results</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',-apple-system,sans-serif;background:#f4f5f9;color:#1e1e2e;line-height:1.6}}
.hdr{{background:linear-gradient(135deg,#1e1b4b,#312e81);color:#fff;padding:28px 36px 22px}}
.hdr h1{{font-size:1.4rem;font-weight:800}}.hdr h1 span{{color:#818cf8;font-weight:400}}
.stats{{display:flex;gap:24px;margin-top:14px;flex-wrap:wrap}}
.stat{{background:rgba(255,255,255,.1);border-radius:10px;padding:10px 20px}}
.stat-val{{font-size:1.5rem;font-weight:800;color:#818cf8}}.stat-lbl{{font-size:.75rem;opacity:.7}}
.wrap{{max-width:1100px;margin:24px auto;padding:0 20px}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:12px;overflow:hidden;
  box-shadow:0 2px 12px rgba(0,0,0,.06)}}
th{{background:#f8f9fc;text-align:left;padding:12px 14px;font-size:.78rem;color:#8b8da0;
  font-weight:600;text-transform:uppercase;letter-spacing:.5px;border-bottom:2px solid #e8e9f0}}
td{{padding:10px 14px;font-size:.85rem;border-bottom:1px solid #f0f1f5}}
tr:hover{{background:#f8f9ff}}
.q{{max-width:350px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.ok{{color:#16a34a;font-weight:700}}.wrong{{color:#dc2626;font-weight:700}}
.na{{color:#8b8da0}}
.foot{{text-align:center;padding:20px;font-size:.72rem;color:#8b8da0;margin-top:24px}}
</style></head><body>

<div class="hdr">
  <h1>Science <span>Parliament</span> — {_esc(bench_name)}</h1>
  <div class="stats">
    <div class="stat"><div class="stat-val">{accuracy:.1%}</div><div class="stat-lbl">Accuracy</div></div>
    <div class="stat"><div class="stat-val">{correct}/{total}</div><div class="stat-lbl">Correct</div></div>
    <div class="stat"><div class="stat-val">{len(gpu_ids)}</div><div class="stat-lbl">GPUs</div></div>
    <div class="stat"><div class="stat-val">{avg_rounds:.1f}</div><div class="stat-lbl">Avg Rounds</div></div>
    <div class="stat"><div class="stat-val">{early_pct:.0f}%</div><div class="stat-lbl">Early Stop</div></div>
    <div class="stat"><div class="stat-val">{unanswered}</div><div class="stat-lbl">Unanswered</div></div>
    <div class="stat"><div class="stat-val">{_esc(timestamp)}</div><div class="stat-lbl">Run</div></div>
  </div>
</div>

<div class="wrap">
<table>
<thead><tr>
<th>#</th><th>Question</th><th>Answer</th><th>Truth</th>
<th>Result</th><th>Rounds</th><th>Early Stop</th><th>GPU</th>
</tr></thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>

<div class="foot">Generated {now} · Click any row to view the parliament discussion</div>
</body></html>"""

    out_path = os.path.join(bench_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
