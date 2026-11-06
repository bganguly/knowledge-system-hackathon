"""
Build an HTML comparison report from all archived eval runs.
Usage: python3 scripts/build_report.py
Writes: results/comparison.html
"""
from __future__ import annotations
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "results" / "runs"
EVAL_FILE = REPO.parent / "hackathon-knowledge-kit-participant" / "eval" / "public_eval.jsonl"
WIP_FILE = REPO / "results" / "wip.jsonl"
OUT = REPO / "results" / "comparison.html"

eval_meta: dict[str, dict] = {}
if EVAL_FILE.exists():
    with open(EVAL_FILE) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                eval_meta[r["id"]] = r

run_files = sorted(RUNS_DIR.glob("run_*.jsonl"))
report_files = {
    re.sub(r"^run_", "", p.stem): RUNS_DIR / f"report_{re.sub(r'^run_', '', p.stem)}.json"
    for p in run_files
}

runs: list[dict] = []
for rp in run_files:
    ts = re.sub(r"^run_", "", rp.stem)
    display_ts = ts.replace("_", " ")
    questions: dict[str, dict] = {}
    with open(rp) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                questions[row["id"]] = row
    final_score = None
    rjson = report_files.get(ts)
    if rjson and Path(rjson).exists():
        with open(rjson) as f:
            rdata = json.load(f)
        final_score = rdata.get("final_score")
    qa_score = None
    if rjson and Path(rjson).exists():
        qa_score = rdata.get("qa_score")
    runs.append({"ts": display_ts, "questions": questions, "final_score": final_score, "qa_score": qa_score})

if not runs:
    print("No run files found in", RUNS_DIR)
    raise SystemExit(1)

# Load WIP questions (overrides last-run answers for tested questions)
wip_questions: dict[str, dict] = {}
if WIP_FILE.exists():
    with open(WIP_FILE) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                wip_questions[r["id"]] = r

all_ids = sorted(
    {qid for run in runs for qid in run["questions"]},
    key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0,
)

def esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

RUN_COLS = len(runs)
HAS_WIP = bool(wip_questions)

header_cells = ""
run_labels = []
for ri, run in enumerate(runs):
    fs = run["final_score"]
    qs = run.get("qa_score")
    fs_str = f"{fs:.4f}" if fs is not None else "—"
    qs_str = f"{qs:.4f}" if qs is not None else "—"
    header_cells += (
        f'<th class="run-col" data-run="{ri}">'
        f'<div class="run-ts">{esc(run["ts"])}</div>'
        f'<div class="run-fs">final: {fs_str} &nbsp; debug_qa: {qs_str}</div>'
        f'</th>'
        f'<th class="score-col" data-run="{ri}">score</th>'
    )
    run_labels.append(f"{run['ts']} ({fs_str})")

# WIP header (always last column)
wip_header = (
    '<th class="run-col wip-col" data-run="wip">'
    '<div class="run-ts">WIP</div>'
    '<div class="run-fs">token-check score (est.)</div>'
    '</th>'
    '<th class="score-col wip-col" data-run="wip">est.</th>'
) if HAS_WIP else ""

run_score_sums = [0.0] * RUN_COLS
run_score_counts = [0] * RUN_COLS

# WIP projected score: use wip score for tested questions, last run score for rest
last_run_questions = runs[-1]["questions"] if runs else {}
wip_total_weight = 0.0
wip_weighted_score = 0.0

rows_html = ""
for row_num, qid in enumerate(all_ids, 1):
    meta = eval_meta.get(qid, {})
    question = esc(meta.get("question", qid))
    expected = esc(meta.get("expected_answer", ""))

    must_include = meta.get("must_include") or []
    must_text = ", ".join(esc(str(m)) for m in must_include)

    # collect snapshot ids from first run that has this question
    snap_ids_text = ""
    for run in runs:
        row = run["questions"].get(qid)
        if row:
            sources = (row.get("answer") or {}).get("sources") or []
            snap_ids = sorted({s.get("snapshot_id", "") for s in sources if s.get("snapshot_id")})
            snap_ids_text = ", ".join(snap_ids)
            break

    weight = meta.get("weight", 1) or 1

    td_meta = (
        f'<td class="num-col">{row_num}</td>'
        f'<td class="snap-col">{esc(snap_ids_text)}</td>'
        f'<td class="wt-col">{weight}</td>'
        f'<td class="q-col">{question}</td>'
        f'<td class="exp-col">{expected}</td>'
        f'<td class="must-col">{must_text}</td>'
    )

    td_runs = ""
    for i, run in enumerate(runs):
        row = run["questions"].get(qid)
        if row is None:
            td_runs += f'<td class="run-col" data-run="{i}">—</td><td class="score-col" data-run="{i}">—</td>'
            continue
        ans = row.get("answer") or {}
        atext = esc((ans.get("text") or "")[:300])
        sc = row.get("score")
        sc_str = f"{sc:.2f}" if sc is not None else "—"
        if sc is not None:
            run_score_sums[i] += sc
            run_score_counts[i] += 1

        td_runs += (
            f'<td class="run-col ans-col" data-run="{i}">'
            f'<div class="ans-text">{atext}</div>'
            f'</td>'
            f'<td class="score-col" data-run="{i}">{sc_str}</td>'
        )

    # WIP column
    td_wip = ""
    if HAS_WIP:
        if qid in wip_questions:
            wrow = wip_questions[qid]
            wans = wrow.get("answer") or {}
            watext = esc((wans.get("text") or "")[:300])
            wsc = wrow.get("score")
            wsc_str = f"{wsc:.2f}" if wsc is not None else "?"
            wip_src = "wip"
        else:
            # fall back to last run
            lrow = last_run_questions.get(qid) or {}
            lans = lrow.get("answer") or {}
            watext = esc((lans.get("text") or "")[:300])
            wsc = lrow.get("score")
            wsc_str = f"{wsc:.2f}" if wsc is not None else "—"
            wip_src = "last"

        # accumulate projected score
        wip_total_weight += weight
        wip_weighted_score += (wsc or 0.0) * weight

        highlight = ' style="background:#fffbe6"' if wip_src == "wip" else ""
        td_wip = (
            f'<td class="run-col wip-col ans-col" data-run="wip"{highlight}>'
            f'<div class="ans-text">{watext}</div>'
            f'</td>'
            f'<td class="score-col wip-col" data-run="wip"{highlight}>{wsc_str}</td>'
        )

    rows_html += f'<tr>{td_meta}{td_runs}{td_wip}</tr>\n'

# WIP projected qa_score
wip_proj = (wip_weighted_score / wip_total_weight) if wip_total_weight > 0 and HAS_WIP else None
wip_tested_count = len(wip_questions)

avg_cells = ""
for i, run in enumerate(runs):
    qs = run.get("qa_score")
    fs = run.get("final_score")
    qs_str = f"{qs:.4f}" if qs is not None else "—"
    fs_str = f"{fs:.4f}" if fs is not None else "—"
    avg_cells += (
        f'<td class="run-col avg-cell" data-run="{i}">'
        f'<span style="font-size:0.78rem">debug_qa: {qs_str} &nbsp; final: {fs_str}</span>'
        f'</td>'
        f'<td class="score-col avg-cell" data-run="{i}"></td>'
    )

wip_avg_cell = ""
if HAS_WIP:
    proj_str = f"{wip_proj:.4f}" if wip_proj is not None else "—"
    wip_avg_cell = (
        f'<td class="run-col wip-col avg-cell" data-run="wip" style="background:#fffbe6">'
        f'<span style="font-size:0.78rem">est. debug_qa: {proj_str} '
        f'&nbsp; ({wip_tested_count}/{len(all_ids)} tested)</span>'
        f'</td>'
        f'<td class="score-col wip-col avg-cell" data-run="wip" style="background:#fffbe6"></td>'
    )

rows_html += (
    f'<tr class="avg-row">'
    f'<td class="num-col avg-cell"></td>'
    f'<td class="snap-col avg-cell"></td>'
    f'<td class="wt-col avg-cell"></td>'
    f'<td class="q-col avg-cell" colspan="3">Summary (from report.json)</td>'
    f'{avg_cells}{wip_avg_cell}'
    f'</tr>\n'
)

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval Comparison</title>
<style>
:root {{
  --bg: #fafafa; --fg: #111; --border: #e4e4e7;
  --muted: #71717a; --code-bg: #f4f4f5;
  font-family: system-ui, sans-serif;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #18181b; --fg: #e4e4e7; --border: #3f3f46;
    --muted: #a1a1aa; --code-bg: #27272a;
    color-scheme: dark;
  }}
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--fg); margin: 0; padding: 16px; font-size: 13px; font-weight: 400; }}
h1 {{ font-size: 1rem; margin: 0 0 4px; }}
.sub {{ color: var(--muted); font-size: 0.78rem; margin-bottom: 12px; }}
.wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: max-content; min-width: 100%; }}
th, td {{ padding: 8px 10px; border: 1px solid var(--border); vertical-align: top; font-weight: 400; }}
thead th {{ background: var(--code-bg); font-size: 0.72rem; text-transform: uppercase;
            letter-spacing: .05em; color: var(--muted); white-space: nowrap; font-weight: 400; }}
.run-ts {{ font-size: 0.78rem; white-space: nowrap; color: var(--muted); }}
.run-fs {{ font-size: 0.78rem; margin-top: 2px; color: var(--muted); }}
.num-col {{ min-width: 32px; max-width: 36px; text-align: center; color: var(--muted); font-size: 0.78rem; }}
.snap-col {{ min-width: 90px; max-width: 120px; font-size: 0.78rem; color: var(--muted); font-family: monospace; }}
.wt-col {{ min-width: 32px; max-width: 42px; text-align: center; font-size: 0.78rem; color: var(--muted); }}
.q-col {{ min-width: 165px; max-width: 210px; font-size: 0.78rem; color: var(--muted); }}
.exp-col {{ min-width: 135px; max-width: 195px; font-size: 0.78rem; color: var(--muted); }}
.must-col {{ min-width: 100px; max-width: 160px; font-size: 0.78rem; color: var(--muted); }}
.run-col {{ min-width: 260px; max-width: 340px; font-size: 0.78rem; color: var(--muted); }}
.score-col {{ min-width: 56px; max-width: 56px; text-align: center; font-size: 0.78rem; color: var(--muted); }}
.ans-col {{ font-size: 0.78rem; color: var(--muted); }}
.ans-text {{ max-height: 80px; overflow: hidden; line-height: 1.45; }}
.avg-row td {{ border-top: 2px solid var(--border); font-size: 0.85rem; }}
.avg-cell {{ background: var(--code-bg); }}
.run-filter {{ display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-bottom:10px; }}
.run-filter button {{ font-size:0.75rem; padding:3px 8px; border:1px solid var(--border);
  background:var(--code-bg); color:var(--fg); border-radius:4px; cursor:pointer; white-space:nowrap; }}
.run-filter button:hover {{ background:var(--border); }}
.run-filter .sep {{ width:1px; height:18px; background:var(--border); margin:0 2px; }}
.run-cb-row {{ display:flex; flex-wrap:wrap; gap:6px; align-items:center; }}
.run-cb-label {{ font-size:0.75rem; color:var(--muted); display:flex; align-items:center; gap:3px;
  cursor:pointer; padding:2px 6px; border:1px solid var(--border); border-radius:4px;
  background:var(--code-bg); white-space:nowrap; }}
.run-cb-label input {{ margin:0; cursor:pointer; }}
[data-hidden] {{ display:none; }}
.wip-col {{ border-left: 2px solid #f59e0b !important; }}
</style>
</head>
<body>
<h1>Eval Run Comparison</h1>
<p class="sub">{RUN_COLS} run(s) · {len(all_ids)} questions</p>
<div class="run-filter">
  <button onclick="showAll()">All</button>
  <button onclick="showLatest()">Latest only</button>
  <div class="sep"></div>
  <div class="run-cb-row" id="cbRow"></div>
</div>
<div class="wrap">
<table>
<thead>
  <tr>
    <th class="num-col">#</th>
    <th class="snap-col">snapshot</th>
    <th class="wt-col">wt</th>
    <th>Question</th>
    <th>Expected Answer</th>
    <th class="must-col">must_include</th>
    {header_cells}
    {wip_header}
  </tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>
<script>
const RUN_COUNT = {RUN_COLS};
const RUN_LABELS = {json.dumps(run_labels)};

function setCols(visSet) {{
  for (let i = 0; i < RUN_COUNT; i++) {{
    const show = visSet.has(i);
    document.querySelectorAll('[data-run="' + i + '"]').forEach(el => {{
      if (show) el.removeAttribute('data-hidden');
      else el.setAttribute('data-hidden', '');
    }});
    const cb = document.getElementById('cb' + i);
    if (cb) cb.checked = show;
  }}
}}

function showAll() {{ setCols(new Set([...Array(RUN_COUNT).keys()])); }}
function showLatest() {{ setCols(new Set([RUN_COUNT - 1])); }}

function buildCheckboxes() {{
  const row = document.getElementById('cbRow');
  for (let i = 0; i < RUN_COUNT; i++) {{
    const lbl = document.createElement('label');
    lbl.className = 'run-cb-label';
    lbl.innerHTML = '<input type="checkbox" id="cb' + i + '" checked onchange="toggleRun(' + i + ')"> ' + RUN_LABELS[i];
    row.appendChild(lbl);
  }}
}}

function toggleRun(i) {{
  const cb = document.getElementById('cb' + i);
  document.querySelectorAll('[data-run="' + i + '"]').forEach(el => {{
    if (cb.checked) el.removeAttribute('data-hidden');
    else el.setAttribute('data-hidden', '');
  }});
}}

buildCheckboxes();
</script>
</body>
</html>
"""

OUT.write_text(html, encoding="utf-8")
print(f"Written: {OUT}  ({len(all_ids)} questions, {RUN_COLS} run(s))")
