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
CUSTOM_EVAL_FILE = REPO / "eval" / "custom_eval.jsonl"
CHUNK_DEBUG_FILE = REPO / "results" / "chunks_debug.json"
WIP_FILE = REPO / "results" / "wip_bm25.jsonl"
OUT = REPO / "results" / "comparison.html"

eval_meta: dict[str, dict] = {}
if EVAL_FILE.exists():
    with open(EVAL_FILE) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                eval_meta[r["id"]] = r
if CUSTOM_EVAL_FILE.exists():
    with open(CUSTOM_EVAL_FILE) as f:
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
    display_ts = ts.replace("_", " ").replace(" custom", "")
    questions: dict[str, dict] = {}
    with open(rp) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                questions[row["id"]] = row
    final_score = None
    qa_score = None
    rjson = report_files.get(ts)
    if rjson and Path(rjson).exists():
        with open(rjson) as f:
            rdata = json.load(f)
        final_score = rdata.get("final_score")
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

eval_order: list[str] = []
if EVAL_FILE.exists():
    with open(EVAL_FILE) as f:
        for line in f:
            if line.strip():
                eval_order.append(json.loads(line)["id"])
if CUSTOM_EVAL_FILE.exists():
    with open(CUSTOM_EVAL_FILE) as f:
        for line in f:
            if line.strip():
                eval_order.append(json.loads(line)["id"])
all_ids = [qid for qid in eval_order if any(qid in run["questions"] for run in runs)]

chunk_debug: dict[str, list] = {}
if CHUNK_DEBUG_FILE.exists():
    try:
        chunk_debug = json.loads(CHUNK_DEBUG_FILE.read_text())
    except Exception:
        pass

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
        f'<div class="run-fs">final: {fs_str} &nbsp; qa: {qs_str}</div>'
        f'</th>'
        f'<th class="score-col" data-run="{ri}">score</th>'
        f'<th class="judge-col" data-run="{ri}">judge</th>'
    )
    run_labels.append(f"{run['ts']} ({fs_str})")

# WIP header
_wip_latest = max((r.get("ts", "") for r in wip_questions.values()), default="") if wip_questions else ""
wip_header = (
    '<th class="run-col wip-col" data-run="wip">'
    f'<div class="run-ts">WIP &nbsp;<span style="font-weight:400">({len(wip_questions)} tested)</span></div>'
    f'<div class="run-fs">est. token-check · last update: {esc(_wip_latest)}</div>'
    '</th>'
    '<th class="score-col wip-col" data-run="wip">est.</th>'
    '<th class="chunks-col wip-col" data-run="wip">BM25 Chunks</th>'
)

run_score_sums = [0.0] * RUN_COLS
run_score_counts = [0] * RUN_COLS

last_run_questions = runs[-1]["questions"] if runs else {}
wip_total_weight = 0.0
wip_weighted_score = 0.0

FLAGGED_ROWS = {10, 11, 13, 22, 23, 26, 31, 33, 34, 38}

rows_html = ""
for row_num, qid in enumerate(all_ids, 1):
    meta = eval_meta.get(qid, {})
    question = esc(meta.get("question", qid))
    expected = esc(meta.get("expected_answer", ""))
    category = esc(meta.get("category", ""))

    must_include = meta.get("must_include") or []
    must_text = ", ".join(esc(str(m)) for m in must_include)

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
        f'<td class="cat-col">{category}</td>'
        f'<td class="q-col">{question}</td>'
        f'<td class="exp-col">{expected}</td>'
        f'<td class="must-col">{must_text}</td>'
    )

    td_runs = ""
    for i, run in enumerate(runs):
        row = run["questions"].get(qid)
        if row is None:
            td_runs += (
                f'<td class="run-col" data-run="{i}">—</td>'
                f'<td class="score-col" data-run="{i}">—</td>'
                f'<td class="judge-col" data-run="{i}">—</td>'
            )
            continue
        ans = row.get("answer") or {}
        atext = esc(ans.get("text") or "")
        sc = row.get("score")
        sc_str = f"{sc:.2f}" if sc is not None else "—"
        if sc is not None:
            run_score_sums[i] += sc
            run_score_counts[i] += 1

        # Judge info: matched, missed, trap_hits, latency
        matched = row.get("matched") or []
        missed = row.get("missed") or []
        traps = row.get("trap_hits") or []
        lat = row.get("latency_s")
        lat_str = f"{lat:.1f}s" if lat is not None else ""
        judge_parts = []
        if matched:
            judge_parts.append(f'<span class="jok">✓ {esc(", ".join(str(m) for m in matched))}</span>')
        if missed:
            judge_parts.append(f'<span class="jmiss">✗ {esc(", ".join(str(m) for m in missed))}</span>')
        if traps:
            judge_parts.append(f'<span class="jtrap">⚠ trap: {esc(", ".join(str(t) for t in traps))}</span>')
        if lat_str:
            judge_parts.append(f'<span class="jlat">{lat_str}</span>')
        judge_html = "<br>".join(judge_parts) if judge_parts else "—"

        chunk_dbg_html = ""
        if i == RUN_COLS - 1 and sc is not None and sc < 1.0:
            dbg_key = meta.get("question", qid) + "|||" + (meta.get("as_of") or "")
            dbg_chunks = chunk_debug.get(dbg_key, [])
            if dbg_chunks:
                cp = []
                for ci, c in enumerate(dbg_chunks, 1):
                    path = (c.get("url") or "").replace("https://www.dupaco.com", "")
                    snippet = esc(c.get("text") or "")
                    auth = c.get("authority", "")
                    cp.append(
                        f'<div class="chunk-item"><b>{ci}. [{auth}] {esc(path)}</b>'
                        f'<pre class="chunk-pre">{snippet}</pre></div>'
                    )
                chunk_dbg_html = (
                    f'<details class="chunk-toggle">'
                    f'<summary>{len(dbg_chunks)} chunk(s) retrieved</summary>'
                    f'<div class="chunk-body">{"".join(cp)}</div>'
                    f'</details>'
                )

        td_runs += (
            f'<td class="run-col ans-col" data-run="{i}">'
            f'<div class="ans-text">{atext}</div>'
            f'{chunk_dbg_html}'
            f'</td>'
            f'<td class="score-col" data-run="{i}">{sc_str}</td>'
            f'<td class="judge-col" data-run="{i}">{judge_html}</td>'
        )

    # WIP column
    if qid in wip_questions:
        wrow = wip_questions[qid]
        wans = wrow.get("answer") or {}
        watext = esc(wans.get("text") or "")
        wsc = wrow.get("score")
        wsc_str = f"{wsc:.2f}" if wsc is not None else "?"
        wip_src = "wip"
        chunks = wrow.get("chunks") or []
        chunk_parts = []
        for ci, c in enumerate(chunks, 1):
            path = (c.get("url") or "").replace("https://www.dupaco.com", "")
            snippet = esc(c.get("text") or "")
            auth = c.get("authority", "")
            chunk_parts.append(
                f'<div class="chunk-item"><b>{ci}. [{auth}] {esc(path)}</b>'
                f'<pre class="chunk-pre">{snippet}</pre></div>'
            )
        chunks_inner = "".join(chunk_parts)
        if chunks_inner:
            chunks_html = (
                f'<details class="chunk-toggle">'
                f'<summary>{len(chunks)} chunk(s)</summary>'
                f'<div class="chunk-body">{chunks_inner}</div>'
                f'</details>'
            )
        else:
            chunks_html = ""
    else:
        lrow = last_run_questions.get(qid) or {}
        lans = lrow.get("answer") or {}
        watext = esc(lans.get("text") or "")
        wsc = lrow.get("score")
        wsc_str = f"{wsc:.2f}" if wsc is not None else "—"
        wip_src = "last"
        chunks_html = ""

    if wsc is not None:
        wip_total_weight += weight
        wip_weighted_score += wsc * weight

    is_zero = wip_src == "wip" and wsc == 0.0
    highlight = ' style="background:#fee2e2"' if is_zero else ""
    td_wip = (
        f'<td class="run-col wip-col ans-col" data-run="wip"{highlight}>'
        f'<div class="ans-text">{watext}</div>'
        f'</td>'
        f'<td class="score-col wip-col" data-run="wip"{highlight}>{wsc_str}</td>'
        f'<td class="chunks-col wip-col" data-run="wip">{chunks_html}</td>'
    )

    rows_html += f'<tr>{td_meta}{td_runs}{td_wip}</tr>\n'

# WIP projected qa_score
wip_proj = (wip_weighted_score / wip_total_weight) if wip_total_weight > 0 and HAS_WIP else None
wip_tested_count = len(wip_questions)

def _wqa(qmap: dict) -> float | None:
    ws, wt = 0.0, 0.0
    for qid, row in qmap.items():
        sc = row.get("score")
        if sc is not None:
            w = (eval_meta.get(qid) or {}).get("weight", 1) or 1
            ws += sc * w
            wt += w
    return ws / wt if wt > 0 else None

avg_cells = ""
for i, run in enumerate(runs):
    qs = run.get("qa_score")
    fs = run.get("final_score")
    qs_str = f"{qs:.4f}" if qs is not None else "—"
    fs_str = f"{fs:.4f}" if fs is not None else "—"
    pub_q = {qid: r for qid, r in run["questions"].items() if qid.startswith("dup-")}
    cust_q = {qid: r for qid, r in run["questions"].items() if qid.startswith("cust-")}
    if cust_q:
        pqs = _wqa(pub_q)
        cqs = _wqa(cust_q)
        n_pub, n_cust = len(pub_q), len(cust_q)
        pqs_str = f"{pqs:.4f}" if pqs is not None else "—"
        cqs_str = f"{cqs:.4f}" if cqs is not None else "—"
        inner = (
            f'{n_pub}q qa: {pqs_str} &nbsp;·&nbsp; '
            f'{n_cust}q qa: {cqs_str} &nbsp;·&nbsp; '
            f'{n_pub+n_cust}q qa: {qs_str} &nbsp; final: {fs_str}'
        )
    else:
        inner = f'qa: {qs_str} &nbsp; final: {fs_str}'
    avg_cells += (
        f'<td class="run-col avg-cell" data-run="{i}" colspan="3">'
        f'<span style="font-size:0.78rem">{inner}</span>'
        f'</td>'
    )

proj_str = f"{wip_proj:.4f}" if wip_proj is not None else "—"
wip_avg_cell = (
    f'<td class="run-col wip-col avg-cell" data-run="wip" style="background:#fffbe6" colspan="3">'
    f'<span style="font-size:0.78rem">est. qa: {proj_str} '
    f'&nbsp; ({wip_tested_count}/{len(all_ids)} tested)</span>'
    f'</td>'
)

rows_html += (
    f'<tr class="avg-row">'
    f'<td class="num-col avg-cell"></td>'
    f'<td class="snap-col avg-cell"></td>'
    f'<td class="wt-col avg-cell"></td>'
    f'<td class="cat-col avg-cell"></td>'
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
.cat-col {{ min-width: 110px; max-width: 150px; font-size: 0.72rem; color: var(--muted); }}
.q-col {{ min-width: 165px; max-width: 220px; font-size: 0.78rem; }}
.exp-col {{ min-width: 135px; max-width: 200px; font-size: 0.78rem; color: var(--muted); }}
.must-col {{ min-width: 100px; max-width: 160px; font-size: 0.78rem; color: var(--muted); }}
.run-col {{ min-width: 280px; max-width: 400px; font-size: 0.78rem; }}
.score-col {{ min-width: 52px; max-width: 52px; text-align: center; font-size: 0.78rem; color: var(--muted); }}
.judge-col {{ min-width: 160px; max-width: 220px; font-size: 0.72rem; vertical-align: top; }}
.ans-col {{ font-size: 0.78rem; }}
.ans-text {{ line-height: 1.5; white-space: pre-wrap; word-break: break-word; }}
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
.chunks-col {{ min-width: 220px; max-width: 500px; font-size: 0.72rem; color: var(--muted); vertical-align: top; }}
.chunk-toggle summary {{ cursor:pointer; font-size:0.72rem; color:var(--muted); padding:2px 0; }}
.chunk-toggle summary:hover {{ color:var(--fg); }}
.chunk-body {{ margin-top:4px; }}
.chunk-item {{ margin-bottom: 8px; border-left: 2px solid var(--border); padding-left: 6px; }}
.chunk-pre {{ white-space: pre-wrap; word-break: break-word; font-size: 0.68rem; margin: 2px 0 0; line-height: 1.4; }}
.jok {{ color: #16a34a; display:block; font-size:0.72rem; }}
.jmiss {{ color: #dc2626; display:block; font-size:0.72rem; }}
.jtrap {{ color: #d97706; display:block; font-size:0.72rem; }}
.jlat {{ color: var(--muted); display:block; font-size:0.70rem; margin-top:2px; }}
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
    <th class="num-col" id="num-th" onclick="sortByNum()" style="cursor:pointer" title="Sort by row number">#</th>
    <th class="snap-col">snapshot</th>
    <th class="wt-col" id="wt-th" onclick="sortByWt()" style="cursor:pointer" title="Sort by weight">wt</th>
    <th class="cat-col">category</th>
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
showLatest();

let _wtAsc = null, _numAsc = null;
function _sortTable(colClass, asc) {{
  const tbody = document.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr:not(.avg-row)'));
  const avgRow = tbody.querySelector('.avg-row');
  rows.sort((a, b) => {{
    const av = parseFloat(a.querySelector('.' + colClass)?.textContent) || 0;
    const bv = parseFloat(b.querySelector('.' + colClass)?.textContent) || 0;
    return asc ? av - bv : bv - av;
  }});
  rows.forEach(r => tbody.insertBefore(r, avgRow));
}}
function sortByWt() {{
  _wtAsc = _wtAsc === true ? false : true; _numAsc = null;
  _sortTable('wt-col', _wtAsc);
  document.getElementById('wt-th').textContent = 'wt ' + (_wtAsc ? '▲' : '▼');
  document.getElementById('num-th').textContent = '#';
}}
function sortByNum() {{
  _numAsc = _numAsc === true ? false : true; _wtAsc = null;
  _sortTable('num-col', _numAsc);
  document.getElementById('num-th').textContent = '# ' + (_numAsc ? '▲' : '▼');
  document.getElementById('wt-th').textContent = 'wt';
}}
</script>
</body>
</html>
"""

OUT.write_text(html, encoding="utf-8")
print(f"Written: {OUT}  ({len(all_ids)} questions, {RUN_COLS} run(s))")
