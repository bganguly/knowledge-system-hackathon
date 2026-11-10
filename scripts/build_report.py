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
    cons_score = None
    speed_score = None
    rjson = report_files.get(ts)
    if rjson and Path(rjson).exists():
        with open(rjson) as f:
            rdata = json.load(f)
        final_score = rdata.get("final_score")
        qa_score = rdata.get("qa_score")
        cons_score = (rdata.get("consistency") or {}).get("score")
        speed_score = (rdata.get("speed") or {}).get("score")
    runs.append({"ts": display_ts, "questions": questions, "final_score": final_score, "qa_score": qa_score,
                 "cons_score": cons_score, "speed_score": speed_score})

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
)

LAST_TWO = {RUN_COLS - 1, RUN_COLS - 2}

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

        # Judge info: matched, missed, trap_hits, cache
        matched = row.get("matched") or []
        missed = row.get("missed") or []
        traps = row.get("trap_hits") or []
        lat = row.get("latency_s")
        cache_hit = lat is not None and lat < 0.1
        judge_parts = []
        if matched:
            judge_parts.append(f'<span class="jok">✓ {esc(", ".join(str(m) for m in matched))}</span>')
        if missed:
            judge_parts.append(f'<span class="jmiss">✗ {esc(", ".join(str(m) for m in missed))}</span>')
        if traps:
            judge_parts.append(f'<span class="jtrap">⚠ trap: {esc(", ".join(str(t) for t in traps))}</span>')
        if cache_hit:
            judge_parts.append(f'<span class="jcache">cache used</span>')
        if i == RUN_COLS - 1 and sc is not None and sc > 0:
            ans_sources = (row.get("answer") or {}).get("sources") or []
            exp_srcs = meta.get("expected_sources") or []
            exp_base = {e.split("#")[0].rstrip("/") for e in exp_srcs}
            if ans_sources:
                src_parts = []
                for si, s in enumerate(ans_sources, 1):
                    url = (s.get("url") or "")
                    path = url.replace("https://www.dupaco.com", "") or "/"
                    base = url.split("#")[0].rstrip("/")
                    tick = " ✓" if (exp_base and base in exp_base) else ""
                    src_parts.append(f"#{si}:{esc(path)}{tick}")
                judge_parts.append(
                    f'<span class="jsrc">{"<br>".join(src_parts)}</span>'
                )
        judge_html = "<br>".join(judge_parts) if judge_parts else "—"

        chunk_dbg_html = ""
        if sc is not None and sc < 1.0:
            ans_chunks = (row.get("answer") or {}).get("chunks") or []
            if ans_chunks:
                cp = []
                for ci, c in enumerate(ans_chunks, 1):
                    path = (c.get("url") or "").replace("https://www.dupaco.com", "")
                    snippet = esc(c.get("quote") or "")
                    snap = esc(c.get("snapshot_id") or "")
                    cp.append(
                        f'<div class="chunk-item"><b>{ci}. {esc(path)}</b>'
                        f' <span style="color:var(--muted)">({snap})</span>'
                        f'<pre class="chunk-pre">{snippet}</pre></div>'
                    )
                chunk_dbg_html = (
                    f'<details class="chunk-toggle">'
                    f'<summary>{len(ans_chunks)} chunk(s) fed to LLM</summary>'
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
    )

    rows_html += f'<tr>{td_meta}{td_runs}<td class="num-col">{row_num}</td></tr>\n'

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
    if i in LAST_TWO and cust_q:
        pqs = _wqa(pub_q)
        all_q = {**pub_q, **cust_q}
        n_pub = len(pub_q)
        n_all = len(all_q)
        pqs_str = f"{pqs:.4f}" if pqs is not None else "—"
        r_cons = run.get("cons_score")
        r_spd = run.get("speed_score")
        if pqs is not None and r_cons is not None and r_spd is not None:
            pfs = pqs * (0.55 + r_cons * 0.15 + r_spd * 0.15) / 0.85
            pfs_str = f"{pfs:.4f}"
        else:
            pfs_str = "—"
        inner = (
            f'{n_pub}q &nbsp; qa: {pqs_str} &nbsp; final: {pfs_str} &nbsp;'
            f'&nbsp;|&nbsp; '
            f'{n_all}q &nbsp; qa: {qs_str} &nbsp; final: {fs_str}'
        )
    elif cust_q:
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
    f'<td class="run-col wip-col avg-cell" data-run="wip" style="background:#fffbe6" colspan="2">'
    f'<span style="font-size:0.78rem">est. qa: {proj_str} '
    f'&nbsp; ({wip_tested_count}/{len(all_ids)} tested)</span>'
    f'</td>'
)

rows_html += (
    f'<tr class="avg-row">'
    f'<td class="snap-col avg-cell"></td>'
    f'<td class="wt-col avg-cell"></td>'
    f'<td class="cat-col avg-cell"></td>'
    f'<td class="q-col avg-cell" colspan="3">Summary (from report.json)</td>'
    f'{avg_cells}'
    f'<td class="num-col avg-cell"></td>'
    f'</tr>\n'
)

_TEMPLATE = (Path(__file__).parent / "report_template.html").read_text(encoding="utf-8")
html = (
    _TEMPLATE
    .replace("%%RUN_COLS%%", str(RUN_COLS))
    .replace("%%QUESTION_COUNT%%", str(len(all_ids)))
    .replace("%%HEADER_CELLS%%", header_cells)
    .replace("%%ROWS_HTML%%", rows_html)
    .replace("%%RUN_LABELS_JSON%%", json.dumps(run_labels))
)

OUT.write_text(html, encoding="utf-8")
print(f"Written: {OUT}  ({len(all_ids)} questions, {RUN_COLS} run(s))")
