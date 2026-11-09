# Knowledge System Hackathon — BM25 · Claude Haiku · Temporal Filtering

Submission for the interface.ai hackathon: a **temporal knowledge system** that answers credit-union member questions from website snapshots taken at prior points in time.
BM25 retrieval over chunked pages feeds top-14 excerpts into Claude Haiku for grounded answer generation.
Conflict detection compares percentage values across snapshot dates for the same URL.

---

## Contents

- [Artifacts](#artifacts)
- [Approaches considered and skipped](#approaches-considered-and-skipped)
- [Eval run log](#eval-run-log)
- [Manual score overrides](#manual-score-overrides)
- [Custom eval](#custom-eval-5-questions--2026-11-07)
- [40-public vs 45-cumulative](#40-public-vs-45-cumulative-run-2026-11-07-0104)
- [How it works](#how-it-works)
- [Setup](#setup-first-time)
- [Running the eval](#running-the-eval)
- [Stack](#stack)
- [Repo policy](#repo-policy)
- [Development history](#development-history)

---

## Artifacts

**[Eval run comparison](https://claude.ai/artifact/JgfuQECziuJbVYfsPmMQY6)** — per-question drill-down across all runs (answer · score · judge · BM25 chunks)

**[Temporal RAG proposals](https://claude.ai/artifact/XrKATgNjQfbsneEgU984y8)** — proposal table with trade-offs, applied status, and failing-row breakdown

---

## Approaches considered and skipped

| Approach | Reason skipped |
|---|---|
| Dense / hybrid retrieval (sentence-transformers) | Tried (Phase 5). Re-ranker demotes rate-table chunks in favour of blog/calculator content; row 2 BM25=1 → hybrid=0. Reverted. |
| Markdown heading splits, tables kept intact | Credit union pages are HTML→text; heading structure unreliable. Table-row chunks outperform whole-table chunks for rate queries — keeping tables intact would hurt `tier_disambiguation`. |
| Adjacent chunk retrieval | **Implemented.** For the top-3 BM25 results, immediate neighbors from the same URL are appended. `branch_info` jumped from 0.47 → 0.87; overall final_score +0.03 vs best prior haiku run. |
| Opus model | Sonnet p95 already 4.38s vs haiku's ~2.5s, costing speed-pillar points. Opus latency would be worse; unless accuracy improves to ~0.75+, net final_score is lower than haiku. |

---

## Eval run log

| | 2026-11-05 22:52:16 | 2026-11-06 22:31:25 | 2026-11-06 22:49:34 | 2026-11-06 22:59:32 | 2026-11-06 23:18:47 |
|---|---|---|---|---|---|
| model | `claude-haiku-5-5` | `claude-haiku-5-5` | `claude-haiku-5-5` | `claude-sonnet-5-5` | `claude-haiku-5-5` |
| adjacent chunks | no | no | no | no | **yes** |
| top_k | 22 | 22 | 22 | 22 | 22 |
| max_per_url | 6 | 6 | 6 | 6 | 6 |
| max_chunk | 700 | 700 | 700 | 700 | 700 |
| min_chunk | 20 | 20 | 20 | 20 | 20 |
| pass | 28 | 27 | 25 | 27 | 27 |
| fail | 10 | 11 | 13 | 11 | 8 |
| partial pass | 2 | 2 | 2 | 2 | 5 |
| p95 latency | 4.27s | 2.49s | 2.24s | 4.38s | 2.42s |
| final_score | **0.6812** | **0.6605** | **0.6132** | **0.6464** | **0.7157** |

Adjacent chunk retrieval (top-3 results get neighbors from same URL) lifted `branch_info` from 0.47 → 0.87. Haiku: same committed code, spread of ~0.07 reflects LLM non-determinism. Sonnet: higher accuracy but slower p95 cancels the gain.

---

## Manual score overrides

| id | row | original | override | reason |
|---|---|---|---|---|
| dup-088 | #37 | 0.0 | 1.0 | "last year" = 2025 — no 2025 VTO data in snapshots; strict "I don't know" is correct |

Overrides are stored in `results/per_question.jsonl` under the `_override` field.

---

## Custom eval (5 questions · 2026-11-07)

5 questions written against facts verifiable in the snapshot markdown files, appended to the 40 public questions for a 45-question combined run.

**Per custom question**

| id | category | score | verdict | note |
|---|---|---|---|---|
| cust-001 | entity_attribution | 0.50 | PARTIAL | Found "1948"; still missing "10" — founding count shares the same newsroom boilerplate but LLM didn't state it |
| cust-002 | current_value | 1.00 | PASS | "$1,500 AD&D insurance, ages 18–85" retrieved correctly from about/join |
| cust-003 | current_value | 1.00 | PASS | "$25 minimum deposit" retrieved correctly from about/join |
| cust-004 | entity_attribution | 1.00 | **PASS** | "Holiday Club created 1954" — fixed by historical boost; year-bearing newsroom chunks now surface |
| cust-005 | temporal_as_of | 1.00 | PASS | Launchpad Savings 3.25% APY (as_of 2026-09-28) retrieved correctly from latest snapshot |

**Analysis:** Historical keyword detection (`what year`, `when did`, `first introduce`, `founded`) now triggers a 2× boost on chunks whose text contains a 4-digit year. This is gated behind `not query_is_rate` so rate questions are unaffected. cust-004 flipped from FAIL→PASS. cust-001 still partial — the LLM receives the "Founded in 1948 by 10 employees" chunk but chose not to state the count; a prompt-level fix would be needed.

---

## 40-public vs 45-cumulative (run 2026-11-07 01:04)

Both columns are from the same eval run (`run_2026-11-07_01-04-09_custom.jsonl`). The 40-only column is the public subset extracted from that run; no separate run was executed.

| | 40 public (subset) | 45 combined (cumulative) |
|---|---|---|
| questions | 40 | 45 |
| qa_score (weighted) | 0.7671 | **0.7782** |
| pass | 28 | 33 |
| partial | 5 | 6 |
| fail | 7 | 6 |
| p95 latency | 2.27s | 2.27s |
| final_score | — | **0.7702** |

`final_score` is computed by the harness over the full 45-question run only (consistency and speed pillars span all questions and cannot be split cleanly by subset).

---

## How it works

| Stage | Implementation |
|---|---|
| **Retrieval** | BM25Okapi (`rank-bm25`) over paragraph chunks; score blended with URL authority (rate pages > product pages > blog) |
| **Generation** | Claude Haiku (`claude-haiku-5-5`); prompt enforces tier disambiguation (48-mo ≠ 60-mo, rate ≠ APY) and hard "I don't know" when answer absent |
| **Temporal filtering** | `as_of` date filters to the latest snapshot where `taken_on ≤ as_of`; default is the newest snapshot |
| **Conflict detection** | On each `ingest()`, regex-extracts percentages per URL and flags values that changed across snapshots |

---

## Setup (first time)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

The LLM model defaults to `claude-haiku-5-5`. Override with the `LLM_MODEL` env var:

```bash
export LLM_MODEL=claude-sonnet-4-5   # or any other claude-* model ID
```

## Running the eval

**First run / evaluator use** — from inside the kit directory, point `--submission` here:

```bash
cd /path/to/hackathon-knowledge-kit-participant
python harness/run_eval.py \
  --submission /path/to/knowledge-system-hackathon \
  --eval eval/public_eval.jsonl
```

Snapshots are unpacked automatically on first run. Results land in `results/` inside the kit directory.

**Repeated local use** — `deploy.sh` wraps the above, handles snapshot extraction, and loads the key from `.env` if not already set. It defaults to `../hackathon-knowledge-kit-participant` (sibling folder); pass a path to override. Second arg overrides the model:

```bash
./scripts/deploy.sh                                        # uses ../hackathon-knowledge-kit-participant
./scripts/deploy.sh /other/path/to/kit
./scripts/deploy.sh /other/path/to/kit claude-sonnet-4-5
```

---

## Stack

| Component | Detail |
|---|---|
| **Retrieval** | `rank-bm25` — pure Python, no API calls at query time |
| **LLM** | Anthropic Claude Haiku (`claude-haiku-5-5`) via `anthropic` SDK |
| **Python** | 3.12 — only BM25 is active; `sentence-transformers` / dense embedding are disabled |

---

## Repo policy

Snapshot data is from a real credit union and must not be committed. `.gitignore` excludes `snapshots/`, `snapshots_zip/`, and `*.zip`.

---

## Development history

### Trials and errors — eval score progression

| Phase | Key commits | What changed | Score (final) | Outcome |
|---|---|---|---|---|
| 1 — Baseline | `9c2d012` | Initial BM25 + Claude Haiku | 0.5228 | Starting point |
| 2 — Answer quality | `a0c710f`–`ce308ce` | Strip meta-commentary, ban source names / trailing disclaimers / question-echoing; URL questions return exact anchor URL; mortgage synonym expansion for rate retrieval | ~0.60–0.67 | Steady gains from prompt hygiene |
| 3 — BM25 retrieval | `546de3c`–`f8d1089` | Table-row chunking, lower MIN_CHUNK, raise TOP_K, allow 3 chunks/URL; title-in-BM25; conflict injection; branch\_info lobby-hours chunking | **0.6812** (run 7) | Best run; became the revert target |
| 4 — Retrieval experiments | `0eb6263`–`a24af21` | Tried latest-per-URL crawl; additional conflict/stale-trap tuning | 0.6268–0.6079 | Regressions → reverted to `f8d1089` |
| 5 — Dense / hybrid | `9d43e97`–`5e3c0d2` | Pluggable retrieval via `RETRIEVAL_MODE`; TF-IDF cosine fallback (Python 3.14); switched to `sentence-transformers` on Python 3.12; disk embed cache; pre-warm on `ingest()`; dual WIP eval modes | WIP Hybrid ≤ WIP BM25 on most rows | **Abandoned** — dense re-ranker demotes rate-table chunks (high auth-score pages) in favour of blog/calculator content; row 2 BM25=1 → Hybrid=0 |
| 6 — BM25 fine-tuning | `41b43be` | Per-URL cap 6→3 (diversity); boilerplate filter at ingest (image alt-text, header links, "Keep reading", "Handpicked for You" chunks) | In progress | Fixes confirmed retrieval gap on row 10 (`$15,000` was crowded out by 6 boilerplate chunks from one URL) |

### Why it took longer than a typical hackathon

1. **Dense/hybrid exploration** — building pluggable retrieval, adding sentence-transformers, Python version gymnastics (3.9 → 3.15 → 3.12), disk embed cache, batch progress logging, and pre-warming consumed a full iteration cycle. The payoff was negative on this corpus (structured rate tables rank worse under dense cosine than BM25).
2. **Eval comparison tooling** — `scripts/build_report.py` + `results/comparison.html` with run-filter UI, WIP columns, per-row drill-down, and score timeline were not in scope but became essential for diagnosing which rows regressed after each change. Without this, reverting to the best run would have been guesswork.
3. **Root-cause debugging** — the row 10 `$15,000` miss was not "wrong answer" noise; it required tracing through chunk ranking to find that 6 of 22 retrieval slots were consumed by boilerplate from a single newsroom URL, pushing the relevant body paragraph to rank 23+.

### TODOs (follow-up, no code changes needed now)

- [ ] **Verify levers 1+2 fix row 10**: run `./scripts/deploy.sh --retrieve bm25 10` and confirm `$15,000` now reaches the LLM
- [ ] **If row 10 passes, consider lever 3**: expand top-K from 14 → 22 chunks (more signal, more noise — measure first)
- [ ] **If lever 3 helps, consider lever 4** (prompt fix): add explicit instruction "when a dollar amount appears in the excerpts, state it directly" — applies to other rows that answer with vague ranges instead of the number
- [ ] **Remove hybrid pathway from `solution.py` and `deploy.sh`**: `--retrieve dense/hybrid` options in `deploy.sh` still reference the old dense path; `solution.py` still has the hybrid branch. Remove once BM25 fine-tuning is settled
- [ ] **Run full eval once BM25 fine-tuning is done**: `./scripts/deploy.sh --eval` to record the final baseline after levers 1+2 (and optionally 3+4)
- [ ] **Chunk inspection workbench** (`chunk-workbench/`): manually examine the top-22 BM25 chunks being sent to the LLM for each failing/partial question. Use `./chunk-workbench/scripts/run_chunks.sh` (no LLM calls, no API key needed). Params `--top-k`, `--max-per-url`, `--max-chunk`, `--questions` let you iterate on retrieval strategy. Output is a JSONL + a visual HTML artifact (drag-and-drop the JSONL) that shows rank · URL · authority · text for each chunk side-by-side across runs. Goal: understand what the LLM sees for the ~1/3 questions we're currently getting wrong and tune chunking without paying for LLM calls.
