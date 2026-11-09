# Knowledge System Hackathon — BM25 · Claude Haiku · Temporal Filtering

Submission for the interface.ai hackathon: a **temporal knowledge system** that answers credit-union member questions from website snapshots taken at prior points in time.
BM25 retrieval over chunked pages feeds top-14 excerpts into Claude Haiku for grounded answer generation.
Conflict detection compares percentage values across snapshot dates for the same URL.

---

## Contents

- [Setup](#setup-first-time)
- [Running the eval](#running-the-eval)
- [Current eval status](#current-eval-status)
- [Artifacts](#artifacts)
- [Approaches considered and skipped](#approaches-considered-and-skipped)
- [Sample eval run log](#sample-eval-run-log)
- [Manual score overrides](#manual-score-overrides)
- [Sample custom eval](#sample-custom-eval)
- [How it works](#how-it-works)
- [Stack](#stack)
- [Repo policy](#repo-policy)
- [Development history](#development-history)

---

## Current eval status

Latest run: `run_2026-11-08_22-06-45_custom` · 56 questions (40 provided + 16 custom)

| Subset | Score 1.0 | Score 0.85 | Score 0.50 | Score 0.0 | qa\_score | final\_score |
|---|---|---|---|---|---|---|
| 40 provided | 32 | 2 | 1 | 5 | 0.8434 | 0.8346 |
| 16 custom | 10 | 0 | 2 | 4 | 0.6538 | 0.6470 |
| **56 total** | **42** | **2** | **3** | **9** | **0.7994** | **0.7911** |

Consistency: 1.00 · Speed: 0.94 (p95 3.1s)

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

## Sample eval run log

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

## Sample custom eval

Sample of 5 of 16 questions written against facts verifiable in the snapshot markdown files.

| # | Question | Expected answer | Must include |
|---|---|---|---|
| 1 | In what year was Dupaco Credit Union founded, and how many people started it? | Dupaco was founded in 1948 by 10 meatpacking employees who each pooled $5. | 1948, 10 |
| 2 | What free insurance benefit does Dupaco automatically provide to member-owners, and what is the coverage amount? | Dupaco provides free $1,500 Accidental Death and Dismemberment (AD&D) insurance to member-owners ages 18 to 85. | 1,500, 18 |
| 3 | What is the minimum deposit required to open a primary Dupaco membership? | The minimum deposit to open a primary Dupaco membership share is $25. | 25 |
| 4 | What year did Dupaco first introduce its Holiday Club savings account? | Dupaco's Holiday Club savings account was created in 1954. | 1954 |
| 5 | What APY does the Launchpad Savings account offer? | The Launchpad Savings account pays 3.25% APY on balances from $5 to $1,000. | 3.25 |

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
- [ ] **Remove chunk inspection workbench** (`chunk-workbench/`): no longer needed once BM25 tuning is settled
- [ ] **Provide unit tests**: add tests for retrieval, chunking, and temporal filtering logic
