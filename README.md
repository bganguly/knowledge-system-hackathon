# Knowledge System Hackathon — BM25 · Claude Haiku · Temporal Filtering

Submission for the interface.ai hackathon: a **temporal knowledge system** that answers credit-union member questions from website snapshots taken at prior points in time.
BM25 retrieval over chunked pages feeds top-14 excerpts into Claude Haiku for grounded answer generation.
Conflict detection compares percentage values across snapshot dates for the same URL.

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
| **Python** | 3.13+ ([download](https://www.python.org/downloads/)) |

---

## Repo policy

Snapshot data is from a real credit union and must not be committed. `.gitignore` excludes `snapshots/`, `snapshots_zip/`, and `*.zip`.
