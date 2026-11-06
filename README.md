# Knowledge System Hackathon — BM25 · Claude Haiku · Temporal Filtering

Submission for the interface.ai hackathon: a **temporal knowledge system** that answers credit-union member questions from crawled website snapshots.
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

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running the eval

```bash
./scripts/deploy.sh /path/to/hackathon-knowledge-kit-participant
```

Unzips the five snapshot archives into `snapshots/`, runs the harness scorer, writes results to `results/`.

---

## Stack

| Component | Detail |
|---|---|
| **Retrieval** | `rank-bm25` — pure Python, no API calls at query time |
| **LLM** | Anthropic Claude Haiku (`claude-haiku-5-5`) via `anthropic` SDK |
| **Python** | 3.9+ |

---

## Repo policy

Snapshot data is from a real credit union and must not be committed. `.gitignore` excludes `snapshots/`, `snapshots_zip/`, and `*.zip`.
