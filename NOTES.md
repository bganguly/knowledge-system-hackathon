# Knowledge System — Hackathon Notes

## Approach

**Retrieval:** BM25 keyword search (`rank-bm25`) over paragraph-level chunks extracted
from every ingested snapshot. Each chunk retains its URL, snapshot date, and an
authority score derived from the URL pattern:

| Authority | URL patterns |
|-----------|-------------|
| 5 — official rate page | `/personal/loans/`, `/personal/savings`, `/rates` |
| 4 — product page | `/personal/`, `/business/` |
| 3 — general | everything else |
| 2 — calculator | `calculator`, `calc`, `tool` |
| 1 — blog/newsroom | `/newsroom/`, `/blog/`, `/press` |

BM25 scores are blended with an authority multiplier so official rate pages
rank above calculators for the same keyword match.

**Generation:** Top-14 chunks are sent to **Claude Haiku** (`claude-haiku-5-5`) with
a prompt that enforces:
- Tier disambiguation (48-month ≠ 60-month, rate ≠ APY, standard ≠ jumbo)
- Authority preference (rate page over calculator when both present)
- Hard "I don't know" when the answer isn't in the excerpts

**Temporal filtering (`as_of`):** When `as_of` is provided, only chunks from the
latest snapshot whose `taken_on ≤ as_of` date are searched.

**Conflict detection:** During each `ingest()` call, percentage values (regex
`\d+\.\d{1,3}%`) are extracted from every page. When the same URL reappears in
a later snapshot with different percentage values the change is recorded as a
`Conflict` with the older snapshot as the stale source and the newer snapshot
as the resolution.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Running the eval

```bash
./deploy.sh /path/to/hackathon-knowledge-kit-participant
```

This unzips the five snapshot archives into `snapshots/`, then runs the harness
scorer. Results are written to `results/`.

## Repo policy

The website snapshot data is from a real credit union and **must not be
committed** to this repo. The `.gitignore` excludes all `snapshots/`,
`snapshots_zip/`, and `*.zip` paths.
