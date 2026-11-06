#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
VENV="$REPO/.venv"
PY312="/usr/local/bin/python3.12"
RETRIEVAL_MODE="bm25"
POSITIONAL=()

usage() {
  cat <<EOF
Usage: $0 [--retrieval bm25|dense|hybrid] [<row_number> | --clear | --clear-all | --eval]

  --retrieval   Retrieval strategy (default: bm25)
                  bm25   - keyword BM25 (fast, default)
                  dense  - TF-IDF cosine or sentence-transformer embeddings
                  hybrid - BM25 top-50 candidates re-ranked by dense

  <row_number>  Run WIP test for a single eval row (1–40)
  --clear       Remove a row from wip.jsonl (requires <row_number> before it)
  --clear-all   Wipe wip.jsonl entirely
  --eval        Run the full harness eval (slow, hits the LLM for all 40 rows)

Examples:
  ./deploy.sh 10
  ./deploy.sh --retrieval hybrid 10
  ./deploy.sh --retrieval dense --eval
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --retrieval)
      RETRIEVAL_MODE="$2"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ ! "$RETRIEVAL_MODE" =~ ^(bm25|dense|hybrid)$ ]]; then
  echo "ERROR: --retrieval must be bm25, dense, or hybrid" >&2
  exit 1
fi

export RETRIEVAL_MODE

echo "==> Retrieval mode: $RETRIEVAL_MODE"

# Use Python 3.12 venv so sentence-transformers (torch) can install
if [[ ! -x "$PY312" ]]; then
  echo "ERROR: $PY312 not found — install Python 3.12 first" >&2
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  echo "==> Creating Python 3.12 venv at $VENV"
  "$PY312" -m venv "$VENV"
fi

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

echo "==> Installing Python dependencies ($("$PY" --version))"
"$PIP" install -q -r "$REPO/requirements.txt"

echo "==> Ensuring snapshots are extracted"
"$REPO/scripts/ensure_snapshots.sh" 2>/dev/null || true

if [[ "${POSITIONAL[0]:-}" == "--eval" ]]; then
  echo "==> Running full harness eval (RETRIEVAL_MODE=$RETRIEVAL_MODE)"
  cd "$REPO/../../hackathon-knowledge-kit-participant"
  "$PY" harness/run_eval.py --solution "$REPO/solution.py"
elif [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  ARGS=("${POSITIONAL[@]}")
  echo "==> Running WIP question (args: ${ARGS[*]})"
  "$PY" "$REPO/scripts/wip_question.py" "${ARGS[@]}"
  echo "==> Rebuilding comparison report"
  "$PY" "$REPO/scripts/build_report.py"
  echo "==> Done — results/comparison.html updated (WIP column shows #${POSITIONAL[0]})"
else
  usage
fi
