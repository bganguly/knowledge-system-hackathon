#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
VENV="$REPO/.venv"
PY312="/usr/local/bin/python3.12"
RETRIEVE_ARG="bm25"
POSITIONAL=()

usage() {
  cat <<EOF
Usage: $0 [--retrieve bm25|dense|hybrid|bm25,hybrid] [<row_number> | --clear | --clear-all | --eval]

  --retrieve    Retrieval strategy (default: bm25); comma-separate to run multiple
                  bm25         - keyword BM25 (fast, default)
                  dense        - TF-IDF cosine or sentence-transformer embeddings
                  hybrid       - BM25 top-50 candidates re-ranked by dense
                  bm25,hybrid  - run both modes in sequence

  <row_number>  Run WIP test for a single eval row (1–40)
  --clear       Remove a row from wip_<mode>.jsonl (requires <row_number> before it)
  --clear-all   Wipe wip_<mode>.jsonl entirely
  --eval        Run the full harness eval (slow, hits the LLM for all 40 rows)

Examples:
  ./deploy.sh 10
  ./deploy.sh --retrieve hybrid 10
  ./deploy.sh --retrieve bm25,hybrid 2
  ./deploy.sh --retrieve dense --eval
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --retrieve|--retrieval)
      RETRIEVE_ARG="$2"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      POSITIONAL+=("$1"); shift ;;
  esac
done

IFS=',' read -ra MODES <<< "$RETRIEVE_ARG"
for mode in "${MODES[@]}"; do
  if [[ ! "$mode" =~ ^(bm25|dense|hybrid)$ ]]; then
    echo "ERROR: each mode in --retrieve must be bm25, dense, or hybrid (got: $mode)" >&2
    exit 1
  fi
done

echo "==> Retrieval mode(s): ${MODES[*]}"

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
  RETRIEVAL_MODE="${MODES[0]}"
  export RETRIEVAL_MODE
  echo "==> Running full harness eval (RETRIEVAL_MODE=$RETRIEVAL_MODE)"
  cd "$REPO/../../hackathon-knowledge-kit-participant"
  "$PY" harness/run_eval.py --solution "$REPO/solution.py"
elif [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  ARGS=("${POSITIONAL[@]}")
  for mode in "${MODES[@]}"; do
    export RETRIEVAL_MODE="$mode"
    echo "==> Running WIP question (mode=$mode, args: ${ARGS[*]})"
    "$PY" "$REPO/scripts/wip_question.py" "${ARGS[@]}"
  done
  echo "==> Rebuilding comparison report"
  "$PY" "$REPO/scripts/build_report.py"
  echo "==> Done — results/comparison.html updated (WIP columns show #${POSITIONAL[0]})"
else
  usage
fi
