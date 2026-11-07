#!/usr/bin/env bash
set -euo pipefail

WORKBENCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="$(cd "$WORKBENCH_DIR/.." && pwd)"
VENV="$REPO_DIR/.venv"
PY312="/usr/local/bin/python3.12"

DEFAULT_KIT="$(cd "$REPO_DIR/../hackathon-knowledge-kit-participant" && pwd 2>/dev/null || echo "")"
DEFAULT_EVAL="$DEFAULT_KIT/eval/public_eval.jsonl"

usage() {
  cat <<EOF
Usage: $0 [options]

Run BM25 chunk retrieval without LLM calls. Output: chunk-workbench/results/chunk_run_<ts>.jsonl

Options:
  --kit PATH        Path to hackathon-knowledge-kit-participant (default: $DEFAULT_KIT)
  --eval PATH       Eval JSONL file (default: $DEFAULT_EVAL)
  --top-k N         Chunks per question (default: 22)
  --max-per-url N   Max chunks from one URL (default: 6)
  --max-chunk N     Max chars per chunk at ingest time (default: 700)
  --questions LIST  Comma-separated 1-based indices or IDs, e.g. '1,5,10' or 'dup-010'
  --out PATH        Output JSONL path (default: results/chunk_run_<ts>.jsonl)
  -h, --help        Show this help

Examples:
  ./scripts/run_chunks.sh
  ./scripts/run_chunks.sh --top-k 30
  ./scripts/run_chunks.sh --questions 1,5,10
  ./scripts/run_chunks.sh --max-per-url 3 --top-k 22
  ./scripts/run_chunks.sh --max-chunk 500 --out results/smaller_chunks.jsonl
EOF
}

KIT="${DEFAULT_KIT}"
EVAL="${DEFAULT_EVAL}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kit)        KIT="$2"; shift 2 ;;
    --eval)       EVAL="$2"; shift 2 ;;
    --top-k)      EXTRA_ARGS+=("--top-k" "$2"); shift 2 ;;
    --max-per-url) EXTRA_ARGS+=("--max-per-url" "$2"); shift 2 ;;
    --max-chunk)  EXTRA_ARGS+=("--max-chunk" "$2"); shift 2 ;;
    --questions)  EXTRA_ARGS+=("--questions" "$2"); shift 2 ;;
    --out)        EXTRA_ARGS+=("--out" "$2"); shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$KIT" ]]; then
  echo "ERROR: --kit not set and default path not found."
  usage
  exit 1
fi

PY="$PY312"
if [[ -f "$VENV/bin/python" ]]; then
  PY="$VENV/bin/python"
fi

echo "Kit:   $KIT"
echo "Eval:  $EVAL"
echo "Py:    $PY"
echo ""

"$PY" "$WORKBENCH_DIR/chunk_runner.py" \
  --kit "$KIT" \
  --eval "$EVAL" \
  "${EXTRA_ARGS[@]}"
