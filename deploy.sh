#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

KIT_DIR="${1:-}"
if [[ -z "$KIT_DIR" ]]; then
  echo "Usage: $0 <path-to-hackathon-knowledge-kit-participant>"
  exit 1
fi

KIT_DIR="$(cd "$KIT_DIR" && pwd)"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set."
  exit 1
fi

echo "==> Installing Python dependencies"
pip install -q -r "$REPO_DIR/requirements.txt"

SNAP_DIR="$REPO_DIR/snapshots"
mkdir -p "$SNAP_DIR"

echo "==> Unzipping snapshots from $KIT_DIR/snapshots_zip/"
for zip_file in "$KIT_DIR/snapshots_zip/"*.zip; do
  base="$(basename "$zip_file" .zip)"
  if [[ ! -d "$SNAP_DIR/$base" ]]; then
    echo "    Extracting $base ..."
    unzip -q "$zip_file" -d "$SNAP_DIR/$base"
  else
    echo "    $base already extracted, skipping."
  fi
done

RESULTS_DIR="$REPO_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "==> Running harness eval"
python "$KIT_DIR/harness/run_eval.py" \
  --submission "$REPO_DIR" \
  --eval "$KIT_DIR/eval/public_eval.jsonl" \
  --snapshots "$SNAP_DIR" \
  --out "$RESULTS_DIR"

echo ""
echo "==> Results written to $OUT_FILE"
