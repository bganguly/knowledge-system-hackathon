#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

KIT_DIR="${1:-"$REPO_DIR/../hackathon-knowledge-kit-participant"}"
KIT_DIR="$(cd "$KIT_DIR" && pwd)"
DEFAULT_MODEL="${2:-${LLM_MODEL:-claude-haiku-5-5}}"
read -r -p "Model [$DEFAULT_MODEL]: " _model_input
export LLM_MODEL="${_model_input:-$DEFAULT_MODEL}"
echo "==> Using model: $LLM_MODEL"

ENV_FILE="$REPO_DIR/.env"
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  if [[ -f "$ENV_FILE" ]]; then
    echo "==> Loading ANTHROPIC_API_KEY from $ENV_FILE"
    export ANTHROPIC_API_KEY="$(grep -E '^ANTHROPIC_API_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2-)"
  fi
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  read -r -p "ANTHROPIC_API_KEY not set. Enter it now? [y/N] " yn
  yn="${yn:-N}"
  if [[ "$yn" =~ ^[Yy]$ ]]; then
    read -r -s -p "ANTHROPIC_API_KEY: " ANTHROPIC_API_KEY
    echo
    export ANTHROPIC_API_KEY
  else
    echo "ERROR: ANTHROPIC_API_KEY required. Set it in .env or the environment."
    exit 1
  fi
fi

PYTHON="${PYTHON:-python3}"
echo "==> Installing Python dependencies"
"$PYTHON" -m pip install -q -r "$REPO_DIR/requirements.txt"

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
"$PYTHON" "$KIT_DIR/harness/run_eval.py" \
  --submission "$REPO_DIR" \
  --eval "$KIT_DIR/eval/public_eval.jsonl" \
  --snapshots "$SNAP_DIR" \
  --out "$RESULTS_DIR"

echo ""
echo "==> Results written to $RESULTS_DIR"
