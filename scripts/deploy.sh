#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
ARG="${1:-}"

# ── Load API key ──────────────────────────────────────────────────────────────
ENV_FILE="$REPO_DIR/.env"
if [[ -z "${ANTHROPIC_API_KEY:-}" && -f "$ENV_FILE" ]]; then
  export ANTHROPIC_API_KEY="$(grep -E '^ANTHROPIC_API_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2-)"
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

_ensure_snapshots() {
  local kit_dir="$1"
  local snap_dir="$REPO_DIR/snapshots"
  mkdir -p "$snap_dir"
  for zip_file in "$kit_dir/snapshots_zip/"*.zip; do
    base="$(basename "$zip_file" .zip)"
    if [[ ! -f "$snap_dir/$base/manifest.json" ]]; then
      echo "    Extracting $base ..."
      unzip -q "$zip_file" -d "$snap_dir"
    fi
  done
}

# ── Single-question WIP mode ──────────────────────────────────────────────────
if [[ "$ARG" =~ ^[0-9]+$ ]]; then
  ROW="$ARG"
  KIT_DIR="$(cd "$REPO_DIR/../hackathon-knowledge-kit-participant" && pwd)"
  echo "==> Installing Python dependencies"
  "$PYTHON" -m pip install -q -r "$REPO_DIR/requirements.txt"
  echo "==> Ensuring snapshots are extracted"
  _ensure_snapshots "$KIT_DIR"
  echo "==> Running question #$ROW"
  "$PYTHON" "$REPO_DIR/scripts/wip_question.py" "$ROW"
  echo "==> Rebuilding comparison report"
  "$PYTHON" "$REPO_DIR/scripts/build_report.py"
  echo "==> Done — results/comparison.html updated (WIP column shows #$ROW)"
  exit 0
fi

# ── Full eval run ─────────────────────────────────────────────────────────────
KIT_DIR="${ARG:-"$REPO_DIR/../hackathon-knowledge-kit-participant"}"
KIT_DIR="$(cd "$KIT_DIR" && pwd)"
DEFAULT_MODEL="${LLM_MODEL:-claude-haiku-5-5}"
read -r -p "Switch model? (current: $DEFAULT_MODEL) [y/N] " _switch
_switch="${_switch:-N}"
if [[ "$_switch" =~ ^[Yy]$ ]]; then
  echo "  1) claude-haiku-5-5    — Anthropic  (fast, cheap)"
  echo "  2) claude-sonnet-5-5   — Anthropic  (balanced, latest)"
  echo "  3) claude-opus-5-5     — Anthropic  (most capable)"
  echo "  4) claude-sonnet-4-5   — Anthropic  (stable sonnet)"
  read -r -p "  Pick [1-4] or type a model ID: " _pick
  case "$_pick" in
    1) export LLM_MODEL="claude-haiku-5-5" ;;
    2) export LLM_MODEL="claude-sonnet-5-5" ;;
    3) export LLM_MODEL="claude-opus-5-5" ;;
    4) export LLM_MODEL="claude-sonnet-4-5" ;;
    *) export LLM_MODEL="${_pick:-$DEFAULT_MODEL}" ;;
  esac
else
  export LLM_MODEL="$DEFAULT_MODEL"
fi
echo "==> Using model: $LLM_MODEL"

echo "==> Installing Python dependencies"
"$PYTHON" -m pip install -q -r "$REPO_DIR/requirements.txt"

echo "==> Ensuring snapshots are extracted"
_ensure_snapshots "$KIT_DIR"

RESULTS_DIR="$REPO_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "==> Running harness eval"
"$PYTHON" "$KIT_DIR/harness/run_eval.py" \
  --submission "$REPO_DIR" \
  --eval "$KIT_DIR/eval/public_eval.jsonl" \
  --snapshots "$REPO_DIR/snapshots" \
  --out "$RESULTS_DIR"

echo ""
echo "==> Results written to $RESULTS_DIR"

RUNS_DIR="$RESULTS_DIR/runs"
mkdir -p "$RUNS_DIR"
RUN_TS="$(date '+%Y-%m-%d_%H-%M-%S')"
cp "$RESULTS_DIR/per_question.jsonl" "$RUNS_DIR/run_${RUN_TS}.jsonl"
cp "$RESULTS_DIR/report.json"        "$RUNS_DIR/report_${RUN_TS}.json"
echo "==> Run archived to $RUNS_DIR/run_${RUN_TS}.jsonl"

echo "==> Rebuilding comparison report"
"$PYTHON" "$REPO_DIR/scripts/build_report.py"
echo "==> Done — results/comparison.html updated"
