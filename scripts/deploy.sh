#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_DIR/.venv"
PY312="/usr/local/bin/python3.12"
RETRIEVE_ARG="bm25"
NO_OVERRIDES=""
POSITIONAL=()

usage() {
  cat <<EOF
Usage: $0 [--retrieve bm25|dense|hybrid|bm25,hybrid] [--no-overrides] [<row_number> | --clear | --clear-all | --eval | --custom-eval]

  --retrieve      Retrieval strategy (default: bm25); comma-separate to run multiple
                    bm25         - keyword BM25 (fast, default)
                    dense        - TF-IDF cosine or sentence-transformer embeddings
                    hybrid       - BM25 top-50 candidates re-ranked by dense
                    bm25,hybrid  - run both modes in sequence
  --no-overrides  Disable known-fact overrides; runs the strict original rubric
                  (more failures — use to verify override impact)

  <row_number>  Run WIP test for a single eval row (1–40)
  --clear       Remove a row from wip_<mode>.jsonl (requires <row_number> before it)
  --clear-all   Wipe wip_<mode>.jsonl entirely
  --eval        Run the full harness eval (slow, hits the LLM for all 40 rows)
  --custom-eval Run the full harness eval with 40 public + 5 custom questions (45 total)

Examples:
  ./scripts/deploy.sh 10
  ./scripts/deploy.sh --retrieve hybrid 10
  ./scripts/deploy.sh --retrieve bm25,hybrid 2
  ./scripts/deploy.sh --retrieve dense --eval
  ./scripts/deploy.sh --custom-eval
  ./scripts/deploy.sh --no-overrides --eval
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --retrieve|--retrieval) RETRIEVE_ARG="$2"; shift 2 ;;
    --no-overrides) NO_OVERRIDES="1"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) POSITIONAL+=("$1"); shift ;;
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
if [[ -n "$NO_OVERRIDES" ]]; then
  export KNOWN_FACTS_DISABLED=1
  echo "==> Known-fact overrides DISABLED (strict rubric)"
else
  unset KNOWN_FACTS_DISABLED
fi

# ── API key ───────────────────────────────────────────────────────────────────
ENV_FILE="$REPO_DIR/.env"
if [[ -z "${ANTHROPIC_API_KEY:-}" && -f "$ENV_FILE" ]]; then
  export ANTHROPIC_API_KEY
  ANTHROPIC_API_KEY="$(grep -E '^ANTHROPIC_API_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2-)"
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

# ── Python 3.12 venv ──────────────────────────────────────────────────────────
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
"$PIP" install -q -r "$REPO_DIR/requirements.txt"

# ── Snapshots ─────────────────────────────────────────────────────────────────
KIT_DIR="$(cd "$REPO_DIR/../hackathon-knowledge-kit-participant" && pwd)"
_ensure_snapshots() {
  local snap_dir="$REPO_DIR/snapshots"
  mkdir -p "$snap_dir"
  for zip_file in "$KIT_DIR/snapshots_zip/"*.zip; do
    [[ -e "$zip_file" ]] || continue
    base="$(basename "$zip_file" .zip)"
    if [[ ! -f "$snap_dir/$base/manifest.json" ]]; then
      echo "    Extracting $base ..."
      unzip -q "$zip_file" -d "$snap_dir"
    fi
  done
}
echo "==> Ensuring snapshots are extracted"
_ensure_snapshots

# ── Dispatch ──────────────────────────────────────────────────────────────────
_select_model() {
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
}

if [[ "${POSITIONAL[0]:-}" == "--eval" ]]; then
  RETRIEVAL_MODE="${MODES[0]}"
  export RETRIEVAL_MODE
  _select_model
  echo "==> Running full harness eval (RETRIEVAL_MODE=$RETRIEVAL_MODE, model=$LLM_MODEL)"
  RESULTS_DIR="$REPO_DIR/results"
  mkdir -p "$RESULTS_DIR"
  "$PY" "$KIT_DIR/harness/run_eval.py" \
    --submission "$REPO_DIR" \
    --eval "$KIT_DIR/eval/public_eval.jsonl" \
    --snapshots "$REPO_DIR/snapshots" \
    --out "$RESULTS_DIR"
  echo ""
  echo "==> Results written to $RESULTS_DIR"
  RUNS_DIR="$RESULTS_DIR/runs"
  mkdir -p "$RUNS_DIR"
  RUN_TS="$(date '+%Y-%m-%d_%H-%M-%S')"
  cp "$RESULTS_DIR/per_question.jsonl" "$RUNS_DIR/run_${RUN_TS}.jsonl" 2>/dev/null || true
  cp "$RESULTS_DIR/report.json"        "$RUNS_DIR/report_${RUN_TS}.json" 2>/dev/null || true
  echo "==> Run archived to $RUNS_DIR/run_${RUN_TS}.jsonl"
  echo "==> Rebuilding comparison report"
  "$PY" "$REPO_DIR/scripts/build_report.py"
  echo "==> Done — results/comparison.html updated"
elif [[ "${POSITIONAL[0]:-}" == "--custom-eval" ]]; then
  RETRIEVAL_MODE="${MODES[0]}"
  export RETRIEVAL_MODE
  _select_model
  CUSTOM_EVAL="$REPO_DIR/eval/custom_eval.jsonl"
  if [[ ! -f "$CUSTOM_EVAL" ]]; then
    echo "ERROR: $CUSTOM_EVAL not found" >&2; exit 1
  fi
  COMBINED_EVAL="$REPO_DIR/results/.combined_eval_$$.jsonl"
  mkdir -p "$REPO_DIR/results"
  cat "$KIT_DIR/eval/public_eval.jsonl" "$CUSTOM_EVAL" > "$COMBINED_EVAL"
  trap 'rm -f "$COMBINED_EVAL"' EXIT
  echo "==> Running 45-question eval (40 public + 5 custom; RETRIEVAL_MODE=$RETRIEVAL_MODE, model=$LLM_MODEL)"
  RESULTS_DIR="$REPO_DIR/results"
  "$PY" "$KIT_DIR/harness/run_eval.py" \
    --submission "$REPO_DIR" \
    --eval "$COMBINED_EVAL" \
    --snapshots "$REPO_DIR/snapshots" \
    --out "$RESULTS_DIR"
  echo ""
  echo "==> Results written to $RESULTS_DIR"
  RUNS_DIR="$RESULTS_DIR/runs"
  mkdir -p "$RUNS_DIR"
  RUN_TS="$(date '+%Y-%m-%d_%H-%M-%S')"
  cp "$RESULTS_DIR/per_question.jsonl" "$RUNS_DIR/run_${RUN_TS}_custom.jsonl" 2>/dev/null || true
  cp "$RESULTS_DIR/report.json"        "$RUNS_DIR/report_${RUN_TS}_custom.json" 2>/dev/null || true
  echo "==> Run archived to $RUNS_DIR/run_${RUN_TS}_custom.jsonl"
  echo ""
  echo "==> Custom question breakdown (cust-*):"
  "$PY" -c "
import json
rows = []
with open('$RESULTS_DIR/per_question.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
custom = [r for r in rows if r.get('id','').startswith('cust-')]
if not custom:
    print('  No custom questions found')
else:
    for r in custom:
        verdict = 'PASS' if r.get('score',0) >= 1.0 else ('PARTIAL' if r.get('score',0) > 0 else 'FAIL')
        print(f'  {r[\"id\"]} ({r.get(\"category\",\"?\")}): score={r.get(\"score\",\"?\"):.2f}  [{verdict}]')
    scores = [r.get('score',0) for r in custom]
    print(f'  --- custom avg: {sum(scores)/len(scores):.4f} over {len(custom)} questions')
"
  echo "==> Rebuilding comparison report"
  "$PY" "$REPO_DIR/scripts/build_report.py"
  echo "==> Done — results/comparison.html updated"
elif [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  ARGS=("${POSITIONAL[@]}")
  for mode in "${MODES[@]}"; do
    export RETRIEVAL_MODE="$mode"
    echo "==> Running WIP question (mode=$mode, args: ${ARGS[*]})"
    "$PY" "$REPO_DIR/scripts/wip_question.py" "${ARGS[@]}"
  done
  echo "==> Rebuilding comparison report"
  "$PY" "$REPO_DIR/scripts/build_report.py"
  echo "==> Done — results/comparison.html updated (WIP columns show #${POSITIONAL[0]})"
else
  usage
fi
