#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Removing extracted snapshots (data files, not code)"
rm -rf "$REPO_DIR/snapshots"

echo "==> Removing eval result outputs"
rm -rf "$REPO_DIR/results"

echo "Done. Source files and requirements.txt are untouched."
