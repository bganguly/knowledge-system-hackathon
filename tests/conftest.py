"""Add the harness to sys.path so tests can import solution.py directly."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_HARNESS = _REPO.parent / "hackathon-knowledge-kit-participant" / "harness"
if _HARNESS.exists() and str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))
