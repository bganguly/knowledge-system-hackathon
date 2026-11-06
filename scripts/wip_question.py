"""
Test a single question by artifact row number without running the full harness.
Saves/updates results/wip.jsonl for use in the comparison report WIP column.

Usage:
  python3 scripts/wip_question.py <row_number>        # e.g. 2 or 7
  python3 scripts/wip_question.py <row_number> --clear # remove this question from wip.jsonl
  python3 scripts/wip_question.py --clear-all          # wipe wip.jsonl entirely
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SNAP_DIR = REPO / "snapshots"
EVAL_FILE = REPO.parent / "hackathon-knowledge-kit-participant" / "eval" / "public_eval.jsonl"
WIP_FILE = REPO / "results" / "wip.jsonl"

# ---------------------------------------------------------------------------
# Load eval metadata and row-to-id mapping
# ---------------------------------------------------------------------------

def _load_eval() -> tuple[list[str], dict]:
    all_ids = []
    meta = {}
    with open(EVAL_FILE) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                all_ids.append(r["id"])
                meta[r["id"]] = r
    all_ids.sort(key=lambda x: int(x.split("-")[1]))
    return all_ids, meta

# ---------------------------------------------------------------------------
# Load snapshot data via solution.py build()
# ---------------------------------------------------------------------------

def _build_ks():
    sys.path.insert(0, str(REPO.parent / "hackathon-knowledge-kit-participant" / "harness"))
    sys.path.insert(0, str(REPO))
    import importlib
    import solution as sol
    importlib.reload(sol)
    from contract import Snapshot, Page  # noqa: F401

    ks = sol.KnowledgeSystem()
    for snap_dir in sorted(SNAP_DIR.iterdir()):
        manifest_path = snap_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        pages = []
        for entry in manifest.get("pages", []):
            md_path = snap_dir / entry["file"]
            if md_path.exists():
                pages.append(Page(url=entry["url"], title=entry.get("title", ""),
                                  markdown=md_path.read_text(encoding="utf-8")))
        snapshot = Snapshot(
            snapshot_id=manifest["snapshot_id"],
            taken_on=manifest["taken_on"],
            pages=pages,
        )
        ks.ingest(snapshot)
    return ks

# ---------------------------------------------------------------------------
# Simple score estimate (no LLM judge — just token check)
# ---------------------------------------------------------------------------

def _estimate_score(answer_text: str, meta: dict) -> float:
    must_inc = meta.get("must_include") or []
    must_not = meta.get("must_not_include") or []
    text_lower = answer_text.lower()

    for tok in must_not:
        if tok.lower() in text_lower:
            return 0.0

    if must_inc:
        hits = sum(1 for tok in must_inc if tok.lower() in text_lower)
        correctness = hits / len(must_inc)
    else:
        # no must_include — assume correct if answer is not "I don't know"
        correctness = 0.0 if "i don't know" in text_lower else 0.85

    return round(0.85 * correctness, 4)

# ---------------------------------------------------------------------------
# WIP file helpers
# ---------------------------------------------------------------------------

def _read_wip() -> dict[str, dict]:
    result = {}
    if WIP_FILE.exists():
        with open(WIP_FILE) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    result[r["id"]] = r
    return result

def _write_wip(rows: dict[str, dict]):
    WIP_FILE.parent.mkdir(exist_ok=True)
    with open(WIP_FILE, "w") as f:
        for row in rows.values():
            f.write(json.dumps(row) + "\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if "--clear-all" in args:
        if WIP_FILE.exists():
            WIP_FILE.unlink()
        print("Cleared wip.jsonl")
        return

    if not args:
        print("Usage: python3 scripts/wip_question.py <row_number> [--clear]")
        sys.exit(1)

    row_num = int(args[0])
    clear = "--clear" in args

    all_ids, meta = _load_eval()
    if row_num < 1 or row_num > len(all_ids):
        print(f"Row must be 1–{len(all_ids)}")
        sys.exit(1)

    qid = all_ids[row_num - 1]
    m = meta[qid]

    wip = _read_wip()

    if clear:
        wip.pop(qid, None)
        _write_wip(wip)
        print(f"Removed #{row_num} ({qid}) from wip.jsonl")
        return

    # Load API key
    env_file = REPO / ".env"
    if not os.environ.get("ANTHROPIC_API_KEY") and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
                break

    print(f"Building knowledge system from snapshots...")
    ks = _build_ks()

    print(f"\n#{row_num} ({qid}): {m['question']}")
    print(f"  expected: {m.get('expected_answer','')[:80]}")
    print(f"  must_include: {m.get('must_include')}")
    print(f"  must_not_include: {m.get('must_not_include')}")

    as_of = m.get("as_of")
    t0 = time.time()
    answer = ks.answer(m["question"], as_of=as_of)
    elapsed = time.time() - t0

    est = _estimate_score(answer.text, m)
    print(f"\nAnswer ({elapsed:.1f}s):\n{answer.text}")
    print(f"\nEstimated score (token check only): {est:.4f}")

    wip[qid] = {
        "id": qid,
        "question": m["question"],
        "score": est,
        "answer": {
            "text": answer.text,
            "confident": answer.confident,
            "sources": [
                {"url": s.url, "snapshot_id": s.snapshot_id}
                for s in (answer.sources or [])
            ],
        },
    }
    _write_wip(wip)
    print(f"\nSaved to wip.jsonl  ({len(wip)} question(s) tracked)")

if __name__ == "__main__":
    main()
