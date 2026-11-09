#!/usr/bin/env python3
"""
Update eval/gold_cache.json with score=1.0 entries from a run JSONL.

Usage:
    python3 scripts/update_gold_cache.py <run.jsonl> <eval1.jsonl> [<eval2.jsonl> ...]
"""
import datetime
import json
import sys
from pathlib import Path

GOLD_CACHE_PATH = Path(__file__).parent.parent / "eval" / "gold_cache.json"
TTL_DAYS = 90

def main():
    if len(sys.argv) < 3:
        print("Usage: update_gold_cache.py <run.jsonl> <eval1.jsonl> [eval2.jsonl ...]", file=sys.stderr)
        sys.exit(1)

    run_path = Path(sys.argv[1])
    eval_paths = [Path(p) for p in sys.argv[2:]]

    by_id: dict[str, dict] = {}
    for ep in eval_paths:
        if not ep.exists():
            continue
        with open(ep) as f:
            for line in f:
                r = json.loads(line)
                by_id[r["id"]] = r

    run_rows: list[dict] = []
    with open(run_path) as f:
        for line in f:
            run_rows.append(json.loads(line))

    existing: dict[str, dict] = {}
    if GOLD_CACHE_PATH.exists():
        try:
            existing = json.loads(GOLD_CACHE_PATH.read_text())
        except Exception:
            pass

    today = datetime.date.today().isoformat()
    cutoff = datetime.date.today() - datetime.timedelta(days=TTL_DAYS)

    # Prune expired
    pruned = {k: v for k, v in existing.items()
              if datetime.date.fromisoformat(v.get("ts", "2000-01-01")) >= cutoff}

    added = refreshed = 0
    for r in run_rows:
        if r.get("score", 0) < 1.0:
            continue
        eq = by_id.get(r["id"], {})
        q = eq.get("question", "")
        as_of = str(eq.get("as_of") or "None")
        if not q:
            continue
        key = q + "||" + as_of
        answer_text = r.get("answer", {}).get("text", "")
        if not answer_text:
            continue
        raw_sources = r.get("answer", {}).get("sources") or []
        sources = [{"url": s["url"], "snapshot_id": s.get("snapshot_id")}
                   for s in raw_sources if isinstance(s, dict) and s.get("url")]
        if key in pruned:
            refreshed += 1
        else:
            added += 1
        pruned[key] = {"text": answer_text, "sources": sources, "ts": today}

    GOLD_CACHE_PATH.parent.mkdir(exist_ok=True)
    GOLD_CACHE_PATH.write_text(json.dumps(pruned, indent=2))
    print(f"[gold-cache] {len(pruned)} entries total — {added} new, {refreshed} refreshed, "
          f"{len(existing) - len(pruned) + added + refreshed} expired pruned")

if __name__ == "__main__":
    main()
