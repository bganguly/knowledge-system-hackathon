#!/usr/bin/env python3
"""
BM25 chunk workbench — no LLM calls.

Loads snapshots, runs retrieval for each question, writes a chunk JSONL for
visual inspection. Tweak --top-k / --max-per-url / --max-chunk to compare
how different retrieval parameters affect which chunks are selected.

Usage:
  python chunk_runner.py --kit /path/to/hackathon-knowledge-kit-participant \\
                         --eval /path/to/eval.jsonl \\
                         [--top-k 22] [--max-per-url 6] [--max-chunk 700] \\
                         [--questions 1,5,10] \\
                         [--out results/my_run.jsonl]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

# No LLM calls — stub key so KnowledgeSystem.__init__ doesn't raise.
os.environ.setdefault("ANTHROPIC_API_KEY", "chunk-runner-no-llm")

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

import solution  # noqa: E402  (must come after path setup)
from solution import KnowledgeSystem  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 chunk retrieval — no LLM")
    parser.add_argument("--kit", required=True,
                        help="Path to hackathon-knowledge-kit-participant directory")
    parser.add_argument("--eval", required=True,
                        help="Eval JSONL file (questions)")
    parser.add_argument("--top-k",      type=int, default=solution._TOP_K,
                        help=f"Chunks fed to LLM slot (default: {solution._TOP_K})")
    parser.add_argument("--max-per-url", type=int, default=solution._MAX_PER_URL,
                        help=f"Max chunks from one URL (default: {solution._MAX_PER_URL})")
    parser.add_argument("--max-chunk",  type=int, default=solution._MAX_CHUNK,
                        help=f"Max chars per chunk (default: {solution._MAX_CHUNK})")
    parser.add_argument("--questions",
                        help="Comma-separated 1-based indices or question IDs to filter, "
                             "e.g. '1,5,10' or 'dup-010,dup-022'")
    parser.add_argument("--out", default=None,
                        help="Output JSONL path (default: results/chunk_run_<ts>.jsonl)")
    args = parser.parse_args()

    # ----- load eval questions -----
    questions: list[dict] = []
    with open(args.eval) as fh:
        for line in fh:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if args.questions:
        raw = [x.strip() for x in args.questions.split(",")]
        indices = {int(x) for x in raw if x.isdigit()}
        ids     = {x for x in raw if not x.isdigit()}
        questions = [
            q for i, q in enumerate(questions)
            if (i + 1) in indices or q.get("id") in ids
        ]

    # ----- patch globals BEFORE ingest (max_chunk affects chunking) -----
    solution._MAX_CHUNK   = args.max_chunk
    solution._TOP_K       = args.top_k
    solution._MAX_PER_URL = args.max_per_url

    # ----- load snapshots via harness loader -----
    kit = Path(args.kit)
    harness_dir = kit / "harness"
    sys.path.insert(0, str(harness_dir))
    from run_eval import load_snapshots  # noqa: E402

    snapshots_root = kit / "snapshots"
    print(f"Loading snapshots from {snapshots_root} ...", flush=True)
    snaps = load_snapshots(snapshots_root)
    print(f"  {len(snaps)} snapshot(s) loaded", flush=True)

    # ----- ingest -----
    ks = KnowledgeSystem()
    for snap in snaps:
        ks.ingest(snap)
    print(f"Ingest done.", flush=True)

    # ----- retrieve -----
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = Path(args.out) if args.out else \
        Path(__file__).parent / "results" / f"chunk_run_{ts}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "ts": ts,
        "params": {
            "top_k":       args.top_k,
            "max_per_url": args.max_per_url,
            "max_chunk":   args.max_chunk,
        },
    }

    rows: list[dict] = []
    for q in questions:
        question_text = q["question"]
        as_of         = q.get("as_of")
        pool          = ks._chunks_as_of(as_of)
        chunks        = ks._retrieve(question_text, pool)

        row = {
            "id":       q.get("id"),
            "question": question_text,
            "as_of":    as_of,
            "chunks": [
                {
                    "rank":        idx + 1,
                    "url":         c.url,
                    "title":       c.title,
                    "authority":   c.authority,
                    "snapshot_id": c.snapshot_id,
                    "taken_on":    c.taken_on,
                    "text":        c.text,
                }
                for idx, c in enumerate(chunks)
            ],
        }
        rows.append(row)
        print(f"  {q.get('id', f'q{questions.index(q)+1}'):10s}  {len(chunks)} chunks  "
              f"as_of={as_of or 'latest'}", flush=True)

    with open(out_path, "w") as fh:
        fh.write(json.dumps(meta) + "\n")
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    print(f"\nSaved {len(rows)} question(s) → {out_path}")


if __name__ == "__main__":
    main()
