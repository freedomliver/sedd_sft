#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_sft import get_answer_text, get_final_answer_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit fixed-block answer-field coverage")
    parser.add_argument("json_path")
    parser.add_argument("--show", type=int, default=5)
    args = parser.parse_args()

    rows = json.loads(Path(args.json_path).read_text())
    counts = {}
    missing = 0
    for row in rows:
        source_answer = get_answer_text(row, "solution")
        _, source = get_final_answer_text(
            row,
            "answer",
            source_answer=source_answer,
            return_source=True,
        )
        counts[source] = counts.get(source, 0) + 1
        if source == "missing":
            missing += 1

    print(f"file: {args.json_path}")
    print(f"rows: {len(rows)}")
    print(f"missing_final_answer: {missing}")
    print("source_counts:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")

    print("samples:")
    shown = 0
    for idx, row in enumerate(rows):
        source_answer = get_answer_text(row, "solution")
        answer, source = get_final_answer_text(
            row,
            "answer",
            source_answer=source_answer,
            return_source=True,
        )
        if not answer:
            continue
        print(f"[{idx}] source={source} answer={answer[:120]}")
        shown += 1
        if shown >= args.show:
            break
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
