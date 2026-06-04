#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def short(text: str, limit: int) -> str:
    text = str(text or "").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether fixed-block eval64 is good enough to launch full599"
    )
    parser.add_argument("jsonl")
    parser.add_argument("--min-match", type=float, default=0.90)
    parser.add_argument("--min-eos", type=float, default=0.90)
    parser.add_argument("--max-empty", type=float, default=0.05)
    parser.add_argument("--show", type=int, default=8)
    parser.add_argument("--width", type=int, default=140)
    args = parser.parse_args()

    path = Path(args.jsonl)
    rows = load_rows(path)
    total = len(rows)
    if total == 0:
        print(f"gate=FAIL reason=no_rows file={path}")
        return 2

    match = sum(bool(row.get("match")) for row in rows)
    eos = sum(bool(row.get("eos_present")) for row in rows)
    empty = sum(not str(row.get("pred_answer", "")).strip() for row in rows)
    match_rate = match / total
    eos_rate = eos / total
    empty_rate = empty / total

    checks = {
        "answer_block_match": match_rate >= args.min_match,
        "eos_present": eos_rate >= args.min_eos,
        "empty_pred_answer": empty_rate <= args.max_empty,
    }
    passed = all(checks.values())

    print(f"file: {path}")
    print(f"rows: {total}")
    print(f"answer_block_match: {match}/{total} = {match_rate:.1%} threshold>={args.min_match:.1%}")
    print(f"eos_present: {eos}/{total} = {eos_rate:.1%} threshold>={args.min_eos:.1%}")
    print(f"empty_pred_answer: {empty}/{total} = {empty_rate:.1%} threshold<={args.max_empty:.1%}")
    print(f"gate={'PASS' if passed else 'FAIL'}")
    if passed:
        print("recommendation=launch_full599")
    else:
        print("recommendation=inspect_eval64_failures_before_full599")
        failures = [row for row in rows if not row.get("match")]
        for row in failures[: args.show]:
            print(
                f"[{row.get('idx')}] gold={short(row.get('gold_answer', ''), args.width)} "
                f"pred={short(row.get('pred_answer', ''), args.width)} "
                f"eos={row.get('eos_present')} eos_idx={row.get('eos_index')} "
                f"pred_tokens={row.get('pred_answer_token_len')}"
            )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
