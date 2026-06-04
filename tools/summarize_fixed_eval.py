#!/usr/bin/env python3
import argparse
import json
from collections import Counter
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


def main():
    parser = argparse.ArgumentParser(description="Summarize fixed-block eval JSONL output")
    parser.add_argument("jsonl")
    parser.add_argument("--show", type=int, default=8)
    parser.add_argument("--width", type=int, default=180)
    parser.add_argument("--raw-width", type=int, default=160)
    args = parser.parse_args()

    path = Path(args.jsonl)
    rows = load_rows(path)
    total = len(rows)
    if total == 0:
        print(f"{path}: no rows")
        return

    match = sum(bool(row.get("match")) for row in rows)
    exact = sum(bool(row.get("exact")) for row in rows)
    eos = sum(bool(row.get("eos_present")) for row in rows)
    empty = sum(not str(row.get("pred_answer", "")).strip() for row in rows)
    answer_lens = [len(str(row.get("pred_answer", ""))) for row in rows]
    gold_lens = [len(str(row.get("gold_answer", ""))) for row in rows]
    pred_token_lens = [
        int(row.get("pred_answer_token_len", 0) or 0)
        for row in rows
    ]
    eos_indexes = [
        int(row["eos_index"])
        for row in rows
        if row.get("eos_index") is not None
    ]
    first_chars = Counter(
        (str(row.get("pred_answer", "")).strip()[:1] or "<empty>")
        for row in rows
    )

    print(f"file: {path}")
    print(f"rows: {total}")
    print(f"answer_block_match: {match}/{total} = {match / total:.1%}")
    print(f"exact_answer_match: {exact}/{total} = {exact / total:.1%}")
    print(f"eos_present: {eos}/{total} = {eos / total:.1%}")
    print(f"empty_pred_answer: {empty}/{total} = {empty / total:.1%}")
    print(
        "pred_answer_chars min/max/avg: "
        f"{min(answer_lens)} {max(answer_lens)} {sum(answer_lens) / total:.1f}"
    )
    print(
        "gold_answer_chars min/max/avg: "
        f"{min(gold_lens)} {max(gold_lens)} {sum(gold_lens) / total:.1f}"
    )
    print(
        "pred_answer_tokens min/max/avg: "
        f"{min(pred_token_lens)} {max(pred_token_lens)} "
        f"{sum(pred_token_lens) / total:.1f}"
    )
    if eos_indexes:
        print(
            "eos_index min/max/avg: "
            f"{min(eos_indexes)} {max(eos_indexes)} "
            f"{sum(eos_indexes) / len(eos_indexes):.1f}"
        )
    else:
        print("eos_index min/max/avg: none")
    print("pred_first_char_top:", first_chars.most_common(10))

    failures = [row for row in rows if not row.get("match")]
    if failures:
        print(f"\nfirst_failures shown={min(args.show, len(failures))}/{len(failures)}")
        for row in failures[: args.show]:
            print(
                f"[{row.get('idx')}] gold={short(row.get('gold_answer', ''), args.width)} "
                f"pred={short(row.get('pred_answer', ''), args.width)} "
                f"eos={row.get('eos_present')} "
                f"eos_idx={row.get('eos_index')} "
                f"pred_tokens={row.get('pred_answer_token_len')}"
            )
            raw_answer = row.get("raw_answer_block_prefix")
            if raw_answer and args.raw_width > 0:
                print(f"    raw={short(raw_answer, args.raw_width)}")


if __name__ == "__main__":
    main()
