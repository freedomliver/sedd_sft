#!/usr/bin/env python3
import argparse
import json
import re
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


def max_char_run(text: str) -> int:
    best = 0
    prev = None
    cur = 0
    for ch in str(text or ""):
        if ch == prev:
            cur += 1
        else:
            prev = ch
            cur = 1
        best = max(best, cur)
    return best


def repeated_token(text: str, min_repeats: int) -> str | None:
    tokens = re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", str(text or ""))
    if not tokens:
        return None
    cur = tokens[0]
    count = 1
    for tok in tokens[1:]:
        if tok == cur:
            count += 1
            if count >= min_repeats:
                return tok
        else:
            cur = tok
            count = 1
    return None


def classify(row: dict, args) -> list[str]:
    pred = str(row.get("pred_answer", "") or "")
    labels = []
    if row.get("match"):
        labels.append("match")
    else:
        labels.append("mismatch")
    if not pred.strip():
        labels.append("empty_pred")
    if not row.get("eos_present"):
        labels.append("no_eos")
    eos_index = row.get("eos_index")
    if eos_index is not None and int(eos_index) > args.late_eos_tokens:
        labels.append("late_eos")
    pred_tokens = int(row.get("pred_answer_token_len", 0) or 0)
    if pred_tokens > args.long_pred_tokens:
        labels.append("long_pred")
    if max_char_run(pred) >= args.repeat_char_run:
        labels.append("char_repeat")
    rep = repeated_token(pred, args.repeat_token_count)
    if rep is not None:
        labels.append("token_repeat")
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze fixed-block eval JSONL failure modes")
    parser.add_argument("jsonl")
    parser.add_argument("--show", type=int, default=12)
    parser.add_argument("--width", type=int, default=140)
    parser.add_argument("--raw-width", type=int, default=140)
    parser.add_argument("--late-eos-tokens", type=int, default=64)
    parser.add_argument("--long-pred-tokens", type=int, default=32)
    parser.add_argument("--repeat-char-run", type=int, default=8)
    parser.add_argument("--repeat-token-count", type=int, default=4)
    args = parser.parse_args()

    path = Path(args.jsonl)
    rows = load_rows(path)
    total = len(rows)
    if total == 0:
        print(f"file: {path}")
        print("rows: 0")
        return 2

    label_counts = Counter()
    combo_counts = Counter()
    failure_rows = []
    for row in rows:
        labels = classify(row, args)
        label_counts.update(labels)
        combo_counts.update(["+".join(label for label in labels if label != "match")])
        if not row.get("match"):
            failure_rows.append((row, labels))

    match = label_counts["match"]
    failures = total - match
    print(f"file: {path}")
    print(f"rows: {total}")
    print(f"match: {match}/{total} = {match / total:.1%}")
    print(f"failures: {failures}/{total} = {failures / total:.1%}")
    print("label_counts:")
    for key, value in label_counts.most_common():
        print(f"  {key}: {value}/{total} = {value / total:.1%}")
    print("failure_mode_combos:")
    for key, value in combo_counts.most_common(12):
        if key == "":
            continue
        print(f"  {key}: {value}")

    if failure_rows:
        print(f"failure_examples shown={min(args.show, len(failure_rows))}/{len(failure_rows)}")
        for row, labels in failure_rows[: args.show]:
            pred = row.get("pred_answer", "")
            raw = row.get("raw_answer_block_prefix", "")
            print(
                f"[{row.get('idx')}] labels={','.join(labels)} "
                f"gold={short(row.get('gold_answer', ''), args.width)} "
                f"pred={short(pred, args.width)} "
                f"eos={row.get('eos_present')} eos_idx={row.get('eos_index')} "
                f"pred_tokens={row.get('pred_answer_token_len')}"
            )
            if raw and args.raw_width > 0:
                print(f"    raw={short(raw, args.raw_width)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
