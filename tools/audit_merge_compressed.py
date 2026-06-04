#!/usr/bin/env python3
"""Revalidate and merge compressed S1K part outputs.

The per-part compressor writes validation results at generation time. This
auditor intentionally recomputes validation with the current code so old false
negatives do not leak into the final train JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from compress_s1k import validate_compression


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_candidates(base_dir: Path) -> dict[int, dict[str, Any]]:
    candidates: dict[int, dict[str, Any]] = {}
    priority_paths: list[Path] = []
    priority_paths.extend(sorted(base_dir.glob("part_*/output.jsonl")))
    priority_paths.extend(sorted(base_dir.glob("repair*/output.jsonl")))
    priority_paths.extend(sorted(base_dir.glob("manual*/output.jsonl")))
    for path in priority_paths:
        for row in read_jsonl(path):
            if "compressed" not in row:
                continue
            try:
                index = int(row["index"])
            except Exception:
                continue
            row["_source_file"] = str(path)
            candidates[index] = row
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--input", default="data/s1K_train_599.json")
    parser.add_argument("--max_question_tokens", type=int, default=220)
    parser.add_argument("--max_reasoning_tokens", type=int, default=512)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    samples = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise TypeError(f"Expected list input, got {type(samples).__name__}")

    candidates = load_candidates(base_dir)
    ok_rows: list[dict[str, Any]] = []
    bad_rows: list[dict[str, Any]] = []
    missing: list[int] = []

    for index, item in enumerate(samples):
        row = candidates.get(index)
        if row is None:
            missing.append(index)
            continue
        validation = validate_compression(
            item,
            row.get("compressed") or {},
            args.max_question_tokens,
            args.max_reasoning_tokens,
        )
        row = {**row, "validation": validation}
        if validation.get("ok"):
            ok_rows.append(row)
        else:
            bad_rows.append(row)

    out_jsonl = base_dir / "merged_validated_output.jsonl"
    out_train = base_dir / "merged_train.json"
    out_errors = base_dir / "merged_validated_errors.jsonl"
    out_failed = base_dir / "failed_indices.txt"
    out_summary = base_dir / "merged_validated_summary.txt"

    out_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ok_rows + bad_rows),
        encoding="utf-8",
    )
    out_errors.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bad_rows),
        encoding="utf-8",
    )
    failed_indices = missing + [int(row["index"]) for row in bad_rows]
    out_failed.write_text("\n".join(str(i) for i in failed_indices) + ("\n" if failed_indices else ""), encoding="utf-8")

    train_rows = []
    for row in ok_rows:
        compressed = row.get("compressed") or {}
        train_rows.append(
            {
                "question": compressed.get("compressed_question", ""),
                "solution": compressed.get("compressed_reasoning", ""),
                "source_id": row.get("sample_id"),
                "metadata": row.get("metadata"),
            }
        )
    out_train.write_text(json.dumps(train_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = [
        f"total_samples {len(samples)}",
        f"candidate_rows {len(candidates)}",
        f"ok_rows {len(ok_rows)}",
        f"bad_rows {len(bad_rows)}",
        f"missing_rows {len(missing)}",
        f"failed_indices_count {len(failed_indices)}",
        f"train_rows {len(train_rows)}",
    ]
    if ok_rows or bad_rows:
        rows = ok_rows + bad_rows
        q_tokens = [(row.get("validation") or {}).get("compressed_question_tokens", 0) for row in rows]
        r_tokens = [(row.get("validation") or {}).get("compressed_reasoning_tokens", 0) for row in rows]
        lines.append(
            f"question_tokens_min_max_avg {min(q_tokens)} {max(q_tokens)} {round(sum(q_tokens) / len(q_tokens), 1)}"
        )
        lines.append(
            f"reasoning_tokens_min_max_avg {min(r_tokens)} {max(r_tokens)} {round(sum(r_tokens) / len(r_tokens), 1)}"
        )
    error_counts: dict[str, int] = {}
    for row in bad_rows:
        for error in (row.get("validation") or {}).get("errors") or []:
            error_counts[error] = error_counts.get(error, 0) + 1
    lines.append(f"error_counts {error_counts}")
    if failed_indices:
        lines.append("failed_indices_preview " + ",".join(str(i) for i in failed_indices[:50]))
    lines.extend(
        [
            f"merged_jsonl {out_jsonl}",
            f"merged_train {out_train}",
            f"merged_errors {out_errors}",
            f"failed_indices {out_failed}",
        ]
    )
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.write(out_summary.read_text(encoding="utf-8"))
    return 0 if not failed_indices else 1


if __name__ == "__main__":
    raise SystemExit(main())
