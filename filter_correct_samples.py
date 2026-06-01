"""
筛选 s1K-1.1 中满足以下条件的样本：
1. deepseek_grade == "Yes"（答案正确）
2. deepseek_attempt 中包含 \boxed{}（有明确答案标记）

保存到 data/s1K_correct.json。
"""

import json
import os
import re
from collections import Counter
from datasets import load_dataset

os.makedirs("data", exist_ok=True)


def extract_boxed(text):
    results = []
    pos = 0
    text = text or ""
    while pos < len(text):
        idx = text.find("\\boxed{", pos)
        if idx == -1:
            break
        start = idx + 7
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start : i - 1].strip())
        pos = i
    return results[-1] if results else None


print("Loading s1K-1.1 ...")
ds = load_dataset("simplescaling/s1K-1.1")["train"]
print(f"Total samples: {len(ds)}")

grade_counter = Counter(ds["deepseek_grade"])
print("\n=== deepseek_grade distribution ===")
for grade, count in sorted(grade_counter.items(), key=lambda x: -x[1]):
    print(f"  {grade!r:20s}: {count:4d}  ({count / len(ds) * 100:.1f}%)")

correct_grades = {"Yes"}

keep_fields = [
    "question",
    "solution",
    "deepseek_thinking_trajectory",
    "deepseek_attempt",
    "deepseek_grade",
    "deepseek_grade_reason",
    "source_type",
    "metadata",
]

correct_records = []
no_boxed_count = 0
incorrect_count = 0

for i in range(len(ds)):
    row = ds[i]
    if row["deepseek_grade"] not in correct_grades:
        incorrect_count += 1
        continue
    boxed = extract_boxed(row.get("deepseek_attempt", ""))
    if boxed is None:
        no_boxed_count += 1
        continue
    record = {k: row[k] for k in keep_fields if k in row}
    correct_records.append(record)

print(f"\nFiltering results:")
print(f"  Total:                  {len(ds)}")
print(f"  Incorrect (grade!=Yes): {incorrect_count}")
print(f"  Correct but no boxed:   {no_boxed_count}")
print(f"  Final (correct+boxed):  {len(correct_records)}")

with open("data/s1K_correct.json", "w", encoding="utf-8") as f:
    json.dump(correct_records, f, ensure_ascii=False, indent=2)

print(f"\n=== 抽样检查（前 5 条）===")
for i in range(min(5, len(correct_records))):
    r = correct_records[i]
    sol = r.get("solution", "").strip()
    ds_boxed = extract_boxed(r.get("deepseek_attempt", ""))
    print(f"  [{i}] solution={sol[:60]!r}  ds_boxed={ds_boxed!r}")

print(f"\nSaved: data/s1K_correct.json — {len(correct_records)} samples")
