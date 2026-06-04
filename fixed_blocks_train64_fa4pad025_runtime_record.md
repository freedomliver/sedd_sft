# Fixed-Blocks Train64 FA4PAD025 Runtime Record

## Launch 2026-06-03 06:04 CST

```text
run_name: fixed_blocks_train64_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step10000
train_process: running pid=67683
eval64_watcher: running pid=67684
fixed_layout_final_answer_weight: 4.0
fixed_layout_final_answer_pad_weight: 0.25
fixed_layout_supervise_pad: true
```

## Status 2026-06-03 06:07 CST

```text
train_process: running pid=67683
eval64_watcher: running pid=67684
latest_step: 50 / 10000
latest_loss: 2.2264
gpu: 23695 / 32607 MiB, util 97%
eval64_jsonl: missing
```

## Status 2026-06-03 06:10 CST

```text
train_process: running pid=67683
eval64_watcher: running pid=67684
latest_step: 100 / 10000
latest_loss: 1.9266
gpu: 23695 / 32607 MiB, util 97%
eval64_jsonl: missing
```

## Status 2026-06-03 06:16 CST

```text
train_process: running pid=67683
eval64_watcher: running pid=67684
latest_step: 200 / 10000
latest_loss: 1.8049
gpu: 23915 / 32607 MiB, util 99%
eval64_jsonl: missing
```

## Status 2026-06-03 06:21 CST

```text
train_process: running pid=67683
eval64_watcher: running pid=67684
latest_step: 300 / 10000
latest_loss: 1.7139
gpu: 23915 / 32607 MiB, util 99%
eval64_jsonl: missing
```

## Status 2026-06-03 06:26 CST

```text
train_process: running pid=67683
eval64_watcher: running pid=67684
latest_step: 400 / 10000
latest_loss: 1.6461
gpu: 23915 / 32607 MiB, util 99%
eval64_jsonl: missing
```

## Status 2026-06-03 06:32 CST

```text
train_process: running pid=67683
eval64_watcher: running pid=67684
latest_step: 500 / 10000
latest_loss: 1.6334
gpu: 23915 / 32607 MiB, util 99%
eval64_jsonl: missing
```

## Status 2026-06-03 06:37 CST

```text
train_process: running pid=67683
eval64_watcher: running pid=67684
latest_step: 600 / 10000
latest_loss: 1.6006
gpu: 23915 / 32607 MiB, util 99%
eval64_jsonl: missing
```

## Status 2026-06-03 06:43 CST

```text
train_process: running pid=67683
eval64_watcher: running pid=67684
latest_step: 700 / 10000
latest_loss: 1.5628
gpu: 23915 / 32607 MiB, util 84%
eval64_jsonl: missing
```

## Status 2026-06-03 06:48 CST

```text
train_process: running pid=67683
eval64_watcher: running pid=67684
latest_step: 800 / 10000
latest_loss: 1.5036
gpu: 23915 / 32607 MiB, util 84%
eval64_jsonl: missing
```

## Status 2026-06-03 06:54 CST

```text
train_process: running pid=67683
eval64_watcher: running pid=67684
latest_step: 900 / 10000
latest_loss: 1.5484
gpu: 24115 / 32607 MiB, util 99%
eval64_jsonl: missing
```

## Status 2026-06-03 06:57 CST

```text
train_process: running pid=67683
eval64_watcher: running pid=67684
latest_step: 950 / 10000
latest_loss: 1.5343
gpu: 24115 / 32607 MiB, util 98%
eval64_jsonl: missing
```

## Eval64 2026-06-03 06:59 CST

```text
checkpoint: /root/autodl-tmp/sedd_outputs/lora_fixed_blocks_train64_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step10000/checkpoint_step_1000.pt
train_process: running pid=67683
eval64_jsonl: outputs/fixed_blocks_eval/fixed_blocks_train64_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step10000_train64_argmax128.jsonl
answer_block_match: 0/64 = 0.0%
exact_answer_match: 0/64 = 0.0%
empty_answer_blocks: 63/64 = 98.4%
no_eos_answer_blocks: 0/64 = 0.0%
avg_answer_tokens: 0.1
gate: FAIL
```

## Diagnosis 2026-06-03 07:10 CST

```text
root_cause: fixed-layout final_answer_field=boxed_inner resolved to empty for this dataset because merged_train.json only carries question/solution at top level; the real final answer lives in metadata fields such as answer / Correct Answer / Pre-Revision Correct Answer
effect: training was effectively supervising EOS / empty final answers, which explains the 63/64 empty pred_answer collapse
fix: data_sft.get_answer_text now falls back to metadata answer fields for boxed_inner/final_answer lookups
next_step: rerun fixed-layout 64 after code sync; keep all EOS/CE exclusions unchanged
```

## Data audit 2026-06-03 07:18 CST

```text
raw_json_audit: 599 / 599 records now resolve a non-empty final_answer after fallbacking boxed_inner -> metadata.answer / metadata.Correct Answer / metadata.Pre-Revision Correct Answer
implication: the fixed-layout final-answer target is no longer empty on this dataset, so the next 64 run should no longer be supervising EOS by accident
```

## Config update 2026-06-03 07:32 CST

```text
default_final_answer_field: answer
reason: the current dataset exposes true final answers primarily via top-level answer or metadata answer/Correct Answer fields, while boxed_inner is absent
impact: future fixed-layout launches now target the actual answer field by default instead of the misleading boxed_inner placeholder
```
