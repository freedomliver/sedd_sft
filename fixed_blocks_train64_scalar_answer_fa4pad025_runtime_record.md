# Fixed Blocks Train64 Scalar Answer FA4 Pad025 Runtime Record

## 2026-06-03 09:35 CST - checkpoint1000 test of previous answer_fa4pad025 run

Previous run:
`fixed_blocks_train64_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step10000`

The run reached `checkpoint_step_1000.pt`, but a lightweight eval exposed that
the final-answer target was still wrong.

Eval command shape:
- checkpoint: `checkpoint_step_1000.pt`
- split: train
- limit: 16
- sampling: argmax
- steps: 64
- eval_batch_size: 1

Result:
- answer_block_match: 0/16 = 0.0%
- exact_answer_match: 0/16 = 0.0%
- empty_answer_blocks: 8/16 = 50.0%
- no_eos_answer_blocks: 0/16 = 0.0%
- avg_answer_tokens: 56.0

Diagnosis:
- The eval gold answer for the first 12 train samples was the full
  `Method...Steps...Answer: \boxed{...}` text, not the scalar answer.
- This means the `answer` fallback fixed non-empty coverage, but still allowed
  long solution text into the fixed final-answer block.
- Continuing that run would train the wrong target, so it was stopped at about
  step1040.

## 2026-06-03 09:48 CST - scalar final-answer parser fix

Code fix:
- Added `get_final_answer_text()` in `data_sft.py`.
- Fixed-layout final answer now uses this parser instead of raw
  `get_answer_text()`.
- Long answer/solution candidates first extract the last `\boxed{...}` inner
  value.
- Short metadata fields such as `answer`, `Correct Answer`, and
  `Pre-Revision Correct Answer` remain usable as scalar answers.
- Singleton-list strings such as `['$0.0254$']` are unwrapped.

Local audit:
- rows: 599
- missing_final_answer: 0
- source_counts:
  - solution: 432
  - metadata.answer: 115
  - metadata.Correct Answer: 28
  - metadata.final_answer: 24

Seed-10 train16 scalar answer check:
- 0: `61`
- 1: `AD`
- 2: `626`
- 3: `263`
- 4: `567`
- 5: `2 + 2\sqrt{2}`
- 6: `16`
- 7: `f(x)=k`
- 8: `151`
- 9: `0.47`
- 10: `ACD`
- 11: `C`
- 12: `15`
- 13: `BCD`
- 14: `\triangle ABC`
- 15: `-2013`

Next run:
`fixed_blocks_train64_scalar_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step10000`

Intended parameters:
- TRAIN_SIZE=64
- VALID_SIZE=0
- TEST_SIZE=0
- MAX_STEPS=10000
- BATCH_SIZE=16
- GRAD_ACCUM=8
- SAVE_FREQ=1000
- LOG_FREQ=10
- FINAL_ANSWER_FIELD=answer
- FIXED_LAYOUT_SUPERVISE_PAD=1
- FIXED_LAYOUT_FINAL_ANSWER_WEIGHT=4.0
- FIXED_LAYOUT_FINAL_ANSWER_PAD_WEIGHT=0.25

## 2026-06-03 09:43 CST - scalar-answer run launched

Run:
`fixed_blocks_train64_scalar_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step10000`

Output dir:
`/root/autodl-tmp/sedd_outputs/lora_fixed_blocks_train64_scalar_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step10000`

Processes:
- train pid: 76660
- eval watcher pid: 76661
- gate watcher pid: 76662

Full599 run configured for gate pass:
`fixed_blocks_full599_scalar_answer_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step100000`

Initial status:
- step=10
- loss=5.5738
- dwdse=5.5738
- gpu_memory=23675/32607 MiB
- gpu_util=83%

Remote formatted-target check for seed-10 train16:
- 0: `61`, len=2
- 1: `AD`, len=2
- 2: `626`, len=2
- 3: `263`, len=2
- 4: `567`, len=3
- 5: `2 + 2\sqrt{2}`, len=10
- 6: `16`, len=2
- 7: `f(x)=k`, len=6
- 8: `151`, len=2
- 9: `0.47`, len=4
- 10: `ACD`, len=3
- 11: `C`, len=2
- 12: `15`, len=2
- 13: `BCD`, len=3
- 14: `\triangle ABC`, len=5
- 15: `-2013`, len=3

## 2026-06-03 11:10 CST - checkpoint1000 train16 eval

Checkpoint:
`/root/autodl-tmp/sedd_outputs/lora_fixed_blocks_train64_scalar_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step10000/checkpoint_step_1000.pt`

Eval:
- split: train
- limit: 16
- sampling: argmax
- steps: 64
- eval_batch_size: 1
- out_jsonl: `outputs/fixed_blocks_eval/fixed_blocks_train64_scalar_answer_fa4pad025_step1000_train16_argmax64.jsonl`

Result:
- answer_block_match: 0/16 = 0.0%
- exact_answer_match: 0/16 = 0.0%
- empty_answer_blocks: 9/16 = 56.2%
- no_eos_answer_blocks: 0/16 = 0.0%
- avg_answer_tokens: 0.8

Observed predictions:
- `61 -> ""`
- `AD -> ""`
- `626 -> ""`
- `263 -> ""`
- `567 -> "6767"`
- `2 + 2\sqrt{2} -> "22}}}"`
- `16 -> ""`
- `f(x)=k -> "ffxxx"`
- `151 -> ""`
- `0.47 -> ".."`
- `ACD -> "ACAC"`
- `C -> ""`
- `15 -> ""`
- `BCD -> "BCBC"`
- `\triangle ABC -> ""`
- `-2013 -> "--"`

Conclusion:
- The gold target is now correct scalar final answer text.
- `checkpoint_step_1000.pt` still fails generation overfit; the answer block is
  mostly empty or short repetition.
- Continue training and retest at `checkpoint_step_2000.pt` before judging this
  scalar-answer run.

## 2026-06-03 11:55 CST - checkpoint2000 train16 eval

Checkpoint:
`/root/autodl-tmp/sedd_outputs/lora_fixed_blocks_train64_scalar_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step10000/checkpoint_step_2000.pt`

Eval:
- split: train
- limit: 16
- sampling: argmax
- steps: 64
- eval_batch_size: 1
- out_jsonl: `outputs/fixed_blocks_eval/fixed_blocks_train64_scalar_answer_fa4pad025_step2000_train16_argmax64.jsonl`

Result:
- answer_block_match: 0/16 = 0.0%
- exact_answer_match: 0/16 = 0.0%
- empty_answer_blocks: 5/16 = 31.2%
- no_eos_answer_blocks: 0/16 = 0.0%
- avg_answer_tokens: 1.6

Observed predictions:
- `61 -> ""`
- `AD -> "ADAD"`
- `626 -> ""`
- `263 -> ""`
- `567 -> "6767"`
- `2 + 2\sqrt{2} -> "22 2 2 2"`
- `16 -> ""`
- `f(x)=k -> "ff)=)=)="`
- `151 -> "151151"`
- `0.47 -> "00"`
- `ACD -> "ACAC"`
- `C -> "CC"`
- `15 -> ""`
- `BCD -> "BCBC"`
- `\triangle ABC -> "tritri"`
- `-2013 -> "--"`

Conclusion:
- Loss continues to decrease normally, and the model now often emits short
  non-empty answer-block text.
- Generation is still not overfitting: failures shifted from mostly empty
  blocks toward duplicated answer fragments.
- Keep the current 30893 training running, but use the new swdd_sf2 instance
  for simple-sample overfit tests to isolate whether the fixed layout is
  learnable on easier data.

## 2026-06-03 13:39 CST - stopped at checkpoint4000

Reason:
- The run had reached `checkpoint_step_4000.pt`, but loss had plateaued around
  0.38-0.42 after step3000.
- Prior generation evals at step1000 and step2000 were both 0/16.
- The newer simple-sample sft2 experiments are now the higher-signal direction.

Stopped processes on 30893:
- train pid: 76675
- eval watcher pid: 76661
- full599 gate watcher pid: 76662

Final available checkpoint:
`/root/autodl-tmp/sedd_outputs/lora_fixed_blocks_train64_scalar_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step10000/checkpoint_step_4000.pt`

Checkpoint4000 train16 eval:
- split: train
- limit: 16
- sampling: argmax
- steps: 64
- out_jsonl:
  `outputs/fixed_blocks_eval/fixed_blocks_train64_scalar_answer_fa4pad025_step4000_train16_argmax64.jsonl`
- log:
  `logs/fixed_blocks/eval_fixed_blocks_train64_scalar_answer_fa4pad025_step4000_train16_argmax64.log`

Result:
- answer_block_match: 0/16 = 0.0%
- exact_answer_match: 0/16 = 0.0%
- empty_answer_blocks: 5/16 = 31.2%
- no_eos_answer_blocks: 0/16 = 0.0%
- avg_answer_tokens: 1.8

Observed predictions:
- `61 -> ""`
- `AD -> "ADAD"`
- `626 -> "626626"`
- `263 -> ""`
- `567 -> "55"`
- `2 + 2\sqrt{2} -> "22sqsqsq"`
- `16 -> ""`
- `f(x)=k -> "(()=)=)="`
- `151 -> "151151"`
- `0.47 -> "00"`
- `ACD -> "ACAC"`
- `C -> ""`
- `15 -> ""`
- `BCD -> "BCBC"`
- `\triangle ABC -> "tritri"`
- `-2013 -> "20132013"`

Conclusion:
- Continuing to step10000 is not justified. The run still cannot overfit
  train16 generation after step4000, and duplicated answer fragments remain the
  dominant failure mode.
- GPU on 30893 was released after stopping.
