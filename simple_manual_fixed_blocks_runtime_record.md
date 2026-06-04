# Simple Manual Fixed-Blocks Runtime Record

## 2026-06-03 11:53 CST - dataset build

Source files:
- `/Users/jinc_air/Documents/resume/maril/server_samples/s1K_manual_synthetic_200.json`
- `/Users/jinc_air/Documents/resume/maril/server_samples/s1K_manual_synthetic_200_v2.json`

Generated files:
- `/Users/jinc_air/Documents/resume/maril/server_samples/s1K_manual_synthetic_400_simple_seed10_train320_valid40_test40.json`
- `/Users/jinc_air/Documents/resume/maril/server_samples/s1K_manual_synthetic_400_simple_seed10_train.json`
- `/Users/jinc_air/Documents/resume/maril/server_samples/s1K_manual_synthetic_400_simple_seed10_valid.json`
- `/Users/jinc_air/Documents/resume/maril/server_samples/s1K_manual_synthetic_400_simple_seed10_test.json`

Split:
- seed: 10
- train: 320
- valid: 40
- test: 40

Remote locations:
- 30893: `/root/sedd/data/simple_manual_synthetic/`
- swdd_sf2: `/root/sedd/data/simple_manual_synthetic/`

swdd_sf2 fixed-layout audit with GPT-2 tokenizer:
- all rows: 400
- missing_final_answer: 0
- over_limit: 0
- format_bad: 0
- question tokens: min 12, p50 19.5, p95 30, max 40, limit 231
- reasoning tokens: min 37, p50 54.5, p95 75, max 81, limit 530
- answer+eos tokens: min 2, p50 2, p95 8, max 9, limit 263

Notes:
- Use `REASONING_FIELD=compressed_reasoning`.
- Use `FINAL_ANSWER_FIELD=compressed_answer`.
- This dataset is intentionally much easier and shorter than the 599 compressed
  DeepSeek set, to test whether the fixed-block input/output format can overfit
  before returning to long-reasoning samples.

## 2026-06-03 11:57 CST - swdd_sf2 code migration

Instance:
- name: swdd_sf2
- ssh: `ssh -p 10314 root@connect.bjb2.seetacloud.com`
- GPU: NVIDIA GeForce RTX 5090, 32607 MiB

Migration:
- Synced current local fixed-block code and scripts to `/root/sedd`.
- Excluded runtime assets: `.git`, `pretrained`, `data`, `logs`, `outputs`,
  checkpoints, and binary model weights.
- Remote backup dir for overwritten files:
  `/root/sedd_migration_backup_20260603_115445`
- `py_compile` passed for `data_sft.py`, `train_sft.py`,
  `eval_fixed_blocks.py`, `losses.py`, `eval_correct.py`, `lora.py`,
  `sft_utils.py`, and `tools/*.py`.
- GPT-2 tokenizer loads locally on swdd_sf2.

## 2026-06-03 11:58 CST - simple64 overfit launch

Run:
`fixed_blocks_simple64_manual400_compressed_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step5000`

Output dir:
`/root/autodl-tmp/sedd_outputs/lora_fixed_blocks_simple64_manual400_compressed_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step5000`

Training:
- data_json: `data/simple_manual_synthetic/s1K_manual_synthetic_400_simple_seed10_train320_valid40_test40.json`
- train_size: 64
- valid_size: 0
- test_size: 0
- seed: 10
- fixed layout: question `[0,231)`, reasoning `[231,761)`,
  final answer `[761,1024)`
- reasoning_field: `compressed_reasoning`
- final_answer_field: `compressed_answer`
- fixed_layout_supervise_pad: 1
- fixed_layout_final_answer_weight: 4.0
- fixed_layout_final_answer_pad_weight: 0.25
- batch_size: 16
- grad_accum: 8
- max_steps: 5000
- save_freq: 500
- log_freq: 10
- all_mask_ce_weight: 0

Initial status:
- train pid: 1949
- step=10: loss=6.1724, dwdse=6.1724, lr=6.00e-05
- step=20: loss=1.3009, dwdse=1.3009, lr=1.20e-04
- step=30: loss=0.7298, dwdse=0.7298, lr=1.80e-04
- step=40: loss=0.5528, dwdse=0.5528, lr=2.40e-04
- step=50: loss=0.4874, dwdse=0.4874, lr=3.00e-04
- step=60: loss=0.4657, dwdse=0.4657, lr=3.00e-04
- gpu_memory=23.3GB
- gpu_util=96-99%

Next checkpoint to test:
`checkpoint_step_500.pt`

Step500 one-shot eval watcher:
- watcher pid: 2258
- watcher log:
  `logs/fixed_blocks/watch_eval_fixed_blocks_simple64_manual400_compressed_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step5000_step500.log`
- eval jsonl:
  `outputs/fixed_blocks_eval/fixed_blocks_simple64_manual400_compressed_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step5000_step500_train16_argmax64.jsonl`
- eval mode: train16, argmax64, eval_batch_size=1

## 2026-06-03 12:25 CST - checkpoint500 train16 eval

Training status:
- process still running: pid 1949
- checkpoint saved:
  `/root/autodl-tmp/sedd_outputs/lora_fixed_blocks_simple64_manual400_compressed_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step5000/checkpoint_step_500.pt`
- step=500: loss=0.1046, dwdse=0.1046
- latest observed step=520: loss=0.0984, dwdse=0.0984
- gpu_memory=23.9GB/32GB
- gpu_util=99%

Eval:
- checkpoint: `checkpoint_step_500.pt`
- split: train
- limit: 16
- sampling: argmax
- steps: 64
- eval_batch_size: 1
- out_jsonl:
  `outputs/fixed_blocks_eval/fixed_blocks_simple64_manual400_compressed_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step5000_step500_train16_argmax64.jsonl`

Result:
- answer_block_match: 0/16 = 0.0%
- exact_answer_match: 0/16 = 0.0%
- empty_answer_blocks: 12/16 = 75.0%
- no_eos_answer_blocks: 0/16 = 0.0%
- avg_answer_tokens: 0.5

Observed predictions:
- `5 -> ""`
- `\frac{8}{17} -> "fracfrac{{{"`
- `5 -> ""`
- `17 -> ""`
- `48 -> ""`
- `220/13 -> "//"`
- `32 -> "3232"`
- `42 -> ""`
- `8 -> ""`
- `8 -> ""`
- `8\pi -> "\\"`
- `6 -> ""`
- `156 -> ""`
- `3 -> ""`
- `47 -> ""`
- `7 -> ""`

Conclusion:
- Teacher-forcing DWDSE loss fits the simple samples quickly.
- Generation is still not overfit at checkpoint500; answer blocks are mostly EOS
  immediately, with a few short repeated fragments.
- Continue to checkpoint1000 before judging the simple-sample fixed-block setup.

Step1000 one-shot eval watcher:
- watcher pid: 3224
- watcher log:
  `logs/fixed_blocks/watch_eval_fixed_blocks_simple64_manual400_compressed_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step5000_step1000.log`
- eval jsonl:
  `outputs/fixed_blocks_eval/fixed_blocks_simple64_manual400_compressed_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step5000_step1000_train16_argmax64.jsonl`
- eval mode: train16, argmax64, eval_batch_size=1

## 2026-06-03 12:40 CST - switch to sft2-only small-model framework tests

Decision:
- Do not update or launch new jobs on the other server for now.
- Continue testing only on swdd_sf2/sft2.
- Medium is slow and has not shown generation overfit yet, so use
  `pretrained/sedd-small` to validate the fixed-block training framework first.
- If small can overfit simple samples, port the working settings back to medium.

Code changes synced to sft2 only:
- `losses.py`
- `train_sft.py`
- `run_fixed_blocks_sft.sh`
- `launch_simple_small_after_medium1000.sh`

New training controls:
- `--sft_num_t_per_sample`: average multiple independently sampled t values per
  batch.
- `--sft_t_grid`: fixed t grid; overrides random multi-t.
- Shell envs: `SFT_NUM_T_PER_SAMPLE`, `SFT_T_GRID`.

Controller:
- old medium step1000 watcher pid 3224 was stopped.
- new controller pid: 3750
- controller log:
  `logs/fixed_blocks/controller_simple_small_after_medium1000.log`
- behavior:
  1. wait for medium `checkpoint_step_1000.pt`;
  2. stop the medium training process;
  3. evaluate medium checkpoint1000 on train16/argmax64;
  4. run small baseline random1 to step1000;
  5. run small random4 same-forward-budget to step250;
  6. run small fixed grid `[0.001,0.01,0.1,0.6]` same-forward-budget to step250.

Medium status before controller handoff:
- latest observed step=790
- loss=0.0779
- gpu_memory=23.9GB/32GB
- gpu_util=90%

## 2026-06-03 12:50 CST - multi-t implementation corrected

Implementation requirement:
- `train_sft.py` owns multi-t averaging.
- For each batch, repeat `loss_fn` K times and backpropagate
  `loss / K / grad_accum` each time.
- `losses.py` response-only loss handles exactly one t per call and accepts:
  - `t=None`
  - `perturbed_batch=None`

## 2026-06-03 16:14 CST - full simple400 short-window small run

Decision:
- The long 1024-token fixed window over-supervised EOS padding and caused
  answer collapse.
- The short fixed window covers all 400 simple samples without truncation:
  question max 40 <= 48, reasoning max 81 <= 96, answer+eos max 9 <= 16.
- Keep the first successful overfit recipe and scale from train64 to the full
  split.

Run:
`fixed_blocks_simple400_small_shortwin_q48_r96_a16_fa4pad025_seed10_train320_valid40_test40_r32a64_b32ga4_step10000`

Output dir:
`/root/autodl-tmp/sedd_outputs/lora_fixed_blocks_simple400_small_shortwin_q48_r96_a16_fa4pad025_seed10_train320_valid40_test40_r32a64_b32ga4_step10000`

Logs:
- train:
  `/root/sedd/logs/fixed_blocks/train_fixed_blocks_simple400_small_shortwin_q48_r96_a16_fa4pad025_seed10_train320_valid40_test40_r32a64_b32ga4_step10000.log`
- nohup:
  `/root/sedd/logs/fixed_blocks/nohup_fixed_blocks_simple400_small_shortwin_q48_r96_a16_fa4pad025_seed10_train320_valid40_test40_r32a64_b32ga4_step10000.log`
- pid:
  `/root/sedd/logs/fixed_blocks/fixed_blocks_simple400_small_shortwin_q48_r96_a16_fa4pad025_seed10_train320_valid40_test40_r32a64_b32ga4_step10000.pid`

Training:
- data_json:
  `data/simple_manual_synthetic/s1K_manual_synthetic_400_simple_seed10_train320_valid40_test40.json`
- split: train 320, valid 40, test 40
- base: `pretrained/sedd-small`
- max_length: 160
- fixed layout: question `[0,48)`, reasoning `[48,144)`,
  final answer `[144,160)`
- reasoning_field: `compressed_reasoning`
- final_answer_field: `compressed_answer`
- fixed_layout_supervise_pad: 1
- fixed_layout_final_answer_weight: 4.0
- fixed_layout_final_answer_pad_weight: 0.25
- all_mask_ce_weight: 0
- sft_num_t_per_sample: 1
- batch_size: 32
- grad_accum: 4
- effective update size: 128 samples
- max_steps: 10000
- save_freq: 250
- eval_freq: 250
- log_freq: 10
- lr: 3e-4
- warmup_steps: 50
- LoRA: r32 alpha64 dropout0.0

Initial status:
- pid: 11806
- step 10: loss=6.2830
- step 50: loss=1.9148
- step 100: loss=1.2885
- step 150: loss=0.9824
- step 200: loss=0.8949
- step 250: loss=0.8293
- step 250 valid_loss=0.6759
- saved: `checkpoint_step_250.pt`
- gpu_memory: about 4.7GB / 32GB

Code changes:
- `train_sft.py`
  - added canonical `--num_t_per_sample`;
  - kept compatibility alias `--sft_num_t_per_sample`;
  - parses `--sft_t_grid`;
  - builds `dwdse_t_values = t_grid if provided else [None] * num_t`;
  - loops over `dwdse_t_values` for the same batch;
  - fixed grid uses `torch.full((batch_size,), t_value, device=device)`;
  - random multi-t passes `t=None`, so loss samples a fresh random t per call.
- `losses.py`
  - removed internal multi-t averaging from response-only loss;
  - added `t=None` and `perturbed_batch=None` to response-only loss;
  - when `t` is provided, uses that fixed t for the batch;
  - when `perturbed_batch` is provided, uses it instead of sampling transition.
- `run_fixed_blocks_sft.sh`
  - passes `--num_t_per_sample`.

Verification:
- local `python3 -m py_compile losses.py train_sft.py`: passed
- local `bash -n run_fixed_blocks_sft.sh launch_simple_small_after_medium1000.sh`: passed
- sft2 `/root/miniconda3/bin/python -m py_compile losses.py train_sft.py`: passed
- sft2 `bash -n run_fixed_blocks_sft.sh launch_simple_small_after_medium1000.sh`: passed

sft2 status after sync:
- controller pid: 3750
- medium latest observed step=910
- medium loss=0.0626
- controller still waiting for medium `checkpoint_step_1000.pt`

## 2026-06-03 12:53 CST - medium checkpoint1000 and small baseline launch

Medium checkpoint1000:
- checkpoint:
  `/root/autodl-tmp/sedd_outputs/lora_fixed_blocks_simple64_manual400_compressed_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step5000/checkpoint_step_1000.pt`
- step=1000
- loss=0.0603
- dwdse=0.0603

Medium train16/argmax64 eval:
- out_jsonl:
  `outputs/fixed_blocks_eval/fixed_blocks_simple64_manual400_compressed_answer_fa4pad025_q231_r530_a263_dwdseonly_seed10_r32a64_b16ga8_step5000_step1000_train16_argmax64.jsonl`
- answer_block_match: 0/16 = 0.0%
- exact_answer_match: 0/16 = 0.0%
- empty_answer_blocks: 10/16 = 62.5%
- no_eos_answer_blocks: 0/16 = 0.0%
- avg_answer_tokens: 0.9

Observed medium predictions:
- `5 -> ""`
- `\frac{8}{17} -> "\\{{{"`
- `5 -> ""`
- `17 -> ""`
- `48 -> ""`
- `220/13 -> "//"`
- `32 -> "3232"`
- `42 -> ""`
- `8 -> ""`
- `8 -> "88"`
- `8\pi -> "\\pipipi"`
- `6 -> ""`
- `156 -> "156156"`
- `3 -> ""`
- `47 -> ""`
- `7 -> ""`

Conclusion:
- Medium DWDSE loss reached about 0.06 on simple64, but generation still does
  not overfit. It improved slightly from checkpoint500 but remains mostly empty
  or duplicated answer fragments.

Small baseline launch:
- run:
  `fixed_blocks_simple64_small_baseline_random1_q231_r530_a263_seed10_r32a64_b16ga8_step1000`
- pretrained: `pretrained/sedd-small`
- train pid: 4405
- params:
  - num_t_per_sample: 1
  - t_grid: random
  - max_steps: 1000
  - save_freq: 500
  - batch_size: 16
  - grad_accum: 8
  - DWDSE-only
- initial gpu_memory: 13.2GB/32GB

Small baseline early training:
- step=10: loss=8.0453, dwdse=8.0453
- step=20: loss=2.4662, dwdse=2.4662
- step=30: loss=1.3157, dwdse=1.3157
- step=40: loss=0.6702, dwdse=0.6702
- step=50: loss=0.5564, dwdse=0.5564
- step=60: loss=0.5268, dwdse=0.5268
- gpu_memory=13.2GB/32GB
- gpu_util=96%

## 2026-06-03 13:18 CST - small baseline1000 eval and random4 progress

Small baseline random1 checkpoint1000:
- run:
  `fixed_blocks_simple64_small_baseline_random1_q231_r530_a263_seed10_r32a64_b16ga8_step1000`
- checkpoint:
  `/root/autodl-tmp/sedd_outputs/lora_fixed_blocks_simple64_small_baseline_random1_q231_r530_a263_seed10_r32a64_b16ga8_step1000/checkpoint_step_1000.pt`
- final observed step=1000
- loss=0.1061
- dwdse=0.1061
- gpu_memory=13.7GB/32GB

Small baseline random1 loss trend:
- step=500: loss=0.1703
- step=650: loss=0.1334
- step=800: loss=0.1180
- step=880: loss=0.1111
- step=940: loss=0.1035
- step=980: loss=0.1065
- step=1000: loss=0.1061

Small baseline train16/argmax64 eval:
- out_jsonl:
  `outputs/fixed_blocks_eval/fixed_blocks_simple64_small_baseline_random1_q231_r530_a263_seed10_r32a64_b16ga8_step1000_step1000_train16_argmax64.jsonl`
- answer_block_match: 0/16 = 0.0%
- exact_answer_match: 0/16 = 0.0%
- empty_answer_blocks: 14/16 = 87.5%
- no_eos_answer_blocks: 0/16 = 0.0%
- avg_answer_tokens: 0.2

Observed small baseline predictions:
- `5 -> ""`
- `\frac{8}{17} -> "\\{{{"`
- `5 -> ""`
- `17 -> ""`
- `48 -> ""`
- `220/13 -> "//"`
- `32 -> ""`
- `42 -> ""`
- `8 -> ""`
- `8 -> ""`
- `8\pi -> ""`
- `6 -> ""`
- `156 -> ""`
- `3 -> ""`
- `47 -> ""`
- `7 -> ""`

Conclusion:
- The small random1 baseline is cheaper and loss descends normally, but
  generation is worse than medium checkpoint1000: most answer blocks emit EOS at
  index 0.

Small random4 same-forward-budget progress:
- run:
  `fixed_blocks_simple64_small_random4_samefwd250_q231_r530_a263_seed10_r32a64_b16ga8_step250`
- pretrained: `pretrained/sedd-small`
- num_t_per_sample: 4
- t_grid: random
- max_steps: 250
- same forward budget as random1 step1000
- current observed process pid: 5455
- latest observed step=60
- loss trend:
  - step=10: loss=7.9755
  - step=20: loss=2.4843
  - step=30: loss=0.9962
  - step=40: loss=0.6949
  - step=50: loss=0.5744
  - step=60: loss=0.5040
- gpu_memory=13.4GB/32GB
- gpu_util=98%

## 2026-06-03 13:37 CST - small random4 finished and fixed-grid launched

Small random4 same-forward-budget checkpoint250:
- run:
  `fixed_blocks_simple64_small_random4_samefwd250_q231_r530_a263_seed10_r32a64_b16ga8_step250`
- checkpoint:
  `/root/autodl-tmp/sedd_outputs/lora_fixed_blocks_simple64_small_random4_samefwd250_q231_r530_a263_seed10_r32a64_b16ga8_step250/checkpoint_step_250.pt`
- final step=250
- loss=0.2175
- dwdse=0.2175
- gpu_memory=13.7GB/32GB

Small random4 loss trend:
- step=50: loss=0.5744
- step=100: loss=0.3748
- step=150: loss=0.2891
- step=200: loss=0.2371
- step=240: loss=0.2197
- step=250: loss=0.2175

Small random4 train16/argmax64 eval:
- out_jsonl:
  `outputs/fixed_blocks_eval/fixed_blocks_simple64_small_random4_samefwd250_q231_r530_a263_seed10_r32a64_b16ga8_step250_step250_train16_argmax64.jsonl`
- answer_block_match: 0/16 = 0.0%
- exact_answer_match: 0/16 = 0.0%
- empty_answer_blocks: 16/16 = 100.0%
- no_eos_answer_blocks: 0/16 = 0.0%
- avg_answer_tokens: 0.0

Conclusion:
- Multi-random-t with the same forward budget did not improve generation.
- It reduced DWDSE more slowly than random1 step1000 and collapsed to immediate
  EOS in every answer block.

Small fixed-grid same-forward-budget launch:
- run:
  `fixed_blocks_simple64_small_tgrid001_01_1_6_samefwd250_q231_r530_a263_seed10_r32a64_b16ga8_step250`
- pretrained: `pretrained/sedd-small`
- max_steps: 250
- effective t_grid: `(0.001, 0.01, 0.1, 0.6)`
- controller command passed:
  `--num_t_per_sample 1 --sft_t_grid 0.001,0.01,0.1,0.6`
- latest observed:
  - step=20
  - loss=2.0407
  - gpu_memory=12.5GB/32GB
  - gpu_util=97%

## 2026-06-03 13:57 CST - fixed-grid finished

Small fixed-grid same-forward-budget checkpoint250:
- run:
  `fixed_blocks_simple64_small_tgrid001_01_1_6_samefwd250_q231_r530_a263_seed10_r32a64_b16ga8_step250`
- checkpoint:
  `/root/autodl-tmp/sedd_outputs/lora_fixed_blocks_simple64_small_tgrid001_01_1_6_samefwd250_q231_r530_a263_seed10_r32a64_b16ga8_step250/checkpoint_step_250.pt`
- effective t_grid: `(0.001, 0.01, 0.1, 0.6)`
- final step=250
- loss=0.1386
- dwdse=0.1386
- gpu_memory=12.5GB/32GB

Small fixed-grid loss trend:
- step=50: loss=0.4396
- step=100: loss=0.3862
- step=150: loss=0.2153
- step=180: loss=0.1796
- step=200: loss=0.1935
- step=210: loss=0.1627
- step=230: loss=0.1446
- step=240: loss=0.2041
- step=250: loss=0.1386

Small fixed-grid train16/argmax64 eval:
- out_jsonl:
  `outputs/fixed_blocks_eval/fixed_blocks_simple64_small_tgrid001_01_1_6_samefwd250_q231_r530_a263_seed10_r32a64_b16ga8_step250_step250_train16_argmax64.jsonl`
- answer_block_match: 0/16 = 0.0%
- exact_answer_match: 0/16 = 0.0%
- empty_answer_blocks: 16/16 = 100.0%
- no_eos_answer_blocks: 0/16 = 0.0%
- avg_answer_tokens: 0.0

Comparison of small experiments:
- random1 baseline step1000:
  - forward budget: 1000 random t calls
  - final loss: 0.1061
  - train16 answer match: 0/16
  - empty answer blocks: 14/16
- random4 same-forward step250:
  - forward budget: 250 * 4 random t calls
  - final loss: 0.2175
  - train16 answer match: 0/16
  - empty answer blocks: 16/16
- fixed-grid same-forward step250:
  - forward budget: 250 * 4 fixed-grid t calls
  - final loss: 0.1386
  - train16 answer match: 0/16
  - empty answer blocks: 16/16

Conclusion:
- Fixed t grid improves DWDSE loss versus random4 at the same forward budget,
  but does not fix generation. The answer block still collapses to immediate EOS
  on all 16 train samples.
- The controller completed all configured sft2 experiments and exited.
- sft2 GPU was idle after completion.
