# A800 le32 Full-Data Experiment Summary

Scope: A800-only records for le32 full-data work. Target remains unseen valid accuracy >60%, about 70GB VRAM, response-only DWDSE only.

Split used for current full-data runs unless noted: train_size=522, valid_size=64, test_size=0, seed=10, data=data/s1K_train_answer_inner_le32.json.

## Current Best Verified Unseen Valid Accuracy

- Best verified valid64 generation so far: 4/64 = 6.2%.
- This is far below the >60% target.
- Low scalar DWDSE validation loss has not implied better generation accuracy.

## Key Runs

| Run | Main config | VRAM log | Best/important scalar valid loss | Verified generation | Paths |
| --- | --- | --- | --- | --- | --- |
| full_le32_t522v64_seed10_lr3e4_b54_r64_dwdse_only | LoRA r64 alpha128, batch54, lr=3e-4, boxed_prompt, response-only DWDSE | 70.1GB | step400 2.3749 | valid64 step400 4/64=6.2%; train64 step400 21/64=32.8%; argmax128 step400 also 4/64 | train log: logs/train_full_le32_t522v64_seed10_lr3e4_b54_r64_dwdse_only.log; out: /root/autodl-tmp/sedd_outputs/lora_full_le32_t522v64_seed10_lr3e4_b54_r64_dwdse_only |
| full_le32_t522v64_seed10_lr1e4_b54_r64_dwdse_only | LoRA r64 alpha128, batch54, lr=1e-4, boxed_prompt, response-only DWDSE | 70.1GB | step400 2.0134; step1000 2.0638 | valid64: step200 1/64, step300 2/64, step400 2/64, step1000 0/64 | train log: logs/train_full_le32_t522v64_seed10_lr1e4_b54_r64_dwdse_only.log; out: /root/autodl-tmp/sedd_outputs/lora_full_le32_t522v64_seed10_lr1e4_b54_r64_dwdse_only |
| full_le32_t522v64_seed10_lr1e4_b54_r128_dwdse_only | LoRA r128 alpha256, batch54, boxed_prompt, response-only DWDSE | 71.0GB | step300 2.1335 | valid64: step200 2/64, step300 4/64, step400 3/64; train64 step300 1/64 | train log: logs/train_full_le32_t522v64_seed10_lr1e4_b54_r128_dwdse_only.log; out: /root/autodl-tmp/sedd_outputs/lora_full_le32_t522v64_seed10_lr1e4_b54_r128_dwdse_only |
| full_le32_t522v64_seed10_lr1e4_b54_r64_ans16_dwdse_only | LoRA r64, batch54, max_answer_len=16, response-only DWDSE | 70.1GB | step400 1.9917 | valid64 ans16: step200 1/64, step300 0/64, step400 2/64 | train log: logs/train_full_le32_t522v64_seed10_lr1e4_b54_r64_ans16_dwdse_only.log; out: /root/autodl-tmp/sedd_outputs/lora_full_le32_t522v64_seed10_lr1e4_b54_r64_ans16_dwdse_only |
| full_le32_t522v64_seed10_lr2e4_b54_r64_mixh50_tmax0995_dwdse_only | LoRA r64, high_t_frac=0.5, lr=2e-4, response-only DWDSE | 70.1GB | step200 1.9224 | valid64 step100 1/64; step200 4/64 | train log: logs/train_full_le32_t522v64_seed10_lr2e4_b54_r64_mixh50_tmax0995_dwdse_only.log; out: /root/autodl-tmp/sedd_outputs/lora_full_le32_t522v64_seed10_lr2e4_b54_r64_mixh50_tmax0995_dwdse_only |
| full_le32_t522v64_seed10_lr1e4_b54_r64_mixh20_dwdse_only | LoRA r64, high_t_frac=0.2, lr=1e-4, response-only DWDSE | 70.1GB | step300 2.1995 | valid64 step200 0/64, step300 0/64 | train log: logs/train_full_le32_t522v64_seed10_lr1e4_b54_r64_mixh20_dwdse_only.log; out: /root/autodl-tmp/sedd_outputs/lora_full_le32_t522v64_seed10_lr1e4_b54_r64_mixh20_dwdse_only |
| full_le32_t522v64_seed10_lr1e4_b54_r64_tmin075_tmax0995_dwdse_only | LoRA r64, high-noise-only t in [0.75,0.995], response-only DWDSE | 70.1GB | step200 2.9480 | valid64 step100 1/64 | train log: logs/train_full_le32_t522v64_seed10_lr1e4_b54_r64_tmin075_tmax0995_dwdse_only.log; out: /root/autodl-tmp/sedd_outputs/lora_full_le32_t522v64_seed10_lr1e4_b54_r64_tmin075_tmax0995_dwdse_only |
| full_le32_t522v64_seed10_lr1e4_b54_r64_outhead_dwdse_only | LoRA r64 plus output_layer.linear, response-only DWDSE | 70.5GB | step200 2.3558 | valid64 step100 1/64, step200 0/64 | train log: logs/train_full_le32_t522v64_seed10_lr1e4_b54_r64_outhead_dwdse_only.log; out: /root/autodl-tmp/sedd_outputs/lora_full_le32_t522v64_seed10_lr1e4_b54_r64_outhead_dwdse_only |
| full_le32_t522v64_seed10_from_overfit64s600_lr5e5_b54_dwdse_only | Resume from 64-sample overfit LoRA, r32, batch54, lr=5e-5, response-only DWDSE | 69.7GB | step400 2.1383 | valid64 step100 1/64, step200 1/64, step300 2/64, step400 1/64; train64 step200 36/64 | train log: logs/train_full_le32_t522v64_seed10_from_overfit64s600_lr5e5_b54_dwdse_only.log; out: /root/autodl-tmp/sedd_outputs/lora_full_le32_t522v64_seed10_from_overfit64s600_lr5e5_b54_dwdse_only |
| full_le32_t522v64_seed10_lr5e5_b36_fullft_dwdse_only | Full-parameter finetune, batch36, lr=5e-5, response-only DWDSE | 70.5GB | step200 2.1992 | valid64 step200 0/64, no_eos 64/64 | train log: logs/train_full_le32_t522v64_seed10_lr5e5_b36_fullft_dwdse_only.log; out: /root/autodl-tmp/sedd_outputs/lora_full_le32_t522v64_seed10_lr5e5_b36_fullft_dwdse_only |
| full_le32_t522v64_seed10_lr1e5_b36_fullft_dwdse_only | Full-parameter finetune, batch36, lr=1e-5, response-only DWDSE | 70.5GB | step400 3.7237 | not generation-evaluated; scalar poor | train log: logs/train_full_le32_t522v64_seed10_lr1e5_b36_fullft_dwdse_only.log; out: /root/autodl-tmp/sedd_outputs/lora_full_le32_t522v64_seed10_lr1e5_b36_fullft_dwdse_only |
| full_le32_t522v64_seed10_lr1e4_b54_r64_windoweos_dwdse_only | LoRA r64, batch54, response-only DWDSE, supervised full answer window as EOS after true answer | 70.1GB | step300 0.2880; step400 0.3099 | A800 local valid64: step300 0/64, step400 0/64 | train log: logs/train_full_le32_t522v64_seed10_lr1e4_b54_r64_windoweos_dwdse_only.log; eval: logs/a800_local_windoweos; out: /root/autodl-tmp/sedd_outputs/lora_full_le32_t522v64_seed10_lr1e4_b54_r64_windoweos_dwdse_only |

## Observations

- The response-only DWDSE validation scalar can become very low without improving answer generation.
- Best verified generalization remains 4/64. Several variants reach the same 4/64 but do not exceed it.
- Full fine-tuning at about 70GB did not help; lr=5e-5 broke EOS behavior, lr=1e-5 kept scalar loss poor.
- Window-EOS supervision reduced scalar loss and no_eos somewhat, but did not improve answer correctness.
- High-noise-only and high-noise mixing did not improve valid generation.
- Output-head LoRA and sigma_map LoRA were not useful; sigma_map exceeded memory or ran at too high memory.

## Record Locations

- Training logs: logs/train_full_le32*.log
- Nohup logs: logs/nohup_full_le32*.log
- Output/checkpoints: /root/autodl-tmp/sedd_outputs/lora_full_le32*
- A800 local eval logs: logs/a800_local_*
