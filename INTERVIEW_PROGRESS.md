# SEDD Supervised Fine-Tuning Interview Notes

## Objective

Build a supervised fine-tuning workflow for SEDD on `simplescaling/s1K-1.1` math problems without destroying the pretrained discrete diffusion model's score-ratio structure.

The current minimal deliverable is:

1. Base SEDD generation baseline.
2. Prior Full FT failure analysis: full-parameter/boxed-only runs produced unreadable text and `0/200` boxed-match accuracy.
3. LoRA-SFT main experiment: response-only DWDSE loss, prompt-clamped sampling, saved demo generations.

## Theory Summary

SEDD is not an autoregressive next-token model. Its model output parameterizes score ratios used by the reverse process of a discrete diffusion model. Fine-tuning should therefore keep the original score-entropy / DWDSE objective instead of replacing it with causal language-model cross entropy.

For supervised conditional generation, the task is reformulated as conditional denoising:

```text
Question:
{question}

Answer:
{solution}<|endoftext|>
```

During training:

1. Prompt tokens are clean and clamped.
2. Answer tokens are forward-noised and trained with DWDSE.
3. Padding tokens stay clean and do not contribute to loss.

During sampling:

1. Prompt positions are fixed every reverse-diffusion step.
2. Answer positions start from the graph limit distribution.
3. The decoded answer is truncated at EOS.

## Implementation Status

Implemented framework pieces:

1. `data_sft.py`: s1K loading, train/valid/test split, `Question/Answer` formatting, `prompt_mask`, `answer_mask`, `pad_mask`.
2. `losses.py`: `get_response_only_sft_loss_fn`, which noised only answer positions and normalizes loss by answer token count.
3. `lora.py`: local LoRA wrapper for `attn_qkv`, `attn_out`, `mlp.0`, `mlp.2`; all base parameters are frozen.
4. `train_sft.py`: LoRA response-only SFT training loop with validation loss and LoRA-only checkpoints.
5. `sampling.py`: prompt-clamped conditional sampler.
6. `infer_sft.py`: base or LoRA generation demo, saved as JSONL.
7. `eval_correct.py`: math generation evaluation with boxed-match, invalid-output stats, optional response-only DWDSE eval.

Continuation updates:

1. Small local JSON files such as `s1K_train_599.json` now keep validation/test holdouts instead of putting every row in train.
2. `train_sft.py` supports `--resume_lora_ckpt` for continuing from `checkpoint_last.pt` or another LoRA checkpoint.
3. Legacy diagnostics that pass `condition_len` into `get_sft_loss_fn` still run through the response-only DWDSE path.
4. `run_train_eval.sh` is repo-location agnostic and accepts `RESUME_LORA_CKPT`, `TRAIN_SIZE`, `VALID_SIZE`, and `TEST_SIZE`.

## Experiment Plan

Base:

```bash
python -u infer_sft.py \
  --pretrained pretrained/sedd-medium \
  --data_json data/s1K_train_599.json \
  --split test \
  --limit 5 \
  --steps 128 \
  --out_jsonl logs/base_sedd_generations.jsonl \
  --offline
```

Full FT:

Prior run is enough for the interview narrative. It produced unreadable mixed-token output, no boxed predictions, and `0/200` boxed-match accuracy. The likely causes were small data, full-parameter updates, over-focused boxed-only loss, and breaking the pretrained score-ratio geometry.

LoRA-SFT:

```bash
python -u train_sft.py \
  --pretrained pretrained/sedd-medium \
  --data_json data/s1K_train_599.json \
  --out_dir outputs/lora_sft \
  --max_steps 1000 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 5e-5 \
  --warmup_steps 100 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --max_answer_len 512 \
  --offline
```

Evaluate LoRA:

```bash
python -u eval_correct.py \
  --pretrained pretrained/sedd-medium \
  --lora_ckpt outputs/lora_sft/lora_final.pt \
  --data_json data/s1K_train_599.json \
  --split test \
  --limit 100 \
  --steps 128 \
  --out_jsonl logs/eval_lora_sft.jsonl \
  --offline
```

If `sedd-medium` weights are unavailable on the server, use `pretrained/sedd-small` only for smoke tests and environment validation. Main results should use `sedd-medium`.

## RL Exploration

A practical RL path for discrete diffusion fine-tuning is preference/reward-weighted denoising rather than standard autoregressive policy gradients.

Candidate approaches:

1. Generate multiple answers per prompt with prompt-clamped reverse diffusion.
2. Score generations using boxed exact match, verifier reward, or format/readability reward.
3. Use reward-weighted DWDSE on selected trajectories or answer token states.
4. Explore DPO-style preference optimization over paired samples by comparing denoising likelihood or response-only DWDSE surrogate scores.

Risks:

1. Exact reverse-process likelihood is expensive.
2. Sparse math rewards are high variance.
3. Aggressive reward tuning can reintroduce unreadable text.

Interview framing: RL is a second-stage alignment layer after LoRA-SFT has a readable supervised baseline.

## Findings And Challenges

1. SEDD SFT must preserve score-entropy training; causal CE is the wrong loss family.
2. Prompt/answer formatting matters. Previous `question + eos` mismatch helped explain bad evaluation behavior, but fixing it alone did not recover a broken checkpoint.
3. Full FT on tiny math data is unstable and produced mixed symbolic/English fragments.
4. LoRA keeps the base model frozen and reduces the blast radius of SFT.
5. Response-only masking is necessary: prompt and padding should not be noised or trained.

## Demo Checklist

Before the final interview:

1. Show base output JSONL and a few readable/unreadable examples.
2. Show Full FT failure examples and explain why they matter.
3. Show LoRA train log with trainable parameter ratio and validation DWDSE.
4. Show LoRA generation JSONL with question, generated answer, boxed prediction, and match status.
5. Show the code path from data masks to loss to sampler clamp.

## Remote Run Log

June 1, 2026 remote server run:

1. Cleaned `/root/autodl-tmp/sedd_checkpoints` by removing intermediate full-FT checkpoints while preserving `sedd_sft_v8_final.pt`, `sedd_sft_v9_final.pt`, `checkpoints_sft/sedd_sft_final.pt`, and `logs/eval_v9_200.jsonl`.
2. Fixed LoRA target matching so only `blocks.*.{attn_qkv,attn_out,mlp.0,mlp.2}` are wrapped. Earlier suffix matching accidentally included `sigma_map.mlp.0/2`.
3. Trained `pretrained/sedd-medium` LoRA-SFT with `batch_size=12`, `grad_accum=4`, effective batch 48, rank 8, alpha 16, LR `5e-5`, `max_steps=1000`, `max_answer_len=512`.
4. Training artifacts:
   - `outputs/lora_sft/checkpoint_step_250.pt`
   - `outputs/lora_sft/checkpoint_step_500.pt`
   - `outputs/lora_sft/checkpoint_step_750.pt`
   - `outputs/lora_sft/checkpoint_step_1000.pt`
   - `outputs/lora_sft/lora_final.pt`
5. Validation response-DWDSE: step 100 `3.7506`, step 200 `3.2296`, step 400 `2.9564`, step 900 `2.8677`, step 1000 `4.3327`.
6. Generation eval on 79 held-out boxed-answer samples:
   - `lora_final.pt`: `0/79`, `100%` no boxed, `89.9%` no EOS.
   - `checkpoint_step_750.pt`: `0/79`, `100%` no boxed, `84.8%` no EOS.

Conclusion: the LoRA response-only DWDSE path trains and reduces validation loss, but this configuration still does not recover boxed math-answer generation. The result is useful evidence that preserving the loss family and using LoRA are necessary but not sufficient; next experiments should focus on answer formatting pressure, sampler settings, shorter answer lengths, output-layer LoRA, or replay/regularization rather than more of the same run.

Clamp audit:

1. Training was already response-only: `answer_mask & pad_mask` positions are noised and trained; prompt and padding stay clean.
2. Eval used the prompt-clamped sampler, but the sampler only clamped before each reverse update and at final return. It did not immediately reclamp after each predictor update.
3. Fixed `sampling.py` so `get_pc_sampler` supports explicit `init_x`, starts from `prompt + graph.sample_limit(answer_len)`, clamps before and after every predictor update, and clamps again after denoise.
4. With the absorbing graph, `graph.sample_limit(B, answer_len)` is the absorbing/mask token, so answer initialization is now explicit and matches the intended SEDD conditional sampling flow.
5. Strict-clamp eval on all 79 held-out boxed-answer samples still produced `0/79`, `100%` no boxed, `89.9%` no EOS, so the final LoRA result is not explained only by prompt drift during sampling.

Target-format audit:

1. The completed LoRA run used the default `--answer_field solution`.
2. In `data/s1K_train_599.json`, only `109/399` train-split `solution` targets contain `\boxed{...}`, while `deepseek_attempt` contains `\boxed{...}` for `399/399`.
3. Evaluation extracts only generated `\boxed{...}`. This means the run mostly trained non-boxed answers but evaluated boxed-only predictions.
4. Directly switching to `deepseek_attempt` with the previous head-truncation would also be wrong: at budgets 128/256, head truncation kept `0/399` boxed answers, while tail truncation kept `399/399`.
5. Added `answer_field=final_boxed`, which extracts the last boxed answer and trains the short target `The answer is \boxed{...}.`; train answer length is about 13 tokens on average and all `399/399` targets contain EOS and boxed answer.
6. Added `answer_truncate=tail` for long-attempt experiments so final boxed answers are not truncated away.

Recommended next experiment: train a fresh LoRA with `--answer_field final_boxed --max_answer_len 64`, then evaluate with strict prompt clamp and `--max_answer_len 64`. This tests whether SEDD can learn boxed final-answer formatting before trying long-form reasoning.

Overfit sanity check:

1. Updated `train_sft.py` defaults to `batch_size=16`, `grad_accum=4`, and `lr=5e-5`.
2. Ran a 16-sample overfit experiment with `answer_field=final_boxed`, `answer_prefix="Final Answer:"`, `max_answer_len=64`, `max_steps=500`.
3. Training loss dropped from about `6.86` at step 10 to `0.43` at step 500, confirming the response-only DWDSE objective can fit the selected samples.
4. Train-split generation on the same 16 prompts still produced `0/16` boxed outputs and no decoded EOS markers.
5. Interpretation: the current loss path can reduce denoising loss, but the learned LoRA updates are not translating into prompt-clamped reverse sampling that generates the target short boxed answer. The next debugging layer should inspect the sampling distribution/token probabilities at answer positions, not just scalar DWDSE loss.
