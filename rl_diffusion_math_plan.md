# Diffusion RL Math Reasoning Plan

Date: 2026-06-04

## Goal

Test whether SEDD reverse diffusion sampling can be optimized as a multi-step
policy so that different denoising times learn useful math reasoning/search
behavior, rather than only imitating a fixed supervised reasoning string.

The first target is the 400 simple manual synthetic dataset. This dataset is
small, programmatic, and has topic metadata, so it is suitable for verifier
reward design and fast failure analysis.

## Starting Point

- Do not train from scratch.
- Start from a previous fixed-block simple400 checkpoint that can overfit the
  training set under supervised DWDSE.
- Use the short fixed layout from the successful simple experiments:
  - max_length: 256
  - question block: q = 48
  - reasoning block: r = 96 or 128 depending on the checkpoint
  - answer block: a = 16
  - fixed positions with question/reasoning/answer gaps clamped
- Keep LoRA fine-tuning first. Do not introduce a value network in the first
  pass.

## Core Idea

Treat one reverse diffusion sample as a trajectory:

```text
x_T -> x_t -> ... -> x_0
```

For each question, sample K trajectories. Decode the final reasoning and answer
blocks. A verifier computes a reward from the generated text. Then update the
model to increase probability of high-reward trajectories and decrease low
reward trajectories.

Initial RL objective should be trajectory-level and simple:

```text
adv_i = reward_i - mean(reward_group_for_same_question)
loss_rl = - adv_i * logprob_trajectory_i
loss = loss_rl + beta * KL(current || SFT_checkpoint)
```

If exact denoising-step logprobs are hard to expose initially, use a faster
proxy experiment:

```text
sample K trajectories -> keep high-reward/winner outputs -> reward-weighted
DWDSE or winner replay fine-tuning
```

This proxy is not full RL, but it can quickly test whether verifier-guided
sampling improves generation.

## Reward Design

Start with a simple, robust reward. Avoid rewarding exact wording.

Suggested components:

- final answer exact match after normalization: +1.0
- valid boxed/fixed answer syntax: +0.1
- no malformed fractions/braces/repetition: +0.1
- reasoning contains expected topic method keyword: +0.05
- topic-specific numeric check: +0.1 to +0.5
- degenerate repeated text or invalid answer block: negative reward

The 400 simple generator has explicit topic types, so dense rewards can be
programmatic:

- linear_equation: parse/verify final x.
- arithmetic: evaluate expression.
- geometry: perimeter/area/circumference/distance/slope formulas.
- number_theory: prime factor, divisor count, multiple count, modular result.
- probability: fraction arithmetic and reduction.
- sequence/counting: formula and final integer.
- linear_system: verify solved variable.

The first implementation can use only answer exact match plus syntax checks.
Then add topic rewards if oracle pass@K is too sparse.

## First Experiments

1. Locate the best previous simple400 fixed-block checkpoint.
2. Reproduce supervised generation metrics:
   - train64 argmax64
   - valid40 argmax64
   - test40 argmax64 if available
3. Measure oracle sampling:
   - K = 8, 16, 32 per question
   - train/valid pass@K by verifier answer match
   - record how many questions have at least one correct sample
4. If oracle pass@K is nonzero enough:
   - implement GRPO/REINFORCE or winner replay.
5. If oracle pass@K is very low:
   - add dense topic-specific verifier reward before full RL.

## Success Criteria

Minimum viability:

- RL/winner replay improves valid40 answer accuracy over the SFT checkpoint.
- Improvement appears on newly generated simple variants, not only the fixed 400
  rows.

Stronger evidence:

- pass@1 improves while pass@K remains stable or improves.
- Reasoning text has fewer malformed equations/fractions.
- The model handles out-of-range numeric variants from the same generator.

## Risks

- Reward is too sparse if the SFT checkpoint rarely samples correct answers.
- Exact trajectory logprobs may require modifying the sampler.
- The model may optimize answer-only shortcuts instead of readable reasoning.
- Verifier bugs can create exploitable reward loopholes.
- Continuing on fixed 400 samples alone can reinforce memorization, so use
  generated variants for RL evaluation as early as possible.

## Next Concrete Step

Find the best simple400 checkpoint and run oracle pass@K. This determines
whether to implement true trajectory RL immediately or first build dense verifier
rewards / winner replay.
