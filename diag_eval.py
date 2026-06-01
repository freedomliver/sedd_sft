"""
Diagnostic evaluation script for SEDD SFT model.
Tests 4 hypotheses about why full eval showed 0% accuracy.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import GPT2TokenizerFast
from datasets import load_dataset
from model import SEDD
from model.ema import ExponentialMovingAverage
import graph_lib
import noise_lib
import losses
from data_sft import get_sft_dataloader, format_sample
from sampling import get_conditional_sampler, get_pc_sampler

# ─────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED = "pretrained/sedd-small"
SFT_CKPT = "checkpoints_sft/sedd_sft_final.pt"

print(f"Device: {DEVICE}")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
SEP = tokenizer.eos_token_id


def load_pretrained_model():
    model = SEDD.from_pretrained(PRETRAINED)
    cfg = model.config
    model = model.to(DEVICE).eval()
    return model, cfg


def load_sft_model():
    model = SEDD.from_pretrained(PRETRAINED)
    ckpt = torch.load(SFT_CKPT, map_location="cpu")
    # The final checkpoint was saved as plain state_dict (not a wrapped dict)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(DEVICE).eval()
    return model


def get_graph_noise(cfg):
    graph = graph_lib.get_graph(cfg, DEVICE)
    noise = noise_lib.get_noise(cfg).to(DEVICE)
    return graph, noise


# ─────────────────────────────────────────────────────────────────
# Load dataset – fixed 3 training samples for reproducibility
# ─────────────────────────────────────────────────────────────────
print("\nLoading dataset (3 training samples)...")
ds = load_dataset("simplescaling/s1K-1.1")["train"]
TRAIN_SAMPLES = [ds[i] for i in range(3)]


# ─────────────────────────────────────────────────────────────────
# TEST 1: Overfitting test – can the SFT model reproduce training data?
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 1: Overfitting Test (conditional generation on training samples)")
print("="*70)

pretrained_model, cfg = load_pretrained_model()
graph, noise = get_graph_noise(cfg)
sft_model = load_sft_model()

for idx, sample in enumerate(TRAIN_SAMPLES):
    fmt = format_sample(sample)
    input_ids = torch.tensor(fmt["input_ids"], dtype=torch.long)
    cond_len = fmt["condition_len"]

    # Extract the question tokens as conditioning
    q_ids = input_ids[:cond_len].unsqueeze(0)  # shape [1, cond_len]

    # Ground-truth answer tokens (after the condition, excluding padding)
    answer_tokens = input_ids[cond_len:]
    # Find where padding starts
    non_pad_mask = answer_tokens != SEP
    # answers might have SEP as the separator at position 0, so be careful
    # The SEP at position cond_len-1 is the separator; real answer comes after
    gt_text_tokens = input_ids[cond_len:]
    # Decode original answer
    gt_text = tokenizer.decode(gt_text_tokens.tolist(), skip_special_tokens=True)

    print(f"\n--- Sample {idx} ---")
    print(f"  Question: {sample['question'][:100]}...")
    print(f"  condition_len={cond_len}, total_len={1024}")

    # Run conditional sampler with SFT model
    sampler = get_conditional_sampler(
        graph, noise, q_ids, steps=128, device=DEVICE
    )
    with torch.no_grad():
        gen = sampler(sft_model)

    gen_answer = gen[0, cond_len:]
    gen_text = tokenizer.decode(gen_answer.tolist(), skip_special_tokens=True)

    print(f"  [GT  answer] {gt_text[:200]}")
    print(f"  [SFT answer] {gen_text[:200]}")

    # Token overlap metric (unigram)
    gt_set = set(gt_text_tokens[gt_text_tokens != SEP].tolist())
    gen_set = set(gen_answer[gen_answer != SEP].tolist())
    if gt_set:
        overlap = len(gt_set & gen_set) / len(gt_set)
        print(f"  Token overlap (unigram Jaccard): {overlap:.3f}")
    else:
        print("  Token overlap: N/A (empty GT)")


# ─────────────────────────────────────────────────────────────────
# TEST 2: Raw generation quality – pretrained vs SFT unconditional
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 2: Raw Generation Quality (unconditional, pretrained vs SFT)")
print("="*70)

pc_sampler_kwargs = dict(
    graph=graph,
    noise=noise,
    batch_dims=(2, 1024),
    predictor="analytic",
    steps=128,
    denoise=True,
    device=DEVICE,
)

pretrained_sampler = get_pc_sampler(**pc_sampler_kwargs)
sft_sampler = get_pc_sampler(**pc_sampler_kwargs)

print("\nGenerating 2 unconditional samples from PRETRAINED model...")
with torch.no_grad():
    pt_gen = pretrained_sampler(pretrained_model)
for i, seq in enumerate(pt_gen):
    text = tokenizer.decode(seq.tolist(), skip_special_tokens=True)
    n_mask = (seq == graph.dim - 1).sum().item() if hasattr(graph, 'dim') else 0
    print(f"  [PRETRAINED {i}] (first 200 chars): {text[:200]}")
    print(f"  Non-EOS tokens: {(seq != SEP).sum().item()}/1024")

print("\nGenerating 2 unconditional samples from SFT model...")
with torch.no_grad():
    sft_gen = sft_sampler(sft_model)
for i, seq in enumerate(sft_gen):
    text = tokenizer.decode(seq.tolist(), skip_special_tokens=True)
    print(f"  [SFT {i}] (first 200 chars): {text[:200]}")
    print(f"  Non-EOS tokens: {(seq != SEP).sum().item()}/1024")

# Compute vocab entropy as a diversity proxy
def vocab_entropy(seqs, eos_id):
    """Shannon entropy over token distribution (excluding EOS/PAD)."""
    all_tokens = []
    for seq in seqs:
        toks = seq[seq != eos_id].tolist()
        all_tokens.extend(toks)
    if not all_tokens:
        return 0.0
    counts = np.bincount(all_tokens)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

pt_ent = vocab_entropy(pt_gen, SEP)
sft_ent = vocab_entropy(sft_gen, SEP)
print(f"\nVocab entropy (bits) — Pretrained: {pt_ent:.2f}, SFT: {sft_ent:.2f}")
print("(Higher = more diverse; collapse to EOS → ~0)")


# ─────────────────────────────────────────────────────────────────
# TEST 3: Loss decomposition on 10 training samples
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 3: Loss Decomposition (SFT loss, pretrained vs SFT model, 10 samples)")
print("="*70)

loss_fn = losses.get_sft_loss_fn(noise, graph)

# Build a fixed batch of 10 training samples
batch_samples = [format_sample(ds[i]) for i in range(10)]
input_ids_list = [s["input_ids"] for s in batch_samples]
cond_lens = [s["condition_len"] for s in batch_samples]

input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long).to(DEVICE)
cond_lens_tensor = torch.tensor(cond_lens, dtype=torch.long).to(DEVICE)

# Run multiple times to get stable estimate (loss uses random t)
N_RUNS = 5
pretrained_losses = []
sft_losses = []

with torch.no_grad():
    for run in range(N_RUNS):
        pt_loss = loss_fn(pretrained_model, input_ids_tensor, cond_lens_tensor)
        sft_loss = loss_fn(sft_model, input_ids_tensor, cond_lens_tensor)
        pretrained_losses.append(pt_loss.mean().item())
        sft_losses.append(sft_loss.mean().item())

pt_mean = np.mean(pretrained_losses)
pt_std = np.std(pretrained_losses)
sft_mean = np.mean(sft_losses)
sft_std = np.std(sft_losses)

print(f"  Pretrained model SFT loss: {pt_mean:.4f} ± {pt_std:.4f}")
print(f"  SFT model SFT loss:        {sft_mean:.4f} ± {sft_std:.4f}")
delta = pt_mean - sft_mean
print(f"  Delta (pretrained - SFT):  {delta:.4f}")
if delta > 0.1:
    print("  VERDICT: SFT model has LOWER loss → training helped")
elif delta > -0.1:
    print("  VERDICT: Losses are SIMILAR → training barely changed the model")
else:
    print("  VERDICT: SFT model has HIGHER loss → training DIVERGED or broke model")

# Per-sample breakdown
print("\n  Per-sample losses (pretrained | sft):")
with torch.no_grad():
    pt_per = loss_fn(pretrained_model, input_ids_tensor, cond_lens_tensor)
    sft_per = loss_fn(sft_model, input_ids_tensor, cond_lens_tensor)
for i in range(10):
    print(f"    sample {i:2d}: pretrained={pt_per[i].item():.4f}, sft={sft_per[i].item():.4f}, "
          f"delta={pt_per[i].item() - sft_per[i].item():.4f}")


# ─────────────────────────────────────────────────────────────────
# TEST 4: Mask inspection
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 4: Mask Inspection (condition_len, answer ratio, padding ratio)")
print("="*70)

for idx in range(3):
    sample = ds[idx]
    fmt = format_sample(sample)
    ids = torch.tensor(fmt["input_ids"], dtype=torch.long)
    cond_len = fmt["condition_len"]

    total_tokens = 1024
    non_pad_tokens = (ids != SEP).sum().item()
    answer_tokens = max(0, non_pad_tokens - cond_len)
    pad_tokens = total_tokens - non_pad_tokens

    # Note: the SEP between question and answer is counted in cond_len
    answer_ratio = answer_tokens / total_tokens
    pad_ratio = pad_tokens / total_tokens

    print(f"\n  Sample {idx}:")
    print(f"    condition_len (question + SEP): {cond_len}")
    print(f"    total non-padding tokens:       {non_pad_tokens}")
    print(f"    answer tokens (non-pad - cond): {answer_tokens}")
    print(f"    padding tokens:                 {pad_tokens}")
    print(f"    answer/total ratio:             {answer_ratio:.3f}")
    print(f"    padding/total ratio:            {pad_ratio:.3f}")
    print(f"    Question (first 80 chars): {sample['question'][:80]}")

    if answer_ratio < 0.05:
        print("    WARNING: Very few answer tokens — training signal may be diluted by padding!")
    elif answer_ratio > 0.5:
        print("    OK: Substantial answer content.")
    else:
        print("    Moderate answer content.")

    # Show the actual token boundary
    q_tokens = tokenizer.encode(sample["question"])
    t_tokens = tokenizer.encode(sample["deepseek_thinking_trajectory"])
    a_tokens = tokenizer.encode(sample["deepseek_attempt"])
    print(f"    Raw lengths: Q={len(q_tokens)}, T={len(t_tokens)}, A={len(a_tokens)}, "
          f"total_pre_clip={len(q_tokens)+1+len(t_tokens)+len(a_tokens)}")
    if len(q_tokens) + 1 + len(t_tokens) + len(a_tokens) > 1024:
        print("    NOTE: Sequence was CLIPPED to 1024 tokens — answer may be partially/fully cut off!")


print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
