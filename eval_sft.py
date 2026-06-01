import json
import torch
import re
import os
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from datasets import load_dataset
from omegaconf import OmegaConf

import graph_lib
import noise_lib
import losses
from model import utils as mutils
from model import SEDD
from sampling import get_conditional_sampler
from data_sft import get_sft_dataloader, format_sample


def extract_answer(text):
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else None


def load_sft_model(pretrained_path, ckpt_path, device):
    from model import SEDD
    model = SEDD.from_pretrained(pretrained_path)
    cfg = model.config
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    return model, cfg


def compute_sft_elbo(model, noise, graph, test_loader, device, n_samples=100):
    loss_fn = losses.get_sft_loss_fn(noise, graph)
    total_loss = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            if count >= n_samples:
                break
            ids = batch["input_ids"].to(device)
            clen = batch["condition_len"].to(device)
            loss = loss_fn(model, ids, clen).mean()
            total_loss += loss.item()
            count += 1
    return total_loss / count


def evaluate():
    device = torch.device("cuda")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    pretrained_path = "pretrained/sedd-small"
    ckpt_path = "/root/autodl-tmp/sedd_checkpoints/sedd_sft_v5_final.pt"

    model, cfg = load_sft_model(pretrained_path, ckpt_path, device)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)

    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    ds = load_dataset("simplescaling/s1K-1.1")["train"]
    test_ds = ds.select(range(900, 1000))

    # 1. ELBO
    test_loader = get_sft_dataloader(batch_size=4)
    elbo = compute_sft_elbo(model, noise, graph, test_loader, device)
    print(f"SFT ELBO loss (lower is better): {elbo:.4f}")

    # 2. Answer accuracy (20 samples)
    correct = 0
    total = 20
    for i in range(total):
        sample = test_ds[i]
        q_ids = torch.tensor(
            tokenizer.encode(sample["question"])[:200], dtype=torch.long
        ).unsqueeze(0)
        sampler = get_conditional_sampler(graph, noise, q_ids, steps=128, device=device)
        gen = sampler(model)
        gen_text = tokenizer.decode(gen[0, q_ids.shape[1]:])
        pred = extract_answer(gen_text)
        gold = extract_answer(sample["deepseek_attempt"])
        if pred and gold and pred == gold:
            correct += 1
        print(f"[{i}] pred={pred}, gold={gold}, {'OK' if pred == gold else 'FAIL'}")
        print(f"  Generated: {gen_text[:200]}\n")

    print(f"\nAnswer Accuracy: {correct}/{total} = {correct / total:.1%}")

    # 3. Generative perplexity (GPT-2-large eval)
    eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
    q_ids = torch.tensor(
        tokenizer.encode(test_ds[0]["question"])[:200], dtype=torch.long
    ).unsqueeze(0)
    sampler = get_conditional_sampler(graph, noise, q_ids, steps=128, device=device)
    samples = sampler(model)
    with torch.no_grad():
        out = eval_model(samples, labels=samples)
        gen_ppl = out.loss.exp().item()
    print(f"Generative Perplexity (GPT-2-large eval): {gen_ppl:.2f}")


if __name__ == "__main__":
    evaluate()
