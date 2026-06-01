import argparse
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from omegaconf import OmegaConf
from transformers import GPT2TokenizerFast
from model import SEDD
import graph_lib, noise_lib
from sampling import get_conditional_sampler

def make_condition_ids(tokenizer, question, max_question_tokens=200):
    # Training uses question + eos + answer, so inference must condition on the same separator.
    ids = tokenizer.encode(question)[:max_question_tokens]
    ids.append(tokenizer.eos_token_id)
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def extract_boxed(text):
    results = []
    pos = 0
    text = text or ""
    while pos < len(text):
        idx = text.find(r"\boxed{", pos)
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
            results.append(text[start:i - 1].strip())
        pos = i
    return results[-1] if results else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--out_jsonl", default="logs/infer_v9_samples.jsonl")
    args = parser.parse_args()

    device = torch.device("cuda")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    with open("pretrained/sedd-small/config.json") as f:
        cfg = OmegaConf.create(json.load(f))
    model = SEDD(cfg)
    ckpt = torch.load("/root/autodl-tmp/sedd_checkpoints/sedd_sft_v9_final.pt", map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    print("Model loaded", flush=True)

    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)

    with open("data/s1K_train_599.json") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        for i, sample in enumerate(data[:args.limit]):
            q_ids = make_condition_ids(tokenizer, sample["question"])
            sampler = get_conditional_sampler(graph, noise, q_ids, steps=args.steps, device=device)
            with torch.no_grad():
                gen = sampler(model)
            gen_text = tokenizer.decode(gen[0, q_ids.shape[1]:])
            pred_boxed = extract_boxed(gen_text)

            row = {
                "idx": i,
                "question": sample["question"],
                "generated": gen_text,
                "pred_boxed": pred_boxed,
                "has_boxed": pred_boxed is not None,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

            print(f"=== Sample {i} ===", flush=True)
            print(f"Q: {sample['question'][:80]}...", flush=True)
            print(f"Has boxed: {pred_boxed is not None}", flush=True)
            print(f"Pred boxed: {pred_boxed}", flush=True)
            print(f"Text[:1000]: {gen_text[:1000]}", flush=True)
            print(flush=True)

    print(f"Saved raw generations to {args.out_jsonl}", flush=True)


if __name__ == "__main__":
    os.chdir("/root/sedd")
    main()
