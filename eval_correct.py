"""
评估脚本：在 s1K_train_599.json 样本上做条件采样推理，
提取模型输出 \boxed{} 与数据集 \boxed{} 对比准确度。
"""

import json
import argparse
import torch
import re
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import GPT2TokenizerFast
from omegaconf import OmegaConf
from safetensors.torch import load_file
from model import SEDD
import graph_lib
import noise_lib
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
            results.append(text[start : i - 1].strip())
        pos = i
    return results[-1] if results else None


def get_gold_answer(sample):
    sol = sample.get("solution", "").strip()
    sol_boxed = extract_boxed(sol)
    if sol_boxed:
        return sol_boxed
    ds_boxed = extract_boxed(sample.get("deepseek_attempt", ""))
    if ds_boxed:
        return ds_boxed
    if sol and len(sol) < 100:
        return sol
    return None


def normalize(s):
    if s is None:
        return None
    s = s.strip().replace(" ", "").replace(",", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = s.replace("\\", "").replace(" ", "").lower()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--out_jsonl", default="logs/eval_v9_generations.jsonl")
    args = parser.parse_args()

    device = torch.device("cuda")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    pretrained_path = "pretrained/sedd-small"
    ckpt_path = "/root/autodl-tmp/sedd_checkpoints/sedd_sft_v9_final.pt"

    print(f"Loading model from {pretrained_path} + {ckpt_path} ...")
    t0 = time.time()
    with open(os.path.join(pretrained_path, "config.json")) as f:
        cfg = OmegaConf.create(json.load(f))
    model = SEDD(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)

    with open("data/s1K_train_599.json") as f:
        correct_data = json.load(f)

    eval_samples = [r for r in correct_data if get_gold_answer(r) is not None]
    print(f"Samples with gold answer: {len(eval_samples)} / {len(correct_data)}")

    total = min(args.limit, len(eval_samples))
    correct = 0
    no_boxed_pred = 0

    print(f"\nEvaluating {total} samples ...\n")
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        for i in range(total):
            sample = eval_samples[i]
            gold = get_gold_answer(sample)

            q_ids = make_condition_ids(tokenizer, sample["question"])

            sampler = get_conditional_sampler(graph, noise, q_ids, steps=args.steps, device=device)
            with torch.no_grad():
                gen = sampler(model)

            gen_text = tokenizer.decode(gen[0, q_ids.shape[1]:])
            pred_boxed = extract_boxed(gen_text)

            if pred_boxed is None:
                no_boxed_pred += 1

            match = normalize(pred_boxed) == normalize(gold)
            if match:
                correct += 1

            row = {
                "idx": i,
                "question": sample["question"],
                "gold": gold,
                "generated": gen_text,
                "pred_boxed": pred_boxed,
                "match": match,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

            status = "OK" if match else "FAIL"
            print(f"[{i:3d}] {status} | gold={gold[:50] if gold else None} | pred={pred_boxed}")
            if i < 5:
                print(f"      gen[:500]={gen_text[:500]}")

            if (i + 1) % 50 == 0:
                print(f"--- Progress: {correct}/{i+1} = {correct/(i+1):.1%} | no_boxed_pred={no_boxed_pred} ---\n")

    print("\n" + "=" * 60)
    print(f"Answer Accuracy (boxed match): {correct}/{total} = {correct/total:.1%}")
    print(f"Predictions without boxed: {no_boxed_pred}/{total} = {no_boxed_pred/total:.1%}")
    print(f"Raw generations: {args.out_jsonl}")
    print("=" * 60)


if __name__ == "__main__":
    os.chdir("/root/sedd")
    main()
