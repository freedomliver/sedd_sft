import argparse
import json
import os
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

from data_sft import DEFAULT_DATASET, build_prompt, get_answer_text, load_s1k_records, split_records
import graph_lib
import noise_lib
from sampling import get_prompt_clamped_sampler
from sft_utils import decode_until_eos, extract_boxed, load_sedd_for_inference, normalize_answer, resolve_dtype


def make_prompt_ids(
    tokenizer,
    question: str,
    answer_prefix: str = "Answer:",
    answer_leading_newline: bool = False,
    boxed_prompt: bool = False,
    max_length: int = 1024,
    min_answer_len: int = 32,
):
    if boxed_prompt:
        prompt = build_prompt(question, answer_prefix, trailing_newline=True) + r"\boxed{"
    else:
        prompt = build_prompt(
            question,
            answer_prefix,
            trailing_newline=not answer_leading_newline,
        )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    original_len = len(ids)
    max_prompt_len = max(1, max_length - min_answer_len)
    truncated = original_len > max_prompt_len
    if truncated:
        ids = ids[:max_prompt_len]
        prompt = tokenizer.decode(ids)
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0), prompt, truncated, original_len


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Prompt-clamped SEDD SFT inference")
    parser.add_argument("--pretrained", default="louaaron/sedd-medium")
    parser.add_argument("--lora_ckpt", default=None)
    parser.add_argument("--data_json", default=None)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--split", choices=("train", "valid", "validation", "test", "all"), default="test")
    parser.add_argument("--train_size", type=int, default=800)
    parser.add_argument("--valid_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--question", default=None)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--predictor", choices=("analytic", "euler", "none"), default="analytic")
    parser.add_argument("--sampling_mode", choices=("sample", "argmax"), default="sample")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_answer_len", type=int, default=32)
    parser.add_argument("--min_answer_len", type=int, default=32)
    parser.add_argument("--answer_field", default="final_boxed")
    parser.add_argument("--answer_prefix", default="Answer:")
    parser.add_argument("--answer_leading_newline", action="store_true")
    parser.add_argument("--boxed_prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--out_jsonl", default="outputs/infer_sft.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype)
    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    print("Loading tokenizer: gpt2", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=args.offline)

    print(f"Loading base model: {args.pretrained}", flush=True)
    model, cfg, lora_info = load_sedd_for_inference(args.pretrained, args.lora_ckpt, device, dtype)
    if lora_info:
        print(f"Loaded LoRA checkpoint: {args.lora_ckpt}", flush=True)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)

    if args.question:
        records = [{"question": args.question}]
    else:
        rows = load_s1k_records(args.data_json, args.dataset_name, args.hf_split, args.cache_dir)
        records = split_records(
            rows,
            args.split,
            train_size=args.train_size,
            valid_size=args.valid_size,
            test_size=args.test_size,
            seed=args.seed,
        )[: args.limit]

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    boxed_total = 0
    match_total = 0
    exact_total = 0
    eos_total = 0
    truncated_total = 0
    with out_path.open("w", encoding="utf-8") as out:
        for idx, record in enumerate(records[: args.limit]):
            prompt_ids, prompt_text, prompt_truncated, prompt_original_len = make_prompt_ids(
                tokenizer,
                record["question"],
                args.answer_prefix,
                args.answer_leading_newline,
                args.boxed_prompt,
                args.max_length,
                args.min_answer_len,
            )
            if prompt_ids.shape[1] >= args.max_length:
                raise ValueError(f"Prompt {idx} is too long: {prompt_ids.shape[1]} tokens")

            sampler = get_prompt_clamped_sampler(
                graph=graph,
                noise=noise,
                prompt_ids=prompt_ids,
                max_answer_len=args.max_answer_len,
                max_length=args.max_length,
                steps=args.steps,
                predictor=args.predictor,
                denoise=True,
                device=device,
                sampling_mode=args.sampling_mode,
            )
            with torch.no_grad(), torch.amp.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=autocast_enabled,
            ):
                generated = sampler(model)
            answer_ids = generated[0, prompt_ids.shape[1] :]
            eos_present = tokenizer.eos_token_id in answer_ids.detach().cpu().tolist()
            answer_suffix = decode_until_eos(tokenizer, answer_ids)
            if args.boxed_prompt:
                pred_boxed = answer_suffix.strip()
                answer_text = f"\\boxed{{{pred_boxed}}}"
            else:
                answer_text = answer_suffix
                pred_boxed = extract_boxed(answer_text)
            target_answer = get_answer_text(record, args.answer_field) if args.answer_field else None
            target_boxed = extract_boxed(target_answer) if target_answer else None
            match = (
                normalize_answer(pred_boxed) == normalize_answer(target_boxed)
                if target_boxed is not None
                else None
            )
            exact = answer_text.strip() == target_answer.strip() if target_answer else None

            row = {
                "idx": idx,
                "question": record["question"],
                "prompt": prompt_text,
                "prompt_len": int(prompt_ids.shape[1]),
                "prompt_original_len": prompt_original_len,
                "prompt_truncated": prompt_truncated,
                "generated": answer_text,
                "generated_suffix": answer_suffix,
                "pred_boxed": pred_boxed,
                "target_answer": target_answer,
                "target_boxed": target_boxed,
                "match": match,
                "exact": exact,
                "eos_present": eos_present,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            total += 1
            boxed_total += int(pred_boxed is not None)
            eos_total += int(eos_present)
            truncated_total += int(prompt_truncated)
            if match is not None:
                match_total += int(match)
            if exact is not None:
                exact_total += int(exact)

            print(f"=== Sample {idx} ===", flush=True)
            print(f"Q: {record['question'][:120]}", flush=True)
            print(f"boxed: {pred_boxed}", flush=True)
            if target_boxed is not None:
                print(f"target: {target_boxed} match={match} exact={exact} eos={eos_present}", flush=True)
            print(answer_text[:1000], flush=True)
            print(flush=True)

    print(f"Saved generations to {out_path}", flush=True)
    print(
        f"summary rows={total} boxed={boxed_total}/{total} eos={eos_total}/{total} "
        f"prompt_truncated={truncated_total}/{total}",
        flush=True,
    )
    if args.answer_field:
        print(
            f"summary target_match={match_total}/{total} exact={exact_total}/{total}",
            flush=True,
        )


if __name__ == "__main__":
    main()
