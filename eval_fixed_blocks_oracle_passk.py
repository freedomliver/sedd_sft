import argparse
import json
import os
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

import graph_lib
import noise_lib
from data_sft import (
    DEFAULT_DATASET,
    SFTFormatConfig,
    format_sft_record,
    load_s1k_records,
    split_records,
)
from eval_fixed_blocks import sample_fixed_batch
from sft_utils import decode_until_eos, load_sedd_for_inference, normalize_answer, resolve_dtype


def short_text(text: str, limit: int = 600) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-block SEDD SFT oracle pass@K generations"
    )
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
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Number of generated trajectories per sampler call.",
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--predictor", choices=("analytic", "euler", "none"), default="analytic")
    parser.add_argument("--sampling_mode", choices=("sample", "argmax"), default="sample")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--question_block_len", type=int, default=231)
    parser.add_argument("--reasoning_block_len", type=int, default=530)
    parser.add_argument("--final_answer_block_len", type=int, default=263)
    parser.add_argument("--fixed_layout_reasoning_start", type=int, default=None)
    parser.add_argument("--fixed_layout_final_answer_start", type=int, default=None)
    parser.add_argument("--reasoning_field", default="solution")
    parser.add_argument("--final_answer_field", default="answer")
    parser.add_argument("--answer_truncate", choices=("head", "tail"), default="head")
    parser.add_argument("--fixed_layout_supervise_pad", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--out_jsonl", default="outputs/eval_fixed_blocks_oracle_passk.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    return parser


def build_format_cfg(args):
    return SFTFormatConfig(
        max_length=args.max_length,
        answer_truncate=args.answer_truncate,
        fixed_layout=True,
        question_block_len=args.question_block_len,
        reasoning_block_len=args.reasoning_block_len,
        final_answer_block_len=args.final_answer_block_len,
        fixed_layout_reasoning_start=args.fixed_layout_reasoning_start,
        fixed_layout_final_answer_start=args.fixed_layout_final_answer_start,
        reasoning_field=args.reasoning_field,
        final_answer_field=args.final_answer_field,
        fixed_layout_supervise_pad=args.fixed_layout_supervise_pad,
    )


def main():
    args = build_arg_parser().parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval_batch_size must be positive")

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype)
    print("Loading tokenizer: gpt2", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=args.offline)

    print(f"Loading base model: {args.pretrained}", flush=True)
    model, cfg, lora_info = load_sedd_for_inference(args.pretrained, args.lora_ckpt, device, dtype)
    if lora_info:
        print(f"Loaded checkpoint: {args.lora_ckpt}", flush=True)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)

    rows = load_s1k_records(args.data_json, args.dataset_name, args.hf_split, args.cache_dir)
    records = split_records(
        rows,
        args.split,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    records = records[args.start_idx : args.start_idx + args.limit]
    format_cfg = build_format_cfg(args)
    items = [format_sft_record(row, tokenizer, format_cfg) for row in records]

    reasoning_start_for_log = (
        args.fixed_layout_reasoning_start
        if args.fixed_layout_reasoning_start is not None
        else args.question_block_len
    )
    answer_start_for_log = (
        args.fixed_layout_final_answer_start
        if args.fixed_layout_final_answer_start is not None
        else reasoning_start_for_log + args.reasoning_block_len
    )
    print(
        f"Evaluating oracle pass@{args.num_samples}: rows={len(items)} "
        f"split={args.split} q={args.question_block_len} "
        f"reasoning_start={reasoning_start_for_log} r={args.reasoning_block_len} "
        f"answer_start={answer_start_for_log} a={args.final_answer_block_len} "
        f"steps={args.steps} sampling={args.sampling_mode}",
        flush=True,
    )

    per_item = []
    for local_idx, (record, item) in enumerate(zip(records, items)):
        idx = args.start_idx + local_idx
        per_item.append(
            {
                "idx": idx,
                "question": record.get("question", ""),
                "gold_answer": item["final_answer"].strip(),
                "samples": [],
            }
        )

    expanded = [
        (item_idx, sample_idx)
        for item_idx in range(len(items))
        for sample_idx in range(args.num_samples)
    ]
    total_samples = len(expanded)
    generated_count = 0

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for batch_start in range(0, total_samples, args.eval_batch_size):
        batch_pairs = expanded[batch_start : batch_start + args.eval_batch_size]
        batch_items = [items[item_idx] for item_idx, _ in batch_pairs]
        generated, reasoning_start, answer_start = sample_fixed_batch(
            model,
            graph,
            noise,
            tokenizer,
            batch_items,
            args,
            device,
            dtype,
        )

        for row_offset, (item_idx, sample_idx) in enumerate(batch_pairs):
            item = items[item_idx]
            answer_ids = generated[
                row_offset,
                answer_start : answer_start + args.final_answer_block_len,
            ]
            reasoning_ids = generated[
                row_offset,
                reasoning_start : reasoning_start + args.reasoning_block_len,
            ]
            answer_list = answer_ids.detach().cpu().tolist()
            eos_present = tokenizer.eos_token_id in answer_list
            eos_index = answer_list.index(tokenizer.eos_token_id) if eos_present else None
            pred_answer = decode_until_eos(tokenizer, answer_ids).strip()
            generated_reasoning = decode_until_eos(tokenizer, reasoning_ids).strip()
            gold_answer = item["final_answer"].strip()
            match = normalize_answer(pred_answer) == normalize_answer(gold_answer)
            per_item[item_idx]["samples"].append(
                {
                    "sample_idx": sample_idx,
                    "pred_answer": pred_answer,
                    "match": match,
                    "eos_present": eos_present,
                    "eos_index": eos_index,
                    "pred_answer_token_len": len(
                        tokenizer.encode(pred_answer, add_special_tokens=False)
                    ),
                    "generated_reasoning": generated_reasoning,
                    "generated_reasoning_prefix": short_text(generated_reasoning, 500),
                }
            )
            generated_count += 1

        print(
            f"generated {generated_count}/{total_samples} trajectories",
            flush=True,
        )

    pass1 = 0
    oracle = 0
    total_correct_samples = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row in per_item:
            samples = sorted(row["samples"], key=lambda x: x["sample_idx"])
            correct_samples = [s for s in samples if s["match"]]
            row["samples"] = samples
            row["correct_count"] = len(correct_samples)
            row["pass_at_1"] = bool(samples and samples[0]["match"])
            row["oracle_pass_at_k"] = bool(correct_samples)
            row["first_correct_sample_idx"] = (
                correct_samples[0]["sample_idx"] if correct_samples else None
            )
            pass1 += int(row["pass_at_1"])
            oracle += int(row["oracle_pass_at_k"])
            total_correct_samples += len(correct_samples)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = max(len(per_item), 1)
    print("=" * 72)
    print(f"rows: {len(per_item)}")
    print(f"sampled_trajectories: {total_samples}")
    print(f"pass@1_by_first_sample: {pass1}/{len(per_item)} = {pass1 / total:.1%}")
    print(
        f"oracle_pass@{args.num_samples}: "
        f"{oracle}/{len(per_item)} = {oracle / total:.1%}"
    )
    print(
        f"correct_sample_rate: {total_correct_samples}/{total_samples} = "
        f"{total_correct_samples / max(total_samples, 1):.1%}"
    )
    print(f"raw_generations: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
