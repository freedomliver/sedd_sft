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
from sampling import get_pc_sampler
from sft_utils import decode_until_eos, load_sedd_for_inference, normalize_answer, resolve_dtype


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate fixed-block staged SEDD decoding")
    parser.add_argument("--pretrained", default="pretrained/sedd-small")
    parser.add_argument("--lora_ckpt", required=True)
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--split", choices=("train", "valid", "validation", "test", "all"), default="test")
    parser.add_argument("--train_size", type=int, default=320)
    parser.add_argument("--valid_size", type=int, default=40)
    parser.add_argument("--test_size", type=int, default=40)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--reasoning_steps", type=int, default=None)
    parser.add_argument("--answer_steps", type=int, default=None)
    parser.add_argument("--use_gold_reasoning", action="store_true")
    parser.add_argument("--predictor", choices=("analytic", "euler", "none"), default="analytic")
    parser.add_argument("--sampling_mode", choices=("sample", "argmax"), default="argmax")
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--question_block_len", type=int, default=48)
    parser.add_argument("--reasoning_block_len", type=int, default=96)
    parser.add_argument("--final_answer_block_len", type=int, default=16)
    parser.add_argument("--fixed_layout_reasoning_start", type=int, default=None)
    parser.add_argument("--fixed_layout_final_answer_start", type=int, default=None)
    parser.add_argument("--reasoning_field", default="compressed_reasoning")
    parser.add_argument("--final_answer_field", default="compressed_answer")
    parser.add_argument("--answer_truncate", choices=("head", "tail"), default="head")
    parser.add_argument("--fixed_layout_supervise_pad", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--out_jsonl", default="outputs/eval_fixed_blocks_staged.jsonl")
    parser.add_argument("--seed", type=int, default=10)
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


@torch.no_grad()
def run_stage(model, graph, noise, init_x, generation_mask, clamp_values, args, device, steps):
    def clamp(x):
        x = x.clone()
        x[~generation_mask] = clamp_values[~generation_mask]
        return x

    sampler = get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=tuple(init_x.shape),
        predictor=args.predictor,
        steps=steps,
        denoise=True,
        eps=1e-4,
        device=device,
        proj_fun=clamp,
        init_x=init_x,
        sampling_mode=args.sampling_mode,
    )
    return sampler(model)


@torch.no_grad()
def sample_staged_batch(model, graph, noise, tokenizer, items, args, device, dtype):
    batch_size = len(items)
    reasoning_start = (
        args.fixed_layout_reasoning_start
        if args.fixed_layout_reasoning_start is not None
        else args.question_block_len
    )
    answer_start = (
        args.fixed_layout_final_answer_start
        if args.fixed_layout_final_answer_start is not None
        else reasoning_start + args.reasoning_block_len
    )
    reasoning_end = reasoning_start + args.reasoning_block_len
    answer_end = answer_start + args.final_answer_block_len

    clamp_values = torch.full(
        (batch_size, args.max_length),
        tokenizer.eos_token_id,
        dtype=torch.long,
        device=device,
    )
    for row_idx, item in enumerate(items):
        input_ids = item["input_ids"].to(device)
        clamp_values[row_idx, : args.question_block_len] = input_ids[: args.question_block_len]

    init_x = clamp_values.clone()
    reasoning_mask = torch.zeros_like(init_x, dtype=torch.bool)
    reasoning_mask[:, reasoning_start:reasoning_end] = True
    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    if args.use_gold_reasoning:
        stage1 = clamp_values.clone()
        for row_idx, item in enumerate(items):
            input_ids = item["input_ids"].to(device)
            stage1[row_idx, reasoning_start:reasoning_end] = input_ids[
                reasoning_start:reasoning_end
            ]
    else:
        noise_x = graph.sample_limit(batch_size, args.max_length).to(device)
        init_x[reasoning_mask] = noise_x[reasoning_mask]

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            stage1 = run_stage(
                model,
                graph,
                noise,
                init_x,
                reasoning_mask,
                clamp_values,
                args,
                device,
                args.reasoning_steps or args.steps,
            )

    clamp_values2 = clamp_values.clone()
    clamp_values2[:, reasoning_start:reasoning_end] = stage1[:, reasoning_start:reasoning_end]
    init_x2 = clamp_values2.clone()
    noise_x2 = graph.sample_limit(batch_size, args.max_length).to(device)
    answer_mask = torch.zeros_like(init_x2, dtype=torch.bool)
    answer_mask[:, answer_start:answer_end] = True
    init_x2[answer_mask] = noise_x2[answer_mask]
    with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
        generated = run_stage(
            model,
            graph,
            noise,
            init_x2,
            answer_mask,
            clamp_values2,
            args,
            device,
            args.answer_steps or args.steps,
        )
    return generated, reasoning_start, answer_start


def main():
    args = build_arg_parser().parse_args()
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=args.offline)
    model, cfg, _ = load_sedd_for_inference(args.pretrained, args.lora_ckpt, device, dtype)
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
    items = [format_sft_record(row, tokenizer, build_format_cfg(args)) for row in records]

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    with out_path.open("w", encoding="utf-8") as out:
        for batch_start in range(0, len(items), args.eval_batch_size):
            batch_items = items[batch_start : batch_start + args.eval_batch_size]
            batch_records = records[batch_start : batch_start + args.eval_batch_size]
            generated, reasoning_start, answer_start = sample_staged_batch(
                model, graph, noise, tokenizer, batch_items, args, device, dtype
            )
            for row_offset, (record, item) in enumerate(zip(batch_records, batch_items)):
                idx = args.start_idx + batch_start + row_offset
                pred_answer = decode_until_eos(
                    tokenizer,
                    generated[
                        row_offset,
                        answer_start : answer_start + args.final_answer_block_len,
                    ],
                ).strip()
                generated_reasoning = decode_until_eos(
                    tokenizer,
                    generated[
                        row_offset,
                        reasoning_start : reasoning_start + args.reasoning_block_len,
                    ],
                ).strip()
                gold_answer = item["final_answer"].strip()
                match = normalize_answer(pred_answer) == normalize_answer(gold_answer)
                correct += int(match)
                out.write(
                    json.dumps(
                        {
                            "idx": idx,
                            "question": record.get("question", ""),
                            "gold_answer": gold_answer,
                            "pred_answer": pred_answer,
                            "match": match,
                            "generated_reasoning": generated_reasoning,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                print(
                    f"[{idx:3d}] {'OK' if match else 'FAIL'} gold={gold_answer} pred={pred_answer[:120]}",
                    flush=True,
                )

    total = max(len(items), 1)
    print("=" * 60)
    print(f"staged_answer_match: {correct}/{len(items)} = {correct / total:.1%}")
    print(f"raw_generations: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
