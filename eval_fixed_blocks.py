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


def short_text(text: str, limit: int = 600) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate fixed-block SEDD SFT generations")
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
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
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
    parser.add_argument("--out_jsonl", default="outputs/eval_fixed_blocks.jsonl")
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


@torch.no_grad()
def sample_fixed_batch(model, graph, noise, tokenizer, items, args, device, dtype):
    batch_size = len(items)
    question_len = args.question_block_len
    reasoning_start = (
        args.fixed_layout_reasoning_start
        if args.fixed_layout_reasoning_start is not None
        else args.question_block_len
    )
    reasoning_end = reasoning_start + args.reasoning_block_len
    answer_start = (
        args.fixed_layout_final_answer_start
        if args.fixed_layout_final_answer_start is not None
        else reasoning_end
    )
    answer_end = answer_start + args.final_answer_block_len
    eos_token_id = tokenizer.eos_token_id
    clamp_values = torch.full(
        (batch_size, args.max_length),
        eos_token_id,
        dtype=torch.long,
        device=device,
    )
    generation_mask = torch.zeros(
        (batch_size, args.max_length),
        dtype=torch.bool,
        device=device,
    )
    for row_idx, item in enumerate(items):
        input_ids = item["input_ids"].to(device)
        clamp_values[row_idx, :question_len] = input_ids[:question_len]
        if (
            args.fixed_layout_reasoning_start is None
            and args.fixed_layout_final_answer_start is None
        ):
            generation_mask[row_idx, question_len:] = True
        else:
            generation_mask[row_idx, reasoning_start:reasoning_end] = True
            generation_mask[row_idx, answer_start:answer_end] = True

    init_x = clamp_values.clone()
    init_noise = graph.sample_limit(batch_size, args.max_length).to(device)
    init_x[generation_mask] = init_noise[generation_mask]

    def clamp_question(x):
        x = x.clone()
        x[~generation_mask] = clamp_values[~generation_mask]
        return x

    sampler = get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=(batch_size, args.max_length),
        predictor=args.predictor,
        steps=args.steps,
        denoise=True,
        eps=1e-4,
        device=device,
        proj_fun=clamp_question,
        init_x=init_x,
        sampling_mode=args.sampling_mode,
    )
    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
        generated = sampler(model)
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
        f"Evaluating {len(items)} fixed-block samples from split={args.split} "
        f"q={args.question_block_len} "
        f"reasoning_start={reasoning_start_for_log} "
        f"r={args.reasoning_block_len} "
        f"a={args.final_answer_block_len} "
        f"answer_start={answer_start_for_log}",
        flush=True,
    )

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    exact = 0
    empty = 0
    no_eos = 0
    total_answer_tokens = 0

    with out_path.open("w", encoding="utf-8") as out:
        for batch_start in range(0, len(items), args.eval_batch_size):
            batch_items = items[batch_start : batch_start + args.eval_batch_size]
            batch_records = records[batch_start : batch_start + args.eval_batch_size]
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

            for row_offset, (record, item) in enumerate(zip(batch_records, batch_items)):
                local_idx = batch_start + row_offset
                idx = args.start_idx + local_idx
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
                gold_answer = item["final_answer"].strip()
                generated_reasoning = decode_until_eos(tokenizer, reasoning_ids).strip()
                pred_answer_token_len = len(
                    tokenizer.encode(pred_answer, add_special_tokens=False)
                )
                raw_answer_block = tokenizer.decode(answer_list, skip_special_tokens=False)
                match = normalize_answer(pred_answer) == normalize_answer(gold_answer)
                exact_match = pred_answer == gold_answer
                correct += int(match)
                exact += int(exact_match)
                empty += int(not pred_answer)
                no_eos += int(not eos_present)
                total_answer_tokens += pred_answer_token_len

                row = {
                    "idx": idx,
                    "question": record.get("question", ""),
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                    "match": match,
                    "exact": exact_match,
                    "eos_present": eos_present,
                    "eos_index": eos_index,
                    "pred_answer_token_len": pred_answer_token_len,
                    "answer_block_token_len": args.final_answer_block_len,
                    "raw_answer_block_prefix": short_text(raw_answer_block),
                    "answer_token_ids_prefix": answer_list[:64],
                    "generated_reasoning": generated_reasoning,
                    "question_block_len": args.question_block_len,
                    "fixed_layout_reasoning_start": args.fixed_layout_reasoning_start,
                    "reasoning_block_len": args.reasoning_block_len,
                    "final_answer_block_len": args.final_answer_block_len,
                    "fixed_layout_final_answer_start": args.fixed_layout_final_answer_start,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print(
                    f"[{idx:3d}] {'OK' if match else 'FAIL'} "
                    f"gold={gold_answer} pred={pred_answer[:160]}",
                    flush=True,
                )
                if idx < 5:
                    print(f"      reasoning[:400]={generated_reasoning[:400]}", flush=True)

    total = max(len(items), 1)
    print("=" * 60)
    print(f"answer_block_match: {correct}/{len(items)} = {correct / total:.1%}")
    print(f"exact_answer_match: {exact}/{len(items)} = {exact / total:.1%}")
    print(f"empty_answer_blocks: {empty}/{len(items)} = {empty / total:.1%}")
    print(f"no_eos_answer_blocks: {no_eos}/{len(items)} = {no_eos / total:.1%}")
    print(f"avg_answer_tokens: {total_answer_tokens / total:.1f}")
    print(f"raw_generations: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
