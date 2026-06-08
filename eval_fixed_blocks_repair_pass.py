import argparse
import json
import os
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

import noise_lib
from data_sft import SFTFormatConfig, format_sft_record, load_s1k_records, split_records
from sft_utils import decode_until_eos, load_sedd_for_inference, normalize_answer, resolve_dtype


def build_arg_parser():
    parser = argparse.ArgumentParser(description="One-pass repair eval from corrupted generations")
    parser.add_argument("--pretrained", default="pretrained/sedd-small")
    parser.add_argument("--repair_lora_ckpt", required=True)
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--corrupt_jsonl", required=True)
    parser.add_argument("--split", choices=("train", "valid", "validation", "test", "all"), default="test")
    parser.add_argument("--train_size", type=int, default=320)
    parser.add_argument("--valid_size", type=int, default=40)
    parser.add_argument("--test_size", type=int, default=40)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--repair_t", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--question_block_len", type=int, default=48)
    parser.add_argument("--reasoning_block_len", type=int, default=96)
    parser.add_argument("--final_answer_block_len", type=int, default=16)
    parser.add_argument("--reasoning_field", default="compressed_reasoning")
    parser.add_argument("--final_answer_field", default="compressed_answer")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--out_jsonl", required=True)
    return parser


def build_format_cfg(args):
    return SFTFormatConfig(
        max_length=args.max_length,
        fixed_layout=True,
        question_block_len=args.question_block_len,
        reasoning_block_len=args.reasoning_block_len,
        final_answer_block_len=args.final_answer_block_len,
        reasoning_field=args.reasoning_field,
        final_answer_field=args.final_answer_field,
        fixed_layout_supervise_pad=True,
    )


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@torch.no_grad()
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
    model, cfg, _ = load_sedd_for_inference(
        args.pretrained,
        args.repair_lora_ckpt,
        device,
        dtype,
    )
    noise = noise_lib.get_noise(cfg).to(device)

    records = split_records(
        load_s1k_records(args.data_json),
        args.split,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )[: args.limit]
    corrupt_rows = load_jsonl(args.corrupt_jsonl)[: args.limit]
    cfg_fmt = build_format_cfg(args)
    clean_items = [format_sft_record(row, tokenizer, cfg_fmt) for row in records]

    corrupted_items = []
    for record, corrupt in zip(records, corrupt_rows):
        corrupted = dict(record)
        corrupted[args.reasoning_field] = corrupt.get("generated_reasoning", "")
        corrupted[args.final_answer_field] = corrupt.get("pred_answer", "")
        corrupted_items.append(format_sft_record(corrupted, tokenizer, cfg_fmt))

    reasoning_start = args.question_block_len
    answer_start = reasoning_start + args.reasoning_block_len
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    with out_path.open("w", encoding="utf-8") as out:
        for batch_start in range(0, len(corrupted_items), args.eval_batch_size):
            batch_corrupt = corrupted_items[batch_start : batch_start + args.eval_batch_size]
            batch_clean = clean_items[batch_start : batch_start + args.eval_batch_size]
            x = torch.stack([item["input_ids"] for item in batch_corrupt]).to(device)
            t = torch.full((x.shape[0],), float(args.repair_t), device=device)
            sigma, _ = noise(t)
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                logits = model(x, sigma.reshape(-1))
            pred = logits.argmax(dim=-1)
            repaired = x.clone()
            repaired[:, reasoning_start : answer_start + args.final_answer_block_len] = pred[
                :, reasoning_start : answer_start + args.final_answer_block_len
            ]
            for row_offset, item in enumerate(batch_clean):
                idx = batch_start + row_offset
                pred_answer = decode_until_eos(
                    tokenizer,
                    repaired[
                        row_offset,
                        answer_start : answer_start + args.final_answer_block_len,
                    ],
                ).strip()
                repaired_reasoning = decode_until_eos(
                    tokenizer,
                    repaired[
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
                            "gold_answer": gold_answer,
                            "pred_answer": pred_answer,
                            "match": match,
                            "repaired_reasoning": repaired_reasoning,
                            "corrupt_pred_answer": corrupt_rows[idx].get("pred_answer", ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                print(
                    f"[{idx:3d}] {'OK' if match else 'FAIL'} gold={gold_answer} pred={pred_answer[:120]}",
                    flush=True,
                )
    total = max(len(corrupted_items), 1)
    print("=" * 60)
    print(f"repair_pass_match: {correct}/{len(corrupted_items)} = {correct / total:.1%}")
    print(f"raw_generations: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
