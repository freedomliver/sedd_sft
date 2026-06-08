import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast

import graph_lib
import losses
import noise_lib
from data_sft import (
    DEFAULT_DATASET,
    SFTFormatConfig,
    format_sft_record,
    load_s1k_records,
    split_records,
)
from lora import load_lora_checkpoint, save_lora_checkpoint, trainable_parameter_summary
from model import SEDD


TENSOR_KEYS = (
    "input_ids",
    "corrupted_input_ids",
    "answer_mask",
    "pad_mask",
    "prompt_len",
    "answer_window_len",
    "final_answer_start",
    "final_answer_len",
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sedd(pretrained: str, device: torch.device, dtype: torch.dtype):
    local_config = os.path.join(pretrained, "config.json")
    if os.path.exists(local_config):
        with open(local_config) as f:
            cfg = OmegaConf.create(json.load(f))
        model = SEDD(cfg)
        safetensors_path = os.path.join(pretrained, "model.safetensors")
        bin_path = os.path.join(pretrained, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            state = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            state = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model weights found under {pretrained}")
        model.load_state_dict(state, strict=False)
    else:
        model = SEDD.from_pretrained(pretrained)
        cfg = model.config
    model = model.to(device)
    if dtype != torch.float32:
        model = model.to(dtype)
    return model, cfg


def resolve_dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


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


class RepairDataset(Dataset):
    def __init__(self, clean_records, repair_pairs, tokenizer, format_cfg):
        self.clean_records = clean_records
        self.repair_pairs = repair_pairs
        self.tokenizer = tokenizer
        self.format_cfg = format_cfg

    def __len__(self):
        return len(self.repair_pairs)

    def __getitem__(self, idx):
        pair = self.repair_pairs[idx]
        clean_record = self.clean_records[int(pair["source_idx"])]
        clean_item = format_sft_record(clean_record, self.tokenizer, self.format_cfg)

        corrupted_record = dict(clean_record)
        corrupted_record[self.format_cfg.reasoning_field] = pair["corrupted_reasoning"]
        corrupted_record[self.format_cfg.final_answer_field] = pair["corrupted_answer"]
        corrupted_item = format_sft_record(corrupted_record, self.tokenizer, self.format_cfg)

        out = {
            "input_ids": clean_item["input_ids"],
            "corrupted_input_ids": corrupted_item["input_ids"],
            "answer_mask": clean_item["answer_mask"],
            "pad_mask": clean_item["pad_mask"],
            "prompt_len": clean_item["prompt_len"],
            "answer_window_len": clean_item["answer_window_len"],
            "final_answer_start": clean_item["final_answer_start"],
            "final_answer_len": clean_item["final_answer_len"],
        }
        return out


def collate(batch):
    return {key: torch.stack([item[key] for item in batch]) for key in TENSOR_KEYS}


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Fixed-block semantic repair training")
    parser.add_argument("--pretrained", default="pretrained/sedd-small")
    parser.add_argument("--resume_lora_ckpt", required=True)
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--repair_json", required=True)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_size", type=int, default=320)
    parser.add_argument("--valid_size", type=int, default=40)
    parser.add_argument("--test_size", type=int, default=40)
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--repair_t", type=float, default=0.1)
    parser.add_argument("--loss_type", choices=("dwdse", "ce"), default="ce")
    parser.add_argument("--final_answer_weight", type=float, default=1.0)
    parser.add_argument("--final_answer_pad_weight", type=float, default=0.25)
    parser.add_argument("--save_freq", type=int, default=250)
    parser.add_argument("--log_freq", type=int, default=20)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--offline", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype)
    if device.type != "cuda" and dtype != torch.float32:
        dtype = torch.float32

    print(f"Loading base model: {args.pretrained}", flush=True)
    model, cfg = load_sedd(args.pretrained, device, dtype)
    lora_info = load_lora_checkpoint(model, args.resume_lora_ckpt, map_location="cpu")
    lora_cfg = lora_info["lora_config"]
    print(f"Resumed LoRA: {args.resume_lora_ckpt}", flush=True)
    summary = trainable_parameter_summary(model)
    print(
        f"Trainable params: {summary['trainable']:,}/{summary['total']:,} "
        f"({summary['ratio']:.4%})",
        flush=True,
    )

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=args.offline)
    all_records = load_s1k_records(args.data_json, args.dataset_name, args.hf_split, args.cache_dir)
    train_records = split_records(
        all_records,
        "train",
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    repair_pairs = json.loads(Path(args.repair_json).read_text(encoding="utf-8"))
    print(f"Repair pairs: {len(repair_pairs)}", flush=True)
    dataset = RepairDataset(train_records, repair_pairs, tokenizer, build_format_cfg(args))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
    )
    if len(loader) == 0:
        raise ValueError("Repair loader is empty")

    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    loss_fn = losses.get_response_only_sft_loss_fn(
        noise,
        graph,
        final_answer_weight=args.final_answer_weight,
        final_answer_pad_weight=args.final_answer_pad_weight,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    global_step = 0
    micro_step = 0
    running = []
    start = time.time()
    while global_step < args.max_steps:
        for batch in loader:
            batch = move_batch(batch, device)
            t = torch.full(
                (batch["input_ids"].shape[0],),
                float(args.repair_t),
                device=device,
            )
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                if args.loss_type == "dwdse":
                    loss = loss_fn(
                        model,
                        batch["input_ids"],
                        batch["answer_mask"],
                        batch["pad_mask"],
                        prompt_len=batch["prompt_len"],
                        answer_window_len=int(batch["answer_window_len"][0].item()),
                        final_answer_start=batch["final_answer_start"],
                        final_answer_len=batch["final_answer_len"],
                        t=t,
                        perturbed_batch=batch["corrupted_input_ids"],
                    ).mean()
                else:
                    sigma, _ = noise(t)
                    logits = model(batch["corrupted_input_ids"], sigma.reshape(-1))
                    ce = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]).float(),
                        batch["input_ids"].reshape(-1),
                        reduction="none",
                    ).reshape_as(batch["input_ids"])
                    mask = batch["answer_mask"].bool() & batch["pad_mask"].bool()
                    loss = (ce * mask.to(ce.dtype)).sum() / mask.sum().clamp_min(1)
            (loss / args.grad_accum).backward()
            running.append(float(loss.detach().float().item()))
            micro_step += 1
            if micro_step % args.grad_accum != 0:
                continue

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.log_freq == 0:
                n = min(len(running), args.log_freq)
                mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                print(
                    f"step={global_step} repair_loss={sum(running[-n:]) / n:.4f} "
                    f"type={args.loss_type} t={args.repair_t:.3f} mem={mem_gb:.1f}GB "
                    f"elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
            if args.save_freq > 0 and global_step % args.save_freq == 0:
                save_lora_checkpoint(
                    str(out_dir / f"checkpoint_step_{global_step}.pt"),
                    model,
                    lora_cfg,
                    step=global_step,
                    optimizer=optimizer.state_dict(),
                    args=vars(args),
                )
                save_lora_checkpoint(
                    str(out_dir / "checkpoint_last.pt"),
                    model,
                    lora_cfg,
                    step=global_step,
                    optimizer=optimizer.state_dict(),
                    args=vars(args),
                )
                print(f"saved checkpoint_step_{global_step}.pt", flush=True)
            if global_step >= args.max_steps:
                break

    save_lora_checkpoint(
        str(out_dir / "lora_final.pt"),
        model,
        lora_cfg,
        step=global_step,
        optimizer=optimizer.state_dict(),
        args=vars(args),
    )
    print(f"Training complete. Saved {out_dir / 'lora_final.pt'}", flush=True)


if __name__ == "__main__":
    main()
