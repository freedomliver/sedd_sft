import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from transformers import GPT2TokenizerFast

from data_sft import DEFAULT_DATASET, SFTFormatConfig, get_sft_dataloader
import graph_lib
import losses
from lora import (
    LoRAConfig,
    apply_lora,
    load_lora_checkpoint,
    save_lora_checkpoint,
    trainable_parameter_summary,
)
from model import SEDD
import noise_lib


def save_full_checkpoint(path: str, model: torch.nn.Module, **extra):
    extra = dict(extra)
    extra.pop("optimizer", None)
    payload = {
        "checkpoint_type": "full",
        "model": model.state_dict(),
        **extra,
    }
    torch.save(payload, path)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sedd(pretrained: str, device: torch.device):
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
    return model.to(device), cfg


def resolve_dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def move_batch(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    for key in ("input_ids", "prompt_mask", "answer_mask", "pad_mask", "prompt_len", "answer_len"):
        out[key] = batch[key].to(device)
    return out


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


@torch.no_grad()
def evaluate_loss(
    model,
    loader,
    loss_fn,
    device,
    max_batches: int | None = None,
    dtype=torch.float32,
    answer_window_len: int | None = None,
):
    model.eval()
    losses_out = []
    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    for idx, batch in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        batch = move_batch(batch, device)
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            loss = loss_fn(
                model,
                batch["input_ids"],
                batch["answer_mask"],
                batch["pad_mask"],
                prompt_len=batch["prompt_len"],
                answer_window_len=answer_window_len,
            )
        losses_out.append(loss.mean().detach())
    model.train()
    if not losses_out:
        return None
    return torch.stack(losses_out).mean().item()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="LoRA response-only SEDD SFT")
    parser.add_argument("--pretrained", default="louaaron/sedd-medium")
    parser.add_argument("--data_json", default=None)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--out_dir", default="outputs/lora_sft")
    parser.add_argument("--resume_lora_ckpt", default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_answer_len", type=int, default=32)
    parser.add_argument("--min_answer_len", type=int, default=32)
    parser.add_argument("--answer_field", default="final_boxed")
    parser.add_argument("--answer_truncate", choices=("head", "tail"), default="head")
    parser.add_argument("--answer_prefix", default="Answer:")
    parser.add_argument("--answer_leading_newline", action="store_true")
    parser.add_argument("--boxed_prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--supervise_answer_window_eos", action="store_true")
    parser.add_argument("--train_size", type=int, default=800)
    parser.add_argument("--valid_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--all_mask_ce_weight", type=float, default=0.0)
    parser.add_argument(
        "--sft_t_min",
        type=float,
        default=0.0,
        help="Lower bound for response-only DWDSE diffusion-time sampling.",
    )
    parser.add_argument(
        "--sft_t_max",
        type=float,
        default=None,
        help="Optional upper bound for response-only DWDSE diffusion-time sampling.",
    )
    parser.add_argument(
        "--sft_high_t_frac",
        type=float,
        default=0.0,
        help="Fraction of response-only DWDSE samples drawn from a high-noise t range.",
    )
    parser.add_argument("--sft_high_t_min", type=float, default=0.75)
    parser.add_argument("--sft_high_t_max", type=float, default=None)
    parser.add_argument("--all_mask_ce_t_values", default="0.0001,0.001,0.01,0.1,0.5,1.0")
    parser.add_argument("--all_mask_ce_cropped", action="store_true")
    parser.add_argument(
        "--all_mask_ce_window_eos",
        action="store_true",
        help="Mask the full answer window as CE input; CE targets still exclude post-EOS padding.",
    )
    parser.add_argument("--all_mask_ce_answer_weight", type=float, default=4.0)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_targets",
        default="attn_qkv,attn_out,mlp.0,mlp.2",
        help="Comma-separated module-name suffixes to wrap with LoRA",
    )
    parser.add_argument("--finetune_mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=10)
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
    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype)
    if device.type != "cuda" and dtype != torch.float32:
        print(f"Non-CUDA device detected; using float32 instead of {args.dtype}", flush=True)
        dtype = torch.float32

    print(f"Loading base model: {args.pretrained}", flush=True)
    t0 = time.time()
    model, cfg = load_sedd(args.pretrained, device)
    if dtype != torch.float32:
        model = model.to(dtype)

    resume_payload = None
    if args.finetune_mode == "full":
        if args.resume_lora_ckpt:
            raise ValueError("--resume_lora_ckpt is only supported in lora finetune_mode")
        for param in model.parameters():
            param.requires_grad = True
        lora_cfg = None
        target_suffixes = ()
        replaced = []
        print("Full-parameter fine-tuning enabled", flush=True)
    elif args.resume_lora_ckpt:
        lora_info = load_lora_checkpoint(model, args.resume_lora_ckpt, map_location="cpu")
        lora_cfg = lora_info["lora_config"]
        resume_payload = lora_info["checkpoint"]
        target_suffixes = lora_cfg.target_suffixes
        replaced = [name for name, module in model.named_modules() if module.__class__.__name__ == "LoRALinear"]
        print(f"Resumed LoRA weights from {args.resume_lora_ckpt}", flush=True)
    else:
        target_suffixes = tuple(item.strip() for item in args.lora_targets.split(",") if item.strip())
        lora_cfg = LoRAConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_suffixes=target_suffixes,
        )
        replaced = apply_lora(model, lora_cfg)
    summary = trainable_parameter_summary(model)
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)
    if args.finetune_mode == "lora":
        print(f"LoRA modules: {len(replaced)} matched ({', '.join(target_suffixes)})", flush=True)
    print(
        f"Trainable params: {summary['trainable']:,}/{summary['total']:,} "
        f"({summary['ratio']:.4%})",
        flush=True,
    )

    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    loss_fn = losses.get_response_only_sft_loss_fn(
        noise,
        graph,
        t_min=args.sft_t_min,
        t_max=args.sft_t_max,
        high_t_frac=args.sft_high_t_frac,
        high_t_min=args.sft_high_t_min,
        high_t_max=args.sft_high_t_max,
    )
    print(
        f"Using response-only DWDSE t_range=[{args.sft_t_min:.4f}, "
        f"{args.sft_t_max if args.sft_t_max is not None else 'default'}] "
        f"high_t_frac={args.sft_high_t_frac:.2f} "
        f"high_t_range=[{args.sft_high_t_min:.4f}, "
        f"{args.sft_high_t_max if args.sft_high_t_max is not None else 'default'}]",
        flush=True,
    )

    all_mask_ce_loss_fn = None
    all_mask_ce_t_values = tuple(
        float(item.strip()) for item in args.all_mask_ce_t_values.split(",") if item.strip()
    )
    if args.all_mask_ce_weight > 0:
        all_mask_ce_loss_fn = losses.get_answer_all_mask_ce_loss_fn(
            noise,
            graph,
            t_values=all_mask_ce_t_values,
            cropped=args.all_mask_ce_cropped,
            include_window_eos=args.all_mask_ce_window_eos,
            answer_weight=args.all_mask_ce_answer_weight,
        )
        print(
            f"Using all-mask answer CE: weight={args.all_mask_ce_weight} "
            f"t_values={all_mask_ce_t_values} cropped={args.all_mask_ce_cropped} "
            f"window_eos={args.all_mask_ce_window_eos} "
            f"answer_weight={args.all_mask_ce_answer_weight}",
            flush=True,
        )

    print("Loading tokenizer: gpt2", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=args.offline)
    format_cfg = SFTFormatConfig(
        max_length=args.max_length,
        max_answer_len=args.max_answer_len,
        min_answer_len=args.min_answer_len,
        answer_field=args.answer_field,
        answer_truncate=args.answer_truncate,
        answer_prefix=args.answer_prefix,
        answer_leading_newline=args.answer_leading_newline,
        boxed_prompt=args.boxed_prompt,
        supervise_answer_window_eos=args.supervise_answer_window_eos,
    )
    loader_kwargs = dict(
        tokenizer=tokenizer,
        data_json=args.data_json,
        dataset_name=args.dataset_name,
        hf_split=args.hf_split,
        cache_dir=args.cache_dir,
        seed=args.seed,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        format_cfg=format_cfg,
        num_workers=args.num_workers,
    )
    train_loader = get_sft_dataloader(
        batch_size=args.batch_size,
        split="train",
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    valid_loader = get_sft_dataloader(
        batch_size=args.batch_size,
        split="valid",
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    if len(train_loader) == 0:
        raise ValueError(
            "Train loader is empty; reduce --batch_size, disable train holdouts, "
            "or provide more training records."
        )
    if len(valid_loader) == 0:
        print("Warning: validation split is empty; periodic eval will be skipped", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if resume_payload and resume_payload.get("optimizer"):
        optimizer.load_state_dict(resume_payload["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
            group["weight_decay"] = args.weight_decay
        move_optimizer_state(optimizer, device)
        print(
            f"Resumed optimizer state; using lr={args.lr:.2e} "
            f"weight_decay={args.weight_decay:.2e}",
            flush=True,
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = int(resume_payload.get("step", 0)) if resume_payload else 0
    micro_step = 0
    running = []
    running_dwdse = []
    running_ce = []
    max_steps = args.max_steps if args.max_steps > 0 else None
    autocast_enabled = device.type == "cuda" and dtype != torch.float32

    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = move_batch(batch, device)
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                dwdse_loss = loss_fn(
                    model,
                    batch["input_ids"],
                    batch["answer_mask"],
                    batch["pad_mask"],
                    prompt_len=batch["prompt_len"],
                    answer_window_len=args.max_answer_len,
                ).mean()
            (dwdse_loss / args.grad_accum).backward()

            if all_mask_ce_loss_fn is None:
                ce_loss = dwdse_loss.new_zeros(())
            else:
                with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                    ce_loss = all_mask_ce_loss_fn(
                        model,
                        batch["input_ids"],
                        batch["answer_mask"],
                        batch["pad_mask"],
                        prompt_len=batch["prompt_len"],
                        answer_window_len=args.max_answer_len,
                    )
                (args.all_mask_ce_weight * ce_loss / args.grad_accum).backward()

            loss_value = dwdse_loss.detach().float() + args.all_mask_ce_weight * ce_loss.detach().float()
            running.append(loss_value.item())
            running_dwdse.append(dwdse_loss.detach().float().item())
            running_ce.append(ce_loss.detach().float().item())
            micro_step += 1

            if micro_step % args.grad_accum != 0:
                continue

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
            global_step += 1
            lr_scale = min(global_step / max(args.warmup_steps, 1), 1.0)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_scale
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_freq == 0:
                avg_loss = sum(running[-args.log_freq:]) / min(len(running), args.log_freq)
                avg_dwdse = sum(running_dwdse[-args.log_freq:]) / min(len(running_dwdse), args.log_freq)
                avg_ce = sum(running_ce[-args.log_freq:]) / min(len(running_ce), args.log_freq)
                mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                print(
                    f"epoch={epoch} step={global_step} loss={avg_loss:.4f} "
                    f"dwdse={avg_dwdse:.4f} all_mask_ce={avg_ce:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} mem={mem_gb:.1f}GB",
                    flush=True,
                )

            if args.eval_freq > 0 and global_step % args.eval_freq == 0:
                val_loss = evaluate_loss(
                    model,
                    valid_loader,
                    loss_fn,
                    device,
                    args.eval_batches,
                    dtype,
                    answer_window_len=args.max_answer_len,
                )
                if val_loss is None:
                    print(f"eval step={global_step} skipped: no validation batches", flush=True)
                else:
                    print(f"eval step={global_step} valid_loss={val_loss:.4f}", flush=True)

            if args.save_freq > 0 and global_step % args.save_freq == 0:
                if args.finetune_mode == "full":
                    save_full_checkpoint(
                        str(out_dir / f"checkpoint_step_{global_step}.pt"),
                        model,
                        step=global_step,
                        optimizer=optimizer.state_dict(),
                        args=vars(args),
                    )
                    save_full_checkpoint(
                        str(out_dir / "checkpoint_last.pt"),
                        model,
                        step=global_step,
                        optimizer=optimizer.state_dict(),
                        args=vars(args),
                    )
                else:
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

            if max_steps is not None and global_step >= max_steps:
                break

        if max_steps is not None and global_step >= max_steps:
            break

    final_name = "full_final.pt" if args.finetune_mode == "full" else "lora_final.pt"
    if args.finetune_mode == "full":
        save_full_checkpoint(
            str(out_dir / final_name),
            model,
            step=global_step,
            optimizer=optimizer.state_dict(),
            args=vars(args),
        )
    else:
        save_lora_checkpoint(
            str(out_dir / final_name),
            model,
            lora_cfg,
            step=global_step,
            optimizer=optimizer.state_dict(),
            args=vars(args),
        )
    print(f"Training complete. Saved {out_dir / final_name}", flush=True)


if __name__ == "__main__":
    main()
