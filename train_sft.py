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
    for key in BATCH_TENSOR_KEYS:
        out[key] = batch[key].to(device)
    return out


BATCH_TENSOR_KEYS = (
        "input_ids",
        "prompt_mask",
        "answer_mask",
        "pad_mask",
        "prompt_len",
        "answer_len",
        "answer_window_len",
        "reasoning_len",
        "final_answer_len",
        "question_block_len",
        "reasoning_start",
        "reasoning_block_len",
        "final_answer_block_len",
        "final_answer_start",
)


class PerturbedReplayCache:
    def __init__(self, t_grid: tuple[float, ...], max_size: int, seed: int = 0):
        if not t_grid:
            raise ValueError("PerturbedReplayCache requires a non-empty t_grid")
        self.t_grid = tuple(float(t) for t in t_grid)
        self.max_per_bucket = max(1, int(max_size) // len(self.t_grid))
        self.rng = random.Random(seed)
        self.buckets: list[list[dict[str, torch.Tensor]]] = [
            [] for _ in self.t_grid
        ]
        self.write_pos = [0 for _ in self.t_grid]

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self.buckets)

    def bucket_sizes(self) -> list[int]:
        return [len(bucket) for bucket in self.buckets]

    def add_batch(
        self,
        batch: dict[str, torch.Tensor],
        perturbed_batch: torch.Tensor,
        t_values: torch.Tensor,
        t_bucket_ids: torch.Tensor,
    ) -> None:
        source = {
            key: batch[key].detach().cpu()
            for key in BATCH_TENSOR_KEYS
            if key in batch
        }
        perturbed_cpu = perturbed_batch.detach().cpu()
        t_cpu = t_values.detach().cpu()
        bucket_cpu = t_bucket_ids.detach().cpu()
        for row in range(perturbed_cpu.shape[0]):
            bucket_id = int(bucket_cpu[row].item())
            item = {key: value[row].clone() for key, value in source.items()}
            item["perturbed_batch"] = perturbed_cpu[row].clone()
            item["t"] = t_cpu[row].clone()
            bucket = self.buckets[bucket_id]
            if len(bucket) < self.max_per_bucket:
                bucket.append(item)
            else:
                pos = self.write_pos[bucket_id] % self.max_per_bucket
                bucket[pos] = item
                self.write_pos[bucket_id] += 1

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor] | None:
        available = [idx for idx, bucket in enumerate(self.buckets) if bucket]
        if not available:
            return None
        per_bucket = _balanced_bucket_ids(batch_size, len(self.t_grid), device=torch.device("cpu"))
        items = []
        for bucket_id in per_bucket.tolist():
            if not self.buckets[bucket_id]:
                bucket_id = self.rng.choice(available)
            items.append(self.rng.choice(self.buckets[bucket_id]))
        out = {}
        keys = list(items[0].keys())
        for key in keys:
            out[key] = torch.stack([item[key] for item in items], dim=0).to(device)
        return out


def _balanced_bucket_ids(batch_size: int, num_buckets: int, device: torch.device) -> torch.Tensor:
    repeats = (batch_size + num_buckets - 1) // num_buckets
    ids = torch.arange(num_buckets, device=device).repeat(repeats)[:batch_size]
    perm = torch.randperm(batch_size, device=device)
    return ids[perm]


def build_fixed_t_batch(
    batch_size: int,
    t_grid: tuple[float, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    bucket_ids = _balanced_bucket_ids(batch_size, len(t_grid), device)
    grid = torch.tensor(t_grid, dtype=torch.float32, device=device)
    return grid[bucket_ids], bucket_ids


@torch.no_grad()
def sample_perturbed_batch(
    graph,
    noise,
    input_ids: torch.Tensor,
    t_values: torch.Tensor,
) -> torch.Tensor:
    sigma, _ = noise(t_values)
    return graph.sample_transition(input_ids, sigma[:, None])


def merge_loss_batch(
    online_batch: dict[str, torch.Tensor],
    online_perturbed: torch.Tensor | None,
    online_t: torch.Tensor | None,
    replay_batch: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    if replay_batch is None:
        out = dict(online_batch)
        if online_perturbed is not None:
            out["perturbed_batch"] = online_perturbed
        if online_t is not None:
            out["t"] = online_t
        return out

    out = {}
    for key in BATCH_TENSOR_KEYS:
        out[key] = torch.cat([online_batch[key], replay_batch[key]], dim=0)
    if online_perturbed is not None and online_t is not None:
        out["perturbed_batch"] = torch.cat(
            [online_perturbed, replay_batch["perturbed_batch"]], dim=0
        )
        out["t"] = torch.cat([online_t, replay_batch["t"]], dim=0)
    return out


def get_batch_answer_window_len(batch: dict, fallback: int | None = None) -> int | None:
    values = batch.get("answer_window_len")
    if values is None:
        return fallback
    unique = torch.unique(values.detach())
    if unique.numel() != 1:
        raise ValueError(f"mixed answer_window_len in batch: {unique.detach().cpu().tolist()}")
    return int(unique.item())


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
    perturb_answer_mask_only: bool = False,
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
                prompt_len=None if perturb_answer_mask_only else batch["prompt_len"],
                answer_window_len=None
                if perturb_answer_mask_only
                else get_batch_answer_window_len(batch, answer_window_len),
                final_answer_start=batch["final_answer_start"],
                final_answer_len=batch["final_answer_len"],
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
    parser.add_argument("--fixed_layout", action="store_true")
    parser.add_argument("--question_block_len", type=int, default=231)
    parser.add_argument("--reasoning_block_len", type=int, default=530)
    parser.add_argument("--final_answer_block_len", type=int, default=263)
    parser.add_argument("--fixed_layout_reasoning_start", type=int, default=None)
    parser.add_argument("--fixed_layout_final_answer_start", type=int, default=None)
    parser.add_argument("--reasoning_field", default="solution")
    parser.add_argument("--final_answer_field", default="answer")
    parser.add_argument("--fixed_layout_supervise_pad", action="store_true")
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
    parser.add_argument(
        "--num_t_per_sample",
        "--sft_num_t_per_sample",
        dest="num_t_per_sample",
        type=int,
        default=1,
        help="Number of independently sampled diffusion times to average per batch.",
    )
    parser.add_argument(
        "--sft_t_grid",
        default="",
        help="Comma-separated fixed diffusion-time grid. Overrides random multi-t sampling.",
    )
    parser.add_argument(
        "--replay_batch_size",
        type=int,
        default=0,
        help="Number of cached perturbed examples to concatenate with each online batch.",
    )
    parser.add_argument(
        "--replay_t_grid",
        default="",
        help="Comma-separated diffusion-time grid used for online/replay perturbed cache.",
    )
    parser.add_argument(
        "--batch_t_grid",
        default="",
        help="Comma-separated diffusion-time grid assigned across each fresh online batch.",
    )
    parser.add_argument(
        "--replay_cache_size",
        type=int,
        default=4096,
        help="Maximum number of cached perturbed examples kept across all t buckets.",
    )
    parser.add_argument(
        "--fixed_layout_final_answer_weight",
        type=float,
        default=1.0,
        help="Weight multiplier for final-answer tokens inside the fixed layout answer window.",
    )
    parser.add_argument(
        "--fixed_layout_final_answer_pad_weight",
        type=float,
        default=1.0,
        help="Weight multiplier for EOS pad tokens inside the fixed layout final-answer block.",
    )
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
    if args.num_t_per_sample < 1:
        raise ValueError("--num_t_per_sample must be >= 1")
    sft_t_grid = tuple(
        float(item.strip()) for item in args.sft_t_grid.split(",") if item.strip()
    )
    replay_t_grid = tuple(
        float(item.strip()) for item in args.replay_t_grid.split(",") if item.strip()
    )
    batch_t_grid = tuple(
        float(item.strip()) for item in args.batch_t_grid.split(",") if item.strip()
    )
    if args.replay_batch_size < 0:
        raise ValueError("--replay_batch_size must be >= 0")
    if args.replay_batch_size > 0:
        if not replay_t_grid:
            raise ValueError("--replay_t_grid is required when --replay_batch_size > 0")
        if batch_t_grid:
            raise ValueError("--batch_t_grid cannot be combined with replay cache")
        if args.num_t_per_sample != 1:
            raise ValueError("replay cache requires --num_t_per_sample 1")
        if sft_t_grid:
            raise ValueError("use --replay_t_grid, not --sft_t_grid, with replay cache")
        if args.all_mask_ce_weight != 0:
            raise ValueError("replay cache experiment currently supports DWDSE-only")
    if batch_t_grid:
        if args.num_t_per_sample != 1:
            raise ValueError("fresh batch t-grid requires --num_t_per_sample 1")
        if sft_t_grid:
            raise ValueError("use --batch_t_grid, not --sft_t_grid, for per-sample t buckets")
    loss_fn = losses.get_response_only_sft_loss_fn(
        noise,
        graph,
        t_min=args.sft_t_min,
        t_max=args.sft_t_max,
        high_t_frac=args.sft_high_t_frac,
        high_t_min=args.sft_high_t_min,
        high_t_max=args.sft_high_t_max,
        final_answer_weight=args.fixed_layout_final_answer_weight,
        final_answer_pad_weight=args.fixed_layout_final_answer_pad_weight,
    )
    print(
        f"Using response-only DWDSE t_range=[{args.sft_t_min:.4f}, "
        f"{args.sft_t_max if args.sft_t_max is not None else 'default'}] "
        f"high_t_frac={args.sft_high_t_frac:.2f} "
        f"high_t_range=[{args.sft_high_t_min:.4f}, "
        f"{args.sft_high_t_max if args.sft_high_t_max is not None else 'default'}] "
        f"num_t_per_sample={args.num_t_per_sample} "
        f"t_grid={sft_t_grid if sft_t_grid else 'random'} "
        f"final_answer_weight={args.fixed_layout_final_answer_weight:.2f} "
        f"final_answer_pad_weight={args.fixed_layout_final_answer_pad_weight:.2f}",
        flush=True,
    )
    if args.replay_batch_size > 0:
        print(
            f"Using perturbed replay cache: online_batch={args.batch_size} "
            f"replay_batch={args.replay_batch_size} "
            f"total_loss_batch={args.batch_size + args.replay_batch_size} "
            f"t_grid={replay_t_grid} cache_size={args.replay_cache_size}",
            flush=True,
        )
    if batch_t_grid:
        print(
            f"Using fresh per-sample t grid: batch_size={args.batch_size} "
            f"t_grid={batch_t_grid}",
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
        fixed_layout=args.fixed_layout,
        question_block_len=args.question_block_len,
        reasoning_block_len=args.reasoning_block_len,
        final_answer_block_len=args.final_answer_block_len,
        fixed_layout_reasoning_start=args.fixed_layout_reasoning_start,
        fixed_layout_final_answer_start=args.fixed_layout_final_answer_start,
        reasoning_field=args.reasoning_field,
        final_answer_field=args.final_answer_field,
        fixed_layout_supervise_pad=args.fixed_layout_supervise_pad,
    )
    if args.fixed_layout:
        reasoning_start_for_log = (
            args.fixed_layout_reasoning_start
            if args.fixed_layout_reasoning_start is not None
            else args.question_block_len
        )
        final_answer_start_for_log = (
            args.fixed_layout_final_answer_start
            if args.fixed_layout_final_answer_start is not None
            else reasoning_start_for_log + args.reasoning_block_len
        )
        print(
            "Using fixed layout: "
            f"question=[0,{args.question_block_len}) "
            f"reasoning=[{reasoning_start_for_log},"
            f"{reasoning_start_for_log + args.reasoning_block_len}) "
            f"final_answer=[{final_answer_start_for_log},"
            f"{final_answer_start_for_log + args.final_answer_block_len}) "
            f"supervise_pad={args.fixed_layout_supervise_pad}",
            flush=True,
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
    dwdse_t_values = sft_t_grid if sft_t_grid else (None,) * args.num_t_per_sample
    replay_cache = None
    if args.replay_batch_size > 0:
        replay_cache = PerturbedReplayCache(
            replay_t_grid,
            max_size=args.replay_cache_size,
            seed=args.seed + 1009,
        )
    perturb_answer_mask_only = (
        args.fixed_layout_reasoning_start is not None
        or args.fixed_layout_final_answer_start is not None
    )

    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = move_batch(batch, device)
            replay_used = 0
            if replay_cache is not None or batch_t_grid:
                active_t_grid = replay_t_grid if replay_cache is not None else batch_t_grid
                online_t, online_t_bucket = build_fixed_t_batch(
                    batch["input_ids"].shape[0],
                    active_t_grid,
                    device,
                )
                online_perturbed = sample_perturbed_batch(
                    graph,
                    noise,
                    batch["input_ids"],
                    online_t,
                )
                replay_batch = None
                if replay_cache is not None:
                    replay_batch = replay_cache.sample(args.replay_batch_size, device)
                    if replay_batch is not None:
                        replay_used = int(replay_batch["input_ids"].shape[0])
                loss_batch = merge_loss_batch(
                    batch,
                    online_perturbed,
                    online_t,
                    replay_batch,
                )
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=autocast_enabled,
                ):
                    dwdse_loss = loss_fn(
                        model,
                        loss_batch["input_ids"],
                        loss_batch["answer_mask"],
                        loss_batch["pad_mask"],
                        prompt_len=None
                        if perturb_answer_mask_only
                        else loss_batch["prompt_len"],
                        answer_window_len=None
                        if perturb_answer_mask_only
                        else get_batch_answer_window_len(
                            loss_batch,
                            args.max_answer_len,
                        ),
                        final_answer_start=loss_batch["final_answer_start"],
                        final_answer_len=loss_batch["final_answer_len"],
                        t=loss_batch["t"],
                        perturbed_batch=loss_batch["perturbed_batch"],
                    ).mean()
                (dwdse_loss / args.grad_accum).backward()
                if replay_cache is not None:
                    replay_cache.add_batch(
                        batch,
                        online_perturbed,
                        online_t,
                        online_t_bucket,
                    )
            else:
                dwdse_loss_sum = None
                for t_value in dwdse_t_values:
                    if t_value is None:
                        t_arg = None
                    else:
                        t_arg = torch.full(
                            (batch["input_ids"].shape[0],),
                            float(t_value),
                            device=device,
                        )
                    with torch.amp.autocast(
                        device_type=device.type,
                        dtype=dtype,
                        enabled=autocast_enabled,
                    ):
                        dwdse_component = loss_fn(
                            model,
                            batch["input_ids"],
                            batch["answer_mask"],
                            batch["pad_mask"],
                            prompt_len=None
                            if perturb_answer_mask_only
                            else batch["prompt_len"],
                            answer_window_len=None
                            if perturb_answer_mask_only
                            else get_batch_answer_window_len(batch, args.max_answer_len),
                            final_answer_start=batch["final_answer_start"],
                            final_answer_len=batch["final_answer_len"],
                            t=t_arg,
                        ).mean()
                    (
                        dwdse_component
                        / args.grad_accum
                        / len(dwdse_t_values)
                    ).backward()
                    if dwdse_loss_sum is None:
                        dwdse_loss_sum = dwdse_component.detach().float()
                    else:
                        dwdse_loss_sum = dwdse_loss_sum + dwdse_component.detach().float()
                dwdse_loss = dwdse_loss_sum / len(dwdse_t_values)

            if all_mask_ce_loss_fn is None:
                ce_loss = dwdse_loss.new_zeros(())
            else:
                with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                    ce_loss = all_mask_ce_loss_fn(
                        model,
                        batch["input_ids"],
                        batch["answer_mask"],
                        batch["pad_mask"],
                        prompt_len=None
                        if perturb_answer_mask_only
                        else batch["prompt_len"],
                        answer_window_len=None
                        if perturb_answer_mask_only
                        else get_batch_answer_window_len(batch, args.max_answer_len),
                        final_answer_start=batch["final_answer_start"],
                        final_answer_len=batch["final_answer_len"],
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
                    f"lr={optimizer.param_groups[0]['lr']:.2e} mem={mem_gb:.1f}GB"
                    + (
                        f" replay={replay_used} cache={len(replay_cache)}"
                        if replay_cache is not None
                        else ""
                    ),
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
                    perturb_answer_mask_only=perturb_answer_mask_only,
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
