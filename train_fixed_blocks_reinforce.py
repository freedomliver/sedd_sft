import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
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
from lora import load_lora_checkpoint, save_lora_checkpoint, trainable_parameter_summary
from model import SEDD
from sft_utils import decode_until_eos, normalize_answer, resolve_dtype


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sedd_base(pretrained: str, device: torch.device, dtype=torch.float32):
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


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Fixed-block trajectory REINFORCE for SEDD")
    parser.add_argument("--pretrained", default="pretrained/sedd-small")
    parser.add_argument("--resume_lora_ckpt", required=True)
    parser.add_argument("--data_json", required=True)
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
    parser.add_argument("--question_batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument(
        "--logprob_microbatch_size",
        type=int,
        default=8,
        help="Number of sampled trajectories to replay per differentiable logprob microbatch.",
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--trace_steps", type=int, default=8)
    parser.add_argument("--max_updates", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--entropy_bonus", type=float, default=0.0)
    parser.add_argument("--reward_mode", choices=("exact", "dense"), default="exact")
    parser.add_argument("--answer_reward", type=float, default=1.0)
    parser.add_argument("--trace_reward", type=float, default=0.3)
    parser.add_argument("--format_reward", type=float, default=0.1)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--log_freq", type=int, default=5)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--offline", action="store_true")
    return parser


def _compact_math(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace(" ", "")
    text = text.replace(",", "")
    text = text.replace("\\cdot", "*")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\operatorname", "")
    return text


def _math_atoms(text: str) -> set[str]:
    text = str(text or "")
    atoms = set()
    patterns = [
        r"\\frac\{[-+]?\d+\}\{[-+]?\d+\}",
        r"[-+]?\d+\\pi",
        r"[-+]?\d+/\d+",
        r"[-+]?\d+\^\{?\d+\}?",
        r"[-+]?\d+\s*[+\-*/]\s*[-+]?\d+\s*=\s*[-+]?\d+",
        r"[-+]?\d+",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            atom = _compact_math(match)
            if len(atom) > 1 or atom.isdigit():
                atoms.add(atom)
    return atoms


def trace_overlap_score(generated_reasoning: str, gold_reasoning: str) -> float:
    gold_atoms = _math_atoms(gold_reasoning)
    if not gold_atoms:
        return 0.0
    generated = _compact_math(generated_reasoning)
    hits = sum(1 for atom in gold_atoms if atom in generated)
    return hits / max(len(gold_atoms), 1)


def format_score(generated_reasoning: str, pred_answer: str) -> float:
    text = str(generated_reasoning or "").lower()
    score = 0.0
    if "method" in text:
        score += 0.25
    if "step" in text:
        score += 0.25
    if pred_answer.strip():
        score += 0.25
    if len(text.strip()) >= 20:
        score += 0.25
    return score


def compute_reward(pred_answer: str, generated_reasoning: str, item: dict, args) -> float:
    answer_match = float(
        normalize_answer(pred_answer) == normalize_answer(item["final_answer"].strip())
    )
    if args.reward_mode == "exact":
        return answer_match
    trace_score = trace_overlap_score(generated_reasoning, item.get("answer", ""))
    fmt_score = format_score(generated_reasoning, pred_answer)
    return (
        args.answer_reward * answer_match
        + args.trace_reward * trace_score
        + args.format_reward * fmt_score
    )


def compute_answer_match(pred_answer: str, item: dict) -> float:
    return float(normalize_answer(pred_answer) == normalize_answer(item["final_answer"].strip()))


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
        fixed_layout_supervise_pad=True,
    )


def fixed_spans(args):
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
    return reasoning_start, answer_start


def build_clamp(items, args, device):
    batch_size = len(items)
    reasoning_start, answer_start = fixed_spans(args)
    reasoning_end = reasoning_start + args.reasoning_block_len
    answer_end = answer_start + args.final_answer_block_len
    eos_token_id = items[0]["input_ids"].new_tensor(50256).item()
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
        clamp_values[row_idx, : args.question_block_len] = input_ids[: args.question_block_len]
        generation_mask[row_idx, reasoning_start:reasoning_end] = True
        generation_mask[row_idx, answer_start:answer_end] = True
    return clamp_values, generation_mask, reasoning_start, answer_start


def choose_tokens(probs):
    gumbel_norm = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
    return (probs / gumbel_norm).argmax(dim=-1)


@torch.no_grad()
def sample_trace(model, graph, noise, items, tokenizer, args, device, dtype):
    batch_size = len(items)
    clamp_values, generation_mask, reasoning_start, answer_start = build_clamp(items, args, device)
    init_noise = graph.sample_limit(batch_size, args.max_length).to(device)
    x = clamp_values.clone()
    x[generation_mask] = init_noise[generation_mask]

    timesteps = torch.linspace(1, 1e-4, args.steps + 1, device=device)
    dt = (1 - 1e-4) / args.steps
    trace_every = max(1, args.steps // args.trace_steps)
    trace = []
    model.eval()
    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    for step_idx in range(args.steps):
        x[~generation_mask] = clamp_values[~generation_mask]
        x_before = x.clone()
        t = timesteps[step_idx] * torch.ones(batch_size, 1, device=device)
        curr_sigma = noise(t)[0]
        next_sigma = noise(t - dt)[0]
        dsigma = curr_sigma - next_sigma
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            score = model(x, curr_sigma.reshape(-1)).exp()
            stag_score = graph.staggered_score(score, dsigma)
            probs = stag_score * graph.transp_transition(x, dsigma)
        x = choose_tokens(probs)
        x[~generation_mask] = clamp_values[~generation_mask]
        if step_idx % trace_every == 0 or step_idx == args.steps - 1:
            trace.append(
                {
                    "x_before": x_before.detach().cpu(),
                    "x_after": x.detach().cpu(),
                    "t": t.detach().cpu(),
                    "step_idx": step_idx,
                }
            )

    t = timesteps[-1] * torch.ones(batch_size, 1, device=device)
    x_before = x.clone()
    x[~generation_mask] = clamp_values[~generation_mask]
    with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
        sigma = noise(t)[0]
        score = model(x, sigma.reshape(-1)).exp()
        stag_score = graph.staggered_score(score, sigma)
        probs = stag_score * graph.transp_transition(x, sigma)
        if graph.absorb:
            probs = probs[..., :-1]
    x = choose_tokens(probs)
    x[~generation_mask] = clamp_values[~generation_mask]
    trace.append(
        {
            "x_before": x_before.detach().cpu(),
            "x_after": x.detach().cpu(),
            "t": t.detach().cpu(),
            "step_idx": args.steps,
        }
    )

    rewards = []
    answer_matches = []
    pred_answers = []
    for row_idx, item in enumerate(items):
        answer_ids = x[
            row_idx,
            answer_start : answer_start + args.final_answer_block_len,
        ]
        pred_answer = decode_until_eos(tokenizer, answer_ids).strip()
        generated_reasoning = decode_until_eos(
            tokenizer,
            x[row_idx, reasoning_start : reasoning_start + args.reasoning_block_len],
        ).strip()
        rewards.append(compute_reward(pred_answer, generated_reasoning, item, args))
        answer_matches.append(compute_answer_match(pred_answer, item))
        pred_answers.append(pred_answer)
    return (
        trace,
        torch.tensor(rewards, dtype=torch.float32),
        torch.tensor(answer_matches, dtype=torch.float32),
        pred_answers,
    )


def logprob_loss_for_trace(
    model,
    graph,
    noise,
    trace,
    generation_mask,
    advantages,
    args,
    device,
    dtype,
    row_indices=None,
):
    losses = []
    entropies = []
    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    if row_indices is None:
        row_indices = torch.arange(generation_mask.shape[0], dtype=torch.long)
    row_indices = row_indices.to(device)
    gen_mask = generation_mask.to(device).index_select(0, row_indices)
    adv = advantages.to(device).index_select(0, row_indices)
    for event in trace:
        x_before = event["x_before"].to(device).index_select(0, row_indices)
        x_after = event["x_after"].to(device).index_select(0, row_indices)
        t = event["t"].to(device).index_select(0, row_indices)
        if int(event["step_idx"]) >= args.steps:
            sigma = noise(t)[0]
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                score = model(x_before, sigma.reshape(-1)).exp()
                stag_score = graph.staggered_score(score, sigma)
                probs = stag_score * graph.transp_transition(x_before, sigma)
                if graph.absorb:
                    probs = probs[..., :-1]
        else:
            dt = (1 - 1e-4) / args.steps
            curr_sigma = noise(t)[0]
            next_sigma = noise(t - dt)[0]
            dsigma = curr_sigma - next_sigma
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                score = model(x_before, curr_sigma.reshape(-1)).exp()
                stag_score = graph.staggered_score(score, dsigma)
                probs = stag_score * graph.transp_transition(x_before, dsigma)
        probs = probs.clamp_min(1e-12)
        logp = probs.log().gather(-1, x_after.unsqueeze(-1)).squeeze(-1)
        per_traj_logp = (logp * gen_mask).sum(dim=1) / gen_mask.sum(dim=1).clamp_min(1)
        losses.append(-(adv * per_traj_logp).mean())
        if args.entropy_bonus > 0:
            entropy = -(probs * probs.log()).sum(dim=-1)
            entropies.append((entropy * gen_mask).sum(dim=1).mean() / gen_mask.sum(dim=1).float().mean())
    loss = torch.stack(losses).mean()
    if entropies and args.entropy_bonus > 0:
        loss = loss - args.entropy_bonus * torch.stack(entropies).mean()
    return loss


def backward_logprob_loss_for_trace(
    model,
    graph,
    noise,
    trace,
    generation_mask,
    advantages,
    selected_rows,
    args,
    device,
    dtype,
):
    if selected_rows.numel() == 0:
        return 0.0, 0
    microbatch_size = max(int(args.logprob_microbatch_size), 1)
    total_loss = 0.0
    total_rows = int(selected_rows.numel())
    for start in range(0, total_rows, microbatch_size):
        rows = selected_rows[start : start + microbatch_size]
        loss = logprob_loss_for_trace(
            model,
            graph,
            noise,
            trace,
            generation_mask,
            advantages,
            args,
            device,
            dtype,
            row_indices=rows,
        )
        scaled_loss = loss * (rows.numel() / total_rows)
        scaled_loss.backward()
        total_loss += loss.detach().float().item() * rows.numel()
    return total_loss / max(total_rows, 1), total_rows


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
    model, cfg = load_sedd_base(args.pretrained, device, dtype)
    lora_info = load_lora_checkpoint(model, args.resume_lora_ckpt, map_location="cpu")
    lora_cfg = lora_info["lora_config"]
    print(f"Resumed LoRA: {args.resume_lora_ckpt}", flush=True)
    summary = trainable_parameter_summary(model)
    print(
        f"Trainable params: {summary['trainable']:,}/{summary['total']:,} "
        f"({summary['ratio']:.4%})",
        flush=True,
    )
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=args.offline)

    records = load_s1k_records(args.data_json, args.dataset_name, args.hf_split, args.cache_dir)
    train_records = split_records(
        records,
        "train",
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    format_cfg = build_format_cfg(args)
    items = [format_sft_record(row, tokenizer, format_cfg) for row in train_records]
    if not items:
        raise ValueError("No training items")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    rng = random.Random(args.seed + 2027)
    start = time.time()
    for update in range(1, args.max_updates + 1):
        batch_indices = [rng.randrange(len(items)) for _ in range(args.question_batch_size)]
        batch_items = []
        group_ids = []
        for group_id, idx in enumerate(batch_indices):
            for _ in range(args.num_samples):
                batch_items.append(items[idx])
                group_ids.append(group_id)
        trace, rewards, answer_matches, pred_answers = sample_trace(
            model,
            graph,
            noise,
            batch_items,
            tokenizer,
            args,
            device,
            dtype,
        )
        rewards = rewards.to(device)
        answer_matches = answer_matches.to(device)
        group_ids_t = torch.tensor(group_ids, dtype=torch.long, device=device)
        baselines = torch.zeros_like(rewards)
        for group_id in range(args.question_batch_size):
            mask = group_ids_t == group_id
            baselines[mask] = rewards[mask].mean()
        advantages = rewards - baselines
        selected_rows = []
        mixed_groups = 0
        for group_id in range(args.question_batch_size):
            row_mask = group_ids_t == group_id
            group_rewards = rewards[row_mask]
            if group_rewards.numel() and group_rewards.min().item() < group_rewards.max().item():
                mixed_groups += 1
                selected_rows.extend(torch.nonzero(row_mask, as_tuple=False).flatten().tolist())
        if not selected_rows:
            continue
        selected_rows_t = torch.tensor(selected_rows, dtype=torch.long)
        selected_advantages = advantages[selected_rows_t.to(device)]
        advantages = advantages / selected_advantages.std().clamp_min(1e-6)
        generation_mask = build_clamp(batch_items, args, device)[1].detach().cpu()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value, selected_count = backward_logprob_loss_for_trace(
            model,
            graph,
            noise,
            trace,
            generation_mask,
            advantages.detach().cpu(),
            selected_rows_t,
            args,
            device,
            dtype,
        )
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip,
            )
        optimizer.step()

        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item() if rewards.numel() > 1 else 0.0
        exact_rate = answer_matches.mean().item()
        selected_adv_abs = selected_advantages.abs().mean().item()
        reward_any = 0
        for group_id in range(args.question_batch_size):
            reward_any += int(answer_matches[group_ids_t == group_id].max().item() > 0)
        if update % args.log_freq == 0:
            mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(
                f"update={update} loss={loss_value:.4f} "
                f"reward_mean={reward_mean:.3f} "
                f"reward_std={reward_std:.3f} "
                f"exact_rate={exact_rate:.3f} "
                f"adv_abs={selected_adv_abs:.3f} "
                f"oracle_in_batch={reward_any}/{args.question_batch_size} "
                f"mixed_groups={mixed_groups}/{args.question_batch_size} "
                f"selected_traj={selected_count}/{len(batch_items)} "
                f"mem={mem_gb:.1f}GB elapsed={time.time() - start:.1f}s",
                flush=True,
            )
        if args.save_freq > 0 and update % args.save_freq == 0:
            save_lora_checkpoint(
                str(out_dir / f"checkpoint_update_{update}.pt"),
                model,
                lora_cfg,
                step=update,
                optimizer=optimizer.state_dict(),
                args=vars(args),
            )
            print(f"saved checkpoint_update_{update}.pt", flush=True)

    save_lora_checkpoint(
        str(out_dir / "lora_final.pt"),
        model,
        lora_cfg,
        step=args.max_updates,
        optimizer=optimizer.state_dict(),
        args=vars(args),
    )
    print(f"Training complete. Saved {out_dir / 'lora_final.pt'}", flush=True)


if __name__ == "__main__":
    main()
