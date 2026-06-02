import json
import os
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

import graph_lib
import noise_lib
from data_sft import (
    DEFAULT_DATASET,
    get_answer_text,
    load_s1k_records,
    split_records,
)
from eval_correct import build_arg_parser, get_gold_answer, make_prompt_ids
from sampling import get_pc_sampler
from sft_utils import (
    decode_until_eos,
    extract_boxed,
    load_sedd_for_inference,
    normalize_answer,
    resolve_dtype,
)


@torch.no_grad()
def sample_full_context_batch(
    model,
    graph,
    noise,
    tokenizer,
    prompt_ids_list,
    args,
    device,
    dtype,
):
    batch_size = len(prompt_ids_list)
    eos_token_id = tokenizer.eos_token_id
    clamp_values = torch.full(
        (batch_size, args.max_length),
        eos_token_id,
        dtype=torch.long,
        device=device,
    )
    answer_mask = torch.zeros(
        (batch_size, args.max_length),
        dtype=torch.bool,
        device=device,
    )
    prompt_lens = []
    answer_lens = []
    for row_idx, prompt_ids in enumerate(prompt_ids_list):
        prompt = prompt_ids.squeeze(0).to(device)
        prompt_len = int(prompt.shape[0])
        answer_len = min(args.max_answer_len, args.max_length - prompt_len)
        if answer_len <= 0:
            raise ValueError("prompt is too long for the requested max_length")
        clamp_values[row_idx, :prompt_len] = prompt
        answer_mask[row_idx, prompt_len : prompt_len + answer_len] = True
        prompt_lens.append(prompt_len)
        answer_lens.append(answer_len)

    init_x = clamp_values.clone()
    init_answer = graph.sample_limit(batch_size, args.max_length).to(device)
    init_x[answer_mask] = init_answer[answer_mask]

    def clamp_known_tokens(x):
        x = x.clone()
        x[~answer_mask] = clamp_values[~answer_mask]
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
        proj_fun=clamp_known_tokens,
        init_x=init_x,
        sampling_mode=args.sampling_mode,
    )
    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    with torch.amp.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=autocast_enabled,
    ):
        generated = sampler(model)
    return generated, prompt_lens, answer_lens


def build_parser():
    parser = build_arg_parser()
    parser.description = "SEDD SFT math generation evaluation with batched full-context sampling"
    parser.add_argument("--eval_batch_size", type=int, default=16)
    return parser


def main():
    args = build_parser().parse_args()
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
        print(f"Loaded LoRA checkpoint: {args.lora_ckpt}", flush=True)
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
    records = [row for row in records if get_gold_answer(row, args.answer_field) is not None]
    records = records[args.start_idx : args.start_idx + args.limit]
    print(
        f"Evaluating {len(records)} samples from split={args.split} "
        f"with eval_batch_size={args.eval_batch_size}",
        flush=True,
    )

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pass1_correct = 0
    passk_correct = 0
    no_boxed = 0
    empty = 0
    no_eos = 0
    exact_correct = 0
    prompt_truncated = 0
    total_len = 0
    total_generations = 0

    with out_path.open("w", encoding="utf-8") as out:
        for batch_start in range(0, len(records), args.eval_batch_size):
            batch_records = records[batch_start : batch_start + args.eval_batch_size]
            prompt_ids_list = []
            prompt_meta = []
            for record in batch_records:
                prompt_ids, was_truncated, prompt_original_len = make_prompt_ids(
                    tokenizer,
                    record["question"],
                    args.answer_prefix,
                    args.answer_leading_newline,
                    args.boxed_prompt,
                    args.max_length,
                    args.min_answer_len,
                )
                prompt_ids_list.append(prompt_ids)
                prompt_meta.append((was_truncated, prompt_original_len))

            generated, prompt_lens, answer_lens = sample_full_context_batch(
                model,
                graph,
                noise,
                tokenizer,
                prompt_ids_list,
                args,
                device,
                dtype,
            )

            for row_offset, record in enumerate(batch_records):
                local_idx = batch_start + row_offset
                idx = args.start_idx + local_idx
                gold = get_gold_answer(record, args.answer_field)
                target_answer = get_answer_text(record, args.answer_field)
                was_truncated, prompt_original_len = prompt_meta[row_offset]
                prompt_len = prompt_lens[row_offset]
                answer_len = answer_lens[row_offset]
                prompt_truncated += int(was_truncated)

                answer_ids = generated[row_offset, prompt_len : prompt_len + answer_len]
                answer_id_list = answer_ids.detach().cpu().tolist()
                eos_present = tokenizer.eos_token_id in answer_id_list
                answer_suffix = decode_until_eos(tokenizer, answer_ids)
                if args.boxed_prompt:
                    pred_boxed = answer_suffix.strip()
                    answer_text = f"\\boxed{{{pred_boxed}}}"
                else:
                    answer_text = answer_suffix
                    pred_boxed = extract_boxed(answer_text)

                match = normalize_answer(pred_boxed) == normalize_answer(gold)
                exact = answer_text.strip() == target_answer.strip()
                generations = [
                    {
                        "sample_idx": 0,
                        "generated": answer_text,
                        "generated_suffix": answer_suffix,
                        "pred_boxed": pred_boxed,
                        "match": match,
                        "exact": exact,
                        "eos_present": eos_present,
                    }
                ]

                pass1_correct += int(match)
                passk_correct += int(match)
                exact_correct += int(exact)
                no_boxed += int(pred_boxed is None)
                empty += int(not answer_text.strip())
                no_eos += int(not eos_present)
                total_len += len(tokenizer.encode(answer_text, add_special_tokens=False))
                total_generations += 1

                row = {
                    "idx": idx,
                    "question": record["question"],
                    "gold": gold,
                    "target_answer": target_answer,
                    "generated": answer_text,
                    "pred_boxed": pred_boxed,
                    "match": match,
                    "exact": exact,
                    "eos_present": eos_present,
                    "prompt_len": prompt_len,
                    "prompt_original_len": prompt_original_len,
                    "prompt_truncated": was_truncated,
                    "pass_at_k": match,
                    "num_samples": 1,
                    "generations": generations,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()

                print(
                    f"[{idx:3d}] {'OK' if match else 'FAIL'} "
                    f"gold={gold} pred@1={pred_boxed}",
                    flush=True,
                )
                if idx < 5:
                    print(f"      gen[:500]={answer_text[:500]}", flush=True)

    total = max(len(records), 1)
    gen_total = max(total_generations, 1)
    print("=" * 60)
    print(f"pass@1_boxed_match: {pass1_correct}/{len(records)} = {pass1_correct / total:.1%}")
    print(f"exact_target_match: {exact_correct}/{len(records)} = {exact_correct / total:.1%}")
    print(f"pass@1_boxed_match: {passk_correct}/{len(records)} = {passk_correct / total:.1%}")
    print(f"no_boxed_generations: {no_boxed}/{total_generations} = {no_boxed / gen_total:.1%}")
    print(f"empty_generations: {empty}/{total_generations} = {empty / gen_total:.1%}")
    print(f"no_eos_generations: {no_eos}/{total_generations} = {no_eos / gen_total:.1%}")
    print(f"prompt_truncated: {prompt_truncated}/{len(records)} = {prompt_truncated / total:.1%}")
    print(f"avg_gen_tokens: {total_len / gen_total:.1f}")
    print(f"raw_generations: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
