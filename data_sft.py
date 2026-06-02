import json
import random
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast


DEFAULT_DATASET = "simplescaling/s1K-1.1"
DEFAULT_TOKENIZER = "gpt2"
BOXED_PREFIX = r"\boxed{"


def build_prompt(
    question: str,
    answer_prefix: str = "Answer:",
    trailing_newline: bool = True,
) -> str:
    prompt = f"Question:\n{question.strip()}\n\n{answer_prefix.strip()}"
    return f"{prompt}\n" if trailing_newline else prompt


def get_answer_text(record: dict, answer_field: str = "solution") -> str:
    if answer_field in {"boxed_inner", "final_boxed_inner", "boxed_answer_inner"}:
        boxed = extract_boxed(record.get("deepseek_attempt", "")) or extract_boxed(
            record.get("solution", "")
        )
        if boxed:
            return boxed

    if answer_field in {"boxed", "final_boxed", "boxed_answer"}:
        boxed = extract_boxed(record.get("deepseek_attempt", "")) or extract_boxed(
            record.get("solution", "")
        )
        if boxed:
            return f"\\boxed{{{boxed}}}"

    answer = record.get(answer_field)
    if answer:
        return str(answer).strip()
    for fallback in ("solution", "deepseek_attempt", "answer"):
        answer = record.get(fallback)
        if answer:
            return str(answer).strip()
    return ""


def extract_boxed(text) -> str | None:
    results = []
    pos = 0
    text = str(text or "")
    while pos < len(text):
        idx = text.find(BOXED_PREFIX, pos)
        if idx == -1:
            break
        start = idx + len(BOXED_PREFIX)
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start : i - 1].strip())
        pos = i
    return results[-1] if results else None


def load_s1k_records(
    data_json: str | None = None,
    dataset_name: str = DEFAULT_DATASET,
    hf_split: str = "train",
    cache_dir: str | None = None,
) -> list[dict]:
    if data_json:
        with open(data_json, encoding="utf-8") as f:
            records = json.load(f)
        return list(records)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required when --data_json is not provided"
        ) from exc

    ds = load_dataset(dataset_name, split=hf_split, cache_dir=cache_dir)
    return [dict(row) for row in ds]


def split_records(
    records: list[dict],
    split: str,
    train_size: int = 800,
    valid_size: int = 100,
    test_size: int = 100,
    seed: int = 0,
    shuffle: bool = True,
) -> list[dict]:
    records = list(records)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)

    train_size, valid_size, test_size = resolve_split_sizes(
        len(records),
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
    )
    train_end = train_size
    valid_end = train_end + valid_size
    test_end = valid_end + test_size

    if split == "train":
        return records[:train_end]
    if split in {"valid", "validation", "val"}:
        return records[train_end:valid_end]
    if split == "test":
        return records[valid_end:test_end]
    if split == "all":
        return records
    raise ValueError(f"Unknown split: {split}")


def resolve_split_sizes(
    n_records: int,
    train_size: int = 800,
    valid_size: int = 100,
    test_size: int = 100,
) -> tuple[int, int, int]:
    """Return concrete split sizes while keeping validation/test non-empty on small JSON files."""
    if n_records <= 0:
        return 0, 0, 0

    requested_total = train_size + valid_size + test_size
    if requested_total <= 0:
        return n_records, 0, 0

    if n_records >= requested_total:
        return train_size, valid_size, test_size

    holdout_size = valid_size + test_size
    if holdout_size > 0 and n_records > holdout_size:
        return n_records - holdout_size, valid_size, test_size

    train_n = max(1, round(n_records * train_size / requested_total))
    valid_n = round(n_records * valid_size / requested_total)
    test_n = n_records - train_n - valid_n
    if test_size > 0 and test_n == 0 and n_records > 2:
        test_n = 1
        train_n -= 1
    if valid_size > 0 and valid_n == 0 and n_records > 2:
        valid_n = 1
        train_n -= 1
    return max(train_n, 0), max(valid_n, 0), max(test_n, 0)


@dataclass(frozen=True)
class SFTFormatConfig:
    max_length: int = 1024
    max_answer_len: int = 512
    min_answer_len: int = 32
    answer_field: str = "solution"
    answer_truncate: str = "head"
    answer_prefix: str = "Answer:"
    answer_leading_newline: bool = False
    boxed_prompt: bool = False
    supervise_answer_window_eos: bool = False


def _truncate_answer(
    answer_ids: list[int],
    budget: int,
    eos_token_id: int,
    mode: str = "head",
) -> list[int]:
    if budget <= 0:
        return []
    if len(answer_ids) <= budget:
        return answer_ids
    if budget == 1:
        return [eos_token_id]
    if mode == "head":
        return answer_ids[: budget - 1] + [eos_token_id]
    if mode == "tail":
        return answer_ids[-budget:]
    raise ValueError(f"Unknown answer_truncate mode: {mode}")


def format_sft_record(
    record: dict,
    tokenizer: GPT2TokenizerFast,
    cfg: SFTFormatConfig,
) -> dict:
    pad_token_id = tokenizer.eos_token_id
    question = str(record.get("question", "")).strip()
    answer_text = get_answer_text(record, cfg.answer_field)
    if cfg.boxed_prompt:
        boxed = extract_boxed(answer_text)
        answer_text = boxed if boxed is not None else answer_text.strip()
    elif cfg.answer_leading_newline:
        answer_text = "\n" + answer_text

    if cfg.boxed_prompt:
        prompt_text = build_prompt(question, cfg.answer_prefix, trailing_newline=True) + BOXED_PREFIX
    else:
        prompt_text = build_prompt(
            question,
            cfg.answer_prefix,
            trailing_newline=not cfg.answer_leading_newline,
        )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(
        answer_text + tokenizer.eos_token,
        add_special_tokens=False,
    )

    max_prompt_len = max(1, cfg.max_length - cfg.min_answer_len)
    prompt_ids = prompt_ids[:max_prompt_len]

    answer_budget = min(cfg.max_answer_len, cfg.max_length - len(prompt_ids))
    answer_ids = _truncate_answer(
        answer_ids,
        answer_budget,
        tokenizer.eos_token_id,
        cfg.answer_truncate,
    )
    if cfg.supervise_answer_window_eos and len(answer_ids) < answer_budget:
        answer_ids = answer_ids + [tokenizer.eos_token_id] * (answer_budget - len(answer_ids))

    input_ids = prompt_ids + answer_ids
    nonpad_len = len(input_ids)
    pad_len = cfg.max_length - nonpad_len
    if pad_len < 0:
        raise ValueError("Formatted sample exceeded max_length")
    input_ids = input_ids + [pad_token_id] * pad_len

    prompt_mask = [1] * len(prompt_ids) + [0] * (cfg.max_length - len(prompt_ids))
    answer_mask = (
        [0] * len(prompt_ids)
        + [1] * len(answer_ids)
        + [0] * (cfg.max_length - len(prompt_ids) - len(answer_ids))
    )
    pad_mask = [1] * nonpad_len + [0] * pad_len

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "prompt_mask": torch.tensor(prompt_mask, dtype=torch.bool),
        "answer_mask": torch.tensor(answer_mask, dtype=torch.bool),
        "pad_mask": torch.tensor(pad_mask, dtype=torch.bool),
        "prompt_len": torch.tensor(len(prompt_ids), dtype=torch.long),
        "answer_len": torch.tensor(len(answer_ids), dtype=torch.long),
        "question": question,
        "answer": answer_text,
        "prompt_text": prompt_text,
    }


_TOKENIZER_CACHE = None


def _get_default_tokenizer():
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        _TOKENIZER_CACHE = GPT2TokenizerFast.from_pretrained(DEFAULT_TOKENIZER)
    return _TOKENIZER_CACHE


def format_sample(record: dict, tokenizer: GPT2TokenizerFast | None = None) -> dict:
    """Deprecated compatibility wrapper for older diagnostics.

    New code should use ``format_sft_record`` and the explicit prompt/answer/pad
    masks. The legacy keys are mapped onto the response-only format.
    """
    tokenizer = tokenizer or _get_default_tokenizer()
    item = format_sft_record(record, tokenizer, SFTFormatConfig())
    return {
        **item,
        "condition_len": item["prompt_len"],
        "boxed_mask": item["answer_mask"].long(),
    }


class S1KResponseDataset(Dataset):
    def __init__(
        self,
        records: Iterable[dict],
        tokenizer: GPT2TokenizerFast,
        format_cfg: SFTFormatConfig | None = None,
    ):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.format_cfg = format_cfg or SFTFormatConfig()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return format_sft_record(self.records[idx], self.tokenizer, self.format_cfg)


def collate_sft(batch: list[dict]) -> dict:
    tensor_keys = ("input_ids", "prompt_mask", "answer_mask", "pad_mask", "prompt_len", "answer_len")
    out = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    out["condition_len"] = out["prompt_len"]
    out["boxed_mask"] = out["answer_mask"].long()
    out["question"] = [item["question"] for item in batch]
    out["answer"] = [item["answer"] for item in batch]
    out["prompt_text"] = [item["prompt_text"] for item in batch]
    return out


def build_sft_dataset(
    split: str,
    tokenizer: GPT2TokenizerFast,
    data_json: str | None = None,
    dataset_name: str = DEFAULT_DATASET,
    hf_split: str = "train",
    cache_dir: str | None = None,
    seed: int = 0,
    train_size: int = 800,
    valid_size: int = 100,
    test_size: int = 100,
    shuffle: bool = True,
    format_cfg: SFTFormatConfig | None = None,
) -> S1KResponseDataset:
    records = load_s1k_records(data_json, dataset_name, hf_split, cache_dir)
    split_rows = split_records(records, split, train_size, valid_size, test_size, seed, shuffle)
    return S1KResponseDataset(split_rows, tokenizer, format_cfg)


def get_sft_dataloader(
    batch_size: int = 8,
    split: str = "train",
    tokenizer_name: str = DEFAULT_TOKENIZER,
    tokenizer: GPT2TokenizerFast | None = None,
    data_json: str | None = None,
    json_path: str | None = None,
    dataset_name: str = DEFAULT_DATASET,
    hf_split: str = "train",
    cache_dir: str | None = None,
    seed: int = 0,
    train_size: int = 800,
    valid_size: int = 100,
    test_size: int = 100,
    shuffle: bool | None = None,
    format_cfg: SFTFormatConfig | None = None,
    num_workers: int = 0,
    drop_last: bool | None = None,
) -> DataLoader:
    if data_json is None and json_path is not None:
        data_json = json_path
    tokenizer = tokenizer or GPT2TokenizerFast.from_pretrained(tokenizer_name)
    ds = build_sft_dataset(
        split=split,
        tokenizer=tokenizer,
        data_json=data_json,
        dataset_name=dataset_name,
        hf_split=hf_split,
        cache_dir=cache_dir,
        seed=seed,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
        shuffle=True,
        format_cfg=format_cfg,
    )
    if shuffle is None:
        shuffle = split == "train"
    if drop_last is None:
        drop_last = split == "train"
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_sft,
    )


if __name__ == "__main__":
    tokenizer = GPT2TokenizerFast.from_pretrained(DEFAULT_TOKENIZER)
    sample = {
        "question": "What is 2+2?",
        "solution": "We compute 2+2=4, so the answer is \\boxed{4}.",
    }
    item = format_sft_record(sample, tokenizer, SFTFormatConfig(max_length=128, max_answer_len=64))
    print("input_ids shape:", tuple(item["input_ids"].shape))
    print("prompt tokens:", int(item["prompt_mask"].sum()))
    print("answer tokens:", int(item["answer_mask"].sum()))
    print("nonpad tokens:", int(item["pad_mask"].sum()))
    print(tokenizer.decode(item["input_ids"][item["pad_mask"]]))
