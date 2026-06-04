from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import dataclass
from typing import Iterable

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    DataLoader = None

    class Dataset:
        pass
try:
    from transformers import GPT2TokenizerFast
except ModuleNotFoundError:
    GPT2TokenizerFast = None


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


def _lookup_text(source: dict | None, key: str) -> str | None:
    if not isinstance(source, dict):
        return None
    value = source.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def get_answer_text(record: dict, answer_field: str = "solution") -> str:
    metadata = record.get("metadata") if isinstance(record, dict) else None

    if answer_field in {"boxed_inner", "final_boxed_inner", "boxed_answer_inner"}:
        boxed = extract_boxed(record.get("deepseek_attempt", "")) or extract_boxed(
            record.get("solution", "")
        )
        if boxed:
            return boxed
        for key in ("boxed_inner", "final_answer", "answer", "Correct Answer", "Pre-Revision Correct Answer"):
            answer = _lookup_text(record, key) or _lookup_text(metadata, key)
            if answer:
                boxed = extract_boxed(answer)
                return boxed if boxed is not None else answer

    if answer_field in {"boxed", "final_boxed", "boxed_answer"}:
        boxed = extract_boxed(record.get("deepseek_attempt", "")) or extract_boxed(
            record.get("solution", "")
        )
        if boxed:
            return f"\\boxed{{{boxed}}}"
        for key in ("boxed", "final_boxed", "boxed_answer", "final_answer", "answer", "Correct Answer", "Pre-Revision Correct Answer"):
            answer = _lookup_text(record, key) or _lookup_text(metadata, key)
            if answer:
                boxed = extract_boxed(answer)
                if boxed is not None:
                    return f"\\boxed{{{boxed}}}"
                return f"\\boxed{{{answer}}}"

    answer = _lookup_text(record, answer_field) or _lookup_text(metadata, answer_field)
    if answer:
        return answer
    for fallback in ("solution", "deepseek_attempt", "answer", "final_answer", "Correct Answer", "Pre-Revision Correct Answer"):
        answer = _lookup_text(record, fallback) or _lookup_text(metadata, fallback)
        if answer:
            return answer
    return ""


def _unwrap_singleton_sequence_text(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    if text[0] not in "[(":
        return text
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return str(value[0]).strip()
    return text


def _extract_answer_after_label(text: str) -> str | None:
    matches = list(
        re.finditer(
            r"(?im)(?:^|\n)\s*(?:final\s+)?answer\s*(?:is|=|:)?\s*",
            str(text or ""),
        )
    )
    if not matches:
        return None
    tail = text[matches[-1].end() :].strip()
    if not tail:
        return None
    boxed = extract_boxed(tail)
    if boxed is not None:
        return boxed
    return tail.splitlines()[0].strip()


def _looks_like_reasoning_text(text: str) -> bool:
    text = str(text or "").strip()
    if len(text) > 300:
        return True
    return bool(re.search(r"(?im)(^|\n)\s*(method|steps?)\s*:", text))


def _clean_scalar_answer(text: str) -> str:
    text = _unwrap_singleton_sequence_text(str(text or "").strip())
    text = text.replace(r"\$", "$").strip()
    boxed = extract_boxed(text)
    if boxed is not None:
        return _clean_scalar_answer(boxed)
    labeled = _extract_answer_after_label(text)
    if labeled is not None and labeled != text:
        return _clean_scalar_answer(labeled)
    text = re.sub(
        r"(?is)^\s*(?:final\s+)?answer\s*(?:is|=|:)?\s*",
        "",
        text,
    ).strip()
    for left, right in (("\\(", "\\)"), ("\\[", "\\]"), ("$", "$")):
        if text.startswith(left) and text.endswith(right):
            text = text[len(left) : len(text) - len(right)].strip()
    text = text.strip("'\" ")
    if text.endswith(".") and not re.search(r"\d\.\d\.$", text):
        text = text[:-1].strip()
    return text.strip()


def _clean_final_answer_candidate(text: str | None) -> str | None:
    if not text:
        return None
    text = _unwrap_singleton_sequence_text(str(text).strip())
    if not text:
        return None
    boxed = extract_boxed(text)
    if boxed is not None:
        return _clean_scalar_answer(boxed)
    labeled = _extract_answer_after_label(text)
    if labeled:
        return _clean_scalar_answer(labeled)
    if _looks_like_reasoning_text(text):
        return None
    answer = _clean_scalar_answer(text)
    return answer if answer else None


def iter_final_answer_candidates(
    record: dict,
    answer_field: str = "answer",
    source_answer: str | None = None,
) -> Iterable[tuple[str, str]]:
    metadata = record.get("metadata") if isinstance(record, dict) else None
    seen: set[tuple[str, str]] = set()

    def add(source: str, value: str | None):
        if not value:
            return
        key = (source, value)
        if key in seen:
            return
        seen.add(key)
        yield source, value

    requested_sources = [
        (answer_field, _lookup_text(record, answer_field)),
        (f"metadata.{answer_field}", _lookup_text(metadata, answer_field)),
    ]
    fallback_sources = [
        ("boxed_inner", _lookup_text(record, "boxed_inner")),
        ("metadata.boxed_inner", _lookup_text(metadata, "boxed_inner")),
        ("final_answer", _lookup_text(record, "final_answer")),
        ("metadata.final_answer", _lookup_text(metadata, "final_answer")),
        ("answer", _lookup_text(record, "answer")),
        ("metadata.answer", _lookup_text(metadata, "answer")),
        ("Correct Answer", _lookup_text(record, "Correct Answer")),
        ("metadata.Correct Answer", _lookup_text(metadata, "Correct Answer")),
        (
            "Pre-Revision Correct Answer",
            _lookup_text(record, "Pre-Revision Correct Answer"),
        ),
        (
            "metadata.Pre-Revision Correct Answer",
            _lookup_text(metadata, "Pre-Revision Correct Answer"),
        ),
        ("deepseek_attempt", _lookup_text(record, "deepseek_attempt")),
        ("solution", _lookup_text(record, "solution")),
    ]
    if source_answer:
        fallback_sources.append(("source_answer", source_answer))

    for source, value in requested_sources + fallback_sources:
        yield from add(source, value)


def get_final_answer_text(
    record: dict,
    answer_field: str = "answer",
    source_answer: str | None = None,
    return_source: bool = False,
):
    for source, value in iter_final_answer_candidates(record, answer_field, source_answer):
        answer = _clean_final_answer_candidate(value)
        if answer:
            return (answer, source) if return_source else answer
    return ("", "missing") if return_source else ""


def extract_boxed(text) -> str | None:
    spans = extract_boxed_spans(text)
    return spans[-1][0] if spans else None


def extract_boxed_spans(text) -> list[tuple[str, int, int]]:
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
            results.append((text[start : i - 1].strip(), idx, i))
        pos = i
    return results


def strip_last_boxed(text) -> str:
    text = str(text or "")
    spans = extract_boxed_spans(text)
    if not spans:
        return text.strip()

    _, start, end = spans[-1]
    before = text[:start].rstrip()
    after = text[end:].strip()
    if after and not all(ch in ".。,，;；:：!?！？ \n\t" for ch in after):
        cleaned = f"{before} {after}".strip()
    else:
        cleaned = before
    cleaned = re.sub(
        r"(?i)(?:so|therefore|hence)?\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)?\s*$",
        "",
        cleaned,
    )
    return cleaned.rstrip(" .,:;，。；：").strip()


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
    fixed_layout: bool = False
    question_block_len: int = 231
    reasoning_block_len: int = 530
    final_answer_block_len: int = 263
    fixed_layout_reasoning_start: int | None = None
    fixed_layout_final_answer_start: int | None = None
    reasoning_field: str = "solution"
    final_answer_field: str = "answer"
    fixed_layout_supervise_pad: bool = False


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


def _fit_block(
    tokenizer: GPT2TokenizerFast,
    text: str,
    block_len: int,
    eos_token_id: int,
    add_eos: bool = False,
    truncate: str = "head",
) -> tuple[list[int], list[int], int]:
    if block_len <= 0:
        return [], [], 0
    ids = tokenizer.encode(str(text or ""), add_special_tokens=False)
    if add_eos:
        ids = ids + [eos_token_id]
    ids = _truncate_answer(ids, block_len, eos_token_id, truncate)
    true_len = len(ids)
    padded = ids + [eos_token_id] * (block_len - true_len)
    real_mask = [1] * true_len + [0] * (block_len - true_len)
    return padded, real_mask, true_len


def format_fixed_sft_record(
    record: dict,
    tokenizer: GPT2TokenizerFast,
    cfg: SFTFormatConfig,
) -> dict:
    reasoning_start = (
        cfg.fixed_layout_reasoning_start
        if cfg.fixed_layout_reasoning_start is not None
        else cfg.question_block_len
    )
    final_answer_start = (
        cfg.fixed_layout_final_answer_start
        if cfg.fixed_layout_final_answer_start is not None
        else reasoning_start + cfg.reasoning_block_len
    )
    if cfg.question_block_len < 0 or cfg.reasoning_block_len < 0 or cfg.final_answer_block_len < 0:
        raise ValueError("fixed layout block lengths must be non-negative")
    if reasoning_start < cfg.question_block_len:
        raise ValueError("fixed layout reasoning_start overlaps question block")
    if reasoning_start + cfg.reasoning_block_len > cfg.max_length:
        raise ValueError("fixed layout reasoning block exceeds max_length")
    if final_answer_start < reasoning_start + cfg.reasoning_block_len:
        raise ValueError("fixed layout final_answer_start overlaps reasoning block")
    if final_answer_start + cfg.final_answer_block_len > cfg.max_length:
        raise ValueError("fixed layout final answer block exceeds max_length")
    if (
        cfg.fixed_layout_reasoning_start is None
        and cfg.fixed_layout_final_answer_start is None
        and cfg.question_block_len + cfg.reasoning_block_len + cfg.final_answer_block_len != cfg.max_length
    ):
        raise ValueError(
            "contiguous fixed layout block lengths must sum to max_length: "
            f"{cfg.question_block_len}+{cfg.reasoning_block_len}+"
            f"{cfg.final_answer_block_len}!={cfg.max_length}"
        )

    eos_token_id = tokenizer.eos_token_id
    question = str(record.get("question", "")).strip()
    source_answer = get_answer_text(record, cfg.reasoning_field)
    final_answer = get_final_answer_text(
        record,
        cfg.final_answer_field,
        source_answer=source_answer,
    )
    if not final_answer:
        final_answer = extract_boxed(source_answer) or ""
    reasoning_text = strip_last_boxed(source_answer)

    question_ids, question_real_mask, question_len = _fit_block(
        tokenizer,
        f"Question:\n{question}",
        cfg.question_block_len,
        eos_token_id,
        add_eos=False,
        truncate="head",
    )
    reasoning_ids, reasoning_real_mask, reasoning_len = _fit_block(
        tokenizer,
        reasoning_text,
        cfg.reasoning_block_len,
        eos_token_id,
        add_eos=False,
        truncate=cfg.answer_truncate,
    )
    final_answer_ids, final_answer_real_mask, final_answer_len = _fit_block(
        tokenizer,
        final_answer,
        cfg.final_answer_block_len,
        eos_token_id,
        add_eos=True,
        truncate="head",
    )

    input_ids = [eos_token_id] * cfg.max_length
    question_block_mask = [0] * cfg.max_length
    generation_block_mask = [0] * cfg.max_length
    real_generation_mask = [0] * cfg.max_length
    real_token_mask = [0] * cfg.max_length

    input_ids[: cfg.question_block_len] = question_ids
    question_block_mask[: cfg.question_block_len] = [1] * cfg.question_block_len
    real_token_mask[: cfg.question_block_len] = question_real_mask

    reasoning_end = reasoning_start + cfg.reasoning_block_len
    input_ids[reasoning_start:reasoning_end] = reasoning_ids
    generation_block_mask[reasoning_start:reasoning_end] = [1] * cfg.reasoning_block_len
    real_generation_mask[reasoning_start:reasoning_end] = reasoning_real_mask
    real_token_mask[reasoning_start:reasoning_end] = reasoning_real_mask

    final_answer_end = final_answer_start + cfg.final_answer_block_len
    input_ids[final_answer_start:final_answer_end] = final_answer_ids
    generation_block_mask[final_answer_start:final_answer_end] = [1] * cfg.final_answer_block_len
    real_generation_mask[final_answer_start:final_answer_end] = final_answer_real_mask
    real_token_mask[final_answer_start:final_answer_end] = final_answer_real_mask

    if cfg.fixed_layout_supervise_pad:
        answer_mask = generation_block_mask
        pad_mask = [1] * cfg.max_length
    else:
        answer_mask = real_generation_mask
        pad_mask = real_token_mask

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "prompt_mask": torch.tensor(question_block_mask, dtype=torch.bool),
        "answer_mask": torch.tensor(answer_mask, dtype=torch.bool),
        "pad_mask": torch.tensor(pad_mask, dtype=torch.bool),
        "prompt_len": torch.tensor(cfg.question_block_len, dtype=torch.long),
        "answer_len": torch.tensor(reasoning_len + final_answer_len, dtype=torch.long),
        "answer_window_len": torch.tensor(
            cfg.reasoning_block_len + cfg.final_answer_block_len,
            dtype=torch.long,
        ),
        "reasoning_len": torch.tensor(reasoning_len, dtype=torch.long),
        "final_answer_len": torch.tensor(final_answer_len, dtype=torch.long),
        "question_block_len": torch.tensor(cfg.question_block_len, dtype=torch.long),
        "reasoning_start": torch.tensor(reasoning_start, dtype=torch.long),
        "reasoning_block_len": torch.tensor(cfg.reasoning_block_len, dtype=torch.long),
        "final_answer_block_len": torch.tensor(cfg.final_answer_block_len, dtype=torch.long),
        "final_answer_start": torch.tensor(final_answer_start, dtype=torch.long),
        "question": question,
        "answer": reasoning_text,
        "final_answer": final_answer,
        "prompt_text": f"Question:\n{question}",
    }


def format_sft_record(
    record: dict,
    tokenizer: GPT2TokenizerFast,
    cfg: SFTFormatConfig,
) -> dict:
    if cfg.fixed_layout:
        return format_fixed_sft_record(record, tokenizer, cfg)

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
        "answer_window_len": torch.tensor(cfg.max_answer_len, dtype=torch.long),
        "reasoning_len": torch.tensor(0, dtype=torch.long),
        "final_answer_len": torch.tensor(len(answer_ids), dtype=torch.long),
        "question_block_len": torch.tensor(len(prompt_ids), dtype=torch.long),
        "reasoning_block_len": torch.tensor(0, dtype=torch.long),
        "final_answer_block_len": torch.tensor(cfg.max_answer_len, dtype=torch.long),
        "final_answer_start": torch.tensor(len(prompt_ids), dtype=torch.long),
        "question": question,
        "answer": answer_text,
        "final_answer": answer_text,
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
    tensor_keys = (
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
    out = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    out["condition_len"] = out["prompt_len"]
    out["boxed_mask"] = out["answer_mask"].long()
    out["question"] = [item["question"] for item in batch]
    out["answer"] = [item["answer"] for item in batch]
    out["final_answer"] = [item["final_answer"] for item in batch]
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
