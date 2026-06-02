from transformers import GPT2TokenizerFast
import torch

from data_sft import SFTFormatConfig, format_sft_record, resolve_split_sizes


def main():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    cfg = SFTFormatConfig(max_length=128, max_answer_len=48)
    record = {
        "question": "Compute 2+2.",
        "solution": "We add the two numbers and get \\boxed{4}.",
    }
    item = format_sft_record(record, tokenizer, cfg)

    prompt_mask = item["prompt_mask"]
    answer_mask = item["answer_mask"]
    pad_mask = item["pad_mask"]

    assert prompt_mask.dtype == torch.bool
    assert answer_mask.dtype == torch.bool
    assert pad_mask.dtype == torch.bool
    assert not torch_overlap(prompt_mask, answer_mask)
    assert int(answer_mask.sum()) > 0
    assert int(pad_mask.sum()) == int(prompt_mask.sum() + answer_mask.sum())
    assert resolve_split_sizes(1000) == (800, 100, 100)
    assert resolve_split_sizes(599) == (399, 100, 100)

    boxed_item = format_sft_record(
        {"question": "Compute 2+2.", "deepseek_attempt": "Thus \\boxed{4}."},
        tokenizer,
        SFTFormatConfig(
            max_length=128,
            max_answer_len=32,
            answer_field="final_boxed",
            answer_prefix="Final Answer:",
        ),
    )
    boxed_text = tokenizer.decode(boxed_item["input_ids"][boxed_item["answer_mask"]])
    assert "\\boxed{4}" in boxed_text
    assert "Final Answer:" in boxed_item["prompt_text"]

    print("prompt_len:", int(item["prompt_len"]))
    print("answer_len:", int(item["answer_len"]))
    print("nonpad_len:", int(pad_mask.sum()))
    print(tokenizer.decode(item["input_ids"][pad_mask]))


def torch_overlap(a, b):
    return bool((a & b).any().item())


if __name__ == "__main__":
    main()
