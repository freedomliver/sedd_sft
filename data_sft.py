import json
import torch
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
SEP = tokenizer.eos_token_id
BOXED_PREFIX = r"\boxed{"


def _find_boxed_char_positions(text):
    positions = set()
    search_start = 0
    while True:
        idx = text.find(BOXED_PREFIX, search_start)
        if idx == -1:
            break
        depth = 0
        end = idx
        for j in range(idx, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        positions.update(range(idx, end + 1))
        search_start = end + 1
    return positions


def format_sample(record):
    q = tokenizer.encode(record["question"])
    a_text = record["deepseek_attempt"]

    a_enc = tokenizer(a_text, return_offsets_mapping=True, add_special_tokens=False)
    a = a_enc["input_ids"]
    a_offsets = a_enc["offset_mapping"]

    max_a = 1024 - len(q) - 1
    if max_a < 50:
        q = q[:973]
        max_a = 50

    a_tail = a[-max_a:]
    a_offsets_tail = a_offsets[-max_a:]

    full = q + [SEP] + a_tail
    q_len = len(q) + 1
    pad_len = 1024 - len(full)
    full = full + [SEP] * pad_len

    boxed_char_pos = _find_boxed_char_positions(a_text)
    boxed_mask = [0] * 1024
    if boxed_char_pos:
        for tok_idx, (start, end) in enumerate(a_offsets_tail):
            if start < end and not boxed_char_pos.isdisjoint(range(start, end)):
                boxed_mask[q_len + tok_idx] = 1

    return {
        "input_ids": torch.tensor(full, dtype=torch.long),
        "condition_len": torch.tensor(q_len, dtype=torch.long),
        "boxed_mask": torch.tensor(boxed_mask, dtype=torch.long),
    }


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, json_path="data/s1K_correct.json"):
        with open(json_path, encoding="utf-8") as f:
            self.records = json.load(f)
        print(f"Loaded {len(self.records)} correct samples from {json_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return format_sample(self.records[idx])


def get_sft_dataloader(batch_size=8, json_path="data/s1K_correct.json"):
    ds = SFTDataset(json_path)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == "__main__":
    loader = get_sft_dataloader(batch_size=2)
    batch = next(iter(loader))
    print("input_ids shape:", batch["input_ids"].shape)
    print("condition_len:", batch["condition_len"])
    print("boxed_mask nonzero pos:", batch["boxed_mask"][0].nonzero().squeeze()[:20])
    text = tokenizer.decode(batch["input_ids"][0, :50])
    print("Decoded start:", text[:200])
