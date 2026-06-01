import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from data_sft import format_sample
from datasets import load_dataset
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
ds = load_dataset('simplescaling/s1K-1.1')['train']

for i in range(5):
    s = format_sample(ds[i])
    total_real = 1024 - s['input_ids'].count(50256) + 1  # approx non-padding
    cond = s['condition_len']
    ids = s['input_ids']
    print(f"Sample {i}: cond_len={cond}, total_tokens={len(ids)}, approx_real={total_real}")
    print(f"  Question: {tokenizer.decode(ids[:30])}...")
    ans_start = cond
    print(f"  Answer start: {tokenizer.decode(ids[ans_start:ans_start+30])}...")
    print()
