import json, torch
from transformers import GPT2TokenizerFast
from model import SEDD
import graph_lib, noise_lib
from sampling import get_conditional_sampler

device = torch.device('cuda')
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

print("Loading model...", flush=True)
model = SEDD.from_pretrained('pretrained/sedd-small')
cfg = model.config
ckpt = torch.load('/root/autodl-tmp/sedd_checkpoints/sedd_sft_v7_final.pt', map_location='cpu')
model.load_state_dict(ckpt)
model = model.to(device).eval()

graph = graph_lib.get_graph(cfg, device)
noise = noise_lib.get_noise(cfg).to(device)

with open('data/s1K_train_599.json') as f:
    data = json.load(f)

print(f"Generating 3 samples to inspect output...\n", flush=True)
for i in range(3):
    sample = data[i]
    q_ids = torch.tensor(tokenizer.encode(sample['question'])[:200], dtype=torch.long).unsqueeze(0)
    sampler = get_conditional_sampler(graph, noise, q_ids, steps=128, device=device)
    with torch.no_grad():
        gen = sampler(model)
    gen_text = tokenizer.decode(gen[0, q_ids.shape[1]:])
    print(f'=== Sample {i} ===', flush=True)
    print(f'Question: {sample["question"][:100]}...', flush=True)
    print(f'Generated ({len(gen_text)} chars):', flush=True)
    print(gen_text[:500], flush=True)
    print(flush=True)
