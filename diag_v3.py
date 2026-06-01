import torch, os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from transformers import GPT2TokenizerFast
from model import SEDD
import graph_lib, noise_lib, losses
from sampling import get_conditional_sampler
from datasets import load_dataset
from data_sft import get_sft_dataloader

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
device = torch.device('cuda')

# Load SFT model
sft_model = SEDD.from_pretrained('pretrained/sedd-small')
sft_model.load_state_dict(torch.load('checkpoints_sft/sedd_sft_final.pt', map_location='cpu'))
sft_model = sft_model.to(device).eval()
cfg = sft_model.config
graph = graph_lib.get_graph(cfg, device)
noise = noise_lib.get_noise(cfg).to(device)

# Load pretrained model for comparison
pre_model = SEDD.from_pretrained('pretrained/sedd-small').to(device).eval()

# Loss comparison on 10 training samples
loss_fn = losses.get_sft_loss_fn(noise, graph)
loader = get_sft_dataloader(batch_size=4)
print('=== LOSS COMPARISON (SFT vs Pretrained) ===')
sft_losses, pre_losses = [], []
for i, batch in enumerate(loader):
    if i >= 3: break
    ids = batch['input_ids'].to(device)
    clen = batch['condition_len'].to(device)
    with torch.no_grad():
        sl = loss_fn(sft_model, ids, clen).mean().item()
        pl = loss_fn(pre_model, ids, clen).mean().item()
    sft_losses.append(sl)
    pre_losses.append(pl)
    print(f'  Batch {i}: pretrained={pl:.1f}, sft={sl:.1f}, delta={pl-sl:.1f}')
print(f'  Mean: pretrained={sum(pre_losses)/len(pre_losses):.1f}, sft={sum(sft_losses)/len(sft_losses):.1f}')

del pre_model
torch.cuda.empty_cache()

# Conditional generation on 5 training samples
ds = load_dataset('simplescaling/s1K-1.1')['train']
print()
print('=== CONDITIONAL GENERATION (5 training samples) ===')
for i in range(5):
    sample = ds[i]
    q_ids = torch.tensor(tokenizer.encode(sample['question'])[:200], dtype=torch.long).unsqueeze(0)
    sampler = get_conditional_sampler(graph, noise, q_ids, steps=128, device=device)
    gen = sampler(sft_model)
    gen_text = tokenizer.decode(gen[0, q_ids.shape[1]:], skip_special_tokens=True)
    gt_text = sample['deepseek_attempt']
    print(f'[{i}] Q: {sample["question"][:80]}...')
    print(f'    GT tail: ...{gt_text[-100:]}')
    print(f'    Generated: {gen_text[:150]}...')
    print()
