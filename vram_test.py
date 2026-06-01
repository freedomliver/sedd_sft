import torch
from model import SEDD
from model.ema import ExponentialMovingAverage
import graph_lib, noise_lib, losses

device = torch.device('cuda')

model = SEDD.from_pretrained('pretrained/sedd-medium').to(device).to(torch.bfloat16)
cfg = model.config
graph = graph_lib.get_graph(cfg, device)
noise = noise_lib.get_noise(cfg).to(device)
loss_fn = losses.get_sft_loss_fn(noise, graph)

print(f'Model loaded. VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB')
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

for bs in [8, 4, 2, 1]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        x = torch.randint(0, 50257, (bs, 1024), device=device)
        clen = torch.full((bs,), 100, device=device, dtype=torch.long)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = loss_fn(model, x, clen).mean()
        loss.backward()
        
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f'batch_size={bs}: peak VRAM = {peak:.2f} GB  OK')
        model.zero_grad(set_to_none=True)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'batch_size={bs}: OOM')
            torch.cuda.empty_cache()
        else:
            raise
