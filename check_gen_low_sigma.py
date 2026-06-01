"""
用匹配训练的 sigma_max=0.05 (t_max~0.0488) 做推理，看效果。
同时对比用完整 t=[1, eps] 范围推理的结果。
"""
import json, torch, math
from transformers import GPT2TokenizerFast
from model import SEDD
import graph_lib, noise_lib
from catsample import sample_categorical

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

from model import utils as mutils


def conditional_sampler_with_tmax(graph, noise, question_ids, steps=128, eps=1e-4, t_start=1.0, device="cuda"):
    """支持自定义 t_start 的条件采样器"""

    @torch.no_grad()
    def sampler(model):
        score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        B, q_len = question_ids.shape
        ans_len = 1024 - q_len

        # 初始化答案部分
        if t_start >= 0.99:
            # 完整噪声：从 absorb 极限分布采样
            x_ans = graph.sample_limit(B, ans_len).to(device)
        else:
            # 低噪声起点：用随机 token + 部分 absorb
            sigma_start = noise(torch.tensor([t_start], device=device))[0]
            random_tokens = torch.randint(0, 50257, (B, ans_len), device=device)
            x_ans = graph.sample_transition(random_tokens, sigma_start.expand(ans_len))

        x = torch.cat([question_ids.to(device), x_ans], dim=1)

        timesteps = torch.linspace(t_start, eps, steps + 1, device=device)
        dt = (t_start - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(B, 1, device=device)
            sigma, dsigma = noise(t)
            score = score_fn(x, sigma.squeeze(-1))

            stag_score = graph.staggered_score(score, dsigma.squeeze(-1) * dt)
            probs = stag_score * graph.transp_transition(x, dsigma.squeeze(-1) * dt)

            x_new = sample_categorical(probs)
            x = torch.cat([question_ids.to(device), x_new[:, q_len:]], dim=1)

        # denoise step
        t = timesteps[-1] * torch.ones(B, 1, device=device)
        sigma = noise(t)[0]
        score = score_fn(x, sigma.squeeze(-1))
        stag_score = graph.staggered_score(score, sigma.squeeze(-1))
        probs = stag_score * graph.transp_transition(x, sigma.squeeze(-1))
        if graph.absorb:
            probs = probs[..., :-1]
        x_ans_final = sample_categorical(probs)[:, q_len:]
        x = torch.cat([question_ids.to(device), x_ans_final], dim=1)
        return x

    return sampler


def extract_boxed(text):
    results = []
    pos = 0
    text = text or ""
    while True:
        idx = text.find(r"\boxed{", pos)
        if idx == -1:
            break
        start = idx + 7
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


with open('data/s1K_train_599.json') as f:
    data = json.load(f)

sigma_max = 0.05
eps_noise = 1e-3
t_max = (1 - math.exp(-sigma_max)) / (1 - eps_noise)

print(f"\n{'='*60}")
print(f"Test 1: t_start = {t_max:.4f} (matching training sigma_max={sigma_max})")
print(f"{'='*60}\n")

for i in range(5):
    sample = data[i]
    q_ids = torch.tensor(tokenizer.encode(sample['question'])[:200], dtype=torch.long).unsqueeze(0)
    sampler = conditional_sampler_with_tmax(graph, noise, q_ids, steps=128, t_start=t_max, device=device)
    gen = sampler(model)
    gen_text = tokenizer.decode(gen[0, q_ids.shape[1]:])
    boxed = extract_boxed(gen_text)
    print(f'[{i}] boxed={boxed}', flush=True)
    print(f'    text: {gen_text[:300]}', flush=True)
    print(flush=True)

print(f"\n{'='*60}")
print(f"Test 2: t_start = 1.0 (full range, standard inference)")
print(f"{'='*60}\n")

for i in range(5):
    sample = data[i]
    q_ids = torch.tensor(tokenizer.encode(sample['question'])[:200], dtype=torch.long).unsqueeze(0)
    sampler = conditional_sampler_with_tmax(graph, noise, q_ids, steps=128, t_start=1.0, device=device)
    gen = sampler(model)
    gen_text = tokenizer.decode(gen[0, q_ids.shape[1]:])
    boxed = extract_boxed(gen_text)
    print(f'[{i}] boxed={boxed}', flush=True)
    print(f'    text: {gen_text[:300]}', flush=True)
    print(flush=True)
