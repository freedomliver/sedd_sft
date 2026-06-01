import torch
import os
import sys
import time
import json
from omegaconf import OmegaConf
from safetensors.torch import load_file

from model import SEDD
from model.ema import ExponentialMovingAverage
import graph_lib
import noise_lib
import losses
from data_sft import get_sft_dataloader

CKPT_DIR = "/root/autodl-tmp/sedd_checkpoints"


def fast_load_sedd(pretrained_path, device):
    t0 = time.time()
    with open(os.path.join(pretrained_path, "config.json")) as f:
        cfg = OmegaConf.create(json.load(f))
    model = SEDD(cfg)
    weights_path = os.path.join(pretrained_path, "model.safetensors")
    if os.path.exists(weights_path):
        state_dict = load_file(weights_path)
    else:
        weights_path = os.path.join(pretrained_path, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)
    return model, cfg


def get_sigma_max(step, n_iters):
    """Sigma annealing: linear 0.01 -> 5.0 over first 90% of training."""
    anneal_end = int(n_iters * 0.9)
    if step >= anneal_end:
        return 5.0
    return 0.01 + (5.0 - 0.01) * step / anneal_end


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_path = "pretrained/sedd-small"

    n_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200000

    print(f"Loading pretrained model from {pretrained_path}...", flush=True)
    model, cfg = fast_load_sedd(pretrained_path, device)
    model = model.to(torch.bfloat16)

    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)

    lr = 3e-5
    batch_size = 24
    grad_accum = 11
    warmup = 500
    log_freq = 100
    save_freq = 20000
    grad_clip = 1000.0

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    loss_fn = losses.get_sft_loss_fn(noise, graph)
    dataloader = get_sft_dataloader(batch_size=batch_size, json_path="data/s1K_train_599.json")

    eff_batch = batch_size * grad_accum
    print(f"Training v9 (boxed-only loss, sigma annealing 0.01->5.0, clip={grad_clip})", flush=True)
    print(f"  {batch_size}x{grad_accum}={eff_batch} eff batch | lr={lr} | warmup={warmup} | n_iters={n_iters}", flush=True)

    step = 0
    cur_lr = 0.0
    optimizer.zero_grad()
    grad_norm_accum = 0.0
    accum_count = 0

    for epoch in range(100000):
        for batch in dataloader:
            if step >= n_iters:
                break

            input_ids = batch["input_ids"].to(device)
            condition_len = batch["condition_len"].to(device)
            boxed_mask = batch["boxed_mask"].to(device, dtype=torch.bfloat16)

            cur_sigma_max = get_sigma_max(step, n_iters)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model, input_ids, condition_len, boxed_mask=boxed_mask, sigma_max=cur_sigma_max).mean() / grad_accum

            loss.backward()

            is_update_step = (step + 1) % grad_accum == 0
            if is_update_step:
                raw_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
                grad_norm_accum += raw_grad_norm
                accum_count += 1

                cur_lr = lr * min((step + 1) / warmup, 1.0)
                for g in optimizer.param_groups:
                    g["lr"] = cur_lr
                optimizer.step()
                ema.update(model.parameters())
                optimizer.zero_grad()

            if step % log_freq == 0:
                avg_grad_norm = grad_norm_accum / max(accum_count, 1)
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                print(
                    f"step={step:6d} | loss={loss.item() * grad_accum:8.4f} | lr={cur_lr:.2e} | "
                    f"grad_norm={avg_grad_norm:7.1f} | sigma_max={cur_sigma_max:.4f} | mem={mem_gb:.1f}GB",
                    flush=True,
                )
                grad_norm_accum = 0.0
                accum_count = 0

            if step % save_freq == 0 and step > 0:
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, os.path.join(CKPT_DIR, f"sft_v9_ckpt_{step}.pt"))
                print(f"Saved checkpoint at step {step}", flush=True)

            step += 1

        if step >= n_iters:
            break

    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "sedd_sft_v9_final.pt"))
    print(f"Training complete ({step} steps). Saved sedd_sft_v9_final.pt", flush=True)


if __name__ == "__main__":
    main()
