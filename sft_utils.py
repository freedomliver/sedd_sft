import json
import os
import re

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from lora import load_lora_checkpoint
from model import SEDD


def resolve_dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def load_sedd_base(pretrained: str, device: torch.device, dtype=torch.float32):
    local_config = os.path.join(pretrained, "config.json")
    if os.path.exists(local_config):
        with open(local_config) as f:
            cfg = OmegaConf.create(json.load(f))
        model = SEDD(cfg)
        safetensors_path = os.path.join(pretrained, "model.safetensors")
        bin_path = os.path.join(pretrained, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            state = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            state = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model weights found under {pretrained}")
        model.load_state_dict(state, strict=False)
    else:
        model = SEDD.from_pretrained(pretrained)
        cfg = model.config

    model = model.to(device)
    if dtype != torch.float32:
        model = model.to(dtype)
    return model, cfg


def load_sedd_for_inference(pretrained: str, lora_ckpt: str | None, device, dtype):
    model, cfg = load_sedd_base(pretrained, device, dtype)
    lora_info = None
    if lora_ckpt:
        raw = torch.load(lora_ckpt, map_location="cpu")
        if raw.get("checkpoint_type") == "full" or "model" in raw:
            missing, unexpected = model.load_state_dict(raw["model"], strict=False)
            lora_info = {
                "checkpoint": raw,
                "checkpoint_type": "full",
                "missing": missing,
                "unexpected": unexpected,
            }
        else:
            lora_info = load_lora_checkpoint(model, lora_ckpt, map_location="cpu")
    return model.to(device).eval(), cfg, lora_info


def decode_until_eos(tokenizer, token_ids):
    ids = token_ids.detach().cpu().tolist()
    if tokenizer.eos_token_id in ids:
        ids = ids[: ids.index(tokenizer.eos_token_id)]
    return tokenizer.decode(ids, skip_special_tokens=True)


def extract_boxed(text):
    results = []
    pos = 0
    text = text or ""
    while pos < len(text):
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


def normalize_answer(text):
    if text is None:
        return None
    text = text.strip().replace(" ", "").replace(",", "")
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
    text = text.replace("\\", "").lower()
    try:
        value = float(text)
        return str(int(value)) if value == int(value) else str(value)
    except Exception:
        return text
