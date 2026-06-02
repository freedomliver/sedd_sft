from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_suffixes: tuple[str, ...] = ("attn_qkv", "attn_out", "mlp.0", "mlp.2")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_a = nn.Linear(base.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.lora_a.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b.to(device=base.weight.device, dtype=base.weight.dtype)

        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.lora_b(self.dropout(self.lora_a(x))) * self.scaling


def _get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _matches_target(module_name: str, target_suffixes: tuple[str, ...]) -> bool:
    matched = any(module_name.endswith(suffix) for suffix in target_suffixes)
    if not matched:
        return False
    if module_name.endswith(("mlp.0", "mlp.2")) and not module_name.startswith("blocks."):
        return module_name in target_suffixes
    return True


def freeze_non_lora(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def apply_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    freeze_non_lora(model)
    replaced = []

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _matches_target(module_name, config.target_suffixes):
            continue

        parent, child_name = _get_parent_module(model, module_name)
        setattr(parent, child_name, LoRALinear(module, config.r, config.alpha, config.dropout))
        replaced.append(module_name)

    if not replaced:
        targets = ", ".join(config.target_suffixes)
        raise ValueError(f"No Linear modules matched LoRA targets: {targets}")

    return replaced


def trainable_parameter_summary(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "trainable": trainable,
        "total": total,
        "ratio": trainable / max(total, 1),
    }


def lora_state_dict(model: nn.Module) -> dict:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if ".lora_a." in key or ".lora_b." in key
    }


def lora_config_to_dict(config: LoRAConfig) -> dict:
    out = asdict(config)
    out["target_suffixes"] = list(config.target_suffixes)
    return out


def lora_config_from_dict(raw: dict) -> LoRAConfig:
    return LoRAConfig(
        r=int(raw["r"]),
        alpha=int(raw["alpha"]),
        dropout=float(raw["dropout"]),
        target_suffixes=tuple(raw["target_suffixes"]),
    )


def save_lora_checkpoint(path: str, model: nn.Module, config: LoRAConfig, **extra):
    payload = {
        "lora": lora_state_dict(model),
        "lora_config": lora_config_to_dict(config),
        **extra,
    }
    torch.save(payload, path)


def load_lora_checkpoint(model: nn.Module, checkpoint_path: str, map_location="cpu") -> dict:
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    config = lora_config_from_dict(ckpt["lora_config"])
    apply_lora(model, config)
    missing, unexpected = model.load_state_dict(ckpt["lora"], strict=False)
    return {
        "checkpoint": ckpt,
        "lora_config": config,
        "missing": missing,
        "unexpected": unexpected,
    }
