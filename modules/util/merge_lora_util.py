"""Bake canonical LoRA weights into an unquantized base model for export."""

from modules.module.FusedModule import discover_fused_groups
from modules.util.enum.ModelFormat import ModelFormat

import torch
from torch import nn

from tqdm import tqdm


@torch.no_grad()
def merge_lora(model, on_progress=lambda _done, _total: None):
    state = model.lora_state_dict
    if not state:
        raise ValueError("The input contains no LoRA weights")
    component = model.model_type.denoising_model_part()
    denoiser = getattr(model, "prior_prior" if component == "prior" else component)
    parts = [(component, denoiser)] + [
        (names[ModelFormat.DIFFUSERS_LORA], module) for module, names in model.lora_text_encoders()
    ]
    targets = {}
    for prefix, part in parts:
        if part is None:
            continue
        modules = {name: module for name, module in part.named_modules() if isinstance(module, (nn.Linear, nn.Conv2d))}
        targets.update({f"{prefix}.{name}": [module] for name, module in modules.items()})
        if prefix == component:
            for name, _names, leaves in discover_fused_groups(model.fusion_groups(), modules, False):
                targets[f"{prefix}.{name}"] = leaves

    groups = {}
    suffixes = (".lora_down.weight", ".lora_up.weight", ".alpha", ".dora_scale")
    for key, value in state.items():
        suffix = next((suffix for suffix in suffixes if key.endswith(suffix)), None)
        if suffix is None:
            raise ValueError(f"Cannot merge adapter parameter {key}. Merge supports LoRA and DoRA weights.")
        groups.setdefault(key.removesuffix(suffix), {})[suffix] = value
    # Validate every target before changing the base or opening the output path.
    used_targets = set()
    for name, weights in groups.items():
        if name not in targets:
            raise ValueError(f"LoRA target does not exist in the base model: {name}")
        if ".lora_down.weight" not in weights or ".lora_up.weight" not in weights:
            raise ValueError(f"Incomplete LoRA weight pair: {name}")
        down, up = weights[".lora_down.weight"], weights[".lora_up.weight"]
        leaves = targets[name]
        if any(id(leaf) in used_targets for leaf in leaves):
            raise ValueError(f"Adapter contains overlapping fused and split targets: {name}")
        used_targets.update(id(leaf) for leaf in leaves)
        expected = (sum(m.weight.shape[0] for m in leaves), *leaves[0].weight.shape[1:])
        if (down.ndim != len(expected) or up.ndim != len(expected) or down.shape[0] <= 0
                or up.shape[1] != down.shape[0] or any(size != 1 for size in up.shape[2:])
                or (up.shape[0], *down.shape[1:]) != expected):
            raise ValueError(f"LoRA shape does not match the base model: {name}")
        if ".alpha" in weights and weights[".alpha"].numel() != 1:
            raise ValueError(f"LoRA alpha must be a scalar: {name}")
        if ".dora_scale" in weights:
            shape = weights[".dora_scale"].shape
            supported = ((expected[0], *([1] * (len(expected) - 1))), (1, expected[1], *([1] * (len(expected) - 2))))
            if shape not in supported:
                raise ValueError(f"Unsupported DoRA magnitude shape: {name}")
        if any(not torch.isfinite(value).all() for value in weights.values()):
            raise ValueError(f"Non-finite LoRA weights: {name}")

    on_progress(0, len(groups))
    for done, (name, weights) in enumerate(tqdm(groups.items(), desc="merging LoRA", unit="layer"), 1):
        leaves = targets[name]
        device = leaves[0].weight.device
        down = weights[".lora_down.weight"].to(device=device, dtype=torch.float32)
        up = weights[".lora_up.weight"].to(device=device, dtype=torch.float32)
        rank = down.shape[0]
        alpha = float(weights.get(".alpha", rank))
        base = torch.cat([leaf.weight.detach().float() for leaf in leaves], dim=0)
        merged = base + (up.flatten(1) @ down.flatten(1)).reshape(base.shape) * (alpha / rank)
        if ".dora_scale" in weights:
            scale = weights[".dora_scale"].to(device=device, dtype=torch.float32)
            axes = tuple(i for i, size in enumerate(scale.shape) if size == 1)
            norm = merged.square().sum(dim=axes, keepdim=True).sqrt().clamp_min(torch.finfo(torch.float32).eps)
            merged = merged * scale / norm
        if not torch.isfinite(merged).all():
            raise ValueError(f"Merged weights are non-finite: {name}")
        offset = 0
        for leaf in leaves:
            size = leaf.weight.shape[0]
            leaf.weight.copy_(merged[offset:offset + size].to(dtype=leaf.weight.dtype))
            offset += size
        on_progress(done, len(groups))
    model.lora_state_dict = None
