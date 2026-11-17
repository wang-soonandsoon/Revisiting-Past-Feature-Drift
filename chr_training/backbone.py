"""Offline construction of the original SLDC ViT and its sequential LoRA state.

The upstream convenience factory targets timm 0.5; loading its model class
directly also works with the installed timm 1.x, without changing upstream code.
"""

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn


@lru_cache(maxsize=1)
def _upstream():
    source = Path(__file__).resolve().parents[1] / "external/SLDC/convs/vits.py"
    spec = importlib.util.spec_from_file_location("_chr_sldc_vits", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_backbone(checkpoint, rank=4, device="cpu", initialize_lora=True,
                   only_lora=True, backbone_type="in21k"):
    """Load a local pre-trained ViT and return an SLDC feature backbone.

    ``model(images)['features']`` has shape [batch, 768]. Set ``rank=0`` for
    Frozen. LoRA uses upstream SVD initialization once; choose a CUDA device
    here for fast SVD. ``initialize_lora=False`` is intended for checkpoint
    reloads, where all LoRA A and B tensors will immediately be restored.
    The default follows upstream ``only_lora=True`` training. ``in21k`` reads
    the Google AugReg .npz; ``mocov3`` reads the official MoCo-v3 base encoder
    converted to {'model': state_dict} by its released convert_to_deit script.
    Both choices reset the final LayerNorm, as the SLDC factories do.
    """
    if backbone_type not in ("in21k", "mocov3"):
        raise ValueError(f"Unknown backbone_type: {backbone_type}")
    upstream = _upstream()
    model = upstream.VisionTransformer(
        patch_size=16, num_classes=21843, embed_dim=768, depth=12,
        num_heads=12, lora_rank=rank,
    )
    if backbone_type == "in21k":
        upstream._load_weights(model, str(checkpoint))
        del model.head
    else:
        del model.head
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        message = model.load_state_dict(state, strict=False)
        expected_missing = {name for name in model.state_dict() if "lora" in name}
        if set(message.missing_keys) != expected_missing or message.unexpected_keys:
            raise ValueError(f"MoCo-v3 encoder keys disagree: {message}")
    # These operations reproduce both upstream SLDC pre-training factories.
    model.norm = nn.LayerNorm(768)
    model.num_used_layers = 1
    model.to(device)
    if rank > 0 and initialize_lora:
        with torch.no_grad():
            for block in model.blocks:
                block.mlp.init_lora()
                block.attn.init_lora()
    for name, parameter in model.named_parameters():
        trainable = ("lora" in name if only_lora else
                     any(key in name for key in ("lora", "norm", "bias", "cls_token")))
        parameter.requires_grad_(rank > 0 and trainable)
    return model


def adapter_state(model):
    """Copy all mutable backbone parameters, including LoRA A initialization."""
    return {name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def save_adapter(model, path, metadata=None):
    """Save a small task snapshot; the immutable pre-trained weights are shared."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": adapter_state(model), "metadata": metadata or {}}, path)


def load_adapter(model, state_or_path):
    """Restore a snapshot into a backbone with matching rank/training policy."""
    payload = (torch.load(state_or_path, map_location="cpu", weights_only=True)
               if isinstance(state_or_path, (str, Path)) else state_or_path)
    state = payload.get("adapter", payload)
    expected = {name for name, parameter in model.named_parameters()
                if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError("Adapter keys differ from the configured trainable backbone parameters")
    model.load_state_dict(state, strict=False)
    return payload.get("metadata", {})
