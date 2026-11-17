"""Equivalent SDPA attention backend for the unchanged upstream ViT weights."""

from types import MethodType
import torch.nn.functional as F


def _attention(self, x):
    batch, tokens, width = x.shape
    weight = self.qkv.weight
    if hasattr(self, "lora_A"):
        weight = weight + (self.lora_B @ self.lora_A) * self.lora_scale
    qkv = F.linear(x, weight, self.qkv.bias)
    qkv = qkv.reshape(batch, tokens, 3, self.num_heads, width // self.num_heads)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    x = F.scaled_dot_product_attention(
        q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0,
        scale=self.scale)
    x = x.transpose(1, 2).reshape(batch, tokens, width)
    return self.proj_drop(self.proj(x))


def enable_sdpa(model):
    for block in model.blocks:
        block.attn.forward = MethodType(_attention, block.attn)
    return model
