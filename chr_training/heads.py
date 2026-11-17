"""Incrementally expanded classifier and symmetric cross entropy."""
import torch
from torch import nn
import torch.nn.functional as F

class ContinualLinear(nn.Module):
    """Exact native SLDC head (external/SLDC/utils/inc_net.py)."""

    def __init__(self, embed_dim, nb_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = nn.ModuleList([nn.Linear(embed_dim, nb_classes, bias=False)])
        self.head_weights = nn.Parameter(torch.ones(nb_classes))
        self.current_output_size = nb_classes

    def update(self, nb_classes, freeze_old=True):
        new_head = nn.Linear(self.embed_dim, nb_classes, bias=False)
        if freeze_old:
            for head in self.heads:
                for param in head.parameters():
                    param.requires_grad = False
        self.heads.append(new_head)
        new_head_weights = nn.Parameter(torch.ones(self.current_output_size + nb_classes))
        with torch.no_grad():
            new_head_weights[:self.current_output_size] = self.head_weights
            new_head_weights[self.current_output_size:] = 1.0
        self.head_weights = new_head_weights
        self.current_output_size += nb_classes

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        combined = torch.cat(outputs, dim=1)
        return combined * self.head_weights


def symmetric_cross_entropy(logits, targets, sce_a=0.5, sce_b=0.5):
    """Native SLDC SCE, including its clamped one-hot CE term."""
    pred = torch.clamp(F.softmax(logits, dim=1), min=1e-7, max=1.0)
    label_one_hot = F.one_hot(targets, pred.size(1)).float().to(pred.device)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    ce_loss = -torch.sum(label_one_hot * torch.log(pred), dim=1).mean()
    rce_loss = -torch.sum(pred * torch.log(label_one_hot), dim=1).mean()
    return sce_a * ce_loss + sce_b * rce_loss
