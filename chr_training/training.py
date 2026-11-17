"""Current-task symmetric cross entropy and feature/norm distillation."""
from contextlib import nullcontext
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .heads import symmetric_cross_entropy


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def train_task(model, head, teacher, loader, known_classes, config, device, on_epoch=None):
    """Train the shared encoder and canonical head on current classes."""
    lr = config["learning_rate"]
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": lr},
        {"params": head.parameters(), "lr": config.get('head_learning_rate',lr * config["head_lr_scale"])},
    ], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=lr / 3)
    if 'head_learning_rate' in config:
        head_floor=config['head_min_learning_rate']/config['head_learning_rate']
        def cosine(floor):
            return lambda epoch:floor+(1-floor)*(1+math.cos(math.pi*epoch/config['epochs']))/2
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,[cosine(1/3),cosine(head_floor)])
    started = time.perf_counter()
    records = []
    use_bf16 = config.get("train_precision") == "bfloat16"
    use_kd = teacher is not None and config["gamma_kd"] > 0
    if teacher is not None:
        teacher.eval().requires_grad_(False)
    for epoch in range(config["epochs"]):
        model.train()
        head.train()
        totals = torch.zeros(5, device=device)
        epoch_started = time.perf_counter()
        for _, images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            context = (torch.autocast(torch.device(device).type, dtype=torch.bfloat16)
                       if use_bf16 else nullcontext())
            with context:
                features = model(images)["features"]
                logits = head(features)
                previous = None
                if use_kd:
                    with torch.no_grad():
                        previous = teacher(images)["features"]
            sce = symmetric_cross_entropy(logits[:, known_classes:].float(),
                                          labels - known_classes,
                                          config["sce_a"], config["sce_b"])
            kd, norm = features.new_zeros(()).float(), features.new_zeros(()).float()
            if use_kd:
                kd = F.mse_loss(features.float(), previous.float())
                norm = F.mse_loss(features.float().norm(dim=1), previous.float().norm(dim=1))
            loss = sce + config["gamma_kd"] * kd + config["gamma_norm"] * norm
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals[:4] += torch.stack((loss.detach(), sce.detach(), kd.detach(), norm.detach())) * len(labels)
            totals[4] += (logits.detach().argmax(1) == labels).sum()
        if not torch.isfinite(totals).all():
            raise ValueError("Nonfinite canonical training loss")
        metrics = (totals / len(loader.dataset)).tolist()
        record = dict(zip(("loss", "sce", "feature_kd", "norm_kd", "train_accuracy"), metrics))
        record.update(epoch=epoch + 1, seconds=time.perf_counter() - epoch_started,
                      lr=optimizer.param_groups[0]["lr"])
        if 'head_learning_rate' in config:record['head_lr']=optimizer.param_groups[1]['lr']
        records.append(record)
        if on_epoch is not None:
            on_epoch(record)
        scheduler.step()
    return {"epochs": records, "seconds": time.perf_counter() - started,
            "training_rows": len(loader.dataset)}
