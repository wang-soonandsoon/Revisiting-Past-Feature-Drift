"""Shared augmented training images through saved encoder endpoints."""
from copy import deepcopy
import random
from time import perf_counter
import numpy as np
import torch
from .acceleration import enable_sdpa
from .backbone import build_backbone, load_adapter
from .data import make_loader
from .training import capture_rng, restore_rng, train_task

def build_model(config, device, initialize=True):
    model = build_backbone(
        config['checkpoint'], rank=config['rank'], device=device,
        initialize_lora=initialize, only_lora=config.get('only_lora', True),
        backbone_type=config.get('backbone_type', 'in21k'))
    if config.get('attention_backend') == 'sdpa':
        model = enable_sdpa(model)
    return model


def train_current(model, head, teacher, records, known, config, seed, task, device):
    """Source current-class SCE + feature/norm KD with an explicit loader RNG.

    Reuses the independently checked native-loss helper. Only current classes
    are in ``records``; full logits supply training accuracy, while the loss
    uses ``logits[:, known:]``. A fresh optimizer/scheduler is made each task.
    """
    loader = make_loader(records, config['batch_size'], config['workers'],
                         shuffle=True, seed=seed * 100 + task)
    return train_task(model, head, teacher, loader, known, config, device)


@torch.no_grad()
def extract_train_snapshots(model, teacher, records, adapters, config, seed, task, device):
    """One current TRAIN view per row shared by endpoints 0..t.

    ``adapters`` contains completed historical endpoints only. A scratch copy
    receives those adapters; the live backbone and its training flag survive
    unchanged. Extraction's dedicated random stream also leaves canonical
    training's Python, NumPy, Torch and device RNG states unchanged. ``teacher``
    is accepted for the shared training API; historical adapters are the single
    endpoint source so the same routine covers every retained endpoint.
    """
    if sorted(adapters) != list(range(task)):
        raise ValueError('Historical endpoints must be exactly 0..task-1')
    started = perf_counter()
    rng = capture_rng(device)
    was_training = model.training
    loader_seed = seed * 10000 + task
    torch.random.default_generator.manual_seed(loader_seed)
    np.random.seed(loader_seed % (2**32))
    random.seed(loader_seed)
    features = {endpoint: [] for endpoint in range(task + 1)}
    label_chunks, seen_indices = [], []
    batch_count = 0
    try:
        model.eval()
        scratch = deepcopy(model).eval() if task else None
        loader = make_loader(records, config['feature_batch_size'], config['workers'],
                             shuffle=False, seed=loader_seed)
        for indices, images, targets in loader:
            images = images.to(device, non_blocking=True)
            for endpoint, state in adapters.items():
                load_adapter(scratch, state)
                features[endpoint].append(scratch(images)['features'].float().cpu())
            features[task].append(model(images)['features'].float().cpu())
            label_chunks.append(targets.long().cpu())
            seen_indices.extend(indices.tolist())
            batch_count += 1
        features = {endpoint: torch.cat(chunks) for endpoint, chunks in features.items()}
        labels = torch.cat(label_chunks)
        if seen_indices != list(range(len(records))):
            raise ValueError('Current TRAIN records were not covered once in their fixed order')
        shape = features[task].shape
        if len(shape) != 2 or shape[0] != len(labels):
            raise ValueError('Expected aligned sample-by-feature rows')
        if any(rows.shape != shape or not torch.isfinite(rows).all() for rows in features.values()):
            raise ValueError('Historical features disagree in shape or contain NaN/Inf')
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
    finally:
        model.train(was_training)
        restore_rng(rng, device)
    image_count = len(labels) * (task + 1)
    metadata = dict(
        seed=seed, task=task, sample_count=len(labels), snapshot_tasks=list(features),
        per_class_counts={int(label): int((labels == label).sum()) for label in labels.unique(sorted=True)},
        selected_record_indices=seen_indices, loader_seed=loader_seed,
        workers=config['workers'], batch_size=config['feature_batch_size'],
        image_forward_count=image_count, feature_precision='float32',
        storage='Current TRAIN rows in RAM only; no historical-class image replay',
        replay='One stochastic TRAIN-augmented view shared across endpoints 0..t',
        costs=dict(image_forward_count=image_count, batch_forward_count=batch_count*(task+1),
                   total_seconds=perf_counter()-started,
                   temporary_feature_bytes=sum(rows.numel()*rows.element_size() for rows in features.values()),
                   temporary_label_bytes=labels.numel()*labels.element_size()),
    )
    if hasattr(records, 'original_indices'):
        metadata['original_record_indices'] = np.asarray(records.original_indices)[seen_indices].tolist()
    return features, labels, metadata
