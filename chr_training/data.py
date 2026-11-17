"""ImageNet-R records, fixed preprocessing, and current-task-only loaders."""

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def class_order(seed, classes=200):
    return np.random.RandomState(seed).permutation(classes).tolist()


def image_transform(training=False):
    spatial = ([transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip()] if training else
               [transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224)])
    return transforms.Compose(spatial + [transforms.ToTensor(), transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class ImageRecords(Dataset):
    def __init__(self, root, split, order, classes=None, training=False):
        folder = ImageFolder(str(Path(root) / split))
        if len(order) != len(folder.classes):
            raise ValueError("Class order and ImageNet-R folders disagree")
        inverse = {original: label for label, original in enumerate(order)}
        selected = set(range(len(order)) if classes is None else classes)
        self.samples = [(path, inverse[label]) for path, label in folder.samples
                        if inverse[label] in selected]
        self.transform = image_transform(training)
        self.class_names = folder.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return index, self.transform(default_loader(path)), label


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, batch_size=64, workers=4, shuffle=False, seed=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0,
                      generator=torch.Generator().manual_seed(seed),
                      worker_init_fn=_seed_worker)
