"""CIFAR-100 Python-pickle records with the existing ViT image preprocessing."""

from pathlib import Path
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .data import image_transform


class Cifar100Records(Dataset):
    """Serve only selected incremental labels; preserve official train/test split.

    ``order[label]`` is the original fine-class ID, as in ImageRecords. Files
    are the already-extracted trusted official ``cifar-100-python`` pickles.
    The selected image array drops every unselected class immediately.
    """

    def __init__(self, root, split, order, classes=None, training=False):
        if split not in ('train', 'test'):
            raise ValueError('CIFAR-100 split must be train or test')
        root = Path(root)
        with (root / 'meta').open('rb') as handle:
            meta = pickle.load(handle, encoding='bytes')
        self.class_names = [name.decode('utf-8') if isinstance(name, bytes) else name
                            for name in meta[b'fine_label_names']]
        if sorted(order) != list(range(len(self.class_names))):
            raise ValueError('Class order must permute the CIFAR fine classes')
        with (root / split).open('rb') as handle:
            payload = pickle.load(handle, encoding='bytes')
        original = np.asarray(payload[b'fine_labels'], dtype=np.int64)
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[np.asarray(order)] = np.arange(len(order))
        labels = inverse[original]
        selected = np.arange(len(order)) if classes is None else np.asarray(list(classes))
        self.original_indices = np.flatnonzero(np.isin(labels, selected))
        self.labels = labels[self.original_indices]
        self.images = np.asarray(payload[b'data'])[self.original_indices].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        unique, counts = np.unique(self.labels, return_counts=True)
        self.class_counts = dict(zip(unique.tolist(), counts.tolist()))
        self.transform = image_transform(training)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index, self.transform(Image.fromarray(self.images[index])), int(self.labels[index])

from torchvision import transforms
from torchvision.transforms import InterpolationMode

def cifar_transform(training=False):
    """Author's 224-pixel CIFAR recipe, including its own normalization."""
    spatial = ([transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63 / 255)] if training else
               [transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224)])
    return transforms.Compose(spatial + [
        transforms.ToTensor(),
        transforms.Normalize((.5071, .4867, .4408), (.2675, .2565, .2761)),
    ])


class NativeCifar100Records(Cifar100Records):
    def __init__(self, root, split, order, classes=None, training=False):
        selected = list(range(len(order))) if classes is None else list(classes)
        super().__init__(root, split, order, selected, training=training)
        # Native DataManager concatenates selected classes, then shuffles in
        # the loader. Preserve that order as well as the original file IDs.
        positions = np.concatenate([np.flatnonzero(self.labels == label) for label in selected])
        self.images = self.images[positions]
        self.labels = self.labels[positions]
        self.original_indices = self.original_indices[positions]
        self.transform = cifar_transform(training)
