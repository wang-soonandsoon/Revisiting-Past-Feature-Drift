"""CUB-200-2011 using the official split and preprocessing."""
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms import InterpolationMode

def cub_transform(training=False):
    spatial = ([transforms.Resize((300, 300), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomCrop((224, 224)), transforms.RandomHorizontalFlip()]
               if training else
               [transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224)])
    return transforms.Compose(spatial + [transforms.ToTensor(), transforms.Normalize(
        [.485, .456, .406], [.229, .224, .225])])


def _indexed_metadata(path, value_type=str):
    rows = [(int(key), value_type(value))
            for key, value in (line.split(maxsplit=1) for line in path.read_text().splitlines())]
    result = dict(rows)
    if len(result) != len(rows):
        raise ValueError(f'duplicate metadata ID in {path.name}')
    return result


class NativeCUB200Records(Dataset):
    def __init__(self, root, split, order, classes=None, training=False):
        if split not in ('train', 'test'):
            raise ValueError('CUB official split must be train or test')
        root = Path(root)
        names = _indexed_metadata(root / 'classes.txt')
        images = _indexed_metadata(root / 'images.txt')
        labels = _indexed_metadata(root / 'image_class_labels.txt', int)
        flags = _indexed_metadata(root / 'train_test_split.txt', int)
        if set(images) != set(labels) or set(images) != set(flags) or min(images) < 1:
            raise ValueError('CUB metadata image ID sets disagree')
        if set(names) != set(range(1, len(names) + 1)) or sorted(order) != list(range(len(names))):
            raise ValueError('CUB class order must permute every official class once')
        if set(labels.values()) != set(names) or any(
                Path(path).parent.name != names[labels[rid]] for rid, path in images.items()):
            raise ValueError('CUB class label and path metadata disagree')
        if set(flags.values()) != {0, 1} or any(
                {labels[rid] for rid in images if flags[rid] == flag} != set(names)
                for flag in (0, 1)):
            raise ValueError('CUB official split must cover every class in TRAIN and TEST')
        selected = list(range(len(order))) if classes is None else list(classes)
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(range(len(order))):
            raise ValueError('CUB selected incremental classes must be distinct valid labels')
        inverse = {original + 1: incremental for incremental, original in enumerate(order)}
        selected_set = set(selected)
        split_flag = int(split == 'train')
        source = [(rid, path, inverse[labels[rid]]) for rid, path in images.items()
                  if flags[rid] == split_flag and inverse[labels[rid]] in selected_set]
        rank = {label: index for index, label in enumerate(selected)}
        source.sort(key=lambda row: (rank[row[2]], row[1]))
        self.original_indices = np.asarray([rid for rid, _, _ in source], dtype=np.int64)
        self.original_paths = [str((root / 'images' / path).resolve()) for _, path, _ in source]
        self.labels = np.asarray([label for _, _, label in source], dtype=np.int64)
        self.samples = list(zip(self.original_paths, self.labels.tolist()))
        self.class_names = [names[label] for label in range(1, len(names) + 1)]
        self.transform = cub_transform(training)
        self.original_record_index_source = (
            'official one-based CUB-200-2011 image_id in images.txt; '
            'train_test_split.txt 1=TRAIN, 0=TEST; distinct from dataset positions')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return index, self.transform(default_loader(path)), label
