"""Dataset dispatch and small, class-balanced smoke-test subsets."""
import numpy as np
from torch.utils.data import Dataset

from .cifar_data import NativeCifar100Records
from .cub_data import NativeCUB200Records
from .imagenet_r import NativeImageNetRRecords


class LimitedRecords(Dataset):
    def __init__(self, records, per_class):
        self.records = records
        labels = np.asarray(records.labels)
        self.positions = np.concatenate([
            np.flatnonzero(labels == c)[:per_class] for c in np.unique(labels)])
        self.labels = labels[self.positions]
        self.original_indices = np.asarray(records.original_indices)[self.positions]
        self.class_names = records.class_names

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        _, image, label = self.records[int(self.positions[index])]
        return index, image, label


def build_records(config, split, order, classes=None, training=False):
    factories = {
        'cifar100': NativeCifar100Records,
        'imagenet_r': NativeImageNetRRecords,
        'cub200_224': NativeCUB200Records,
    }
    records = factories[config['dataset']](
        config['data_root'], split, order, classes, training=training)
    limit = config.get('smoke_samples_per_class')
    return LimitedRecords(records, limit) if limit else records
