"""ImageNet-R on the fixed PILOT train/test split."""
from pathlib import Path
import numpy as np
from .data import ImageRecords

class NativeImageNetRRecords(ImageRecords):
    def __init__(self, root, split, order, classes=None, training=False):
        # Inventory all paths without opening images, so IDs remain those of
        # the original split even after current-class selection/reordering.
        super().__init__(root, split, order, classes=None, training=training)
        selected = list(range(len(order))) if classes is None else list(classes)
        source = self.samples
        self.original_indices = np.asarray([
            index for label in selected for index, (_, target) in enumerate(source)
            if target == label
        ], dtype=np.int64)
        self.samples = [source[index] for index in self.original_indices]
        self.labels = np.asarray([label for _, label in self.samples], dtype=np.int64)
        self.original_paths = [str(Path(path).resolve()) for path, _ in self.samples]
        self.original_record_index_source = (
            f'zero-based ImageFolder({split}).samples row in the complete fixed '
            'split, before incremental class selection/grouping')
