"""Dataset preprocessing and source row/split alignment."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from chr_training.datasets import build_records
from chr_training.data import image_transform as imagenet_r_transform

ROOT = Path(__file__).resolve().parents[1]


def upstream_transform(training):
    """Execute only iImageNetR's declaration, not its download or GPU code."""
    path = ROOT / 'external/SLDC/utils/data.py'
    tree = ast.parse(path.read_text())
    source_class = next(node for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == 'iImageNetR')
    scope = {'iData': object, 'np': np, 'transforms': transforms}
    exec(compile(ast.Module(body=[source_class], type_ignores=[]), str(path), 'exec'), scope)
    native = scope['iImageNetR']
    return transforms.Compose((native.train_trsf if training else native.test_trsf)
                              + native.common_trsf)


def make_split(root):
    # Deliberately create filenames and classes in a different order from the
    # lexicographic ImageFolder inventory used by the source DataManager.
    for split in ('test', 'train'):
        for label in (2, 0, 1):
            directory = root / split / f'class_{label}'
            directory.mkdir(parents=True)
            for name, value in [('z.png', 80), ('a.png', 35)]:
                image = np.full((31, 43, 3), value + label, dtype=np.uint8)
                Image.fromarray(image).save(directory / name)


class ImageNetRTests(unittest.TestCase):
    def test_ir_transforms_match_upstream_seeded_pixels(self):
        image = Image.fromarray(np.random.RandomState(91).randint(
            0, 256, size=(281, 367, 3), dtype=np.uint8))
        for training in (False, True):
            with self.subTest(training=training):
                torch.manual_seed(124)
                expected = upstream_transform(training)(image)
                torch.manual_seed(124)
                transform = imagenet_r_transform(training)
                actual = transform(image)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.shape, (3, 224, 224))
                self.assertEqual(transform.transforms[-1].mean, [.485, .456, .406])
                self.assertEqual(transform.transforms[-1].std, [.229, .224, .225])
                names = [type(step).__name__ for step in transform.transforms]
                self.assertEqual(names, ['RandomResizedCrop', 'RandomHorizontalFlip',
                                         'ToTensor', 'Normalize'] if training else
                                        ['Resize', 'CenterCrop', 'ToTensor', 'Normalize'])

    def test_selected_classes_concatenate_source_rows_with_original_ids_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_split(root)
            config = {'dataset': 'imagenet_r', 'data_root': directory}
            order, selected = [2, 0, 1], [2, 0]
            for split in ('train', 'test'):
                with self.subTest(split=split):
                    source = ImageFolder(str(root / split)).samples
                    # Literal native get_dataset(indices)->_select then
                    # np.concatenate, retaining the original inventory index.
                    expected = [(row, path, order.index(original))
                                for incremental in selected
                                for row, (path, original) in enumerate(source)
                                if order.index(original) == incremental]
                    records = build_records(config, split, order, selected, training=split == 'train')
                    self.assertEqual(records.samples, [(p, label) for _, p, label in expected])
                    self.assertEqual(records.labels.tolist(), [label for _, _, label in expected])
                    self.assertEqual(records.original_indices.tolist(), [row for row, _, _ in expected])
                    self.assertEqual(records.original_paths, [str(Path(p).resolve()) for _, p, _ in expected])
                    self.assertIn('zero-based', records.original_record_index_source)
                    self.assertIn(split, records.original_record_index_source)
                    torch.manual_seed(51)
                    expected_pixels = upstream_transform(split == 'train')(Image.open(expected[0][1]).convert('RGB'))
                    torch.manual_seed(51)
                    index, actual_pixels, target = records[0]
                    self.assertEqual((index, target), (0, expected[0][2]))
                    torch.testing.assert_close(actual_pixels, expected_pixels, rtol=0, atol=0)
            complete = build_records(config, 'train', order)
            self.assertEqual(complete.labels.tolist(), [0, 0, 1, 1, 2, 2])
            self.assertEqual(complete.original_indices.tolist(), [4, 5, 0, 1, 2, 3])
