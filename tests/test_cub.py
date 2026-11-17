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
from chr_training.cub_data import cub_transform

ROOT = Path(__file__).resolve().parents[1]


def upstream_transform(training):
    path = ROOT / 'external/SLDC/utils/data.py'
    tree = ast.parse(path.read_text())
    declaration = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == 'iCUB200_224')
    scope = {'iData': object, 'np': np, 'transforms': transforms}
    exec(compile(ast.Module(body=[declaration], type_ignores=[]), str(path), 'exec'), scope)
    native = scope['iCUB200_224']
    return transforms.Compose((native.train_trsf if training else native.test_trsf)
                              + native.common_trsf)


def make_cub(root):
    # Metadata IDs, metadata line order and source ImageFolder filename order
    # deliberately differ. Labels and official splits remain explicit.
    rows = [(9, 2, 'z_train', 1), (2, 0, 'z_train', 1), (7, 1, 'b_test', 0),
            (4, 2, 'a_train', 1), (1, 0, 'a_test', 0), (10, 1, 'a_train', 1),
            (5, 2, 'm_train', 1), (8, 0, 'a_train', 1), (3, 1, 'a_test', 0),
            (6, 2, 'a_test', 0)]
    names = [f'{label + 1:03d}.class_{label}' for label in range(3)]
    root.mkdir(parents=True, exist_ok=True)
    for filename, values in {
        'classes.txt': [(label + 1, name) for label, name in enumerate(names)],
        'images.txt': [(rid, f'{names[label]}/{name}.jpg') for rid, label, name, _ in rows],
        'image_class_labels.txt': [(rid, label + 1) for rid, label, _, _ in rows],
        'train_test_split.txt': [(rid, split) for rid, _, _, split in rows],
    }.items():
        (root / filename).write_text(''.join(f'{key} {value}\n' for key, value in values))
    for rid, label, name, _ in rows:
        path = root / 'images' / names[label] / f'{name}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.random.RandomState(rid).randint(0, 256, (231, 317, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
    return rows, names


class CUBTests(unittest.TestCase):
    def test_cub_transforms_match_source_seeded_pixels(self):
        image = Image.fromarray(np.random.RandomState(81).randint(
            0, 256, (281, 367, 3), dtype=np.uint8))
        for training in (False, True):
            with self.subTest(training=training):
                torch.manual_seed(124)
                expected = upstream_transform(training)(image)
                torch.manual_seed(124)
                actual = cub_transform(training)(image)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.shape, (3, 224, 224))

    def test_official_ids_splits_and_native_class_filename_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, names = make_cub(root)
            config = {'dataset': 'cub200_224', 'data_root': directory}
            order, selected = [2, 0, 1], [2, 0]
            for split, flag in [('train', 1), ('test', 0)]:
                with self.subTest(split=split):
                    expected = [(rid, str((root / 'images' / names[label] / f'{name}.jpg').resolve()),
                                 order.index(label))
                                for incremental in selected
                                for rid, label, name, is_train in sorted(rows, key=lambda row: row[2])
                                if is_train == flag and order.index(label) == incremental]
                    with patch('PIL.Image.open', side_effect=AssertionError('Pixels opened in metadata build')):
                        records = build_records(config, split, order, selected, training=split == 'train')
                    self.assertEqual(records.samples, [(path, label) for _, path, label in expected])
                    self.assertEqual(records.original_indices.tolist(), [rid for rid, _, _ in expected])
                    self.assertEqual(records.original_paths, [path for _, path, _ in expected])
                    self.assertEqual(records.labels.tolist(), [label for _, _, label in expected])
                    self.assertEqual(records.class_names, names)
                    self.assertIn('one-based', records.original_record_index_source)
                    self.assertIn('image_id', records.original_record_index_source)
                    torch.manual_seed(52)
                    expected_pixels = upstream_transform(split == 'train')(
                        Image.open(expected[0][1]).convert('RGB'))
                    torch.manual_seed(52)
                    index, pixels, target = records[0]
                    self.assertEqual((index, target), (0, expected[0][2]))
                    torch.testing.assert_close(pixels, expected_pixels, rtol=0, atol=0)
            complete = build_records(config, 'train', order)
            self.assertEqual(complete.original_indices.tolist(), [4, 5, 9, 8, 2, 10])
            self.assertEqual(complete.labels.tolist(), [0, 0, 0, 1, 1, 2])
