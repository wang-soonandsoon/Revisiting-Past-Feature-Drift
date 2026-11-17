"""Resume equivalence and test-set metric boundaries on small CPU fixtures."""
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from chr_training.common import load_checkpoint, read_json
from chr_training.evaluate import metrics
from chr_training.train import run


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = nn.Parameter(torch.eye(3) + .02 * torch.randn(3, 3))
        self.dropout = nn.Dropout(.2)

    def forward(self, images):
        return {'features': self.dropout(images) @ self.lora}


class CurrentRecords(torch.utils.data.Dataset):
    def __init__(self, classes):
        self.labels = np.repeat(list(classes), 4)
        self.original_indices = np.arange(len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = int(self.labels[index])
        image = torch.tensor([label + 1., index % 4 + .5, 1.])
        image += .1 * (torch.rand(3) + random.random() + float(np.random.rand()))
        return index, image, label


class TrainingPipelineTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_training_and_reads_only_current_train(self):
        config = read_json(Path(__file__).resolve().parents[1] / 'configs/cifar100_20.json')
        config.update(classes=9, tasks=3, classes_per_task=3, feature_dim=3,
                      epochs=1, batch_size=6, feature_batch_size=6, workers=0,
                      cpu_threads=1)
        calls = []

        def records(cfg, split, order, classes, training=False):
            self.assertEqual(split, 'train')
            self.assertTrue(training)
            classes = list(classes)
            self.assertEqual(len(classes), 3)
            self.assertEqual(classes, list(range(classes[0], classes[0] + 3)))
            calls.append(classes)
            return CurrentRecords(classes)

        with tempfile.TemporaryDirectory() as directory, \
             patch('chr_training.train.build_model', side_effect=lambda *_: TinyBackbone()), \
             patch('chr_training.train.build_records', side_effect=records):
            full = dict(config, output_root=str(Path(directory) / 'full'))
            split = dict(config, output_root=str(Path(directory) / 'resumed'))
            full_path = run(full, 7, 'cpu')
            run(split, 7, 'cpu', max_tasks=1)
            split_path = run(split, 7, 'cpu', resume=True)
            for task in range(3):
                expected = load_checkpoint(full_path / f'task_{task:03d}.pt')
                actual = load_checkpoint(split_path / f'task_{task:03d}.pt')
                for field in ('adapter', 'canonical_head'):
                    for key in expected[field]:
                        torch.testing.assert_close(actual[field][key], expected[field][key], rtol=0, atol=0)
                for method in config['methods']:
                    for key in expected['heads'][method]:
                        torch.testing.assert_close(actual['heads'][method][key], expected['heads'][method][key], rtol=0, atol=0)
                self.assertTrue(torch.equal(actual['rng']['torch'], expected['rng']['torch']))
            expected = load_checkpoint(full_path / 'resume.pt')['statistics']
            actual = load_checkpoint(split_path / 'resume.pt')['statistics']
            self.assertEqual(actual['accumulator']['steps'][1]['count'], 12)
            for key in ('gram', 'cross'):
                torch.testing.assert_close(actual['accumulator']['steps'][1][key],
                                           expected['accumulator']['steps'][1][key], rtol=0, atol=0)
            for group in ('point', 'births'):
                for label, value in expected[group].items():
                    torch.testing.assert_close(actual[group][label].mean, value.mean, rtol=0, atol=0)
                    torch.testing.assert_close(actual[group][label].cov, value.cov, rtol=0, atol=0)
        self.assertEqual(calls, [[0, 1, 2], [3, 4, 5], [6, 7, 8]] * 2)

    def test_filtered_and_full_test_metrics_keep_old_current_boundaries(self):
        logits = torch.tensor([[3., 1., 5.], [1., 5., 0.], [1., 0., 5.], [4., 1., 2.]])
        labels = torch.tensor([0, 1, 2, 2])
        result = metrics(logits, labels, old_count=2, keep=torch.tensor([True, True, True, False]))
        self.assertAlmostEqual(result['accuracy'], 200. / 3., places=4)
        self.assertEqual(result['standard_accuracy'], 50.)
        self.assertEqual(result['old_accuracy'], 50.)
        self.assertEqual(result['old_only_accuracy'], 100.)
        self.assertEqual(result['current_accuracy'], 100.)
        self.assertEqual((result['test_count'], result['excluded_count']), (3, 1))
