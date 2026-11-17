"""Independent native preprocessing, canonical-loss, and shared-view checks."""
from copy import deepcopy
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from chr_training.heads import ContinualLinear
from chr_training.datasets import build_records
from chr_training.observations import ( build_model, train_current, extract_train_snapshots,
)


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = nn.Parameter(torch.tensor([[1.0, .2], [-.1, .8]]))

    def forward(self, x):
        return {'features': x @ self.lora}


class ToyRecords(torch.utils.data.Dataset):
    def __init__(self, stochastic=False):
        self.labels = np.array([2, 3, 2, 3])
        self.original_indices = np.array([11, 13, 17, 19])
        self.stochastic = stochastic

    def __len__(self):
        return 4

    def __getitem__(self, index):
        value = torch.tensor([index + .25, 1.5 - index])
        if self.stochastic:
            value = value + torch.rand(2) + random.random() + float(np.random.rand())
        return index, value, int(self.labels[index])


class ObservationsTests(unittest.TestCase):
    def test_cifar_transform_matches_native_source_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / 'meta').open('wb') as handle:
                pickle.dump({b'fine_label_names': [b'a', b'b', b'c']}, handle)
            image = np.arange(3072, dtype=np.uint8).reshape(3, 32, 32)
            for split, ids in [('train', [0, 1, 2]), ('test', [2, 0])]:
                with (root / split).open('wb') as handle:
                    pickle.dump({b'data': np.stack([image + i for i in range(len(ids))]).reshape(len(ids), -1),
                                 b'fine_labels': ids}, handle)
            config = {'dataset': 'cifar100', 'data_root': directory}
            records = build_records(config, 'train', [2, 0, 1], [1, 2], training=True)
            self.assertEqual(records.labels.tolist(), [1, 2])
            self.assertEqual(records.original_indices.tolist(), [0, 1])
            # Literal external/SLDC/utils/data.py iCIFAR100_224 recipe.
            reference = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=63/255),
                transforms.ToTensor(),
                transforms.Normalize((.5071, .4867, .4408), (.2675, .2565, .2761)),
            ])
            torch.manual_seed(246)
            expected = reference(Image.fromarray(image.transpose(1, 2, 0)))
            torch.manual_seed(246)
            _, actual, label = records[0]
            self.assertEqual(label, 1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            grouped = build_records(config, 'train', [2, 0, 1], [2, 0, 1])
            self.assertEqual(grouped.labels.tolist(), [2, 0, 1])
            self.assertEqual(grouped.original_indices.tolist(), [1, 2, 0])
            test = build_records(config, 'test', [2, 0, 1], [0])
            self.assertEqual(test.original_indices.tolist(), [0])
            self.assertEqual(test[0][1].shape, (3, 224, 224))
            self.assertEqual([type(step).__name__ for step in test.transform.transforms],
                             ['Resize', 'CenterCrop', 'ToTensor', 'Normalize'])

    def test_model_builder_uses_declared_local_backbone_and_sdpa(self):
        marker = object()
        config = {'checkpoint': 'local.npz', 'rank': 4, 'attention_backend': 'sdpa'}
        with patch('chr_training.observations.build_backbone', return_value=marker) as builder, \
             patch('chr_training.observations.enable_sdpa', return_value=marker) as enable:
            self.assertIs(build_model(config, 'cpu', initialize=False), marker)
        builder.assert_called_once_with('local.npz', rank=4, device='cpu', initialize_lora=False,
                                        only_lora=True, backbone_type='in21k')
        enable.assert_called_once_with(marker)

    def test_single_update_matches_literal_native_current_sce_and_kd(self):
        torch.manual_seed(31)
        model = ToyBackbone()
        teacher = deepcopy(model)
        with torch.no_grad():
            teacher.lora.add_(.3)
        head = ContinualLinear(2, 2)
        head.update(2, freeze_old=False)
        expected_model, expected_head = deepcopy(model), deepcopy(head)
        config = dict(learning_rate=1e-4, head_lr_scale=10, weight_decay=3e-5,
                      epochs=1, batch_size=4, workers=0, gamma_kd=1., gamma_norm=.1,
                      sce_a=.5, sce_b=.5, train_precision='float32')
        samples = ToyRecords()
        images = torch.stack([samples[i][1] for i in range(4)])
        labels = torch.tensor(samples.labels)
        optimizer = torch.optim.AdamW([
            {'params': expected_model.parameters(), 'lr': 1e-4},
            {'params': expected_head.parameters(), 'lr': 1e-3},
        ], weight_decay=3e-5)
        features = expected_model(images)['features']
        logits = expected_head(features)
        previous = teacher(images)['features'].detach()
        pred = logits[:, 2:].softmax(1).clamp(1e-7, 1.)
        one_hot = torch.nn.functional.one_hot(labels-2, 2).float().clamp(1e-4, 1.)
        sce = .5 * -(one_hot * pred.log()).sum(1).mean() + .5 * -(pred * one_hot.log()).sum(1).mean()
        kd = (features-previous).square().mean()
        norm = (features.norm(dim=1)-previous.norm(dim=1)).square().mean()
        loss = sce + kd + .1 * norm
        loss.backward()
        optimizer.step()
        info = train_current(model, head, teacher, samples, 2, config, 4001, 1, 'cpu')
        for actual, expected in zip(model.parameters(), expected_model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        for actual, expected in zip(head.parameters(), expected_head.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(info['epochs'][0]['loss'], float(loss.detach()), places=5)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        self.assertEqual(info['training_rows'], 4)

    def test_shared_augmented_views_and_live_model_rng_survive_extraction(self):
        torch.manual_seed(49)
        random.seed(49)
        np.random.seed(49)
        model = ToyBackbone().train()
        original = deepcopy(model.state_dict())
        prior = {'lora': original['lora'] * 2}
        config = dict(feature_batch_size=2, workers=0)
        before_torch = torch.get_rng_state().clone()
        before_numpy = np.random.get_state()
        before_python = random.getstate()
        features, labels, info = extract_train_snapshots(
            model, None, ToyRecords(stochastic=True), {0: prior}, config, 4001, 1, 'cpu')
        self.assertEqual(sorted(features), [0, 1])
        torch.testing.assert_close(features[0], 2*features[1], rtol=1e-6, atol=1e-6)
        self.assertEqual(labels.tolist(), [2, 3, 2, 3])
        self.assertEqual(info['snapshot_tasks'], [0, 1])
        self.assertEqual(info['costs']['image_forward_count'], 8)
        self.assertEqual(info['selected_record_indices'], [0, 1, 2, 3])
        self.assertEqual(info['original_record_indices'], [11, 13, 17, 19])
        self.assertTrue(model.training)
        torch.testing.assert_close(model.state_dict()['lora'], original['lora'], rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))
        self.assertEqual(random.getstate(), before_python)
        after_numpy = np.random.get_state()
        self.assertEqual(before_numpy[0], after_numpy[0])
        np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
        self.assertEqual(before_numpy[2:], after_numpy[2:])
        again, again_labels, _ = extract_train_snapshots(
            model, None, ToyRecords(stochastic=True), {0: prior}, config, 4001, 1, 'cpu')
        torch.testing.assert_close(features[1], again[1], rtol=0, atol=0)
        self.assertTrue(torch.equal(labels, again_labels))

    def test_extra_native_branch_does_not_change_next_task_training(self):
        # Exercise the next head initialization AND stochastic canonical step,
        # instead of checking RNG bytes alone.
        from chr_training.native import cached_noise, fit_native_head, native_birth

        class StochasticBackbone(ToyBackbone):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(.25)

            def forward(self, x):
                return {'features': self.dropout(x) @ self.lora}

        config = dict(learning_rate=1e-4, head_lr_scale=10, weight_decay=3e-5,
                      epochs=1, batch_size=4, workers=0, gamma_kd=1., gamma_norm=.1,
                      sce_a=.5, sce_b=.5, train_precision='float32')

        def run(extra_branch):
            random.seed(303)
            np.random.seed(303)
            torch.manual_seed(303)
            model, head = StochasticBackbone(), ContinualLinear(2, 2)
            first = ToyRecords(stochastic=True)
            first.labels -= 2
            train_current(model, head, None, first, 0, config, 4001, 0, 'cpu')
            if extra_branch:
                # Extra deployment work must neither mutate canonical weights
                # nor consume the RNG needed for a later head/backbone update.
                x = torch.tensor([[1., 0.], [.5, 1.], [0., 1.], [1., .5]])
                stats = native_birth(x, torch.tensor([0, 0, 1, 1]))
                fit_native_head(stats, head, cached_noise(2, 77, rows=5000), 99, 'cpu')
            teacher = deepcopy(model)
            head.update(2, freeze_old=False)
            train_current(model, head, teacher, ToyRecords(stochastic=True),
                          2, config, 4001, 1, 'cpu')
            return model.state_dict(), head.state_dict(), torch.rand(7), random.random(), np.random.rand(3)

        baseline, extra = run(False), run(True)
        for index in (0, 1):
            for key in baseline[index]:
                torch.testing.assert_close(baseline[index][key], extra[index][key], rtol=0, atol=0)
        self.assertTrue(torch.equal(baseline[2], extra[2]))
        self.assertEqual(baseline[3], extra[3])
        np.testing.assert_array_equal(baseline[4], extra[4])

    def test_task_zero_extracts_only_current_endpoint(self):
        samples = ToyRecords()
        samples.labels[:] = np.array([0, 1, 0, 1])
        features, labels, info = extract_train_snapshots(
            ToyBackbone(), None, samples, {}, dict(feature_batch_size=4, workers=0), 4001, 0, 'cpu')
        self.assertEqual(list(features), [0])
        self.assertEqual(info['costs']['image_forward_count'], 4)


if __name__ == '__main__':
    unittest.main()
