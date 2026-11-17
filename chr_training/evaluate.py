"""Evaluate fixed checkpoints on seen test classes and report AIA/Final."""
import argparse
import csv
from pathlib import Path
from statistics import mean

import torch

from .backbone import load_adapter
from .common import head_from_state, load_checkpoint, read_json, setup, write_json
from .data import make_loader
from .datasets import build_records
from .observations import build_model


def metrics(logits, labels, old_count, keep):
    correct = logits.argmax(1).eq(labels)

    def accuracy(mask):
        return 100. * correct[mask].float().mean().item() if mask.any() else None

    old = keep & (labels < old_count)
    old_only = (100. * logits[old, :old_count].argmax(1).eq(labels[old]).float().mean().item()
                if old.any() else None)
    return dict(accuracy=accuracy(keep), standard_accuracy=accuracy(torch.ones_like(keep)),
                old_accuracy=accuracy(old), current_accuracy=accuracy(keep & (labels >= old_count)),
                old_only_accuracy=old_only, test_count=int(keep.sum()),
                excluded_count=int((~keep).sum()))


@torch.no_grad()
def evaluate(run_dir, device='cuda:0', data_root=None, checkpoint=None):
    root = Path(run_dir)
    registry = read_json(root / 'run.json')
    config, seed, order = dict(registry['config']), registry['seed'], registry['class_order']
    if data_root:
        config['data_root'] = data_root
    if checkpoint:
        config['checkpoint'] = checkpoint
    snapshots = sorted(root.glob('task_[0-9][0-9][0-9].pt'))
    if not snapshots:
        raise ValueError('No trained checkpoints found')
    if [int(p.stem[5:]) for p in snapshots] != list(range(len(snapshots))):
        raise ValueError('Evaluation needs a contiguous sequence starting at task 0')
    setup(config, device)
    model = build_model(config, device, initialize=False).eval()
    rows = []
    excluded = torch.tensor(config.get('test_excluded_indices', []), dtype=torch.long)
    for path in snapshots:
        snapshot = load_checkpoint(path)
        task, seen = snapshot['task'], snapshot['classes_seen']
        load_adapter(model, snapshot['adapter'])
        heads = {name: head_from_state(weights, device) for name, weights in snapshot['heads'].items()}
        records = build_records(config, 'test', order, range(seen), training=False)
        loader = make_loader(records, config['feature_batch_size'], config['workers'],
                             seed=seed * 1009 + task)
        scores = {name: [] for name in heads}
        targets, positions = [], []
        for indices, images, labels in loader:
            features = model(images.to(device))['features'].float()
            for name, head in heads.items():
                scores[name].append(head(features).cpu())
            targets.append(labels)
            positions.append(indices)
        targets = torch.cat(targets)
        original_ids = torch.as_tensor(records.original_indices)[torch.cat(positions)]
        keep = ~torch.isin(original_ids, excluded)
        for name in heads:
            row = dict(task=task, method=name, **metrics(
                torch.cat(scores[name]), targets, task * config['classes_per_task'], keep))
            rows.append(row)
        print(f'Task {task + 1}: ' + ', '.join(
            f'{r["method"]}={r["accuracy"]:.2f}%' for r in rows[-len(heads):]), flush=True)
    summary = dict(completed_tasks=len(snapshots), configured_tasks=config['tasks'],
                   smoke=bool(config.get('smoke_samples_per_class')),
                   units='percent', methods={})
    for name in config['methods']:
        selected = [r for r in rows if r['method'] == name]
        summary['methods'][name] = dict(
            AIA=mean(r['accuracy'] for r in selected), Final=selected[-1]['accuracy'],
            Old=mean(r['old_accuracy'] for r in selected if r['old_accuracy'] is not None)
                if len(selected) > 1 else None,
            Current=mean(r['current_accuracy'] for r in selected),
            standard_AIA=mean(r['standard_accuracy'] for r in selected))
    write_json(root / 'evaluation.json', dict(summary=summary, tasks=rows))
    with (root / 'evaluation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data-root')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    print(evaluate(args.run, args.device, args.data_root, args.checkpoint))


if __name__ == '__main__':
    main()
