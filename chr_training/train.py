"""Train canonical encoders and align CHR classifiers without reading test data."""
import argparse
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import torch

from chr_core import (corrected_statistics, empty_accumulator,
                      select_cumulative_chains, select_passive_chains,
                      update_accumulator)
from .backbone import adapter_state, load_adapter
from .common import cpu_state, load_checkpoint, read_json, setup, write_json
from .data import class_order
from .datasets import build_records
from .heads import ContinualLinear
from .native import (cached_noise, fit_native_head, linear_update,
                     native_birth, native_factor, native_factors, weak_update)
from .observations import build_model, extract_train_snapshots, train_current
from .training import capture_rng, restore_rng, seed_everything

METHODS = ('mean', 'joint', 'current_only', 'sldc_linear', 'sldc')


def sampling_factors(point, births_now, device):
    old = {c: stat for c, stat in point.items() if c not in births_now}
    factors = native_factors(old, device)
    factors.update({c: native_factor(stat, 'cpu') for c, stat in births_now.items()})
    return factors


def revise(config, task, features, labels, previous, device, noise, seed):
    """Update native moments, accumulated evidence and complete historical paths."""
    current = native_birth(features[task], labels)
    if task:
        point, bridge, _ = linear_update(
            previous['point'], features[task - 1], features[task], device,
            gamma_1=config['gamma_1'], alpha_t=config['alpha_t'])
        transports = previous['transports'] + [bridge]
        births = dict(previous['births'])
    else:
        point, births = {}, {}
        transports = [torch.eye(features[0].shape[1])]
    point.update(current)
    births.update(current)
    accumulator = previous['accumulator'] if previous else empty_accumulator()
    paths, passive, selections = {}, {}, {}
    if task >= 2:
        historical = {s: features[s] for s in range(task)}
        accumulator = update_accumulator(accumulator, historical, task)
        fits = select_cumulative_chains(historical, labels, transports, task,
                                        config['alphas'], accumulator)
        paths = {b: f['map'] for b, f in fits.items()}
        selections = {b: {'alpha': f['alpha'], 'baseline_score': f['baseline_score'],
                          'selected_score': f['selected_score']} for b, f in fits.items()}
        if 'current_only' in config['methods']:
            fits = select_passive_chains(historical, labels, transports, task, config['alphas'])
            passive = {b: f['map'] for b, f in fits.items()}
    common = (point, births, transports, paths, task, config['classes_per_task'],
              config['transport_jitter'])
    states = {'sldc_linear': point, 'mean': corrected_statistics(*common, mode='mean')}
    factors = sampling_factors(point, current, device)
    branch_factors = {'sldc_linear': factors, 'mean': factors, 'current_only': factors}
    if 'joint' in config['methods']:
        states['joint'] = corrected_statistics(*common, mode='joint')
        changed = {c: stat for c, stat in states['joint'].items() if stat.cov is not point[c].cov}
        branch_factors['joint'] = factors | native_factors(changed, device)
    if 'current_only' in config['methods']:
        states['current_only'] = corrected_statistics(
            point, births, transports, passive, task, config['classes_per_task'],
            config['transport_jitter'], mode='mean')
    saved = dict(point=point, births=births, transports=transports,
                 accumulator=accumulator, paths=paths)
    if 'sldc' in config['methods']:
        weak = {}
        if task:
            weak, _, _ = weak_update(previous['weak'], features[task - 1], features[task],
                                     noise, seed + 43003 * task, device,
                                     gamma_2=config['gamma_2'], factors=previous['weak_factors'])
        weak.update(current)
        states['sldc'] = weak
        branch_factors['sldc'] = sampling_factors(weak, current, device)
        saved.update(weak=weak, weak_factors=branch_factors['sldc'])
    return saved, states, branch_factors, selections


def run(config, seed, device='cuda:0', max_tasks=None, resume=False):
    setup(config, device)
    methods = config['methods']
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
        raise ValueError(f'Choose distinct methods from {METHODS}')
    cpt = config['classes_per_task']
    if cpt < 3 or config['classes'] != cpt * config['tasks']:
        raise ValueError('Use equal tasks of at least three classes covering the dataset')
    if config['transport_jitter'] != .01:
        raise ValueError('Joint CHR must retain the native transport jitter of 0.01')
    count = config['tasks'] if max_tasks is None else max_tasks
    if not 1 <= count <= config['tasks']:
        raise ValueError('max_tasks must be within the configured sequence')
    seed_everything(seed)
    order = class_order(seed, config['classes'])
    root = Path(config['output_root']) / f'seed{seed}'
    registry = dict(config=config, seed=seed, class_order=order)
    if (root / 'run.json').exists():
        if not resume:
            raise ValueError('Run already exists; use --resume or a different --output')
        if read_json(root / 'run.json') != registry:
            raise ValueError('Resume configuration differs from the saved run')
    write_json(root / 'run.json', registry)
    model = build_model(config, device)
    dim = config.get('feature_dim', 768)
    canonical = ContinualLinear(dim, cpt).to(device)
    noise = cached_noise(dim, config['noise_seed'])
    previous, adapters, start = None, {}, 0
    if resume and (root / 'resume.pt').exists():
        saved = load_checkpoint(root / 'resume.pt')
        start, previous = saved['task'] + 1, saved['statistics']
        for task in range(start):
            snapshot = load_checkpoint(root / f'task_{task:03d}.pt')
            adapters[task] = snapshot['adapter']
            if task:
                canonical.update(cpt, freeze_old=False)
        canonical.to(device)
        load_adapter(model, snapshot['adapter'])
        canonical.load_state_dict(snapshot['canonical_head'])
        restore_rng(snapshot['rng'], device)
    for task in range(start, count):
        tick = perf_counter()
        if task:
            canonical.update(cpt, freeze_old=False)
            canonical.to(device)
        records = build_records(config, 'train', order, range(task * cpt, (task + 1) * cpt), training=True)
        teacher = deepcopy(model).eval().requires_grad_(False) if task else None
        print(f'Task {task + 1}/{count}: train {len(records)} images', flush=True)
        training = train_current(model, canonical, teacher, records, task * cpt, config, seed, task, device)
        del teacher
        snapshot = dict(task=task, adapter=adapter_state(model), canonical_head=cpu_state(canonical),
                        rng=capture_rng(device), classes_seen=(task + 1) * cpt)
        features, labels, extraction = extract_train_snapshots(
            model, None, records, adapters, config, seed, task, device)
        saved, states, factors, selection = revise(config, task, features, labels, previous, device, noise, seed)
        heads, alignment = {}, {}
        for method in methods:
            # Every branch starts from the same canonical head and random stream.
            fitted, info = fit_native_head(states[method], canonical, noise,
                                          seed + 10007 * task, device, factors=factors[method])
            heads[method], alignment[method] = cpu_state(fitted), info
            del fitted
        snapshot['heads'] = heads
        torch.save(snapshot, root / f'task_{task:03d}.pt')
        torch.save(dict(task=task, statistics=saved), root / 'resume.pt')
        write_json(root / f'task_{task:03d}.json', dict(
            task=task, training=training, extraction=extraction, selection=selection,
            alignment=alignment, seconds=perf_counter() - tick))
        previous, adapters[task] = saved, snapshot['adapter']
        del features, labels, states, factors, records
        print(f'Task {task + 1}/{count}: saved {", ".join(methods)} ({perf_counter() - tick:.1f}s)', flush=True)
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data-root')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output')
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    parser.add_argument('--max-tasks', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Three tasks, one training epoch and four images per class')
    args = parser.parse_args()
    config = read_json(args.config)
    for key, value in [('data_root', args.data_root), ('checkpoint', args.checkpoint),
                       ('output_root', args.output), ('methods', args.methods)]:
        if value is not None:
            config[key] = value
    if args.smoke:
        config.update(epochs=1, workers=0, smoke_samples_per_class=4)
        config['output_root'] += '_smoke'
        args.max_tasks = min(3, config['tasks'])
    print(run(config, args.seed, args.device, args.max_tasks, args.resume))


if __name__ == '__main__':
    main()
