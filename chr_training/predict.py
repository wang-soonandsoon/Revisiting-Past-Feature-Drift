"""Predict one image with one saved encoder and aligned classifier."""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from .backbone import load_adapter
from .cifar_data import cifar_transform
from .common import head_from_state, load_checkpoint, read_json, setup
from .cub_data import cub_transform
from .data import image_transform
from .observations import build_model


@torch.no_grad()
def predict(run_dir, image, method='joint', task=None, device='cuda:0', checkpoint=None):
    root = Path(run_dir)
    registry = read_json(root / 'run.json')
    config = dict(registry['config'])
    if checkpoint:
        config['checkpoint'] = checkpoint
    setup(config, device)
    path = (sorted(root.glob('task_[0-9][0-9][0-9].pt'))[-1] if task is None
            else root / f'task_{task:03d}.pt')
    snapshot = load_checkpoint(path)
    model = build_model(config, device, initialize=False).eval()
    load_adapter(model, snapshot['adapter'])
    head = head_from_state(snapshot['heads'][method], device)
    transform = {'cifar100': cifar_transform, 'imagenet_r': image_transform,
                 'cub200_224': cub_transform}[config['dataset']](False)
    with Image.open(image) as im:
        inputs = transform(im.convert('RGB')).unsqueeze(0).to(device)
    label = int(head(model(inputs)['features']).argmax(1))
    return dict(task=snapshot['task'], method=method, incremental_class=label,
                original_class=registry['class_order'][label])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--method', default='joint')
    parser.add_argument('--task', type=int, help='Zero-based task; default is the last checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    print(json.dumps(predict(args.run, args.image, args.method, args.task, args.device, args.checkpoint)))


if __name__ == '__main__':
    main()
