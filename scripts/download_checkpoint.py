"""Download the pretrained ViT weights once."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

NAME = 'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz'
URL = 'https://storage.googleapis.com/vit_models/augreg/' + NAME
SHA256 = 'a157033e309d33e79cbff2408dd77b4458014fcedc46f0ad853e2d08ecb493f6'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('checkpoints') / NAME)
    args = parser.parse_args()
    if args.output.exists():
        print(f'Using existing weights: {args.output}')
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix('.download')
    urlretrieve(URL, partial)
    digest = hashlib.sha256()
    with partial.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise RuntimeError(f'Checkpoint checksum mismatch: {partial}')
    partial.replace(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
