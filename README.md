# Revisiting Past Feature Drift for Exemplar-Free Class-Incremental Learning

Official implementation of **Revisiting Past Feature Drift for Exemplar-Free Class-Incremental Learning**.

## Installation

Python 3.11 and PyTorch 2.8.0.

```bash
git clone --recurse-submodules https://github.com/wang-soonandsoon/Revisiting-Past-Feature-Drift.git
cd Revisiting-Past-Feature-Drift
pip install -e ".[train]"
```

For an existing clone: `git submodule update --init --recursive`.
Prepare the datasets and pretrained weights as described in [Data](docs/DATA.md).

## Training and evaluation

```bash
python -m chr_training.train --config configs/cifar100_20.json --seed 4001 --device cuda:0
python -m chr_training.evaluate --run outputs/cifar100_20/seed4001 --device cuda:0
```

| Configuration | Tasks | Seeds |
| --- | ---: | --- |
| `configs/cifar100_20.json` | 20 | 4001, 4002, 4003 |
| `configs/imagenet_r_20.json` | 20 | 5001, 5002, 5003 |
| `configs/cub200_20.json` | 20 | 6001, 6002, 6003 |
| `configs/imagenet_r_50.json` | 50 | 5101, 5102, 5103 |

Mean and Joint CHR share one training trajectory. Use `--methods mean joint current_only sldc_linear sldc` to include the controls. Set paths with `--data-root`, `--checkpoint`, and `--output`; continue a run with `--resume`.

`--smoke` runs three tasks with one epoch and four images per class, saving to a separate `_smoke` directory. Evaluation writes per-task results and AIA/Final to `evaluation.csv` and `evaluation.json`.

## Inference

```bash
python -m chr_training.predict --run outputs/cifar100_20/seed4001 --image example.jpg --method joint
```

## Tests

```bash
python -m unittest discover -s tests -v
```

[Code map](docs/METHOD.md) · [Acknowledgments](docs/PROVENANCE.md)
