# Data and weights

Set `data_root` and `checkpoint` in the configuration or pass CLI overrides.

| Dataset | Expected path | Source |
| --- | --- | --- |
| CIFAR-100 | `datasets/cifar-100-python/{train,test,meta}` | [Download](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) |
| ImageNet-R | `datasets/pilot/{train,test}/<class>/*` | [PILOT preprocessed split](https://github.com/LAMDA-CL/LAMDA-PILOT#datasets) |
| CUB-200-2011 | `datasets/CUB_200_2011/` with images and official metadata | [Download](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1) |

ImageNet-R uses the fixed 24,000/6,000 split. CUB uses `train_test_split.txt`. CIFAR evaluation reports both the paper's 9,990-image policy (ten excluded indices are in the config) and the standard test accuracy.

Download the ImageNet-21k ViT-B/16 weights:

```bash
python scripts/download_checkpoint.py
```

Training uses only current-task training images. Evaluation is a separate command.
