
# M2A: Mask to Adapt — Simple Random Masking Surprisingly Enables Robust Continual Test-Time Learning

This repository contains code for M2A (Mask to Adapt).
  
The CIFAR runners implementing M2A are:
- `cifar/cifar10c_vit_m2a.py`
- `cifar/cifar100c_vit_m2a.py`

## Installation

See INSTALL.md for detailed environment setup:
- [INSTALL.md](INSTALL.md)

Quick summary (ROCm example is in INSTALL.md): create a conda env, install PyTorch/torchvision, then install project requirements from `requirements.txt`.

## Data: CIFAR-C

Download the CIFAR-C datasets and note the directory you place them in (pass as `--data_dir` when running):
- CIFAR-10-C: https://zenodo.org/records/2535967
- CIFAR-100-C: https://zenodo.org/records/3555552

RobustBench loaders in `cifar/cifar10c_vit_m2a.py` and `cifar/cifar100c_vit_m2a.py` implement M2A and will read from `--data_dir`.

## CIFAR Experiments

Below is a minimal setup and the exact commands to reproduce M2A on CIFAR-10-C and CIFAR-100-C.

### Environment setup

```bash
conda init
conda activate rem
cd M2A
```

If you are inside the repository root, the `cifar/` folder is at `M2A/cifar/`.

### CIFAR-10 → CIFAR-10-C

Run the following from inside the `cifar/` directory (so that paths like `cfgs/...` resolve):

```bash
python -m cifar10c_vit_m2a \
     --cfg cfgs/cifar10/m2a.yaml \
     --data_dir data_path \
     --patch_size 8 \
     --lr 0.001 \
     --lamb 1.0 \
     --margin 0.0 \
     --random_masking \
     --num_squares 1 \
     --mask_type binary \
     --m 0.10 --n 3 \
     --logm2a_enable beta \
     --logm2a_lr_mult 5.0 \
```

### CIFAR-100 → CIFAR-100-C

```bash
python -m cifar100c_vit_m2a \
     --cfg cfgs/cifar100/m2a.yaml \
     --data_dir data_path \
     --patch_size 8 \
     --lr 0.0001 \
     --lamb 1.0 \
     --margin 0.0 \
     --random_masking \
     --num_squares 1 \
     --mask_type binary \
     --m 0.10 --n 3 \
     --logm2a_enable beta \
     --logm2a_lr_mult 5.0 \
```

### Notes

- Checkpoints:
  - CIFAR-10: `cifar/cifar10c_vit_m2a.py` loads a ViT checkpoint from `/users/doloriel/work/Repo/M2A/ckpt/vit_base_384_cifar10.t7`.
  - CIFAR-100: `cifar/cifar100c_vit_m2a.py` loads a checkpoint from `/users/doloriel/work/Repo/M2A/ckpt/pretrain_cifar100.t7`.
  - If your checkpoints are elsewhere, update those paths in the scripts or place the files accordingly.
- Input size and patch size:
  - The default input resize is `--size 384` (see `cifar/conf.py`). If using M2A masking, the input size must be divisible by `--patch_size` (e.g., 384 divisible by 8).
- Config knobs:
  - YAMLs under `cifar/cfgs/cifar10/m2a.yaml` and `cifar/cfgs/cifar100/m2a.yaml` set defaults for learning rate, masking schedule (`m`, `n`), and M2A options. CLI flags override the YAML.

## Acknowledgements

This codebase builds upon and was inspired by the following works and repositories:

+ REM [official](https://github.com/pilsHan/rem)
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ ViDA [official](https://github.com/Yangsenqiao/vida)
+ Continual-MAE [official](https://github.com/RanXu2000/continual-mae)
+ MaskedKD [official](https://github.com/effl-lab/MaskedKD)
+ KATANA [official](https://github.com/giladcohen/KATANA) 
+ Robustbench [official](https://github.com/RobustBench/robustbench) 