# Mask to Adapt (M2A)

This directory contains the **test-time adaptation (TTA)** code used for the M2A (Mask to Adapt) experiments on **ImageNet-C** and related robustness benchmarks.

## 1. Environment & Requirements

The recommended way to set up the environment for this submission is via the provided **Conda environment file**:

```bash
conda create -n m2a python=3.8
conda activate m2a
```

```bash
# for AMD ROCm PyTorch
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/rocm5.6
pip install -r requirements.txt

# for NVIDIA CUDA PyTorch
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The provided `requirements.txt` already includes the necessary versions and fetches RobustBench & AutoAttack from GitHub.

To validate the installation you can run (inside the activated env):

```bash
python -c "import torch, numpy, timm, robustbench; print(torch.__version__, numpy.__version__)"
```

If imports succeed without errors, the environment is ready for the scripts under `m2a/classification/`.

## 2. Data: ImageNet-C

This submission evaluates M2A and other TTA methods on **ImageNet-C** using the runners in `classification/`.

### 2.1. Download ImageNet-C

Download **ImageNet-C** corruptions from the official release:

   - ImageNet-C: https://zenodo.org/records/2235448

Place the data under a directory of your choice (referred to here as `/path/to/datasets/imagenet`). A typical layout, matching this repository, is:

```text
/path/to/datasets/imagenet/
  ImageNet-C/                # ImageNet-C corruptions from the Zenodo archive
```

The exact folder names must match those expected by RobustBench / ImageNet-C; the provided loaders in `classification/datasets/` assume the standard ImageNet-C directory structure used in the official release.

### 2.2. Pointing the code to ImageNet-C

The classification code uses a config field `DATA_DIR` (see `classification/conf.py` and YAML files under `classification/cfgs/`) to locate datasets.

You can set this in one of two ways:

- **Via config YAML**: edit the corresponding config file (e.g. `classification/cfgs/imagenet_c/m2a.yaml`) and set:

  ```yaml
  DATA_DIR: /path/to/datasets/imagenet
  ```

- **Via command-line override** (if desired):

  ```bash
  python test_time.py \
      --cfg cfgs/imagenet_c/m2a.yaml \
      DATA_DIR /path/to/datasets/imagenet \
      CORRUPTION.DATASET imagenet_c \
      CORRUPTION.NUM_EX 5000
  ```

In both cases, the ImageNet-C loaders under `classification/datasets/` will read from `DATA_DIR` and expect the ImageNet-C directory to be under that root.

## 3. Running ImageNet-C Experiments (M2A)

All commands below assume you have activated the environment and are inside this submission directory.

```bash
conda activate m2a   # or your custom env
cd m2a/classification
```

### 3.1. Direct Python call

A minimal example to run **M2A** on ImageNet-C is:

```bash
cd classification
python test_time.py \
      --cfg cfgs/imagenet_c/m2a.yaml \
      DATA_DIR "/path/to/datasets/imagenet" \
      TEST.BATCH_SIZE 64 \
      OPTIM.STEPS 1 \
      CORRUPTION.DATASET imagenet_c \
      CORRUPTION.NUM_EX 5000 \
      RNG_SEED 1 \
      M2A.SEED 1 \
      M2A.M 0.1 \
      M2A.N 3 \
      M2A.NUM_SQUARES 1 \
      M2A.LAMBDA_EML 1.0 \
      M2A.RANDOM_MASKING spatial \
      M2A.SPATIAL_TYPE patch \
      M2A.SPECTRAL_TYPE low \
      M2A.DISABLE_MCL False \
      M2A.DISABLE_EML False \
```

This mirrors the configuration used in the included helper script and assumes that `DATA_DIR` is correctly set in the YAML or overridden on the command line.

The script calls `test_time.py` with the M2A-specific hyperparameters for ImageNet-C.

## 4. Repository Layout (Submission)

Key directories under `m2a/`:

- `classification/`
  - `conf.py` – configuration system (model, optimization, corruption, testing).
  - `cfgs/` – YAML configs (e.g. `imagenet_c/m2a.yaml`).
  - `datasets/` – dataset and corruption loaders for ImageNet-C, CIFAR-C, DomainNet, etc.
  - `methods/` – TTA methods including **M2A** (`m2a.py`) and baselines (TENT, CoTTA, REM, etc.).
  - `utils/` – evaluation utilities, losses, registries, CDC helpers.
  - `test_time.py` – main entrypoint for test-time adaptation evaluation.
  - `scripts/` – helper shell scripts for running experiments.

## 5. Acknowledgements

This submission builds upon and is inspired by the following works and repositories (see the original project for full details):

- ROID – [official](https://github.com/mariodoebler/test-time-adaptation)
- REM – [official](https://github.com/pilsHan/rem)
- CoTTA – [official](https://github.com/qinenergy/cotta)
- ViDA – [official](https://github.com/Yangsenqiao/vida)
- Continual-MAE – [official](https://github.com/RanXu2000/continual-mae)
- MaskedKD – [official](https://github.com/effl-lab/MaskedKD)
- KATANA – [official](https://github.com/giladcohen/KATANA)
- RobustBench – [official](https://github.com/RobustBench/robustbench)