import argparse
import os
import random
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple

from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from robustbench.data import load_cifar10c, load_cifar100c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.model_zoo.cifar10 import cifar_10_models
from robustbench.model_zoo.cifar100 import cifar_100_models


METHOD_TO_ARCH = {
    'source': 'Standard_VITB',
    'tent': 'Standard_VITB',
    'cotta': 'Standard_VITB',
    'continual_mae': 'Standard_VITB_MAE',
    'rem': 'Standard_VITB_REM',
    'm2a': 'Standard_VITB_M2A',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Extract penultimate ViT features from CIFAR-10-C / CIFAR-100-C using '
            'TTA-adapted checkpoints and visualize them with t-SNE.'
        )
    )
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint file containing adapted ViT weights (saved with {"model": state_dict}).')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10c', 'cifar100c'],
                        help='Dataset to use: cifar10c or cifar100c.')
    parser.add_argument('--outdir', type=str, default='data_cifar/plots/features',
                        help='Output directory for t-SNE plots (default: data_cifar/plots/features).')
    parser.add_argument('--method', type=str, required=True,
                        choices=['source', 'tent', 'cotta', 'continual_mae', 'rem', 'm2a'],
                        help='Adaptation method used to produce the checkpoint.')
    parser.add_argument('--severity', type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help='Corruption severity level (1-5).')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility (default: 1).')
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For reproducible convolutions / cuDNN behaviour (may slow down slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_base_model(dataset: str, method: str, device: torch.device) -> torch.nn.Module:
    method_key = method.lower()
    if method_key not in METHOD_TO_ARCH:
        raise ValueError(f"Unknown method '{method}'. Expected one of {list(METHOD_TO_ARCH.keys())}.")
    arch_name = METHOD_TO_ARCH[method_key]

    if dataset == 'cifar10c':
        models_by_threat = cifar_10_models[ThreatModel.corruptions]
        num_classes = 10
    elif dataset == 'cifar100c':
        models_by_threat = cifar_100_models[ThreatModel.corruptions]
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'.")

    if arch_name not in models_by_threat:
        raise ValueError(
            f"Architecture '{arch_name}' not found for dataset '{dataset}'. "
            f"Available keys: {list(models_by_threat.keys())}"
        )

    model_entry = models_by_threat[arch_name]
    base_model = model_entry['model']()
    base_model.to(device)
    base_model.eval()
    return base_model


def load_tta_weights(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logging.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    # Remove optional 'module.' prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[len('module.'):]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        logging.warning(f"Missing keys when loading state_dict: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys when loading state_dict: {unexpected}")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def forward_penultimate(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return penultimate features from ViT-like models.

    For the ViT variants in this repo (Standard_VITB, *_REM, *_M2A, *_MAE),
    `forward_features` returns the representation before the classifier head
    (CLS token or pre_logits). We use that as the feature vector.
    """
    if hasattr(model, 'forward_features'):
        feats = model.forward_features(x)
        # Some variants may return (features, aux) or (features, attn)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return feats

    # Fallback: register a temporary hook on the classifier head input
    penultimate = []

    def hook_fn(_module, _input, _output):
        # _input is a tuple; we want the tensor being fed into the head
        penultimate.append(_input[0].detach())

    handle = None
    try:
        if hasattr(model, 'head') and isinstance(model.head, torch.nn.Module):
            handle = model.head.register_forward_hook(hook_fn)
        logits = model(x)
        _ = logits  # avoid lints
        if not penultimate:
            raise RuntimeError('Penultimate hook did not capture any features.')
        feats = penultimate[0]
    finally:
        if handle is not None:
            handle.remove()

    return feats


def extract_features(model: torch.nn.Module,
                     x: torch.Tensor,
                     batch_size: int,
                     device: torch.device) -> torch.Tensor:
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats_list = []

    for (xb,) in loader:
        xb = xb.to(device)
        with torch.no_grad():
            feats = forward_penultimate(model, xb)
        feats_list.append(feats.cpu())

    return torch.cat(feats_list, dim=0)


def load_corrupted_data(dataset: str, severity: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Number of examples: CIFAR test sets have 10k examples
    n_examples = 10000

    if dataset == 'cifar10c':
        x, y = load_cifar10c(n_examples=n_examples,
                             severity=severity,
                             data_dir="/scratch/project_465002264/datasets/cifar10c",
                             shuffle=False)
    elif dataset == 'cifar100c':
        x, y = load_cifar100c(n_examples=n_examples,
                              severity=severity,
                              data_dir="/scratch/project_465002264/datasets/cifar100c",
                              shuffle=False)
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'.")

    # Resize to 384x384 to match ViT-B/16-384 input resolution
    x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)

    return x, y


def run_tsne(features: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=30.0,
        learning_rate='auto',
        init='pca',
        random_state=seed,
    )
    emb = tsne.fit_transform(features)

    # Normalize each dimension to [-1, 1]
    min_vals = emb.min(axis=0, keepdims=True)
    max_vals = emb.max(axis=0, keepdims=True)
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0
    emb_norm = 2.0 * (emb - min_vals) / denom - 1.0
    return emb_norm


def plot_tsne(embeddings_2d: np.ndarray,
              labels: np.ndarray,
              dataset: str,
              method: str,
              out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    num_classes = 10 if dataset == 'cifar10c' else 100
    plt.figure(figsize=(8, 8))

    cmap_name = 'tab10' if num_classes <= 10 else 'tab20'
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap=cmap_name,
        s=5,
        alpha=0.7,
    )

    # plt.title(f't-SNE of penultimate features ({dataset}, method={method})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Build model matching dataset+method and load TTA-adapted weights
    model = build_base_model(args.dataset, args.method, device)
    model = load_tta_weights(model, args.ckpt, device)

    # Load corrupted data at specified severity
    logging.info(f'Loading {args.dataset} with severity={args.severity}')
    x, y = load_corrupted_data(args.dataset, args.severity)

    # Extract features
    logging.info('Extracting penultimate features...')
    feats = extract_features(model, x, batch_size=128, device=device)
    feats_np = feats.numpy().astype(np.float32)
    labels_np = y.numpy().astype(np.int64)

    logging.info('Running t-SNE...')
    emb_2d = run_tsne(feats_np, labels_np, seed=args.seed)

    # Save plot with same basename as checkpoint but .jpg extension
    ckpt_basename = os.path.basename(args.ckpt)
    stem, _ = os.path.splitext(ckpt_basename)
    out_fname = stem + '.jpg'
    out_path = os.path.join(args.outdir, out_fname)

    logging.info(f'Saving t-SNE plot to {out_path}')
    plot_tsne(emb_2d, labels_np, args.dataset, args.method, out_path)


if __name__ == '__main__':
    main()
