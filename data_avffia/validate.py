"""Validate a trained AVFFIA ViT-B/16 (384) checkpoint on the validation set.

This script reuses the data pipeline and model definition from train.py and
loads a checkpoint specified via --ckpt, then evaluates on the validation set.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

from train import (
    TrainConfig,
    build_dataloaders,
    build_model,
    set_seed_all,
    validate as run_validate,
)


def parse_args():
    p = argparse.ArgumentParser("Validate AVFFIA ViT-B/16 (384) checkpoint")
    p.add_argument(
        "--val-root",
        type=str,
        default=TrainConfig().val_root,
        help="Path to validation ImageFolder root",
    )
    p.add_argument(
        "--class-map-path",
        type=str,
        default=TrainConfig().class_map_path,
        help="Path to JSON mapping {class_name: class_id}",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint file saved by train.py",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=TrainConfig().batch_size,
        help="Batch size for validation",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=TrainConfig().num_workers,
        help="Number of DataLoader workers",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=TrainConfig().seed,
        help="RNG seed (matches train.py behavior)",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Build a minimal TrainConfig to reuse build_dataloaders / build_model
    cfg = TrainConfig()
    cfg.train_root = args.val_root
    cfg.val_root = args.val_root
    cfg.class_map_path = args.class_map_path
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.seed = args.seed

    set_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data (we only need val_loader and num_classes)
    _, val_ds, _, val_loader, num_classes = build_dataloaders(cfg)

    # Model
    model = build_model(num_classes, cfg)
    model.to(device)

    # Load checkpoint
    ckpt_path = os.path.expanduser(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)

    criterion = nn.CrossEntropyLoss()
    amp_enabled = (device.type == "cuda")

    val_loss, val_top1, val_map1 = run_validate(
        model,
        val_loader,
        criterion,
        device,
        epoch=0,
        epochs=1,
        amp_enabled=amp_enabled,
        num_classes=num_classes,
    )

    print(
        f"Validation: loss={val_loss:.4f} top1={val_top1:.2f}% mAP@1={val_map1:.2f}%"
    )


if __name__ == "__main__":
    main()
