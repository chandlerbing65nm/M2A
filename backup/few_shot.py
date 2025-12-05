from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from robustbench.model_zoo.vit import create_model as rb_create_model

from train import (
    DEFAULT_TRAIN_ROOT,
    DEFAULT_VAL_ROOT,
    DEFAULT_CLASS_MAP,
    RemappedImageFolder,
    build_transforms,
    set_seed_all,
    worker_init_fn,
    train_one_epoch,
    validate,
)


FEWSHOT_CKPT_DIR = "/flash/project_465002264/projects/m2a/ckpt/shot"
MODEL_ARCH = "vit_base_patch16_224"
MODEL_ARCH_NAME = "vitb16_224"
DEFAULT_DATASET_NAME = "mrsffia"


def parse_shot_string(s: str) -> List[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    shots: List[int] = []
    for p in parts:
        try:
            v = int(p)
        except ValueError:
            continue
        if v <= 0:
            continue
        if v not in shots:
            shots.append(v)
    if not shots:
        raise ValueError("Invalid --shot argument: must contain at least one positive integer, e.g. '5,10' or '1,2,4'.")
    return shots


def parse_args():
    parser = argparse.ArgumentParser("MRSFFIA ViT-B/16 few-shot fine-tuning")
    parser.add_argument("--train-root", type=str, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--val-root", type=str, default=DEFAULT_VAL_ROOT)
    parser.add_argument("--class-map-path", type=str, default=DEFAULT_CLASS_MAP,
                        help="Path to JSON mapping {class_name: class_id}")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to base checkpoint to fine-tune from")

    parser.add_argument("--epoch", type=int, default=50,
                        help="Maximum number of fine-tuning epochs")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience in epochs (0 disables early stopping)")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Mini-batch size for few-shot training and validation")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay for optimizer")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision (AMP)")

    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_NAME,
                        help="Dataset name used in checkpoint filename")

    parser.add_argument("--shot", type=str, default="5,10",
                        help="Comma-separated shot counts to run sequentially, e.g. '5,10' or '1,2,4' (order matters)")

    return parser.parse_args()


def build_full_datasets(args):
    """Build full train/val datasets using the same transforms and mapping as train.py."""
    train_tf, val_tf = build_transforms(None)

    train_base = datasets.ImageFolder(args.train_root, transform=train_tf)
    val_base = datasets.ImageFolder(args.val_root, transform=val_tf)

    with open(args.class_map_path, "r") as f:
        class_map = json.load(f)
    class_map = {str(k): int(v) for k, v in class_map.items()}

    train_ds_full = RemappedImageFolder(train_base, class_map)
    val_ds = RemappedImageFolder(val_base, class_map)
    num_classes = train_ds_full.num_classes
    return train_ds_full, val_ds, num_classes


def build_few_shot_indices(train_ds_full, num_classes: int, shots: List[int], rng: random.Random) -> Dict[int, List[int]]:
    """Return per-shot indices ensuring larger shots are supersets of smaller ones per class."""
    shots = sorted(list(set(int(s) for s in shots)))
    max_shot = max(shots)

    indices_by_class: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    # RemappedImageFolder precomputes remapped_targets
    for idx, y in enumerate(train_ds_full.remapped_targets):
        indices_by_class[int(y)].append(idx)

    per_class_selection: Dict[int, List[int]] = {}
    for c in range(num_classes):
        candidates = indices_by_class[c]
        if len(candidates) < max_shot:
            raise ValueError(
                f"Not enough samples for class {c}: required {max_shot}, found {len(candidates)}"
            )
        perm = list(candidates)
        rng.shuffle(perm)
        per_class_selection[c] = perm[:max_shot]

    shot_to_indices: Dict[int, List[int]] = {}
    for shot in shots:
        current_indices: List[int] = []
        for c in range(num_classes):
            current_indices.extend(per_class_selection[c][:shot])
        rng.shuffle(current_indices)
        shot_to_indices[shot] = current_indices

    return shot_to_indices


def build_model(num_classes: int, ckpt_path: str, device: torch.device) -> nn.Module:
    """Create ViT-B/16 model and load weights from a checkpoint instead of ImageNet."""
    model = rb_create_model(
        MODEL_ARCH,
        pretrained=False,
        num_classes=num_classes,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    return model


def build_shot_checkpoint_path(shot: int, model_arch_name: str, dataset_name: str, ckpt_path: str) -> str:
    ckpt_basename = os.path.basename(ckpt_path)
    ckpt_stem, _ = os.path.splitext(ckpt_basename)
    ckpt_firstword = ckpt_stem.split("_")[0] if ckpt_stem else ckpt_stem
    filename = f"{shot}_{model_arch_name}_{dataset_name}_{ckpt_firstword}_best.pth"
    return os.path.join(FEWSHOT_CKPT_DIR, filename)


def run_few_shot_experiment(
    shot: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    args,
    device: torch.device,
    amp_enabled: bool,
    ckpt_path: str,
    model_arch_name: str,
    dataset_name: str,
) -> None:
    model = build_model(num_classes, ckpt_path, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_top1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(args.epoch):
        train_loss, train_top1, train_map1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            args.epoch,
            num_classes,
        )

        val_loss, val_top1, val_map1 = validate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            args.epoch,
            amp_enabled,
            num_classes,
        )

        print(
            f"[{shot}-shot] Epoch {epoch + 1}/{args.epoch}: "
            f"train_loss={train_loss:.4f} train_top1={train_top1:.2f}% train_mAP@1={train_map1:.2f}% | "
            f"val_loss={val_loss:.4f} val_top1={val_top1:.2f}% val_mAP@1={val_map1:.2f}%"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            epochs_no_improve = 0
            best_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "shot": shot,
                "best_val_top1": best_val_top1,
            }
        else:
            epochs_no_improve += 1

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(
                f"[{shot}-shot] Early stopping triggered: no val_top1 improvement for "
                f"{args.patience} epoch(s). Best val_top1={best_val_top1:.2f}%."
            )
            break

    if best_state is None:
        best_state = {
            "epoch": args.epoch - 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "shot": shot,
            "best_val_top1": best_val_top1,
        }

    os.makedirs(FEWSHOT_CKPT_DIR, exist_ok=True)
    out_path = build_shot_checkpoint_path(shot, model_arch_name, dataset_name, ckpt_path)
    torch.save(best_state, out_path)
    print(f"[{shot}-shot] Saved best model to {out_path} with val_top1={best_val_top1:.2f}%")


def main():
    args = parse_args()

    set_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp

    train_ds_full, val_ds, num_classes = build_full_datasets(args)
    print(
        f"Loaded train set with {len(train_ds_full)} images, "
        f"val set with {len(val_ds)} images, num_classes={num_classes}."
    )

    rng = random.Random(args.seed)
    shots = parse_shot_string(args.shot)
    shot_to_indices = build_few_shot_indices(train_ds_full, num_classes, shots, rng)

    # Shared val loader (full test set)
    g_val = torch.Generator()
    g_val.manual_seed(args.seed)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=g_val,
        persistent_workers=args.num_workers > 0,
    )

    for shot in shots:
        indices = shot_to_indices[shot]
        train_subset = Subset(train_ds_full, indices)

        g_train = torch.Generator()
        g_train.manual_seed(args.seed)
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            worker_init_fn=worker_init_fn,
            generator=g_train,
            persistent_workers=args.num_workers > 0,
        )

        print(f"Starting {shot}-shot fine-tuning with {len(train_subset)} images.")

        run_few_shot_experiment(
            shot=shot,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            args=args,
            device=device,
            amp_enabled=amp_enabled,
            ckpt_path=args.ckpt,
            model_arch_name=MODEL_ARCH_NAME,
            dataset_name=args.dataset,
        )


if __name__ == "__main__":
    main()
