"""
AVFFIA ViT-B/16 fine-tuning script

Pip requirements (minimum):
  pip install timm torch torchvision tqdm

- Finetunes ViT-B/16 (224) with ImageNet-1k pretrained weights (timm).
- Uses EXACT ImageNet preprocessing for fine-tuning:
    Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet mean/std)
- Expects ImageFolder directory structure: /path/class_name/*.JPEG
- Remaps ImageFolder targets via a JSON class map: {"class_name": class_id}
- Mixed-precision (AMP) training with torch.cuda.amp (ROCm-compatible PyTorch exposes CUDA APIs).
- Saves checkpoints each epoch: model + optimizer + scheduler + epoch + args

Default paths:
  train root: /flash/project_465002264/datasets/avffia/AVFFIA-C/clean/train
  val root:   /flash/project_465002264/datasets/avffia/AVFFIA-C/clean/test
  ckpt dir:   /users/doloriel/work/Repo/M2A/ckpt

Usage example:
python data_avffia/train.py --epochs 50 --batch-size 128 --lr 1e-3 --weight-decay 0.05 --num-workers 4 --use-randaugment

"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import sys
import shlex

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from robustbench.model_zoo.vit import create_model as rb_create_model

import timm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_TRAIN_ROOT = \
    "/flash/project_465002264/datasets/avffia/split/clean/train"
DEFAULT_VAL_ROOT = \
    "/flash/project_465002264/datasets/avffia/split/clean/test"
DEFAULT_CKPT_DIR = \
    "/users/doloriel/work/Repo/M2A/ckpt"
DEFAULT_CLASS_MAP = \
    "data_avffia/avffia_class_to_id_map.json"


@dataclass
class TrainConfig:
    train_root: str = DEFAULT_TRAIN_ROOT
    val_root: str = DEFAULT_VAL_ROOT
    ckpt_dir: str = DEFAULT_CKPT_DIR
    class_map_path: str = DEFAULT_CLASS_MAP
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.05
    num_workers: int = 4
    resume: Optional[str] = None
    seed: int = 1
    patience: int = 0
    tag: Optional[str] = None
    trainable: Optional[str] = None
    # Augmentation (RandAugment) and regularization
    use_randaugment: bool = True
    rand_n: int = 2
    rand_m: int = 9
    random_erasing: float = 0.0
    # Model regularization knobs (disabled)
    # AMP control
    no_amp: bool = False


class RemappedImageFolder(Dataset):
    """Wraps an ImageFolder to remap its target labels according to a provided mapping.

    mapping: dict[class_name] -> class_id
    """
    def __init__(self, base: datasets.ImageFolder, mapping: Dict[str, int]):
        self.base = base
        self.mapping = dict(mapping)
        # Validate mapping covers all classes in dataset
        missing = [c for c in self.base.class_to_idx.keys() if c not in self.mapping]
        if missing:
            raise ValueError(
                f"Class mapping missing entries for dataset classes: {missing[:5]}" +
                ("..." if len(missing) > 5 else "")
            )
        # Precompute remapped targets for __len__ fast path
        self.remapped_targets = [self.mapping[self.base.classes[idx]] for idx in self.base.targets]
        # Determine number of classes from mapping (max id + 1 or unique count)
        self.num_classes = max(self.mapping.values()) + 1 if self.mapping else 0

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        img, orig_target = self.base[idx]
        # orig_target is the ImageFolder index; map via class name
        class_name = self.base.classes[orig_target]
        target = self.mapping[class_name]
        return img, target


def build_transforms(cfg: TrainConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    """ImageNet-style transforms with optional RandAugment and RandomErasing for training.
    Train: Resize->CenterCrop->(Flip)->(RandAugment)->ToTensor->Normalize->(RandomErasing)
    Val:   Resize->CenterCrop->ToTensor->Normalize
    """
    train_list = [
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        # transforms.Resize(64),
        # transforms.Resize(256),
        transforms.RandomResizedCrop(384),
        transforms.RandomHorizontalFlip(),
    ]
    train_list.extend([
        transforms.ToTensor(),
        # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_tf = transforms.Compose(train_list)
    val_tf = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, val_tf


def set_seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def worker_init_fn(worker_id: int):
    # Ensure deterministic dataloader workers
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    try:
        import numpy as np
        np.random.seed((seed + worker_id) % 2**32)
    except Exception:
        pass


def build_dataloaders(cfg: TrainConfig) -> Tuple[Dataset, Dataset, DataLoader, DataLoader, int]:
    train_tf, val_tf = build_transforms(cfg)

    train_base = datasets.ImageFolder(cfg.train_root, transform=train_tf)
    val_base = datasets.ImageFolder(cfg.val_root, transform=val_tf)

    with open(cfg.class_map_path, "r") as f:
        class_map = json.load(f)
        # Ensure keys are strings
        class_map = {str(k): int(v) for k, v in class_map.items()}

    train_ds = RemappedImageFolder(train_base, class_map)
    val_ds = RemappedImageFolder(val_base, class_map)

    num_classes = train_ds.num_classes

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=g,
        persistent_workers=cfg.num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=g,
        persistent_workers=cfg.num_workers > 0,
    )

    return train_ds, val_ds, train_loader, val_loader, num_classes


def build_model(num_classes: int, cfg: TrainConfig) -> nn.Module:
    model = rb_create_model(
        "vit_base_patch16_384",
        pretrained=True,
        num_classes=num_classes,
    )
    return model


def select_trainable_params(model: nn.Module, trainable_spec: Optional[str]):
    if not trainable_spec:
        return list(model.parameters())
    try:
        parts = [s.strip() for s in str(trainable_spec).split(',') if s.strip()]
        indices = set(int(p) for p in parts)
    except Exception:
        return list(model.parameters())
    try:
        model.requires_grad_(False)
        if hasattr(model, 'blocks') and isinstance(model.blocks, torch.nn.ModuleList) or True:
            try:
                n_blocks = len(model.blocks)
            except Exception:
                n_blocks = 0
            for i in range(n_blocks):
                if i in indices:
                    for p in model.blocks[i].parameters():
                        p.requires_grad = True
        if (-1 in indices) and hasattr(model, 'head'):
            for p in model.head.parameters():
                p.requires_grad = True
    except Exception:
        return list(model.parameters())
    params = [p for p in model.parameters() if getattr(p, 'requires_grad', False)]
    return params if params else list(model.parameters())


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        num_classes = output.size(1)
        # Clamp requested ks to valid range [1, num_classes]
        clamped_topk = tuple(max(1, min(int(k), num_classes)) for k in topk)
        maxk = max(clamped_topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in clamped_topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state: dict, ckpt_dir: str, tag: str = "last"):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"avffia_vitb16_384_{tag}.pth")
    torch.save(state, path)


def load_checkpoint_if_any(model: nn.Module, optimizer, scheduler, resume_path: Optional[str]) -> int:
    if resume_path is None or not os.path.isfile(resume_path):
        return 0
    ckpt = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(ckpt["model"]) if "model" in ckpt else model.load_state_dict(ckpt)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    return start_epoch


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch: int, epochs: int, num_classes: int):
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    count = 0
    pred_counts = [0] * int(num_classes)
    correct_pred_counts = [0] * int(num_classes)

    pbar = tqdm(loader, desc=f"Train epoch {epoch+1}/{epochs}")
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        top1 = accuracy(outputs, targets, topk=(1,))[0].item()
        running_loss += loss.item() * images.size(0)
        running_top1 += top1 * images.size(0) / 100.0
        count += images.size(0)

        # Update precision@1 counts per predicted class
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            for p, t in zip(preds.view(-1).tolist(), targets.view(-1).tolist()):
                pred_counts[p] += 1
                if p == t:
                    correct_pred_counts[p] += 1

        avg_loss = running_loss / max(1, count)
        avg_top1 = 100.0 * (running_top1 / max(1, count))
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "top1": f"{avg_top1:.2f}%"})

    # Compute macro precision@1 (mAP@1)
    per_class_prec = [ (correct_pred_counts[c] / pred_counts[c]) if pred_counts[c] > 0 else 0.0 for c in range(int(num_classes)) ]
    map_at1 = 100.0 * (sum(per_class_prec) / float(max(1, int(num_classes))))
    return running_loss / max(1, count), 100.0 * (running_top1 / max(1, count)), map_at1


def validate(model, loader, criterion, device, epoch: int, epochs: int, amp_enabled: bool, num_classes: int):
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    count = 0
    pred_counts = [0] * int(num_classes)
    correct_pred_counts = [0] * int(num_classes)

    pbar = tqdm(loader, desc=f"Val epoch {epoch+1}/{epochs}")
    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, targets)
            top1 = accuracy(outputs, targets, topk=(1,))[0].item()
            running_loss += loss.item() * images.size(0)
            running_top1 += top1 * images.size(0) / 100.0
            count += images.size(0)

            # Update precision@1 counts per predicted class
            preds = outputs.argmax(dim=1)
            for p, t in zip(preds.view(-1).tolist(), targets.view(-1).tolist()):
                pred_counts[p] += 1
                if p == t:
                    correct_pred_counts[p] += 1

            avg_loss = running_loss / max(1, count)
            avg_top1 = 100.0 * (running_top1 / max(1, count))
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "top1": f"{avg_top1:.2f}%"})

    per_class_prec = [ (correct_pred_counts[c] / pred_counts[c]) if pred_counts[c] > 0 else 0.0 for c in range(int(num_classes)) ]
    map_at1 = 100.0 * (sum(per_class_prec) / float(max(1, int(num_classes))))
    return (
        running_loss / max(1, count),
        100.0 * (running_top1 / max(1, count)),
        map_at1,
    )


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser("AVFFIA ViT-B/16 fine-tuning")
    p.add_argument("--train-root", type=str, default=DEFAULT_TRAIN_ROOT)
    p.add_argument("--val-root", type=str, default=DEFAULT_VAL_ROOT)
    p.add_argument("--ckpt-dir", type=str, default=DEFAULT_CKPT_DIR)

    p.add_argument("--class-map-path", type=str, default=DEFAULT_CLASS_MAP,
                   help="Path to JSON mapping {class_name: class_id}")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.00)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--trainable", type=str, default=None,
                   help="Comma-separated indices of ViT blocks to train; use -1 to also train the head. Example: '9,10,11,-1'.")
    # Augmentations and regularization knobs

    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--seed", type=int, default=1)

    args = p.parse_args()
    return TrainConfig(
        train_root=args.train_root,
        val_root=args.val_root,
        ckpt_dir=args.ckpt_dir,
        class_map_path=args.class_map_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        resume=args.resume,
        seed=args.seed,
        patience=int(args.patience),
        tag=args.tag,
        trainable=args.trainable,
        no_amp=False,
    )


def main():
    cfg = parse_args()
    # Print the exact command used to launch training
    try:
        script_path = os.path.abspath(sys.argv[0]) if sys.argv else "train.py"
        args_joined = " ".join(shlex.quote(a) for a in sys.argv[1:])
        cmd = f"python {shlex.quote(script_path)}"
        if args_joined:
            cmd = f"{cmd} {args_joined}"
        print(cmd)
    except Exception:
        pass

    set_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_ds, val_ds, train_loader, val_loader, num_classes = build_dataloaders(cfg)

    # Model
    model = build_model(num_classes, cfg)
    model.to(device)

    # Optim / Loss / Sched
    criterion = nn.CrossEntropyLoss()
    opt_params = select_trainable_params(model, cfg.trainable)
    optimizer = optim.Adam(opt_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    amp_enabled = (device.type == "cuda" and not cfg.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # Optionally resume
    start_epoch = load_checkpoint_if_any(model, optimizer, None, cfg.resume)

    # Train loop
    best_top1 = -1.0
    epochs_no_improve = 0
    for epoch in range(start_epoch, cfg.epochs):
        train_loss, train_top1, train_map1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, cfg.epochs, num_classes
        )
        val_loss, val_top1, val_map1 = validate(
            model, val_loader, criterion, device, epoch, cfg.epochs, amp_enabled, num_classes
        )

        # No scheduler

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": cfg.__dict__,
        }
        # Save last every epoch
        if val_top1 > best_top1:
            best_top1 = val_top1
            epochs_no_improve = 0
            state_best = dict(state)
            state_best["best_top1"] = best_top1
            save_checkpoint(state_best, cfg.ckpt_dir, tag=("best" + (f"_{cfg.tag}" if cfg.tag else "")))
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch+1}/{cfg.epochs}: "
            f"train_loss={train_loss:.4f} train_top1={train_top1:.2f}% train_mAP@1={train_map1:.2f}% | "
            f"val_loss={val_loss:.4f} val_top1={val_top1:.2f}% val_mAP@1={val_map1:.2f}%"
        )

        if cfg.patience > 0 and epochs_no_improve >= cfg.patience:
            print(f"Early stopping triggered: no val_top1 improvement for {cfg.patience} epoch(s). Best val_top1={best_top1:.2f}%.")
            break


if __name__ == "__main__":
    main()
