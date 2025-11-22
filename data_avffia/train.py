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
  train root: /scratch/project_465002264/datasets/avffia/AVFFIA-C/clean/train
  val root:   /scratch/project_465002264/datasets/avffia/AVFFIA-C/clean/test
  ckpt dir:   /users/doloriel/work/Repo/SPARC/ckpt

Usage example:
  python avffia/train.py \
    --epochs 50 --batch-size 128 --lr 5e-4 --weight-decay 0.05 --num-workers 8 \
    --class-map-path avffia/robustbench/data/avffia_class_to_id_map.json
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

import timm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_TRAIN_ROOT = \
    "/scratch/project_465002264/datasets/avffia/AVFFIA-C/clean/train"
DEFAULT_VAL_ROOT = \
    "/scratch/project_465002264/datasets/avffia/AVFFIA-C/clean/test"
DEFAULT_CKPT_DIR = \
    "/users/doloriel/work/Repo/SPARC/ckpt"
DEFAULT_CLASS_MAP = \
    "/users/doloriel/work/Repo/SPARC/avffia/robustbench/data/avffia_class_to_id_map.json"


@dataclass
class TrainConfig:
    train_root: str = DEFAULT_TRAIN_ROOT
    val_root: str = DEFAULT_VAL_ROOT
    ckpt_dir: str = DEFAULT_CKPT_DIR
    class_map_path: str = DEFAULT_CLASS_MAP
    epochs: int = 50
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 0.05
    num_workers: int = 8
    resume: Optional[str] = None
    seed: int = 1
    # Augmentation (RandAugment) and regularization
    use_randaugment: bool = True
    rand_n: int = 2
    rand_m: int = 9
    random_erasing: float = 0.0
    # Model regularization knobs
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    # Scheduler warmup
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.01
    # Label smoothing
    label_smoothing: float = 0.1
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
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if cfg.use_randaugment:
        try:
            train_list.append(transforms.RandAugment(num_ops=cfg.rand_n, magnitude=cfg.rand_m))
        except TypeError:
            train_list.append(transforms.RandAugment(cfg.rand_n, cfg.rand_m))
    train_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    if cfg.random_erasing and cfg.random_erasing > 0.0:
        train_list.append(transforms.RandomErasing(p=cfg.random_erasing, scale=(0.33, 0.5), ratio=(0.3, 3.3), value='random'))

    train_tf = transforms.Compose(train_list)
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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
    # Load ViT-B/16 with ImageNet pretrained weights, set classifier to num_classes
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
    )
    return model


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
    path = os.path.join(ckpt_dir, f"avffia_vitb16_{tag}.pth")
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
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=8)
    # Augmentations and regularization knobs
    p.add_argument("--use-randaugment", action="store_true", help="Enable RandAugment in training transforms")
    p.add_argument("--rand-n", type=int, default=2, help="RandAugment num_ops")
    p.add_argument("--rand-m", type=int, default=9, help="RandAugment magnitude")
    p.add_argument("--random-erasing", type=float, default=0.0, help="RandomErasing probability (0 to disable)")
    p.add_argument("--drop-rate", type=float, default=0.0, help="Model dropout rate")
    p.add_argument("--drop-path-rate", type=float, default=0.1, help="Stochastic depth rate")
    # Scheduler warmup and label smoothing
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--warmup-start-factor", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision (AMP)")

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
        use_randaugment=bool(args.use_randaugment),
        rand_n=int(args.rand_n),
        rand_m=int(args.rand_m),
        random_erasing=float(args.random_erasing),
        drop_rate=float(args.drop_rate),
        drop_path_rate=float(args.drop_path_rate),
        warmup_epochs=int(args.warmup_epochs),
        warmup_start_factor=float(args.warmup_start_factor),
        label_smoothing=float(args.label_smoothing),
        no_amp=bool(args.no_amp),
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
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Warmup + Cosine scheduler
    total_main_epochs = max(1, cfg.epochs - max(0, cfg.warmup_epochs))
    sched_cos = CosineAnnealingLR(optimizer, T_max=total_main_epochs)
    if cfg.warmup_epochs > 0:
        sched_wu = LinearLR(optimizer, start_factor=cfg.warmup_start_factor, end_factor=1.0, total_iters=cfg.warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[sched_wu, sched_cos], milestones=[cfg.warmup_epochs])
    else:
        scheduler = sched_cos

    amp_enabled = (device.type == "cuda" and not cfg.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # Optionally resume
    start_epoch = load_checkpoint_if_any(model, optimizer, scheduler, cfg.resume)

    # Train loop
    best_top1 = -1.0
    for epoch in range(start_epoch, cfg.epochs):
        train_loss, train_top1, train_map1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, cfg.epochs, num_classes
        )
        val_loss, val_top1, val_map1 = validate(
            model, val_loader, criterion, device, epoch, cfg.epochs, amp_enabled, num_classes
        )

        scheduler.step()

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": cfg.__dict__,
        }
        # Save last every epoch
        save_checkpoint(state, cfg.ckpt_dir, tag="last")
        # Save best if improved
        if val_top1 > best_top1:
            best_top1 = val_top1
            state_best = dict(state)
            state_best["best_top1"] = best_top1
            save_checkpoint(state_best, cfg.ckpt_dir, tag="best")

        print(
            f"Epoch {epoch+1}/{cfg.epochs}: "
            f"train_loss={train_loss:.4f} train_top1={train_top1:.2f}% train_mAP@1={train_map1:.2f}% | "
            f"val_loss={val_loss:.4f} val_top1={val_top1:.2f}% val_mAP@1={val_map1:.2f}%"
        )


if __name__ == "__main__":
    main()
