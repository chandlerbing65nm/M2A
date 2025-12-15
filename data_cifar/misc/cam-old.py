import argparse
import os
import sys
import math
import random
from collections import OrderedDict

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from robustbench.data import load_cifar10c, load_cifar100c, CORRUPTIONS
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model


METHOD_TO_ARCH = {
    "source": "Standard_VITB",
    "tent": "Standard_VITB",
    "cotta": "Standard_VITB",
    "rem": "Standard_VITB_REM",
    "m2a": "Standard_VITB_M2A",
    "continual_mae": "Standard_VITB_MAE",
}


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a ViT attention CAM overlay for a CIFAR-C checkpoint."
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint file to load into the model.")
    parser.add_argument("--method", type=str, required=True,
                        choices=["source", "tent", "cotta", "rem", "m2a", "continual_mae"],
                        help="Adaptation method; used to choose the base ViT architecture.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for CAM JPEG. Defaults to data_cifar/plots/cam.")
    parser.add_argument("--data_format", type=str, default="cifar10c",
                        help="Data format. Supports 'cifar10c' or 'cifar100c'.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducible image selection.")
    parser.add_argument("--severity", type=int, default=5,
                        help="CIFAR-C severity level in [1,5] for the sampled image.")
    parser.add_argument("--attn_only", action="store_true",
                        help="If set, save only the attention/CAM map as a standalone image (no overlay).")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Opacity of the CAM in [0,1]: 0 = only image, 1 = only CAM.")
    parser.add_argument("--class", dest="class_name", type=str, default=None,
                        help="If set (CIFAR-10-C only), sample an image from this class name (e.g., 'airplane').")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_default_out_dir() -> str:
    # This file lives in data_cifar/misc/cam.py
    data_cifar_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(data_cifar_root, "plots", "cam")


def rm_substr_from_state_dict(state_dict, substr: str):
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if key.startswith(substr):
            new_key = key[len(substr):]
            new_state_dict[new_key] = val
        else:
            new_state_dict[key] = val
    return new_state_dict


def load_base_model(method: str, data_format: str) -> torch.nn.Module:
    method_l = method.lower()
    if method_l not in METHOD_TO_ARCH:
        raise ValueError(f"Unsupported method '{method}'. Supported: {sorted(METHOD_TO_ARCH.keys())}")

    arch = METHOD_TO_ARCH[method_l]

    df = data_format.lower()
    if df == "cifar10c":
        dataset = "cifar10"
    elif df == "cifar100c":
        dataset = "cifar100"
    else:
        raise ValueError(f"Unsupported data_format '{data_format}'. Use 'cifar10c' or 'cifar100c'.")

    model = load_model(arch, "./ckpt", dataset, ThreatModel.corruptions)
    return model


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    state_dict = rm_substr_from_state_dict(state_dict, "module.")
    model.load_state_dict(state_dict, strict=True)
    return model


def get_random_sample(data_format: str,
                      seed: int,
                      img_size: int = 384,
                      severity: int = 5,
                      class_name: str = None):
    set_seed(seed)
    if not (1 <= int(severity) <= 5):
        raise ValueError(f"severity must be in [1,5], got {severity}")
    sev = int(severity)
    df = data_format.lower()
    if df == "cifar10c":
        data_dir = "/scratch/project_465002264/datasets/cifar10c"
        x_all, y_all = load_cifar10c(
            n_examples=10000,
            severity=sev,
            data_dir=data_dir,
            shuffle=False,
            corruptions=CORRUPTIONS,
        )
        if class_name is not None:
            cname = class_name.lower()
            if cname not in CIFAR10_CLASSES:
                raise ValueError(f"Unknown CIFAR-10 class name '{class_name}'. Valid: {CIFAR10_CLASSES}")
            target_idx = CIFAR10_CLASSES.index(cname)
            mask = (y_all == target_idx)
            idxs = mask.nonzero(as_tuple=False).squeeze(1)
            if idxs.numel() == 0:
                raise RuntimeError(f"No samples found for class '{class_name}' at severity {sev}.")
            N = idxs.shape[0]
            sel = int(np.random.randint(0, N))
            idx = int(idxs[sel].item())
            x_all = x_all[idx:idx + 1]
            y_all = y_all[idx:idx + 1]
    elif df == "cifar100c":
        if class_name is not None:
            raise ValueError("Class selection by name is only supported for cifar10c.")
        data_dir = "/scratch/project_465002264/datasets/cifar100c"
        x_all, y_all = load_cifar100c(
            n_examples=10000,
            severity=sev,
            data_dir=data_dir,
            shuffle=False,
            corruptions=CORRUPTIONS,
        )
    else:
        raise ValueError(f"Unsupported data_format '{data_format}'.")
    N = x_all.shape[0]
    idx = np.random.randint(0, N)
    x = x_all[idx:idx + 1]
    y = y_all[idx:idx + 1]
    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x, y


def generate_vit_cam(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    """Generate an attention-based CAM from the last ViT block.

    Mirrors the extraction used in utils_trend/m2a_masking_trend_cifar10c.py:
    - take mean attention over heads
    - use CLS-to-patch attentions from the last block
    - reshape to a square grid, upsample to image size, and normalize.
    """
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        out = model(x, return_attn=True)

    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise RuntimeError("Model did not return (logits, attn) when called with return_attn=True.")

    _, attn = out[0], out[1]
    if attn.ndim != 4:
        raise RuntimeError(f"Unexpected attention shape {attn.shape}, expected [B, H, T, T].")

    # Mean over heads, CLS-to-patch attentions from last block
    attn_tokens = attn.mean(dim=1)[:, 0, 1:]  # [B, T-1]
    cam_vec = attn_tokens[0]                  # [T-1]

    T = cam_vec.shape[0]
    token_side = int(round(math.sqrt(T)))
    if token_side * token_side != T:
        token_side = int(math.floor(math.sqrt(T)))
        cam_vec = cam_vec[: token_side * token_side]

    cam_grid = cam_vec.view(1, 1, token_side, token_side)
    cam_up = F.interpolate(cam_grid, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]

    cam = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min() + 1e-8)
    cam = cam.detach().cpu().numpy()
    cam = np.clip(cam, 0.0, 1.0)
    return cam


def save_cam_overlay(x: torch.Tensor, cam: np.ndarray, out_path: str, alpha: float = 0.5) -> None:
    """Save CAM on top of the input image as JPEG.

    ``alpha`` controls CAM opacity directly:
      - alpha = 0: only image
      - alpha = 1: only CAM
    """
    x_np = x.squeeze(0).cpu().numpy()          # [C, H, W]
    x_np = np.transpose(x_np, (1, 2, 0))       # [H, W, C] RGB in [0,1]
    x_np = np.clip(x_np, 0.0, 1.0)
    img_rgb = (x_np * 255.0).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Build heatmap using OpenCV's inferno colormap
    cam_map = np.clip(cam, 0.0, 1.0)
    cam_u8 = (cam_map * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_INFERNO)

    # Blend image and CAM: image weight = 1 - alpha, CAM weight = alpha
    overlay_bgr = cv2.addWeighted(img_bgr, float(1.0 - alpha), heat_bgr, float(alpha), 0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, overlay_bgr)


def save_cam_only(cam: np.ndarray, out_path: str) -> None:
    """Save the CAM alone as a colored JPEG (no underlying image).

    Uses matplotlib's imshow with a high-quality colormap for nicer
    visualization (no axes, tightly cropped).
    """
    cam_map = np.clip(cam, 0.0, 1.0)
    cam_u8 = (cam_map * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_INFERNO)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, heat_bgr)


def main():
    args = parse_args()

    out_dir = args.out_dir if args.out_dir is not None else get_default_out_dir()
    os.makedirs(out_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_base_model(args.method, args.data_format)
    model = load_checkpoint_into_model(model, args.ckpt)
    model.to(device)

    if not (0.0 <= float(args.alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {args.alpha}")

    x, y = get_random_sample(
        args.data_format,
        args.seed,
        img_size=384,
        severity=args.severity,
        class_name=args.class_name,
    )

    if args.method.lower() == "source":
        input_path = os.path.join(out_dir, "input.jpg")
        x_np = x.squeeze(0).cpu().numpy()
        x_np = np.transpose(x_np, (1, 2, 0))
        x_np = np.clip(x_np, 0.0, 1.0)
        img = (x_np * 255.0).astype(np.uint8)
        Image.fromarray(img).save(input_path, format="JPEG")
        print(f"Saved input image to {input_path}")

    cam = generate_vit_cam(model, x, device)

    ckpt_base = os.path.splitext(os.path.basename(args.ckpt))[0]
    meta_parts = [
        f"alpha-{args.alpha}",
        f"sev-{args.severity}",
        f"seed-{args.seed}",
    ]
    if args.class_name is not None:
        safe_class = str(args.class_name).replace(" ", "_")
        meta_parts.append(f"class-{safe_class}")
    meta_suffix = "_" + "_".join(meta_parts)
    base_name = ckpt_base + meta_suffix
    overlay_path = os.path.join(out_dir, base_name + ".jpg")

    # Always save overlay; optionally save attention-only image as well
    save_cam_overlay(x, cam, overlay_path, alpha=float(args.alpha))
    print(f"Saved CAM overlay to {overlay_path}")

    if args.attn_only:
        attn_path = os.path.join(out_dir, base_name + "_attn.jpg")
        save_cam_only(cam, attn_path)
        print(f"Saved attention-only CAM to {attn_path}")


if __name__ == "__main__":
    main()


# python data_cifar/misc/cam.py --ckpt /flash/project_465002264/projects/m2a/ckpt/source_vitb16_cifar10c.pth --method source --severity 3 --seed 8 --alpha 0.4 --class airplane