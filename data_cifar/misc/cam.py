import argparse
import os
import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image

# Optional dependency for visualization
try:
    import cv2
except Exception:
    cv2 = None

# Ensure we can import local robustbench (data_cifar/robustbench)
try:
    THIS_DIR = Path(__file__).resolve().parent
    DATA_CIFAR_DIR = THIS_DIR.parent
    if str(DATA_CIFAR_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_CIFAR_DIR))
except Exception:
    pass

# Local robustbench fork
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(substr):
            new_state_dict[k[len(substr):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class ViTGradCam:
    """Grad-CAM for ViT-like models.

    - Hooks the target layer (e.g., last block norm1) to capture features and grads.
    - reshape_transform maps tokens -> (C,H,W) with grid_size tokens.
    """
    def __init__(self, model: nn.Module, target: nn.Module, grid_size: int):
        self.model = model.eval()
        self.target = target
        self.grid = int(grid_size)
        self.feature = None
        self.gradient = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.feature = self.reshape_transform(out)
        def bwd_hook(module, grad_input, grad_output):
            # grad_output is a tuple with one element (gradient wrt output)
            grad = grad_output[0]
            self.gradient = self.reshape_transform(grad)
        self.target.register_forward_hook(fwd_hook)
        self.target.register_full_backward_hook(bwd_hook)

    def reshape_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        # Expect tensor shape [B, N_tokens, C]; drop CLS token
        b, n, c = tensor.shape
        h = w = self.grid
        # remove CLS (first token), reshape to (B, H, W, C) then (B, C, H, W)
        x = tensor[:, 1:, :].reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x

    def __call__(self, inputs: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        # target class: top-1
        top_idx = int(torch.argmax(logits.detach(), dim=1).item())
        score = logits[:, top_idx]
        score.backward(retain_graph=False)
        # compute weights and cam
        grad = self.gradient[0].detach().cpu().numpy()  # (C,H,W)
        feat = self.feature[0].detach().cpu().numpy()   # (C,H,W)
        weights = grad.mean(axis=(1, 2))                # (C,)
        cam = (feat * weights[:, None, None]).sum(axis=0)
        cam = np.maximum(cam, 0)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam  # (H,W) in [0,1]


def method_to_arch(method: str) -> str:
    m = (method or '').strip().lower()
    # Accept direct arch names or map common aliases
    direct = {
        'standard_vitb': 'Standard_VITB',
        'standard_vitb_rem': 'Standard_VITB_REM',
        'standard_vitb_m2a': 'Standard_VITB_M2A',
        'standard_vitb_mae': 'Standard_VITB_MAE',
    }
    if method in direct.values():
        return method
    if m in direct:
        return direct[m]
    if m in ['vit', 'vitb', 'vit-b', 'vit_base', 'vit_base_patch16_384']:
        return 'Standard_VITB'
    if m in ['rem', 'vitb_rem']:
        return 'Standard_VITB_REM'
    if m in ['m2a', 'vitb_m2a']:
        return 'Standard_VITB_M2A'
    if m in ['mae', 'vitb_mae']:
        return 'Standard_VITB_MAE'
    # Fallback: assume user passed a valid robustbench arch key
    return method


def load_vit_model(arch_key: str, dataset: str = 'cifar10') -> nn.Module:
    # ThreatModel.corruptions is used in CIFAR-C evals
    model = load_model(arch_key, './ckpt', dataset, ThreatModel.corruptions)
    return model


def load_checkpoint_into_model(model: nn.Module, ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model'] if (isinstance(ckpt, dict) and 'model' in ckpt) else ckpt
    # strip common prefixes
    state = rm_substr_from_state_dict(state, 'module.')
    state = rm_substr_from_state_dict(state, 'model.')
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        # As a last resort, wrap with DataParallel suffixes
        try:
            from collections import OrderedDict
            new_state = OrderedDict()
            for k, v in state.items():
                new_state['module.' + k] = v
            model.load_state_dict(new_state, strict=False)
        except Exception:
            raise
    model.to(device).eval()
    return model


def preprocess_image(img_path: str, size: int = 384, device: torch.device = torch.device('cpu')):
    # Load image as RGB
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0  # H,W,3 in [0,1]
    # To torch tensor NCHW
    x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    # Normalize with mean/std 0.5 (matches ViT cfg in this repo)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    x = (x - mean) / std
    x = x.to(device)
    # Also return resized original in [0,1] HWC for overlay
    img_resized = F.interpolate(torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0),
                                size=(size, size), mode='bilinear', align_corners=False)[0]
    img_resized = img_resized.permute(1, 2, 0).cpu().numpy()  # H,W,3 in [0,1]
    return x, img_resized


def colorize_and_overlay(base_img_hwc: np.ndarray, cam_hw: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h, w, _ = base_img_hwc.shape
    cam_resized = cv2.resize(cam_hw, (w, h)) if cv2 is not None else cam_hw
    if cv2 is not None:
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255.0
        img = np.float32(base_img_hwc)
        overlay = (1 - alpha) * heatmap + alpha * img
        overlay = overlay / np.maximum(overlay.max(), 1e-8)
        out = np.uint8(255 * overlay)
    else:
        # Fallback: grayscale overlay
        cam_rgb = np.stack([cam_resized] * 3, axis=-1)
        out = np.uint8(255 * (alpha * base_img_hwc + (1 - alpha) * cam_rgb))
    return out


def main():
    parser = argparse.ArgumentParser(description='ViT Grad-CAM visualization')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint to load')
    parser.add_argument('--method', type=str, required=True, help='Model arch/method (e.g., Standard_VITB_M2A)')
    parser.add_argument('--outdir', '--out_dir', dest='outdir', type=str, default='data_cifar/plots/cam', help='Output directory')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--input', type=str, required=True, help='Path to input JPG image')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

    set_seed(int(args.seed))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine arch key
    arch_key = method_to_arch(args.method)
    logger.info(f'Using arch: {arch_key}')

    # Instantiate model (assume CIFAR-10 ViT by default)
    model = load_vit_model(arch_key, dataset='cifar10')

    # Load checkpoint weights
    model = load_checkpoint_into_model(model, args.ckpt, device)

    # Choose target layer from printed architecture (last block norm1)
    target_layer = None
    try:
        if hasattr(model, 'blocks') and len(model.blocks) > 0 and hasattr(model.blocks[-1], 'norm1'):
            target_layer = model.blocks[-1].norm1
        elif hasattr(model, 'blocks') and len(model.blocks) > 0 and hasattr(model.blocks[-1], 'norm'):
            target_layer = model.blocks[-1].norm
    except Exception:
        target_layer = None
    if target_layer is None:
        raise RuntimeError('Could not locate target norm layer in last transformer block for Grad-CAM.')

    # Prepare input (resize to 384 via F.interpolate and normalize)
    x, img_resized = preprocess_image(args.input, size=384, device=device)

    # ViT-B/16 @ 384 has 24x24 token grid
    grad_cam = ViTGradCam(model, target_layer, grid_size=24)

    with torch.enable_grad():
        cam = grad_cam(x)  # (H,W) in [0,1]

    # Overlay and save
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.outdir, f"{ckpt_stem}_{input_stem}.jpg")
    vis = colorize_and_overlay(img_resized, cam, alpha=0.5)
    if cv2 is not None:
        cv2.imwrite(out_path, vis[:, :, ::-1])  # convert RGB->BGR for cv2
    else:
        Image.fromarray(vis).save(out_path)
    logger.info(f'Saved CAM visualization to: {out_path}')


if __name__ == '__main__':
    main()


# python data_cifar/misc/cam.py --ckpt /flash/project_465002264/projects/m2a/ckpt/m2a_vitb16_spatial_cifar10c.pth --method Standard_VITB_M2A --input /scratch/project_465002264/datasets/cifar10c/CIFAR-10-C/samples/zoom_blur/severity_5/zoom_blur_s5_45595.jpg --outdir data_cifar/plots/cam --seed 1