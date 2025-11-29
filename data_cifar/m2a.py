from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List
from contextlib import contextmanager


class Entropy(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy(x: torch.Tensor, x_ema: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between current logits and a detached target distribution."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy_temp(x: torch.Tensor, x_ema: torch.Tensor, temperature: float) -> torch.Tensor:
    """Cross-entropy with temperature scaling: softmax(logits/temperature).

    Args:
        x: current logits [B, K]
        x_ema: target logits [B, K] (detached)
        temperature: > 0 scalar
    """
    t = torch.tensor(temperature, dtype=x.dtype, device=x.device)
    t = torch.clamp(t, min=1e-6)
    return -((x_ema / t).softmax(1) * (x / t).log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy_temp_teacher(x: torch.Tensor, x_ema: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature scaling applied to teacher (target) only."""
    t = torch.tensor(temperature, dtype=x.dtype, device=x.device)
    t = torch.clamp(t, min=1e-6)
    return -((x_ema / t).softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy_temp_student(x: torch.Tensor, x_ema: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature scaling applied to student (current) only."""
    t = torch.tensor(temperature, dtype=x.dtype, device=x.device)
    t = torch.clamp(t, min=1e-6)
    return -(x_ema.softmax(1) * (x / t).log_softmax(1)).sum(1)


def copy_model_and_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer,
                             model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model: nn.Module) -> nn.Module:
    """Enable grads where needed for M2A CTTA and keep BatchNorm in special mode (as in REM).

    All TALN-related wrapping has been removed.
    """
    # Move model to an appropriate device if needed
    target_device = None
    if isinstance(model, nn.DataParallel) and torch.cuda.is_available() and len(model.device_ids) > 0:
        target_device = torch.device(f"cuda:{model.device_ids[0]}")
    else:
        for p in model.parameters():
            target_device = p.device
            break
    if target_device is None:
        target_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(target_device)

    # Train-time settings for TTA
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def collect_params(model: nn.Module, ln_quarter: str = 'default'):
    """Collect trainable parameters (LayerNorm weights/bias) with optional quarter selection.

    Quarter selection applies to ViT-style transformer blocks named like 'blocks.X'.
    Options:
      - 'default': original policy (skip LayerNorms in blocks 9,10,11 and any top-level 'norm')
      - 'q1'|'q2'|'q3'|'q4': only LayerNorms inside the corresponding quarter of transformer blocks
      - 'all': LayerNorms inside all transformer blocks

    For CNN backbones, the original exclusions (skip 'layer4', skip top-level 'norm') still apply.
    """
    ln_quarter = str(ln_quarter).lower()
    params = []
    names = []

    # First pass: gather transformer block indices if present
    block_indices = set()
    for nm, _ in model.named_modules():
        if 'blocks.' in nm:
            try:
                after = nm.split('blocks.')[1]
                idx_str = after.split('.')[0]
                if idx_str.isdigit():
                    block_indices.add(int(idx_str))
            except Exception:
                pass

    sorted_blocks = sorted(block_indices)
    n_blocks = len(sorted_blocks)

    def quarter_index_ranges(n: int):
        # Returns list of (start_inclusive, end_inclusive) for 4 quarters dividing [0, n)
        # Use rounding to distribute remainder evenly.
        bounds = [int(round(i * n / 4.0)) for i in range(5)]  # 0, ~n/4, ~n/2, ~3n/4, n
        return [(bounds[i], bounds[i + 1] - 1) for i in range(4)]

    allowed_blocks = None  # None means use default policy
    if ln_quarter in ['q1', 'q2', 'q3', 'q4', 'all'] and n_blocks > 0:
        if ln_quarter == 'all':
            allowed_blocks = set(sorted_blocks)
        else:
            q_map = {'q1': 0, 'q2': 1, 'q3': 2, 'q4': 3}
            q_idx = q_map[ln_quarter]
            ranges = quarter_index_ranges(n_blocks)
            # Map back to actual block indices using their sorted order
            start_pos, end_pos = ranges[q_idx]
            start_pos = max(0, min(start_pos, n_blocks - 1))
            end_pos = max(start_pos, min(end_pos, n_blocks - 1))
            allowed_blocks = set(sorted_blocks[start_pos:end_pos + 1])

    # Second pass: collect LayerNorm parameters according to policy
    for nm, m in model.named_modules():
        # Exclusions common to both policies
        if 'layer4' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        # Determine if this module is inside a specific transformer block
        this_block_idx = None
        if 'blocks.' in nm:
            try:
                after = nm.split('blocks.')[1]
                idx_str = after.split('.')[0]
                if idx_str.isdigit():
                    this_block_idx = int(idx_str)
            except Exception:
                pass

        # Apply selection policy
        if allowed_blocks is None:
            # default policy: skip LNs in blocks 9,10,11 (ViT-B typical last quarter)
            if any(f'blocks.{k}' in nm for k in ['9', '10', '11']):
                continue
        else:
            # quarter/all policy: only allow LNs in allowed transformer blocks
            if this_block_idx is None or this_block_idx not in allowed_blocks:
                continue

        if isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def _gaussian_kernel1d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    center = (kernel_size - 1) / 2.0
    xs = torch.arange(kernel_size, device=device, dtype=dtype) - center
    kernel = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
    kernel = kernel / (kernel.sum() + 1e-12)
    return kernel


def gaussian_blur2d(x: torch.Tensor, kernel_size: int = 11, sigma: float = None) -> torch.Tensor:
    """
    Simple Gaussian blur using depthwise separable 2D convolution.
    x: [B,C,H,W]
    kernel_size: odd integer
    sigma: if None, use a common heuristic based on kernel_size
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    if sigma is None:
        # Heuristic similar to OpenCV
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype
    k1d = _gaussian_kernel1d(kernel_size, sigma, device, dtype)
    k2d = torch.outer(k1d, k1d)
    kernel = k2d.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.to(device=device, dtype=dtype)
    kernel = kernel.expand(C, 1, kernel_size, kernel_size).contiguous()
    padding = kernel_size // 2
    # Depthwise conv
    return F.conv2d(x, kernel, bias=None, stride=1, padding=padding, groups=C)


def apply_frequency_mask(x: torch.Tensor, mask_percent: float, spectral_type: str = 'all') -> torch.Tensor:
    """
    Random frequency masking per image: zero-out a percentage of frequency bins
    (shared across channels) in the 2D FFT domain. Returns a masked image.

    Args:
        x: [B,C,H,W] input in float, expected in [0,1]
        mask_percent: percentage (0..100) of frequency bins to zero per image
    """
    mask_percent = float(mask_percent)
    mask_percent = max(0.0, min(100.0, mask_percent))
    B, C, H, W = x.shape
    if mask_percent <= 0.0:
        return x
    X = torch.fft.fft2(x.to(torch.float32), dim=(-2, -1), norm='ortho')
    total_bins = H * W
    k = int(math.ceil((mask_percent / 100.0) * total_bins))
    if k <= 0:
        X_masked = X
    else:
        st = str(spectral_type).lower()
        if st not in ['all', 'low', 'high']:
            st = 'all'
        base_mask = torch.ones((H, W), device=x.device, dtype=X.dtype)
        if st == 'all':
            allowed = torch.ones((H, W), device=x.device, dtype=torch.bool)
        else:
            h2 = H // 2
            w2 = W // 2
            if st == 'low':
                allowed = torch.zeros((H, W), device=x.device, dtype=torch.bool)
                allowed[:h2, :w2] = True
            else:  # 'high' -> only bottom-right quadrant (Q4)
                allowed = torch.zeros((H, W), device=x.device, dtype=torch.bool)
                allowed[h2:, w2:] = True
        allowed_idx = allowed.view(-1).nonzero().squeeze(1)
        max_choose = int(allowed_idx.numel())
        choose_k = min(k, max_choose)
        mask_batch = []
        for _ in range(B):
            if choose_k > 0:
                perm = torch.randperm(max_choose, device=x.device)[:choose_k]
                pick = allowed_idx[perm]
                flat = base_mask.new_ones((H * W,))
                flat[pick] = 0
                mask_i = flat.view(1, H, W)
            else:
                mask_i = base_mask.view(1, H, W)
            mask_batch.append(mask_i)
        mask = torch.stack(mask_batch, dim=0)  # [B,1,H,W]
        X_masked = X * mask
    x_rec = torch.fft.ifft2(X_masked, dim=(-2, -1), norm='ortho').real
    x_rec = x_rec.clamp(0.0, 1.0)
    return x_rec


def build_random_square_mask(H: int,
                             W: int,
                             ratio: float,
                             num_squares: int = 1,
                             generator: torch.Generator = None) -> torch.Tensor:
    """
    Place num_squares equal-sized squares to cover ~ratio of the image area at random positions.
    Attempts to avoid overlaps first; if not enough positions, allows overlaps to reach the desired count.
    Returns a binary mask [H, W].
    """
    device = 'cpu'
    total_area = int(round(ratio * H * W))
    if total_area <= 0 or num_squares <= 0:
        return torch.zeros((H, W), dtype=torch.float32)

    # Square side in pixels (not grid-aligned); clamp to valid range
    side = int(round(math.sqrt(total_area / float(max(num_squares, 1)))))
    side = max(1, min(side, min(H, W)))

    max_y0 = max(0, H - side)
    max_x0 = max(0, W - side)

    mask = torch.zeros((H, W), dtype=torch.float32)
    placed = []  # list of (y0, x0, side)

    def overlaps(y0, x0, s, others):
        for (yy, xx, ss) in others:
            if not (x0 + s <= xx or xx + ss <= x0 or y0 + s <= yy or yy + ss <= y0):
                return True
        return False

    # Try to place without overlap using random sampling (per-pixel positions)
    attempts = 0
    max_attempts = 2000
    while len(placed) < num_squares and attempts < max_attempts:
        if max_y0 > 0:
            y0 = int(torch.randint(low=0, high=max_y0 + 1, size=(1,), generator=generator).item())
        else:
            y0 = 0
        if max_x0 > 0:
            x0 = int(torch.randint(low=0, high=max_x0 + 1, size=(1,), generator=generator).item())
        else:
            x0 = 0
        if not overlaps(y0, x0, side, placed):
            placed.append((y0, x0, side))
        attempts += 1

    # If we still need more squares, allow overlaps
    while len(placed) < num_squares:
        if max_y0 > 0:
            y0 = int(torch.randint(low=0, high=max_y0 + 1, size=(1,), generator=generator).item())
        else:
            y0 = 0
        if max_x0 > 0:
            x0 = int(torch.randint(low=0, high=max_x0 + 1, size=(1,), generator=generator).item())
        else:
            x0 = 0
        placed.append((y0, x0, side))

    for (y0, x0, s) in placed:
        mask[y0:y0 + s, x0:x0 + s] = 1.0

    return mask

class M2A(nn.Module):
    """
    M2A: Stochastic Patch Erasing with Adaptive Residual Correction for Continual Test-Time Adaptation.

    Masking modes:
    - Random mode (random_masking='spatial'): place `num_squares` equal-size square masks at
      random pixel positions so that the union covers ~m% of the image area (non-grid-aligned).

    We then compute the REM losses across masking levels and update the model online during CTTA.
    """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 steps: int = 1, episodic: bool = False,
                 m: float = 0.1, n: int = 3, lamb: float = 1.0, margin: float = 0.0,
                 random_masking: str = 'spatial',
                 num_squares: int = 1,
                 mask_type: str = 'binary',
                 spatial_type: str = 'patch',
                 spectral_type: str = 'all',
                 seed: int = None,
                 # Plotting options
                 plot_loss: bool = False,
                 plot_loss_path: str = "",
                 plot_ema_alpha: float = 0.98,
                 # MCL temperature
                 mcl_temperature: float = 1.0,
                 mcl_temperature_apply: str = 'both',
                 mcl_distance: str = 'ce',
                 # ERL activation selection
                 erl_activation: str = 'relu',
                 erl_leaky_relu_slope: float = 0.01,
                 erl_softplus_beta: float = 1.0,
                 # (Removed progressive/adaptive masking and internal pruning controls)
                 # Disable specific losses
                 disable_mcl: bool = False,
                 disable_erl: bool = False,
                 disable_eml: bool = False,
                 # Logm2a options
                 logm2a_enable: str = 'none',
                 logm2a_lr_mult: float = 1.0,
                 logm2a_reg: float = 0.0,
                 logm2a_temp: float = 0.0,
                 ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "M2A requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

        self.m = float(m)
        self.n = int(n)
        self.mn = [i * self.m for i in range(self.n)]
        self.lamb = lamb
        self.margin = margin

        self.entropy = Entropy()

        # Entropy-masking params (fixed settings)
        # Masking domain: 'spatial' (random squares) or 'spectral' (random frequency bins)
        rm = str(random_masking).lower()
        if rm not in ['spatial', 'spectral']:
            rm = 'spatial'
        self.random_masking = rm
        # Number of equal-size squares per masking level
        self.num_squares = max(1, int(num_squares))
        # Mask fill type: 'binary' (zeros), 'mean' (per-image mean), 'gaussian' (blurred)
        mt = str(mask_type).lower()
        assert mt in ['binary', 'mean', 'gaussian'], "mask_type must be one of ['binary','mean','gaussian']"
        self.mask_type = mt
        st_sp = str(spatial_type).lower()
        if st_sp not in ['patch', 'pixel']:
            st_sp = 'patch'
        self.spatial_type = st_sp
        st_spec = str(spectral_type).lower()
        if st_spec not in ['all', 'low', 'high']:
            st_spec = 'all'
        self.spectral_type = st_spec
        # Local RNG for deterministic masking when seed is provided
        self._rng = torch.Generator(device='cpu')
        try:
            if seed is not None:
                self._rng.manual_seed(int(seed))
        except Exception:
            pass

        # Plotting state
        self.plot_loss = bool(plot_loss)
        self.plot_loss_path = str(plot_loss_path) if plot_loss_path is not None else ""
        self.plot_ema_alpha = float(plot_ema_alpha)
        self._ema_mcl = None
        self._ema_erl = None
        self._ema_mcl_hist = []
        self._ema_erl_hist = []
        self._steps_seen = 0

        # MCL temperature
        self.mcl_temperature = float(mcl_temperature)
        if self.mcl_temperature <= 0:
            raise ValueError("mcl_temperature must be > 0")
        mta = str(mcl_temperature_apply).lower()
        if mta not in ['teacher', 'student', 'both']:
            raise ValueError("mcl_temperature_apply must be one of ['teacher','student','both']")
        self.mcl_temperature_apply = mta
        # MCL distance metric
        mdist = str(mcl_distance).lower()
        if mdist not in ['ce', 'kl', 'js', 'mse', 'mae']:
            raise ValueError("mcl_distance must be one of ['ce','kl','js','mse','mae']")
        self.mcl_distance = mdist

        # ERL activation configuration
        act = str(erl_activation).lower()
        if act not in ['relu', 'leaky_relu', 'softplus', 'gelu', 'sigmoid', 'identity']:
            raise ValueError("erl_activation must be one of ['relu','leaky_relu','softplus','gelu','sigmoid','identity']")
        self.erl_activation = act
        self.erl_leaky_relu_slope = float(erl_leaky_relu_slope)
        self.erl_softplus_beta = float(erl_softplus_beta)

        # Disable flags
        self.disable_mcl = bool(disable_mcl)
        self.disable_erl = bool(disable_erl)
        self.disable_eml = bool(disable_eml)
        # eval-only flag to bypass adaptation updates
        self._eval_only = False

        # Logm2a state
        self.logm2a_enable = str(logm2a_enable).lower()
        if self.logm2a_enable not in ['none', 'gamma', 'beta', 'gammabeta']:
            self.logm2a_enable = 'none'
        self.logm2a_lr_mult = float(logm2a_lr_mult)
        self.logm2a_reg = float(logm2a_reg)
        self.logm2a_head: nn.Linear = None  # lazy init when CLS/logits dim known
        self._logm2a_params_added = False
        self.logm2a_temp = float(logm2a_temp)

        self.last_mcl = 0.0
        self.last_erl = 0.0
        self.last_eml = 0.0
        self.mcl_sum = 0.0
        self.erl_sum = 0.0
        self.eml_sum = 0.0
        self.loss_count = 0.0

    @contextmanager
    def no_adapt_mode(self):
        """Context manager to temporarily disable adaptation updates."""
        prev_eval_only = self._eval_only
        self._eval_only = True
        try:
            yield
        finally:
            self._eval_only = prev_eval_only

    def _get_cls_and_logits(self, x: torch.Tensor):
        """Try to extract class token feature [B,D] and logits [B,K] in one pass when possible.
        Falls back to calling model(x) for logits and returns (None, logits) if unsupported.
        """
        base = self.model.module if hasattr(self.model, 'module') else self.model
        cls = None
        logits = None
        try:
            if hasattr(base, 'forward_features'):
                # Many ViTs support this
                feats = base.forward_features(x)
                if isinstance(feats, tuple):
                    cls = feats[0]
                else:
                    cls = feats
                if hasattr(base, 'forward_head'):
                    logits = base.forward_head(cls)
                elif hasattr(base, 'head') and isinstance(base.head, nn.Module):
                    logits = base.head(cls)
                else:
                    # Can't map to logits, fallback
                    cls = None
                    logits = self.model(x, return_attn=False)
            else:
                logits = self.model(x, return_attn=False)
        except Exception:
            # Last resort
            logits = self.model(x, return_attn=False)
            cls = None
        return cls, logits

    def _current_levels(self):
        """Compute masking levels for this batch.
        Levels are [0, m, 2m, ..., (n-1)m] clamped to [0,1] with a static m.
        """
        levels = [max(0.0, min(1.0, i * self.m)) for i in range(self.n)]
        if len(levels) > 0:
            levels[0] = 0.0
        return levels

    def _update_and_plot_losses(self, mcl_val: torch.Tensor, erl_val: torch.Tensor):
        if not self.plot_loss:
            return
        # Detach to CPU scalars
        mcl = float(mcl_val.detach().item())
        erl = float(erl_val.detach().item())
        alpha = self.plot_ema_alpha
        # Initialize or update EMA
        if self._ema_mcl is None:
            self._ema_mcl = mcl
            self._ema_erl = erl
        else:
            self._ema_mcl = alpha * self._ema_mcl + (1.0 - alpha) * mcl
            self._ema_erl = alpha * self._ema_erl + (1.0 - alpha) * erl
        self._steps_seen += 1
        self._ema_mcl_hist.append(self._ema_mcl)
        self._ema_erl_hist.append(self._ema_erl)

        # Guard against empty path
        if not self.plot_loss_path:
            return
        # Ensure directory exists
        out_dir = os.path.dirname(self.plot_loss_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Plot
        try:
            plt.figure(figsize=(8, 5))
            xs = list(range(1, self._steps_seen + 1))
            plt.plot(xs, self._ema_mcl_hist, label='EMA MCL', color='tab:blue')
            plt.plot(xs, self._ema_erl_hist, label='EMA ERL', color='tab:orange')
            plt.xlabel('Batch steps')
            plt.ylabel('Loss (EMA)')
            plt.title('EMA of MCL and ERL over steps')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plot_loss_path)
            plt.close()
        except Exception:
            # Silently ignore plotting errors to avoid disrupting adaptation
            pass

    

    def _apply_erl_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.erl_activation == 'relu':
            return F.relu(x)
        elif self.erl_activation == 'leaky_relu':
            return F.leaky_relu(x, negative_slope=self.erl_leaky_relu_slope)
        elif self.erl_activation == 'softplus':
            return F.softplus(x, beta=self.erl_softplus_beta)
        elif self.erl_activation == 'gelu':
            return F.gelu(x)
        elif self.erl_activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.erl_activation == 'identity':
            return x
        else:
            # Should never happen due to validation
            return F.relu(x)

    def _mcl_pair_distance(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute per-sample distance between student and teacher distributions according
        to the configured MCL distance and temperature application.

        Returns a tensor [B].
        """
        dist = self.mcl_distance
        apply = self.mcl_temperature_apply
        t = self.mcl_temperature

        if dist == 'ce':
            # Use existing TorchScript CE helpers for efficiency and parity
            if apply == 'teacher':
                return softmax_entropy_temp_teacher(student_logits, teacher_logits.detach(), t)
            elif apply == 'student':
                return softmax_entropy_temp_student(student_logits, teacher_logits.detach(), t)
            else:  # both
                return softmax_entropy_temp(student_logits, teacher_logits.detach(), t)

        # Prepare temperature-adjusted logits for other distances
        s = student_logits
        te = teacher_logits.detach()
        if apply == 'teacher':
            te_t = te / t
            s_t = s
        elif apply == 'student':
            te_t = te
            s_t = s / t
        else:  # both
            te_t = te / t
            s_t = s / t

        # Probabilities and log-probabilities as needed
        p = te_t.softmax(dim=1)            # teacher distribution
        logp = te_t.log_softmax(dim=1)
        q = s_t.softmax(dim=1)             # student distribution
        logq = s_t.log_softmax(dim=1)

        if dist == 'kl':
            # KL(p || q)
            kl = (p * (logp - logq)).sum(dim=1)
            return kl
        elif dist == 'js':
            # JS(p || q) = 0.5 * KL(p||m) + 0.5 * KL(q||m), m=(p+q)/2
            m = 0.5 * (p + q)
            eps = 1e-8
            logm = (m + eps).log()
            kl_p_m = (p * (logp - logm)).sum(dim=1)
            kl_q_m = (q * (logq - logm)).sum(dim=1)
            return 0.5 * (kl_p_m + kl_q_m)
        elif dist == 'mse':
            # Mean squared error between distributions
            mse = ((p - q) ** 2).mean(dim=1)
            return mse
        elif dist == 'mae':
            # Mean absolute error between distributions
            mae = (p - q).abs().mean(dim=1)
            return mae
        else:
            # Fallback to CE (should not happen)
            return softmax_entropy_temp(student_logits, teacher_logits.detach(), t)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

    def reset_loss_stats(self):
        self.mcl_sum = 0.0
        self.erl_sum = 0.0
        self.eml_sum = 0.0
        self.loss_count = 0.0

    @torch.enable_grad()
    def forward(self, x: torch.Tensor):
        if self.episodic:
            self.reset()
        out = None
        for _ in range(self.steps):
            out = self.forward_and_adapt(x, self.optimizer)
        return out

    def forward_and_adapt(self, x: torch.Tensor, optimizer: torch.optim.Optimizer,
                          **kwargs) -> torch.Tensor:
        """Forward pass for M2A (CTTA) with multiple masked views and adaptation update."""
        # If in eval-only probe mode, bypass masking/adaptation and return base logits
        if getattr(self, "_eval_only", False):
            self.model.eval()
            with torch.no_grad():
                return self.model(x, return_attn=False)

        B, C, H, W = x.shape

        outputs_list = []
        # For Logm2a regularizer: store gamma/beta per level (including m=0 for reg only)
        logm2a_gamma_levels: List[torch.Tensor] = []
        logm2a_beta_levels: List[torch.Tensor] = []
        self.model.eval()
        levels = self._current_levels()
        for m in levels:
            if m == 0.0:
                # Compute logits normally; also get class token for regularizer only
                cls0, out0 = self._get_cls_and_logits(x)
                if isinstance(out0, tuple):
                    out0 = out0[0]
                
                z0_ref = out0.detach()
                z0_after = z0_ref
                margin_teacher = None
                # For regularizer at m=0, compute gamma/beta if needed
                need_gb_m0 = (self.logm2a_enable != 'none') and (cls0 is not None) and (self.logm2a_reg > 0.0)
                g0 = None
                b0 = None
                if need_gb_m0:
                    # Lazy-create Logm2a head
                    if self.logm2a_head is None:
                        in_dim = int(cls0.shape[-1])
                        out_dim = 2
                        self.logm2a_head = nn.Linear(in_dim, out_dim).to(cls0.device)
                        # Add params with LR multiplier
                        if not self._logm2a_params_added:
                            try:
                                base_lr = self.optimizer.param_groups[0].get('lr', None)
                                if base_lr is None:
                                    self.optimizer.add_param_group({'params': self.logm2a_head.parameters()})
                                else:
                                    self.optimizer.add_param_group({'params': self.logm2a_head.parameters(), 'lr': base_lr * self.logm2a_lr_mult})
                            except Exception:
                                pass
                            self._logm2a_params_added = True
                    # Compute scalar gamma/beta per sample
                    raw = self.logm2a_head(cls0)  # [B,2]
                    raw_g = raw[:, 0]
                    raw_b = raw[:, 1]
                    if self.logm2a_temp > 0.0:
                        raw_b = raw_b / self.logm2a_temp
                    gb_g = F.softplus(raw_g)
                    gb_b = F.softplus(raw_b)
                    g0 = gb_g   # [B]
                    b0 = gb_b   # [B]
                # Do not transform unmasked logits; only store for regularizer if enabled
                logm2a_gamma_levels.append(g0 if (need_gb_m0 and self.logm2a_reg > 0.0) else None)
                logm2a_beta_levels.append(b0 if (need_gb_m0 and self.logm2a_reg > 0.0) else None)
                outputs_list.append(out0)
                
            else:
                mfrac = m
                if self.random_masking == 'spatial':
                    xb_masked = x.clone()
                    x_blur = None
                    if self.mask_type == 'gaussian':
                        x_blur = gaussian_blur2d(xb_masked, kernel_size=11, sigma=None)
                    if self.spatial_type == 'patch':
                        for bi in range(B):
                            mask_bw = build_random_square_mask(
                                H, W, ratio=mfrac, num_squares=self.num_squares, generator=self._rng
                            ).to(x.device)
                            mask_c = mask_bw.unsqueeze(0)
                            if self.mask_type == 'binary':
                                xb_masked[bi] = x[bi] * (1.0 - mask_c)
                            elif self.mask_type == 'mean':
                                mean_val = x[bi].mean(dim=(1, 2), keepdim=True)
                                xb_masked[bi] = x[bi] * (1.0 - mask_c) + mean_val * mask_c
                            elif self.mask_type == 'gaussian':
                                xb_masked[bi] = x[bi] * (1.0 - mask_c) + x_blur[bi] * mask_c
                    else:
                        total_pixels = H * W
                        k_pix = int(round(mfrac * total_pixels))
                        k_pix = max(0, min(k_pix, total_pixels))
                        for bi in range(B):
                            if k_pix > 0:
                                flat = torch.zeros((total_pixels,), device=x.device, dtype=torch.float32)
                                idx = torch.randperm(total_pixels, generator=self._rng)[:k_pix]
                                flat[idx] = 1.0
                                mask_bw = flat.view(H, W)
                            else:
                                mask_bw = torch.zeros((H, W), device=x.device, dtype=torch.float32)
                            mask_c = mask_bw.unsqueeze(0)
                            if self.mask_type == 'binary':
                                xb_masked[bi] = x[bi] * (1.0 - mask_c)
                            elif self.mask_type == 'mean':
                                mean_val = x[bi].mean(dim=(1, 2), keepdim=True)
                                xb_masked[bi] = x[bi] * (1.0 - mask_c) + mean_val * mask_c
                            elif self.mask_type == 'gaussian':
                                xb_masked[bi] = x[bi] * (1.0 - mask_c) + x_blur[bi] * mask_c
                    for mod in self.model.modules():
                        if hasattr(mod, 'current_m'):
                            mod.current_m = float(mfrac)
                else:
                    xb_masked = apply_frequency_mask(x, mask_percent=(mfrac * 100.0), spectral_type=self.spectral_type)

                # Compute class token and logits for masked input
                cls_m, out_m = self._get_cls_and_logits(xb_masked)
                if isinstance(out_m, tuple):
                    out_m = out_m[0]

                # Apply LogM2A only for masked images if enabled and class token available
                if (self.logm2a_enable != 'none') and (cls_m is not None):
                    # Lazy-create Logm2a head
                    if self.logm2a_head is None:
                        in_dim = int(cls_m.shape[-1])
                        out_dim = 2
                        self.logm2a_head = nn.Linear(in_dim, out_dim).to(cls_m.device)
                        if not self._logm2a_params_added:
                            try:
                                base_lr = self.optimizer.param_groups[0].get('lr', None)
                                if base_lr is None:
                                    self.optimizer.add_param_group({'params': self.logm2a_head.parameters()})
                                else:
                                    self.optimizer.add_param_group({'params': self.logm2a_head.parameters(), 'lr': base_lr * self.logm2a_lr_mult})
                            except Exception:
                                pass
                            self._logm2a_params_added = True
                    # Compute gamma/beta depending on type
                    # scalar per-sample
                    raw = self.logm2a_head(cls_m)  # [B,2]
                    raw_g = raw[:, 0]
                    raw_b = raw[:, 1]
                    if self.logm2a_temp > 0.0:
                        raw_b = raw_b / self.logm2a_temp
                    gb_g = F.softplus(raw_g)  # non-negative
                    gb_b = F.softplus(raw_b)
                    gamma_apply = gb_g.unsqueeze(1)  # [B,1]
                    beta_apply = gb_b.unsqueeze(1)   # [B,1]
                    gamma_b = gamma_apply.squeeze(1)  # [B]
                    beta_b = beta_apply.squeeze(1)   # [B]
                    # Mode selection (which parameters to use)
                    if self.logm2a_enable == 'gamma':
                        beta_use = torch.zeros_like(beta_apply)
                        gamma_use = gamma_apply
                    elif self.logm2a_enable == 'beta':
                        beta_use = beta_apply
                        gamma_use = torch.ones_like(beta_apply)
                    else:  # 'gammabeta'
                        beta_use = beta_apply
                        gamma_use = gamma_apply
                    # Apply gamma/beta transform followed by L2 normalization
                    xform = out_m * gamma_use + beta_use
                    eps = 1e-6
                    mag = torch.norm(xform, p=2, dim=1, keepdim=True).clamp_min(eps)
                    out_m = xform / mag
                    # Save gamma/beta for regularizer
                    logm2a_gamma_levels.append(gamma_b)
                    logm2a_beta_levels.append(beta_b)
                else:
                    logm2a_gamma_levels.append(None)
                    logm2a_beta_levels.append(None)

                outputs_list.append(out_m)
        self.model.train()

        # Losses computed on raw outputs (MARN removed)
        outputs_for_losses = outputs_list
        levels_for_losses = levels

        # Compute entropies early if ERL needed
        entropys = None
        if not self.disable_erl:
            entropys = [self.entropy(o) for o in outputs_list]

        mcl_loss = None
        erl_loss = None
        eml_loss = None

        # Mask Consistency Loss (MCL)
        if not self.disable_mcl:
            total_mcl = None
            # Need at least 2 levels for any pairwise distance
            for i in range(1, len(self.mn)):
                term = self._mcl_pair_distance(outputs_for_losses[i], outputs_for_losses[0]).mean()
                total_mcl = term if total_mcl is None else (total_mcl + term)
                for j in range(1, i):
                    term_ij = self._mcl_pair_distance(outputs_for_losses[i], outputs_for_losses[j]).mean()
                    total_mcl = term_ij if total_mcl is None else (total_mcl + term_ij)
            mcl_loss = total_mcl  # may remain None if no pairs

        # Entropy Ranking Loss (ERL)
        if not self.disable_erl:
            margin = self.margin * math.log(outputs_list[0].shape[-1])
            total_erl = None
            levels_len = len(self._current_levels())
            for i in range(levels_len):
                for j in range(i + 1, levels_len):
                    ent_i = entropys[i]
                    ent_j = entropys[j].detach()
                    diff = ent_i - ent_j + margin
                    actv = self._apply_erl_activation(diff).mean()
                    total_erl = actv if total_erl is None else (total_erl + actv)
            erl_loss = total_erl  # may remain None if no pairs

        if not self.disable_eml:
            try:
                eml_terms = []
                for out in outputs_for_losses:
                    eml_terms.append(self.entropy(out).mean())
                if len(eml_terms) > 0:
                    eml_loss = sum(eml_terms) / float(len(eml_terms))
            except Exception:
                eml_loss = None

        # Logm2a monotonic regularizer across levels (including m=0 baseline if available)
        logm2a_reg_loss = None
        if (self.logm2a_enable != 'none') and (self.logm2a_reg > 0.0):
            try:
                # Build tensors for consecutive level penalties where gamma/beta exist
                penalties = []
                for seq in (logm2a_gamma_levels, logm2a_beta_levels):
                    prev = None
                    for val in seq:
                        if val is None:
                            prev = None
                            continue
                        if prev is None:
                            prev = val
                            continue
                        # Enforce non-decreasing: penalize decreases
                        penalties.append(F.relu(prev - val).mean())
                        prev = val
                if len(penalties) > 0:
                    logm2a_reg_loss = sum(penalties) / float(len(penalties))
            except Exception:
                logm2a_reg_loss = None

        # Total loss and optimizer step
        loss_terms = []
        if isinstance(mcl_loss, torch.Tensor) and mcl_loss.requires_grad:
            loss_terms.append(mcl_loss)
        if isinstance(erl_loss, torch.Tensor) and erl_loss.requires_grad:
            loss_terms.append(self.lamb * erl_loss)
        if isinstance(eml_loss, torch.Tensor) and eml_loss.requires_grad:
            loss_terms.append(eml_loss)
        # taln_reg is a no-op and not included

        if (logm2a_reg_loss is not None) and (self.logm2a_reg > 0.0):
            loss_terms.append(self.logm2a_reg * logm2a_reg_loss)
        if len(loss_terms) > 0:
            loss = loss_terms[0]
            for lt in loss_terms[1:]:
                loss = loss + lt
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Update EMA trackers and save plot if enabled (zero when disabled)
        mcl_plot_val = mcl_loss if mcl_loss is not None else torch.tensor(0.0, device=x.device)
        erl_plot_val = erl_loss if erl_loss is not None else torch.tensor(0.0, device=x.device)
        self._update_and_plot_losses(mcl_val=mcl_plot_val, erl_val=erl_plot_val)

        # Convert to scalars and accumulate per-corruption sums
        try:
            mcl_val = float(mcl_plot_val.detach().item())
        except Exception:
            mcl_val = 0.0
        try:
            erl_val = float(erl_plot_val.detach().item())
        except Exception:
            erl_val = 0.0
        try:
            eml_val = float(eml_loss.detach().item()) if eml_loss is not None else 0.0
        except Exception:
            eml_val = 0.0

        batch_weight = float(x.shape[0])
        self.mcl_sum += mcl_val * batch_weight
        self.erl_sum += erl_val * batch_weight
        self.eml_sum += eml_val * batch_weight
        self.loss_count += batch_weight

        self.last_mcl = mcl_val
        self.last_erl = erl_val
        self.last_eml = eml_val

        # Return full-batch predictions
        return outputs_list[0]
