from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import torchvision

import numpy as np
import PIL
from time import time
import logging
import math

import matplotlib.pyplot as plt


@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def __call__(self, logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def _gaussian_kernel1d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    center = (kernel_size - 1) / 2.0
    xs = torch.arange(kernel_size, device=device, dtype=dtype) - center
    kernel = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
    kernel = kernel / (kernel.sum() + 1e-12)
    return kernel


def gaussian_blur2d(x: torch.Tensor, kernel_size: int = 11, sigma: float = None) -> torch.Tensor:
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    if sigma is None:
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
    return F.conv2d(x, kernel, bias=None, stride=1, padding=padding, groups=C)


def apply_frequency_mask(x: torch.Tensor, mask_percent: float, spectral_type: str = 'all') -> torch.Tensor:
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
            else:  # 'high'
                allowed = torch.ones((H, W), device=x.device, dtype=torch.bool)
                allowed[:h2, :w2] = False
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


def build_random_square_mask(H: int, W: int, ratio: float, num_squares: int = 1,
                             generator: torch.Generator = None) -> torch.Tensor:
    device = 'cpu'
    total_area = int(round(ratio * H * W))
    if total_area <= 0 or num_squares <= 0:
        return torch.zeros((H, W), dtype=torch.float32)
    side = int(round(math.sqrt(total_area / float(max(num_squares, 1)))))
    side = max(1, min(side, min(H, W)))
    max_y0 = max(0, H - side)
    max_x0 = max(0, W - side)
    mask = torch.zeros((H, W), dtype=torch.float32)
    placed = []

    def overlaps(y0, x0, s, others):
        for (yy, xx, ss) in others:
            if not (x0 + s <= xx or xx + ss <= x0 or y0 + s <= yy or yy + ss <= y0):
                return True
        return False

    attempts = 0
    max_attempts = 2000
    while len(placed) < num_squares and attempts < max_attempts:
        y0 = int(torch.randint(low=0, high=max_y0 + 1, size=(1,), generator=generator).item()) if max_y0 > 0 else 0
        x0 = int(torch.randint(low=0, high=max_x0 + 1, size=(1,), generator=generator).item()) if max_x0 > 0 else 0
        if not overlaps(y0, x0, side, placed):
            placed.append((y0, x0, side))
        attempts += 1

    while len(placed) < num_squares:
        y0 = int(torch.randint(low=0, high=max_y0 + 1, size=(1,), generator=generator).item()) if max_y0 > 0 else 0
        x0 = int(torch.randint(low=0, high=max_x0 + 1, size=(1,), generator=generator).item()) if max_x0 > 0 else 0
        placed.append((y0, x0, side))

    for (y0, x0, s) in placed:
        mask[y0:y0 + s, x0:x0 + s] = 1.0
    return mask


def _forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    try:
        return model(x, return_attn=False)
    except TypeError:
        try:
            return model(x)
        except Exception:
            return model(x)


class M2A(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, m=0.1, n=3, lamb=1.0, margin=0.0,
                 random_masking: str = 'spatial', num_squares: int = 1, mask_type: str = 'binary',
                 spatial_type: str = 'patch', spectral_type: str = 'all', seed: int = None,
                 disable_mcl: bool = False, disable_erl: bool = False, disable_eml: bool = False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic

        self.model_state, self.optimizer_state, _, _ = copy_model_and_optimizer(self.model, self.optimizer)

        self.m = float(m)
        self.n = int(n)
        self.mn = [i * self.m for i in range(self.n)]
        self.lamb = lamb
        self.margin = margin * math.log(1000)

        self.entropy = Entropy()
        self.disable_mcl = bool(disable_mcl)
        self.disable_erl = bool(disable_erl)
        self.disable_eml = bool(disable_eml)

        rm = str(random_masking).lower()
        self.random_masking = rm if rm in ['spatial', 'spectral'] else 'spatial'
        self.num_squares = max(1, int(num_squares))
        mt = str(mask_type).lower()
        self.mask_type = mt if mt in ['binary', 'gaussian', 'mean'] else 'binary'
        st_sp = str(spatial_type).lower()
        self.spatial_type = st_sp if st_sp in ['patch', 'pixel'] else 'patch'
        st_spec = str(spectral_type).lower()
        self.spectral_type = st_spec if st_spec in ['all', 'low', 'high'] else 'all'
        self._rng = torch.Generator(device='cpu')
        try:
            if seed is not None:
                self._rng.manual_seed(int(seed))
        except Exception:
            pass

    def forward(self, x):
        if self.episodic:
            self.reset()
        outputs = None
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.optimizer)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

    @torch.enable_grad()
    def forward_and_adapt(self, x, optimizer):
        B, C, H, W = x.shape
        self.model.eval()
        outputs0 = _forward_logits(self.model, x)

        outputs_list = [outputs0]
        for m in self.mn[1:]:
            mfrac = m
            if self.random_masking == 'spatial':
                xb = x.clone()
                x_blur = gaussian_blur2d(xb, kernel_size=11, sigma=None) if self.mask_type == 'gaussian' else None
                if self.spatial_type == 'patch':
                    for bi in range(B):
                        mask_bw = build_random_square_mask(H, W, ratio=mfrac, num_squares=self.num_squares, generator=self._rng).to(x.device)
                        mask_c = mask_bw.unsqueeze(0)
                        if self.mask_type == 'binary':
                            xb[bi] = x[bi] * (1.0 - mask_c)
                        elif self.mask_type == 'mean':
                            mean_val = x[bi].mean(dim=(1, 2), keepdim=True)
                            xb[bi] = x[bi] * (1.0 - mask_c) + mean_val * mask_c
                        elif self.mask_type == 'gaussian':
                            xb[bi] = x[bi] * (1.0 - mask_c) + x_blur[bi] * mask_c
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
                            xb[bi] = x[bi] * (1.0 - mask_c)
                        elif self.mask_type == 'mean':
                            mean_val = x[bi].mean(dim=(1, 2), keepdim=True)
                            xb[bi] = x[bi] * (1.0 - mask_c) + mean_val * mask_c
                        elif self.mask_type == 'gaussian':
                            xb[bi] = x[bi] * (1.0 - mask_c) + x_blur[bi] * mask_c
            else:
                xb = apply_frequency_mask(x, mask_percent=(mfrac * 100.0), spectral_type=self.spectral_type)

            out_m = _forward_logits(self.model, xb)
            outputs_list.append(out_m)
        self.model.train()

        total_loss_terms = []
        if not self.disable_mcl:
            mcl = 0.0
            for i in range(1, len(self.mn)):
                mcl = mcl + softmax_entropy(outputs_list[i], outputs_list[0].detach()).mean()
                for j in range(1, i):
                    mcl = mcl + softmax_entropy(outputs_list[i], outputs_list[j].detach()).mean()
            if isinstance(mcl, torch.Tensor) and mcl.requires_grad:
                total_loss_terms.append(mcl)

        if not self.disable_erl:
            entropys = [self.entropy(out) for out in outputs_list]
            erl = 0.0
            for i in range(len(self.mn)):
                for j in range(i + 1, len(self.mn)):
                    erl = erl + (F.relu(entropys[i] - entropys[j].detach() + self.margin)).mean()
            if isinstance(erl, torch.Tensor) and erl.requires_grad:
                total_loss_terms.append(self.lamb * erl)

        if not self.disable_eml:
            eml_terms = [self.entropy(out).mean() for out in outputs_list]
            if len(eml_terms) > 0:
                eml = sum(eml_terms) / float(len(eml_terms))
                if isinstance(eml, torch.Tensor) and eml.requires_grad:
                    total_loss_terms.append(eml)

        if len(total_loss_terms) > 0:
            loss = total_loss_terms[0]
            for lt in total_loss_terms[1:]:
                loss = loss + lt
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return outputs0

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []

    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue
        if isinstance(m, nn.LayerNorm):
           for np, p in m.named_parameters():
               if np in ['weight', 'bias'] and p.requires_grad:
                   params.append(p)
                   #print(nm, np)

    return params


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
        # if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            #m.track_running_stats = True
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
