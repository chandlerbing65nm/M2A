import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import PIL

from methods.base import TTAMethod, forward_decorator
from utils.registry import ADAPTATION_REGISTRY


def get_gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array (ported from m2a operators)."""

    def _gaussian_fn(kernlen, std):
        n = torch.arange(0, kernlen).float()
        n -= n.mean()
        n /= std
        w = torch.exp(-0.5 * n ** 2)
        return w

    gkern1d = _gaussian_fn(kernlen, std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d / gkern2d.sum()


class HOGLayerC(nn.Module):
    """HOG feature extractor (simplified port from m2a.data_cifar.operators.HOGLayerC)."""

    def __init__(self, nbins: int = 9, pool: int = 7, gaussian_window: int = 0):
        super().__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi

        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer("weight_x", weight_x)
        self.register_buffer("weight_y", weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gkern = get_gkern(gaussian_window, gaussian_window // 2)
            self.register_buffer("gkern", gkern)

    @torch.no_grad()
    def forward(self, x, ori_img=None, hog_img=None, index: int = 0):
        # x: B x 3 x H x W
        x = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")
        gx_rgb = F.conv2d(x, self.weight_x, bias=None, stride=1, padding=0, groups=3)
        gy_rgb = F.conv2d(x, self.weight_y, bias=None, stride=1, padding=0, groups=3)

        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins  # [-nbins, nbins]

        b, c, h, w = norm_rgb.shape
        out = torch.zeros((b, c, self.nbins, h, w), dtype=torch.float, device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)

        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, f"h {h} gw {self.gaussian_window}"
                repeat_rate = h // self.gaussian_window
                temp_gkern = self.gkern.repeat([repeat_rate, repeat_rate])
            else:
                temp_gkern = self.gkern
            norm_rgb *= temp_gkern

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        out = torch.nn.functional.normalize(out, p=2, dim=2)
        return out  # B x 3 x nbins x H//pool x W//pool


class GaussianNoise(nn.Module):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.mean = float(mean)
        self.std = float(std)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(img) * self.std + self.mean
        return img + noise


class Clip(nn.Module):
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return torch.clamp(img, self.min_val, self.max_val)


class ColorJitterPro(T.ColorJitter):
    """Color jitter with optional gamma, mirroring mae_transforms.ColorJitterPro."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, 'gamma')

    @staticmethod
    def get_params(brightness, contrast, saturation, hue, gamma):
        transforms = []
        rng = torch.Generator()
        rng.manual_seed(torch.randint(0, 2 ** 31 - 1, (1,)).item())

        def _uniform(tup):
            return torch.empty(1, generator=rng).uniform_(tup[0], tup[1]).item()

        if brightness is not None:
            brightness_factor = _uniform(brightness)
            transforms.append(T.Lambda(lambda img: T.functional.adjust_brightness(img, brightness_factor)))
        if contrast is not None:
            contrast_factor = _uniform(contrast)
            transforms.append(T.Lambda(lambda img: T.functional.adjust_contrast(img, contrast_factor)))
        if saturation is not None:
            saturation_factor = _uniform(saturation)
            transforms.append(T.Lambda(lambda img: T.functional.adjust_saturation(img, saturation_factor)))
        if hue is not None:
            hue_factor = _uniform(hue)
            transforms.append(T.Lambda(lambda img: T.functional.adjust_hue(img, hue_factor)))
        if gamma is not None:
            gamma_factor = _uniform(gamma)
            transforms.append(T.Lambda(lambda img: T.functional.adjust_gamma(img, gamma_factor)))

        if len(transforms) == 0:
            return T.Identity()
        order = torch.randperm(len(transforms), generator=rng).tolist()
        ordered = [transforms[i] for i in order]
        return T.Compose(ordered)

    def forward(self, img):
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                img = T.functional.adjust_brightness(img, torch.empty(1).uniform_(self.brightness[0], self.brightness[1]).item())
            if fn_id == 1 and self.contrast is not None:
                img = T.functional.adjust_contrast(img, torch.empty(1).uniform_(self.contrast[0], self.contrast[1]).item())
            if fn_id == 2 and self.saturation is not None:
                img = T.functional.adjust_saturation(img, torch.empty(1).uniform_(self.saturation[0], self.saturation[1]).item())
            if fn_id == 3 and self.hue is not None:
                img = T.functional.adjust_hue(img, torch.empty(1).uniform_(self.hue[0], self.hue[1]).item())
            if fn_id == 4 and self.gamma is not None:
                gamma_factor = torch.empty(1).uniform_(self.gamma[0], self.gamma[1]).item()
                img = img.clamp(1e-8, 1.0)
                img = T.functional.adjust_gamma(img, gamma_factor)
        return img


def get_tta_transforms(img_size, gaussian_std: float = 0.005, soft: bool = False):
    # img_size is (H, W) or int; assume square for TTA
    if isinstance(img_size, int):
        n_pixels = img_size
    else:
        n_pixels = int(img_size[0])

    clip_min, clip_max = 0.0, 1.0
    p_hflip = 0.5

    return T.Compose([
        Clip(0.0, 1.0),
        ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3],
        ),
        T.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        T.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1 / 16, 1 / 16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=PIL.Image.BILINEAR,
        ),
        T.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        T.CenterCrop(size=n_pixels),
        T.RandomHorizontalFlip(p=p_hflip),
        GaussianNoise(0.0, gaussian_std),
        Clip(clip_min, clip_max),
    ])


@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


@ADAPTATION_REGISTRY.register()
class CMAE(TTAMethod):
    """Continual MAE-style test-time adaptation (ported as 'cmae')."""

    def __init__(self, cfg, model, num_classes):
        # First run base initializer (sets up model, device, img_size, optimizer on norm params)
        super().__init__(cfg, model, num_classes)

        # Continual-MAE specific hyperparameters from config
        self.block_size = int(getattr(cfg, "block_size", 16))
        self.mask_ratio = float(getattr(cfg, "mask_ratio", 0.5))
        self.use_hog = bool(getattr(cfg, "use_hog", False))
        self.hog_ratio = float(getattr(cfg, "hog_ratio", 1.0))

        # Discover embedding dim for mask token from underlying ViT backbone
        core = self.model
        for _ in range(3):
            if hasattr(core, "model"):
                core = core.model
                continue
            if hasattr(core, "module"):
                core = core.module
                continue
            break
        embed_dim = getattr(core, "embed_dim", 768)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim, device=self.device))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Optional HOG encoder and projection head
        if self.use_hog:
            nbins = 9
            cell_sz = 8
            self.hogs = HOGLayerC(nbins=nbins, pool=cell_sz, gaussian_window=0).to(self.device)
            for p in self.hogs.parameters():
                p.requires_grad_(False)

            head_dim = embed_dim
            num_class = int(nbins * 3 * (16 / cell_sz) * (16 / cell_sz))
            self.projections = nn.Linear(head_dim, num_class, bias=True).to(self.device)
            nn.init.trunc_normal_(self.projections.weight, std=0.02)
            if self.projections.bias is not None:
                nn.init.constant_(self.projections.bias, 0.0)

            self.mse_func = nn.MSELoss(reduction="mean")
        else:
            self.hogs = None
            self.projections = None
            self.mse_func = None

        # Extend optimizer params to include mask_token and (optionally) projection head
        self.params.append(self.mask_token)
        if self.projections is not None:
            self.params.extend(list(self.projections.parameters()))
        self.optimizer = self.setup_optimizer()

        self.mt = float(getattr(cfg.OPTIM, "MT", 0.999))
        self.rst = float(getattr(cfg.OPTIM, "RST", 0.01))
        self.ap = float(getattr(cfg.OPTIM, "AP", 0.92))

        # Teacher and anchor models (EMA + source copy)
        self.model_temp = self.copy_model(self.model)
        for p in self.model_temp.parameters():
            p.detach_()
        self.model_anchor = self.copy_model(self.model)
        for p in self.model_anchor.parameters():
            p.detach_()

        # Extend models list for reset/restore bookkeeping
        self.models = [self.model, self.model_temp, self.model_anchor]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()
        self.model_state = self.model_states[0]

        self.transform = get_tta_transforms(self.img_size)

    def loss_calculation(self, x):
        imgs_test = x[0]

        # Teacher predictions for EMA baseline
        with torch.no_grad():
            outputs_ema = self.model_temp(imgs_test)

        # Uncertainty-based mask selection using teacher with TTA
        n_forward = 10
        with torch.no_grad():
            n_outputs = []
            for _ in range(n_forward):
                aug = self.transform(imgs_test)
                # Use teacher without masking to estimate per-patch variance
                _, tokens = self.model_temp(aug, None, None, return_norm=True)
                # tokens: (B, N+1, D); drop CLS and average over channels
                tokens_patches = tokens[:, 1:, :]
                pooled = tokens_patches.mean(dim=2)
                n_outputs.append(pooled)
            stacked = torch.stack(n_outputs, dim=0)
            variance = torch.var(stacked, dim=0)
            sorted_vals, sorted_idx = torch.sort(variance, dim=1, descending=True)
            top_k = max(1, int(sorted_idx.shape[1] * self.mask_ratio))
            mask_idx = sorted_idx[:, :top_k]
            mask_chosed = torch.zeros_like(sorted_vals)
            mask_chosed.scatter_(1, mask_idx, 1.0)

        # Student update with masked tokens
        outputs_student, tokens_student = self.model(imgs_test, self.mask_token, mask_chosed, return_norm=True)
        loss_ori = softmax_entropy(outputs_student, outputs_ema).mean(0)

        # Optional HOG reconstruction loss on masked patches
        if self.hogs is not None and self.projections is not None and self.use_hog:
            output_mask = mask_chosed.to(torch.bool)
            # Project token features to HOG space (skip CLS token)
            hog_preds = self.projections(tokens_student[:, 1:, :])
            hog_preds = hog_preds[output_mask]
            hog_labels = self._get_hog_label_2d(imgs_test, output_mask, block_size=self.block_size)
            hog_loss = self.mse_func(hog_preds, hog_labels)
            loss = loss_ori + self.hog_ratio * hog_loss
        else:
            loss = loss_ori

        return outputs_ema, loss

    def _get_hog_label_2d(self, input_frames: torch.Tensor, output_mask: torch.Tensor, block_size: int):
        """Compute block-wise HOG labels corresponding to masked tokens.

        Ported from m2a Continual_MAE._get_hog_label_2d.
        """
        # input_frames: B x C x H x W
        feat_size = input_frames.shape[-1] // block_size  # number of windows per side
        tmp_hog = self.hogs(input_frames).flatten(1, 2)   # B x (3*nbins) x H' x W'
        unfold_size = tmp_hog.shape[-1] // feat_size
        tmp_hog = (
            tmp_hog.permute(0, 2, 3, 1)
            .unfold(1, unfold_size, unfold_size)
            .unfold(2, unfold_size, unfold_size)
            .flatten(1, 2)
            .flatten(2)
        )
        tmp_hog = tmp_hog[output_mask]
        return tmp_hog

    @forward_decorator
    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs_ema, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs_ema, loss = self.loss_calculation(x)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Stochastic restore on student
        with torch.no_grad():
            if self.rst > 0.0:
                for nm, m in self.model.named_modules():
                    for npp, p in m.named_parameters(recurse=False):
                        if npp in ["weight", "bias"] and p.requires_grad:
                            mask = (torch.rand_like(p) < self.rst).float()
                            full_name = f"{nm}.{npp}" if nm else npp
                            if full_name in self.model_state:
                                p.data = self.model_state[full_name] * mask + p * (1.0 - mask)

        # Update teacher to match student (one-model EMA style)
        with torch.no_grad():
            for p_t, p_s in zip(self.model_temp.parameters(), self.model.parameters()):
                p_t.data.mul_(self.mt).add_(p_s.data * (1.0 - self.mt))

        return outputs_ema

    @torch.no_grad()
    def forward_sliding_window(self, x):
        imgs_test = x[0]
        return self.model_temp(imgs_test)

    def configure_model(self):
        """Configure model for CMAE updates (similar to other ViT-based methods)."""
        self.model.eval()
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

    def collect_params(self):
        """Collect norm parameters (BN/LN/GN) as in the original continual_mae strategy.

        The learnable mask_token is created and added to the optimizer separately in __init__.
        """
        params = []
        param_names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"] and p.requires_grad:
                        params.append(p)
                        param_names.append(f"{nm}.{np}")
        return params, param_names
