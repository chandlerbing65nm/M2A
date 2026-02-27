"""
Builds upon: https://github.com/DequanWang/tent
Corresponding paper: https://arxiv.org/abs/2006.10726
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy


@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


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
    b, c, h, w = x.shape
    device = x.device
    dtype = x.dtype
    k1d = _gaussian_kernel1d(kernel_size, sigma, device, dtype)
    k2d = torch.outer(k1d, k1d)
    kernel = k2d.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.to(device=device, dtype=dtype)
    kernel = kernel.expand(c, 1, kernel_size, kernel_size).contiguous()
    padding = kernel_size // 2
    return F.conv2d(x, kernel, bias=None, stride=1, padding=padding, groups=c)


def _build_self_conjugate_mask(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Return a bool mask that is True for every self-conjugate (Hermitian-fixed)
    bin of an h x w real-valued 2-D DFT.

    A bin (u, v) is self-conjugate when (-u mod h, -v mod w) == (u, v).
    This always includes DC (0, 0) and, for even dimensions, the Nyquist
    row/column/corner bins.
    """
    sc = torch.zeros((h, w), device=device, dtype=torch.bool)
    sc[0, 0] = True
    if h % 2 == 0:
        sc[h // 2, 0] = True
    if w % 2 == 0:
        sc[0, w // 2] = True
    if h % 2 == 0 and w % 2 == 0:
        sc[h // 2, w // 2] = True
    return sc


def _build_conjugate_pairs(h: int, w: int, device: torch.device):
    """Return (pair_uv, pair_conj) – two (P, 2) long tensors that list every
    unique conjugate pair in an h x w DFT grid, excluding self-conjugate bins.

    For each pair index p, bin pair_uv[p] and bin pair_conj[p] are conjugate
    partners: X[pair_uv[p]] == conj(X[pair_conj[p]]).
    """
    us = torch.arange(h, device=device)
    vs = torch.arange(w, device=device)
    uu, vv = torch.meshgrid(us, vs, indexing="ij")
    uu_flat = uu.reshape(-1)
    vv_flat = vv.reshape(-1)

    cu = (-uu_flat) % h
    cv = (-vv_flat) % w

    # Linear indices for the bin and its conjugate
    idx = uu_flat * w + vv_flat
    cidx = cu * w + cv

    # Keep only one representative per pair: the one with the smaller linear index
    keep = idx < cidx
    pair_uv = torch.stack([uu_flat[keep], vv_flat[keep]], dim=1)       # (P, 2)
    pair_conj = torch.stack([cu[keep], cv[keep]], dim=1)               # (P, 2)
    return pair_uv, pair_conj


def _radial_distance(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Compute the normalised radial distance of every DFT bin from DC,
    using centered frequency coordinates (equivalent to fftshift logic).

    Returns a (h, w) float tensor in [0, 1] where 0 is DC and 1 is the
    corner of the Nyquist box.
    """
    # Centered frequency indices: [-h//2 .. (h-1)//2], same for w
    fu = torch.arange(h, device=device, dtype=torch.float32)
    fv = torch.arange(w, device=device, dtype=torch.float32)
    # Map to centered coords: freq u for index i is  (i + h//2) % h - h//2
    # but equivalently: i  if i <= h//2  else i - h  (standard DFT convention)
    fu = torch.where(fu <= h // 2, fu, fu - h).float()
    fv = torch.where(fv <= w // 2, fv, fv - w).float()
    gu, gv = torch.meshgrid(fu, fv, indexing="ij")
    # Normalise so that the Nyquist corner has distance 1
    max_r = math.sqrt((h // 2) ** 2 + (w // 2) ** 2)
    if max_r == 0:
        max_r = 1.0
    r = torch.sqrt(gu * gu + gv * gv) / max_r
    return r  # (h, w)


def apply_frequency_mask(x: torch.Tensor, mask_percent: float, spectral_type: str = "all") -> torch.Tensor:
    """Zero-out a random subset of DFT coefficients while preserving the
    Hermitian symmetry required for a real-valued inverse transform.

    Fixes over the previous implementation
    ----------------------------------------
    1. **Conjugate symmetry**: bins are always masked in conjugate pairs
       (u, v) and (-u mod h, -v mod w) so that ifft2(...).imag ≈ 0.
    2. **Radial frequency bands**: "low" / "high" are defined by a radial
       distance from DC (using centered-frequency coordinates, i.e. the
       fftshift convention) with a cutoff at 0.5 × max_radius.
    3. **Self-conjugate bin protection**: DC (0, 0) *and* all Nyquist /
       self-conjugate bins are excluded from masking so the output is
       guaranteed real-valued.
    """
    mask_percent = float(mask_percent)
    mask_percent = max(0.0, min(100.0, mask_percent))
    b, c, h, w = x.shape
    if mask_percent <= 0.0:
        return x

    x_fft = torch.fft.fft2(x.to(torch.float32), dim=(-2, -1), norm="ortho")
    st = str(spectral_type).lower()
    if st not in ("all", "low", "high"):
        st = "all"

    device = x.device

    # ------------------------------------------------------------------
    # 1. Enumerate every unique conjugate pair (excluding self-conjugate bins)
    # ------------------------------------------------------------------
    pair_uv, pair_conj = _build_conjugate_pairs(h, w, device)
    # pair_uv: (P, 2), pair_conj: (P, 2)  where P = number of unique pairs

    # ------------------------------------------------------------------
    # 2. Build a radial-distance map and determine which *pairs* fall
    #    inside the requested spectral band.
    # ------------------------------------------------------------------
    r = _radial_distance(h, w, device)  # (h, w), values in [0, 1]
    cutoff = 0.5  # normalised radius separating low from high

    if st == "all":
        pair_in_band = torch.ones(pair_uv.shape[0], device=device, dtype=torch.bool)
    else:
        # Use the radial distance of the representative bin of each pair
        r_pair = r[pair_uv[:, 0], pair_uv[:, 1]]
        if st == "low":
            pair_in_band = r_pair <= cutoff
        else:  # "high"
            pair_in_band = r_pair > cutoff

    # Indices (into the pair list) that are eligible for masking
    eligible_idx = pair_in_band.nonzero(as_tuple=False).squeeze(1)
    num_eligible = int(eligible_idx.numel())

    if num_eligible <= 0:
        x_masked = x_fft
    else:
        # Number of *pairs* to zero-out
        k = int(math.ceil((mask_percent / 100.0) * num_eligible))
        k = min(k, num_eligible)
        if k <= 0:
            x_masked = x_fft
        else:
            # ----------------------------------------------------------
            # 3. Per-sample random pair selection → symmetric mask
            # ----------------------------------------------------------
            mask_batch = []
            for _ in range(b):
                perm = torch.randperm(num_eligible, device=device)[:k]
                chosen = eligible_idx[perm]  # indices into pair_uv / pair_conj

                flat_mask = torch.ones(h * w, device=device, dtype=x_fft.dtype)

                # Zero both the representative and its conjugate partner
                uv = pair_uv[chosen]          # (k, 2)
                cv = pair_conj[chosen]         # (k, 2)
                flat_mask[uv[:, 0] * w + uv[:, 1]] = 0
                flat_mask[cv[:, 0] * w + cv[:, 1]] = 0

                mask_batch.append(flat_mask.view(1, h, w))

            mask = torch.stack(mask_batch, dim=0)  # (B, 1, h, w)
            x_masked = x_fft * mask

    x_rec = torch.fft.ifft2(x_masked, dim=(-2, -1), norm="ortho").real
    # x_rec = x_rec.clamp(0.0, 1.0)
    return x_rec


def build_random_square_mask(h: int, w: int, ratio: float, num_squares: int = 1,
                             generator: torch.Generator = None) -> torch.Tensor:
    total_area = int(round(ratio * h * w))
    if total_area <= 0 or num_squares <= 0:
        return torch.zeros((h, w), dtype=torch.float32)

    side = int(round(math.sqrt(total_area / float(max(num_squares, 1)))))
    side = max(1, min(side, min(h, w)))
    max_y0 = max(0, h - side)
    max_x0 = max(0, w - side)
    mask = torch.zeros((h, w), dtype=torch.float32)
    placed = []

    def overlaps(y0, x0, s, others):
        for (yy, xx, ss) in others:
            if not (x0 + s <= xx or xx + ss <= x0 or y0 + s <= yy or yy + ss <= y0):
                return True
        return False

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


def _forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    try:
        return model(x, return_attn=False)
    except TypeError:
        try:
            return model(x)
        except Exception:
            return model(x)


@ADAPTATION_REGISTRY.register()
class M2A(TTAMethod):
    """M2A adaptation with random spatial/spectral masking.
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.entropy = Entropy()

        self.m = float(cfg.M2A.M)
        self.n = int(cfg.M2A.N)
        self.mn = [i * self.m for i in range(self.n)]
        self.lamb_erl = float(cfg.M2A.LAMBDA_ERL)
        self.lamb_eml = float(cfg.M2A.LAMBDA_EML)
        self.margin = float(cfg.M2A.MARGIN) * math.log(1000.0)

        self.disable_mcl = bool(cfg.M2A.DISABLE_MCL)
        self.disable_erl = bool(cfg.M2A.DISABLE_ERL)
        self.disable_eml = bool(cfg.M2A.DISABLE_EML)

        rm = str(cfg.M2A.RANDOM_MASKING).lower()
        self.random_masking = rm if rm in ["spatial", "spectral"] else "spatial"
        self.num_squares = max(1, int(cfg.M2A.NUM_SQUARES))

        mt = str(cfg.M2A.MASK_TYPE).lower()
        self.mask_type = mt if mt in ["binary", "gaussian", "mean"] else "binary"

        st_sp = str(cfg.M2A.SPATIAL_TYPE).lower()
        self.spatial_type = st_sp if st_sp in ["patch", "pixel"] else "patch"

        st_spec = str(cfg.M2A.SPECTRAL_TYPE).lower()
        self.spectral_type = st_spec if st_spec in ["all", "low", "high"] else "all"

        self._rng = torch.Generator(device="cpu")
        try:
            if cfg.M2A.SEED is not None and cfg.M2A.SEED >= 0:
                self._rng.manual_seed(int(cfg.M2A.SEED))
        except Exception:
            pass

    def loss_calculation(self, x):
        imgs_test = x[0]
        b, c, h, w = imgs_test.shape

        # Reset cached masked batch for this forward
        self._last_masked = None

        self.model.eval()
        outputs0 = _forward_logits(self.model, imgs_test)

        outputs_list = [outputs0]
        for m_val in self.mn[1:]:
            mfrac = m_val
            if self.random_masking == "spatial":
                xb = imgs_test.clone()
                x_blur = None
                if self.mask_type == "gaussian":
                    x_blur = gaussian_blur2d(xb, kernel_size=11, sigma=None)

                if self.spatial_type == "patch":
                    for bi in range(b):
                        mask_bw = build_random_square_mask(h, w, ratio=mfrac, num_squares=self.num_squares,
                                                           generator=self._rng).to(imgs_test.device)
                        mask_c = mask_bw.unsqueeze(0)
                        if self.mask_type == "binary":
                            xb[bi] = imgs_test[bi] * (1.0 - mask_c)
                        elif self.mask_type == "mean":
                            mean_val = imgs_test[bi].mean(dim=(1, 2), keepdim=True)
                            xb[bi] = imgs_test[bi] * (1.0 - mask_c) + mean_val * mask_c
                        elif self.mask_type == "gaussian" and x_blur is not None:
                            xb[bi] = imgs_test[bi] * (1.0 - mask_c) + x_blur[bi] * mask_c
                else:
                    total_pixels = h * w
                    k_pix = int(round(mfrac * total_pixels))
                    k_pix = max(0, min(k_pix, total_pixels))
                    for bi in range(b):
                        if k_pix > 0:
                            flat = torch.zeros((total_pixels,), device=imgs_test.device, dtype=torch.float32)
                            idx = torch.randperm(total_pixels, device=imgs_test.device)[:k_pix]
                            flat[idx] = 1.0
                            mask_bw = flat.view(h, w)
                        else:
                            mask_bw = torch.zeros((h, w), device=imgs_test.device, dtype=torch.float32)
                        mask_c = mask_bw.unsqueeze(0)
                        if self.mask_type == "binary":
                            xb[bi] = imgs_test[bi] * (1.0 - mask_c)
                        elif self.mask_type == "mean":
                            mean_val = imgs_test[bi].mean(dim=(1, 2), keepdim=True)
                            xb[bi] = imgs_test[bi] * (1.0 - mask_c) + mean_val * mask_c
                        elif self.mask_type == "gaussian" and x_blur is not None:
                            xb[bi] = imgs_test[bi] * (1.0 - mask_c) + x_blur[bi] * mask_c
            else:
                xb = apply_frequency_mask(imgs_test, mask_percent=(mfrac * 100.0), spectral_type=self.spectral_type)

            # Cache the most recent masked batch for optional analysis image saving
            try:
                self._last_masked = xb.detach().cpu()
            except Exception:
                self._last_masked = None

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

        erl = None
        if not self.disable_erl:
            entropys = [self.entropy(out) for out in outputs_list]
            erl_tmp = 0.0
            for i in range(len(self.mn)):
                for j in range(i + 1, len(self.mn)):
                    erl_tmp = erl_tmp + (F.relu(entropys[i] - entropys[j].detach() + self.margin)).mean()
            erl = erl_tmp
            if isinstance(erl, torch.Tensor) and erl.requires_grad:
                total_loss_terms.append(self.lamb_erl * erl)

        eml = None
        if not self.disable_eml:
            eml_terms = [self.entropy(out).mean() for out in outputs_list]
            if len(eml_terms) > 0:
                eml = sum(eml_terms) / float(len(eml_terms))
                if isinstance(eml, torch.Tensor) and eml.requires_grad:
                    total_loss_terms.append(self.lamb_eml * eml)

        if len(total_loss_terms) > 0:
            loss = total_loss_terms[0]
            for lt in total_loss_terms[1:]:
                loss = loss + lt
        else:
            loss = torch.zeros((), device=imgs_test.device, dtype=outputs0.dtype)

        return outputs0, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return outputs

    def collect_params(self):
        """Collect all trainable parameters following the original M2A strategy.

        Skip top layers and some normalization layers, then select LayerNorm weights/bias.
        """
        params = []
        param_names = []

        for nm, m in self.model.named_modules():
            if "layer4" in nm:
                continue
            if "blocks.9" in nm:
                continue
            if "blocks.10" in nm:
                continue
            if "blocks.11" in nm:
                continue
            if "norm." in nm:
                continue
            if nm in ["norm"]:
                continue
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"] and p.requires_grad:
                        params.append(p)
                        param_names.append(f"{nm}.{np}")

        return params, param_names

    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

