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


@ADAPTATION_REGISTRY.register()
class REM(TTAMethod):

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.entropy = Entropy()

        self.m = float(cfg.REM.M)
        self.n = int(cfg.REM.N)
        self.mn = [i * self.m for i in range(self.n)]
        self.lamb = float(cfg.REM.LAMBDA)
        self.margin = float(cfg.REM.MARGIN) * math.log(1000.0)

    def configure_model(self):
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
        params = []
        names = []

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
                        names.append(f"{nm}.{np}")

        return params, names

    def loss_calculation(self, x):
        imgs_test = x[0]

        self.model.eval()
        outputs0, attn = self.model(imgs_test, return_attn=True)

        attn_score = attn.mean(dim=1)[:, 0, 1:]
        num_tokens = attn_score.shape[-1]

        outputs_list = []

        for m_val in self.mn:
            if m_val == 0.0:
                outputs_list.append(outputs0)
            else:
                num_keep = int(num_tokens * (1.0 - m_val))
                num_keep = max(1, min(num_keep, num_tokens))
                len_keep = torch.topk(attn_score, num_keep, largest=False).indices
                out = self.model(imgs_test, len_keep=len_keep, return_attn=False)
                outputs_list.append(out)

        self.model.train()

        loss = torch.zeros((), device=imgs_test.device, dtype=outputs0.dtype)
        if len(self.mn) > 1:
            for i in range(1, len(self.mn)):
                loss = loss + softmax_entropy(outputs_list[i], outputs_list[0].detach()).mean()
                for j in range(1, i):
                    loss = loss + softmax_entropy(outputs_list[i], outputs_list[j].detach()).mean()

        entropys = [self.entropy(out) for out in outputs_list]
        lossn = torch.zeros((), device=imgs_test.device, dtype=outputs0.dtype)
        for i in range(len(self.mn)):
            for j in range(i + 1, len(self.mn)):
                lossn = lossn + (F.relu(entropys[i] - entropys[j].detach() + self.margin)).mean()

        loss = loss + self.lamb * lossn
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

