import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from robustbench.data import load_avffiacc

from robustbench.model_zoo.rem_vit import create_model_rem

import rem
from conf import cfg, load_cfg_fom_args
import operators

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

def _build_model_from_ckpt(num_classes: int, device: torch.device) -> nn.Module:
    # Build ViT-Base/16 REM variant with desired num_classes
    model = create_model_rem("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    # Load local checkpoint trained with data_avffia/train.py
    ckpt_path = getattr(cfg.TEST, 'ckpt', None)
    if ckpt_path is None or (isinstance(ckpt_path, str) and ckpt_path.strip() == ""):
        ckpt_path = "/users/doloriel/work/Repo/M2A/ckpt/uffia_vitb16_best.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model
 

def evaluate(description):
    args = load_cfg_fom_args(description)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # configure model
    base_model = _build_model_from_ckpt(num_classes=4, device=device)
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    elif cfg.MODEL.ADAPTATION == "REM":
        logger.info("test-time adaptation: REM")
        model = setup_rem(base_model)
    else:
        logger.info("test-time adaptation: %s not supported, defaulting to NONE", cfg.MODEL.ADAPTATION)
        model = setup_source(base_model)
    # evaluate on each severity and type of corruption in turn
    for ii, severity in enumerate(cfg.CORRUPTION.SEVERITY):
        for i_x, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            try:
                if i_x == 0:
                    model.reset()
                    logger.info("resetting model")
                else:
                    logger.warning(" ")
                    logger.warning("not resetting model")
            except Exception:
                logger.warning(" ")
                logger.warning("not resetting model")
            x_test, y_test = load_avffiacc(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            acc, nll, ece, max_softmax, entropy = compute_metrics(
                model, x_test, y_test, cfg.TEST.BATCH_SIZE, device=device
            )
            err = 1. - acc
            logger.info(f"Error % [{corruption_type}{severity}]: {err:.2%}")
            logger.info(f"NLL [{corruption_type}{severity}]: {nll:.4f}")
            logger.info(f"ECE [{corruption_type}{severity}]: {ece:.4f}")
            logger.info(f"Max Softmax [{corruption_type}{severity}]: {max_softmax:.4f}")
            logger.info(f"Entropy [{corruption_type}{severity}]: {entropy:.4f}")
            

def setup_source(model):
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_optimizer_rem(params):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam([{"params": params, "lr": cfg.OPTIM.LR}],
                          lr=cfg.OPTIM.LR,
                          betas=(cfg.OPTIM.BETA, 0.999),
                          weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD([{"params": params, "lr": cfg.OPTIM.LR}],
                         lr=cfg.OPTIM.LR,
                         momentum=0.9,
                         dampening=0,
                         weight_decay=cfg.OPTIM.WD,
                         nesterov=True)
    else:
        raise NotImplementedError


def setup_rem(model):
    model = rem.configure_model(model)
    params = rem.collect_params(model)
    optimizer = setup_optimizer_rem(params)
    rem_model = rem.REM(model, optimizer,
                           len_num_keep=cfg.OPTIM.KEEP,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           m = cfg.OPTIM.M,
                           n = cfg.OPTIM.N,
                           lamb = cfg.OPTIM.LAMB,
                           margin = cfg.OPTIM.MARGIN,
                           )
    # logger.info(f"model for adaptation: %s", model)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return rem_model


def compute_metrics(model: nn.Module,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    batch_size: int = 64,
                    device: torch.device = None):
    if device is None:
        device = x.device
    if isinstance(device, str):
        device = torch.device(device)
    total_N = x.shape[0]
    n_batches = int((total_N + batch_size - 1) // batch_size)

    correct = 0
    total_eval = 0
    nll_sum = 0.0
    max_softmax_sum = 0.0
    entropy_sum = 0.0
    confs_all = []
    correct_all = []

    with torch.no_grad():
        for b in range(n_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, total_N)
            x_b = x[start:end].to(device)
            y_b = y[start:end].to(device)

            logits = model(x_b)
            preds = logits.argmax(dim=1)
            correct += (preds == y_b).float().sum().item()
            total_eval += y_b.shape[0]

            nll_sum += F.cross_entropy(logits, y_b, reduction='sum').item()
            probs = logits.softmax(dim=1)
            confs = probs.max(dim=1).values
            ents = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)

            confs_all.append(confs.detach().cpu())
            correct_all.append((preds == y_b).detach().cpu())
            max_softmax_sum += float(confs.sum().item())
            entropy_sum += float(ents.sum().item())

    if total_eval == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    acc = correct / total_eval
    nll = nll_sum / total_eval
    max_softmax = max_softmax_sum / total_eval if total_eval > 0 else 0.0
    entropy = entropy_sum / total_eval if total_eval > 0 else 0.0
    confs_all = torch.cat(confs_all) if len(confs_all) else torch.empty(0)
    correct_all = torch.cat(correct_all).float() if len(correct_all) else torch.empty(0)
    ece = compute_ece(confs_all, correct_all)
    return acc, nll, ece, max_softmax, entropy


def compute_ece(confs: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> float:
    if confs.numel() == 0:
        return 0.0
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1)
    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confs >= lo) & (confs <= hi)
        else:
            in_bin = (confs >= lo) & (confs < hi)
        count = in_bin.sum().item()
        if count == 0:
            continue
        conf_bin = confs[in_bin].mean().item()
        acc_bin = correct[in_bin].mean().item()
        prop = count / confs.numel()
        ece += abs(acc_bin - conf_bin) * prop
    return float(ece)

      
if __name__ == '__main__':
    evaluate('AVFFIA-C evaluation.')
