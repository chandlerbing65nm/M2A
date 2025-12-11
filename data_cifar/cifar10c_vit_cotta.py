import logging

import os

import torch
import torch.optim as optim
import torch.nn.functional as F
import time

from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy
from collections import OrderedDict

import cotta
import torch.nn as nn
from conf import cfg, load_cfg_fom_args

logger = logging.getLogger(__name__)

def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:  # to delete prefix 'module.' if it exists
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict

def evaluate(description):
    """Evaluate M2A: Stochastic Patch Erasing with Adaptive Residual Correction
    for Continual Test-Time Adaptation (CTTA) on CIFAR-10-C.

    The evaluation iterates over corruptions/severities and measures accuracy, NLL, ECE,
    as well as adaptation-time metrics relevant to CTTA.
    """
    args = load_cfg_fom_args(description)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("[cifar10] RNG seed in use: %d", cfg.RNG_SEED)
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions)
    checkpoint = torch.load("/users/doloriel/work/Repo/M2A/ckpt/vit_base_384_cifar10.t7", map_location='cpu')
    checkpoint = rm_substr_from_state_dict(checkpoint['model'], 'module.') if isinstance(checkpoint, dict) else checkpoint
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        base_model.load_state_dict(checkpoint['model'], strict=True)
    else:
        base_model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    # Apply potential adaptation checkpoint (optional)
    if cfg.TEST.ckpt is not None:
        if device.type == 'cuda':
            base_model = torch.nn.DataParallel(base_model)
        ckpt = torch.load(cfg.TEST.ckpt, map_location='cpu')
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        base_model.load_state_dict(state, strict=False)
    else:
        if device.type == 'cuda':
            base_model = torch.nn.DataParallel(base_model)
    base_model.to(device)

    if cfg.MODEL.ADAPTATION == "cotta":
        logger.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model)
    else:
        raise ValueError("Unknown adaptation method: {}".format(cfg.MODEL.ADAPTATION))
    if getattr(cfg, "PRINT_MODEL", False):
        return

    # evaluate on each severity and type of corruption in turn
    All_error = []
    for severity in cfg.CORRUPTION.SEVERITY:
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            if i_c == 0:
                try:
                    model.reset()
                    logger.info("resetting model")
                except:
                    logger.warning("not resetting model")
            else:
                logger.warning("not resetting model")
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test = F.interpolate(x_test, size=(args.size, args.size), \
                mode='bilinear', align_corners=False)

            if hasattr(model, 'reset_loss_stats'):
                try:
                    model.reset_loss_stats()
                except Exception:
                    pass

            acc, nll, ece, max_softmax, entropy, cos_sim, total_cnt, adapt_time_total, adapt_macs_total, mcl_last, erl_last, eml_last = compute_metrics(
                model, x_test, y_test, cfg.TEST.BATCH_SIZE, device=device
            )
            err = 1. - acc
            All_error.append(err)
            logger.info(f"Error % [{corruption_type}{severity}]: {err:.2%}")
            logger.info(f"NLL [{corruption_type}{severity}]: {nll:.4f}")
            logger.info(f"ECE [{corruption_type}{severity}]: {ece:.4f}")
            # logger.info(f"Entropy [{corruption_type}{severity}]: {entropy:.4f}")
            # logger.info(f"Adaptation Time (lower is better) [{corruption_type}{severity}]: {adapt_time_total:.3f}s")

    # Save checkpoint after full evaluation if requested
    try:
        if args.save_ckpt:
            method = str(cfg.MODEL.ADAPTATION).lower()
            arch_tag = str(cfg.MODEL.ARCH).replace('/', '').replace('-', '').replace('_', '').lower()
            dataset_tag = 'cifar10c'
            ckpt_dir = '/flash/project_465002264/projects/m2a/ckpt'
            os.makedirs(ckpt_dir, exist_ok=True)
            filename = f"{method}_{arch_tag}_{dataset_tag}.pth"
            path = os.path.join(ckpt_dir, filename)
            save_model = model
            if hasattr(save_model, 'model'):
                save_model = save_model.model
            if hasattr(save_model, 'module'):
                save_model = save_model.module
            torch.save({'model': save_model.state_dict()}, path)
            logger.info(f"Saved checkpoint to: {path}")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")



def setup_cotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           ap=cfg.OPTIM.AP,
                           size = cfg.size)
    logger.info(f"model for adaptation: %s", cotta_model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


def compute_metrics(model: torch.nn.Module,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    batch_size: int = 100,
                    device: torch.device = None):
    if device is None:
        device = x.device
    if isinstance(device, str):
        device = torch.device(device)
    total_cnt = x.shape[0]
    n_batches = int((total_cnt + batch_size - 1) // batch_size)

    correct = 0
    total_eval = 0
    nll_sum = 0.0
    max_softmax_sum = 0.0
    entropy_sum = 0.0
    cos_sum = 0.0
    confs_all = []
    correct_all = []
    adapt_time_total = 0.0
    adapt_macs_total = 0

    def unwrap_model(m):
        try:
            while True:
                if hasattr(m, 'module'):
                    m = m.module
                elif hasattr(m, 'model'):
                    m = getattr(m, 'model')
                else:
                    break
        except Exception:
            pass
        return m

    def estimate_vit_macs_per_image(stats_src, img_size: int) -> int:
        try:
            m = unwrap_model(stats_src)
            if hasattr(m, 'patch_embed') and hasattr(m.patch_embed, 'proj'):
                ps = m.patch_embed.proj.kernel_size[0]
                seq = (img_size // ps) ** 2 + 1
            else:
                ps = 16
                seq = (img_size // ps) ** 2 + 1
            heads = getattr(getattr(m, 'blocks', [None])[0], 'attn', None)
            num_heads = getattr(heads, 'num_heads', 12) if heads is not None else 12
            d_model = getattr(m, 'embed_dim', 768)
            attn_cost = 2 * (seq ** 2) * num_heads
            proj_cost = 3 * seq * d_model * d_model
            mlp_cost = 2 * seq * d_model * (4 * d_model)
            blocks = len(getattr(m, 'blocks', [])) or 12
            total = blocks * (attn_cost + proj_cost + mlp_cost)
            return int(total)
        except Exception:
            return 0

    per_img_macs = estimate_vit_macs_per_image(model, img_size=x.shape[-1])

    for i in range(n_batches):
        lo = i * batch_size
        hi = min((i + 1) * batch_size, total_cnt)
        x_b = x[lo:hi].to(device)
        y_b = y[lo:hi].to(device)
        t0 = time.time()
        output = model(x_b)
        adapt_time_total += (time.time() - t0)
        adapt_macs_total += per_img_macs * int(x_b.shape[0])

        logits = output if isinstance(output, torch.Tensor) else output[0]
        preds = logits.argmax(dim=1)
        correct += (preds == y_b).float().sum().item()
        total_eval += y_b.shape[0]
        nll_sum += F.cross_entropy(logits, y_b, reduction='sum').item()
        probs = logits.softmax(dim=1)
        confs = probs.max(dim=1).values
        ents = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
        one_hot = F.one_hot(y_b.long(), num_classes=logits.shape[-1]).float()
        cos_b = F.cosine_similarity(probs, one_hot, dim=1)
        confs_all.append(confs.detach().cpu())
        correct_all.append((preds == y_b).detach().cpu())
        max_softmax_sum += float(confs.sum().item())
        entropy_sum += float(ents.sum().item())
        cos_sum += float(cos_b.sum().item())

    if total_eval == 0:
        mcl_last = getattr(model, 'last_mcl', 0.0)
        erl_last = getattr(model, 'last_erl', 0.0)
        eml_last = getattr(model, 'last_eml', 0.0)
        try:
            mcl_last = float(mcl_last)
            erl_last = float(erl_last)
            eml_last = float(eml_last)
        except Exception:
            mcl_last, erl_last, eml_last = 0.0, 0.0, 0.0
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, total_cnt, adapt_time_total, adapt_macs_total, mcl_last, erl_last, eml_last

    acc = correct / total_eval
    nll = nll_sum / total_eval
    max_softmax = max_softmax_sum / total_eval if total_eval > 0 else 0.0
    entropy = entropy_sum / total_eval if total_eval > 0 else 0.0
    cos_sim = cos_sum / total_eval if total_eval > 0 else 0.0
    confs_all = torch.cat(confs_all) if len(confs_all) else torch.empty(0)
    correct_all = torch.cat(correct_all).float() if len(correct_all) else torch.empty(0)
    ece = compute_ece(confs_all, correct_all)

    mcl_last = getattr(model, 'last_mcl', 0.0)
    erl_last = getattr(model, 'last_erl', 0.0)
    eml_last = getattr(model, 'last_eml', 0.0)
    try:
        count = float(getattr(model, 'loss_count', 0.0))
        if count > 0.0:
            mcl_sum = float(getattr(model, 'mcl_sum', 0.0))
            erl_sum = float(getattr(model, 'erl_sum', 0.0))
            eml_sum = float(getattr(model, 'eml_sum', 0.0))
            mcl_last = mcl_sum / count
            erl_last = erl_sum / count
            eml_last = eml_sum / count
        else:
            mcl_last = float(mcl_last)
            erl_last = float(erl_last)
            eml_last = float(eml_last)
    except Exception:
        try:
            mcl_last = float(mcl_last)
            erl_last = float(erl_last)
            eml_last = float(eml_last)
        except Exception:
            mcl_last, erl_last, eml_last = 0.0, 0.0, 0.0

    return acc, nll, ece, max_softmax, entropy, cos_sim, total_cnt, adapt_time_total, adapt_macs_total, mcl_last, erl_last, eml_last


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
    evaluate('"CIFAR-10-C evaluation.')