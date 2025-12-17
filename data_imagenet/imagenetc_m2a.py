import logging
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from robustbench.data import load_imagenetc
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import m2a
from conf import cfg, load_cfg_fom_args
import operators

import numpy as np
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)


def evaluate(description):
    args = load_cfg_fom_args(description)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                            cfg.CORRUPTION.DATASET, ThreatModel.corruptions)
    base_model.to(device)
    method_name = None
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
        method_name = "source"
    if cfg.MODEL.ADAPTATION == "M2A":
        logger.info("test-time adaptation: M2A")
        model = setup_m2a(base_model)
        method_name = "m2a"
    if getattr(args, "print_model", False):
        return
    # evaluate on each severity and type of corruption in turn
    n_recur = max(1, int(getattr(args, "recur", 1)))
    for r in range(n_recur):
        for ii, severity in enumerate(cfg.CORRUPTION.SEVERITY):
            severity_domains = {}
            for i_x, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
                # domain generalization: adapt only on first 10 corruptions, then
                # evaluate without further adaptation on the rest
                domain_gen = bool(getattr(cfg.TEST, "DOMAIN_GEN", False))
                no_adapt_this = domain_gen and (i_x >= 10)

                # reset adaptation only on the first corruption of the first pass
                try:
                    if (r == 0 and i_x == 0) and not no_adapt_this:
                        model.reset()
                        logger.info("resetting model")
                    else:
                        logger.warning(" ")
                        logger.warning("not resetting model")
                except Exception:
                    logger.warning(" ")
                    logger.warning("not resetting model")

                x_test, y_test = load_imagenetc(cfg.CORRUPTION.NUM_EX,
                                               severity, cfg.DATA_DIR, False,
                                               [corruption_type])
                acc, nll, ece, max_softmax, entropy = compute_metrics(
                    model, x_test, y_test, cfg.TEST.BATCH_SIZE, device=device,
                    tag=f"[{corruption_type}{severity}]",
                    no_adapt=bool(no_adapt_this),
                )
                err = 1. - acc
                logger.info(f"Error % [{corruption_type}{severity}]: {err:.2%}")
                logger.info(f"NLL [{corruption_type}{severity}]: {nll:.4f}")
                logger.info(f"ECE [{corruption_type}{severity}]: {ece:.4f}")

                # Optional feature saving per corruption
                if getattr(args, "save_feat", False) and method_name is not None:
                    domain_data = save_domain_features(
                        method_name=method_name,
                        model=model,
                        x=x_test,
                        y=y_test,
                        severity=severity,
                        corruption_type=corruption_type,
                        batch_size=cfg.TEST.BATCH_SIZE,
                        device=device,
                    )
                    if domain_data is not None:
                        domain_id = domain_data.get("domain_id", f"{corruption_type}_{severity}")
                        severity_domains[domain_id] = domain_data

            if getattr(args, "save_feat", False) and method_name is not None:
                save_severity_features(method_name, severity, severity_domains)

    try:
        if getattr(args, "save_ckpt", False):
            method = str(cfg.MODEL.ADAPTATION).lower()
            arch_tag = str(cfg.MODEL.ARCH).replace('/', '').replace('-', '').replace('_', '').lower()
            dataset_tag = 'imagenetc'
            ckpt_dir = '/flash/project_465002264/projects/m2a/ckpt'
            os.makedirs(ckpt_dir, exist_ok=True)
            mask_tag = f"_{str(args.random_masking).lower()}" if (method == 'm2a' and getattr(args, 'random_masking', None)) else ""
            filename = f"{method}_{arch_tag}{mask_tag}_{dataset_tag}.pth"
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
            
def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model

    
def setup_optimizer_m2a(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
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

def setup_m2a(model):
    model = m2a.configure_model(model)
    params, param_names = m2a.collect_params(model)
    optimizer = setup_optimizer_m2a(params)
    m2a_model = m2a.M2A(
        model, optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        m=cfg.OPTIM.M,
        n=cfg.OPTIM.N,
        lamb=cfg.OPTIM.LAMB,
        margin=cfg.OPTIM.MARGIN,
        random_masking=cfg.M2A.RANDOM_MASKING,
        num_squares=cfg.M2A.NUM_SQUARES,
        mask_type=cfg.M2A.MASK_TYPE,
        spatial_type=cfg.M2A.SPATIAL_TYPE,
        spectral_type=cfg.M2A.SPECTRAL_TYPE,
        seed=cfg.RNG_SEED,
        disable_mcl=cfg.M2A.DISABLE_MCL,
        disable_erl=cfg.M2A.DISABLE_ERL,
        disable_eml=cfg.M2A.DISABLE_EML,
    )
    logger.info(f"model for adaptation: %s", m2a_model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return m2a_model


def compute_metrics(model: nn.Module,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    batch_size: int = 64,
                    device: torch.device = None,
                    tag: str = "",
                    no_adapt: bool = False):
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

            if not no_adapt:
                logits = model(x_b)
            else:
                # bypass adaptation: use underlying backbone only
                core = _unwrap_base_model_for_features(model)
                if hasattr(core, 'forward_features'):
                    out = core.forward_features(x_b)
                    feats = out[0] if isinstance(out, (tuple, list)) else out
                    if hasattr(core, 'forward_head'):
                        logits = core.forward_head(feats)
                    elif hasattr(core, 'head') and isinstance(core.head, torch.nn.Module):
                        logits = core.head(feats)
                    else:
                        logits = core(x_b)
                else:
                    logits = core(x_b)
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

            if getattr(cfg.TEST, "BATCH_METRICS", False):
                batch_acc = correct / max(1, total_eval)
                batch_err = 1.0 - batch_acc
                batch_nll = nll_sum / max(1, total_eval)
                batch_ece = compute_ece(confs.detach(), (preds == y_b).float().detach())
                prefix = f"{tag} " if tag else ""
                logger.info(
                    f"[BATCH_METRICS] {prefix}batch {b}: Error %: {batch_err:.2%}, "
                    f"NLL: {batch_nll:.4f}, ECE: {batch_ece:.4f}"
                )

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


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_optimizer_m2a(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
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


def setup_m2a(model):
    model = m2a.configure_model(model)
    params, param_names = m2a.collect_params(model)
    optimizer = setup_optimizer_m2a(params)
    m2a_model = m2a.M2A(
        model, optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        m=cfg.OPTIM.M,
        n=cfg.OPTIM.N,
        lamb=cfg.OPTIM.LAMB,
        margin=cfg.OPTIM.MARGIN,
        random_masking=cfg.M2A.RANDOM_MASKING,
        num_squares=cfg.M2A.NUM_SQUARES,
        mask_type=cfg.M2A.MASK_TYPE,
        spatial_type=cfg.M2A.SPATIAL_TYPE,
        spectral_type=cfg.M2A.SPECTRAL_TYPE,
        seed=cfg.RNG_SEED,
        disable_mcl=cfg.M2A.DISABLE_MCL,
        disable_erl=cfg.M2A.DISABLE_ERL,
        disable_eml=cfg.M2A.DISABLE_EML,
    )
    logger.info(f"model for adaptation: %s", m2a_model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return m2a_model


if __name__ == '__main__':
    evaluate('"Imagenet-C evaluation.')
