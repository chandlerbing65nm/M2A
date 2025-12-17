import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from robustbench.data import load_imagenetc
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import rem
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
    if cfg.MODEL.ADAPTATION == "REM":
        logger.info("test-time adaptation: REM")
        model = setup_rem(base_model)
        method_name = "rem"
    if getattr(args, "print_model", False):
        return
    # evaluate on each severity and type of corruption in turn
    n_recur = max(1, int(getattr(args, "recur", 1)))
    for r in range(n_recur):
        for ii, severity in enumerate(cfg.CORRUPTION.SEVERITY):
            severity_domains = {}
            for i_x, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
                # reset adaptation only on the first full pass
                # note: for evaluation protocol, but not necessarily needed
                domain_gen = bool(getattr(cfg.TEST, "DOMAIN_GEN", False))
                no_adapt_this = domain_gen and (i_x >= 10)
                try:
                    if r == 0 and i_x == 0 and not no_adapt_this:
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
                    model, x_test, y_test, cfg.TEST.BATCH_SIZE,
                    device=device,
                    tag=f"[{corruption_type}{severity}]",
                    no_adapt=bool(no_adapt_this),
                )
                err = 1. - acc
                logger.info(f"Error % [{corruption_type}{severity}]: {err:.2%}")
                logger.info(f"NLL [{corruption_type}{severity}]: {nll:.4f}")
                logger.info(f"ECE [{corruption_type}{severity}]: {ece:.4f}")

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
            
def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model

    
def setup_optimizer_rem(params):
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

def setup_rem(model):
    model = rem.configure_model(model)
    params, param_names = rem.collect_params(model)
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
    logger.info(f"model for adaptation: %s", rem_model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return rem_model


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
                core = _unwrap_base_model_for_features(model)
                if hasattr(core, 'forward_features'):
                    out = core.forward_features(x_b)
                    feats = out[0] if isinstance(out, (tuple, list)) else out
                    if hasattr(core, 'forward_head'):
                        logits = core.forward_head(feats)
                    elif hasattr(core, 'head') and isinstance(core.head, nn.Module):
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


def _unwrap_base_model_for_features(model: torch.nn.Module) -> torch.nn.Module:
    m = model
    for _ in range(4):
        if hasattr(m, "model"):
            m = getattr(m, "model")
            continue
        if hasattr(m, "module"):
            m = getattr(m, "module")
            continue
        break
    return m


def _extract_features_and_logits(base_model: torch.nn.Module,
                                 x_b: torch.Tensor):
    core = base_model
    if hasattr(core, "module"):
        core = core.module

    feats = None
    logits = None
    if hasattr(core, "forward_features"):
        out = core.forward_features(x_b)
        feats = out[0] if isinstance(out, (tuple, list)) else out
        if hasattr(core, "forward_head"):
            logits = core.forward_head(feats)
        elif hasattr(core, "head") and isinstance(core.head, nn.Module):
            logits = core.head(feats)
        else:
            logits = core(x_b)
    else:
        logits = core(x_b)
        feats = logits

    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return feats, logits


def _extract_blockwise_tokens_and_logits(core: torch.nn.Module, x_b: torch.Tensor):
    if hasattr(core, "module"):
        core = core.module
    if not (hasattr(core, "patch_embed") and hasattr(core, "cls_token") and hasattr(core, "pos_embed")
            and hasattr(core, "pos_drop") and hasattr(core, "blocks") and hasattr(core, "norm")):
        feats, logits = _extract_features_and_logits(core, x_b)
        return feats, logits, [], [], []

    x = core.patch_embed(x_b)
    cls_token = core.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x = core.pos_drop(x + core.pos_embed)

    class_tokens = []
    patch_means = []
    patch_stds = []

    for blk in core.blocks:
        x = blk(x)
        cls_t = x[:, 0, :]
        patches = x[:, 1:, :]
        patch_mean = patches.mean(dim=1)
        patch_std = patches.std(dim=1, unbiased=False)
        class_tokens.append(cls_t)
        patch_means.append(patch_mean)
        patch_stds.append(patch_std)

    x_norm = core.norm(x)
    feats = x_norm[:, 0]
    if hasattr(core, "forward_head"):
        logits = core.forward_head(feats)
    elif hasattr(core, "head") and isinstance(core.head, nn.Module):
        logits = core.head(feats)
    else:
        logits = core(x_b)

    return feats, logits, class_tokens, patch_means, patch_stds


def save_domain_features(method_name: str,
                         model: torch.nn.Module,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         severity: int,
                         corruption_type: str,
                         batch_size: int,
                         device: torch.device):
    try:
        base_model = _unwrap_base_model_for_features(model)
        total = x.shape[0]
        feat_list = []
        logit_list = []
        label_list = []

        cls_blocks_lists = None
        pmean_blocks_lists = None
        pstd_blocks_lists = None
        with torch.no_grad():
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                x_b = x[start:end].to(device)
                y_b = y[start:end].to(device)
                feats_b, logits_b, cls_blocks, pmeans_b, pstds_b = _extract_blockwise_tokens_and_logits(base_model, x_b)
                feat_list.append(feats_b.detach().cpu())
                logit_list.append(logits_b.detach().cpu())
                label_list.append(y_b.detach().cpu())
                if cls_blocks_lists is None:
                    L = len(cls_blocks)
                    cls_blocks_lists = [[] for _ in range(L)]
                    pmean_blocks_lists = [[] for _ in range(L)]
                    pstd_blocks_lists = [[] for _ in range(L)]
                for i_b, (cb, mb, sb) in enumerate(zip(cls_blocks, pmeans_b, pstds_b)):
                    cls_blocks_lists[i_b].append(cb.detach().cpu())
                    pmean_blocks_lists[i_b].append(mb.detach().cpu())
                    pstd_blocks_lists[i_b].append(sb.detach().cpu())

        if not feat_list:
            return None

        feats_t = torch.cat(feat_list, dim=0)
        logits_t = torch.cat(logit_list, dim=0)
        labels_t = torch.cat(label_list, dim=0)
        probs_t = logits_t.softmax(dim=1)
        preds_t = probs_t.argmax(dim=1)

        domain_id = f"{corruption_type}_{severity}"
        domain_data = {
            "domain_id": domain_id,
            "method": method_name,
            "features": feats_t.numpy(),
            "logits": logits_t.numpy(),
            "probabilities": probs_t.numpy(),
            "predictions": preds_t.numpy(),
            "labels": labels_t.numpy(),
        }
        if cls_blocks_lists is not None:
            for i in range(len(cls_blocks_lists)):
                cls_cat = torch.cat(cls_blocks_lists[i], dim=0).numpy()
                pm_cat = torch.cat(pmean_blocks_lists[i], dim=0).numpy()
                ps_cat = torch.cat(pstd_blocks_lists[i], dim=0).numpy()
                domain_data[f"class_features_{i+1}"] = cls_cat
                domain_data[f"patch_mean_{i+1}"] = pm_cat
                domain_data[f"patch_std_{i+1}"] = ps_cat
        return domain_data
    except Exception as e:
        logger.warning(f"Failed to build features for {corruption_type}{severity}: {e}")
        return None


def save_severity_features(method_name: str,
                           severity: int,
                           domains: dict) -> None:
    try:
        if not domains:
            return
        save_dir = "/flash/project_465002264/projects/m2a/feat"
        os.makedirs(save_dir, exist_ok=True)
        dataset_tag = "imagenetc"
        filename = f"{method_name}_{severity}_{dataset_tag}.npy"
        path = os.path.join(save_dir, filename)
        np.save(path, domains, allow_pickle=True)
        logger.info(f"Saved features to: {path}")
    except Exception as e:
        logger.warning(f"Failed to save severity features for {severity}: {e}")


      
if __name__ == '__main__':
    evaluate('"Imagenet-C evaluation.')
