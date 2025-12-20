import logging

import os

import torch
import torch.optim as optim
import torch.nn.functional as F

from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import tent
import cotta
import vida
import continual_mae
import torch.nn as nn
from conf import cfg, load_cfg_fom_args
import operators
from utils_cdc import create_cdc_sequence
from collections import OrderedDict
import numpy as np
import random

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
    args = load_cfg_fom_args(description)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions)
    checkpoint = torch.load("/users/doloriel/work/Repo/M2A/ckpt/vit_base_384_cifar10.t7")
    checkpoint = rm_substr_from_state_dict(checkpoint['model'], 'module.')
    base_model.load_state_dict(checkpoint, strict=True)
    if cfg.TEST.ckpt is not None:
        base_model = torch.nn.DataParallel(base_model) # make parallel
        checkpoint = torch.load(cfg.TEST.ckpt)
        base_model.load_state_dict(checkpoint['model'], strict=False)
    else:
        base_model = torch.nn.DataParallel(base_model) # make parallel
    base_model.to(device)

    head_dim = 768
    if cfg.use_hog:
        nbins = 9
        cell_sz = 8
        hogs = operators.HOGLayerC(
                nbins=nbins,
                pool=cell_sz
            )
        hogs = nn.DataParallel(hogs) # make parallel
        hogs.to(device)

        # hog_projection
        num_class = int(nbins*3*(16/cell_sz)*(16/cell_sz)) 
        projections = nn.Linear(head_dim, num_class, bias=True)
        if isinstance(projections, nn.Linear):
            nn.init.trunc_normal_(projections.weight, std=0.02)
            if isinstance(projections, nn.Linear) and projections.bias is not None:
                nn.init.constant_(projections.bias, 0)
        projections = nn.DataParallel(projections) # make parallel
        projections.to(device)
    else:
        hogs = None
        projections = None
    

    # mask_token
    mask_token_dim = (1, 1, head_dim)
    mask_token = nn.Parameter(torch.zeros(*mask_token_dim), requires_grad=True)
    mask_token = nn.DataParallel(mask_token) # make parallel
    mask_token.to(device)


    if cfg.MODEL.ADAPTATION == "Continual_MAE":
        logger.info("test-time adaptation: Continual_MAE")
        model = setup_continual_mae(base_model, hogs=hogs, projections=projections, mask_token=mask_token, hog_ratio=cfg.hog_ratio)
        method_name = "continual_mae"
    else:
        raise ValueError(f"Unknown adaptation method: {cfg.MODEL.ADAPTATION}")
    if getattr(cfg, "PRINT_MODEL", False):
        return
    if bool(getattr(cfg.TEST, "ENABLE_CDC", False)):
        corruptions = cfg.CORRUPTION.TYPE
        num_total_batches = cfg.CORRUPTION.NUM_EX // cfg.TEST.BATCH_SIZE + 1
        domain_order = create_cdc_sequence(num_total_batches=num_total_batches)

        corruption_res = [0] * len(corruptions)
        corruption_idx = [0] * len(corruptions)
        total = 0

        for severity in cfg.CORRUPTION.SEVERITY:
            all_loaders = []
            for i_x, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
                x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                               severity, cfg.DATA_DIR, False,
                                               [corruption_type])
                # Keep x_test compact; upsample per-batch only.
                all_loaders.append((x_test, y_test))
                logger.info(f"[{corruption_type}{severity}] Loaded")

            for i_xx, order_i in enumerate(domain_order):
                x_test, y_test = all_loaders[order_i]
                start = corruption_idx[order_i]
                end = start + cfg.TEST.BATCH_SIZE
                if start >= x_test.shape[0]:
                    continue
                x_curr = x_test[start:end]
                x_curr = F.interpolate(x_curr, size=(args.size, args.size),
                                       mode='bilinear', align_corners=False)
                x_curr = x_curr.to(device)
                y_curr = y_test[start:end].to(device)
                corruption_idx[order_i] += cfg.TEST.BATCH_SIZE

                output = model(x_curr)
                output = output[0] if isinstance(output, tuple) else output
                preds = output.max(1)[1]
                correct = (preds == y_curr).float().sum().item()
                corruption_res[order_i] += correct
                batch_size_curr = y_curr.shape[0]
                total += batch_size_curr
                if batch_size_curr > 0:
                    acc_curr = correct / batch_size_curr
                else:
                    acc_curr = 0.0
                err_curr = 1.0 - acc_curr
                if total > 0:
                    acc_running = sum(corruption_res) / total
                else:
                    acc_running = 0.0
                err_running = 1.0 - acc_running
                logger.info(
                    f"[{i_xx}/{len(domain_order)}: {corruptions[order_i]}] current error: {err_curr:.2%}, running error: {err_running:.2%}"
                )

        All_error = [1.0 - (float(corr) / float(cfg.CORRUPTION.NUM_EX)) for corr in corruption_res]
        all_error_res = ' '.join([f"{e:.2%}" for e in All_error])
        logger.info(f"All error: {all_error_res}")
        logger.info(f"Mean error: {sum(All_error) / len(All_error):.2%}")
        return

    # evaluate on each severity and type of corruption in turn
    All_error = []
    use_rand_domain = bool(getattr(cfg.TEST, "RAND_DOMAIN", False))
    n_permutations = 10 if use_rand_domain else 1
    for severity in cfg.CORRUPTION.SEVERITY:
        severity_domains = {}
        for perm_idx in range(n_permutations):
            if use_rand_domain and perm_idx > 0:
                order = np.random.permutation(len(cfg.CORRUPTION.TYPE))
            else:
                order = range(len(cfg.CORRUPTION.TYPE))
            for pos_in_order, idx in enumerate(order):
                corruption_type = cfg.CORRUPTION.TYPE[idx]
                domain_gen = bool(getattr(cfg.TEST, "DOMAIN_GEN", False))
                no_adapt_this = domain_gen and (pos_in_order >= 10)
                if not no_adapt_this:
                    if pos_in_order == 0:
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
                    model, x_test, y_test, cfg.TEST.BATCH_SIZE, device=device,
                    tag=f"[{corruption_type}{severity}]", no_adapt=bool(no_adapt_this)
                )
                err = 1. - acc
                All_error.append(err)
                logger.info(f"Error % [{corruption_type}{severity}]: {err:.2%}")
                logger.info(f"NLL [{corruption_type}{severity}]: {nll:.4f}")
                logger.info(f"ECE [{corruption_type}{severity}]: {ece:.4f}")
            # logger.info(f"Entropy [{corruption_type}{severity}]: {entropy:.4f}")
            # logger.info(f"Adaptation Time (lower is better) [{corruption_type}{severity}]: {adapt_time_total:.3f}s")

                if getattr(args, "save_feat", False):
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

            if use_rand_domain:
                print(f"{perm_idx + 1} random iteration done!")

        if getattr(args, "save_feat", False):
            save_severity_features(method_name, severity, severity_domains)

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

def setup_continual_mae(model, hogs=None, projections=None, mask_token=None, hog_ratio=None):
    model = continual_mae.configure_model(model)
    params, param_names = continual_mae.collect_params(model, projections, mask_token)
    optimizer = setup_optimizer(params)
    cmae_model = continual_mae.Continual_MAE(model, optimizer,
                           hogs = hogs,
                           projections = projections,
                           mask_token = mask_token,
                           hog_ratio = hog_ratio,
                           block_size=cfg.block_size,
                           mask_ratio=cfg.mask_ratio,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           ap=cfg.OPTIM.AP)
    logger.info(f"model for adaptation: %s", cmae_model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cmae_model


def compute_metrics(model: torch.nn.Module,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    batch_size: int = 100,
                    device: torch.device = None,
                    tag: str = "",
                    no_adapt: bool = False):
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

    for i in range(n_batches):
        lo = i * batch_size
        hi = min((i + 1) * batch_size, total_cnt)
        x_b = x[lo:hi].to(device)
        y_b = y[lo:hi].to(device)
        if not no_adapt:
            output = model(x_b)
            logits = output if isinstance(output, torch.Tensor) else output[0]
        else:
            # bypass adaptation: use underlying ViT backbone only
            core = unwrap_model(model)
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
        correct_batch = (preds == y_b).float().sum().item()
        total_batch = y_b.shape[0]
        correct += correct_batch
        total_eval += total_batch
        batch_nll_sum = F.cross_entropy(logits, y_b, reduction='sum').item()
        nll_sum += batch_nll_sum
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

        if getattr(cfg.TEST, "BATCH_METRICS", False):
            batch_acc = correct_batch / total_batch if total_batch > 0 else 0.0
            batch_err = 1.0 - batch_acc
            batch_nll = batch_nll_sum / total_batch if total_batch > 0 else 0.0
            batch_ece = compute_ece(confs.detach(), (preds == y_b).float().detach())
            prefix = f"{tag} " if tag else ""
            logger.info(
                f"[BATCH_METRICS] {prefix}batch {i}: Error %: {batch_err:.2%}, "
                f"NLL: {batch_nll:.4f}, ECE: {batch_ece:.4f}"
            )

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
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, total_cnt, 0.0, 0.0, mcl_last, erl_last, eml_last

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

    return acc, nll, ece, max_softmax, entropy, cos_sim, total_cnt, 0.0, 0.0, mcl_last, erl_last, eml_last


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
    """Best-effort unwrapping to reach the underlying ViT backbone for feature extraction.

    This function does not change any training/eval modes or optimizer state.
    """
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
    """Extract CLS features and logits from a ViT-like backbone without adaptation.

    Tries to use forward_features + head/forward_head when available.
    """
    core = base_model
    if hasattr(core, "module"):
        core = core.module

    feats = None
    logits = None
    if hasattr(core, "forward_features"):
        out = core.forward_features(x_b)
        feats = out[0] if isinstance(out, tuple) else out
        if hasattr(core, "forward_head"):
            logits = core.forward_head(feats)
        elif hasattr(core, "head") and isinstance(core.head, nn.Module):
            logits = core.head(feats)
        else:
            logits = core(x_b)
    else:
        logits = core(x_b)
        feats = logits

    if isinstance(logits, tuple):
        logits = logits[0]
    return feats, logits


def _extract_blockwise_tokens_and_logits(core: torch.nn.Module, x_b: torch.Tensor):
    if hasattr(core, "module"):
        core = core.module
    if not (hasattr(core, "patch_embed") and hasattr(core, "cls_token") and hasattr(core, "pos_embed") and hasattr(core, "pos_drop") and hasattr(core, "blocks") and hasattr(core, "norm")):
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
    """Save per-domain forward features/logits/probabilities/predictions/labels.

    This runs an extra pure forward pass through the underlying ViT backbone
    and does not modify any adaptation state.
    """
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
        dataset_tag = "cifar10c"
        filename = f"{method_name}_{severity}_{dataset_tag}.npy"
        path = os.path.join(save_dir, filename)
        np.save(path, domains, allow_pickle=True)
        logger.info(f"Saved features to: {path}")
    except Exception as e:
        logger.warning(f"Failed to save severity features for {severity}: {e}")


if __name__ == '__main__':
    evaluate('CIFAR-10-C evaluation.')