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

import rem
import torch.nn as nn
from conf import cfg, load_cfg_fom_args
import operators
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

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
    checkpoint = torch.load("/users/doloriel/work/Repo/M2A/ckpt/vit_base_384_cifar10.t7", map_location='cpu')
    checkpoint = rm_substr_from_state_dict(checkpoint['model'], 'module.')
    base_model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    if cfg.TEST.ckpt is not None:
        # make parallel only if CUDA is available
        if device.type == 'cuda':
            base_model = torch.nn.DataParallel(base_model)
        checkpoint = torch.load(cfg.TEST.ckpt, map_location='cpu')
        base_model.load_state_dict(checkpoint['model'], strict=False)
    else:
        if device.type == 'cuda':
            base_model = torch.nn.DataParallel(base_model)
    base_model.to(device)

    head_dim = 768

    method_name = None
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
        method_name = "source"
    if cfg.MODEL.ADAPTATION == "REM":
        logger.info("test-time adaptation: REM")
        model = setup_rem(base_model)
        method_name = "rem"
    if method_name is None:
        raise ValueError(f"Unknown adaptation method: {cfg.MODEL.ADAPTATION}")
    if getattr(cfg, "PRINT_MODEL", False):
        return
    # evaluate on each severity and type of corruption in turn
    All_error = []
    for severity in cfg.CORRUPTION.SEVERITY:
        severity_domains = {}
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
            x_test = torch.nn.functional.interpolate(x_test, size=(args.size, args.size), \
                mode='bilinear', align_corners=False)
            # Reset per-corruption REM loss statistics if available
            if hasattr(model, 'reset_loss_stats'):
                try:
                    model.reset_loss_stats()
                except Exception:
                    pass
            acc, nll, ece, max_softmax, entropy, cos_sim, total_cnt, adapt_time_total, adapt_macs_total, mcl_last, erl_last, eml_last = compute_metrics(
                model, x_test, y_test, cfg.TEST.BATCH_SIZE, device=device,
                tag=f"[{corruption_type}{severity}]"
            )
            err = 1. - acc
            All_error.append(err)
            logger.info(f"Error % [{corruption_type}{severity}]: {err:.2%}")
            logger.info(f"NLL [{corruption_type}{severity}]: {nll:.4f}")
            logger.info(f"ECE [{corruption_type}{severity}]: {ece:.4f}")
            # logger.info(f"Max Softmax [{corruption_type}{severity}]: {max_softmax:.4f}")
            # logger.info(f"Entropy [{corruption_type}{severity}]: {entropy:.4f}")
            # logger.info(f"Cosine(pred_softmax, target_onehot) [{corruption_type}{severity}]: {cos_sim:.4f}")
            # logger.info(f"Adaptation Time (lower is better) [{corruption_type}{severity}]: {adapt_time_total:.3f}s")
            # logger.info(f"Adaptation MACs (lower is better) [{corruption_type}{severity}]: {fmt_sci(adapt_macs_total)}")
            logger.info(f"MCL (avg per corruption) [{corruption_type}{severity}]: {mcl_last:.6f}")
            logger.info(f"ERL (avg per corruption) [{corruption_type}{severity}]: {erl_last:.6f}")
            logger.info(f"EML (avg per corruption) [{corruption_type}{severity}]: {eml_last:.6f}")

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


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


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


def setup_rem(model):
    model = rem.configure_model(model)
    params, param_names = rem.collect_params(model)
    optimizer = setup_optimizer(params)
    rem_model = rem.REM(model, optimizer,
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

def fmt_sci(n: int) -> str:
    try:
        if n == 0:
            return "0"
        import math
        exp = int(math.floor(math.log10(abs(float(n)))))
        mant = float(n) / (10 ** exp)
        return f"{mant:.3f} x 10^{exp}"
    except Exception:
        return str(n)


def compute_metrics(model: torch.nn.Module,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    batch_size: int = 100,
                    device: torch.device = None,
                    tag: str = ""):
    """Compute ACC, NLL, ECE, mean Max-Softmax, mean Entropy, mean cosine(pred_softmax, target_onehot)
    and adaptation timing/MACs.
    Returns (acc, nll, ece, max_softmax, entropy, cos_sim, total_cnt, adapt_time_total, adapt_macs_total, mcl_last, erl_last, eml_last)
    """
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
            # crude estimate for ViT compute
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
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, total_cnt, adapt_time_total, adapt_macs_total, mcl_last, erl_last, eml_last

    acc = correct / total_eval
    nll = nll_sum / total_eval
    max_softmax = max_softmax_sum / total_eval if total_eval > 0 else 0.0
    entropy = entropy_sum / total_eval if total_eval > 0 else 0.0
    cos_sim = cos_sum / total_eval if total_eval > 0 else 0.0
    confs_all = torch.cat(confs_all) if len(confs_all) else torch.empty(0)
    correct_all = torch.cat(correct_all).float() if len(correct_all) else torch.empty(0)
    ece = compute_ece(confs_all, correct_all)
    # Prefer per-corruption averages if REM exposes accumulators
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
        mask = (confs >= lo) & (confs < hi)
        if mask.any():
            acc_bin = correct[mask].float().mean().item()
            conf_bin = confs[mask].float().mean().item()
            ece += (mask.float().mean().item()) * abs(acc_bin - conf_bin)
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
        filename = f"{method_name}_{severity}.npy"
        path = os.path.join(save_dir, filename)
        np.save(path, domains, allow_pickle=True)
        logger.info(f"Saved features to: {path}")
    except Exception as e:
        logger.warning(f"Failed to save severity features for {severity}: {e}")


if __name__ == '__main__':
    evaluate('CIFAR-10-C evaluation.')
