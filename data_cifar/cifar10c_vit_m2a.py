import logging
import time
import os
from contextlib import nullcontext
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

from conf import cfg, load_cfg_fom_args
import m2a

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

    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE (source)")
        model = setup_source(base_model)
        method_name = "source"
    elif cfg.MODEL.ADAPTATION == "M2A":
        logger.info("test-time adaptation: M2A")
        model = setup_m2a(base_model)
        method_name = _build_m2a_method_name()
    else:
        logger.info("Unknown adaptation; defaulting to source")
        model = setup_source(base_model)
        method_name = "source"
    if getattr(cfg, "PRINT_MODEL", False):
        return

    # evaluate on each severity and type of corruption in turn
    all_error = []
    accs_so_far = []  # for domain shift robustness
    prev_x = None
    prev_y = None
    prev_acc_at_time = None

    # Helper: format large numbers in base-10 scientific notation
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
    for severity in cfg.CORRUPTION.SEVERITY:
        severity_domains = {}
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            domain_gen = bool(getattr(cfg.TEST, "DOMAIN_GEN", False))
            no_adapt_this = domain_gen and (i_c >= 10)
            if not no_adapt_this:
                if i_c == 0:
                    try:
                        if hasattr(model, 'reset'):
                            model.reset()
                            logger.info("")
                            logger.info("resetting model")
                    except Exception:
                        logger.info("")
                        logger.warning("not resetting model")
                else:
                    logger.info("")
                    logger.warning("not resetting model")
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test = F.interpolate(x_test, size=(args.size, args.size),
                                   mode='bilinear', align_corners=False)
            # No divisibility requim2aent for spatial masking (pixel-level squares)
            # Compute metrics (acc, nll, ece, max-softmax, entropy)
            # No directional metrics reset; focusing on standard/confidence metrics

            # Reset per-corruption M2A loss statistics if available
            if hasattr(model, 'reset_loss_stats'):
                try:
                    model.reset_loss_stats()
                except Exception:
                    pass
            metrics = compute_metrics(
                model, x_test, y_test, cfg.TEST.BATCH_SIZE, device=device,
                tag=f"[{corruption_type}{severity}]", no_adapt=bool(no_adapt_this)
            )
            acc, nll, ece, max_softmax, entropy, cos_sim, total_cnt, adapt_time_total, adapt_macs_total, mcl_last, erl_last, eml_last = metrics
            err = 1. - acc
            all_error.append(err)
            logger.info(f"Error % [{corruption_type}{severity}]: {err:.2%}")
            logger.info(f"NLL [{corruption_type}{severity}]: {nll:.4f}")
            logger.info(f"ECE [{corruption_type}{severity}]: {ece:.4f}")
            # logger.info(f"Max Softmax [{corruption_type}{severity}]: {max_softmax:.4f}")
            # logger.info(f"Entropy [{corruption_type}{severity}]: {entropy:.4f}")
            # logger.info(f"Cosine(pred_softmax, target_onehot) [{corruption_type}{severity}]: {cos_sim:.4f}")
            logger.info(f"MCL (avg per corruption) [{corruption_type}{severity}]: {mcl_last:.6f}")
            logger.info(f"ERL (avg per corruption) [{corruption_type}{severity}]: {erl_last:.6f}")
            logger.info(f"EML (avg per corruption) [{corruption_type}{severity}]: {eml_last:.6f}")
            # New metrics per corruption (averaged per corruption)
            # - Adaptation Time (s): total wall-clock time spent adapting; lower is better
            # - Adaptation MACs: total MACs for adapted samples; lower is better
            # logger.info(f"Adaptation Time (lower is better) [{corruption_type}{severity}]: {adapt_time_total:.3f}s")
            # logger.info(f"Adaptation MACs (lower is better) [{corruption_type}{severity}]: {fmt_sci(adapt_macs_total)}")

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

            # Domain Shift Robustness: std of accuracies across types so far (lower is better)
            accs_so_far.append(acc)
            if len(accs_so_far) >= 2:
                import math
                mean_acc = sum(accs_so_far) / float(len(accs_so_far))
                var_acc = sum((a - mean_acc) ** 2 for a in accs_so_far) / float(len(accs_so_far))
                dsr = math.sqrt(var_acc)
            else:
                dsr = 0.0
            # logger.info(f"Domain Shift Robustness (std, lower is better) up to [{corruption_type}{severity}]: {dsr:.4f}")
            # Update previous corruption cache for next iteration
            prev_x, prev_y, prev_acc_at_time = x_test, y_test, acc

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
            mask_tag = f"_{str(args.random_masking).lower()}" if (method == 'm2a' and getattr(args, 'random_masking', None)) else ""
            disable_tag = ""
            if method == 'm2a':
                try:
                    if getattr(cfg.M2A, 'DISABLE_MCL', False):
                        disable_tag += '_disable_mcl'
                    if getattr(cfg.M2A, 'DISABLE_EML', False):
                        disable_tag += '_disable_eml'
                except Exception:
                    pass
            filename = f"{method}_{arch_tag}{mask_tag}{disable_tag}_{dataset_tag}.pth"
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

    # No MotivationExp1 summary; only standard/confidence metrics are reported


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def compute_metrics(model: nn.Module,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    batch_size: int = 100,
                    device: torch.device = None,
                    tag: str = "",
                    no_adapt: bool = False):
    """Compute ACC, NLL, ECE, mean Max-Softmax, mean Entropy, mean cosine(pred_softmax, target_onehot)
    and accumulate adaptation timing/MACs during M2A CTTA.
    Returns (acc, nll, ece, max_softmax, entropy, cos_sim, total_cnt, adapt_time_total, adapt_macs_total, mcl_last, erl_last, eml_last)
    """
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
    # For ECE, accumulate confidences and correctness
    confs_all = []
    correct_all = []
    total_cnt = x.shape[0]
    cos_sum = 0.0

    # Adaptation timing and MACs accumulators (averaged per corruption outside)
    adapt_time_total = 0.0
    adapt_macs_total = 0

    # Estimate per-image MACs for ViT-like models
    def estimate_vit_macs_per_image(stats_src, img_size: int) -> int:
        try:
            m = stats_src
            # Unwrap M2A and DataParallel to reach underlying ViT
            if hasattr(m, 'module'):
                m = m.module
            if hasattr(m, 'model'):
                m = m.model
            if hasattr(m, 'module'):
                m = m.module
            # Extract key attributes
            # Patch size from conv proj
            pe = m.patch_embed
            if hasattr(pe, 'inner'):
                pe_inner = pe.inner
            else:
                pe_inner = pe
            ps = pe_inner.proj.weight.shape[2]
            D = getattr(m, 'embed_dim', m.head.in_features)
            depth = len(m.blocks)
            # heads from first block
            h = m.blocks[0].attn.num_heads
            # mlp ratio from first block
            D_m = m.blocks[0].mlp.fc1.out_features
            mlp_ratio = float(D_m) / float(D)
            # tokens length
            Ph = img_size // ps
            Pw = img_size // ps
            L = 1 + (Ph * Pw)
            C_in = pe_inner.proj.weight.shape[1]
            # MACs: Patch embedding conv
            macs_patch = (Ph * Pw) * (C_in * D * ps * ps)
            # Per-block MACs
            # QKV: 3 * L * D * D
            # Attn: 2 * L * L * D (QK^T and AV)
            # Out proj: L * D * D
            # MLP: 2 * L * D * (mlp_ratio * D)
            per_block = (3 * L * D * D) + (2 * L * L * D) + (L * D * D) + (2 * L * D * int(mlp_ratio * D))
            macs_blocks = depth * per_block
            # Head
            num_classes = m.head.out_features if hasattr(m.head, 'out_features') else 1000
            macs_head = D * num_classes
            total = macs_patch + macs_blocks + macs_head
            return int(total)
        except Exception:
            return 0

    with torch.no_grad():
        for b in range(n_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, total_N)
            x_b_full = x[start:end]
            y_b_full = y[start:end]

            # Move full batch to device for potential prediction
            x_b_full = x_b_full.to(device)
            y_b_full = y_b_full.to(device)

            # Single prediction pass (may adapt internally)
            if not no_adapt:
                t0 = time.time()
                output = model(x_b_full)
                adapt_time_total += (time.time() - t0)
                stats_src = model.module if hasattr(model, 'module') else model
                # Count MACs for entire batch
                per_img_macs = estimate_vit_macs_per_image(stats_src, img_size=x_b_full.shape[-1])
                adapt_macs_total += per_img_macs * int(x_b_full.shape[0])
                y_eval = y_b_full
                # Handle outputs (logits)
                logits = output
            else:
                # bypass adaptation: use underlying base model forward only
                base_core = _unwrap_base_model_for_features(model)
                _feats, logits = _extract_features_and_logits(base_core, x_b_full)
                y_eval = y_b_full
            preds = logits.argmax(dim=1)
            correct_batch = (preds == y_eval).float().sum().item()
            total_batch = y_eval.shape[0]
            correct += correct_batch
            total_eval += total_batch
            batch_nll_sum = F.cross_entropy(logits, y_eval, reduction='sum').item()
            nll_sum += batch_nll_sum
            probs = logits.softmax(dim=1)
            confs = probs.max(dim=1).values
            # Entropy per sample: -sum p * log p
            ents = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
            # Cosine similarity between pred softmax and one-hot target
            one_hot = F.one_hot(y_eval.long(), num_classes=logits.shape[-1]).float()
            cos_b = F.cosine_similarity(probs, one_hot, dim=1)
            confs_all.append(confs.detach().cpu())
            correct_all.append((preds == y_eval).detach().cpu())
            max_softmax_sum += float(confs.sum().item())
            entropy_sum += float(ents.sum().item())
            cos_sum += float(cos_b.sum().item())

            if getattr(cfg.TEST, "BATCH_METRICS", False):
                batch_acc = correct_batch / total_batch if total_batch > 0 else 0.0
                batch_err = 1.0 - batch_acc
                batch_nll = batch_nll_sum / total_batch if total_batch > 0 else 0.0
                batch_ece = compute_ece(confs.detach(), (preds == y_eval).float().detach())
                prefix = f"{tag} " if tag else ""
                logger.info(
                    f"[BATCH_METRICS] {prefix}batch {b}: Error %: {batch_err:.2%}, "
                    f"NLL: {batch_nll:.4f}, ECE: {batch_ece:.4f}"
                )

    if total_eval == 0:
        # Best effort to read last losses
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
    # compute ECE
    confs_all = torch.cat(confs_all) if len(confs_all) else torch.empty(0)
    correct_all = torch.cat(correct_all).float() if len(correct_all) else torch.empty(0)
    ece = compute_ece(confs_all, correct_all)
    # Prefer per-corruption averages if M2A exposes accumulators
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
    """Expected Calibration Error with equal-width bins on [0,1].
    confs: [N], correct: [N] in {0,1}
    """
    if confs.numel() == 0:
        return 0.0
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1)
    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        # include right edge in last bin
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


def _build_m2a_method_name() -> str:
    """Build a descriptive method name for M2A based on masking config."""
    base = "m2a"
    try:
        domain = str(getattr(cfg.M2A, "RANDOM_MASKING", "")).lower()
    except Exception:
        domain = ""
    try:
        mask_type = str(getattr(cfg.M2A, "MASK_TYPE", "")).lower()
    except Exception:
        mask_type = ""
    parts = [base]
    if domain:
        parts.append(domain)
    if mask_type:
        parts.append(mask_type)
    return "_".join(parts)


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


def setup_optimizer(params):
    """Set up optimizer for test-time adaptation.

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


def setup_m2a(model):
    model = m2a.configure_model(model)
    params, param_names = m2a.collect_params(model, ln_quarter=cfg.MODEL.LN_QUARTER)
    if cfg.OPTIM.METHOD == 'Adam':
        optimizer = optim.Adam(params,
                               lr=cfg.OPTIM.LR,
                               betas=(cfg.OPTIM.BETA, 0.999),
                               weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        optimizer = optim.SGD(params,
                              lr=cfg.OPTIM.LR,
                              momentum=cfg.OPTIM.MOMENTUM,
                              dampening=cfg.OPTIM.DAMPENING,
                              weight_decay=cfg.OPTIM.WD,
                              nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError
    # Debug logging
    # try:
    #     logger.info(f"model for adaptation: %s", model)
    #     logger.info(f"params for adaptation: %s", param_names)
    #     logger.info(f"optimizer for adaptation: %s", optimizer)
    # except Exception:
    #     pass
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
        plot_loss=cfg.M2A.PLOT_LOSS,
        plot_loss_path=cfg.M2A.PLOT_LOSS_PATH,
        plot_ema_alpha=cfg.M2A.PLOT_EMA_ALPHA,
        mcl_temperature=cfg.M2A.MCL_TEMPERATURE,
        mcl_temperature_apply=cfg.M2A.MCL_TEMPERATURE_APPLY,
        mcl_distance=cfg.M2A.MCL_DISTANCE,
        erl_activation=cfg.M2A.ERL_ACTIVATION,
        erl_leaky_relu_slope=cfg.M2A.ERL_LEAKY_RELU_SLOPE,
        erl_softplus_beta=cfg.M2A.ERL_SOFTPLUS_BETA,
        disable_mcl=cfg.M2A.DISABLE_MCL,
        disable_erl=cfg.M2A.DISABLE_ERL,
        disable_eml=cfg.M2A.DISABLE_EML,
        logm2a_enable=cfg.M2A.LOGM2A_ENABLE,
        logm2a_lr_mult=cfg.M2A.LOGM2A_LR_MULT,
        logm2a_reg=cfg.M2A.LOGM2A_REG,
        logm2a_temp=cfg.M2A.LOGM2A_TEMP,
        
    )
    logger.info(f"model for adaptation: %s", m2a_model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return m2a_model

if __name__ == '__main__':
    evaluate('CIFAR-10-C evaluation with M2A: Stochastic Patch Erasing with Adaptive Residual Correction for Continual Test-Time Adaptation.')
