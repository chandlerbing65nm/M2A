import logging
import time
import os
from contextlib import nullcontext
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

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
    elif cfg.MODEL.ADAPTATION == "M2A":
        logger.info("test-time adaptation: M2A")
        model = setup_m2a(base_model)
    else:
        logger.info("Unknown adaptation; defaulting to source")
        model = setup_source(base_model)

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
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
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
                model, x_test, y_test, cfg.TEST.BATCH_SIZE, device=device
            )
            acc, nll, ece, max_softmax, entropy, cos_sim, total_cnt, adapt_time_total, adapt_macs_total, mcl_last, erl_last, eml_last = metrics
            err = 1. - acc
            all_error.append(err)
            logger.info(f"Error % [{corruption_type}{severity}]: {err:.2%}")
            logger.info(f"NLL [{corruption_type}{severity}]: {nll:.4f}")
            logger.info(f"ECE [{corruption_type}{severity}]: {ece:.4f}")
            logger.info(f"Max Softmax [{corruption_type}{severity}]: {max_softmax:.4f}")
            logger.info(f"Entropy [{corruption_type}{severity}]: {entropy:.4f}")
            logger.info(f"Cosine(pred_softmax, target_onehot) [{corruption_type}{severity}]: {cos_sim:.4f}")
            logger.info(f"MCL (avg per corruption) [{corruption_type}{severity}]: {mcl_last:.6f}")
            logger.info(f"ERL (avg per corruption) [{corruption_type}{severity}]: {erl_last:.6f}")
            logger.info(f"EML (avg per corruption) [{corruption_type}{severity}]: {eml_last:.6f}")
            # New metrics per corruption (averaged per corruption)
            # - Adaptation Time (s): total wall-clock time spent adapting; lower is better
            # - Adaptation MACs: total MACs for adapted samples; lower is better
            logger.info(f"Adaptation Time (lower is better) [{corruption_type}{severity}]: {adapt_time_total:.3f}s")
            logger.info(f"Adaptation MACs (lower is better) [{corruption_type}{severity}]: {fmt_sci(adapt_macs_total)}")

            # Domain Shift Robustness: std of accuracies across types so far (lower is better)
            accs_so_far.append(acc)
            if len(accs_so_far) >= 2:
                import math
                mean_acc = sum(accs_so_far) / float(len(accs_so_far))
                var_acc = sum((a - mean_acc) ** 2 for a in accs_so_far) / float(len(accs_so_far))
                dsr = math.sqrt(var_acc)
            else:
                dsr = 0.0
            logger.info(f"Domain Shift Robustness (std, lower is better) up to [{corruption_type}{severity}]: {dsr:.4f}")

            # Catastrophic Forgetting Rate (measured): re-evaluate previous corruption after current adaptation
            # CFR_current = max(0, prev_acc_at_time - prev_acc_after_current). Lower is better.
            if prev_x is not None and prev_y is not None and prev_acc_at_time is not None:
                try:
                    m2a_model = model.module if hasattr(model, 'module') else model
                    ctx = m2a_model.no_adapt_mode() if hasattr(m2a_model, 'no_adapt_mode') else nullcontext()
                except Exception:
                    ctx = nullcontext()
                with ctx:
                    re_metrics = compute_metrics(
                        model, prev_x, prev_y, cfg.TEST.BATCH_SIZE, device=device
                    )
                    re_acc = re_metrics[0]
                cfr_measured = max(0.0, float(prev_acc_at_time) - float(re_acc))
                logger.info(f"Catastrophic Forgetting Rate (prev-domain, lower is better) after [{corruption_type}{severity}]: {cfr_measured:.4f}")
            # Update previous corruption cache for next iteration
            prev_x, prev_y, prev_acc_at_time = x_test, y_test, acc

    # Save checkpoint after full evaluation if requested
    try:
        if args.save_ckpt:
            method = str(cfg.MODEL.ADAPTATION).lower()
            arch_tag = str(cfg.MODEL.ARCH).replace('/', '').replace('-', '').replace('_', '').lower()
            dataset_tag = 'cifar10c'
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
                    device: torch.device = None):
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
            preds = logits.argmax(dim=1)
            correct += (preds == y_eval).float().sum().item()
            total_eval += y_eval.shape[0]
            nll_sum += F.cross_entropy(logits, y_eval, reduction='sum').item()
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
    try:
        logger.info(f"[setup_m2a] collected params: total={len(param_names)}")
        lrs = [pg.get('lr', None) for pg in optimizer.param_groups]
        logger.info(f"[setup_m2a] optimizer: {optimizer.__class__.__name__}, lrs={lrs}")
    except Exception:
        pass
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
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return m2a_model

if __name__ == '__main__':
    evaluate('CIFAR-10-C evaluation with M2A: Stochastic Patch Erasing with Adaptive Residual Correction for Continual Test-Time Adaptation.')
