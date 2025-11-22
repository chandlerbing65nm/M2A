import logging

import torch
import torch.optim as optim
import torch.nn.functional as F
import time

from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

# import tent
# import norm
# import cotta
# import vida
import torch.nn as nn
# import wandb
from conf import cfg, load_cfg_fom_args
from collections import OrderedDict

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
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions)
    if cfg.TEST.ckpt is not None:
        base_model = torch.nn.DataParallel(base_model) # make parallel
        # checkpoint = torch.load(cfg.TEST.ckpt)
        checkpoint = torch.load("./ckpt/vit_base_16_384.t7")
        checkpoint = rm_substr_from_state_dict(checkpoint['model'], 'module.')
        base_model.load_state_dict(checkpoint['model'], strict=False)
    else:
        base_model = torch.nn.DataParallel(base_model) # make parallel
    base_model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    # if cfg.MODEL.ADAPTATION == "norm":
    #     logger.info("test-time adaptation: NORM")
    #     model = setup_norm(base_model)
    # if cfg.MODEL.ADAPTATION == "tent":
    #     logger.info("test-time adaptation: TENT")
    #     model = setup_tent(base_model)
    # if cfg.MODEL.ADAPTATION == "cotta":
    #     logger.info("test-time adaptation: CoTTA")
    #     model = setup_cotta(base_model)
    # if cfg.MODEL.ADAPTATION == "vida":
    #     logger.info("test-time adaptation: ViDA")
    #     model = setup_vida(args, base_model)
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
            x_test = torch.nn.functional.interpolate(x_test, size=(args.size, args.size), \
                mode='bilinear', align_corners=False)
            acc, nll, ece, max_softmax, entropy, cos_sim, total_cnt, adapt_time_total, adapt_macs_total = compute_metrics(
                model, x_test, y_test, cfg.TEST.BATCH_SIZE, device=device
            )
            err = 1. - acc
            All_error.append(err)
            logger.info(f"Error % [{corruption_type}{severity}]: {err:.2%}")
            logger.info(f"NLL [{corruption_type}{severity}]: {nll:.4f}")
            logger.info(f"ECE [{corruption_type}{severity}]: {ece:.4f}")
            logger.info(f"Max Softmax [{corruption_type}{severity}]: {max_softmax:.4f}")
            logger.info(f"Entropy [{corruption_type}{severity}]: {entropy:.4f}")
            logger.info(f"Cosine(pred_softmax, target_onehot) [{corruption_type}{severity}]: {cos_sim:.4f}")
            logger.info(f"Adaptation Time (lower is better) [{corruption_type}{severity}]: {adapt_time_total:.3f}s")
            logger.info(f"Adaptation MACs (lower is better) [{corruption_type}{severity}]: {fmt_sci(adapt_macs_total)}")


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


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
    logger.info(f"model for adaptation: %s", model)
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
def setup_vida(args, model):
    model = vida.configure_model(model, cfg)
    model_param, vida_param = vida.collect_params(model)
    optimizer = setup_optimizer_vida(model_param, vida_param, cfg.OPTIM.LR, cfg.OPTIM.ViDALR)
    vida_model = vida.ViDA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           unc_thr = args.unc_thr,
                           ema = cfg.OPTIM.MT,
                           ema_vida = cfg.OPTIM.MT_ViDA,
                           )
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return vida_model
def setup_optimizer_vida(params, params_vida, model_lr, vida_lr):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam([{"params": params, "lr": model_lr},
                                  {"params": params_vida, "lr": vida_lr}],
                                 lr=1e-5, betas=(cfg.OPTIM.BETA, 0.999),weight_decay=cfg.OPTIM.WD)

    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD([{"params": params, "lr": model_lr},
                                  {"params": params_vida, "lr": vida_lr}],
                                    momentum=cfg.OPTIM.MOMENTUM,dampening=cfg.OPTIM.DAMPENING,
                                    nesterov=cfg.OPTIM.NESTEROV,
                                 lr=1e-5,weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError
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
                    device: torch.device = None):
    """Compute ACC, NLL, ECE, mean Max-Softmax, mean Entropy and adaptation timing/MACs.
    Returns (acc, nll, ece, max_softmax, entropy, cos_sim, total_cnt, adapt_time_total, adapt_macs_total)
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
    confs_all = []
    correct_all = []
    adapt_time_total = 0.0
    adapt_macs_total = 0
    cos_sum = 0.0

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
            # crude estimate for ViT: O(L^2 * heads) + proj costs; constants omitted
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
        x_b_full = x[lo:hi].to(device)
        y_b_full = y[lo:hi].to(device)
        t0 = time.time()
        output = model(x_b_full)
        adapt_time_total += (time.time() - t0)
        adapt_macs_total += per_img_macs * int(x_b_full.shape[0])

        logits = output if isinstance(output, torch.Tensor) else output[0]
        preds = logits.argmax(dim=1)
        correct += (preds == y_b_full).float().sum().item()
        total_eval += y_b_full.shape[0]
        nll_sum += F.cross_entropy(logits, y_b_full, reduction='sum').item()
        probs = logits.softmax(dim=1)
        confs = probs.max(dim=1).values
        ents = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
        one_hot = F.one_hot(y_b_full.long(), num_classes=logits.shape[-1]).float()
        cos_b = F.cosine_similarity(probs, one_hot, dim=1)
        confs_all.append(confs.detach().cpu())
        correct_all.append((preds == y_b_full).detach().cpu())
        max_softmax_sum += float(confs.sum().item())
        entropy_sum += float(ents.sum().item())
        cos_sum += float(cos_b.sum().item())

    if total_eval == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, total_cnt, adapt_time_total, adapt_macs_total

    acc = correct / total_eval
    nll = nll_sum / total_eval
    max_softmax = max_softmax_sum / total_eval if total_eval > 0 else 0.0
    entropy = entropy_sum / total_eval if total_eval > 0 else 0.0
    cos_sim = cos_sum / total_eval if total_eval > 0 else 0.0
    confs_all = torch.cat(confs_all) if len(confs_all) else torch.empty(0)
    correct_all = torch.cat(correct_all).float() if len(correct_all) else torch.empty(0)
    ece = compute_ece(confs_all, correct_all)
    return acc, nll, ece, max_softmax, entropy, cos_sim, total_cnt, adapt_time_total, adapt_macs_total


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


if __name__ == '__main__':
    evaluate('"CIFAR-10-C evaluation.')
