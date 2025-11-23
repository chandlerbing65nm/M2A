import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from robustbench.data import load_avffiac
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import m2a
from conf import cfg, load_cfg_fom_args
import operators

import numpy as np
import random
from tqdm import tqdm
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions)

    checkpoint = torch.load("/users/doloriel/work/Repo/M2A/ckpt/uffia_vitb16_best.pth", map_location='cpu')
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
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "M2A":
        logger.info("test-time adaptation: M2A")
        model = setup_m2a(base_model)
    # evaluate on each severity and type of corruption in turn
    for ii, severity in enumerate(cfg.CORRUPTION.SEVERITY):
        for i_x, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # reset adaptation for each combination of corruption x severity
            # note: for evaluation protocol, but not necessarily needed
            try:
                if i_x == 0:
                    model.reset()
                    logger.info("resetting model")
                else:
                    logger.warning(" ")
                    logger.warning("not resetting model")
            except:
                logger.warning(" ")
                logger.warning("not resetting model")
            x_test, y_test = load_avffiac(cfg.CORRUPTION.NUM_EX,
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


def setup_optimizer(params):
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
    params = m2a.collect_params(model)
    optimizer = setup_optimizer(params)
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
    # logger.info(f"model for adaptation: %s", model)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return m2a_model


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

            max_softmax_sum += float(confs.sum().item())
            entropy_sum += float(ents.sum().item())

    if total_eval == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    acc = correct / total_eval
    nll = nll_sum / total_eval
    max_softmax = max_softmax_sum / total_eval if total_eval > 0 else 0.0
    entropy = entropy_sum / total_eval if total_eval > 0 else 0.0
    # ECE omitted here to keep parity with imagenetc_m2a baseline
    ece = 0.0
    return acc, nll, ece, max_softmax, entropy


if __name__ == '__main__':
    evaluate('AVFFIA-C M2A evaluation.')
