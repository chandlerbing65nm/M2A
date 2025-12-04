import logging

import torch
import torch.optim as optim

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
    else:
        raise ValueError(f"Unknown adaptation method: {cfg.MODEL.ADAPTATION}")
    
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
            acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE, device = device)
            err = 1. - acc
            All_error.append(err)
            logger.info(f"Error % [{corruption_type}{severity}]: {err:.2%}")

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
    cotta_model = continual_mae.Continual_MAE(model, optimizer,
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
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model



if __name__ == '__main__':
    evaluate('"CIFAR-10-C evaluation.')