import os
import torch
import logging
import numpy as np
import methods

from models.model import get_model
from utils.misc import print_memory_info
from utils.eval_utils import get_accuracy, eval_domain_dict, AnalysisSampleCollector
from utils.registry import ADAPTATION_REGISTRY
from datasets.data_loading import get_test_loader
from utils.utils_cdc import create_cdc_sequence
from conf import cfg, load_cfg_from_args, get_num_classes, ckpt_path_to_domain_seq

logger = logging.getLogger(__name__)


def _build_ckpt_filename(method: str, arch_name: str) -> str:
    """Build standardized checkpoint filename.

    For general methods (source, rem, tent, etc.):
        (method)_(arch)_(dataset).pth

    For M2A:
        (method)_(arch)_(mask_type)_(mask_method)_(loss_tag)_(dataset).pth

    - method: lowercased cfg.MODEL.ADAPTATION (e.g., m2a, tent, source)
    - arch:  short tag derived from cfg.MODEL.ARCH (e.g., vitb16, vitl16, vittiny16, rn50, convnextb)
    - mask_type: "spatial"/"spectral" for M2A, else "none"
    - mask_method: "patch"/"pixel"/"all"/"low"/"high" for M2A, else "none"
    - loss_tag:  "+"-joined subset of [disable_mcl, disable_erl, disable_eml] for M2A, else "none"
    - dataset: cfg.CORRUPTION.DATASET with underscores removed (e.g., imagenetc, cifar10c, mrsffiac)
    """
    m = (method or "").lower()

    arch_key = str(arch_name)
    arch_tag = arch_key.replace('/', '').replace('-', '').replace('_', '').lower()
    low_arch = arch_key.lower()
    if 'vit' in low_arch:
        if 'b16' in low_arch or 'base' in low_arch:
            arch_tag = 'vitb16'
        elif 'l16' in low_arch or 'large' in low_arch:
            arch_tag = 'vitl16'
        elif 'tiny' in low_arch:
            arch_tag = 'vittiny16'
    elif 'rn50' in low_arch or 'resnet50' in low_arch:
        arch_tag = 'rn50'
    elif 'convnext' in low_arch:
        arch_tag = 'convnextb'

    dataset_tag = str(cfg.CORRUPTION.DATASET).replace('_', '').lower()

    if m == 'm2a':
        mask_type = 'none'
        mask_method = 'none'
        loss_tag = 'none'
        try:
            rm = str(getattr(cfg.M2A, 'RANDOM_MASKING', '') or '').lower()
            if rm in ('spatial', 'spectral'):
                mask_type = rm
                if rm == 'spatial':
                    mm = str(getattr(cfg.M2A, 'SPATIAL_TYPE', '') or '').lower()
                else:
                    mm = str(getattr(cfg.M2A, 'SPECTRAL_TYPE', '') or '').lower()
                if mm:
                    mask_method = mm

            disabled = []
            if bool(getattr(cfg.M2A, 'DISABLE_MCL', False)):
                disabled.append('disable_mcl')
            if bool(getattr(cfg.M2A, 'DISABLE_ERL', False)):
                disabled.append('disable_erl')
            if bool(getattr(cfg.M2A, 'DISABLE_EML', False)):
                disabled.append('disable_eml')
            if disabled:
                loss_tag = '+'.join(disabled)
        except Exception:
            pass

        parts = [m, arch_tag, mask_type, mask_method, loss_tag, dataset_tag]
        return f"{'_'.join(parts)}.pth"

    parts = [m, arch_tag, dataset_tag]
    return f"{'_'.join(parts)}.pth"


def _evaluate_cdc(model,
                  device,
                  domain_sequence,
                  severities,
                  model_preprocess):
    """Evaluate using a continual domain curriculum (CDC) over domains.

    This mirrors the CIFAR CTTA CDC evaluation: batches are drawn from
    different domains according to a curriculum sequence, and we accumulate
    per-domain accuracy to report a mean error per corruption/domain and
    an overall mean across domains.
    """

    domain_names = list(domain_sequence)
    if len(domain_names) == 0:
        logger.warning("CDC evaluation requested but domain_sequence is empty; skipping.")
        return

    # Determine effective number of examples per domain for CDC.
    # When NUM_EX == -1, use full dataset size (10k for CIFAR-C, 5k for ImageNet-C).
    dataset_name = cfg.CORRUPTION.DATASET
    if cfg.CORRUPTION.NUM_EX <= 0:
        if dataset_name in ["cifar10_c", "cifar100_c"]:
            effective_num_examples = 10000
        elif dataset_name == "imagenet_c":
            effective_num_examples = 5000
        else:
            effective_num_examples = 1
    else:
        effective_num_examples = cfg.CORRUPTION.NUM_EX

    num_domains = len(domain_names)
    # Approximate number of batches per domain for CDC
    num_total_batches = int(np.ceil(effective_num_examples / max(1, cfg.TEST.BATCH_SIZE)))
    domain_order = create_cdc_sequence(num_domains=num_domains,
                                       num_total_batches=num_total_batches)

    # Accumulate correct/total per domain across all severities
    corruption_correct = {d: 0.0 for d in domain_names}
    corruption_total = {d: 0 for d in domain_names}

    for severity in severities:
        # Build a loader iterator per domain for this severity
        loaders = {}
        for dom_name in domain_names:
            test_loader = get_test_loader(
                setting=cfg.SETTING,
                adaptation=cfg.MODEL.ADAPTATION,
                dataset_name=cfg.CORRUPTION.DATASET,
                preprocess=model_preprocess,
                data_root_dir=cfg.DATA_DIR,
                domain_name=dom_name,
                domain_names_all=domain_names,
                severity=severity,
                num_examples=effective_num_examples,
                rng_seed=cfg.RNG_SEED,
                use_clip=cfg.MODEL.USE_CLIP,
                n_views=cfg.TEST.N_AUGMENTATIONS,
                delta_dirichlet=cfg.TEST.DELTA_DIRICHLET,
                batch_size=cfg.TEST.BATCH_SIZE,
                shuffle=False,
                workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
            )
            loaders[dom_name] = iter(test_loader)

        total_seen = 0
        for step_idx, dom_idx in enumerate(domain_order):
            dom_name = domain_names[dom_idx]
            loader_it = loaders.get(dom_name, None)
            if loader_it is None:
                continue
            try:
                batch = next(loader_it)
            except StopIteration:
                loaders[dom_name] = None
                continue

            imgs, labels = batch[0], batch[1]
            labels = labels.to(device)
            if isinstance(imgs, list):
                imgs_dev = [img.to(device) for img in imgs]
                output = model(imgs_dev)
            else:
                imgs_dev = imgs.to(device)
                output = model(imgs_dev)

            if isinstance(output, (tuple, list)):
                output = output[0]
            preds = output.argmax(1)

            correct = (preds == labels).float().sum().item()
            batch_size_curr = labels.shape[0]
            corruption_correct[dom_name] += correct
            corruption_total[dom_name] += batch_size_curr
            total_seen += batch_size_curr

            acc_curr = correct / batch_size_curr if batch_size_curr > 0 else 0.0
            err_curr = 1.0 - acc_curr
            running_correct = sum(corruption_correct.values())
            running_acc = running_correct / max(1, total_seen)
            err_running = 1.0 - running_acc

            logger.info(
                f"[CDC {step_idx + 1}/{len(domain_order)}: {dom_name}{severity}] "
                f"current error: {err_curr:.2%}, running error: {err_running:.2%}"
            )

    # Per-domain mean error across all severities
    per_domain_errors = []
    for dom_name in domain_names:
        total_dom = corruption_total.get(dom_name, 0)
        if total_dom == 0:
            continue
        acc_dom = corruption_correct[dom_name] / total_dom
        err_dom = 1.0 - acc_dom
        per_domain_errors.append(err_dom)
        logger.info(
            f"[CDC] {cfg.CORRUPTION.DATASET} mean error for {dom_name}: {err_dom:.2%}"
        )

    if per_domain_errors:
        logger.info(
            f"[CDC] mean error across all domains: {np.mean(per_domain_errors):.2%}"
        )


def evaluate(description):
    load_cfg_from_args(description)
    valid_settings = ["reset_each_shift",           # reset the model state after the adaptation to a domain
                      "continual",                  # train on sequence of domain shifts without knowing when a shift occurs
                      "gradual",                    # sequence of gradually increasing / decreasing domain shifts
                      "mixed_domains",              # consecutive test samples are likely to originate from different domains
                      "correlated",                 # sorted by class label
                      "mixed_domains_correlated",   # mixed domains + sorted by class label
                      "gradual_correlated",         # gradual domain shifts + sorted by class label
                      "reset_each_shift_correlated"
                      ]
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    # get the base model and its corresponding input pre-processing (if available)
    base_model, model_preprocess = get_model(cfg, num_classes, device)

    # append the input pre-processing to the base model
    base_model.model_preprocess = model_preprocess

    # setup test-time adaptation method
    available_adaptations = ADAPTATION_REGISTRY.registered_names()
    assert cfg.MODEL.ADAPTATION in available_adaptations, \
        f"The adaptation '{cfg.MODEL.ADAPTATION}' is not supported! Choose from: {available_adaptations}"
    model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, num_classes=num_classes)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION}")

    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET == "domainnet126":
        # extract the domain sequence for a specific checkpoint.
        domain_sequence = ckpt_path_to_domain_seq(ckpt_path=cfg.MODEL.CKPT_PATH)
    elif cfg.CORRUPTION.DATASET in ["imagenet_d", "imagenet_d109"] and not cfg.CORRUPTION.TYPE[0]:
        # domain_sequence = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        domain_sequence = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        domain_sequence = cfg.CORRUPTION.TYPE
    logger.info(f"Using {cfg.CORRUPTION.DATASET} with the following domain sequence: {domain_sequence}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    domain_seq_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else domain_sequence

    # setup the severities for the gradual setting
    if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in ["cifar10_c", "cifar100_c", "imagenet_c"] and len(cfg.CORRUPTION.SEVERITY) == 1:
        severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        logger.info(f"Using the following severity sequence for each domain: {severities}")
    else:
        severities = cfg.CORRUPTION.SEVERITY

    # CDC mode: override the standard evaluation loop
    if bool(getattr(cfg.TEST, "ENABLE_CDC", False)):
        _evaluate_cdc(model, device, domain_sequence, severities, model_preprocess)
        return

    errs = []
    errs_5 = []
    domain_dict = {}

    # Domain generalization mode: adapt on first 10 domains, then evaluate
    # without further adaptation on the remaining domains. Only meaningful for
    # the standard continual setting (non-mixed_domains).
    domain_gen = bool(getattr(cfg.TEST, "DOMAIN_GEN", False)) \
        and cfg.SETTING == "continual" \
        and "mixed_domains" not in cfg.SETTING
    if domain_gen:
        logger.info("Domain generalization enabled: adapting on first 10 corruptions, then evaluating without adaptation on remaining ones.")

    # Optional analysis sample collector for saving selected images
    analysis_collector = None
    if getattr(cfg.TEST, "SAVE_ANALYSIS_IMAGES", False):
        # Only M2A currently exposes masked images via _last_masked
        save_masked = (cfg.MODEL.ADAPTATION == "m2a")
        analysis_collector = AnalysisSampleCollector(
            root_dir="/flash/project_465002264/projects/tca/classification/plots/masks",
            dataset_name=cfg.CORRUPTION.DATASET,
            arch_name=cfg.MODEL.ARCH,
            save_masked=save_masked,
            max_per_type=5,
        )

    # Number of full passes over all domains/severities
    n_lifelong = max(1, int(getattr(cfg.TEST, "LIFELONG", 1)))
    n_shuffle = max(0, int(getattr(cfg.TEST, "SHUFFLE_DOMAIN", 0)))

    # If shuffle-domain is enabled (>0), it overrides LIFELONG for the number of passes
    n_passes = n_shuffle if n_shuffle > 0 else n_lifelong

    # start evaluation (potentially repeated multiple times)
    for r in range(n_passes):
        # track errors for this lifelong pass only
        errs_pass = []
        errs_5_pass = []

        # optionally shuffle domain order for this pass
        if n_shuffle > 0:
            order = np.random.permutation(len(domain_seq_loop))
            domain_seq_this_pass = [domain_seq_loop[i] for i in order]
        else:
            domain_seq_this_pass = domain_seq_loop

        for i_dom, domain_name in enumerate(domain_seq_this_pass):
            # Reset adaptation:
            # - for every domain if using reset_each_shift setting
            # - at the first domain of each pass when shuffle_domain is enabled
            # - only once at the very beginning when neither applies (legacy behavior)
            if "reset_each_shift" in cfg.SETTING:
                do_reset = True
            elif n_shuffle > 0:
                do_reset = (i_dom == 0)
            else:
                do_reset = (r == 0 and i_dom == 0)

            if do_reset:
                try:
                    model.reset()
                    logger.info("resetting model")
                except AttributeError:
                    logger.warning("not resetting model")
            else:
                logger.warning("not resetting model")

            for severity in severities:
                test_data_loader = get_test_loader(
                    setting=cfg.SETTING,
                    adaptation=cfg.MODEL.ADAPTATION,
                    dataset_name=cfg.CORRUPTION.DATASET,
                    preprocess=model_preprocess,
                    data_root_dir=cfg.DATA_DIR,
                    domain_name=domain_name,
                    domain_names_all=domain_sequence,
                    severity=severity,
                    num_examples=cfg.CORRUPTION.NUM_EX,
                    rng_seed=cfg.RNG_SEED,
                    use_clip=cfg.MODEL.USE_CLIP,
                    n_views=cfg.TEST.N_AUGMENTATIONS,
                    delta_dirichlet=cfg.TEST.DELTA_DIRICHLET,
                    batch_size=cfg.TEST.BATCH_SIZE,
                    shuffle=False,
                    workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count())
                )

                if r == 0 and i_dom == 0:
                    # Note that the input normalization is done inside of the model
                    logger.info(f"Using the following data transformation:\n{test_data_loader.dataset.transform}")

                # evaluate the model
                acc, domain_dict, num_samples = get_accuracy(
                    model,
                    data_loader=test_data_loader,
                    dataset_name=cfg.CORRUPTION.DATASET,
                    domain_name=domain_name,
                    setting=cfg.SETTING,
                    domain_dict=domain_dict,
                    print_every=cfg.PRINT_EVERY,
                    device=device,
                    batch_random=cfg.TEST.BATCH_RANDOM,
                    no_adapt=(domain_gen and i_dom >= 10),
                    sample_collector=analysis_collector,
                    severity=severity,
                )

                err = 1. - acc
                errs.append(err)
                errs_pass.append(err)
                if severity == 5 and domain_name != "none":
                    errs_5.append(err)
                    errs_5_pass.append(err)

                logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={num_samples}]: {err:.2%}")

        # summary for this pass
        if len(errs_5_pass) > 0:
            logger.info(
                f"[pass {r + 1}/{n_passes}] mean error: {np.mean(errs_pass):.2%}, "
                f"mean error at 5: {np.mean(errs_5_pass):.2%}"
            )
        else:
            logger.info(
                f"[pass {r + 1}/{n_passes}] mean error: {np.mean(errs_pass):.2%}"
            )

    # overall summary across all passes
    if len(errs_5) > 0:
        logger.info(f"[total over {n_passes} passes] mean error: {np.mean(errs):.2%}, mean error at 5: {np.mean(errs_5):.2%}")
    else:
        logger.info(f"[total over {n_passes} passes] mean error: {np.mean(errs):.2%}")

    if "mixed_domains" in cfg.SETTING and len(domain_dict.values()) > 0:
        # print detailed results for each domain
        eval_domain_dict(domain_dict, domain_seq=domain_sequence)

    if analysis_collector is not None:
        analysis_collector.save_all()

    if cfg.TEST.DEBUG:
        print_memory_info()

    # Optionally save adapted model checkpoint at the end of evaluation
    if bool(getattr(cfg.TEST, "SAVE_CKPT", False)):
        try:
            method = str(cfg.MODEL.ADAPTATION).lower()
            arch_name = str(cfg.MODEL.ARCH)
            ckpt_dir = "/flash/project_465002264/projects/tca/classification/ckpt"
            os.makedirs(ckpt_dir, exist_ok=True)
            filename = _build_ckpt_filename(method, arch_name)
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


if __name__ == '__main__':
    evaluate('"Evaluation.')
