import os
import torch
import logging
import numpy as np
import random
from collections import Counter
from typing import Union
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from datasets.imagenet_subsets import IMAGENET_D_MAPPING


logger = logging.getLogger(__name__)


def _unwrap_base_model(model: torch.nn.Module, max_depth: int = 4) -> torch.nn.Module:
    """Recursively unwrap common container attributes to recover the core model.

    This mirrors the helpers used in other runners (e.g., Imagenet-C M2A) so that
    we can call the backbone directly when bypassing adaptation.
    """
    m = model
    for _ in range(max_depth):
        if hasattr(m, "model"):
            m = getattr(m, "model")
            continue
        if hasattr(m, "module"):
            m = getattr(m, "module")
            continue
        break
    return m


def split_results_by_domain(domain_dict: dict, data: list, predictions: torch.tensor):
    """
    Separates the label prediction pairs by domain
    Input:
        domain_dict: Dictionary, where the keys are the domain names and the values are lists with pairs [[label1, prediction1], ...]
        data: List containing [images, labels, domains, ...]
        predictions: Tensor containing the predictions of the model
    Returns:
        domain_dict: Updated dictionary containing the domain seperated label prediction pairs
    """

    labels, domains = data[1], data[2]
    assert predictions.shape[0] == labels.shape[0], "The batch size of predictions and labels does not match!"

    for i in range(labels.shape[0]):
        if domains[i] in domain_dict.keys():
            domain_dict[domains[i]].append([labels[i].item(), predictions[i].item()])
        else:
            domain_dict[domains[i]] = [[labels[i].item(), predictions[i].item()]]

    return domain_dict


def eval_domain_dict(domain_dict: dict, domain_seq: list):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    Input:
        domain_dict: Dictionary containing the labels and predictions for each domain
        domain_seq: Order to print the results (if all domains are contained in the domain dict)
    """
    correct = []
    num_samples = []
    avg_error_domains = []
    domain_names = domain_seq if all([dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
    logger.info(f"Splitting the results by domain...")
    for key in domain_names:
        label_prediction_arr = np.array(domain_dict[key])  # rows: samples, cols: (label, prediction)
        correct.append((label_prediction_arr[:, 0] == label_prediction_arr[:, 1]).sum())
        num_samples.append(label_prediction_arr.shape[0])
        accuracy = correct[-1] / num_samples[-1]
        error = 1 - accuracy
        avg_error_domains.append(error)
        logger.info(f"{key:<20} error: {error:.2%}")
    logger.info(f"Average error across all domains: {sum(avg_error_domains) / len(avg_error_domains):.2%}")
    # The error across all samples differs if each domain contains different amounts of samples
    logger.info(f"Error over all samples: {1 - sum(correct) / sum(num_samples):.2%}")


class AnalysisSampleCollector:
    def __init__(self, root_dir: str, dataset_name: str, arch_name: str, save_masked: bool = False, max_per_type: int = 5):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.arch_name = arch_name
        self.save_masked = save_masked
        self.max_per_type = max_per_type
        # Per (domain, severity), store at most max_per_type samples keyed by
        # global dataset index so that we can retrieve the same images across
        # all domains without keeping the full dataset in memory.
        self.samples = {}
        # For each severity, remember which domain was seen first (reference
        # domain) and the corresponding least-confident samples.
        self.reference_domain = {}
        self.ref_candidates = {}

    def update(self, domain_name: str, severity: int, images: torch.Tensor,
               labels: torch.Tensor, preds: torch.Tensor, confs: torch.Tensor,
               masked_images: torch.Tensor = None, indices: torch.Tensor = None):
        try:
            images = images.detach().cpu()
            labels = labels.detach().cpu()
            preds = preds.detach().cpu()
            confs = confs.detach().cpu()
            if masked_images is not None:
                masked_images = masked_images.detach().cpu()
            if indices is not None:
                indices = indices.detach().cpu()
        except Exception:
            return

        dom_str = str(domain_name)
        sev = int(severity)

        # Determine and cache the reference domain for this severity (first seen)
        if sev not in self.reference_domain:
            self.reference_domain[sev] = dom_str
        is_ref = (self.reference_domain[sev] == dom_str)

        key_dom = (dom_str, sev)
        # Map from global index -> sample dict for this (domain, severity), but
        # only for the tracked least-confident indices (up to max_per_type)
        per_domain = self.samples.get(key_dom, {})

        # For non-reference domains, determine which indices we care about
        # based on the least-confident candidates selected on the reference
        # domain so far.
        ref_idx_set = set()
        if not is_ref:
            cands = self.ref_candidates.get(sev, [])
            ref_idx_set = {s["index"] for s in cands if s.get("index") is not None}

        b = images.shape[0]
        for i in range(b):
            img = images[i]
            lbl = int(labels[i].item())
            pred = int(preds[i].item())
            conf = float(confs[i].item())
            masked = None
            if masked_images is not None and i < masked_images.shape[0]:
                masked = masked_images[i]

            idx_global = int(indices[i].item()) if indices is not None else None
            if idx_global is None:
                continue

            sample = {"image": img, "label": lbl, "pred": pred, "conf": conf,
                      "masked": masked, "index": idx_global}

            if is_ref:
                # On the reference domain, maintain a running set of
                # least-confident samples globally for this severity.
                cands = self.ref_candidates.get(sev, [])
                cands.append(sample)
                cands.sort(key=lambda s: s["conf"])  # ascending = least confident first
                if len(cands) > self.max_per_type:
                    cands.pop(-1)
                self.ref_candidates[sev] = cands

                # Keep only candidates for this domain in memory to avoid
                # storing the full dataset.
                keep_idx = {s["index"] for s in cands if s.get("index") is not None}
                if idx_global in keep_idx:
                    per_domain[idx_global] = sample
                # Also drop any previously stored samples that are no longer
                # in the current least-confident set.
                for old_idx in list(per_domain.keys()):
                    if old_idx not in keep_idx:
                        per_domain.pop(old_idx, None)
            else:
                # On non-reference domains, only cache samples whose global
                # index matches one of the tracked least-confident indices.
                if idx_global in ref_idx_set:
                    per_domain[idx_global] = sample

        self.samples[key_dom] = per_domain

    def _save_with_footer(self, image_tensor: torch.Tensor, out_path: str,
                           pred: int = None, label: int = None, conf: float = None):
        """Save a tensor as an image with a text footer stacked vertically and centered.

        The footer contains three lines with values, e.g.:
        - Prediction: 123
        - Target: 45
        - Confidence: 97.30%
        """
        try:
            img = to_pil_image(image_tensor.clamp(0.0, 1.0))
        except Exception:
            # Fallback: try saving tensor directly without footer
            try:
                save_image(image_tensor.clamp(0.0, 1.0), out_path)
            except Exception:
                pass
            return

        w, h = img.size
        # Build text lines with actual values if provided
        if conf is not None:
            conf_pct = conf * 100.0
        else:
            conf_pct = None
        lines = [
            f"Prediction: {pred}" if pred is not None else "Prediction",
            f"Target: {label}" if label is not None else "Target",
            f"Confidence: {conf_pct:.2f}%" if conf_pct is not None else "Confidence %",
        ]

        # Choose a simple font
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # Measure text block size
        dummy_draw = ImageDraw.Draw(img)
        line_sizes = []
        max_width = 0
        for line in lines:
            if hasattr(dummy_draw, "textbbox"):
                bbox = dummy_draw.textbbox((0, 0), line, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            else:
                tw, th = dummy_draw.textsize(line, font=font)
            max_width = max(max_width, tw)
            line_sizes.append((tw, th))

        line_spacing = 4
        text_block_height = sum(th for _, th in line_sizes) + line_spacing * (len(lines) - 1)
        padding_y = 4

        new_h = h + text_block_height + 2 * padding_y
        # White footer band for better readability
        new_img = Image.new("RGB", (w, new_h), color=(255, 255, 255))
        new_img.paste(img, (0, 0))

        draw = ImageDraw.Draw(new_img)
        y = h + padding_y
        for (line, (tw, th)) in zip(lines, line_sizes):
            x = (w - tw) // 2
            # Black text on white band
            draw.text((x, y), line, fill=(0, 0, 0), font=font)
            y += th + line_spacing

        try:
            new_img.save(out_path)
        except Exception:
            # If annotated save fails, try raw tensor save as a last resort
            try:
                save_image(image_tensor.clamp(0.0, 1.0), out_path)
            except Exception:
                pass

    def save_all(self):
        if not self.samples:
            return
        # For each severity, extract the global indices of the least-confident
        # samples from the first (reference) domain.
        ref_indices_by_sev = {}
        for sev, cands in self.ref_candidates.items():
            if not cands:
                continue
            # Ensure sorted by increasing confidence
            cands_sorted = sorted(cands, key=lambda s: s["conf"])
            ref_indices_by_sev[sev] = [s["index"] for s in cands_sorted if s.get("index") is not None]

        # Now, for every (domain, severity), only save those reference indices,
        # so that the same underlying images are saved across all domains.
        for (domain, severity), per_domain in self.samples.items():
            sev = int(severity)
            ref_indices = ref_indices_by_sev.get(sev, None)
            if not ref_indices:
                continue

            bucket_name = "tracked_least_confident"
            subdir = self.root_dir / f"{self.dataset_name}_{self.arch_name}" / f"{domain}_s{severity}" / bucket_name
            os.makedirs(subdir, exist_ok=True)

            for rank, idx_global in enumerate(ref_indices):
                sample = per_domain.get(idx_global, None)
                if sample is None:
                    continue

                # Include predicted and true class indices in the filename for easier inspection
                base_name = f"{domain}_s{severity}_p{sample['pred']}_y{sample['label']}_{bucket_name}_{rank}"
                img_path = subdir / f"{base_name}_orig.png"
                try:
                    self._save_with_footer(
                        sample["image"],
                        str(img_path),
                        pred=sample["pred"],
                        label=sample["label"],
                        conf=sample["conf"],
                    )
                except Exception:
                    continue

                if self.save_masked and sample["masked"] is not None:
                    masked_path = subdir / f"{base_name}_masked.png"
                    try:
                        self._save_with_footer(
                            sample["masked"],
                            str(masked_path),
                            pred=sample["pred"],
                            label=sample["label"],
                            conf=sample["conf"],
                        )
                    except Exception:
                        pass


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 print_every: int,
                 device: Union[str, torch.device],
                 batch_random: bool = False,
                 no_adapt: bool = False,
                 sample_collector=None,
                 severity: int = None):

    num_correct = 0.
    num_samples = 0
    # Choose which model to use for forward passes. When no_adapt is True,
    # we bypass adaptation and evaluate using the underlying base model.
    eval_model = _unwrap_base_model(model) if no_adapt else model

    with torch.no_grad():
        if not batch_random:
            for i, data in enumerate(data_loader):
                imgs, labels = data[0], data[1]

                output = eval_model([img.to(device) for img in imgs]) if isinstance(imgs, list) else eval_model(imgs.to(device))
                predictions = output.argmax(1)

                if dataset_name == "imagenet_d" and domain_name != "none":
                    mapping_vector = list(IMAGENET_D_MAPPING.values())
                    predictions = torch.tensor([mapping_vector[pred] for pred in predictions], device=device)

                num_correct += (predictions == labels.to(device)).float().sum()

                if "mixed_domains" in setting and len(data) >= 3:
                    domain_dict = split_results_by_domain(domain_dict, data, predictions)

                # Optionally collect a small set of analysis samples per domain
                if sample_collector is not None and severity is not None:
                    try:
                        if isinstance(imgs, list):
                            imgs_for_save = imgs[0].detach()
                        else:
                            imgs_for_save = imgs.detach()
                        probs = torch.softmax(output, dim=1)
                        confs, _ = probs.max(dim=1)
                        masked_batch = getattr(eval_model, "_last_masked", None)
                        # Use a stable per-sample index within this loader so that
                        # the same samples can be tracked across all domains.
                        batch_size = imgs_for_save.shape[0]
                        start_idx = num_samples
                        indices = torch.arange(start_idx, start_idx + batch_size, dtype=torch.long)
                        sample_collector.update(domain_name=domain_name,
                                                severity=int(severity),
                                                images=imgs_for_save,
                                                labels=labels,
                                                preds=predictions,
                                                confs=confs,
                                                masked_images=masked_batch,
                                                indices=indices)
                    except Exception:
                        pass

                # track progress
                batch_size = imgs[0].shape[0] if isinstance(imgs, list) else imgs.shape[0]
                num_samples += batch_size
                if print_every > 0 and (i+1) % print_every == 0:
                    logger.info(f"#batches={i+1:<6} #samples={num_samples:<9} error = {1 - num_correct / num_samples:.2%}")

                if dataset_name == "ccc" and num_samples >= 7500000:
                    break

        else:
            # Random per-sample batch sizes over the full dataset stream.
            candidate_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
            batch_hist = Counter()
            data_iter = iter(data_loader)

            buffer_imgs = None
            buffer_labels = None
            buffer_domains = [] if "mixed_domains" in setting else None

            done = False
            i_batch = 0

            while not done or (buffer_labels is not None and buffer_labels.shape[0] > 0):
                # Ensure there is at least one sample in the buffer
                if (buffer_labels is None or buffer_labels.shape[0] == 0) and not done:
                    try:
                        data = next(data_iter)
                    except StopIteration:
                        done = True
                    else:
                        imgs, labels = data[0], data[1]

                        if isinstance(imgs, list):
                            if buffer_imgs is None:
                                buffer_imgs = [img for img in imgs]
                            else:
                                buffer_imgs = [torch.cat([buf, img], dim=0) for buf, img in zip(buffer_imgs, imgs)]
                        else:
                            if buffer_imgs is None:
                                buffer_imgs = imgs
                            else:
                                buffer_imgs = torch.cat([buffer_imgs, imgs], dim=0)

                        if buffer_labels is None:
                            buffer_labels = labels
                        else:
                            buffer_labels = torch.cat([buffer_labels, labels], dim=0)

                        if "mixed_domains" in setting and len(data) >= 3:
                            domains = data[2]
                            if buffer_domains is not None:
                                if isinstance(domains, torch.Tensor):
                                    buffer_domains.extend(domains.tolist())
                                else:
                                    buffer_domains.extend(list(domains))

                        continue

                if buffer_labels is None or buffer_labels.shape[0] == 0:
                    break

                # Draw a random target batch size and, if possible, prefetch more
                # data so that the buffer contains at least that many samples.
                target_bs = random.choice(candidate_batch_sizes)
                while buffer_labels.shape[0] < target_bs and not done:
                    try:
                        data = next(data_iter)
                    except StopIteration:
                        done = True
                        break
                    else:
                        imgs, labels = data[0], data[1]

                        if isinstance(imgs, list):
                            if buffer_imgs is None:
                                buffer_imgs = [img for img in imgs]
                            else:
                                buffer_imgs = [torch.cat([buf, img], dim=0) for buf, img in zip(buffer_imgs, imgs)]
                        else:
                            if buffer_imgs is None:
                                buffer_imgs = imgs
                            else:
                                buffer_imgs = torch.cat([buffer_imgs, imgs], dim=0)

                        if buffer_labels is None:
                            buffer_labels = labels
                        else:
                            buffer_labels = torch.cat([buffer_labels, labels], dim=0)

                        if "mixed_domains" in setting and len(data) >= 3:
                            domains = data[2]
                            if buffer_domains is not None:
                                if isinstance(domains, torch.Tensor):
                                    buffer_domains.extend(domains.tolist())
                                else:
                                    buffer_domains.extend(list(domains))

                if buffer_labels is None or buffer_labels.shape[0] == 0:
                    break

                available = buffer_labels.shape[0]
                bs = min(target_bs, available)
                batch_hist[bs] += 1

                # Form the micro-batch from the buffer
                if isinstance(buffer_imgs, list):
                    imgs_sub = [img[:bs].to(device) for img in buffer_imgs]
                    buffer_imgs = [img[bs:] for img in buffer_imgs]
                else:
                    imgs_sub = buffer_imgs[:bs].to(device)
                    buffer_imgs = buffer_imgs[bs:]

                labels_sub = buffer_labels[:bs]
                buffer_labels = buffer_labels[bs:]

                domains_sub = None
                if "mixed_domains" in setting and buffer_domains is not None and len(buffer_domains) > 0:
                    domains_sub = buffer_domains[:bs]
                    buffer_domains = buffer_domains[bs:]

                output = eval_model(imgs_sub) if isinstance(imgs_sub, list) else eval_model(imgs_sub)
                predictions = output.argmax(1)

                if dataset_name == "imagenet_d" and domain_name != "none":
                    mapping_vector = list(IMAGENET_D_MAPPING.values())
                    predictions = torch.tensor([mapping_vector[pred] for pred in predictions], device=device)

                num_correct += (predictions == labels_sub.to(device)).float().sum()

                if "mixed_domains" in setting and domains_sub is not None:
                    data_sub = [None, labels_sub, domains_sub]
                    domain_dict = split_results_by_domain(domain_dict, data_sub, predictions)

                num_samples += bs
                i_batch += 1
                # per-iteration logging removed for clarity at scale

                if dataset_name == "ccc" and num_samples >= 7500000:
                    break

            # Log a compact histogram of used random batch sizes for this evaluation
            if len(batch_hist) > 0:
                logger.info("Random batch size histogram for this evaluation:")
                for bs_val, count in sorted(batch_hist.items()):
                    logger.info(f"  batch_size={bs_val:<3} count={count}")

    accuracy = num_correct.item() / num_samples
    return accuracy, domain_dict, num_samples
