#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Canonical CIFAR-10-C corruption order (15 corruptions)
CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

# For legend names, mirror inter_domain_div.py / batch_metrics.py
VALID_METHODS = {
    "Source",
    "Tent",
    "CoTTA",
    "Continual-MAE",
    "REM",
    "M2A (Spatial)",
    "M2A (Frequency)",
}


def _norm_method_name(name: str) -> str:
    return re.sub(r"[\s_\-()]+", "", name.strip().lower())


def _canonicalize_methods(raw_methods: List[str]) -> List[str]:
    canonical_map = {_norm_method_name(m): m for m in VALID_METHODS}
    methods: List[str] = []
    for m in raw_methods:
        key = _norm_method_name(m)
        if key not in canonical_map:
            raise ValueError(
                f"Invalid method '{m}'. Must be one of: "
                + ", ".join(sorted(VALID_METHODS))
            )
        methods.append(canonical_map[key])
    return methods


def _parse_corruption_name_from_domain_id(domain_id: str) -> str:
    # domain_id like "gaussian_noise_5"
    parts = domain_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return domain_id


def _load_domains_from_npy(path: str) -> Dict[str, Dict]:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, dict):
        data = arr
    else:
        try:
            data = arr.item()
        except Exception as e:
            raise TypeError(f"File {path} does not contain a pickled dict: {e}")

    if not isinstance(data, dict):
        raise TypeError(f"Top-level object in {path} is not a dict")

    domains: Dict[str, Dict] = {}
    for key, dom in data.items():
        if not isinstance(dom, dict):
            continue
        domain_id = dom.get("domain_id", key)
        corr = _parse_corruption_name_from_domain_id(str(domain_id))
        domains[corr] = dom

    missing = [c for c in CORRUPTIONS if c not in domains]
    if missing:
        raise ValueError(
            f"File {path} is missing corruptions: {', '.join(missing)} "
            f"(has: {sorted(domains.keys())})"
        )

    return domains


def _compute_intra_class_divergence(features: np.ndarray, labels: np.ndarray) -> float:
    """Compute intra-class divergence for a single domain.

    For each class: compute centroid and mean L2 distance of samples to centroid.
    Then average these mean distances across classes.
    """
    feats = np.asarray(features, dtype=float)
    labs = np.asarray(labels)

    if feats.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {feats.shape}")
    if feats.shape[0] != labs.shape[0]:
        raise ValueError(
            f"Mismatch between features and labels: {feats.shape[0]} vs {labs.shape[0]}"
        )

    uniq = np.unique(labs)
    class_means: List[float] = []
    for c in uniq:
        mask = labs == c
        feats_c = feats[mask]
        if feats_c.shape[0] <= 1:
            # Not enough samples to define spread
            continue
        centroid = feats_c.mean(axis=0)
        diffs = feats_c - centroid
        sq_dists = np.sum(diffs * diffs, axis=1)
        class_means.append(float(sq_dists.mean()))

    if not class_means:
        return 0.0
    return float(np.mean(class_means))


def compute_intra_class_series_for_file(path: str, use_what: str) -> Tuple[np.ndarray, Dict[str, Dict]]:
    domains = _load_domains_from_npy(path)

    divergences: List[float] = []
    for corr in CORRUPTIONS:
        dom = domains[corr]
        if use_what not in dom:
            feat_keys = sorted([k for k in dom.keys() if (k == "features" or k.startswith("features_"))])
            raise KeyError(
                f"Key '{use_what}' not found in domain '{corr}'. Available feature keys: {feat_keys}"
            )
        feats = np.asarray(dom[use_what])
        labs = np.asarray(dom["labels"])
        div = _compute_intra_class_divergence(feats, labs)
        divergences.append(div)

    return np.array(divergences, dtype=float), domains


def _infer_severity_from_npy(npy_path: str, domains: Dict[str, Dict]) -> str:
    """Infer severity from filename or domain_id suffix.

    Expected patterns like 'source_5.npy' or 'gaussian_noise_5'.
    """
    base = os.path.splitext(os.path.basename(npy_path))[0]
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[1]

    # Fallback: use any domain_id
    if domains:
        any_dom = next(iter(domains.values()))
        dom_id = str(any_dom.get("domain_id", ""))
        dparts = dom_id.rsplit("_", 1)
        if len(dparts) == 2 and dparts[1].isdigit():
            return dparts[1]

    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Measure intra-class feature divergence across corruptions."
    )
    parser.add_argument(
        "--npy",
        nargs="+",
        required=True,
        help="Paths to .npy files (one per method).",
    )
    parser.add_argument(
        "--method",
        nargs="+",
        required=True,
        help=(
            "Method name per npy: "
            "source, tent, cotta, continual_mae, rem, m2a(spatial), m2a(frequency)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (for reproducibility, if any randomness is used).",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(ROOT_DIR, "plots", "divergence"),
        help="Output directory for plots (default: data_cifar/plots/divergence).",
    )
    parser.add_argument(
        "--use_what",
        default="features",
        help="Which representation to use (e.g., 'features' or 'features_1', 'features_2', ...)",
    )
    args = parser.parse_args()

    if len(args.npy) != len(args.method):
        raise ValueError("--npy and --method must have the same length")

    np.random.seed(int(args.seed))

    methods_canonical = _canonicalize_methods(args.method)

    intra_series_list: List[np.ndarray] = []
    severities: List[str] = []

    # For consistency checks between --method and npy content
    expected_token_map = {
        "Source": "source",
        "Tent": "tent",
        "CoTTA": "cotta",
        "Continual-MAE": "continual_mae",
        "REM": "rem",
        "M2A (Spatial)": "m2a_spatial",
        "M2A (Frequency)": "m2a_spectral",
    }

    first_severity: str = "unknown"
    severity_set = set()

    for idx, (npy_path, legend_name) in enumerate(zip(args.npy, methods_canonical)):
        npy_path = os.path.expanduser(npy_path)
        if not os.path.isfile(npy_path):
            raise FileNotFoundError(f"Cannot find npy file: {npy_path}")

        series, domains = compute_intra_class_series_for_file(npy_path, args.use_what)
        intra_series_list.append(series)

        # Inspect one domain to get method token stored in file
        any_dom = next(iter(domains.values()))
        in_file_method = str(any_dom.get("method", "")).lower()

        expected_token = expected_token_map.get(legend_name)
        if expected_token and not in_file_method.startswith(expected_token):
            print(
                f"Warning: file {npy_path} has method '{in_file_method}', "
                f"which does not start with expected token '{expected_token}' "
                f"for legend '{legend_name}'",
                file=sys.stderr,
            )

        severity = _infer_severity_from_npy(npy_path, domains)
        severities.append(severity)
        severity_set.add(severity)
        if idx == 0:
            first_severity = severity

    if not intra_series_list:
        raise RuntimeError("No intra-class divergence series computed")

    # All series should have the same length (#domains = len(CORRUPTIONS))
    num_domains = len(intra_series_list[0])
    for s in intra_series_list:
        if len(s) != num_domains:
            raise RuntimeError("Intra-class series have mismatched lengths")

    # Joint normalization across methods to [0, 1] when comparing multiple inputs
    normalized = False
    if len(args.npy) > 1 and len(args.method) > 1:
        all_vals = np.concatenate(intra_series_list)
        vmin = float(all_vals.min())
        vmax = float(all_vals.max())
        if vmax > vmin:
            intra_series_list = [(vals - vmin) / (vmax - vmin) for vals in intra_series_list]
            normalized = True

    xs = np.arange(1, num_domains + 1)

    plt.figure(figsize=(10, 5))

    for series, legend_name in zip(intra_series_list, methods_canonical):
        plt.plot(xs, series, label=legend_name, linewidth=5)

    base_fs = float(plt.rcParams.get("font.size", 7.0))
    tick_fs = base_fs * 1.5
    legend_fs = base_fs * 1.7

    ax = plt.gca()
    if normalized:
        ax.set_ylim(0.0, 1.0)

    # X-axis: 15 domains labeled c1..c15 but only show c2,c4,c6,c8,c10,c12,c14
    desired_positions = [2, 4, 6, 8, 10, 12, 14]
    xticks = [p for p in desired_positions if 1 <= p <= num_domains]
    xlabels = [f"c{p}" for p in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    # Font sizes: x-axis tick labels same as legend; y-axis slightly smaller
    ax.tick_params(axis="x", which="major", labelsize=legend_fs)
    ax.tick_params(axis="y", which="major", labelsize=tick_fs)

    # Y-axis: keep only 5 tick labels
    yticks = ax.get_yticks()
    if len(yticks) > 5:
        idxs = np.linspace(0, len(yticks) - 1, 5, dtype=int)
        yticks_sel = yticks[idxs]
        ax.set_yticks(yticks_sel)

    plt.legend(prop={"size": legend_fs, "weight": "bold"})

    if num_domains > 0:
        plt.xlim(xs[0], xs[-1])

    os.makedirs(args.outdir, exist_ok=True)

    # If all severities agree, use that; otherwise, mark as 'mixed'
    if len(severity_set) == 1:
        severity_tag = first_severity
    else:
        severity_tag = "mixed"

    outfile = "intra_class_divergence.jpg"
    out_path = os.path.join(args.outdir, outfile)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()


# python data_cifar/misc/intra_class_div.py --npy /flash/project_465002264/projects/m2a/feat/source_5.npy /flash/project_465002264/projects/m2a/feat/rem_5.npy /flash/project_465002264/projects/m2a/feat/m2a_spatial_5.npy /flash/project_465002264/projects/m2a/feat/m2a_spectral_5.npy --method "source" "tent" "continual_mae" "m2a(spatial)" "m2a(frequency)" --seed 1
