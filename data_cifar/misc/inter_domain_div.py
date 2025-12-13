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

# For legend names, mirror batch_metrics.py
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


def _js_from_features(
    feats_a: np.ndarray, feats_b: np.ndarray, num_bins: int = 50
) -> float:
    x = np.asarray(feats_a, dtype=float).ravel()
    y = np.asarray(feats_b, dtype=float).ravel()

    if x.size == 0 or y.size == 0:
        return 0.0

    vmin = float(min(x.min(), y.min()))
    vmax = float(max(x.max(), y.max()))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return 0.0

    hist_a, _ = np.histogram(x, bins=num_bins, range=(vmin, vmax), density=False)
    hist_b, _ = np.histogram(y, bins=num_bins, range=(vmin, vmax), density=False)

    p = hist_a.astype(float)
    q = hist_b.astype(float)
    eps = 1e-12
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def kl_div(r: np.ndarray, s: np.ndarray) -> float:
        return float(np.sum(r * np.log(r / s)))

    js = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
    return js


def compute_js_series_for_file(path: str, use_what: str) -> Tuple[np.ndarray, Dict[str, Dict]]:
    domains = _load_domains_from_npy(path)

    ordered_feats: List[np.ndarray] = []
    for corr in CORRUPTIONS:
        dom = domains[corr]
        arr = np.asarray(dom[use_what])
        ordered_feats.append(arr)

    js_vals: List[float] = []
    for i in range(len(CORRUPTIONS) - 1):
        f_a = ordered_feats[i]
        f_b = ordered_feats[i + 1]
        js = _js_from_features(f_a, f_b)
        js_vals.append(js)

    return np.array(js_vals, dtype=float), domains


def main():
    parser = argparse.ArgumentParser(
        description="Measure JS feature divergence between successive corruptions."
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
        choices=["features", "logits", "probabilities"],
        help="Which representation to use for divergence (features, logits, or probabilities).",
    )
    args = parser.parse_args()

    if len(args.npy) != len(args.method):
        raise ValueError("--npy and --method must have the same length")

    np.random.seed(int(args.seed))

    methods_canonical = _canonicalize_methods(args.method)

    js_series_list: List[np.ndarray] = []
    method_tokens_in_file: List[str] = []

    # For optional consistency checks between --method and npy content
    expected_token_map = {
        "Source": "source",
        "Tent": "tent",
        "CoTTA": "cotta",
        "Continual-MAE": "continual_mae",
        "REM": "rem",
        "M2A (Spatial)": "m2a_spatial",
        "M2A (Frequency)": "m2a_spectral",
    }

    for npy_path, legend_name in zip(args.npy, methods_canonical):
        npy_path = os.path.expanduser(npy_path)
        if not os.path.isfile(npy_path):
            raise FileNotFoundError(f"Cannot find npy file: {npy_path}")

        js_vals, domains = compute_js_series_for_file(npy_path, args.use_what)
        js_series_list.append(js_vals)

        # Inspect one domain to get method token stored in file
        any_dom = next(iter(domains.values()))
        in_file_method = str(any_dom.get("method", "")).lower()
        method_tokens_in_file.append(in_file_method)

        expected_token = expected_token_map.get(legend_name)
        if expected_token and not in_file_method.startswith(expected_token):
            print(
                f"Warning: file {npy_path} has method '{in_file_method}', "
                f"which does not start with expected token '{expected_token}' "
                f"for legend '{legend_name}'",
                file=sys.stderr,
            )

    if not js_series_list:
        raise RuntimeError("No JS divergence series computed")

    # All series should have the same length (#pairs = len(CORRUPTIONS)-1)
    num_pairs = len(js_series_list[0])
    for s in js_series_list:
        if len(s) != num_pairs:
            raise RuntimeError("JS series have mismatched lengths")

    # Joint normalization across methods to [0, 1] when comparing multiple inputs
    normalized = False
    if len(args.npy) > 1 and len(args.method) > 1:
        all_vals = np.concatenate(js_series_list)
        vmin = float(all_vals.min())
        vmax = float(all_vals.max())
        if vmax > vmin:
            js_series_list = [(vals - vmin) / (vmax - vmin) for vals in js_series_list]
            normalized = True

    xs = np.arange(1, num_pairs + 1)
    xtick_labels = [f"c{i}-c{i+1}" for i in range(1, num_pairs + 1)]

    plt.figure(figsize=(10, 5))

    for js_vals, legend_name in zip(js_series_list, methods_canonical):
        plt.plot(xs, js_vals, label=legend_name, linewidth=5)

    base_fs = float(plt.rcParams.get("font.size", 7.0))
    tick_fs = base_fs * 1.5
    legend_fs = base_fs * 1.7

    ax = plt.gca()
    if normalized:
        ax.set_ylim(0.0, 1.0)

    # X-axis: only 4 ticks at specific pairs with custom labels
    # pairs: c3-c4, c6-c7, c9-10, c12-c13
    desired_pairs = [3, 6, 9, 12]
    label_map = {
        3: "c3-c4",
        6: "c6-c7",
        9: "c9-10",
        12: "c12-c13",
    }
    xticks = [p for p in desired_pairs if 1 <= p <= num_pairs]
    xlabels = [label_map[p] for p in xticks]

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

    if num_pairs > 0:
        plt.xlim(xs[0], xs[-1])

    os.makedirs(args.outdir, exist_ok=True)

    stems = [os.path.splitext(os.path.basename(p))[0] for p in args.npy]
    outfile = "inter_domain_divergence.jpg"
    out_path = os.path.join(args.outdir, outfile)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()


# python data_cifar/misc/inter_domain_div.py --npy /flash/project_465002264/projects/m2a/feat/source_5.npy /flash/project_465002264/projects/m2a/feat/rem_5.npy /flash/project_465002264/projects/m2a/feat/m2a_spatial_5.npy /flash/project_465002264/projects/m2a/feat/m2a_spectral_5.npy --method "source" "continual_mae" "rem" "m2a(spatial)" "m2a(frequency)" --seed 1