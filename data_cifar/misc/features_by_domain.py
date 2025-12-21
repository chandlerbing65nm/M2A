#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

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
    # More granular M2A (Spatial) variants (accepted as CLI method names)
    "M2A (Spatial, EML only)",
    "M2A (Spatial, MCL only)",
    "M2A (Spatial, EML+MCL)",
    # Additional granular M2A variants used in feature naming
    "M2A (Spatial, Patch)",
    "M2A (Spatial, Pixel)",
    "M2A (Frequency, All)",
    "M2A (Frequency, Low)",
    "M2A (Frequency, High)",
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
        if use_what not in dom:
            sample_keys = sorted(list(dom.keys()))
            raise KeyError(
                f"Key '{use_what}' not found in domain '{corr}' of file {path}.\n"
                f"Available keys include: {sample_keys}"
            )
        arr = np.asarray(dom[use_what])
        ordered_feats.append(arr)

    js_vals: List[float] = []
    for i in range(len(CORRUPTIONS) - 1):
        f_a = ordered_feats[i]
        f_b = ordered_feats[i + 1]
        js = _js_from_features(f_a, f_b)
        js_vals.append(js)

    return np.array(js_vals, dtype=float), domains


def run_tsne(features: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=30.0,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    feats = np.asarray(features, dtype=np.float32)
    emb = tsne.fit_transform(feats)

    min_vals = emb.min(axis=0, keepdims=True)
    max_vals = emb.max(axis=0, keepdims=True)
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0
    emb_norm = 2.0 * (emb - min_vals) / denom - 1.0
    return emb_norm


def plot_tsne(embeddings_2d: np.ndarray, labels: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    labels_arr = np.asarray(labels)
    num_classes = int(len(np.unique(labels_arr))) if labels_arr.size > 0 else 0

    plt.figure(figsize=(8, 8))

    cmap_name = "tab10" if num_classes <= 10 else "tab20"
    plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels_arr,
        cmap=cmap_name,
        s=5,
        alpha=0.7,
    )

    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize per-class features from a chosen domain using t-SNE."
    )
    parser.add_argument(
        "--npy",
        required=True,
        help="Path to a single .npy feature file.",
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
        "--tag",
        default="",
        help="Optional tag to append to the output filename.",
    )
    parser.add_argument(
        "--use_what",
        default="class_features_12",
        help="Which blockwise key to use (e.g., class_features_1..N, patch_mean_1..N, patch_std_1..N).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=CORRUPTIONS,
        help="Which corruption/domain to visualize (e.g., gaussian_noise, impulse_noise).",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="",
        help="Output filename (without path); saved under --outdir.",
    )
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    npy_path = os.path.expanduser(args.npy)
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"Cannot find npy file: {npy_path}")

    domains = _load_domains_from_npy(npy_path)
    if args.domain not in domains:
        available = sorted(list(domains.keys()))
        raise KeyError(
            f"Domain '{args.domain}' not found in file {npy_path}. Available domains: {available}"
        )

    dom = domains[args.domain]
    if args.use_what not in dom:
        sample_keys = sorted(list(dom.keys()))
        raise KeyError(
            f"Key '{args.use_what}' not found in domain '{args.domain}' of file {npy_path}.\n"
            f"Available keys include: {sample_keys}"
        )
    if "labels" not in dom:
        raise KeyError(
            f"Domain '{args.domain}' in file {npy_path} does not contain 'labels' key."
        )

    feats = np.asarray(dom[args.use_what])
    labels = np.asarray(dom["labels"])

    emb_2d = run_tsne(feats, labels, seed=int(args.seed))

    os.makedirs(args.outdir, exist_ok=True)
    if args.outfile:
        filename = args.outfile
    else:
        stem = os.path.splitext(os.path.basename(npy_path))[0]
        tag_suffix = f"_{args.tag}" if getattr(args, "tag", "") else ""
        filename = f"tsne_{stem}_{args.domain}_{args.use_what}{tag_suffix}.jpg"
    out_path = os.path.join(args.outdir, filename)

    plot_tsne(emb_2d, labels, out_path)
    print(f"Saved t-SNE plot to: {out_path}")


if __name__ == "__main__":
    main()


# python data_cifar/misc/inter_domain_div.py --npy /flash/project_465002264/projects/m2a/feat/source_5.npy /flash/project_465002264/projects/m2a/feat/rem_5.npy /flash/project_465002264/projects/m2a/feat/m2a_spatial_5.npy /flash/project_465002264/projects/m2a/feat/m2a_spectral_5.npy --method "source" "continual_mae" "rem" "m2a(spatial)" "m2a(frequency)" --seed 1