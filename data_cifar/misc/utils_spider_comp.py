#!/usr/bin/env python3
import argparse
import os
import re
import math
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Canonical metric names recognized in logs
CANONICAL_METRICS = [
    "Error",
    "NLL",
    "ECE",
    "Max Softmax",
    "Entropy",
    "Cosine",
]


def canonicalize_metric_name(raw: str) -> str:
    s = raw.strip()
    if s.startswith("Error"):
        return "Error"
    if s.startswith("NLL"):
        return "NLL"
    if s.startswith("ECE"):
        return "ECE"
    if s.startswith("Max Softmax"):
        return "Max Softmax"
    if s.startswith("Entropy"):
        return "Entropy"
    if s.startswith("Cosine"):
        return "Cosine"
    return ""


def parse_log_file(path: str, wanted_metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """Parse a log file returning {metric: {corruption: value}} for the selected metrics.

    Lines look like:
    - "Error % [gaussian_noise5]: 17.53%"
    - "NLL [gaussian_noise5]: 0.6074"
    - "ECE [gaussian_noise5]: 0.0761"
    - "Max Softmax [gaussian_noise5]: 0.9002"
    - "Entropy [gaussian_noise5]: 0.3019"
    - "Cosine(pred_softmax, target_onehot) [gaussian_noise5]: 0.8611"
    - "Adaptation Time ... s" and "Adaptation MACs ..." are ignored here
    """
    metrics: Dict[str, Dict[str, float]] = {m: {} for m in wanted_metrics}
    # Regex to capture the metric name, corruption, and value
    pattern = re.compile(r"\]:\s*(?P<metric>[^\[]+)\[(?P<corr>[^\]]+)\]:\s*(?P<val>[^\s]+)")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            raw_metric = m.group("metric").strip()
            corr = m.group("corr").strip()
            val_str = m.group("val").strip()
            canon = canonicalize_metric_name(raw_metric)
            if canon == "" or canon not in wanted_metrics:
                continue
            # Clean value: strip trailing '%' or 's'
            val_str = val_str.rstrip('%s')
            try:
                val = float(val_str)
            except Exception:
                # Try to handle scientific notation embedded or extra chars
                try:
                    val = float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val_str)[0])
                except Exception:
                    continue
            # For Error %, keep numeric value as-is (percentage number)
            metrics[canon][corr] = val
    return metrics


def shorthand_from_corruption(corr: str) -> str:
    """Convert corruption name (e.g., gaussian_noise5) to initial caps (e.g., GN)."""
    # strip trailing digits at end (severity)
    base = re.sub(r"\d+$", "", corr)
    base = base.strip("_")
    # collapse multiple underscores
    parts = [p for p in base.split("_") if p]
    if not parts:
        return corr.upper()
    initials = ''.join([p[0].upper() for p in parts])
    return initials


def prepare_axes_order(all_corr_sets: List[Dict[str, float]]) -> List[str]:
    """Determine a stable order of corruptions based on first occurrence across logs."""
    seen = []
    for d in all_corr_sets:
        for c in d.keys():
            if c not in seen:
                seen.append(c)
    return seen


def align_values(corr_order: List[str], values_by_corr: Dict[str, float]) -> np.ndarray:
    arr = []
    for c in corr_order:
        v = values_by_corr.get(c, np.nan)
        arr.append(v)
    return np.array(arr, dtype=float)


def make_radar_subplot(ax, labels: List[str], series: List[np.ndarray], names: List[str], colors: List[str]):
    N = len(labels)
    if N == 0:
        return
    # angles for each axis + close the loop
    angles = np.linspace(0, 2 * math.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # set polar
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    # draw axes and labels with font sizing (2x for angular labels)
    ax.set_xticks(angles[:-1])
    base_fs = float(plt.rcParams.get('font.size', 10.0))
    big_fs = base_fs * 2.0
    ax.set_xticklabels(labels, fontsize=big_fs)

    # radial limits auto; grid alpha
    ax.grid(True, alpha=0.3)
    # Radial tick labels keep base size
    try:
        for lab in ax.get_yticklabels():
            lab.set_fontsize(base_fs)
    except Exception:
        pass

    # plot each series
    for y, name, color in zip(series, names, colors):
        y_f = np.array(y, dtype=float)
        # Replace nans with 0 for plotting
        y_f = np.nan_to_num(y_f, nan=0.0)
        y_plot = y_f.tolist() + y_f[:1].tolist()
        ax.plot(angles, y_plot, linewidth=2, label=name, color=color)
        ax.fill(angles, y_plot, alpha=0.10, color=color)


def main():
    parser = argparse.ArgumentParser(description="Spider (radar) comparison of metrics per corruption across logs")
    parser.add_argument('--logs', nargs='+', required=True, help='Paths to log files')
    parser.add_argument('--names', nargs='+', required=True, help='Legend names, one per log')
    parser.add_argument('--metrics', nargs='+', required=True,
                        help='Metrics to plot (choose from: Error, NLL, ECE, Max Softmax, Entropy, Cosine)')
    parser.add_argument('--outdir', default=os.path.dirname('/users/doloriel/work/Repo/M2A/cifar/plots/SPARE/Misc/cifar10/error_rate_samples.png'),
                        help='Output directory (default: same dir as error_rate_samples.png)')
    parser.add_argument('--outfile', default='spider.png', help='Output filename (default: spider.png)')
    args = parser.parse_args()

    if len(args.logs) != len(args.names):
        raise ValueError('--logs and --names must have the same length')

    # Canonicalize metric selection
    wanted = []
    for m in args.metrics:
        cm = canonicalize_metric_name(m)
        if cm == "":
            # Try simple aliases
            ms = m.strip().lower().replace('_', ' ')
            if ms in ['max softmax', 'maxsoftmax']:
                cm = 'Max Softmax'
            elif ms in ['cos', 'cosine']:
                cm = 'Cosine'
            elif ms in ['error', 'err', 'error %']:
                cm = 'Error'
            elif ms in ['nll']:
                cm = 'NLL'
            elif ms in ['ece']:
                cm = 'ECE'
            elif ms in ['entropy']:
                cm = 'Entropy'
        if cm and cm in CANONICAL_METRICS and cm not in wanted:
            wanted.append(cm)
    if not wanted:
        raise ValueError('No valid metrics requested. Choose from: ' + ', '.join(CANONICAL_METRICS))

    # Parse logs
    per_log_metrics: List[Dict[str, Dict[str, float]]] = []
    for log in args.logs:
        per_log_metrics.append(parse_log_file(log, wanted))

    # Build a consistent color list for all subplots and series
    default_colors = plt.rcParams.get('axes.prop_cycle', None)
    if default_colors is not None:
        default_colors = default_colors.by_key().get('color', [])
    if not default_colors:
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors = [default_colors[i % len(default_colors)] for i in range(len(args.names))]

    # For each metric, determine axis order and assemble series for each log
    n_metrics = len(wanted)
    ncols = min(3, n_metrics)
    nrows = int(math.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True),
                             figsize=(5 * ncols, 4.5 * nrows))
    # Normalize axes handling for single subplot
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])

    for idx, metric in enumerate(wanted):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        # Collect corruption sets across logs for this metric
        corr_sets = [m[metric] for m in per_log_metrics]
        corr_order = prepare_axes_order(corr_sets)
        # Axis labels: initials in caps
        labels = [shorthand_from_corruption(c) for c in corr_order]
        # Series per log
        series = [align_values(corr_order, d) for d in corr_sets]
        make_radar_subplot(ax, labels, series, args.names, colors)
        # No subplot titles per requirement

    # Hide any unused subplot axes
    total_axes = nrows * ncols
    for j in range(n_metrics, total_axes):
        r = j // ncols
        c = j % ncols
        fig.delaxes(axes[r][c])

    # Legend with explicit colored handles to ensure matching colors
    handles = [Line2D([0], [0], color=colors[i], lw=2) for i in range(len(args.names))]
    base_fs = float(plt.rcParams.get('font.size', 10.0))
    big_fs = base_fs
    fig.legend(handles, args.names, loc='lower center', ncol=min(len(args.names), 4), prop={'size': big_fs})
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, args.outfile)
    fig.savefig(out_path, dpi=200)
    print(f"Saved spider plot to: {out_path}")


if __name__ == '__main__':
    main()
