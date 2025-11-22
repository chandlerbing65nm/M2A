#!/usr/bin/env python3
import argparse
import os
import re
import math
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


def canonicalize_arg_metric(m: str) -> str:
    cm = canonicalize_metric_name(m)
    if cm:
        return cm
    ms = m.strip().lower().replace("_", " ")
    if ms in ["max softmax", "maxsoftmax"]:
        return "Max Softmax"
    if ms in ["cos", "cosine"]:
        return "Cosine"
    if ms in ["error", "err", "error %"]:
        return "Error"
    if ms in ["nll"]:
        return "NLL"
    if ms in ["ece"]:
        return "ECE"
    if ms in ["entropy"]:
        return "Entropy"
    return ""


def parse_metric_from_log(path: str, wanted_metric: str) -> Dict[str, float]:
    d: Dict[str, float] = {}
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
            if canon == "" or canon != wanted_metric:
                continue
            val_str = val_str.rstrip("%s")
            try:
                val = float(val_str)
            except Exception:
                try:
                    val = float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val_str)[0])
                except Exception:
                    continue
            d[corr] = val
    return d


def shorthand_from_corruption(corr: str) -> str:
    base = re.sub(r"\d+$", "", corr)
    base = base.strip("_")
    parts = [p for p in base.split("_") if p]
    if not parts:
        return corr.upper()
    initials = "".join([p[0].upper() for p in parts])
    return initials


def prepare_axes_order(all_corr_sets: List[Dict[str, float]]) -> List[str]:
    seen: List[str] = []
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


def main():
    parser = argparse.ArgumentParser(description="Ablations radar: compare one metric across logs")
    parser.add_argument("--logs", nargs="+", required=True, help="Paths to log files")
    parser.add_argument("--names", nargs="+", required=True, help="Legend names, one per log")
    parser.add_argument("--metric", default="Error", help="Metric to plot (default: Error)")
    parser.add_argument("--outdir", default=os.getcwd(), help="Output directory")
    parser.add_argument("--outfile", default="ablations_radar.png", help="Output filename")
    args = parser.parse_args()

    if len(args.logs) != len(args.names):
        raise ValueError("--logs and --names must have the same length")

    sel = canonicalize_arg_metric(args.metric)
    if sel == "" or sel not in CANONICAL_METRICS:
        raise ValueError("Invalid --metric. Choose from: " + ", ".join(CANONICAL_METRICS))

    per_log = []
    for p in args.logs:
        per_log.append(parse_metric_from_log(p, sel))

    corr_order = prepare_axes_order(per_log)
    if len(corr_order) == 0:
        raise ValueError("No matching metric entries found in provided logs")

    labels = [shorthand_from_corruption(c) for c in corr_order]
    series = [align_values(corr_order, d) for d in per_log]

    prop = plt.rcParams.get("axes.prop_cycle", None)
    if prop is not None:
        base_colors = prop.by_key().get("color", [])
    else:
        base_colors = []
    if not base_colors:
        base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    colors = [base_colors[i % len(base_colors)] for i in range(len(series))]

    all_vals = np.concatenate(series) if series else np.array([0.0])
    finite_vals = all_vals[np.isfinite(all_vals)]
    if finite_vals.size == 0:
        ymin, ymax = 0.0, 1.0
    else:
        y_min_val = float(np.nanmin(finite_vals))
        y_max_val = float(np.nanmax(finite_vals))
        ymin = max(0.0, y_min_val - 0.1 * abs(y_min_val))
        ymax = y_max_val + 0.1 * abs(y_max_val)

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))

    N = len(labels)
    angles = np.linspace(0, 2 * math.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])

    base_fs = float(plt.rcParams.get("font.size", 10.0))
    big_fs = base_fs * 2.0
    ax.set_xticklabels(labels, fontsize=big_fs)

    for lab in ax.get_yticklabels():
        lab.set_fontsize(base_fs)

    ax.grid(True, alpha=0.3)
    ax.set_ylim(ymin, ymax)

    for y, name, color in zip(series, args.names, colors):
        y_f = np.array(y, dtype=float)
        y_f = np.nan_to_num(y_f, nan=0.0)
        y_plot = y_f.tolist() + y_f[:1].tolist()
        ax.plot(angles, y_plot, linewidth=2, label=name, color=color)
        ax.fill(angles, y_plot, alpha=0.10, color=color)

    handles = [Line2D([0], [0], color=colors[i], lw=2) for i in range(len(args.names))]
    fig.legend(handles, args.names, loc="lower center", ncol=min(len(args.names), 4), prop={"size": big_fs})
    fig.tight_layout(rect=(0, 0.15, 1, 1))

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, args.outfile)
    fig.savefig(out_path, dpi=200)
    print(f"Saved ablations radar to: {out_path}")


if __name__ == "__main__":
    main()
