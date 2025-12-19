#!/usr/bin/env python3
import argparse
import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_metric_series(log_path: str, metric_name: str) -> Tuple[List[str], List[float]]:
    """Parse a log file and extract per-corruption values for the given metric.

    Supported metrics (canonical tokens): ERROR, NLL, ECE.
    Returns (corruptions_in_order, values_in_order).
    """
    token = metric_name.strip().upper()
    assert token in {"ERROR", "NLL", "ECE"}, "--metric must be one of: Error, NLL, ECE"

    # Example lines:
    # [..] Error % [gaussian_noise5]: 15.24%
    # [..] error % [gaussian_noise5]: 60.08%
    # [..] NLL [gaussian_noise5]: 0.5138
    # [..] ECE [gaussian_noise5]: 0.0671
    pattern = re.compile(
        r"\]:\s*(?P<metric>Error|NLL|ECE)\s*(?:%)?\s*\[(?P<corr>[^\]]+)\]:\s*(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*%)?",
        re.IGNORECASE,
    )

    corrs: List[str] = []
    vals: List[float] = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            metric = m.group("metric").strip().upper()
            if metric != token:
                continue
            corr = m.group("corr").strip()
            val_str = m.group("val").strip()
            try:
                val = float(val_str)
            except Exception:
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val_str)
                if not nums:
                    continue
                val = float(nums[0])
            if corr not in corrs:
                corrs.append(corr)
                vals.append(val)
    return corrs, vals


def main():
    parser = argparse.ArgumentParser(
        description="Compare per-corruption metrics (Error/NLL/ECE) across logs.",
    )
    parser.add_argument("--logs", nargs="+", required=True, help="Paths to log files")
    parser.add_argument("--names", nargs="+", required=True, help="Legend names, one per log")
    parser.add_argument(
        "--metric",
        required=True,
        choices=["Error", "NLL", "ECE"],
        help="Which metric to plot (Error, NLL, or ECE)",
    )
    parser.add_argument(
        "--outdir",
        default="/users/doloriel/work/Repo/M2A/data_cifar/plots",
        help="Output directory",
    )
    parser.add_argument(
        "--outfile",
        default="ablations_cifar10c.png",
        help="Output filename",
    )
    args = parser.parse_args()

    if len(args.logs) != len(args.names):
        raise ValueError("--logs and --names must have the same length")

    metric_token = args.metric.strip().upper()

    series_corrs: List[List[str]] = []
    series_y_raw: List[List[float]] = []

    for log in args.logs:
        corrs, vals = parse_metric_series(log, metric_token)
        series_corrs.append(corrs)
        series_y_raw.append(vals)

    def base_corr_name(c: str) -> str:
        _re = re
        base = _re.sub(r"\d+$", "", c).strip("_").lower()
        return base

    global_order: List[str] = []
    for clist in series_corrs:
        for c in clist:
            b = base_corr_name(c)
            if b not in global_order:
                global_order.append(b)

    ABBR = {
        "gaussian_noise": "GN",
        "shot_noise": "SN",
        "impulse_noise": "IN",
        "defocus_blur": "DB",
        "glass_blur": "GB",
        "motion_blur": "MB",
        "zoom_blur": "ZB",
        "snow": "S",
        "frost": "Fr",
        "fog": "F",
        "brightness": "B",
        "contrast": "C",
        "elastic_transform": "ET",
        "pixelate": "P",
        "jpeg_compression": "JC",
    }

    aligned_series_y: List[List[float]] = []
    for corrs, vals in zip(series_corrs, series_y_raw):
        mapping = {base_corr_name(c): v for c, v in zip(corrs, vals)}
        aligned = [mapping.get(b, np.nan) for b in global_order]
        aligned_series_y.append(aligned)

    C = len(global_order)
    xs_common = list(range(1, C + 1))

    plt.figure(figsize=(10, 5))

    for ys, name in zip(aligned_series_y, args.names):
        plt.plot(xs_common, ys, label=name, linewidth=5)

    base_fs = float(plt.rcParams.get("font.size", 7.0))
    tick_fs = base_fs * 1.5
    legend_fs = base_fs * 1.7

    ax = plt.gca()

    tick_labels = [ABBR.get(b, b.upper()) for b in global_order]
    ax.set_xticks(xs_common)
    ax.set_xticklabels(tick_labels)

    ax.tick_params(axis="x", which="major", labelsize=legend_fs)
    ax.tick_params(axis="y", which="major", labelsize=tick_fs)

    yticks = ax.get_yticks()
    if len(yticks) > 5:
        idxs = np.linspace(0, len(yticks) - 1, 5, dtype=int)
        yticks_sel = yticks[idxs]
        ax.set_yticks(yticks_sel)

    plt.legend(prop={"size": legend_fs, "weight": "bold"})

    if C > 0:
        plt.xlim(0.5, C + 0.5)

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, args.outfile)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

    base = os.path.splitext(os.path.basename(args.outfile))[0]
    for name, ys in zip(args.names, aligned_series_y):
        arr = np.array(ys, dtype=float)
        avg = float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else float("nan")
        name_tag = name.lower().replace(" ", "")
        print(f"{base}_{name_tag}_avg: {avg:.2f}")

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
