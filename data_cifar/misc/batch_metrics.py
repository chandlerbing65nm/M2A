#!/usr/bin/env python3
import argparse
import os
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

VALID_METRICS = {"Error", "NLL", "ECE"}
VALID_METHODS = {
    "Source",
    "Tent",
    "CoTTA",
    "Continual-MAE",
    "REM",
    "M2A (Ours)",
    "M2A (Spatial)",
    "M2A (Frequency)",
    "M2A (Spatial) + ERL",
    "M2A (Frequency) + ERL",
}


def parse_batch_metric_series(log_path: str, metric: str) -> List[float]:
    """Parse a log file and extract per-batch values for the given metric.

    Only lines containing "[BATCH_METRICS]" are considered. Expected pattern
    (simplified example):

        [..] [BATCH_METRICS] [gaussian_noise5] batch 0: Error %: 55.00%, NLL: 2.5063, ECE: 0.3612

    We extract the numeric value corresponding to the requested metric.
    """
    token = metric.strip()
    assert token in VALID_METRICS, f"--metric must be one of: {', '.join(sorted(VALID_METRICS))}"

    # Regex to capture corruption name, batch index, and metrics
    # Example matched groups on a line:
    #   corr  -> gaussian_noise5
    #   batch -> 0
    #   err   -> 55.00
    #   nll   -> 2.5063
    #   ece   -> 0.3612
    float_re = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    pattern = re.compile(
        r"\[BATCH_METRICS\]\s*"  # tag
        r"\[(?P<corr>[^\]]+)\]\s*"  # corruption/severity
        r"batch\s+(?P<batch>\d+):\s*"
        rf"Error\s*%:\s*(?P<err>{float_re})%?,\s*"  # value may have a trailing '%'
        rf"NLL:\s*(?P<nll>{float_re}),\s*"
        rf"ECE:\s*(?P<ece>{float_re})"
    )

    values: List[float] = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[BATCH_METRICS]" not in line:
                continue
            m = pattern.search(line)
            if not m:
                continue
            if token == "Error":
                val_str = m.group("err")
                # Convert percent to fraction if desired; here we stay in percent
                val = float(val_str)
            elif token == "NLL":
                val_str = m.group("nll")
                val = float(val_str)
            else:  # "ECE"
                val_str = m.group("ece")
                val = float(val_str)
            values.append(val)

    return values


def main():
    parser = argparse.ArgumentParser(description="Plot per-batch metrics across corruptions and methods")
    parser.add_argument("--logs", nargs="+", required=True, help="Paths to log files (.log or .txt)")
    parser.add_argument("--metric", required=True, choices=[m.lower() for m in VALID_METRICS],
                        help="Metric to plot: error, nll, or ece")
    parser.add_argument("--method", nargs="+", required=True,
                        help=(
                            "Method name per log (Source, Tent, CoTTA, Continual-MAE, REM, "
                            "M2A (Ours), M2A (Spatial), M2A (Frequency), "
                            "M2A (Spatial) + ERL, M2A (Frequency) + ERL)"
                        ))
    parser.add_argument("--out_dir", default="/users/doloriel/work/Repo/M2A/data_cifar/plots/batch_metrics",
                        help="Output directory for plots")
    parser.add_argument("--gap", type=int, default=1,
                        help="Window size for averaging; plot the mean metric over each block of 'gap' batches")
    args = parser.parse_args()

    if len(args.logs) != len(args.method):
        raise ValueError("--logs and --method must have the same length")

    # Normalize metric name: user passes lower case, we map to canonical
    metric_token = args.metric.strip().lower()
    metric_canonical = {"error": "Error", "nll": "NLL", "ece": "ECE"}[metric_token]

    # Validate methods (accept flexible input) and map to canonical legend names
    # We normalize by lowering case and stripping spaces/underscores/hyphens/parentheses.
    def _norm_method_name(name: str) -> str:
        return re.sub(r"[\s_\-()]+", "", name.strip().lower())

    canonical_map = {_norm_method_name(m): m for m in VALID_METHODS}

    methods: List[str] = []
    for m in args.method:
        key = _norm_method_name(m)
        if key not in canonical_map:
            raise ValueError(f"Invalid method '{m}'. Must be one of: {', '.join(sorted(VALID_METHODS))}")
        methods.append(canonical_map[key])

    # Parse series from each log
    series_y: List[List[float]] = []
    max_len = 0
    for log in args.logs:
        ys = parse_batch_metric_series(log, metric_canonical)
        series_y.append(ys)
        if len(ys) > max_len:
            max_len = len(ys)

    if max_len == 0:
        raise RuntimeError("No batch metrics found in the provided logs for the requested metric")

    # Subsample using gap (same for all series) by averaging over blocks of size 'gap'
    gap = max(1, int(args.gap))

    subsampled_series_y: List[List[float]] = []
    for ys in series_y:
        y_arr = np.array(ys, dtype=float)
        if gap > 1:
            block_means: List[float] = []
            for start in range(0, len(y_arr), gap):
                block = y_arr[start:start + gap]
                if block.size == 0:
                    continue
                block_means.append(float(block.mean()))
            ys_sub = block_means
        else:
            ys_sub = y_arr.tolist()
        subsampled_series_y.append(ys_sub)

    # X-axis: global batch index across corruptions; we simply treat them as a flat sequence
    # Example: 15 corruptions * 500 batches -> indices 0..7499 (len = 7500)
    # With gap > 1, each x position corresponds to the first batch index in a window of size 'gap'.
    full_indices = list(range(max_len))
    xs = full_indices[::gap]

    # Plot with style similar to utils_loss_comparison.py
    plt.figure(figsize=(10, 5))

    for ys, name in zip(subsampled_series_y, methods):
        # Align y length with xs (truncate if shorter)
        y_arr = np.array(ys, dtype=float)
        if len(y_arr) > len(xs):
            y_arr = y_arr[: len(xs)]
        plt.plot(xs[: len(y_arr)], y_arr, label=name, linewidth=5)

    # Font sizes similar to utils_loss_comparison.py
    base_fs = float(plt.rcParams.get("font.size", 7.0))
    title_fs = base_fs * 2.5
    xlabel_fs = base_fs * 2.5
    tick_fs = base_fs * 1.5
    legend_fs = base_fs * 1.7

    # Axis labels and title as requested
    plt.xlabel("Batches", fontdict={"size": xlabel_fs, "weight": "bold"})
    # y-axis label blank
    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)

    # Legend labels from methods
    plt.legend(prop={"size": legend_fs, "weight": "bold"})

    # Plot title blank, but keep font scaling in case user wants to modify later
    plt.title("", fontdict={"size": title_fs, "weight": "bold"})

    if len(xs) > 0:
        plt.xlim(xs[0], xs[-1])

    os.makedirs(args.out_dir, exist_ok=True)

    outfile = f"{metric_token}.jpg"
    out_path = os.path.join(args.out_dir, outfile)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()


# python data_cifar/misc/batch_metrics.py   --metric error   --logs logs/cifar10c/output_15332385.txt logs/cifar10c/output_15332387.txt logs/cifar10c/output_15332388.txt logs/cifar10c/output_15332389.txt logs/cifar10c/output_15332390.txt logs/cifar10c/output_15332392.txt logs/cifar10c/output_15332393.txt  --method "source" "tent" "cotta" "continual-mae" "rem" "m2a (spatial)" "m2a (frequency)"  --gap 50

# python data_cifar/misc/batch_metrics.py --metric ece --logs logs/cifar10c/output_15332385.txt logs/cifar10c/output_15332389.txt logs/cifar10c/output_15332390.txt logs/cifar10c/output_15332392.txt --method "source" "continual-mae" "rem" "m2a (spatial)"  --gap 50