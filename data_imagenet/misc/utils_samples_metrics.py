import os
import argparse
import re
import math
import numpy as np
import matplotlib.pyplot as plt


def parse_avg_error(log_path: str) -> float:
    values = []
    pattern = re.compile(r"Error\s*%?\s*\[[^\]]+\]:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pattern.search(line)
                if not m:
                    continue
                try:
                    values.append(float(m.group(1)))
                except Exception:
                    continue
    except FileNotFoundError:
        return float("nan")
    if len(values) == 0:
        return float("nan")
    return float(np.nanmean(np.array(values, dtype=float)))


def main():
    parser = argparse.ArgumentParser(description="Bar chart of avg Error across corruptions vs samples per corruption")
    parser.add_argument("--logs", nargs="+", required=True, help="Paths to log files. Grouped per name in sample order")
    parser.add_argument("--names", nargs="+", required=True, help="Legend names (methods), each with a set of logs")
    parser.add_argument("--labels", nargs="+", default=["1k", "2k", "5k", "10k"], help="Sample sizes labels and grouping size")
    parser.add_argument("--outdir", type=str, default="cifar/plots/SPARE/Misc/cifar10")
    parser.add_argument("--outfile", type=str, default="error_rate_samples.png")
    parser.add_argument("--radar", action="store_true", help="Output a radar (polar) plot instead of bars")
    args = parser.parse_args()

    labels = list(args.labels)
    L = len(labels)
    N = len(args.names)
    expected = L * N
    if len(args.logs) != expected:
        raise ValueError(f"--logs count ({len(args.logs)}) must equal len(--names) * len(--labels) = {N} * {L} = {expected}")

    series = []
    for i in range(N):
        slice_logs = args.logs[i * L : (i + 1) * L]
        vals = [parse_avg_error(p) for p in slice_logs]
        series.append(np.array(vals, dtype=float))

    x = np.arange(L)
    n_methods = N
    bar_width = 0.8 / max(1, n_methods)

    prop = plt.rcParams.get("axes.prop_cycle", None)
    if prop is not None:
        base_colors = prop.by_key().get("color", [])
    else:
        base_colors = []
    if not base_colors:
        base_colors = ["#4C78A8", "#F58518", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    colors = [base_colors[i % len(base_colors)] for i in range(n_methods)]

    all_vals = np.concatenate(series) if series else np.array([0.0])
    finite_vals = all_vals[np.isfinite(all_vals)]
    if finite_vals.size == 0:
        ymin, ymax = 0.0, 1.0
    else:
        y_min_val = float(np.nanmin(finite_vals))
        y_max_val = float(np.nanmax(finite_vals))
        ymin = max(0.0, y_min_val - 0.1 * abs(y_min_val))
        ymax = y_max_val + 0.1 * abs(y_max_val)

    if args.radar:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 7))
        N_axes = len(labels)
        if N_axes > 0:
            angles = np.linspace(0, 2 * math.pi, N_axes, endpoint=False).tolist()
            angles += angles[:1]
            ax.set_theta_offset(math.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            base_fs = float(plt.rcParams.get('font.size', 10.0))
            big_fs = base_fs * 2.0
            ax.set_xticklabels(labels, fontsize=big_fs)
            ax.grid(True, alpha=0.3)
            try:
                for lab in ax.get_yticklabels():
                    lab.set_fontsize(base_fs)
            except Exception:
                pass
            ax.set_ylim(ymin, ymax)
            for i, (vals, name) in enumerate(zip(series, args.names)):
                v = np.array(vals, dtype=float)
                v = np.nan_to_num(v, nan=0.0)
                y_plot = v.tolist() + v[:1].tolist()
                ax.plot(angles, y_plot, linewidth=2, color=colors[i], label=name)
                ax.fill(angles, y_plot, alpha=0.10, color=colors[i])
        # No title per request; place legend below the radar plot
        fig.legend(loc='lower center', ncol=min(len(args.names), 4), prop={'size': big_fs})
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (vals, name) in enumerate(zip(series, args.names)):
            offset = (i - (n_methods - 1) / 2.0) * bar_width
            ax.bar(
                x + offset,
                vals,
                width=bar_width,
                color=colors[i],
                alpha=0.85,
                label=name,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Samples per Corruption")
        ax.set_ylabel("")
        ax.set_title("Classification Error Rate %")
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend()

    os.makedirs(args.outdir, exist_ok=True)
    outfile = os.path.join(args.outdir, args.outfile)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(outfile, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
