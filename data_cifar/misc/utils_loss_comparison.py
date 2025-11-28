#!/usr/bin/env python3
import argparse
import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_loss_series(log_path: str, loss_name: str) -> Tuple[List[str], List[float]]:
    """Parse a log file and extract per-corruption last-batch loss values for the given loss.

    Returns a tuple (corruptions_in_order, values_in_order).
    """
    # Accept common variants of the loss name; normalize to upper-case token
    token = loss_name.strip().upper()
    assert token in {"MCL", "ERL", "EML"}, "--loss must be one of: MCL, ERL, EML"

    # Example line:
    # [..] MCL (last batch) [gaussian_noise5]: 1.240985
    # [..] ERL (last batch) [gaussian_noise5]: 0.000000
    # [..] EML (last batch) [gaussian_noise5]: 0.221503
    pattern = re.compile(
        r"\]:\s*(?P<metric>MCL|ERL|EML)\s*\(last batch\)\s*\[(?P<corr>[^\]]+)\]:\s*(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    corrs: List[str] = []
    vals: List[float] = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            metric = m.group("metric").upper()
            if metric != token:
                continue
            corr = m.group("corr").strip()
            val_str = m.group("val").strip()
            try:
                val = float(val_str)
            except Exception:
                # Best effort fallback to number extraction
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val_str)
                if not nums:
                    continue
                val = float(nums[0])
            # Keep first occurrence per corruption in the log's natural order
            if corr not in corrs:
                corrs.append(corr)
                vals.append(val)
    return corrs, vals


def main():
    parser = argparse.ArgumentParser(description="Compare last-batch loss values across logs with sample-index x-axis")
    parser.add_argument('--logs', nargs='+', required=True, help='Paths to log files')
    parser.add_argument('--names', nargs='+', required=True, help='Legend names, one per log')
    parser.add_argument('--loss', required=True, choices=['MCL', 'ERL', 'EML'], help='Which loss metric to plot')
    parser.add_argument('--samples_per_corruption', type=int, required=True, help='Number of samples per corruption in the dataset')
    parser.add_argument('--outdir', default='/users/doloriel/work/Repo/M2A/data_cifar/plots/M2A/Misc/cifar10', help='Output directory')
    parser.add_argument('--outfile', default='loss_cifar10c.png', help='Output filename')
    parser.add_argument('--moving_avg', type=int, default=None, help='Window size for moving average overlay (optional)')
    args = parser.parse_args()

    if len(args.logs) != len(args.names):
        raise ValueError('--logs and --names must have the same length')

    loss_token = args.loss.upper()

    series_corrs: List[List[str]] = []  # corruption names as found in logs (with severity)
    series_y_raw: List[List[float]] = []

    for log in args.logs:
        corrs, vals = parse_loss_series(log, loss_token)
        series_corrs.append(corrs)
        series_y_raw.append(vals)

    # Build a stable global order of corruptions across all logs
    def base_corr_name(c: str) -> str:
        import re as _re
        base = _re.sub(r"\d+$", "", c).strip("_").lower()
        return base

    global_order: List[str] = []  # base names without severity
    for clist in series_corrs:
        for c in clist:
            b = base_corr_name(c)
            if b not in global_order:
                global_order.append(b)

    # Map base corruption name to abbreviation as specified
    ABBR = {
        'gaussian_noise': 'GN',
        'shot_noise': 'SN',
        'impulse_noise': 'IN',
        'defocus_blur': 'DB',
        'glass_blur': 'GB',
        'motion_blur': 'MB',
        'zoom_blur': 'ZB',
        'snow': 'S',
        'frost': 'Fr',
        'fog': 'F',
        'brightness': 'B',
        'contrast': 'C',
        'elastic_transform': 'ET',
        'pixelate': 'P',
        'jpeg_compression': 'JC',
    }

    # Align each series' y to the global order
    aligned_series_y: List[List[float]] = []
    for corrs, vals in zip(series_corrs, series_y_raw):
        mapping = {base_corr_name(c): v for c, v in zip(corrs, vals)}
        aligned = [mapping.get(b, np.nan) for b in global_order]
        aligned_series_y.append(aligned)

    # Common x positions are 1..C for C corruptions
    C = len(global_order)
    xs_common = list(range(1, C + 1))

    # Plot
    plt.figure(figsize=(10, 5))
    for ys, name in zip(aligned_series_y, args.names):
        line, = plt.plot(xs_common, ys, label=name, linewidth=3)
        if args.moving_avg is not None and isinstance(args.moving_avg, int) and args.moving_avg > 1 and len(ys) > 0:
            w = args.moving_avg
            kernel = np.ones(w, dtype=float)
            y_arr = np.array(ys, dtype=float)
            # NaN-aware moving average
            valid = (~np.isnan(y_arr)).astype(float)
            y_filled = np.nan_to_num(y_arr, nan=0.0)
            num = np.convolve(y_filled, kernel, mode='same')
            den = np.convolve(valid, kernel, mode='same')
            with np.errstate(invalid='ignore', divide='ignore'):
                ys_ma = num / den
            plt.plot(xs_common, ys_ma, color=line.get_color(), alpha=0.4, linewidth=3, label=f"{name} (MA)")

    # Axis labels
    base_fs = float(plt.rcParams.get('font.size', 7.0))
    label_fs = base_fs * 1.7
    tick_fs = base_fs * 1.3
    plt.xlabel('Corruptions', fontdict={'size': label_fs, 'weight': 'bold'})
    plt.ylabel(loss_token, fontdict={'size': label_fs, 'weight': 'bold'})
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    leg = plt.legend(prop={'size': label_fs, 'weight': 'bold'})
    # Dynamic title based on selected loss, styled like legend
    if loss_token == 'MCL':
        title_str = 'Mask Consistency Loss (MCL)'
    elif loss_token == 'EML':
        title_str = 'Entropy Minimization Loss (EML)'
    else:  # 'ERL'
        title_str = 'Entropy Ranking Loss (ERL)'
    plt.title(title_str, fontdict={'size': label_fs, 'weight': 'bold'})
    # X-axis ticks with corruption abbreviations
    tick_labels = [ABBR.get(b, b.upper()) for b in global_order]
    plt.xticks(xs_common, tick_labels)
    if C > 0:
        plt.xlim(0.5, C + 0.5)

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, args.outfile)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    base = os.path.splitext(os.path.basename(args.outfile))[0]
    for name, ys in zip(args.names, aligned_series_y):
        arr = np.array(ys, dtype=float)
        avg = float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else float('nan')
        name_tag = name.lower().replace(' ', '')
        print(f"{base}_{name_tag}_avg: {avg:.2f}")
    print(f"Saved plot to: {out_path}")


if __name__ == '__main__':
    main()
