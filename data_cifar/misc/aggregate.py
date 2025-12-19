#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict

PATTERNS = {
    "Error": re.compile(r"Error % \[([^\]]+)\]:\s*([\d.]+)%"),
    "NLL": re.compile(r"NLL \[([^\]]+)\]:\s*([\d.]+)"),
    "ECE": re.compile(r"ECE \[([^\]]+)\]:\s*([\d.]+)"),
    "Adaptation Time": re.compile(r"Adaptation Time .*?\[([^\]]+)\]:\s*([\d.]+)s"),
    "Adaptation MACs": re.compile(r"Adaptation MACs .*?\[([^\]]+)\]:\s*(.+)$"),
    "Domain Shift Robustness": re.compile(r"Domain Shift Robustness .*?\[([^\]]+)\]:\s*([\d.]+)"),
    "Catastrophic Forgetting Rate": re.compile(r"Catastrophic Forgetting Rate .*? after \[([^\]]+)\]:\s*([\d.]+)")
}

SCIENTIFIC_MACS_RE = re.compile(r"^\s*([\d.]+)\s*[xX]\s*10\^([+\-]?\d+)\s*$")


def parse_macs_value(val_str: str) -> float:
    s = val_str.strip()
    m = SCIENTIFIC_MACS_RE.match(s)
    if m:
        base = float(m.group(1))
        exp = int(m.group(2))
        return base * (10 ** exp)
    m2 = re.search(r"([+-]?[0-9]*\\.?[0-9]+)", s)
    if m2:
        return float(m2.group(1))
    raise ValueError(f"Unparsable MACs value: {val_str}")


def parse_log_file_detailed(path: Path) -> Dict[str, List[Tuple[str, float]]]:
    out: Dict[str, List[Tuple[str, float]]] = {
        "Error": [],
        "NLL": [],
        "ECE": [],
        "Adaptation Time": [],
        "Adaptation MACs": [],
        "Domain Shift Robustness": [],
        "Catastrophic Forgetting Rate": [],
    }
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return out
    for line in text.splitlines():
        for key, pattern in PATTERNS.items():
            m = pattern.search(line)
            if not m:
                continue
            try:
                corr = m.group(1)
                if key == "Adaptation MACs":
                    val = parse_macs_value(m.group(2))
                else:
                    val = float(m.group(2))
                out[key].append((corr, val))
            except Exception:
                pass
    return out


def aggregate_across_logs(paths: List[Path]) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, List[str]], Dict[str, Dict[str, List[float]]]]:
    metrics_order = [
        "Error",
        "NLL",
        "ECE",
        "Adaptation Time",
        "Adaptation MACs",
        "Domain Shift Robustness",
        "Catastrophic Forgetting Rate",
    ]
    agg: Dict[str, DefaultDict[str, List[float]]] = {
        m: defaultdict(list) for m in metrics_order
    }
    order_by_metric: Dict[str, List[str]] = {m: [] for m in metrics_order}
    # Per-log values across corruptions for each metric: per_log_vals[metric][log_path] = [vals across corruptions]
    per_log_vals: Dict[str, Dict[str, List[float]]] = {m: {} for m in metrics_order}
    for p in paths:
        if not p.exists():
            print(f"Warning: {p} does not exist, skipping.", file=sys.stderr)
            continue
        parsed = parse_log_file_detailed(p)
        for metric, pairs in parsed.items():
            for corr, val in pairs:
                if corr not in order_by_metric[metric]:
                    order_by_metric[metric].append(corr)
                agg[metric][corr].append(val)
                per_log_vals[metric].setdefault(str(p), []).append(val)
    return {m: dict(cmap) for m, cmap in agg.items()}, order_by_metric, per_log_vals


def convert_units(values_by_corr: Dict[str, List[float]], metric: str) -> Dict[str, List[float]]:
    if metric in ("Domain Shift Robustness", "Catastrophic Forgetting Rate"):
        return {k: [v * 100.0 for v in vs] for k, vs in values_by_corr.items()}
    return values_by_corr


def compute_stats(vals: List[float]) -> Tuple[int, float, float]:
    if not vals:
        return 0, float("nan"), float("nan")
    c = len(vals)
    m = float(mean(vals))
    s = float(stdev(vals)) if c > 1 else float("nan")
    return c, m, s


def fmt_mean(v: float, decimals: int = 3) -> str:
    if v != v:
        return "nan"
    s = f"{v:.{decimals}f}"
    s = s.rstrip('0').rstrip('.') if '.' in s else s
    return s


def fmt_std(v: float, decimals: int = 2) -> str:
    if v != v:
        return "nan"
    return f"{v:.{decimals}f}"


def print_aggregate(agg: Dict[str, Dict[str, List[float]]], order_by_metric: Dict[str, List[str]], per_log_vals: Dict[str, Dict[str, List[float]]], selected_metric=None, avg_type: str = "per_log"):
    order = [
        "Error",
        "NLL",
        "ECE",
        "Adaptation Time",
        "Adaptation MACs",
        "Domain Shift Robustness",
        "Catastrophic Forgetting Rate",
    ]
    if selected_metric:
        key = selected_metric.strip().lower()
        mapping = {m.lower(): m for m in order}
        if key in mapping:
            order = [mapping[key]]
    for metric in order:
        values_by_corr = convert_units(agg.get(metric, {}), metric)
        print(f"\nMetric: {metric}")
        means_for_overall: List[float] = []
        order_list = order_by_metric.get(metric, [])
        # Ensure we include any keys that might not be in the recorded order (unlikely)
        corrs_iter = order_list + [c for c in values_by_corr.keys() if c not in order_list]
        for corr in corrs_iter:
            c, m, s = compute_stats(values_by_corr[corr])
            if metric in ("Domain Shift Robustness", "Catastrophic Forgetting Rate"):
                m_str = fmt_mean(m) + "%"
                s_str = fmt_std(s) + "%"
            elif metric == "Adaptation MACs":
                m_str = f"{m:.6g}" if m == m else "nan"
                s_str = f"{s:.6g}" if s == s else "nan"
            else:
                m_str = fmt_mean(m)
                s_str = fmt_std(s)
            print(f"  {corr:>28}: count={c:2d}, mean= {m_str} ({s_str})")
            if m == m:
                means_for_overall.append(m)
        if means_for_overall:
            overall_mean = float(mean(means_for_overall))
            overall_std = float(stdev(means_for_overall)) if len(means_for_overall) > 1 else float("nan")
        else:
            overall_mean = float("nan")
            overall_std = float("nan")
        if metric in ("Domain Shift Robustness", "Catastrophic Forgetting Rate"):
            om = fmt_mean(overall_mean) + "%"
            os = fmt_std(overall_std) + "%"
        elif metric == "Adaptation MACs":
            om = f"{overall_mean:.6g}" if overall_mean == overall_mean else "nan"
            os = f"{overall_std:.6g}" if overall_std == overall_std else "nan"
        else:
            om = fmt_mean(overall_mean)
            os = fmt_std(overall_std)
        print(f"  {'Overall (across corruptions)':>28}: mean= {om} ({os})")

        if avg_type == "per_log":
            # Overall across logs: compute per-log mean/std across corruptions, then average these across logs
            log_series = per_log_vals.get(metric, {})
            per_log_means: List[float] = []
            per_log_stds: List[float] = []
            for lp, vals in log_series.items():
                conv_vals = convert_units({"_": vals}, metric)["_"]
                if not conv_vals:
                    continue
                per_log_means.append(float(mean(conv_vals)))
                s_val = float(stdev(conv_vals)) if len(conv_vals) > 1 else float("nan")
                per_log_stds.append(s_val)
            if per_log_means:
                overall_mean_logs = float(mean(per_log_means))
                overall_std_logs = float(stdev(per_log_means)) if len(per_log_means) > 1 else float("nan")
            else:
                overall_mean_logs = float("nan")
                overall_std_logs = float("nan")

            if metric in ("Domain Shift Robustness", "Catastrophic Forgetting Rate"):
                om_logs = fmt_mean(overall_mean_logs) + "%"
                os_logs = fmt_std(overall_std_logs) + "%"
            elif metric == "Adaptation MACs":
                om_logs = f"{overall_mean_logs:.6g}" if overall_mean_logs == overall_mean_logs else "nan"
                os_logs = f"{overall_std_logs:.6g}" if overall_std_logs == overall_std_logs else "nan"
            else:
                om_logs = fmt_mean(overall_mean_logs)
                os_logs = fmt_std(overall_std_logs)
            print(f"  {'Overall (across logs)':>28}: mean= {om_logs} ({os_logs})")
        elif avg_type == "per_count":
            # Per-count: for each count index, average across corruptions
            max_count = 0
            for vs in values_by_corr.values():
                if len(vs) > max_count:
                    max_count = len(vs)
            if max_count == 0:
                continue
            per_count_means: List[float] = []
            print(f"  {'Per-count (across corruptions)':>28}:")
            for idx in range(max_count):
                vals_idx: List[float] = []
                for corr, vs in values_by_corr.items():
                    if idx < len(vs):
                        vals_idx.append(vs[idx])
                if not vals_idx:
                    continue
                c_idx = len(vals_idx)
                m_idx = float(mean(vals_idx))
                s_idx = float(stdev(vals_idx)) if len(vals_idx) > 1 else float("nan")
                if metric in ("Domain Shift Robustness", "Catastrophic Forgetting Rate"):
                    m_str_idx = fmt_mean(m_idx) + "%"
                    s_str_idx = fmt_std(s_idx) + "%"
                elif metric == "Adaptation MACs":
                    m_str_idx = f"{m_idx:.6g}" if m_idx == m_idx else "nan"
                    s_str_idx = f"{s_idx:.6g}" if s_idx == s_idx else "nan"
                else:
                    m_str_idx = fmt_mean(m_idx)
                    s_str_idx = fmt_std(s_idx)
                print(f"  Count {idx+1:2d} (across corr.): count={c_idx:2d}, mean= {m_str_idx} ({s_str_idx})")
                if m_idx == m_idx:
                    per_count_means.append(m_idx)
            if per_count_means:
                overall_mean_counts = float(mean(per_count_means))
                overall_std_counts = float(stdev(per_count_means)) if len(per_count_means) > 1 else float("nan")
            else:
                overall_mean_counts = float("nan")
                overall_std_counts = float("nan")

            if metric in ("Domain Shift Robustness", "Catastrophic Forgetting Rate"):
                om_cnt = fmt_mean(overall_mean_counts) + "%"
                os_cnt = fmt_std(overall_std_counts) + "%"
            elif metric == "Adaptation MACs":
                om_cnt = f"{overall_mean_counts:.6g}" if overall_mean_counts == overall_mean_counts else "nan"
                os_cnt = f"{overall_std_counts:.6g}" if overall_std_counts == overall_std_counts else "nan"
            else:
                om_cnt = fmt_mean(overall_mean_counts)
                os_cnt = fmt_std(overall_std_counts)
            print(f"  {'Overall (across counts)':>28}: mean= {om_cnt} ({os_cnt})")


def metric_name_from_arg(name: str) -> str:
    """Map a CLI metric name (case-insensitive) to one of the known PATTERNS keys."""
    if not name:
        return "Error"
    key = name.strip().lower()
    mapping = {m.lower(): m for m in PATTERNS.keys()}
    return mapping.get(key, "Error")


def print_rand_domain_summary(log_path: Path, metric_arg: str) -> None:
    """Special handling for rand_domain CIFAR-10C CTTA logs.

    Assumes a single log where the 15 CIFAR-10-C corruptions are presented
    in random order for each permutation, with model resets between
    permutations. We ignore resets explicitly and simply group the chosen
    metric values into contiguous blocks of 15 and average within each
    block. The resulting per-block means are printed as a single
    LaTeX-friendly row: v1 & v2 & ...
    """

    metric = metric_name_from_arg(metric_arg)
    parsed = parse_log_file_detailed(log_path)
    pairs = parsed.get(metric, [])
    if not pairs:
        print(f"No metric '{metric}' found in log {log_path}.")
        return

    # Preserve chronological order from the log
    vals = [v for _, v in pairs]

    group_size = 15  # CIFAR-10-C has 15 corruption types
    num_groups = len(vals) // group_size
    if num_groups == 0:
        print(f"Not enough entries for metric '{metric}' in {log_path} to form a 15-corruption block.")
        return

    group_means: List[float] = []
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        chunk = vals[start:end]
        if not chunk:
            continue
        group_means.append(float(mean(chunk)))

    # Format numbers using existing helpers. For Error we keep the value
    # as-is (already in % units from the logs) and just format the mean.
    formatted: List[str] = []
    for m in group_means:
        if metric == "Adaptation MACs":
            formatted.append(f"{m:.6g}")
        elif metric in ("Domain Shift Robustness", "Catastrophic Forgetting Rate"):
            # These are stored as fractions in normal aggregation; multiply by 100 here.
            formatted.append(fmt_mean(m * 100.0))
        else:
            formatted.append(fmt_mean(m))

    print(f"Rand-domain summary for metric '{metric}' from {log_path}:")
    print(" & ".join(formatted))


def main():
    parser = argparse.ArgumentParser(description="Aggregate corruption-wise metrics across multiple logs.")
    parser.add_argument("logs", nargs="*", type=str, help="Paths to log files to parse")
    parser.add_argument("--metric", type=str, default=None, help="Single metric to print (e.g., Error, NLL, ECE)")
    parser.add_argument("--avg_type", type=str, default="per_log", choices=["per_log", "per_count"],
                        help="How to average overall metrics: per_log (default) or per_count (average across corruptions per count index)")
    parser.add_argument("--rand_domain", action="store_true",
                        help="Handle a single CIFAR-10-C CTTA log with randomly permuted corruption domains; "
                             "group the chosen metric into 15-corruption blocks and print the mean per block as a LaTeX row.")
    args = parser.parse_args()

    default_logs = [
        "/users/doloriel/work/Repo/M2A/logs/output_13685926.txt",
        "/users/doloriel/work/Repo/M2A/logs/output_13685937.txt",
        "/users/doloriel/work/Repo/M2A/logs/output_13685946.txt",
    ]
    log_paths = [Path(p) for p in (args.logs if args.logs else default_logs)]

    if args.rand_domain:
        if len(log_paths) != 1:
            print("--rand_domain expects exactly one log file (got {}), please specify a single log.".format(len(log_paths)),
                  file=sys.stderr)
            sys.exit(1)
        metric_for_rand = args.metric if args.metric is not None else "Error"
        print_rand_domain_summary(log_paths[0], metric_for_rand)
        return

    agg, order_by_metric, per_log_vals = aggregate_across_logs(log_paths)
    print_aggregate(agg, order_by_metric, per_log_vals, selected_metric=args.metric, avg_type=args.avg_type)


if __name__ == "__main__":
    main()
