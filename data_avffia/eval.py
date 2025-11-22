from glob import glob
import numpy as np


def _read_metric_from_file(filename, metric_key, is_percent=False):
    lines = open(filename, "r").readlines()
    vals = []
    for l in lines:
        if f"{metric_key} [" in l:
            tail = l.strip().split(":")[-1].strip()
            if is_percent and tail.endswith("%"):
                tail = tail[:-1]
            try:
                vals.append(float(tail))
            except Exception:
                pass
    assert len(vals) == 15, f"expected 15 entries for {metric_key} in {filename}, got {len(vals)}"
    return np.mean(np.array(vals))


def _read_metric_from_files(files, metric_key, is_percent=False):
    res = []
    for f in files:
        res.append(_read_metric_from_file(f, metric_key, is_percent))
    mean = float(np.mean(np.array(res))) if len(res) else float("nan")
    std = float(np.std(np.array(res))) if len(res) else float("nan")
    print(f"{metric_key}: read {len(files)} files.")
    print(res)
    return mean, std


def summarize_group(pattern, group_name):
    files = glob(pattern)
    if not files:
        print(f"no files matched for {group_name} ({pattern})")
        return
    print(f"read {group_name} files:")
    err = _read_metric_from_files(files, "Error %", is_percent=True)
    nll = _read_metric_from_files(files, "NLL")
    ece = _read_metric_from_files(files, "ECE")
    ms  = _read_metric_from_files(files, "Max Softmax")
    ent = _read_metric_from_files(files, "Entropy")
    print(f"{group_name} summary -> Error%(mean,std): {err}, NLL(mean,std): {nll}, ECE(mean,std): {ece}, MaxSoft(mean,std): {ms}, Entropy(mean,std): {ent}")


summarize_group("source_*.txt", "source")
summarize_group("norm_*.txt", "adabn")
summarize_group("tent[0-9]_*.txt", "tent")
summarize_group("cotta[0-9]_*.txt", "cotta")
