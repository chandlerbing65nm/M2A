import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def clean_label(s: str) -> str:
    s = s.replace('_', ' ').lower()
    tokens = [t for t in s.split() if t not in {'blur', 'noise', 'compression', 'transform'}]
    return ' '.join(tokens)


BASE_FS = mpl.rcParams.get('font.size', 14.0)
SCALE = 1.8
mpl.rcParams.update({
    'font.size': BASE_FS * SCALE,
    'axes.titlesize': BASE_FS * SCALE,
    'axes.labelsize': BASE_FS * SCALE,
    'xtick.labelsize': BASE_FS * SCALE,
    'ytick.labelsize': BASE_FS * SCALE,
    'legend.fontsize': BASE_FS * SCALE,
})


def plot_pair(xlabels, bars_mean, bars_std, line_mean, line_std, legend_bars, legend_line, outfile, mode="bar"):
    x = np.arange(len(xlabels))
    fig, ax = plt.subplots(figsize=(14, 5))

    if mode == "bar":
        width = 0.4
        ax.bar(
            x - width / 2,
            bars_mean,
            yerr=bars_std,
            width=width,
            color="#4C78A8",
            alpha=0.85,
            error_kw={"elinewidth": 1, "capsize": 3},
            label=legend_bars,
            zorder=1,
        )
        ax.bar(
            x + width / 2,
            line_mean,
            yerr=line_std,
            width=width,
            color="#F58518",
            alpha=0.85,
            error_kw={"elinewidth": 1, "capsize": 3},
            label=legend_line,
            zorder=2,
        )
    elif mode == "line":
        ax.errorbar(
            x,
            bars_mean,
            yerr=bars_std,
            fmt="o-",
            color="#4C78A8",
            linewidth=2,
            capsize=3,
            label=legend_bars,
            zorder=2,
        )
        ax.errorbar(
            x,
            line_mean,
            yerr=line_std,
            fmt="s-",
            color="#F58518",
            linewidth=2,
            capsize=3,
            label=legend_line,
            zorder=2,
        )
    elif mode == "violin" or mode == "box":
        rng = np.random.default_rng(0)
        n = 500
        left_data = [np.clip(rng.normal(loc=m, scale=max(s, 1e-6), size=n), 0, None) for m, s in zip(bars_mean, bars_std)]
        right_data = [np.clip(rng.normal(loc=m, scale=max(s, 1e-6), size=n), 0, None) for m, s in zip(line_mean, line_std)]
        width = 0.35
        left_pos = x - width
        right_pos = x + width
        if mode == "violin":
            v1 = ax.violinplot(left_data, positions=left_pos, widths=0.35, showmeans=True, showmedians=False, showextrema=False)
            v2 = ax.violinplot(right_data, positions=right_pos, widths=0.35, showmeans=True, showmedians=False, showextrema=False)
            for b in v1["bodies"]:
                b.set_facecolor("#4C78A8")
                b.set_edgecolor("black")
                b.set_alpha(0.6)
            for b in v2["bodies"]:
                b.set_facecolor("#F58518")
                b.set_edgecolor("black")
                b.set_alpha(0.6)
            handles = [Patch(facecolor="#4C78A8", alpha=0.6, edgecolor="black", label=legend_bars), Patch(facecolor="#F58518", alpha=0.6, edgecolor="black", label=legend_line)]
            ax.legend(handles=handles)
        else:
            b1 = ax.boxplot(left_data, positions=left_pos, widths=0.35, patch_artist=True, manage_ticks=False)
            b2 = ax.boxplot(right_data, positions=right_pos, widths=0.35, patch_artist=True, manage_ticks=False)
            for elem in ["boxes", "whiskers", "caps", "medians"]:
                for a in b1[elem]:
                    a.set(color="#4C78A8")
                for a in b2[elem]:
                    a.set(color="#F58518")
            for patch in b1["boxes"]:
                patch.set(facecolor="#4C78A8", alpha=0.35)
            for patch in b2["boxes"]:
                patch.set(facecolor="#F58518", alpha=0.35)
            handles = [Patch(facecolor="#4C78A8", edgecolor="#4C78A8", alpha=0.35, label=legend_bars), Patch(facecolor="#F58518", edgecolor="#F58518", alpha=0.35, label=legend_line)]
            ax.legend(handles=handles)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_ylabel("")
    ax.set_title("Classification Error Rate %")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    if mode in {"bar", "line"}:
        ax.legend()

    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    MODE = "bar"
    order = [
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
    xlabels = [clean_label(o) for o in order]

    freq_means = {
        "gaussian_noise": 17.08,
        "shot_noise": 11.183,
        "impulse_noise": 11.84,
        "defocus_blur": 10.27,
        "glass_blur": 19.06,
        "motion_blur": 10.22,
        "zoom_blur": 7.01,
        "snow": 6.743,
        "frost": 5.483,
        "fog": 6.343,
        "brightness": 4.59,
        "contrast": 5.27,
        "elastic_transform": 14.03,
        "pixelate": 51.76,
        "jpeg_compression": 32.837,
    }
    freq_stds = {
        "gaussian_noise": 0.21,
        "shot_noise": 0.27,
        "impulse_noise": 0.41,
        "defocus_blur": 0.19,
        "glass_blur": 0.31,
        "motion_blur": 0.08,
        "zoom_blur": 0.29,
        "snow": 0.04,
        "frost": 0.07,
        "fog": 0.20,
        "brightness": 0.19,
        "contrast": 0.14,
        "elastic_transform": 0.40,
        "pixelate": 35.65,
        "jpeg_compression": 16.96,
    }

    patch_means = {
        "gaussian_noise": 17.113,
        "shot_noise": 11.363,
        "impulse_noise": 8.327,
        "defocus_blur": 7.357,
        "glass_blur": 14.89,
        "motion_blur": 7.627,
        "zoom_blur": 5.077,
        "snow": 5.817,
        "frost": 4.577,
        "fog": 5.863,
        "brightness": 3.257,
        "contrast": 4.897,
        "elastic_transform": 9.637,
        "pixelate": 7.187,
        "jpeg_compression": 10.77,
    }
    patch_stds = {
        "gaussian_noise": 0.33,
        "shot_noise": 0.19,
        "impulse_noise": 0.31,
        "defocus_blur": 0.14,
        "glass_blur": 0.01,
        "motion_blur": 0.09,
        "zoom_blur": 0.06,
        "snow": 0.29,
        "frost": 0.13,
        "fog": 0.15,
        "brightness": 0.05,
        "contrast": 0.08,
        "elastic_transform": 0.05,
        "pixelate": 0.21,
        "jpeg_compression": 0.44,
    }

    freq_rc_means = {
        "gaussian_noise": 15.473,
        "shot_noise": 10.28,
        "impulse_noise": 9.74,
        "defocus_blur": 9.607,
        "glass_blur": 19.61,
        "motion_blur": 10.14,
        "zoom_blur": 6.88,
        "snow": 6.903,
        "frost": 5.467,
        "fog": 6.153,
        "brightness": 4.05,
        "contrast": 5.027,
        "elastic_transform": 15.217,
        "pixelate": 73.457,
        "jpeg_compression": 45.287,
    }
    freq_rc_stds = {
        "gaussian_noise": 0.16,
        "shot_noise": 0.15,
        "impulse_noise": 0.25,
        "defocus_blur": 0.25,
        "glass_blur": 0.03,
        "motion_blur": 0.13,
        "zoom_blur": 0.07,
        "snow": 0.06,
        "frost": 0.14,
        "fog": 0.12,
        "brightness": 0.14,
        "contrast": 0.03,
        "elastic_transform": 0.98,
        "pixelate": 3.88,
        "jpeg_compression": 24.14,
    }

    patch_rc_means = {
        "gaussian_noise": 15.57,
        "shot_noise": 9.74,
        "impulse_noise": 7.303,
        "defocus_blur": 6.783,
        "glass_blur": 15.313,
        "motion_blur": 7.71,
        "zoom_blur": 4.893,
        "snow": 5.683,
        "frost": 4.537,
        "fog": 5.857,
        "brightness": 3.137,
        "contrast": 4.92,
        "elastic_transform": 9.623,
        "pixelate": 6.593,
        "jpeg_compression": 10.317,
    }
    patch_rc_stds = {
        "gaussian_noise": 0.05,
        "shot_noise": 0.28,
        "impulse_noise": 0.11,
        "defocus_blur": 0.01,
        "glass_blur": 0.90,
        "motion_blur": 0.14,
        "zoom_blur": 0.21,
        "snow": 0.10,
        "frost": 0.15,
        "fog": 0.13,
        "brightness": 0.07,
        "contrast": 0.05,
        "elastic_transform": 0.08,
        "pixelate": 0.12,
        "jpeg_compression": 0.51,
    }

    bars_means_1 = [patch_means[k] for k in order]
    bars_stds_1 = [patch_stds[k] for k in order]
    line_means_1 = [freq_means[k] for k in order]
    line_stds_1 = [freq_stds[k] for k in order]

    plot_pair(
        xlabels,
        bars_means_1,
        bars_stds_1,
        line_means_1,
        line_stds_1,
        legend_bars="Patch Mask",
        legend_line="Frequency Mask",
        outfile="cifar/plots/SPARE/Comp/error_rate_masks.png",
        mode=MODE,
    )

    bars_means_2 = [patch_rc_means[k] for k in order]
    bars_stds_2 = [patch_rc_stds[k] for k in order]
    line_means_2 = [freq_rc_means[k] for k in order]
    line_stds_2 = [freq_rc_stds[k] for k in order]

    plot_pair(
        xlabels,
        bars_means_2,
        bars_stds_2,
        line_means_2,
        line_stds_2,
        legend_bars="Patch Mask + Residual",
        legend_line="Frequency Mask + Residual",
        outfile="cifar/plots/SPARE/Comp/error_rate_masks_residual.png",
        mode=MODE,
    )

    plt.show()
