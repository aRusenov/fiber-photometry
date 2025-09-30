import os
import re
import sys
import h5py
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.stats import ttest_rel
import csv

# Add the parent for import
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from common.lib import (
    Activity,
    Channel,
    ChannelType,
    Processed5CData,
    compute_mean,
    log,
    pad,
)

events = [
    "Initiated trials",
    "Prematures",
    "Reward pickup",
    "Omissions",
    "Correct",
    "Incorrect",
]


def read_processed_data(infile: str) -> Processed5CData:
    activities: list[Activity] = []
    with h5py.File(infile, "r") as f:
        # f.visit(printname)
        events = f["Event"]
        for event in events.keys():
            activity = Activity(event=event, channels=[])
            for channel in f["Event"][event].keys():
                activity.channels.append(
                    Channel(
                        name=ChannelType[channel],
                        bins_dff=np.array(f["Event"][event][channel]["Bins"]["Dff"], dtype=np.float64),
                        bins_zscore=np.array(f["Event"][event][channel]["Bins"]["Zscore"], dtype=np.float64),
                    )
                )

            activities.append(activity)

        filename = os.path.basename(file)
        name = re.search(r"(\d+_\d)", filename).group(0)
        return Processed5CData(name=name, label="TODO", activities=activities, sampling_rate=f['Meta/Sampling_Rate'])


def bootstrap_ci(data, n_boot=2000, ci=95):
    """
    data: numpy array (n_events x n_time)
    n_boot: number of bootstrap resamples
    ci: confidence interval width (e.g. 95)

    Returns:
      mean_trace: average across events
      lower, upper: confidence interval bounds
    """
    n_events, n_time = data.shape
    boot_means = np.zeros((n_boot, n_time))

    for i in range(n_boot):
        resample_idx = np.random.choice(n_events, n_events, replace=True)
        boot_sample = data[resample_idx, :]
        boot_means[i, :] = boot_sample.mean(axis=0)

    lower = np.percentile(boot_means, (100 - ci) / 2, axis=0)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2, axis=0)

    return lower, upper


def compute_auc(time, trace, t_min, t_max):
    """
    Compute area under the curve (AUC) for a single trace.

    time: 1D array of time points
    trace: 1D array (same length as time)
    t_min, t_max: time window over which to compute AUC
    """
    mask = (time >= t_min) & (time <= t_max)
    return np.trapezoid(trace[mask], time[mask])


indir = "/Users/atanas/Documents/workspace/data/analysis/photometry/5C/processed"
out_dir = '/Users/atanas/Documents/workspace/data/analysis/photometry/5C'
pdf = PdfPages(join(out_dir, "output.pdf"))

file = "../mice.csv"
groups = {'nac': [], 'bla': []}
with open(file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        mouse_id = row['mouse']
        properties = {k: row[k].lower() for k in row.keys()}
        area = row['area']
        if area in groups:
            groups[area].append(mouse_id)

print(groups)

files = [f for f in listdir(indir) if isfile(join(indir, f)) and f.endswith(".h2py")]
log(f"Files {files}")

subjects: list[Processed5CData] = []
for file in files:
    data = read_processed_data(join(indir, file))
    subjects.append(data)


def calc_stars(p_val) -> int:
    if p_val < 0.001:
        stars = '***'
    elif p_val < 0.01:
        stars = '**'
    elif p_val < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'
    return stars


for event in events:
    data = {
        "nac": {
            "signal_dff": [],
            "signal_z": [],
        },
        "bla": {
            "signal_dff": [],
            "signal_z": [],
        },
    }
    for sub in subjects:
        activity = next((a for a in sub.activities if a.event == event), None)
        mouse_id = sub.name.replace("_", ".")
        group = None
        if mouse_id in groups['nac']:
            group = 'nac'
        elif mouse_id in groups['bla']:
            group = 'bla'

        if group is None:
            log(f"WARNING: subject {sub.name} does not belong in any group -> skipping")
            continue

        data[group]["signal_dff"].append(
            compute_mean(activity.signal_corr().bins_dff)
        )
        data[group]["signal_z"].append(
            compute_mean(activity.signal_corr().bins_zscore)
        )

    log(f"Plotting {event}")
    time_before_plot = 5
    time_after_plot = 10
    baseline_correction_from = 2
    sampling_rate = 240  # FIXME: read from individual files, handle padding when discrepancies exist?

    fig, axs = plt.subplots(3, 2, gridspec_kw={"width_ratios": [1, 1]})

    for label in ['bla', 'nac']:
        means_dff = pad(data[label]["signal_dff"])

        baseline_window = baseline_correction_from * sampling_rate
        signal_dff_group_mean = np.mean(means_dff, axis=0)
        signal_dff_baseline = np.mean(signal_dff_group_mean[:baseline_correction_from])
        signal_dff_corrected_group_mean = np.subtract(
            signal_dff_group_mean, signal_dff_baseline
        )

        dff_lower_ci, dff_upper_ci = bootstrap_ci(means_dff)

        ax_col = 0 if label == 'bla' else 1
        pseudotime = np.linspace(
            -time_before_plot,
            time_after_plot,
            num=len(means_dff[0]),
        )
        axs[0, ax_col].plot(pseudotime, signal_dff_corrected_group_mean, color="#076e18")
        axs[0, ax_col].fill_between(
            pseudotime,
            dff_lower_ci - signal_dff_baseline,
            dff_upper_ci - signal_dff_baseline,
            alpha=0.5,
            edgecolor="#195423",
            facecolor="#35c44d",
        )
        axs[0, ax_col].set_ylabel("ΔF/F")
        axs[0, ax_col].set_xticks([])
        axs[0, ax_col].set_xlim([-time_before_plot, time_after_plot])
        axs[0, ax_col].set_ylim([-1.0, 1.0])
        axs[0, ax_col].set_title(f"{label} (N={len(means_dff)})")
        axs[0, ax_col].hlines(0, -10000, 10000, linestyle="dashed", color="#000", alpha=0.2)
        axs[0, ax_col].vlines(0, -10000, 10000, linestyle="dotted", color="#000")
        # axs[0, ax_col].legend(loc="upper right")

        # z-score
        means_zscore = pad(data[label]["signal_z"])
        signal_z_group_mean = np.mean(means_zscore, axis=0)
        signal_z_baseline = np.mean(signal_z_group_mean[:baseline_correction_from])
        signal_z_corrected_group_mean = np.subtract(
            signal_z_group_mean, signal_z_baseline
        )

        zscore_lower_ci, zscore_upper_ci = bootstrap_ci(means_zscore)

        axs[1, ax_col].plot(pseudotime, signal_z_corrected_group_mean, color="#ad30e3")
        axs[1, ax_col].fill_between(
            pseudotime,
            zscore_lower_ci - signal_z_baseline,
            zscore_upper_ci - signal_z_baseline,
            alpha=0.5,
            edgecolor="#9a48bd",
            facecolor="#cd7df0",
        )
        axs[1, ax_col].set_ylabel("z-score")
        axs[1, ax_col].set_xticks([-5, 0, 5, 10])
        axs[1, ax_col].set_xlim([-time_before_plot, time_after_plot])
        axs[1, ax_col].set_ylim([-2.0, 2.0])
        # axs[1, ax_col].set_title(f"{label} (N={len(means_zscore)})")
        axs[1, ax_col].hlines(0, -10000, 10000, linestyle="dashed", color="#000", alpha=0.2)
        axs[1, ax_col].vlines(0, -10000, 10000, linestyle="dotted", color="#000")
        # axs[1, ax_col].legend(loc="upper right")

        # AUC
        steps = [(-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3), (3, 4)]
        auc = []
        auc_labels = []
        for step in steps:
            t_min, t_max = step
            auc_value = [compute_auc(pseudotime, transient, t_min, t_max) for transient in means_zscore]
            if len(auc) > 0:
                prev = auc[-1]
                t_stat, p_val = ttest_rel(auc_value, prev)
                print(f"t-stat={t_stat}, p-val={p_val}")
                if p_val < 0.05:
                    x1, x2 = len(auc), len(auc) + 1
                    y, h = max(max(prev), max(auc_value)) + 0.5, 0.2  # height for the line
                    stars = calc_stars(p_val)
                    print(f"{stars}... {t_min}—{t_max}s")
                    axs[2, ax_col].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
                    axs[2, ax_col].text((x1 + x2) * .5, y + h + 0.05, stars, ha='center', va='bottom', fontsize=8)

            auc.append(auc_value)
            auc_labels.append(f'{t_min}—{t_max}s')

        axs[2, ax_col].boxplot(auc, showfliers=False, showmeans=True, bootstrap=2000)
        axs[2, ax_col].set_ylabel("AUC")
        axs[2, ax_col].set_xticklabels(auc_labels, rotation=45, fontsize=8)
        axs[2, ax_col].set_ylim([-3.0, 3.0])

    fig.suptitle(event)
    # fig.set_size_inches(8, 3)
    # plt.subplots_adjust(wspace=0.3)
    pdf.savefig(fig)

pdf.close()
