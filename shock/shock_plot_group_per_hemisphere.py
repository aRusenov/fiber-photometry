import csv
import os
import re
import sys
from os import listdir
from os.path import isfile, join

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from scipy import integrate

# Add the parent for import
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from common.lib import (
    Activity,
    Channel,
    ChannelType,
    Processed5CData,
    log,
)


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
        # FIXME: read from h5py
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

def baseline_correct(traces, pseudotime, baseline_window=(-5, -3)):
    """Subtract mean baseline per epoch."""
    mask = (pseudotime >= baseline_window[0]) & (pseudotime <= baseline_window[1])
    corrected = traces - traces[:, mask].mean(axis=1, keepdims=True)
    return corrected

def hierarchical_bootstrap(subject_epochs, n_boot=500, baseline_window=(-5, -3)):
    """
    subject_epochs: dict {subject: (epochs, t_epoch)}
        epochs: array shape (n_events, n_time)
        t_epoch: array shape (n_time,)
    Returns: bootstrap distribution of group mean traces
    """
    subjects = list(subject_epochs.keys())
    n_time = list(subject_epochs.values())[0][0].shape[1]
    boot_means = np.zeros((n_boot, n_time))

    for b in range(n_boot):
        # resample subjects with replacement
        subj_sample = np.random.choice(subjects, size=len(subjects), replace=True)
        subj_means = []
        for subj in subj_sample:
            epochs, t_epoch = subject_epochs[subj]
            # resample events within subject
            ev_idx = np.random.choice(range(epochs.shape[0]), size=epochs.shape[0], replace=True)
            ep_resampled = epochs[ev_idx]
            # baseline correct per epoch
            ep_bc = baseline_correct(ep_resampled, t_epoch, baseline_window)
            subj_means.append(ep_bc.mean(axis=0))

        boot_means[b] = np.mean(subj_means, axis=0)

    return boot_means
    # group_mean = np.mean(boot_means, axis=0)
    # ci_lower = np.percentile(boot_means, 2.5, axis=0)
    # ci_upper = np.percentile(boot_means, 97.5, axis=0)
    # return group_mean, ci_lower, ci_upper

def compute_auc(time, trace, t_min, t_max):
    """
    Compute area under the curve (AUC) for a single trace.

    time: 1D array of time points
    trace: 1D array (same length as time)
    t_min, t_max: time window over which to compute AUC
    """
    mask = (time >= t_min) & (time <= t_max)
    return np.trapezoid(trace[mask], time[mask])

def compute_boot_auc(boot_means, t_epoch, windows=[(-2,0),(0,2)]):
    """Compute AUCs for each bootstrap replicate and each window."""
    aucs = {w: [] for w in windows}
    for bm in boot_means:
        for w in windows:
            mask = (t_epoch >= w[0]) & (t_epoch <= w[1])
            aucs[w].append(integrate.trapezoid(bm[mask], t_epoch[mask]))
    # convert lists to arrays
    aucs = {w: np.array(v) for w,v in aucs.items()}
    return aucs


def create_text_page():
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')

    font = FontProperties()
    font.set_size(14)

    plt.text(0.1, 0.9, 'Right hemisphere', fontproperties=font, fontweight='bold')
    plt.text(0.1, 0.85, f'Generated on: {os.path.basename(__file__)}', fontproperties=font)
    plt.text(0.1, 0.80, f'Analysis date: {os.path.getctime(__file__)}', fontproperties=font)

    return fig


def baseline_correct(traces, pseudotime, baseline_window=(-5, -3)):
    """Subtract mean baseline per epoch."""
    mask = (pseudotime >= baseline_window[0]) & (pseudotime <= baseline_window[1])
    corrected = traces - traces[:, mask].mean(axis=1, keepdims=True)
    return corrected

def compare_boot_auc(aucs, window1, window2):
    """Perform significance testing by computing distribution of differences."""
    diff = aucs[window2] - aucs[window1]
    p_val = 2 * min(
        np.mean(diff >= 0),
        np.mean(diff <= 0)
    )  # two-tailed p-value
    return diff, p_val


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

events = [
    "Onset",
]

experiment = 'foot-shock'
indir = "/Users/atanas/Documents/workspace/data/analysis/photometry/shock/processed"
out_dir = '/Users/atanas/Documents/workspace/data/analysis/photometry/shock'
files = [f for f in listdir(indir) if isfile(join(indir, f)) and f.endswith(".h2py")]
log(f"Files {files}")

subjects_left: list[Processed5CData] = []
subjects_right: list[Processed5CData] = []
for file in files:
    data = read_processed_data(join(indir, file))
    if 'left' in file:
        subjects_left.append(data)
    elif 'right' in file:
        subjects_right.append(data)
    else:
        raise Exception(f"Unknown hemisphere {file}")

np.random.seed(7092025)  # FIXED SEED FOR REPRODUCIBILITY

pdf = PdfPages(join(out_dir, f"plot-group-{experiment}.pdf"))
# pdf.savefig(create_text_page())

hemispheres = [
    ("left", subjects_left),
    ("right", subjects_right)
]

for hemisphere, subjects in hemispheres:
    for event in events:
        data = {
            "nac": [],
            "bla": []
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
            data[group].append(
                {"sub": mouse_id, "dff": activity.signal_corr().bins_dff, "zscore": activity.signal_corr().bins_zscore})

            # data[group]["signal_dff"].append(
            #     compute_mean(activity.signal_corr().bins_dff)
            # )
            # data[group]["signal_z"].append(
            #     compute_mean(activity.signal_corr().bins_zscore)
            # )

        log(f"Plotting {event}")
        time_before_plot = 5
        time_after_plot = 10
        sampling_rate = 240  # FIXME: read from individual files, handle padding when discrepancies exist?

        fig, axs = plt.subplots(3, 2, gridspec_kw={"width_ratios": [1, 1]})
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        for label in ['bla', 'nac']:
            # means_dff = pad(data[label]["signal_dff"])
            # means_dff = np.array([sub["dff"] for sub in data[label]])
            # means_zscore = np.array([sub["zscore"] for sub in data[label]])

            ax_col = 0 if label == 'bla' else 1

            pseudotime = np.linspace(
                -time_before_plot,
                time_after_plot,
                num=len(data[label][0]["zscore"][0]),
            )

            subs = data[label]

            subject_dff_traces = {s['sub']: (s['dff'], pseudotime) for s in subs}
            means_dff = hierarchical_bootstrap(subject_dff_traces, n_boot=200)
            signal_dff_group_mean = np.mean(means_dff, axis=0)
            ci_lower = np.percentile(means_dff, 2.5, axis=0)
            ci_upper = np.percentile(means_dff, 97.5, axis=0)

            # signal_dff_corrected_baseline_corrected_bins = baseline_correct(means_dff, pseudotime)
            # signal_dff_group_mean = np.mean(signal_dff_corrected_baseline_corrected_bins, axis=0)
            # dff_lower_ci, dff_upper_ci = bootstrap_ci(signal_dff_corrected_baseline_corrected_bins)

            axs[0, ax_col].plot(pseudotime, signal_dff_group_mean, label="GCaMP corrected", color="#076e18")
            axs[0, ax_col].fill_between(
                pseudotime,
                ci_lower,
                ci_upper,
                label="95% CI",
                alpha=0.3,
                color="#076e18",
            )
            axs[0, ax_col].set_ylabel("ΔF/F", fontsize=10)
            axs[0, ax_col].set_xticks([])
            axs[0, ax_col].set_xlim([-time_before_plot, time_after_plot])
            axs[0, ax_col].set_ylim([-0.3, 0.3])
            axs[0, ax_col].set_title(f"{label} (N={len(subs)})")
            axs[0, ax_col].hlines(0, -10000, 10000, linestyle="dashed", color="#000", alpha=0.2)
            axs[0, ax_col].vlines(0, -10000, 10000, linestyle="dotted", color="#000")
            axs[0, ax_col].legend(loc="upper right", fontsize=5)

            # z-score
            # means_zscore = pad(data[label]["signal_z"])
            # signal_z_group_mean = np.mean(means_zscore, axis=0)
            # signal_z_baseline = np.mean(signal_z_group_mean[:baseline_correction_from])
            # signal_z_corrected_group_mean = np.subtract(
            #     signal_z_group_mean, signal_z_baseline
            # )

            # zscore_lower_ci, zscore_upper_ci = bootstrap_ci(means_zscore)

            subject_zscore_traces = {s['sub']: (s['zscore'], pseudotime) for s in subs}
            means_zscore = hierarchical_bootstrap(subject_zscore_traces, n_boot=200)
            signal_zscore_group_mean = np.mean(means_zscore, axis=0)
            ci_lower = np.percentile(means_zscore, 2.5, axis=0)
            ci_upper = np.percentile(means_zscore, 97.5, axis=0)

            axs[1, ax_col].plot(pseudotime, signal_zscore_group_mean, label="GCaMP corrected", color="#f5b642")
            axs[1, ax_col].fill_between(
                pseudotime,
                ci_lower,
                ci_upper,
                label="95% CI",
                alpha=0.3,
                color="#f5b642",
            )
            axs[1, ax_col].set_ylabel("z-score")
            axs[1, ax_col].set_xticks([-5, 0, 5, 10])
            axs[1, ax_col].set_xlim([-time_before_plot, time_after_plot])
            axs[1, ax_col].set_ylim([-1, 3])
            # axs[1, ax_col].set_title(f"{label} (N={len(means_zscore)})")
            axs[1, ax_col].hlines(0, -10000, 10000, linestyle="dashed", color="#000", alpha=0.2)
            axs[1, ax_col].vlines(0, -10000, 10000, linestyle="dotted", color="#000")
            axs[1, ax_col].legend(loc="upper right", fontsize=5)

            # AUC
            steps = [(-4, -2), (-2, 0), (0, 2), (2, 4), (4, 6)]
            auc = []
            auc_labels = []

            aucs = compute_boot_auc(means_zscore, pseudotime, windows=steps)
            for (start, end), _ in aucs.items():
                auc_labels.append(f'{start}—{end}s')

            y_max = 5
            for idx, (window_prev, window_current) in enumerate(zip(steps, steps[1:])):
                _, p_val = compare_boot_auc(aucs, window_prev, window_current)
                print(f"{window_prev} x {window_current}: p-val={p_val}")
                if p_val < 0.05:
                    x1, x2 = idx + 1, idx + 2

                    y, h, col = y_max + 0.05, 0.05, 'k'
                    # x1, x2 = len(auc), len(auc) + 1
                    # y, h = max(max(prev), max(auc_value)) + 0.5, 0.2  # height for the line
                    stars = calc_stars(p_val)
                    print(f"{stars}... {window_prev}—{window_current}s")
                    axs[2, ax_col].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                    axs[2, ax_col].text((x1+x2)*.5, y+h, stars, ha='center', va='bottom', color=col)

            axs[2, ax_col].boxplot([auc for _, auc in aucs.items()], showfliers=False, showmeans=False)
            axs[2, ax_col].set_ylabel("AUC")
            axs[2, ax_col].set_xticklabels(auc_labels, rotation=45, fontsize=8)
            axs[2, ax_col].set_ylim([-3, 7])

        fig.suptitle(f'{event} ({hemisphere})', fontsize=12)
        pdf.savefig(fig)

plt.subplots_adjust(wspace=0.3, hspace=0.8)
pdf.close()
