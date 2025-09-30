import csv
import os
import re

import h5py
import numpy as np

from common.lib import Processed5CData, Activity, Channel, ChannelType
from scipy import integrate
from statsmodels.stats.multitest import multipletests

def load_groups(file: str):
    groups = {'nac': [], 'bla': []}
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mouse_id = row['mouse']
            properties = {k: row[k].lower() for k in row.keys()}
            area = row['area']
            if area in groups:
                groups[area].append(mouse_id)

    return groups

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

        filename = os.path.basename(infile)
        # FIXME: read from h5py
        name = re.search(r"(\d+_\d)", filename).group(0)
        return Processed5CData(name=name, label="TODO", activities=activities, sampling_rate=f['Meta/Sampling_Rate'])


# def baseline_correct(traces, pseudotime, baseline_window=(-5, -3)):
#     """Subtract mean baseline per epoch."""
#     mask = (pseudotime >= baseline_window[0]) & (pseudotime <= baseline_window[1])
#     corrected = traces - traces[:, mask].mean(axis=1, keepdims=True)
#     return corrected


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
            # ep_bc = baseline_correct(ep_resampled, t_epoch, baseline_window)
            subj_means.append(ep_resampled.mean(axis=0))

        boot_means[b] = np.mean(subj_means, axis=0)

    return boot_means
    # group_mean = np.mean(boot_means, axis=0)
    # ci_lower = np.percentile(boot_means, 2.5, axis=0)
    # ci_upper = np.percentile(boot_means, 97.5, axis=0)
    # return group_mean, ci_lower, ci_upper

def subject_mean_auc(epochs, t_epoch, window):
    mask = (t_epoch >= window[0]) & (t_epoch <= window[1])
    # per-trial AUCs
    aucs = [np.trapz(ep[mask], t_epoch[mask]) for ep in epochs]
    return np.mean(aucs)

def hierarchical_boot_between_groups(groupA_epochs, groupB_epochs, t_epoch, window, n_boot=2000):
    subjectsA = list(groupA_epochs.keys())
    subjectsB = list(groupB_epochs.keys())
    boot_diffs = np.zeros(n_boot)
    for b in range(n_boot):
        # Group A
        sampled_A = np.random.choice(subjectsA, size=len(subjectsA), replace=True)
        subj_means_A = []
        for s in sampled_A:
            epochs = groupA_epochs[s]              # shape (n_events, n_time)
            ev_idx = np.random.choice(len(epochs), size=len(epochs), replace=True)
            ep_sample = epochs[ev_idx]
            subj_means_A.append(subject_mean_auc(ep_sample, t_epoch, window))
        meanA = np.mean(subj_means_A)

        # Group B
        sampled_B = np.random.choice(subjectsB, size=len(subjectsB), replace=True)
        subj_means_B = []
        for s in sampled_B:
            epochs = groupB_epochs[s]
            ev_idx = np.random.choice(len(epochs), size=len(epochs), replace=True)
            ep_sample = epochs[ev_idx]
            subj_means_B.append(subject_mean_auc(ep_sample, t_epoch, window))
        meanB = np.mean(subj_means_B)

        boot_diffs[b] = meanA - meanB

    # p-value (two-tailed)
    p = 2 * min(np.mean(boot_diffs >= 0), np.mean(boot_diffs <= 0))
    ci = np.percentile(boot_diffs, [2.5, 97.5])
    return boot_diffs, p, ci

def compute_auc(time, trace, t_min, t_max):
    """
    Compute area under the curve (AUC) for a single trace.

    time: 1D array of time points
    trace: 1D array (same length as time)
    t_min, t_max: time window over which to compute AUC
    """
    mask = (time >= t_min) & (time <= t_max)
    return np.trapezoid(trace[mask], time[mask])


def compute_boot_auc(boot_means, t_epoch, windows):
    """Compute AUCs for each bootstrap replicate and each window."""
    aucs = {w: [] for w in windows}
    for bm in boot_means:
        for w in windows:
            mask = (t_epoch >= w[0]) & (t_epoch <= w[1])
            aucs[w].append(integrate.trapezoid(bm[mask], t_epoch[mask]))
    # convert lists to arrays
    aucs = {w: np.array(v) for w, v in aucs.items()}
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


def compare_boot_bins(aucs, windows):
    """Compare consecutive windows with multiple comparison correction."""
    comparisons = []
    p_vals = []
    diffs = []
    for i in range(len(windows) - 1):
        w1, w2 = windows[i], windows[i + 1]
        diff = aucs[w2] - aucs[w1]
        p_val = 2 * min(np.mean(diff >= 0), np.mean(diff <= 0))
        comparisons.append((w1, w2))
        diffs.append(diff)
        p_vals.append(p_val)

    reject, p_adj, _, _ = multipletests(p_vals, method="holm")
    results = {
        "comparisons": comparisons,
        "diffs": diffs,
        "p_vals": p_vals,
        "p_adj": p_adj,
        "reject": reject
    }
    return results
