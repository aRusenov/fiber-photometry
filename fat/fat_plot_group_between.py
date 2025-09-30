import csv
import os
import re
import sys
from os import listdir
from os.path import isfile, join

import h5py
import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from scipy import integrate
from statsmodels.stats.multitest import multipletests

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


# def hierarchical_bootstrap(subject_epochs, n_boot=500, baseline_window=(-5, -3)):
#     """
#     subject_epochs: dict {subject: (epochs, t_epoch)}
#         epochs: array shape (n_events, n_time)
#         t_epoch: array shape (n_time,)
#     Returns: bootstrap distribution of group mean traces
#     """
#     subjects = list(subject_epochs.keys())
#     n_time = list(subject_epochs.values())[0][0].shape[1]
#     boot_means = np.zeros((n_boot, n_time))
#
#     for b in range(n_boot):
#         # resample subjects with replacement
#         subj_sample = np.random.choice(subjects, size=len(subjects), replace=True)
#         subj_means = []
#         for subj in subj_sample:
#             epochs, t_epoch = subject_epochs[subj]
#             # resample events within subject
#             ev_idx = np.random.choice(range(epochs.shape[0]), size=epochs.shape[0], replace=True)
#             ep_resampled = epochs[ev_idx]
#             # baseline correct per epoch
#             ep_bc = baseline_correct(ep_resampled, t_epoch, baseline_window)
#             subj_means.append(ep_bc.mean(axis=0))
#
#         boot_means[b] = np.mean(subj_means, axis=0)
#
#     return boot_means
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


def compute_boot_auc(boot_means, t_epoch, windows=[(-2, 0), (0, 2)]):
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


def hierarchical_bootstrap_group(group, t_epoch, n_boot=2000, windows=[]):
    """
    Hierarchical bootstrap for one group of subjects.

    group: dict
        {subject: [ {"epochs": np.ndarray}, {"epochs": np.ndarray}, ... ]}
        epochs: array of shape (n_trials, n_time)

    n_boot: int
        number of bootstrap iterations

    t_epoch: array
        time vector for each epoch

    window: tuple
        time window (start, end) for AUC computation

    Returns:
        boot_means: array (n_boot, n_time)
    """
    np.random.seed(0)  # reproducibility
    subjects = list(group.keys())
    n_time = list(group.values())[0][0]["epochs"].shape[1]
    boot_means = np.zeros((n_boot, n_time))
    boot_aucs = {w: np.zeros(n_boot) for w in windows}

    # Precompute masks for each window
    window_masks = {}
    if t_epoch is not None:
        for w in windows:
            mask = (t_epoch >= w[0]) & (t_epoch <= w[1])
            window_masks[w] = mask

    for b in range(n_boot):
        subj_sample = np.random.choice(subjects, size=len(subjects), replace=True)
        subj_means = []

        for subj in subj_sample:
            recs = group[subj]
            # sample recordings with replacement
            rec_idx = np.random.choice(len(recs), size=len(recs), replace=True)

            rec_means = []
            for ri in rec_idx:
                epochs = recs[ri]["epochs"]
                # resample trials
                ev_idx = np.random.choice(epochs.shape[0], size=epochs.shape[0], replace=True)
                ep_resampled = epochs[ev_idx]
                rec_means.append(ep_resampled.mean(axis=0))

            # subject mean = average over its recordings
            subj_means.append(np.mean(rec_means, axis=0))

        # Group mean for this bootstrap iteration
        boot_mean = np.mean(subj_means, axis=0)
        boot_means[b] = boot_mean

        # Compute AUCs for each window
        if t_epoch is not None:
            for w in windows:
                mask = window_masks[w]
                boot_aucs[w][b] = np.trapz(boot_mean[mask], t_epoch[mask])

    return boot_means, boot_aucs


# def hierarchical_boot_between_groups_with_recordings(groupA, groupB, t_epoch, window, n_boot=2000, seed=0,
#                                                      weighting='equal'):
#     """
#     groupA, groupB: dict {subject_id: list_of_recordings}, each recording is {'epochs': np.array}
#     weighting: how to collapse recordings into subject mean ('equal' or 'trial')
#     """
#     np.random.seed(seed)
#     subjectsA = list(groupA.keys())
#     subjectsB = list(groupB.keys())
#     boot_diffs = np.zeros(n_boot)
#     # meanA_diffs = np.zeros(n_boot)
#     # meanB_diffs = np.zeros(n_boot)
#     mask = (t_epoch >= window[0]) & (t_epoch <= window[1])
#
#     for b in range(n_boot):
#         # Group A
#         sampled_A = np.random.choice(subjectsA, size=len(subjectsA), replace=True)
#         subj_means_A = []
#         for s in sampled_A:
#             recs = groupA[s]
#             rec_idx = np.random.choice(len(recs), size=len(recs), replace=True)
#             rec_means = []
#             rec_n = []
#             for ri in rec_idx:
#                 epochs = recs[ri]['epochs']
#                 ev_idx = np.random.choice(range(epochs.shape[0]), size=epochs.shape[0], replace=True)
#                 ep_sample = baseline_correct(epochs[ev_idx], t_epoch)
#                 aucs = [np.trapz(ep[mask], t_epoch[mask]) for ep in ep_sample]
#                 rec_means.append(np.mean(aucs))
#                 rec_n.append(len(aucs))
#             if weighting == 'equal':
#                 subj_means_A.append(np.mean(rec_means))
#             else:
#                 subj_means_A.append(np.sum(np.array(rec_means) * np.array(rec_n)) / np.sum(rec_n))
#         meanA = np.mean(subj_means_A)
#         # meanA_diffs[b] = meanA
#
#         # Group B
#         sampled_B = np.random.choice(subjectsB, size=len(subjectsB), replace=True)
#         subj_means_B = []
#         for s in sampled_B:
#             recs = groupB[s]
#             rec_idx = np.random.choice(len(recs), size=len(recs), replace=True)
#             rec_means = []
#             rec_n = []
#             for ri in rec_idx:
#                 epochs = recs[ri]['epochs']
#                 ev_idx = np.random.choice(range(epochs.shape[0]), size=epochs.shape[0], replace=True)
#                 ep_sample = baseline_correct(epochs[ev_idx], t_epoch)
#                 aucs = [np.trapz(ep[mask], t_epoch[mask]) for ep in ep_sample]
#                 rec_means.append(np.mean(aucs))
#                 rec_n.append(len(aucs))
#             if weighting == 'equal':
#                 subj_means_B.append(np.mean(rec_means))
#             else:
#                 subj_means_B.append(np.sum(np.array(rec_means) * np.array(rec_n)) / np.sum(rec_n))
#         meanB = np.mean(subj_means_B)
#         # meanB_diffs[b] = meanB
#
#         boot_diffs[b] = meanA - meanB
#
#     p = 2 * min(np.mean(boot_diffs >= 0), np.mean(boot_diffs <= 0))
#     ci = np.percentile(boot_diffs, [2.5, 97.5])
#     return boot_diffs, p, ci


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


def get_ylim(signal):
    signal_max = np.max(np.abs(signal))
    if signal_max < 0.1:
        lim = 0.1
    else:
        lim = np.ceil(signal_max * 10) / 10

    lim *= 1.1

    return [-lim, lim]


def compare_boot_bins(aucs_a, windows, aucs_b=None, method="holm"):
    """
    Compare bootstrapped AUCs either:
    - Between groups (aucs_a vs aucs_b), for the same windows
    - Within group (aucs_a only), comparing consecutive windows

    Parameters
    ----------
    aucs_a : dict
        {window: array of bootstrapped AUCs} for group A
    windows : list of tuples
        Time windows to compare (e.g. [(-2,0),(0,2),(2,4)])
    aucs_b : dict or None
        {window: array of bootstrapped AUCs} for group B, or None if within-group
    method : str
        Multiple comparisons correction method (default: "holm")

    Returns
    -------
    results : dict
        {
          "comparisons": list of window pairs,
          "diffs": list of arrays of bootstrap diffs,
          "p_vals": list of raw p-values,
          "p_adj": list of adjusted p-values,
          "reject": list of booleans
        }
    """
    comparisons, diffs, p_vals, ci_upper, ci_lower = [], [], [], [], []
    alpha = 0.05

    if aucs_b is not None:
        # -----------------
        # Between-groups case
        # -----------------
        for w in windows:
            diff = aucs_a[w] - aucs_b[w]
            p_val = 2 * min(np.mean(diff >= 0), np.mean(diff <= 0))
            comparisons.append((w, "between"))
            diffs.append(diff)
            p_vals.append(p_val)
            ci = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            ci_lower.append(ci[0])
            ci_upper.append(ci[1])
    else:
        # -----------------
        # Within-group case (compare consecutive windows)
        # -----------------
        for i in range(1, len(windows)):
            for j in range(i):
                w1, w2 = windows[j], windows[i]
                diff = aucs_a[w2] - aucs_a[w1]
                p_val = 2 * min(np.mean(diff >= 0), np.mean(diff <= 0))
                comparisons.append((w1, w2))
                diffs.append(diff)
                p_vals.append(p_val)
                ci = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
                ci_lower.append(ci[0])
                ci_upper.append(ci[1])

    # Multiple comparison correction
    reject, p_adj, _, _ = multipletests(p_vals, method=method)

    return {
        "comparisons": comparisons,
        "diffs": diffs,
        "p_vals": p_vals,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_adj": p_adj,
        "reject": reject
    }


def compare_boot_bins_arr(aucs, windows):
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
    "Outset"
]

experiment = 'fat'
indir = "/Users/atanas/Documents/workspace/data/analysis/photometry/fat/processed"
out_dir = '/Users/atanas/Documents/workspace/data/analysis/photometry/fat'
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

pdf = PdfPages(join(out_dir, f"plot-group-{experiment}-between.pdf"))
# pdf.savefig(create_text_page())

hemispheres = [
    ("left", subjects_left),
    ("right", subjects_right)
]

all_mice = np.concatenate([np.array(s).flatten() for s in groups.values()])

time_before_plot = 5
time_after_plot = 10
sampling_rate = 240

pseudotime = np.linspace(
    -time_before_plot,
    time_after_plot,
    num=sampling_rate * (time_before_plot + time_after_plot),
)

# import matplotlib as mpl
# mpl.use('macosx')

processed_data = dict()
excl = [('656.3', 'left'), ('656.3', 'right'),
        ('201.3', 'left'),
        ('652.3', 'left'), ('652.3', 'right'),
        ('201.2', 'right')]

for event in events:
    log(f"Processing {event}")
    processed_data[event] = {
        "left": {
            "nac": {"sub": [], "dff": [], "zscore": []},
            "bla": {"sub": [], "dff": [], "zscore": []}
        },
        "right": {
            "nac": {"sub": [], "dff": [], "zscore": []},
            "bla": {"sub": [], "dff": [], "zscore": []}
        }
    }

    data = {"nac": {}, "bla": {}}
    for hemisphere, subjects in hemispheres:
        for sub in subjects:
            activity = next((a for a in sub.activities if a.event == event), None)
            mouse_id = sub.name.replace("_", ".")

            if any(mouse_id == e[0] and hemisphere == e[1] for e in excl):
                log(f"WARNING: subject {mouse_id} ({hemisphere}) in exclusion list -> skipping")
                continue

            group = None
            if mouse_id in groups['nac']:
                group = 'nac'
            elif mouse_id in groups['bla']:
                group = 'bla'

            if group is None:
                log(f"WARNING: subject {sub.name} does not belong in any group -> skipping")
                continue
            # print(mouse_id)

            event_count = len(activity.signal_corr().bins_zscore)

            if sub.name not in data[group]:
                data[group][sub.name] = []

            data[group][sub.name].append(
                {"epochs": activity.signal_corr().bins_zscore,
                 "hemisphere": hemisphere})
            # processed_data[event][hemisphere][group][sub].append(
            #     {"sub": mouse_id, "dff": activity.signal_corr().bins_dff, "zscore": activity.signal_corr().bins_zscore})

        # log(f"Plotting {event}")
        # time_before_plot = 5
        # time_after_plot = 10
        # sampling_rate = 240  # FIXME: read from individual files, handle padding when discrepancies exist?
        #
        # fig, axs = plt.subplots(3, 2, gridspec_kw={"width_ratios": [1, 1]})
        # plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # print(data)

    # Filter subjects with insufficient epochs
    min_count = 5
    for group in ['nac', 'bla']:
        to_remove = []
        for subject in data[group]:
            total_epochs = sum(len(rec['epochs']) for rec in data[group][subject])
            if total_epochs < min_count:
                log(f"WARNING: Subject {subject} has less than {min_count} events ({total_epochs}) -> skipping")
                to_remove.append(subject)
        for subject in to_remove:
            del data[group][subject]

    # windows = [(-2, 0), (0, 2), (2, 4), (4, 6)]
    # Calculate total NAc epochs
    total_nac_epochs = sum(len(rec['epochs']) for subject in data['nac'].values() for rec in subject)
    total_bla_epochs = sum(len(rec['epochs']) for subject in data['nac'].values() for rec in subject)

    windows = [(-2, 0), (0, 2), (2, 4)]

    nac_means, nac_aucs = hierarchical_bootstrap_group(data["nac"], pseudotime, n_boot=2000, windows=windows)
    nac_group_mean = np.mean(nac_means, axis=0)
    nac_ci_lower = np.percentile(nac_means, 2.5, axis=0)
    nac_ci_upper = np.percentile(nac_means, 97.5, axis=0)

    nac_aucs_results = compare_boot_bins(nac_aucs, windows)
    print('NAC', nac_aucs_results)

    bla_means, bla_aucs = hierarchical_bootstrap_group(data["bla"], pseudotime, n_boot=2000, windows=windows)
    bla_group_mean = np.mean(bla_means, axis=0)
    bla_ci_lower = np.percentile(bla_means, 2.5, axis=0)
    bla_ci_upper = np.percentile(bla_means, 97.5, axis=0)

    bla_aucs_results = compare_boot_bins(bla_aucs, windows)
    print('BLA', bla_aucs_results)

    mask = (pseudotime >= -5) & (pseudotime <= 10)
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])  # 2 rows, 2 cols

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(pseudotime[mask], nac_group_mean[mask], label=f"NAc (N={len(data['nac'].keys())})", color="steelblue")
    ax1.fill_between(
        pseudotime[mask],
        nac_ci_lower[mask],
        nac_ci_upper[mask],
        alpha=0.3,
        color="steelblue",
    )
    ax1.plot(pseudotime[mask], bla_group_mean[mask], label=f"BLA (N={len(data['bla'].keys())})", color="salmon")
    ax1.fill_between(
        pseudotime[mask],
        bla_ci_lower[mask],
        bla_ci_upper[mask],
        alpha=0.3,
        color="salmon",
    )
    ax1.set_ylabel("z-score")
    ax1.set_xlabel("Time (s)")
    ax1.set_xlim([-5, 10])
    ax1.set_xticks([-4, -2, 0, 2, 4, 6, 8, 10])
    ax1.set_ylim([-1, 1])
    # ax1.set_yticks([-1.5, 0, 1.5])
    ax1.vlines(0, -10000, 10000, linestyles='dotted', color="#000")
    ax1.hlines(0, -10000, 10000, linestyle="dotted", color="#000", alpha=0.2)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 1.3), fontsize=7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # results = compare_boot_bins(nac_aucs, windows, bla_aucs)
    # # print('AUCS', results)
    # y_max = 3
    # for idx, (comp, raw_p, adj_p, rej, ci_lower, ci_upper) in enumerate(
    #         zip(results["comparisons"], results["p_vals"], results["p_adj"], results["reject"], results["ci_lower"],
    #             results["ci_upper"])):
    #     print(
    #         f"(Between) {comp}: raw p={raw_p:.3f}, adj p={adj_p:.3f}, reject={rej}, CI [{ci_lower:.3f}, {ci_upper:.3f}]")
    #     if rej:
    #         x1, x2 = (idx * 2), ((idx * 2) + 1)
    #         x1 += 1
    #         x2 += 1
    #         y, h, col = y_max + 0.05, 0.05, 'k'
    #         stars = calc_stars(adj_p)
    #         ax2.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    #         ax2.text((x1 + x2) * .5, y + h, stars, ha='center', va='bottom', color=col)

    inner_labels = ["BLA"] * len(windows)
    outer_labels = [f'{start}—{end}s' for start, end in windows]

    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    to_draw_bla = []
    print("-----BLA")
    for idx, (comp, raw_p, adj_p, rej, ci_lower, ci_upper) in enumerate(
            zip(bla_aucs_results["comparisons"], bla_aucs_results["p_vals"], bla_aucs_results["p_adj"],
                bla_aucs_results["reject"], bla_aucs_results["ci_lower"], bla_aucs_results["ci_upper"])):
        print(
            f"(Within){comp}: raw p={raw_p:.3f}, adj p={adj_p:.3f}, reject={rej}, CI [{ci_lower:.3f}, {ci_upper:.3f}]")
        if rej:
            w1, w2 = comp
            x1, x2 = windows.index(w1), windows.index(w2)
            to_draw_bla.append((x1, x2, calc_stars(adj_p)))

    for x1, x2, stars in to_draw_bla:
        x1 += 1 * 0.95
        x2 += 1 * 1.05
        y = 2.5
        h = y / 5
        ax2.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c='k')
        ax2.text((x1 + x2) * .5, y + (h * 2), stars, ha='center', va='top', color='k', fontsize=9)


    bla_aucs_arr = [bla_aucs[w] for w in windows]
    bplot_bla = ax2.boxplot(bla_aucs_arr, showfliers=False, showmeans=False,
                            medianprops=dict(color="black", linewidth=0.5),
                            tick_labels=inner_labels, patch_artist=True)
    # fill with colors
    for patch in bplot_bla['boxes']:
        patch.set_facecolor('salmon')

    ax2.set_xticklabels(outer_labels)
    ax2.set_ylabel("AUC")
    ax2.set_ylim([-3, 3])

    to_draw_nac = []
    print("-----NAc")
    for idx, (comp, raw_p, adj_p, rej, ci_lower, ci_upper) in enumerate(
            zip(nac_aucs_results["comparisons"], nac_aucs_results["p_vals"], nac_aucs_results["p_adj"],
                nac_aucs_results["reject"], nac_aucs_results["ci_lower"], nac_aucs_results["ci_upper"])):
        print(
            f"(Within){comp}: raw p={raw_p:.3f}, adj p={adj_p:.3f}, reject={rej}, CI [{ci_lower:.3f}, {ci_upper:.3f}]")
        if rej:
            w1, w2 = comp
            x1, x2 = (windows.index(w1) * 2) + 1, (windows.index(w2) * 2) + 1
            to_draw_nac.append((x1, x2, calc_stars(adj_p)))

    for x1, x2, stars in to_draw_nac:
        x1 += 1 * 0.95
        x2 += 1 * 1.05
        y = 2.5
        h = y / 5
        ax3.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c='k')
        ax3.text((x1 + x2) * .5, y + (h * 2), stars, ha='center', va='top', color='k', fontsize=9)

    nac_aucs_arr = [nac_aucs[w] for w in windows]
    bplot_nac = ax3.boxplot(nac_aucs_arr, showfliers=False, showmeans=False,
                            medianprops=dict(color="black", linewidth=0.5),
                            tick_labels=inner_labels, patch_artist=True)
    # fill with colors
    for patch in bplot_nac['boxes']:
        patch.set_facecolor('steelblue')

    ax3.set_xticklabels(outer_labels)
    ax3.set_ylabel("AUC")
    ax3.set_ylim([-3, 3])
    ax3.get_yaxis().set_visible(False)

    # to_draw.sort(key=lambda x: (x[1] - x[0], x[0]))

    # y_min = -2.5
    # for idx, (x1, x2, stars) in enumerate(to_draw):
    #     x1 += 1
    #     x2 += 1
    #     y, h, col = y_min - (idx * 1.2), 0.2, 'k'
    #     ax2.plot([x1, x1, x2, x2], [y, y - h, y - h, y], lw=1.0, c=col)
    #     ax2.text((x1 + x2) * .5, y - (h * 2), stars, ha='center', va='top', color=col, fontsize=7)


    # auc_labels = []
    # for start, end in windows:
    #     auc_labels.append(f'{start}—{end}s')



    # auc_plots = []
    # for w in windows:
    #     auc_plots.append(bla_aucs[w])
    #     auc_plots.append(nac_aucs[w])
    #
    # bplot = ax2.boxplot(auc_plots, showfliers=False, showmeans=False, medianprops=dict(color="black", linewidth=0.5),
    #                     labels=inner_labels, patch_artist=True)
    # # fill with colors
    # for patch, color in zip(bplot['boxes'], colors):
    #     patch.set_facecolor(color)
    #
    # group_centers = [(idx * 2) + 1.5 for idx in range(len(windows))]  # [1.5, 3.5, 5.5]  # centers of pairs
    # ax2.set_xticks(group_centers)
    # ax2.set_xticklabels(outer_labels)
    # # ax2.set_xlim()
    # ax2.set_ylabel("AUC")
    # ax2.set_ylim([-6, 3])
    # # ax2.hlines(0, -10000, 10000, linestyle="dotted", color="#000", alpha=0.2)
    # ax2.set_yticks([-3, 0, 3])
    # # bplot.legend(loc="upper right", fontsize=7)


    # pdf.savefig(fig)

    # nac_aucs2 = {}
    # for window in windows:
    #     nac_mean, nac_auc = hierarchical_bootstrap_group(data["nac"], pseudotime, n_boot=2000, windows=[window])
    #     nac_aucs2[window] = nac_auc[window]
    #
    # nac_aucs_results = compare_boot_bins_arr(nac_aucs2, windows)
    # print(nac_aucs_results)
    #
    # bla_aucs2 = {}
    # for window in windows:
    #     bla_mean, bla_auc = hierarchical_bootstrap_group(data["bla"], pseudotime, n_boot=2000, windows=[window])
    #     bla_aucs2[window] = bla_auc[window]
    #
    # bla_aucs_results = compare_boot_bins_arr(bla_aucs2, windows)
    # print(bla_aucs_results)

    # for window in windows:
    #     log(f"Bootstrapping {window}")
    #     boot_diffs, p, ci = hierarchical_boot_between_groups_with_recordings(
    #         data["nac"], data["bla"], pseudotime, window, n_boot=500, seed=42, weighting='equal'
    #     )
    #     if p < 0.05:
    #         stars = calc_stars(p)
    #         log(f"Bootstrapped p-value for {window}: {p:.3f} ({stars})")
    # nac_group =
    # hierarchical_boot_between_groups_with_recordings()

    # for label in ['bla', 'nac']:
    #     # means_dff = pad(data[label]["signal_dff"])
    #     # means_dff = np.array([sub["dff"] for sub in data[label]])
    #     # means_zscore = np.array([sub["zscore"] for sub in data[label]])
    #
    #     ax_col = 0 if label == 'bla' else 1
    #
    #     processed_data[event][group][sub]
    #     pseudotime = np.linspace(
    #         -time_before_plot,
    #         time_after_plot,
    #         num=len(data[label][0]["zscore"][0]),
    #     )
    #
    #     subs = data[label]
    #
    #     subject_dff_traces = {s['sub']: (s['dff'], pseudotime) for s in subs}
    #     # for s in subs:
    #     #     print(s['sub'], s['dff'].shape)
    #
    #     means_dff = hierarchical_bootstrap(subject_dff_traces, n_boot=200)
    #     signal_dff_group_mean = np.mean(means_dff, axis=0)
    #     ci_lower = np.percentile(means_dff, 2.5, axis=0)
    #     ci_upper = np.percentile(means_dff, 97.5, axis=0)
    #
    #     # signal_dff_corrected_baseline_corrected_bins = baseline_correct(means_dff, pseudotime)
    #     # signal_dff_group_mean = np.mean(signal_dff_corrected_baseline_corrected_bins, axis=0)
    #     # dff_lower_ci, dff_upper_ci = bootstrap_ci(signal_dff_corrected_baseline_corrected_bins)
    #
    #     axs[0, ax_col].plot(pseudotime, signal_dff_group_mean, label="GCaMP corrected", color="#076e18")
    #     axs[0, ax_col].fill_between(
    #         pseudotime,
    #         ci_lower,
    #         ci_upper,
    #         label="95% CI",
    #         alpha=0.3,
    #         color="#076e18",
    #     )
    #     axs[0, ax_col].set_ylabel("ΔF/F", fontsize=10)
    #     axs[0, ax_col].set_xticks([])
    #     axs[0, ax_col].set_xlim([-time_before_plot, time_after_plot])
    #     axs[0, ax_col].set_ylim([-1, 1])
    #     axs[0, ax_col].set_title(f"{label} (N={len(subs)})")
    #     axs[0, ax_col].hlines(0, -10000, 10000, linestyle="dashed", color="#000", alpha=0.2)
    #     axs[0, ax_col].vlines(0, -10000, 10000, linestyle="dotted", color="#000")
    #     axs[0, ax_col].legend(loc="upper right", fontsize=5)
    #
    #     # z-score
    #     # means_zscore = pad(data[label]["signal_z"])
    #     # signal_z_group_mean = np.mean(means_zscore, axis=0)
    #     # signal_z_baseline = np.mean(signal_z_group_mean[:baseline_correction_from])
    #     # signal_z_corrected_group_mean = np.subtract(
    #     #     signal_z_group_mean, signal_z_baseline
    #     # )
    #
    #     # zscore_lower_ci, zscore_upper_ci = bootstrap_ci(means_zscore)
    #
    #     subject_zscore_traces = {s['sub']: (s['zscore'], pseudotime) for s in subs}
    #     means_zscore = hierarchical_bootstrap(subject_zscore_traces, n_boot=200)
    #     signal_zscore_group_mean = np.mean(means_zscore, axis=0)
    #     ci_lower = np.percentile(means_zscore, 2.5, axis=0)
    #     ci_upper = np.percentile(means_zscore, 97.5, axis=0)
    #
    #     axs[1, ax_col].plot(pseudotime, signal_zscore_group_mean, label="GCaMP corrected", color="#f5b642")
    #     axs[1, ax_col].fill_between(
    #         pseudotime,
    #         ci_lower,
    #         ci_upper,
    #         label="95% CI",
    #         alpha=0.3,
    #         color="#f5b642",
    #     )
    #     axs[1, ax_col].set_ylabel("z-score")
    #     axs[1, ax_col].set_xticks([-5, 0, 5, 10])
    #     axs[1, ax_col].set_xlim([-time_before_plot, time_after_plot])
    #     axs[1, ax_col].set_ylim([-1.5, 1.5])
    #     # axs[1, ax_col].set_title(f"{label} (N={len(means_zscore)})")
    #     axs[1, ax_col].hlines(0, -10000, 10000, linestyle="dashed", color="#000", alpha=0.2)
    #     axs[1, ax_col].vlines(0, -10000, 10000, linestyle="dotted", color="#000")
    #     axs[1, ax_col].legend(loc="upper right", fontsize=5)
    #
    #     # AUC
    #     steps = [(-4, -2), (-2, 0), (0, 2), (2, 4), (4, 6)]
    #     auc = []
    #     auc_labels = []
    #
    #     aucs = compute_boot_auc(means_zscore, pseudotime, windows=steps)
    #     for (start, end), _ in aucs.items():
    #         auc_labels.append(f'{start}—{end}s')
    #
    #     y_max = 2
    #     for idx, (window_prev, window_current) in enumerate(zip(steps, steps[1:])):
    #         _, p_val = compare_boot_auc(aucs, window_prev, window_current)
    #         print(f"{window_prev} x {window_current}: p-val={p_val}")
    #         if p_val < 0.05:
    #             x1, x2 = (idx + 1) * 1.05, (idx + 2) * 0.95
    #
    #             y, h, col = y_max + 0.05, 0.05, 'k'
    #             # x1, x2 = len(auc), len(auc) + 1
    #             # y, h = max(max(prev), max(auc_value)) + 0.5, 0.2  # height for the line
    #             stars = calc_stars(p_val)
    #             print(f"{stars}... {window_prev}—{window_current}s")
    #             axs[2, ax_col].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    #             axs[2, ax_col].text((x1 + x2) * .5, y + h, stars, ha='center', va='bottom', color=col)
    #
    #     axs[2, ax_col].boxplot([auc for _, auc in aucs.items()], showfliers=False, showmeans=False)
    #     axs[2, ax_col].set_ylabel("AUC")
    #     axs[2, ax_col].set_xticklabels(auc_labels, rotation=45, fontsize=8)
    #     axs[2, ax_col].set_ylim([-4, 4])
    #
    # fig.suptitle(f'{event} ({hemisphere})', fontsize=12)
    fig.suptitle(f'{event}', fontsize=12)
    plt.tight_layout()
    pdf.savefig(fig)

plt.subplots_adjust(wspace=0.6, hspace=0.8)
pdf.close()
