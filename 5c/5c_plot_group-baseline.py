import csv
import os
import re
import sys
from os import listdir
from os.path import isfile, join
import pandas as pd

import h5py
import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.backends.backend_pdf import PdfPages
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


def bootstrap_sem_trace(boot_means):
    # standard error per timepoint
    sem = np.std(boot_means, axis=0, ddof=1)
    return sem


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
                boot_aucs[w][b] = np.trapezoid(boot_mean[mask], t_epoch[mask])

    return boot_means, boot_aucs


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


def compare_boot_bins(aucs_a, windows, aucs_b=None, to_compare=None, method="holm"):
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
        if to_compare is None:
            to_compare = []
            # Compare adjacent windows
            for i in range(1, len(windows)):
                to_compare.append((i, i - 1))

        for i, j in to_compare:
            w1, w2 = windows[i], windows[j]
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
    "Initiated trials",
    "Prematures",
    "Incorrect",
    "Correct",
    "Reward pickup",
    "Omissions",
]

experiment = '5c'
indir = "/Users/atanas/Documents/workspace/data/analysis/photometry/5c/processed"
out_dir = '/Users/atanas/Documents/workspace/data/analysis/photometry/5c'
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
all_mice = np.concatenate([np.array(s).flatten() for s in groups.values()])

time_before_plot = 10
time_after_plot = 10
sampling_rate = 240

pseudotime = np.linspace(
    -time_before_plot,
    time_after_plot,
    num=sampling_rate * (time_before_plot + time_after_plot),
)

limits = {
    "Initiated trials": {"zscore": [-1, 1], "auc": [-2.5, 2.5], "sig_y": 2},
    "Prematures": {"zscore": [-2.5, 2.5], "auc": [-5, 5], "sig_y": 3 }, #, "bin_color": "steelblue"},
    "Reward pickup": {"zscore": [-8, 5], "auc": [-10, 5], "sig_y": 3.5},
    "Omissions": {"zscore": [-1, 1], "auc": [-3, 3], "sig_y": 2.5},
    "Correct": {"zscore": [-6, 6], "auc": [-5, 5], "sig_y": 3.5 }, # "bin_color": "forestgreen"},
    "Incorrect": {"zscore": [-2, 2], "auc": [-5, 5], "sig_y": 3.2 }, # "bin_color": "coral"}
}
# limits = {
#     "Initiated trials": {"zscore": [-1, 1], "auc": [-2, 2], "sig_y": 1.5},
#     "Prematures": {"zscore": [-1, 1], "auc": [-2, 2], "sig_y": 1.5},
#     "Reward pickup": {"zscore": [-1.5, 1.5], "auc": [-4, 4], "sig_y": 2.5},
#     "Omissions": {"zscore": [-1, 1], "auc": [-2, 2], "sig_y": 1.5},
#     "Correct": {"zscore": [-1, 1], "auc": [-2, 2], "sig_y": 1.5},
#     "Incorrect": {"zscore": [-1, 1], "auc": [-2, 2], "sig_y": 1.5}
# }

processed_data = dict()
excl = [('656.3', 'left'), ('656.3', 'right'),
        ('201.3', 'left'), ('201.3', 'right'),
        ('652.3', 'left'), ('652.3', 'right'),
        ('201.2', 'right')]

# y_label = "ΔF/F"
y_label = "z-score"

stats_df = pd.DataFrame()

for event in events:
    log(f"Processing {event}")
    processed_data[event] = {
        "nac": {
            "means": [],
            "aucs": {}
        },
        "bla": {
            "means": [],
            "aucs": {}
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
            min_event_count = 15
            if event_count < min_event_count:
                log(f"WARNING: subject {sub.name} has less than {min_event_count} events ({event_count}) -> skipping")
                continue

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

    total_bla_epochs = sum(len(rec['epochs']) for subject in data['bla'].values() for rec in subject)
    total_nac_epochs = sum(len(rec['epochs']) for subject in data['nac'].values() for rec in subject)
    log(f'{event} BLA events = {total_bla_epochs}')
    log(f'{event} NAc events = {total_nac_epochs}')

    windows = [(-10, -8), (-2, 0), (0, 2), (2, 4), (4, 6)]
    # windows = [(-10, -8), (-4, -2), (-2, 0), (0, 2), (2, 4)]
    # windows = [(-9, -8), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3)]

    nac_means, nac_aucs = hierarchical_bootstrap_group(data["nac"], pseudotime, n_boot=2000, windows=windows)
    nac_group_mean = np.mean(nac_means, axis=0)
    nac_sem = bootstrap_sem_trace(nac_means)
    nac_ci_lower = np.percentile(nac_means, 2.5, axis=0)
    nac_ci_upper = np.percentile(nac_means, 97.5, axis=0)

    processed_data[event]["nac"]["means"] = nac_means
    processed_data[event]["nac"]["aucs"] = nac_aucs

    bla_means, bla_aucs = hierarchical_bootstrap_group(data["bla"], pseudotime, n_boot=2000, windows=windows)
    bla_group_mean = np.mean(bla_means, axis=0)
    bla_sem = bootstrap_sem_trace(bla_means)
    bla_ci_lower = np.percentile(bla_means, 2.5, axis=0)
    bla_ci_upper = np.percentile(bla_means, 97.5, axis=0)

    processed_data[event]["bla"]["means"] = bla_means
    processed_data[event]["bla"]["aucs"] = bla_aucs

    # Export NAc AUCs to Excel
    nac_aucs_df = pd.DataFrame()
    for window in windows:
        window_label = f'{window[0]}—{window[1]}s'
        nac_aucs_df[window_label] = nac_aucs[window]
    nac_aucs_df.to_excel(join(out_dir, f'nac_aucs_{event}.xlsx'), index=False)

    # Export BLA AUCs to Excel
    bla_aucs_df = pd.DataFrame()
    for window in windows:
        window_label = f'{window[0]}—{window[1]}s'
        bla_aucs_df[window_label] = bla_aucs[window]
    bla_aucs_df.to_excel(join(out_dir, f'bla_aucs_{event}.xlsx'), index=False)

    mask = (pseudotime >= -5) & (pseudotime <= 10)

    df = pd.DataFrame({
        "time": pseudotime[mask],
        f"nac_mean": nac_group_mean[mask],
        f"nac_ci_upper": nac_ci_upper[mask],
        f"nac_ci_lower": nac_ci_lower[mask],
        f"bla_mean": bla_group_mean[mask],
        f"bla_ci_upper": bla_ci_upper[mask],
        f"bla_ci_lower": bla_ci_lower[mask],
    })

    df.to_excel(join(out_dir, f"bootstrapped_trace_{event}.xlsx"), index=False)

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])  # 2 rows, 2 cols

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(pseudotime[mask], nac_group_mean[mask], label=f"NAc (N={len(data['nac'])})", color="steelblue")
    ax1.fill_between(
        pseudotime[mask],
        nac_ci_lower[mask],
        nac_ci_upper[mask],
        # label="95% CI",
        alpha=0.3,
        color="steelblue",
    )
    ax1.plot(pseudotime[mask], bla_group_mean[mask], label=f"BLA (N={len(data['bla'])})", color="salmon")
    ax1.fill_between(
        pseudotime[mask],
        bla_ci_lower[mask],
        bla_ci_upper[mask],
        # label="95% CI",
        alpha=0.3,
        color="salmon",
    )
    ax1.set_ylabel(y_label)
    ax1.set_xlabel("Time (s)")
    ax1.set_xlim([-5, 10])
    ax1.set_xticks([-4, -2, 0, 2, 4, 6, 8, 10])
    ax1.set_ylim(limits[event]["zscore"])
    # ax1.set_yticks([-6, 0, 6])
    ax1.vlines(0, -10000, 10000, linestyles='dotted', color="#000")
    ax1.hlines(0, -10000, 10000, linestyle="dotted", color="#000", alpha=0.2)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 1.1), fontsize=7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    predefined_comparisons = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3), (3, 4), (0, 4)]
    # predefined_comparisons = [(0, 1), (1, 2), (2, 3), (3, 4)]
    # predefined_comparisons = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (2, 3), (3 , 4), (4, 5)]
    nac_aucs_results = compare_boot_bins(nac_aucs, windows, to_compare=predefined_comparisons)
    bla_aucs_results = compare_boot_bins(bla_aucs, windows, to_compare=predefined_comparisons)
    # results = compare_boot_bins(nac_aucs, windows, bla_aucs)

    # Collect NAc results in a list
    nac_results_list = []
    for i, (comp, raw_p, adj_p, rej, ci_lower, ci_upper) in enumerate(
            zip(nac_aucs_results["comparisons"], nac_aucs_results["p_vals"],
                nac_aucs_results["p_adj"], nac_aucs_results["reject"],
                nac_aucs_results["ci_lower"], nac_aucs_results["ci_upper"])):
        nac_results_list.append({
            'window1': f'{comp[0][0]}—{comp[0][1]}s',
            'window2': f'{comp[1][0]}—{comp[1][1]}s',
            'area': 'nac',
            'raw_p': raw_p,
            'p': adj_p,
            'reject': rej,
            'CI': f'[{ci_lower:.3f}, {ci_upper:.3f}]'
        })

    bla_results_list = []
    for i, (comp, raw_p, adj_p, rej, ci_lower, ci_upper) in enumerate(
            zip(bla_aucs_results["comparisons"], bla_aucs_results["p_vals"],
                bla_aucs_results["p_adj"], bla_aucs_results["reject"],
                bla_aucs_results["ci_lower"], bla_aucs_results["ci_upper"])):
        bla_results_list.append({
            'window1': f'{comp[0][0]}—{comp[0][1]}s',
            'window2': f'{comp[1][0]}—{comp[1][1]}s',
            'area': 'bla',
            'raw_p': raw_p,
            'p': adj_p,
            'reject': rej,
            'CI': f'[{ci_lower:.3f}, {ci_upper:.3f}]'
        })

    df_results = pd.DataFrame(bla_results_list + nac_results_list)
    excel_path = join(out_dir, f'5c_stats.xlsx')
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df_results.to_excel(writer, sheet_name=event, index=False)
    except FileNotFoundError:
        # File doesn't exist yet, create it
        df_results.to_excel(excel_path, sheet_name=event, index=False)

    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    # print('AUCS', results)
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

    #
    inner_labels = ["BLA"] * len(windows)
    outer_labels = [f'{start}—{end}s' for start, end in windows]
    group_centers = [1, 2, 3]  # centers of pairs

    print("-----BLA")
    to_draw_bla = []
    for idx, (comp, raw_p, adj_p, rej, ci_lower, ci_upper) in enumerate(
            zip(bla_aucs_results["comparisons"], bla_aucs_results["p_vals"], bla_aucs_results["p_adj"],
                bla_aucs_results["reject"], bla_aucs_results["ci_lower"], bla_aucs_results["ci_upper"])):
        print(
            f"(Within){comp}: raw p={raw_p:.3f}, adj p={adj_p:.3f}, reject={rej}, CI [{ci_lower:.3f}, {ci_upper:.3f}]")
        if rej:
            w1, w2 = comp
            x1, x2 = windows.index(w1), windows.index(w2)
            if x1 > 0:
                x1 += 1
            if x2 > 0:
                x2 += 1
            to_draw_bla.append((x1, x2, calc_stars(adj_p)))

    for idx, (x1, x2, stars) in enumerate(to_draw_bla):
        x1 += 1 * 1.05
        x2 += 1 * 0.95
        y = limits[event]["sig_y"]
        h = (y / 5) + (idx / y)
        ax2.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c='k')
        ax2.text((x1 + x2) * .5, y + (h * 4), stars, ha='center', va='top', color='k', fontsize=9)

    bla_aucs_arr = [bla_aucs[w] for w in windows]
    positions = [1, 3, 4, 5, 6]
    # positions = [1, 3, 4, 5]
    bplot_bla = ax2.boxplot(bla_aucs_arr, positions=positions, showfliers=False, showmeans=False,
                            whis=[2.5, 97.5],
                            medianprops=dict(color="black", linewidth=0.5),
                            tick_labels=inner_labels, patch_artist=True)

    # fill with colors
    for idx, patch in enumerate(bplot_bla['boxes']):
        alpha = 1 if idx > 0 else 0.5
        color = "salmon" if "bin_color" not in limits[event] else limits[event]["bin_color"]
        patch.set_facecolor(color)
        patch.set_alpha(alpha)

    # ax2.set_xticklabels(outer_labels)
    ax2.set_xticklabels(outer_labels, rotation=45, fontsize=7)
    ax2.set_ylabel("AUC")
    ax2.set_ylim(limits[event]["auc"])

    print("-----NAc")
    to_draw_nac = []
    for idx, (comp, raw_p, adj_p, rej, ci_lower, ci_upper) in enumerate(
            zip(nac_aucs_results["comparisons"], nac_aucs_results["p_vals"], nac_aucs_results["p_adj"],
                nac_aucs_results["reject"], nac_aucs_results["ci_lower"], nac_aucs_results["ci_upper"])):
        print(
            f"(Within){comp}: raw p={raw_p:.3f}, adj p={adj_p:.3f}, reject={rej}, CI [{ci_lower:.3f}, {ci_upper:.3f}]")
        if rej:
            w1, w2 = comp
            x1, x2 = windows.index(w1), windows.index(w2)
            if x1 > 0:
                x1 += 1
            if x2 > 0:
                x2 += 1
            to_draw_nac.append((x1, x2, calc_stars(adj_p)))

    for idx, (x1, x2, stars) in enumerate(to_draw_nac):
        x1 += 1 * 1.05
        x2 += 1 * 0.95
        y_offset = (idx / limits[event]["sig_y"]) * 2
        y = limits[event]["sig_y"] + y_offset
        h = y / 5
        ax3.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c='k')
        ax3.text((x1 + x2) * .5, y + (h * 4), stars, ha='center', va='top', color='k', fontsize=9)

    nac_aucs_arr = [nac_aucs[w] for w in windows]
    bplot_nac = ax3.boxplot(nac_aucs_arr, positions=positions, showfliers=False, showmeans=False,
                            whis=[2.5, 97.5],
                            medianprops=dict(color="black", linewidth=0.5),
                            tick_labels=inner_labels, patch_artist=True)
    # fill with colors
    for idx, patch in enumerate(bplot_nac['boxes']):
        alpha = 1 if idx > 0 else 0.5
        color = "steelblue" if "bin_color" not in limits[event] else limits[event]["bin_color"]
        patch.set_facecolor(color)
        patch.set_alpha(alpha)

    ax3.set_xticklabels(outer_labels, rotation=45, fontsize=7)
    ax3.get_yaxis().set_visible(False)
    ax3.set_ylim(limits[event]["auc"])

    fig.suptitle(f'{event}', fontsize=12)
    plt.tight_layout()
    pdf.savefig(fig)

# exit(0)
areas = ["bla", "nac"]
for area in areas:
    fig, (ax) = plt.subplots(1, figsize=(4, 2))

    prematures = processed_data["Prematures"][area]
    correct = processed_data["Correct"][area]
    incorrect = processed_data["Incorrect"][area]

    # w = [(-2, -1), (-1, 0), (0, 1), (1, 2)]
    w = [(-10, -8), (-2, 0)]
    prem_vs_incorrect = compare_boot_bins(prematures["aucs"], w, incorrect["aucs"])
    print(f"Prematures vs incorrect: {prem_vs_incorrect}")
    prem_vs_correct = compare_boot_bins(prematures["aucs"], w, correct["aucs"])
    print(f"Prematures vs correct: {prem_vs_correct}")

    prematures_mean = np.mean(prematures["means"], axis=0)
    prematures_sem = bootstrap_sem_trace(prematures["means"])
    correct_mean = np.mean(correct["means"], axis=0)
    correct_sem = bootstrap_sem_trace(correct["means"])
    incorrect_mean = np.mean(incorrect["means"], axis=0)
    incorrect_sem = bootstrap_sem_trace(incorrect["means"])

    mask = (pseudotime >= -5) & (pseudotime <= 10)

    ax.plot(pseudotime[mask], prematures_mean[mask], label=f"Premature", color='steelblue')
    ax.plot(pseudotime[mask], correct_mean[mask], label=f"Correct", color='forestgreen')
    ax.plot(pseudotime[mask], incorrect_mean[mask], label=f"Incorrect", color='coral')

    ax.set_ylabel(y_label)
    ax.set_xlabel("Time (s)")
    ax.set_xlim([-5, 10])
    ax.set_xticks([-4, -2, 0, 2, 4, 6, 8, 10])
    # ax.set_yticks([-4, -3, -2, -1, 0, 1, 2])
    if area == 'nac':
        ax.set_ylim([-5, 2])
    else:
        ax.set_ylim([-2, 2])
    # ax.set_ylim(limits[event]["zscore"])
    # ax1.set_yticks([-6, 0, 6])
    ax.vlines(0, -10000, 10000, linestyles='dotted', color="#000")
    ax.hlines(0, -10000, 10000, linestyle="dotted", color="#000", alpha=0.2)
    ax.legend(loc="upper left", fontsize=5)
    # ax.set_title(f'{area.upper()} overlays')
    ax.fill_between(
        pseudotime[mask],
        (prematures_mean - prematures_sem)[mask],
        (prematures_mean + prematures_sem)[mask],
        # label="95% CI",
        alpha=0.3,
        color="steelblue",
    )

    ax.fill_between(
        pseudotime[mask],
        (incorrect_mean - incorrect_sem)[mask],
        (incorrect_mean + incorrect_sem)[mask],
        # label="95% CI",
        alpha=0.3,
        color="coral",
    )

    ax.fill_between(
        pseudotime[mask],
        (correct_mean - correct_sem)[mask],
        (correct_mean + correct_sem)[mask],
        # label="95% CI",
        alpha=0.3,
        color="forestgreen",
    )

    # ax1.plot(prematures[""])

    fig.suptitle(f'{area} overlays', fontsize=12)
    pdf.savefig(fig)

plt.subplots_adjust(wspace=0.3, hspace=0.8)
pdf.close()
