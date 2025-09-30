import os
import sys
from os import listdir
from os.path import isfile, join

import numpy as np
from h5py.h5ds import iterate
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from common.plot import load_groups, read_processed_data, hierarchical_bootstrap, compute_boot_auc, compare_boot_bins, \
    calc_stars, hierarchical_boot_between_groups

# Add the parent for import
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
from common.lib import (
    Processed5CData,
    log,
)

events = [
    "Onset",
    "Outset"
]
experiment = 'fat-licking'
indir = "/Users/atanas/Documents/workspace/data/analysis/photometry/fat/processed"
out_dir = '/Users/atanas/Documents/workspace/data/analysis/photometry/fat'

groups = load_groups("../mice.csv")
print(groups)

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

hemispheres = [
    ("left", subjects_left),
    ("right", subjects_right)
]

for hemisphere, subjects in hemispheres:
    for event in events:
        data = {
            "nac": {"subjects": [], "dff": [], "zscore": []},
            "bla": {"subjects": [], "dff": [], "zscore": []}
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
            data[group]["subjects"].append(
                {"sub": mouse_id, "dff": activity.signal_corr().bins_dff, "zscore": activity.signal_corr().bins_zscore})

        log(f"Plotting {event}")
        time_before_plot = 5
        time_after_plot = 10
        sampling_rate = 240  # FIXME: read from individual files, handle padding when discrepancies exist?

        fig, axs = plt.subplots(3, 2, gridspec_kw={"width_ratios": [1, 1]})
        plt.subplots_adjust(wspace=0.3, hspace=0.3)


        for label in ['bla', 'nac']:

            ax_col = 0 if label == 'bla' else 1

            subs = data[label]["subjects"]
            pseudotime = np.linspace(
                -time_before_plot,
                time_after_plot,
                num=len(subs[0]["zscore"][0]),
            )


            subject_dff_traces = {s['sub']: (s['dff'], pseudotime) for s in subs}
            means_dff = hierarchical_bootstrap(subject_dff_traces, n_boot=200)
            data[label]["dff"] = means_dff
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
            data[label]["zscore"] = means_zscore
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
            axs[1, ax_col].set_ylim([-1.0, 1.0])
            # axs[1, ax_col].set_title(f"{label} (N={len(means_zscore)})")
            axs[1, ax_col].hlines(0, -10000, 10000, linestyle="dashed", color="#000", alpha=0.2)
            axs[1, ax_col].vlines(0, -10000, 10000, linestyle="dotted", color="#000")
            axs[1, ax_col].legend(loc="upper right", fontsize=5)

            # AUC
            steps = [(-4, -2), (-2, 0), (0, 2), (2, 4)]
            # steps = [(-4, -3), (-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3), (3, 4)]
            # steps = [(-3, -1), (-1, 1), (1, 3)]
            auc = []
            auc_labels = []
            # steps = [(-5, -2.5), (-2.5, 0), (0, 2.5), (2.5, 5)]
            auc = []
            auc_labels = []

            aucs = compute_boot_auc(means_zscore, pseudotime, windows=steps)
            results = compare_boot_bins(aucs, windows=steps)

            # y_max = 1.2
            # for idx, (comp, raw_p, adj_p, rej) in enumerate(
            #         zip(results["comparisons"], results["p_vals"], results["p_adj"], results["reject"])):
            #     print(f"{comp[0]} vs {comp[1]}: raw p={raw_p:.3f}, adj p={adj_p:.3f}, reject={rej}")
            #     if rej:
            #         x1, x2 = (idx + 1) * 1.05, (idx + 2) * 0.95
            #
            #         y, h, col = y_max + 0.05, 0.05, 'k'
            #         # x1, x2 = len(auc), len(auc) + 1
            #         # y, h = max(max(prev), max(auc_value)) + 0.5, 0.2  # height for the line
            #         stars = calc_stars(adj_p)
            #         # print(f"{stars}... {window_prev}—{window_current}s")
            #         axs[2, ax_col].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
            #         axs[2, ax_col].text((x1 + x2) * .5, y + h, stars, ha='center', va='bottom', color=col)

            for (start, end), _ in aucs.items():
                auc_labels.append(f'{start}—{end}s')

            # for idx, (window_prev, window_current) in enumerate(zip(steps, steps[1:])):
            #     _, p_val = compare_boot_auc(aucs, window_prev, window_current)
            #     # print(f"{window_prev} x {window_current}: p-val={p_val}")
            #     if p_val < 0.05:
            #         x1, x2 = (idx + 1) * 1.05, (idx + 2) * 0.95
            #
            #         y, h, col = y_max + 0.05, 0.05, 'k'
            #         # x1, x2 = len(auc), len(auc) + 1
            #         # y, h = max(max(prev), max(auc_value)) + 0.5, 0.2  # height for the line
            #         stars = calc_stars(p_val)
            #         print(f"{stars}... {window_prev}—{window_current}s")
            #         axs[2, ax_col].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            #         axs[2, ax_col].text((x1+x2)*.5, y+h, stars, ha='center', va='bottom', color=col)

            axs[2, ax_col].boxplot([auc for _, auc in aucs.items()], showfliers=False, showmeans=False)
            axs[2, ax_col].set_ylabel("AUC")
            axs[2, ax_col].set_xticklabels(auc_labels, rotation=45, fontsize=8)
            axs[2, ax_col].set_ylim([-2, 2])

        nac_zscore_means = data["nac"]["zscore"]
        bla_zscore_means = data["bla"]["zscore"]

        pseudotime = np.linspace(
            -time_before_plot,
            time_after_plot,
            num=3600,
        )

        windows = [(-2, 0), (0,2), (2,4)]

        nac_zscores = dict()
        for sub in data["nac"]["subjects"]:
            nac_zscores[sub["sub"]] = sub["zscore"]

        bla_zscores = dict()
        for sub in data["bla"]["subjects"]:
            bla_zscores[sub["sub"]] = sub["zscore"]

        for window in windows:
            diff, p, ci = hierarchical_boot_between_groups(nac_zscores, bla_zscores, pseudotime, window, n_boot=500)
            print(f'Diff={np.mean(diff)} p={p}, ci={ci}')
            if p < 0.05:
                print('Signficant!')

        fig.suptitle(f'{event} ({hemisphere})', fontsize=12)
        pdf.savefig(fig)

plt.subplots_adjust(wspace=0.3, hspace=0.8)
pdf.close()
