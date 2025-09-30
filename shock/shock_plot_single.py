import os
import re
import sys
import h5py
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.stats import ttest_rel

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

events = [
    "Onset",
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
        # FIXME: read from h5py
        [name, hemisphere] = re.search(r"(\d+_\d)-(\w+)-", filename).groups()
        return Processed5CData(name=name, label=hemisphere, activities=activities, sampling_rate=f['Meta/Sampling_Rate'])


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

def get_ylim(signal):
    signal_max = np.max(np.abs(signal))
    if signal_max < 0.1:
        lim = 0.1
    else:
        lim = np.ceil(signal_max * 10) / 10

    lim *= 1.1

    return [-lim, lim]

experiment = 'foot-shock'
indir = "/Users/atanas/Documents/workspace/data/analysis/photometry/shock/processed"
out_dir = '/Users/atanas/Documents/workspace/data/analysis/photometry/shock'

def baseline_correct(traces, pseudotime, baseline_window=(-5, -3)):
    """Subtract mean baseline per epoch."""
    mask = (pseudotime >= baseline_window[0]) & (pseudotime <= baseline_window[1])
    corrected = traces - traces[:, mask].mean(axis=1, keepdims=True)
    return corrected

def create_text_page():
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')

    font = FontProperties()
    font.set_size(14)

    plt.text(0.1, 0.9, 'Right hemisphere', fontproperties=font, fontweight='bold')
    plt.text(0.1, 0.85, f'Generated on: {os.path.basename(__file__)}', fontproperties=font)
    plt.text(0.1, 0.80, f'Analysis date: {os.path.getctime(__file__)}', fontproperties=font)

    return fig

def calc_stars(p_val) -> str:
    if p_val < 0.001:
        stars = '***'
    elif p_val < 0.01:
        stars = '**'
    elif p_val < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'
    return stars

files = [f for f in listdir(indir) if isfile(join(indir, f)) and f.endswith(".h2py")]

pdf = PdfPages(join(out_dir, f"single-plots-{experiment}.pdf"))
# pdf.savefig(create_text_page())


log(f"Files {files}")

subjects: list[Processed5CData] = []
for file in files:
    data = read_processed_data(join(indir, file))
    subjects.append(data)

for sub in subjects:
    log(f"Subject {sub.name} {sub.label}")
    mouse_id = sub.name.replace("_", ".")
    for event in events:
        activity = next((a for a in sub.activities if a.event == event), None)
        time_before_plot = 5
        time_after_plot = 10
        sampling_rate = 240 # FIXME: read from individual file

        fig, (ax1, ax2, ax3) = plt.subplots(3)
        plt.subplots_adjust(hspace=0.8)
        pseudotime = np.linspace(-time_before_plot, time_after_plot, num=len(activity.channels[0].bins_dff[0]))

        signal_baseline_corrected_bins = baseline_correct(activity.signal().bins_dff, pseudotime)
        signal_mean = np.mean(signal_baseline_corrected_bins, axis=0)
        signal_lower_ci, signal_upper_ci = bootstrap_ci(signal_baseline_corrected_bins)

        control_baseline_corrected_bins = baseline_correct(activity.control().bins_dff, pseudotime)
        control_mean = np.mean(control_baseline_corrected_bins, axis=0)
        control_lower_ci, control_upper_ci = bootstrap_ci(control_baseline_corrected_bins)

        ax1.plot(pseudotime, signal_mean, label="GCaMP", color='g')
        ax1.plot(pseudotime, control_mean, label="Isosbestic", color='m')
        ax1.fill_between(pseudotime, signal_lower_ci, signal_upper_ci, alpha=0.1, color='g')
        ax1.fill_between(pseudotime, control_lower_ci, control_upper_ci, alpha=0.1, color='m')
        ax1.set_ylabel("ΔF/F")
        ax1.set_xlim([-time_before_plot, time_after_plot])
        ax1.set_ylim(get_ylim(signal_mean))
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_title('GCaMP vs Isosbestic')

        signal_corr_baseline_corrected_bins = baseline_correct(activity.signal_corr().bins_dff, pseudotime)
        signal_corr_mean = np.mean(signal_corr_baseline_corrected_bins, axis=0)
        signal_corr_lower_ci, signal_corr_upper_ci = bootstrap_ci(signal_corr_baseline_corrected_bins)

        ax2.plot(pseudotime, signal_corr_mean, label='GCaMP dff', color='y')
        ax2.fill_between(pseudotime, signal_corr_lower_ci, signal_corr_upper_ci, alpha=0.1, color='y')
        ax2.set_ylabel("ΔF/F")
        ax2.set_title('GCaMP corrected')
        ax2.set_xlim([-time_before_plot, time_after_plot])
        ax2.set_ylim(get_ylim(signal_corr_mean))
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        signalz_corr_baseline_corrected_bins = baseline_correct(activity.signal_corr().bins_zscore, pseudotime)
        signalz_corr_mean = np.mean(signalz_corr_baseline_corrected_bins, axis=0)
        signalz_corr_lower_ci, signalz_corr_upper_ci = bootstrap_ci(signalz_corr_baseline_corrected_bins)

        _ax2 = ax2.twinx()
        color = 'tab:blue'
        _ax2.set_ylabel('z-score', color=color)
        _ax2.set_ylim(get_ylim(signalz_corr_mean))
        _ax2.plot(pseudotime, signalz_corr_mean, color=color)
        _ax2.tick_params(axis='y', labelcolor=color)
        _ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        _ax2.fill_between(pseudotime, signalz_corr_lower_ci, signalz_corr_upper_ci, alpha=0.1, color=color)

        steps = [(-4, -2), (-2, 0), (0, 2), (2, 4)]
        auc = []
        auc_labels = []
        for step in steps:
            t_min, t_max = step
            auc_value = [compute_auc(pseudotime, trace, t_min, t_max) for trace in signalz_corr_baseline_corrected_bins]
            if len(auc) > 0:
                prev = auc[-1]
                t_stat, p_val = ttest_rel(auc_value, prev)
                print(f"t-stat={t_stat}, p-val={p_val}")
                # print(f"X1={prev}, X2={auc_value}")
                if p_val < 0.05:
                    x1, x2 = len(auc), len(auc) + 1
                    y, h = 0.2, 0.2  # height for the line
                    stars = calc_stars(p_val)
                    print(f"{stars}... {t_min}—{t_max}s")
                    _ax2.plot([t_min + (t_max - t_min) / 2, t_min + (t_max - t_min) / 2], [y, y + h], lw=1.5, c='k')
                    _ax2.text((t_min + t_max) * .5, y + h + 0.05, stars, ha='center', va='bottom', fontsize=8)

            auc.append(auc_value)
            auc_labels.append(f'{t_min}—{t_max}s')

        activity_data = np.vstack(activity.signal_corr().bins_dff)
        im = ax3.imshow(
            activity_data,
            aspect='auto',         # stretch to fill axis
            origin='lower',        # trial 0 at bottom
            extent=[-time_before_plot, time_after_plot, 0, activity_data.shape[0]],  # map x to time, y to trials
            cmap='viridis'         # colormap
        )
        # im = ax3.imshow(activity_data, aspect='auto', cmap='viridis',
        #                 extent=[-time_before_plot, time_after_plot, activity_data.shape[0], 0])
        fig.colorbar(im, ax=ax3, label='ΔF/F')
        ax3.set_title('Trial heatmap')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Trial')

        fig.suptitle(f"{mouse_id} {sub.label}. {event} (N={len(activity.signal_corr().bins_dff)})")
        pdf.savefig(fig)
        plt.close(fig)


# for event in events:
#
#     for sub in subjects:
#         activity = next((a for a in sub.activities if a.event == event), None)
#         mouse_id = sub.name.replace("_", ".")
#         group = None
#         if mouse_id in groups['nac']:
#             group = 'nac'
#         elif mouse_id in groups['bla']:
#             group = 'bla'
#
#         if group is None:
#             log(f"WARNING: subject {sub.name} does not belong in any group -> skipping")
#             continue
#
#         data[group]["signal_dff"].append(
#             compute_mean(activity.signal_corr().bins_dff)
#         )
#         data[group]["signal_z"].append(
#             compute_mean(activity.signal_corr().bins_zscore)
#         )
#
#     log(f"Plotting {event}")
#     time_before_plot = 5
#     time_after_plot = 10
#     baseline_correction_from = 2
#     sampling_rate = 240  # FIXME: read from individual files, handle padding when discrepancies exist?
#
#     fig, axs = plt.subplots(3, 2, gridspec_kw={"width_ratios": [1, 1]})
#
#     for label in ['bla', 'nac']:
#         means_dff = pad(data[label]["signal_dff"])
#
#         baseline_window = baseline_correction_from * sampling_rate
#         signal_dff_group_mean = np.mean(means_dff, axis=0)
#         # FIXME: weighted mean
#         signal_dff_baseline = np.mean(signal_dff_group_mean[:baseline_correction_from])
#         signal_dff_corrected_group_mean = np.subtract(
#             signal_dff_group_mean, signal_dff_baseline
#         )
#
#         dff_lower_ci, dff_upper_ci = bootstrap_ci(means_dff)
#
#         ax_col = 0 if label == 'bla' else 1
#         pseudotime = np.linspace(
#             -time_before_plot,
#             time_after_plot,
#             num=len(means_dff[0]),
#         )
#         axs[0, ax_col].plot(pseudotime, signal_dff_corrected_group_mean, color="#076e18")
#         axs[0, ax_col].fill_between(
#             pseudotime,
#             dff_lower_ci - signal_dff_baseline,
#             dff_upper_ci - signal_dff_baseline,
#             alpha=0.5,
#             edgecolor="#195423",
#             facecolor="#35c44d",
#         )
#         axs[0, ax_col].set_ylabel("ΔF/F")
#         axs[0, ax_col].set_xticks([])
#         axs[0, ax_col].set_xlim([-time_before_plot, time_after_plot])
#         axs[0, ax_col].set_ylim([-1.0, 1.0])
#         axs[0, ax_col].set_title(f"{label} (N={len(means_dff)})")
#         axs[0, ax_col].hlines(0, -10000, 10000, linestyle="dashed", color="#000", alpha=0.2)
#         axs[0, ax_col].vlines(0, -10000, 10000, linestyle="dotted", color="#000")
#         # axs[0, ax_col].legend(loc="upper right")
#
#         # z-score
#         means_zscore = pad(data[label]["signal_z"])
#         signal_z_group_mean = np.mean(means_zscore, axis=0)
#         signal_z_baseline = np.mean(signal_z_group_mean[:baseline_correction_from])
#         signal_z_corrected_group_mean = np.subtract(
#             signal_z_group_mean, signal_z_baseline
#         )
#
#         zscore_lower_ci, zscore_upper_ci = bootstrap_ci(means_zscore)
#
#         axs[1, ax_col].plot(pseudotime, signal_z_corrected_group_mean, color="#ad30e3")
#         axs[1, ax_col].fill_between(
#             pseudotime,
#             zscore_lower_ci - signal_z_baseline,
#             zscore_upper_ci - signal_z_baseline,
#             alpha=0.5,
#             edgecolor="#9a48bd",
#             facecolor="#cd7df0",
#         )
#         axs[1, ax_col].set_ylabel("z-score")
#         axs[1, ax_col].set_xticks([-5, 0, 5, 10])
#         axs[1, ax_col].set_xlim([-time_before_plot, time_after_plot])
#         axs[1, ax_col].set_ylim([-1.0, 1.0])
#         # axs[1, ax_col].set_title(f"{label} (N={len(means_zscore)})")
#         axs[1, ax_col].hlines(0, -10000, 10000, linestyle="dashed", color="#000", alpha=0.2)
#         axs[1, ax_col].vlines(0, -10000, 10000, linestyle="dotted", color="#000")
#         # axs[1, ax_col].legend(loc="upper right")
#
#         # AUC
#         steps = [(-4, -2), (-2, 0), (0, 2), (2, 4), (4, 6)]
#         auc = []
#         auc_labels = []
#         for step in steps:
#             t_min, t_max = step
#             auc_value = [compute_auc(pseudotime, trace, t_min, t_max) for trace in means_dff]
#             if len(auc) > 0:
#                 prev = auc[-1]
#                 t_stat, p_val = ttest_rel(auc_value, prev)
#                 print(f"t-stat={t_stat}, p-val={p_val}")
#                 if p_val < 0.05:
#                     x1, x2 = len(auc), len(auc) + 1
#                     y, h = max(max(prev), max(auc_value)) + 0.5, 0.2  # height for the line
#                     stars = calc_stars(p_val)
#                     print(f"{stars}... {t_min}—{t_max}s")
#                     axs[2, ax_col].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
#                     axs[2, ax_col].text((x1 + x2) * .5, y + h + 0.05, stars, ha='center', va='bottom', fontsize=8)
#
#             auc.append(auc_value)
#             auc_labels.append(f'{t_min}—{t_max}s')
#
#         axs[2, ax_col].boxplot(auc, showfliers=False, showmeans=True, bootstrap=2000)
#         axs[2, ax_col].set_ylabel("AUC")
#         axs[2, ax_col].set_xticklabels(auc_labels, rotation=45, fontsize=8)
#         axs[2, ax_col].set_ylim([-3.0, 3.0])

    fig.suptitle(event)
    # fig.set_size_inches(8, 3)

plt.subplots_adjust(wspace=0.3, hspace=0.8)
pdf.close()
