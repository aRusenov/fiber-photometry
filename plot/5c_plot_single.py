import os
import re
import sys
import h5py
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np

# Add the parent for import
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from lib import (
    Activity,
    Channel,
    ChannelType,
    Processed5CData,
    compute_mean,
    compute_stderr,
    printname,
    standard_cli_argparse,
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
        f.visit(printname)
        events = f["Event"]
        for event in events.keys():
            activity = Activity(event=event, channels=[])
            for channel in f["Event"][event].keys():
                activity.channels.append(
                    Channel(
                        name=ChannelType[channel],
                        bins_dff=f["Event"][event][channel]["Bins"]["Dff"][:],
                        bins_zscore=f["Event"][event][channel]["Bins"]["Zscore"][:],
                    )
                )

            activities.append(activity)

        filename = os.path.basename(file)
        name = re.search(r"(\d+_\d)", filename).group(0)
        return Processed5CData(name=name, label="TODO", activities=activities)

parser = standard_cli_argparse("5C plot")

args = parser.parse_args()

files = []
for file in args.file:
    for line in file.splitlines():
        files.append(line)

subject_label = args.label
# log(f"Action {args.action}")

pdf = PdfPages("output.pdf")

# if (args.action == 'avg'):

# else:
subjects: list[Processed5CData] = []
for file in files:
    data = read_processed_data(file)
    subjects.append(data)

time_before_plot = 5
time_after_plot = 10

print(len(subjects[0].activities[0].channels[0].bins_dff))

for subject in subjects:
    for activity in subject.activities:

        fig, (ax1, ax2, ax3) = plt.subplots(3, gridspec_kw={"height_ratios": [2, 2, 1]})

        pseudotime = np.linspace(-time_before_plot, time_after_plot, num=len(activity.channels[0].bins_dff[0]))


        signal_dff_mean = compute_mean(activity.signal().bins_dff)
        signal_dff_se = compute_stderr(activity.signal().bins_dff)
        
        control_dff_mean = compute_mean(activity.control().bins_dff)
        control_dff_se = compute_stderr(activity.control().bins_dff)
        
        signal_corr_dff_mean = compute_mean(activity.signal_corr().bins_dff)
        signal_corr_dff_se = compute_stderr(activity.signal_corr().bins_dff)
        
        signal_corr_zscore_mean = compute_mean(activity.signal_corr().bins_zscore)

        # signal_dff_group = np.array([subject[key].signal_dff_mean for subject in subjects])
        # signal_dff_mean = np.mean(signal_dff_group, axis=0)
        # signal_dff_se = np.std(signal_dff_group, axis=0, ddof=1) / np.sqrt(
        #     np.size(signal_dff_group, axis=0)
        # )

        # control_dff_group = np.array(
        #     [subject[key].control_dff_mean for subject in subjects]
        # )
        # control_dff_mean = np.mean(control_dff_group, axis=0)
        # control_dff_se = np.std(control_dff_group, axis=0, ddof=1) / np.sqrt(
        #     np.size(control_dff_group, axis=0)
        # )

        # signal_dff_corrected_group = np.array(
        #     [subject[key].signal_corrected_dff_mean for subject in subjects]
        # )
        # signal_dff_corrected_mean = np.mean(signal_dff_corrected_group, axis=0)
        # signal_dff_corrected_se = np.std(
        #     signal_dff_corrected_group, axis=0, ddof=1
        # ) / np.sqrt(np.size(signal_dff_corrected_group, axis=0))

        ax1.plot(pseudotime, signal_dff_mean, label="GCaMP", color="#076e18")
        ax1.fill_between(
            pseudotime,
            signal_dff_mean - signal_corr_dff_se,
            signal_dff_mean + signal_corr_dff_se,
            alpha=0.5,
            edgecolor="#195423",
            facecolor="#35c44d",
        )

        ax1.plot(pseudotime, control_dff_mean, label="Isosbestic", color="#ad30e3")
        ax1.fill_between(
            pseudotime,
            control_dff_mean - control_dff_se,
            control_dff_mean + control_dff_se,
            alpha=0.5,
            edgecolor="#9a48bd",
            facecolor="#cd7df0",
        )

        ax1.set_ylabel("ΔF/F")
        ax1.set_xticks([])
        ax1.set_xlim([-time_before_plot, time_after_plot])
        ax1.set_ylim([-1.0, 1.0])
        ax1.set_title(f"{subject.name} (events={len(activity.channels[0].bins_dff)})")
        ax1.legend(loc="upper right")
        ax1.hlines(0, -10000, 10000, linestyle="dashed", color="#000", alpha=0.2)
        ax1.vlines(0, -10000, 10000, linestyle="dotted", color="#000")

        ax2.plot(pseudotime, signal_corr_dff_mean, label="GCaMP Corrected")
        ax2.fill_between(
            pseudotime,
            signal_corr_dff_mean - signal_corr_dff_se,
            signal_corr_dff_mean + signal_corr_dff_se,
            alpha=0.5,
            edgecolor="#42b0f5",
            facecolor="#42c8f5",
        )

        ax2.set_ylabel("ΔF/F")
        ax2.set_xticks([])
        ax2.set_xlim([-time_before_plot, time_after_plot])
        ax2.set_ylim([-1.0, 1.0])
        ax2.legend(loc="upper right")
        ax2.hlines(0, -10000, 10000, linestyle="dashed", color="#000", alpha=0.2)
        ax2.vlines(0, -10000, 10000, linestyle="dotted", color="#000")

        # signal_corrected_z_group = np.array(
        #     [subject[key].signal_corrected_z for subject in subjects]
        # )
        # mean_zscore = np.mean(signal_corrected_z_group, axis=0)
        ax3.plot(pseudotime, signal_corr_zscore_mean)
        ax3.set_xlabel(f"{activity.event} onset (seconds)")
        ax3.set_ylabel("z-score")
        ax3.set_ylim([-2.0, 2.0])
        ax3.set_xlim([-time_before_plot, time_after_plot])
        ax3.hlines(0, -10000, 10000, linestyle="dashed", color="#000", alpha=0.2)
        ax3.vlines(0, -10000, 10000, linestyle="dotted", color="#000")

        plt.subplots_adjust(hspace=0.1)
        pdf.savefig(fig)

pdf.close()
