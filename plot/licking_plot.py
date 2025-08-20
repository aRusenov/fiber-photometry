import os
import re
import sys
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np

# Add the parent for import
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from lib import calculate_transients, read_preprocessed_data, log, standard_cli_argparse


def calculate_blocks(arr, merge_threshold_sec, sampling_rate):
    # Step 1: Detect changes
    diff = np.diff(arr)
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]

    # Step 2: Handle edge cases
    if arr[0] == 1:
        starts = np.insert(starts, 0, 0)
    if arr[-1] == 1:
        ends = np.append(ends, len(arr) - 1)

    # Step 3: Zip and merge based on proximity
    raw_ranges = list(zip(starts, ends))
    merged_ranges = []

    threshold = merge_threshold_sec * sampling_rate
    for current in raw_ranges:
        if not merged_ranges:
            merged_ranges.append(current)
        else:
            last_start, last_end = merged_ranges[-1]
            current_start, current_end = current
            if current_start - last_end < threshold:
                merged_ranges[-1] = (last_start, current_end)
            else:
                merged_ranges.append(current)

    # keep only ranges where combined duration >= 1 second
    arr = np.array(merged_ranges)
    diff = arr[:, 1] - arr[:, 0]
    filtered_arr = arr[(diff / sampling_rate) >= 1]
    # print(filtered_arr)
    return filtered_arr


parser = standard_cli_argparse("FP fat licking plot transients")
parser.add_argument("--dio", help="Source DIO for licking activity", required=True)
args = parser.parse_args()

files = []
for file in args.file:
    for line in file.splitlines():
        files.append(line)

pdf = PdfPages("output.pdf")
subject_label = args.label
for file in files:
    filename = os.path.basename(file)
    log(f"file {file}")
    name = re.search(r"(\d+_\d)", filename).group(0)
    log(
        f"Processing {name} ({subject_label}), reading licking activity from {args.dio}"
    )

    data = read_preprocessed_data(file)
    licks = data.dios[args.dio]
    blocks = calculate_blocks(licks, merge_threshold_sec=1, sampling_rate=data.sampling_rate)

    log("Calculating activity averages (dff & z-score)")
    time_before = 5
    time_after = 10
    signal_corrected_transients = calculate_transients(
        data.signal_corrected_dff, data.time, blocks, time_before, time_after, "dff"
    )
    signal_transients = calculate_transients(
        data.signal_dff, data.time, blocks, time_before, time_after, "dff"
    )
    control_transients = calculate_transients(
        data.control_dff, data.time, blocks, time_before, time_after, "dff"
    )

    signal_corrected_transients_zscore = calculate_transients(
        data.signal_corrected_dff, data.time, blocks, time_before, time_after, "z"
    )

    log("Plotting")
    transients = [
        (signal_transients, "GCaMP"),
        (signal_corrected_transients, "GCaMP corrected"),
        (control_transients, "Isosbestic"),
    ]

    mean_zscore = np.mean(signal_corrected_transients_zscore, axis=0)
    pseudotime_x = np.linspace(-time_before, time_after, num=len(mean_zscore))

    fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={"height_ratios": [3, 1]})
    for transient, label in transients:
        mean_y = np.mean(transient, axis=0)
        stderr_y = np.std(transient, axis=0, ddof=1) / np.sqrt(
            np.size(transient, axis=0)
        )
        ax1.plot(pseudotime_x, mean_y, label=label)
        ax1.fill_between(
            pseudotime_x,
            mean_y - stderr_y,
            mean_y + stderr_y,
            alpha=0.5,
            edgecolor="#42b0f5",
            facecolor="#42c8f5",
        )

    ax1.set_ylabel("ΔF/F")
    ax1.set_xticks([])
    ax1.set_xlim([-time_before, time_after])
    ax1.set_ylim([-1.0, 1.0])
    ax1.set_title(f"{name} {subject_label} ({len(blocks)} trials)")
    ax1.legend(loc="upper right")
    ax1.vlines(0, -10000, 10000, linestyle="dotted", color="#000")

    ax2.plot(pseudotime_x, mean_zscore)
    ax2.set_xlabel("Licking onset (seconds)")
    ax2.set_ylabel("z-score")
    ax2.set_ylim([-2.0, 2.0])
    ax2.set_xlim([-time_before, time_after])
    ax2.hlines(0, -10000, 10000, linestyle="dashed", color="#000", alpha=0.2)
    ax2.vlines(0, -10000, 10000, linestyle="dotted", color="#000")

    plt.subplots_adjust(hspace=0.1)
    pdf.savefig(fig)

pdf.close()
