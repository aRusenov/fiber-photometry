import os
import re
import sys

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

# Add the parent for improt
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from lib import calculate_transients, read_preprocessed_data, standard_cli_argparse, log

MAX_TRIALS = 20


def find_shock_ranges(arr):
    ranges = []
    start = None

    for i, val in enumerate(arr):
        if val == 1:
            if start is None:
                start = i  # start of a new run
        else:
            if start is not None:
                ranges.append((start, i - 1))
                start = None

    # Handle case where array ends with a run of 1s
    if start is not None:
        ranges.append((start, len(arr) - 1))

    return ranges[:MAX_TRIALS]


parser = standard_cli_argparse("FP foot shock")
parser.add_argument("--dio", help="Source DIO for shock activity", required=True)
args = parser.parse_args()

files = []
for file in args.file:
    for line in file.splitlines():
        files.append(line)

pdf = PdfPages("output.pdf")
subject_label = args.label

for file in files:
    filename = os.path.basename(file)
    log(f"File {file}")
    name = re.search(r"(\d+_\d)", filename).group(0)
    log(
        f"Processing {name} ({subject_label}), reading licking activity from {args.dio}"
    )

    data = read_preprocessed_data(file)
    shock_dio = data.dios[args.dio]
    blocks = find_shock_ranges(shock_dio)

    log("Calculating activity averages (dff & z-score)")
    time_before = 25
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

    # Plot only a subset of the transient bins
    time_before_plot = 5
    time_after_plot = 10
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
    ax1.set_xlim([-time_before_plot, time_after_plot])
    ax1.set_ylim([-1.0, 1.0])
    ax1.set_title(f"{name} {subject_label} ({len(blocks)} trials)")
    ax1.legend(loc="upper right")
    ax1.vlines(0, -10000, 10000, linestyle="dotted", color="#000")

    ax2.plot(pseudotime_x, mean_zscore)
    ax2.set_xlabel("Shock onset (seconds)")
    ax2.set_ylabel("z-score")
    ax2.set_ylim([-2.0, 2.0])
    ax2.set_xlim([-time_before_plot, time_after_plot])
    ax2.hlines(0, -10000, 10000, linestyle="dashed", color="#000", alpha=0.2)
    ax2.vlines(0, -10000, 10000, linestyle="dotted", color="#000")

    plt.subplots_adjust(hspace=0.1)
    pdf.savefig(fig)

pdf.close()
