import argparse
import os
from lib import (
    log,
    read_doric_file,
    run_preprocessing_pipeline,
    save_preprocessed_data,
)
import re

parser = argparse.ArgumentParser("FP pre-processing")
parser.add_argument("--file", nargs="+", help="Input file(s)", required=True)
parser.add_argument(
    "--channel",
    help="The analog channel for signal and isosbestic channels (e.g. AIN01)",
    required=True,
)
parser.add_argument(
    "--dio", nargs="*", help="Additional DIO doric channels to preserve (e.g. DIO01)"
)
parser.add_argument(
    "--label", help="Label"
)
parser.add_argument(
    "--outdir", help="Output directory (optional). Defaults to /preprocessed in the base dir of each file"
)
args = parser.parse_args()

files = []
for file in args.file:
    for line in file.splitlines():
        files.append(line)
        
channel = args.channel
for file in files:
    basedir = os.path.dirname(file)
    filename = os.path.basename(file)

    # pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")

    # activity_dio = "DIO01"
    # files = os.listdir(basedir)
    # doric_files = [file for file in files if file.endswith("102_1_Det12_DIO1_0003.doric")]
    # for file in doric_files:
    # for hemisphere in ["L"]:
    label = args.label or channel
    name = re.search(r"(\d+_\d)", filename).group(0)
    log(f"Processing {name} ({label})")

    # if hemisphere == "L":
    #     channel = "AIN01"
    # else:
    #     channel = "AIN02"

    # filepath = os.path.join(basedir, file)
    time, control, signal, dios, sampling_rate = read_doric_file(
        file, channel, args.dio
    )

    data = run_preprocessing_pipeline(signal, control, time, dios)


    outfilename = f"{name}-{label}.h2py"
    subdir = args.outdir or os.path.join(basedir, "preprocessed")
    outfilepath = os.path.join(subdir, outfilename)
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    save_preprocessed_data(data, outpath=outfilepath)
    log(f'Saved to {outfilepath}')

    # log("Calculating activity averages (dff & z-score)")
    # time_before = 5
    # time_after = 10
    # signal_corrected_transients = calculate_transients(
    #     signal_corrected_dff, time, blocks, time_before, time_after, "dff"
    # )
    # signal_transients = calculate_transients(
    #     signal_dff, time, blocks, time_before, time_after, "dff"
    # )
    # control_transients = calculate_transients(
    #     control_dff, time, blocks, time_before, time_after, "dff"
    # )

    # signal_corrected_transients_zscore = calculate_transients(
    #     signal_corrected_dff, time, blocks, time_before, time_after, "z"
    # )

    # log("Plotting")
    # transients = [
    #     (signal_transients, "GCaMP"),
    #     (signal_corrected_transients, "GCaMP corrected"),
    #     (control_transients, "Isosbestic"),
    # ]

    # mean_zscore = np.mean(signal_corrected_transients_zscore, axis=0)
    # pseudotime_x = np.linspace(-time_before, time_after, num=len(mean_zscore))

    # fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={"height_ratios": [3, 1]})
    # for transient, label in transients:
    #     mean_y = np.mean(transient, axis=0)
    #     stderr_y = np.std(transient, axis=0, ddof=1) / np.sqrt(
    #         np.size(transient, axis=0)
    #     )
    #     ax1.plot(pseudotime_x, mean_y, label=label)
    #     ax1.fill_between(
    #         pseudotime_x,
    #         mean_y - stderr_y,
    #         mean_y + stderr_y,
    #         alpha=0.5,
    #         edgecolor="#42b0f5",
    #         facecolor="#42c8f5",
    #     )

    # ax1.set_ylabel("ΔF/F")
    # ax1.set_xticks([])
    # ax1.set_xlim([-time_before, time_after])
    # ax1.set_ylim([-1.0, 1.0])
    # ax1.set_title(f"{name} {hemisphere}")
    # ax1.legend(loc="upper right")
    # ax1.vlines(0, -10000, 10000, linestyle="dotted", color="#000")

    # ax2.plot(pseudotime_x, mean_zscore)
    # ax2.set_xlabel("Licking onset (seconds)")
    # ax2.set_ylabel("z-score")
    # ax2.set_ylim([-2.0, 2.0])
    # ax2.set_xlim([-time_before, time_after])
    # ax2.hlines(0, -10000, 10000, linestyle="dashed", color="#000", alpha=0.2)
    # ax2.vlines(0, -10000, 10000, linestyle="dotted", color="#000")

    # plt.subplots_adjust(hspace=0.1)
    # pdf.savefig(fig)

# pdf.close()
# plt.savefig("report.pdf", format="pdf", bbox_inches="tight")
# plt.show()
