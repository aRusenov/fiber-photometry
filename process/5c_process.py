import os
import re
import sys
import h5py
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
    log,
    read_preprocessed_data,
    standard_cli_argparse,
)


def find_5c_events(dio, time, duration_from, duration_to):
    """
    Finds all continuous ranges of 1s in the input array.

    Args:
        arr (list of int): List containing 0s and 1s.

    Returns:
        list of tuples: Each tuple contains the start and end index of a range of 1s.
                        The end index is inclusive.
    """
    ranges = []
    start = None

    for i, val in enumerate(dio):
        if val == 1:
            if start is None:
                start = i
        elif start is not None:
            ranges.append((start, i - 1))
            start = None

    # Handle case where array ends with a 1-range
    if start is not None:
        ranges.append((start, len(dio) - 1))

    events = []
    for start, end in ranges:
        duration = time[end] - time[start]
        if duration >= duration_from and duration <= duration_to:
            events.append((start, end))

    return events


def prepare_outfile(data: Processed5CData):
    outfilename = f"{data.name}-{data.label}.h2py"
    basedir = os.path.dirname(file)
    subdir = args.outdir or os.path.join(basedir, "processed")
    outfilepath = os.path.join(subdir, outfilename)
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    return outfilepath


def save_processed_data(data: Processed5CData):
    outpath = prepare_outfile(data)
    log(f"Saving to {outpath}")

    with h5py.File(outpath, "w") as f_out:
        # print(asdict(data))
        f_out.create_dataset('Meta/Sampling_Rate', data=data.sampling_rate)
        for activity in data.activities:
            for channel in activity.channels:
                if channel.bins_dff is not None:
                    f_out.create_dataset(
                        f"Event/{activity.event}/{channel.name.name}/Bins/Dff",
                        data=channel.bins_dff,
                    )

                if channel.bins_zscore is not None:
                    f_out.create_dataset(
                        f"Event/{activity.event}/{channel.name.name}/Bins/Zscore",
                        data=channel.bins_zscore,
                    )


# Pad in case of length discrepancies
def pad(unpadded):
    if len(unpadded) < 2:
        return unpadded

    max_length = max(len(bin) for bin in unpadded)
    bins = np.array([np.pad(bin, (0, max_length - len(bin))) for bin in unpadded])
    return bins


parser = standard_cli_argparse("5C plot transients")
parser.add_argument(
    "--dio01",
    help="Source DIO for correct, incorrect and trial start activity",
    default="DIO01",
)
parser.add_argument(
    "--dio02",
    help="Source DIO for omission, premature and reward pickup activity",
    default="DIO02",
)
parser.add_argument(
    "--time-before",
    help="Time before (in seconds) for baseline",
    default="DIO02",
)

args = parser.parse_args()

files = []
for file in args.file:
    for line in file.splitlines():
        files.append(line)

# pdf = PdfPages("output.pdf")
subject_label = args.label

baseline_correction_from = 5
time_before = 5
time_after = 10

bins: dict[str, Activity] = {}
for file in files:
    filename = os.path.basename(file)
    log(f"file {file}")
    name = re.search(r"(\d+_\d)", filename).group(0)
    log(
        f"Processing {name} ({subject_label}), reading activity from DIOs '{args.dio01}' and '{args.dio02}"
    )

    data = read_preprocessed_data(file)

    dio1 = data.dios[args.dio01]
    dio2 = data.dios[args.dio02]
    maxlen = max(len(dio1), len(dio2))
    
    # FIXME: pad during preprocessing
    time = np.pad(data.time, (0, maxlen - len(data.time)))

    reward_pickup = find_5c_events(dio2, time, 0.03, 0.05)
    prematures = find_5c_events(dio2, time, 0.9, 1.0)
    omissions = find_5c_events(dio2, time, 0.15, 0.25)
    correct = find_5c_events(dio1, time, 0.15, 0.25)
    incorrect = find_5c_events(dio1, time, 0.9, 1.0)
    initiated_trials = find_5c_events(dio1, time, 0.03, 0.05)

    events = {
        "Initiated trials": initiated_trials,
        "Prematures": prematures,
        "Reward pickup": reward_pickup,
        "Omissions": omissions,
        "Correct": correct,
        "Incorrect": incorrect,
    }

    print("--- Task descriptives ---")
    for event in events.keys():
        print(f"{event: <20} | {len(events[event])}")

    print("-")

    for event in events:
        if event not in bins:
            bins[event] = Activity(
                event=event,
                channels=[
                    Channel(name=ChannelType.Signal),
                    Channel(name=ChannelType.Control),
                    Channel(name=ChannelType.Signal_Corrected),
                ],
            )

    # Pool all bins
    for event in events.keys():
        activity = events[event]
        log(f"Processing {event}")

        for start, end in activity:
            fromIdx = start - (data.sampling_rate * time_before)
            toIdx = end + (data.sampling_rate * time_after)

            bins[event].signal().bins_dff.append(data.signal_dff[fromIdx:toIdx])
            bins[event].control().bins_dff.append(data.control_dff[fromIdx:toIdx])
            bins[event].signal_corr().bins_dff.append(
                data.signal_corrected_dff[fromIdx:toIdx]
            )
            bins[event].signal_corr().bins_zscore.append(
                data.signal_corrected_zscore[fromIdx:toIdx]
            )


log("--------------")
log(f"Pooled data from {len(files)} files")
log("--------------")
activities: list[Activity] = []
maxlen = 0
for event in bins.keys():
    activity = bins[event]

    # Pad in case of bin length descrepancies
    for channel in activity.channels:
        channel.bins_dff = pad(channel.bins_dff)
        channel.bins_zscore = pad(channel.bins_zscore)

    bin_count = len(activity.signal().bins_dff)
    log(f"{event: <20} | {bin_count}")

    activities.append(activity)

processed_data = Processed5CData(name=name, label=subject_label, activities=activities)
save_processed_data(processed_data)
