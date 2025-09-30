import os
import sys

from scipy.linalg.interpolative import estimate_rank

from common.processing import EventBatch, process_events, save_processed_data

# Add the parent for import
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from common.lib import (
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


parser = standard_cli_argparse("5C process transients")
parser.add_argument("--name", help="Name")
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

name = args.name
subject_label = args.label

sampling_rate = None
event_batches = []
activity_bins = []
for file in files:
    log(f"Reading from file {file}")
    log(
        f"Processing {name} ({subject_label}), reading activity from DIOs '{args.dio01}' and '{args.dio02}"
    )

    data = read_preprocessed_data(file)
    sampling_rate = data.sampling_rate
    dio1 = data.dios[args.dio01]
    dio2 = data.dios[args.dio02]
    time = data.time

    # maxlen = max(len(dio1), len(dio2))
    # # FIXME: pad during preprocessing
    # time = np.pad(data.time, (0, maxlen - len(data.time)))

    reward_pickup = find_5c_events(dio2, time, 0.03, 0.05)
    prematures = find_5c_events(dio2, time, 0.9, 1.0)
    omissions = find_5c_events(dio2, time, 0.15, 0.25)
    correct = find_5c_events(dio1, time, 0.15, 0.25)
    incorrect = find_5c_events(dio1, time, 0.9, 1.0)
    initiated_trials = find_5c_events(dio1, time, 0.03, 0.05)

    for (start, _) in initiated_trials:
        # 7s waiting + 2s response + 20s eating = ~30s
        est_trial_end_offset = start + (sampling_rate * (20))
        activity_bins.append((start, est_trial_end_offset))

    events = {
        "Initiated trials": initiated_trials,
        "Prematures": prematures,
        "Reward pickup": reward_pickup,
        "Omissions": omissions,
        "Correct": correct,
        "Incorrect": incorrect,
    }

    event_batches.append(EventBatch(events=events, data=data))
    print("--- Task descriptives ---")
    for event in events.keys():
        print(f"{event: <20} | {len(events[event])}")

    print("-")

log("--------------")
log(f"Pooled data from {len(files)} files")
log("--------------")

time_before = 5
time_after = 10
activities = process_events(event_batches, time_before, time_after, z_scoring='session', z_baseline_strategy='last_non_overlapping', baseline_window=10, activity_bins=activity_bins)

processed_data = Processed5CData(name=name, label=subject_label, activities=activities, sampling_rate=sampling_rate)
save_processed_data(args.outdir, processed_data)
