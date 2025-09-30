import numpy as np

from common.lib import read_preprocessed_data, log, Processed5CData, standard_cli_argparse
from common.processing import process_events, save_processed_data, EventBatch


def calculate_blocks(arr, merge_threshold_sec, min_bout_duration_sec, sampling_rate, drop_previous_interval=None):
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

    # keep only ranges where combined duration >= min_bout_duration_sec second
    arr = np.array(merged_ranges)
    diff = arr[:, 1] - arr[:, 0]
    filtered_arr = arr[(diff / sampling_rate) >= min_bout_duration_sec]

    # keep only where gap between blocks >= drop_previous_interval seconds
    if drop_previous_interval is not None and len(filtered_arr) > 1:
        gaps = filtered_arr[1:, 0] - filtered_arr[:-1, 1]
        mask = np.ones(len(filtered_arr), dtype=bool)
        mask[1:] = gaps >= (sampling_rate * 3)
        filtered_arr = filtered_arr[mask]

    return filtered_arr


parser = standard_cli_argparse("Fat licking process transients")
parser.add_argument("--name", help="Name")
parser.add_argument(
    "--dio01",
    help="Source DIO for licking activity",
    default="DIO01",
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
        f"Processing {name} ({subject_label}), reading activity from DIO '{args.dio01}'"
    )

    data = read_preprocessed_data(file)
    sampling_rate = data.sampling_rate
    dio1 = data.dios[args.dio01]

    dio_events = np.where(dio1 == 1)[0]
    print(f"Total licks: {dio_events}")
    licks = calculate_blocks(dio1,
                             merge_threshold_sec=0.5,
                             min_bout_duration_sec=2,
                             sampling_rate=data.sampling_rate,
                             drop_previous_interval=2)

    licks_onset = np.array([(start, end) for start, end in licks])
    licks_outset = np.array([(end, end + sampling_rate) for start, end in licks])

    activity_bins = licks
    # print([(data.time[start], data.time[end]) for (start, end) in licks])
    events = {
        "Onset": licks_onset,
        "Outset": licks_outset
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
activities = process_events(event_batches,
                            time_before=time_before,
                            time_after=time_after,
                            z_scoring='session',
                            z_baseline_strategy='last_non_overlapping',
                            baseline_window=10,
                            activity_bins=activity_bins)

processed_data = Processed5CData(name=name, label=subject_label, activities=activities, sampling_rate=sampling_rate)
save_processed_data(args.outdir, processed_data)
