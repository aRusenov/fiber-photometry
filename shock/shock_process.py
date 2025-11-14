import numpy as np

from common.lib import read_preprocessed_data, log, Processed5CData, standard_cli_argparse
from common.processing import process_events, save_processed_data, EventBatch
from fat.fat_process import activity_bins

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


parser = standard_cli_argparse("Foot shock process traces")
parser.add_argument("--name", help="Name")
parser.add_argument(
    "--dio01",
    help="Source DIO for the delivered shock",
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
    shocks = find_shock_ranges(dio1)
    for start, end in shocks:
        print(f'Shock duration {(end - start) / sampling_rate}')

    activity_bins = np.array([(start - (sampling_rate * (5)), end) for start, end in shocks])

    # print([(data.time[start], data.time[end]) for (start, end) in licks])
    events = {
        "Onset": shocks,
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
                            z_scoring='baseline',
                            z_baseline_strategy='last_non_overlapping',
                            baseline_window=20,
                            activity_bins=activity_bins)

processed_data = Processed5CData(name=name, label=subject_label, activities=activities, sampling_rate=sampling_rate)
save_processed_data(args.outdir, processed_data)
