import os
from dataclasses import dataclass

import h5py
import numpy as np

from common.lib import log, Processed5CData, Activity, Channel, ChannelType, PreprocessedData


@dataclass
class EventBatch:
    events: dict[str, list[tuple[int, int]]]
    data: PreprocessedData


def prepare_outfile(outdir: str, data: Processed5CData):
    outfilename = f"{data.name}-{data.label}.h2py"
    outfilepath = os.path.join(outdir, outfilename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    return outfilepath


def save_processed_data(outdir: str, data: Processed5CData):
    outpath = prepare_outfile(outdir, data)
    log(f"Saving to {outpath}")
    log("***")

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

                if channel.bin_zscore_baseline is not None:
                    f_out.create_dataset(
                        f"Event/{activity.event}/{channel.name.name}/Bins/Zscore_baseline",
                        data=channel.bin_zscore_baseline,
                    )


# Pad in case of length discrepancies
def pad(unpadded):
    if len(unpadded) < 2:
        return unpadded

    max_length = max(len(bin) for bin in unpadded)
    bins = np.array([np.pad(bin, (0, max_length - len(bin))) for bin in unpadded])
    return bins

from typing import Literal

_Z_SCORING = Literal["session", "baseline"]
_Z_BASELINE_STRATEGY = Literal["last_simple", "last_non_overlapping"]

def find_non_overlapping_window(activity: list[tuple[int, int]], from_idx: int, window_size: int, sampling_rate: int) -> \
tuple[int, int]:
    window_samples = window_size * sampling_rate
    current_start = from_idx - window_samples

    while current_start >= 0:
        current_end = current_start + window_samples
        overlaps = False
        for start, end in activity:
            if not (current_end <= start or current_start >= end):
                overlaps = True
                break
        if not overlaps:
            return current_start, current_end
        current_start -= sampling_rate

    return max(0, from_idx - window_samples), from_idx


def process_events(event_batches: list[EventBatch], time_before: float, time_after: float,
                   z_scoring: _Z_SCORING = 'session', baseline_window: int = 20, z_baseline_strategy: _Z_BASELINE_STRATEGY = 'last_simple', activity_bins = None):
    bins: dict[str, Activity] = {}
    for batch in event_batches:
        for event in batch.events.keys():
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
        for event in batch.events.keys():
            activity = batch.events[event][:-1]
            # log(f"Processing {event}")

            for start, end in activity:
                fromIdx = start - (batch.data.sampling_rate * time_before)
                toIdx = start + (batch.data.sampling_rate * time_after)

                bins[event].signal().bins_dff.append(batch.data.signal_dff[fromIdx:toIdx])
                bins[event].control().bins_dff.append(batch.data.control_dff[fromIdx:toIdx])
                bins[event].signal_corr().bins_dff.append(
                    batch.data.signal_corrected_dff[fromIdx:toIdx]
                )

                if z_scoring == 'baseline':
                    if z_baseline_strategy == 'last_simple':
                        baseline_start = max(fromIdx - (batch.data.sampling_rate * baseline_window), 0)
                        baseline_end = fromIdx
                    elif z_baseline_strategy == 'last_non_overlapping':
                        baseline_start, baseline_end = find_non_overlapping_window(
                            activity_bins, start, baseline_window, batch.data.sampling_rate
                        )

                    log(f'Calculating zscore of trace {int(batch.data.time[fromIdx])}-{int(batch.data.time[toIdx])} using baseline window {int(batch.data.time[baseline_start])} - {int(batch.data.time[baseline_end])}')
                    baseline = batch.data.signal_corrected_dff[baseline_start:baseline_end]
                    trace = batch.data.signal_corrected_dff[fromIdx:toIdx]
                    bins[event].signal_corr().bins_zscore.append(
                        np.subtract(trace, np.mean(baseline)) / np.std(baseline)
                    )
                    bins[event].signal_corr().bin_zscore_baseline = batch.data.signal_corrected_zscore[baseline_start:baseline_end]
                elif z_scoring == 'session':
                    bins[event].signal_corr().bins_zscore.append(
                        batch.data.signal_corrected_zscore[fromIdx:toIdx]
                    )

    activities: list[Activity] = []
    for event in bins.keys():
        activity = bins[event]

        # Pad in case of bin length discrepancies
        for channel in activity.channels:
            channel.bins_dff = pad(channel.bins_dff)
            channel.bins_zscore = pad(channel.bins_zscore)

        bin_count = len(activity.signal().bins_dff)
        log(f"{event: <20} | {bin_count}")

        activities.append(activity)

    return activities
