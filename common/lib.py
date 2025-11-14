import math
from argparse import ArgumentParser
from dataclasses import dataclass, field
from enum import Enum
from sklearn.linear_model import LinearRegression

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as ss
from scipy.optimize import curve_fit
from scipy.stats import linregress


@dataclass
class PreprocessedData:
    signal_dff: np.ndarray
    signal_corrected_dff: np.ndarray
    control_dff: np.ndarray
    signal_corrected_zscore: np.ndarray
    time: np.ndarray
    dios: dict[str, np.ndarray]
    sampling_rate: int


@dataclass
class Preprocessed5CData:
    signal_dff: np.ndarray
    signal_corrected_dff: np.ndarray
    control_dff: np.ndarray
    time: np.ndarray
    dios: dict[str, np.ndarray]


@dataclass
class Descriptive:
    bins = []
    mean: np.ndarray = None
    se: np.ndarray = None


class ChannelType(Enum):
    Signal = ("Signal",)
    Control = ("Control",)
    Signal_Corrected = "Signal_Corrected"


@dataclass
class Channel:
    name: ChannelType
    bins_dff: list[np.ndarray] = field(default_factory=list)
    ranges: list[tuple[float, float]] = field(default_factory=list)
    bins_zscore: list[np.ndarray] = field(default_factory=list)
    bin_zscore_baseline: list[np.ndarray] = field(default_factory=list)


@dataclass
class Activity:
    event: str
    channels: list[Channel]

    def signal(self) -> Channel:
        return next((c for c in self.channels if c.name == ChannelType.Signal), None)

    def control(self) -> Channel:
        return next((c for c in self.channels if c.name == ChannelType.Control), None)

    def signal_corr(self) -> Channel:
        return next(
            (c for c in self.channels if c.name == ChannelType.Signal_Corrected), None
        )


@dataclass
class Processed5CData:
    name: str
    label: str
    activities: list[Activity]
    sampling_rate: int


# https://github.com/ThomasAkam/photometry_preprocessing/blob/master/Photometry%20data%20preprocessing.ipynb
def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    """Compute a double exponential function with constant offset.
    Parameters:
    t       : Time vector in seconds.
    const   : Amplitude of the constant offset.
    amp_fast: Amplitude of the fast component.
    amp_slow: Amplitude of the slow component.
    tau_slow: Time constant of slow component in seconds.
    tau_multiplier: Time constant of fast component relative to slow.
    """
    tau_fast = tau_slow * tau_multiplier
    return const + amp_slow * np.exp(-t / tau_slow) + amp_fast * np.exp(-t / tau_fast)


def correct_photobleaching(signal, time):
    max_sig = np.max(signal)
    inital_params = [max_sig / 2, max_sig / 4, max_sig / 4, 3600, 0.1]
    bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
    parms, parm_cov = curve_fit(
        double_exponential, time, signal, p0=inital_params, bounds=bounds, maxfev=1000
    )
    expfit = double_exponential(time, *parms)
    return expfit


# Pad in case of length discrepancies
def pad(unpadded):
    if len(unpadded) < 2:
        return unpadded

    max_length = max(len(bin) for bin in unpadded)
    bins = np.array([np.pad(bin, (0, max_length - len(bin))) for bin in unpadded])
    return bins


def compute_mean(transients):
    return np.mean(transients, axis=0)


def correct_baseline(signal, baseline_window):
    # bin_signal = signal[fromIdx:toIdx]

    baseline = np.mean(signal[:baseline_window])
    baseline_corrected = np.subtract(signal, baseline)

    return baseline_corrected


def compute_stderr(transients):
    return np.std(transients, axis=0, ddof=1) / np.sqrt(np.size(transients, axis=0))


def compute_zscore(dff):
    numerator = np.subtract(dff, np.nanmean(dff))
    zscore = np.divide(numerator, np.nanstd(dff))
    return zscore


# Applies a digital filter to smoothen the signal
def filter_signal(signal, filter_window):
    b = np.divide(np.ones((filter_window,)), filter_window)
    a = 1
    return ss.filtfilt(b, a, signal)


def delta_ff(signal, control):
    res = np.subtract(signal, control)
    normData = np.divide(res, control) * 100

    return normData


def fit_polynomial(control, signal):
    fit = np.polyfit(control, signal, 1)
    arr = (fit[0] * control) + fit[1]
    return arr


# def calculate_transients_v2(signal, time, blocks, )


def calculate_transients(signal, time, blocks, time_before, time_after, unit="dff"):
    bins = []
    for block in blocks:
        # Keep this for foot shock
        # if block[0] == block[1]:
        #     continue
        startTime = time[block[0]]

        zeroTimeIdx = block[0]
        fromIdx = np.where(time > startTime - time_before)[0][0]
        toIdx = np.where(
            np.logical_and(
                time >= startTime - time_before, time <= startTime + time_after
            )
        )[0][-1]
        bin_signal = signal[fromIdx:toIdx]
        if unit == "z":
            bin_signal = compute_zscore(bin_signal)

        length = zeroTimeIdx - fromIdx
        baseline = np.mean(bin_signal[:length])
        baselineCorrection = np.subtract(bin_signal, baseline)
        bins.append(baselineCorrection)

    # Pad in case of length discrepancies
    max_length = max(len(bin) for bin in bins)
    bins = np.array([np.pad(bin, (0, max_length - len(bin))) for bin in bins])

    return bins


def plot_transients(transients, zs, timeBefore, timeAfter, labels):
    fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={"height_ratios": [3, 1]})
    for transient, label in transients:
        mean_y = np.mean(transient, axis=0)
        stderr_y = np.std(transient, axis=0, ddof=1) / np.sqrt(
            np.size(transient, axis=0)
        )
        pseudotime_x = np.linspace(-timeBefore, timeAfter, num=len(mean_y))
        ax1.plot(pseudotime_x, mean_y, label=label)
        ax1.fill_between(
            pseudotime_x,
            mean_y - stderr_y,
            mean_y + stderr_y,
            alpha=0.5,
            edgecolor="#42b0f5",
            facecolor="#42c8f5",
        )

    # laser_times = np.array([0, 5])
    # ax1.axvspan(0, 5, facecolor='yellow', alpha=0.2)
    # reward_ticks = ax1.plot(laser_times, np.full(np.size(laser_times), 0.08), label='Reward Cue',color='w', marker="v", mfc='k', mec='k', ms=8)
    ax1.set_ylim([-1.0, 1.0])
    ax1.set_title(labels["title"])
    ax1.set_ylabel("ΔF/F")
    ax1.set_xticks([])
    ax1.legend(loc="upper right")

    mean_zscore = np.mean(zs, axis=0)
    ax2.plot(pseudotime_x, mean_zscore)
    ax2.set_xlabel(labels["x"])
    ax2.set_ylabel("z-score")
    new_var = 2.0
    ax2.set_ylim([-new_var, 2.0])
    ax2.set_xlim([-timeBefore, timeAfter])

    plt.subplots_adjust(hspace=0.1)
    plt.show()


def read_csv_file(file, channel, dio_keys):
    dio_template = "Digital I/O | Ch.1"
    cols = [
        "---",
        "Analog In. | Ch.1",
        "Analog In. | Ch.1.1",
        "Analog In. | Ch.2",
        "Digital I/O | Ch.1",
        "Digital I/O | Ch.2",
    ]
    df = pd.read_csv(file, usecols=cols)

    return (
        df["---"][1:].astype(float),
        df["Analog In. | Ch.1"][1:].astype(float),
        df["Analog In. | Ch.1.1"][1:].astype(float),
    )


def read_doric_file(file, channel, dio_keys, downsample_factor=None):
    # with h5py.File(file, "r") as f:
    #     f.visit(printname)
    log(f"Reading from {file}")
    with h5py.File(file, "r") as f:
        time = np.array(
            f["DataAcquisition/FPConsole/Signals/Series0001/LockInAOUT01/Time"]
        )
        control = np.array(
            f[f"DataAcquisition/FPConsole/Signals/Series0001/LockInAOUT01/{channel}"]
        )
        signal = np.array(
            f[f"DataAcquisition/FPConsole/Signals/Series0001/LockInAOUT02/{channel}"]
        )
        dios = {}
        for key in dio_keys:
            dios[key] = np.array(
                f[f"DataAcquisition/FPConsole/Signals/Series0001/DigitalIO/{key}"]
            )

    sampling_rate = math.floor(1 / (time[1] - time[0]))
    log(f"Sampling rate: {sampling_rate}")

    return time, control, signal, dios, sampling_rate


def mad(arr):
    """Compute the Median Absolute Deviation (MAD) of an array."""
    median = np.median(arr)
    deviation = np.abs(arr - median)
    return np.median(deviation)


def detect_extremes_mad(arr, window_size=5, mad_threshold=4.0, stride=1):
    """
    Detects extreme windows in an array using MAD instead of standard deviation.

    Parameters:
        arr (list or np.array): Input array of positive real numbers.
        window_size (int): Size of the sliding window.
        mad_threshold (float): Threshold for MAD to flag a window as extreme.
        stride (int): Step size to move the sliding window.

    Returns:
        List of tuples: (start_index, window_values, MAD) for extreme windows.
    """
    arr = np.array(arr)
    extremes = []

    max = 0
    for i in range(0, len(arr) - window_size + 1, stride):
        window = arr[i: i + window_size]
        mad_value = mad(window)
        max = mad_value if mad_value > max else max
        if mad_value > mad_threshold:
            extremes.append((i, window.tolist(), mad_value))

    print(f"max={max}")
    return extremes


def group_adjacent_indices(indices, max_gap=1):
    """
    Groups a list of indices into start-end ranges where indices are adjacent or close.

    Parameters:
        indices (list of int): Sorted list of indices.
        max_gap (int): Maximum gap between indices to still be considered part of the same group.

    Returns:
        List of (start, end) tuples.
    """
    if len(indices) == 0:
        return []

    grouped = []
    start = prev = indices[0]

    for i in indices[1:]:
        if i - prev <= max_gap:
            prev = i
        else:
            grouped.append((start, prev))
            start = prev = i

    grouped.append((start, prev))
    return grouped


def detect_peaks_mad(signal, threshold=5.0, baseline_window=None):
    """
    Detects peaks in a photometry signal using MAD-based thresholding.

    Parameters:
        signal (list or np.array): 1D photometry signal (positive real values).
        threshold (float): Multiplier for MAD above the median to consider a point a peak.
        baseline_window (tuple): Optional (start, end) indices to define baseline for MAD calculation.

    Returns:
        List of indices where peaks occur.
    """

    # Use full signal or baseline slice for median and MAD
    if baseline_window:
        baseline = signal[baseline_window[0]: baseline_window[1]]
    else:
        baseline = signal

    med = np.median(baseline)
    mad = np.median(np.abs(baseline - med))

    # Avoid divide-by-zero error
    if mad == 0:
        return []

    # Threshold for peak detection
    threshold_value = med + threshold * mad
    abs_signal = np.abs(signal)
    peaks = np.where(abs_signal > threshold_value)[0]
    peak_ranges = group_adjacent_indices(peaks, max_gap=10)

    return peak_ranges


def sliding_mad_peaks(signal, window_size=100, mad_threshold=5.0, stride=1):
    """
    Detects peaks in a decaying photometry signal using a sliding MAD threshold.

    Parameters:
        signal (list or np.array): Photometry signal (positive real values).
        window_size (int): Size of the sliding window for MAD calculation.
        mad_threshold (float): Threshold in MAD units above local median.
        stride (int): Step size for sliding window.

    Returns:
        List of peak indices (relative to signal) that exceed local MAD threshold.
    """
    signal = np.array(signal)
    peaks = set()

    for i in range(0, len(signal) - window_size + 1, stride):
        window = signal[i: i + window_size]
        median = np.median(window)
        mad = np.median(np.abs(window - median))

        if mad == 0:
            continue  # skip flat or uniform windows

        threshold = median + mad_threshold * mad

        # Identify values above threshold within the window
        for j in range(window_size):
            if signal[i + j] > threshold:
                peaks.add(i + j)

    peaks = sorted(peaks)
    peak_ranges = group_adjacent_indices(peaks, max_gap=10)
    return peak_ranges

def compute_dff(signal_465, signal_405):
    """Regress 405 from 465, compute ΔF/F."""
    model = LinearRegression().fit(signal_405.reshape(-1, 1), signal_465)
    fit_405 = model.predict(signal_405.reshape(-1, 1))
    dff = (signal_465 - fit_405) / np.mean(fit_405)
    return dff

def run_preprocessing_pipeline(
        signal: np.ndarray,
        control: np.ndarray,
        time: np.ndarray,
        dios: dict[str, np.ndarray],
        sampling_rate: int,
        label = None
) -> PreprocessedData:
    # log("Plotting raw")
    # fig, (ax1) = plt.subplots(1)
    # ax1.plot(time, signal, label="signal")
    # ax1.plot(time, control, label="control")
    # if (label is not None):
    #     plt.title(label)
    # plt.legend()
    # plt.show()

    log("Filtering")
    window = 100
    signal = filter_signal(signal, window)
    control = filter_signal(control, window)

    log("Fitting")
    signal_expfit = correct_photobleaching(signal, time)
    signal_detrended = signal - signal_expfit

    control_expfit = correct_photobleaching(control, time)
    control_detrended = control - control_expfit

    # log("Plotting fitted")
    # fig, (ax1) = plt.subplots(1)
    # ax1.plot(time, signal_detrended, label="signal")
    # ax1.plot(time, control_detrended, label="control")
    # if (label is not None):
    #     plt.title(label)
    # plt.legend()
    # plt.show()

    log("Artifact detection")
    peak_ranges = detect_peaks_mad(control_detrended)
    if len(peak_ranges) > 0:
        log(
            f"WARNING: {len(peak_ranges)} potential motion artifact(s) detected -> consider cleanup"
        )
    else:
        log("No artifacts detected")

    log("Motion correcting")
    slope, intercept, r_value, p_value, std_err = linregress(
        x=control_detrended, y=signal_detrended
    )
    signal_est_motion = intercept + slope * control_detrended
    signal_motion_corrected = signal_detrended - signal_est_motion
    print("Slope    : {:.3f}".format(slope))
    print("R-squared: {:.3f}".format(r_value ** 2))

    log("Dff")
    signal_dff = 100 * signal_detrended / signal_expfit
    signal_corrected_dff = 100 * signal_motion_corrected / signal_expfit
    control_dff = 100 * control_detrended / control_expfit
    # Session-wide z-score
    signal_corrected_zscore = (
                                      signal_motion_corrected - np.mean(signal_motion_corrected)
                              ) / np.std(signal_motion_corrected)

    return PreprocessedData(
        signal_dff=signal_dff,
        signal_corrected_dff=signal_corrected_dff,
        control_dff=control_dff,
        signal_corrected_zscore=signal_corrected_zscore,
        time=time,
        dios=dios,
        sampling_rate=sampling_rate
    )


DATASET_DIO_PREFIX = "DIO"


def save_processed_data(data: PreprocessedData, outpath):
    with h5py.File(outpath, "w") as f_out:
        f_out.create_dataset("Time", data=data.time)
        f_out.create_dataset("Signal", data=data.signal_dff)
        f_out.create_dataset("Signal_Corrected", data=data.signal_corrected_dff)
        f_out.create_dataset("Control", data=data.control_dff)
        for key in data.dios.keys():
            f_out.create_dataset(
                f"{DATASET_DIO_PREFIX}/{key}",
                data=data.dios[key],
            )


def save_preprocessed_data(data: PreprocessedData, outpath):
    with h5py.File(outpath, "w") as f_out:
        f_out.create_dataset("Time", data=data.time)
        f_out.create_dataset("Signal", data=data.signal_dff)
        f_out.create_dataset("Control", data=data.control_dff)
        f_out.create_dataset("Signal_Corrected", data=data.signal_corrected_dff)
        f_out.create_dataset("Signal_Corrected_Zscore", data=data.signal_corrected_zscore)
        f_out.create_dataset('Meta/Sampling_Rate', data=data.sampling_rate)
        for key in data.dios.keys():
            f_out.create_dataset(
                f"{DATASET_DIO_PREFIX}/{key}",
                data=data.dios[key],
            )


def read_preprocessed_data(inpath) -> PreprocessedData:
    with h5py.File(inpath, "r") as f:
        # f.visit(printname)
        time = np.array(f["Time"])
        signal_dff = np.array(f["Signal"])
        signal_corrected_dff = np.array(f["Signal_Corrected"])
        signal_corrected_zscore = np.array(f["Signal_Corrected_Zscore"])
        control_dff = np.array(f["Control"])
        sampling_rate = f['Meta/Sampling_Rate'][()] or math.floor(1 / (time[-1] - time[-2]))
        dios = {}
        for key in f[DATASET_DIO_PREFIX].keys():
            dios[key] = np.array(f[DATASET_DIO_PREFIX][key])

    log(f"Sampling rate: {sampling_rate}")

    return PreprocessedData(
        signal_dff=signal_dff,
        signal_corrected_dff=signal_corrected_dff,
        control_dff=control_dff,
        signal_corrected_zscore=signal_corrected_zscore,
        sampling_rate=sampling_rate,
        time=time,
        dios=dios,
    )


def log(txt):
    print(txt)


def printname(name):
    print(name)


def standard_cli_argparse(title="FP data processor") -> ArgumentParser:
    parser = ArgumentParser(title)
    parser.add_argument("--file", nargs="+", help="Input file", required=True)
    parser.add_argument("--label", help="Label")
    parser.add_argument(
        "--outdir",
        help="Output directory (optional). Defaults to working dir",
    )
    return parser
