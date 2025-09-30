import os

import matplotlib
from matplotlib import pyplot as plt
# matplotlib.use('QtAgg')

from common.lib import read_doric_file

import numpy as np
from scipy.optimize import curve_fit

def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    """Double exponential with constant offset.
    Allows both upward and downward drift depending on signs of amplitudes.
    """
    tau_fast = tau_slow * tau_multiplier
    return const + amp_slow * np.exp(-t / tau_slow) + amp_fast * np.exp(-t / tau_fast)


def correct_photobleaching(signal, time):
    max_sig = np.max(signal)
    min_sig = np.min(signal)

    # Shared fitting bounds
    bounds = (
        [min_sig, -np.inf, -np.inf, 600, 0],   # lower
        [max_sig,  np.inf,  np.inf, 36000, 1]  # upper
    )

    # Base guess (can handle decay)
    base_guess = [
        np.median(signal),          # const
        (max_sig - min_sig) / 4,    # amp_fast
        (max_sig - min_sig) / 4,    # amp_slow
        3600,                       # tau_slow
        0.1                         # tau_multiplier
    ]

    # Alternative guess (inverted amplitudes, handles growth)
    alt_guess = [
        base_guess[0],
        -base_guess[1],
        -base_guess[2],
        base_guess[3],
        base_guess[4]
    ]

    # Try both fits
    for guess in (base_guess, alt_guess):
        try:
            parms, parm_cov = curve_fit(
                double_exponential,
                time,
                signal,
                p0=guess,
                bounds=bounds,
                maxfev=5000
            )
            expfit = double_exponential(time, *parms)

            # Simple quality check: R² > 0.5
            residuals = signal - expfit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((signal - np.mean(signal))**2)
            r2 = 1 - (ss_res / ss_tot)
            if r2 > 0.5:
                return expfit, parms, r2
        except RuntimeError:
            continue

    # If both fits fail, just return raw signal
    return signal, None, None

def correct_photobleaching_v1(signal, time):
    max_sig = np.max(signal)
    inital_params = [max_sig / 2, max_sig / 4, max_sig / 4, 3600, 0.1]
    bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
    parms, parm_cov = curve_fit(
        double_exponential, time, signal, p0=inital_params, bounds=bounds, maxfev=1000
    )
    expfit = double_exponential(time, *parms)
    return expfit

def calculate_blocks(arr, merge_threshold_sec, min_bout_duration_sec, sampling_rate):
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

    # keep only ranges where combined duration >= min_bout_duration_sec second and gap between blocks >= 5 seconds
    arr = np.array(merged_ranges)
    diff = arr[:, 1] - arr[:, 0]
    filtered_arr = arr[(diff / sampling_rate) >= min_bout_duration_sec]

    return filtered_arr

file = '662.6/662_6R_det1_DIO12_653_2R_det2_DIO34_0000.doric'
data_dir = '/Users/atanas/Documents/workspace/data/analysis/photometry/5C/raw'
channel = 'AIN01'

time, control, signal, dios, sampling_rate = read_doric_file(os.path.join(data_dir, file), channel, ['DIO01'])
fig, (ax) = plt.subplots(1)
ax.plot(time, signal, label='signal')
ax.plot(time, control, label='control')
ax.legend()
plt.title('Raw')
plt.show()

signal_expfit = correct_photobleaching_v1(signal, time)
control_expfit = correct_photobleaching_v1(control, time)
fig, (ax) = plt.subplots(1)
ax.plot(time, signal_expfit, label='signal fit')
ax.plot(time, control_expfit, label='control fit')
ax.legend()
plt.title('Fits')
plt.show()

signal_detrended = signal - signal_expfit
control_detrended = control - control_expfit
fig, (ax) = plt.subplots(1)
# ax.plot(time[0:20*sampling_rate], signal_detrended[0:20*sampling_rate], label='signal corrected')
# ax.plot(time[0:20*sampling_rate], control_detrended[0:20*sampling_rate], label='control corrected')
ax.plot(time, signal_detrended, label='signal corrected')
ax.plot(time, control_detrended, label='control corrected')
ax.legend()
plt.title('Subtracted')
plt.show()

dio01 = dios['DIO01']
licks = calculate_blocks(dio01, merge_threshold_sec=1, min_bout_duration_sec=1, sampling_rate=sampling_rate)
for start, end in licks:
    print(f'lick: {time[start]} - {time[end]}')