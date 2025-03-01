import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math
from scipy import signal as ss
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from scipy.stats import linregress

from matplotlib.pyplot import figure

# https://github.com/ThomasAkam/photometry_preprocessing/blob/master/Photometry%20data%20preprocessing.ipynb
def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    '''Compute a double exponential function with constant offset.
    Parameters:
    t       : Time vector in seconds.
    const   : Amplitude of the constant offset. 
    amp_fast: Amplitude of the fast component.  
    amp_slow: Amplitude of the slow component.  
    tau_slow: Time constant of slow component in seconds.
    tau_multiplier: Time constant of fast component relative to slow. 
    '''
    tau_fast = tau_slow*tau_multiplier
    return const+amp_slow*np.exp(-t/tau_slow)+amp_fast*np.exp(-t/tau_fast)

filename = "/Users/atanas/Downloads/2nd Stimulation Optomization 14-2-25/5mw/662_6L_5mw_40hz_20ms_0003.doric"

def correctPhotobleaching(signal, time):
    max_sig = np.max(signal)
    inital_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
    bounds = ([0      , 0      , 0      , 600  , 0],
          [max_sig, max_sig, max_sig, 36000, 1])
    parms, parm_cov = curve_fit(double_exponential, time, signal, 
                                  p0=inital_params, bounds=bounds, maxfev=1000)
    expfit = double_exponential(time, *parms)
    return expfit

def pad(array, target_shape):
    return np.pad(
        array,
        [(0, target_shape[i] - array.shape[i]) for i in range(len(array.shape))],
        "constant",
    )
    
def computeZScore(dff):
    numerator = np.subtract(dff, np.nanmean(dff))
    zscore = np.divide(numerator, np.nanstd(dff))
    return zscore

# Applies a digital filter to smoothen the signal
def filterSignal(signal, filter_window):
    b = np.divide(np.ones((filter_window,)), filter_window)
    a = 1
    return ss.filtfilt(b, a, signal)

def deltaFF(signal, control):    
	res = np.subtract(signal, control)
	normData = np.divide(res, control) * 100

	return normData

def fitPolynomial(control, signal):
    fit = np.polyfit(control, signal, 1)
    arr = (fit[0] * control) + fit[1]
    return arr

def calculateTransients(signal, time, activity, timeBefore, timeAfter):
    # Identify transients (i.e. activity blocks)
    idxs = np.where(activity == 1.0)[0]
    gaps = np.diff(idxs)
    split_points = np.where(gaps > 5)[0] + 1
    blocks = np.split(idxs, split_points)
    log(f'Transients: {[time[i[0]].item() for i in blocks]}')    
    
    # 
    bins = []
    for block in blocks:
        startTime = time[block[0]]
        
        zeroTimeIdx = block[0]
        fromIdx = np.where(time > startTime - timeBefore)[0][0]
        toIdx = np.where(np.logical_and(time >= startTime - timeBefore, time <= startTime + timeAfter))[0][-1]
        # TODO: extend to prev 25 seconds
        bin_zscore = signal[fromIdx:toIdx]
        bin = computeZScore(bin_zscore)
        # bin = bin_zscore
        
        length = zeroTimeIdx - fromIdx
        baseline = np.mean(bin[:length])
        baselineCorrection = np.subtract(bin, baseline)
        bin = baselineCorrection
        bins.append(bin)
    
    # Pad in case of length discrepancies
    max_length = max(len(bin) for bin in bins)
    bins = np.array([np.pad(bin, (0, max_length - len(bin))) for bin in bins])

    return bins

def plotTransients(transients, timeBefore, timeAfter):
    fig, ax1 = plt.subplots()  
    for (transient, label) in transients:
        log(transient.shape)
        mean_y = np.mean(transient, axis = 0)
        stderr_y = np.std(transient, axis = 0, ddof=1) / np.sqrt(np.size(transient, axis = 0))
        pseudotime_x = np.linspace(-timeBefore, timeAfter, num = len(mean_y))
        ax1.plot(pseudotime_x, mean_y, label = label)
        ax1.fill_between(pseudotime_x, mean_y - stderr_y, mean_y + stderr_y, alpha=0.5, edgecolor='#42b0f5', facecolor='#42c8f5')
    
    laser_times = np.array([0, 5])
    ax1.axvspan(0, 5, facecolor='yellow', alpha=0.2)
    # reward_ticks = ax1.plot(laser_times, np.full(np.size(laser_times), 0.08), label='Reward Cue',color='w', marker="v", mfc='k', mec='k', ms=8)
    ax1.set_title(f'662.6 laser stimulation (10mW, 40hz, 20ms)')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("z-score")
    plt.legend()
    plt.show()

def log(txt):
    print(txt)

def printname(name):
    print(name)
with h5py.File(filename, 'r') as f:
    f.visit(printname)

with h5py.File(filename, 'r') as f:
    time = np.array(f["DataAcquisition/FPConsole/Signals/Series0001/LockInAOUT01/Time"])
    control = np.array(f["DataAcquisition/FPConsole/Signals/Series0001/LockInAOUT01/AIN01"])
    signal = np.array(f["DataAcquisition/FPConsole/Signals/Series0001/LockInAOUT02/AIN01"])
    laserDIO = np.array(f["DataAcquisition/FPConsole/Signals/Series0001/DigitalIO/DIO01"])

# Preprocess   
sampling_rate = math.floor(1 / (time[-1] - time[-2]))

log(f'Sampling rate: {sampling_rate}')
downsample_factor = None

if (downsample_factor is not None):
    log(f'Down-sampling factor: {downsample_factor}')    
    time = time[1::downsample_factor]
    control = control[1::downsample_factor]
    signal = signal[1::downsample_factor]
    laserDIO = laserDIO[1::downsample_factor]

removeFirstSec = None
if (removeFirstSec is not None):
    removeOffset = np.where(time >= removeFirstSec)[0][0]
    log(f'Removing first {removeFirstSec} seconds at range (0, {removeOffset})')
    time = time[removeOffset:]
    control = control[removeOffset:]
    signal = signal[removeOffset:]
    laserDIO = laserDIO[removeOffset:]

# plt.plot(time, signal, label = 'GCaMP')
# plt.plot(time, control, label = 'Isosbestic')
# plt.show()
# exit(0)


window = 100
signal = filterSignal(signal, window)
control = filterSignal(control, window)
# control_fitted = fitPolynomial(control, signal)
# signal_corrected = deltaFF(signal, control_fitted)

# fit
signal_expfit = correctPhotobleaching(signal, time)
signal_detrended = signal - signal_expfit

control_expfit = correctPhotobleaching(control, time)
control_detrended = control - control_expfit

# motion correction
slope, intercept, r_value, p_value, std_err = linregress(x=control_detrended, y=signal_detrended)
signal_est_motion = intercept + slope * control_detrended
signal_motion_corrected = signal_detrended - signal_est_motion
print('Slope    : {:.3f}'.format(slope))
print('R-squared: {:.3f}'.format(r_value**2))

# dff
signal_dff = 100 * signal_detrended / signal_expfit
signal_corrected_dff = 100 * signal_motion_corrected / signal_expfit
control_dff = 100 * control_detrended / control_expfit


# signal_dff = deltaFF(signal, control_fitted)
# signal_zscore = computeZScore(signal_dff)
# control_zscore = computeZScore(control_dff)

timeBefore = 10
timeAfter = 15

signal_corrected_transients = calculateTransients(signal_corrected_dff, time, laserDIO, timeBefore, timeAfter)
signal_transients = calculateTransients(signal_dff, time, laserDIO, timeBefore, timeAfter)
control_transients = calculateTransients(control_dff, time, laserDIO, timeBefore, timeAfter)

plotTransients([
                (signal_transients, 'GCaMP'),
                (signal_corrected_transients, 'GCaMP corrected'),
                (control_transients, 'Isosbestic')
                ], timeBefore, timeAfter)
