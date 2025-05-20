import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import SpanSelector, Button
from scipy.optimize import curve_fit
import argparse

from lib import read_doric_file

parser = argparse.ArgumentParser("FP artifact removal")
parser.add_argument("--file", help="Input file", required=True)
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

args = parser.parse_args()

channel = args.channel
basedir = os.path.dirname(args.file)
filename = os.path.basename(args.file)
label = args.label or channel

time, control, signal, dios, sampling_rate = read_doric_file(
    args.file, channel, args.dio
)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)

ax.plot(time, control, label="Isosbestic")
ax.plot(time, signal, label="GCaMP")
ax.legend()

# Store selections as (xmin, xmax)
selections = []
highlight_lines = []


# Callback: when user selects a range
def onselect(xmin, xmax):
    xmin, xmax = sorted([xmin, xmax])  # make sure xmin <= xmax
    selections.append((xmin, xmax))

    # Highlight selected range on the plot
    mask = (time >= xmin) & (time <= xmax)
    (line,) = ax.plot(time[mask], control[mask], "r", lw=2)
    highlight_lines.append(line)

    fig.canvas.draw()
    print(f"Added selection: {xmin:.2f} to {xmax:.2f}")


# Attach span selector
span = SpanSelector(ax, onselect, "horizontal", useblit=True)


# Define exponential decay model
def exp_decay(x, A, k):
    return A * np.exp(-k * x)


def impute_v2(arr, start, end, decay_rate=0.5):
    arr = np.asarray(arr)
    arr[start:end] = np.nan
    isnan = np.isnan(arr)
    n = len(arr)
    i = 0

    while i < n:
        if isnan[i]:
            # Find the full nan run
            start = i
            while i < n and isnan[i]:
                i += 1
            end = i - 1

            left = start - 1
            right = end + 1

            if left >= 0 and right < n:
                left_val = arr[left]
                right_val = arr[right]
                gap_size = right - left - 1

                imputed = []
                for j in range(gap_size):
                    weight = np.exp(-decay_rate * j)
                    val = left_val * weight + right_val * (1 - weight)
                    imputed.append(val)

                return np.array(imputed)

            else:
                raise ValueError(f"Cannot impute range {start}–{end}: missing boundary value(s).")

        i += 1

    raise ValueError("No missing range found in array.")

def impute(x, y, start, end):
    x = np.arange(len(y))
    y_missing = y.copy()
    y_missing[start:end] = np.nan

    # Use only non-NaN values to fit the model
    mask = ~np.isnan(y_missing)
    x_fit = x[mask]
    y_fit = y_missing[mask]

    # Fit the exponential model to the available data
    params, _ = curve_fit(exp_decay, x_fit, y_fit, p0=(1, 0.1))  # initial guess (A, k)
    A_fit, k_fit = params

    # Predict missing values
    x_missing = x[~mask]
    y_predicted = exp_decay(x_missing, A_fit, k_fit)

    return y_predicted


# --- Button Callbacks ---
def undo(event):
    if selections:
        removed = selections.pop()
        line = highlight_lines.pop()
        line.remove()
        fig.canvas.draw()
        print(f"Undid selection: {removed[0]:.2f} to {removed[1]:.2f}")
    else:
        print("No selections to undo.")


def clear(event):
    selections.clear()
    for line in highlight_lines:
        line.remove()
    highlight_lines.clear()
    fig.canvas.draw()
    print("All selections cleared.")


def submit(event):
    if not selections:
        print("No selections to submit.")
        return

    plt.close()

    print("Imputing removed ranges...")
    for i, (xmin, xmax) in enumerate(selections, 1):
        start = np.searchsorted(time, xmin)
        end = np.searchsorted(time, xmax)
        # signal_imputed = impute(time, signal, start, end)
        # control_imputed = impute(time, control, start, end)
        signal_imputed = impute_v2(signal, start, end)
        control_imputed = impute_v2(control, start, end)

        signal[start:end] = signal_imputed
        control[start:end] = control_imputed
        for key in dios.keys():
            # TODO: zero?
            dios[key][start:end] = np.nan

    # Plot for final preview
    fig, ax = plt.subplots()
    ax.plot(time, control, label="Control")
    ax.plot(time, signal, label="Signal")
    # ax.plot(time, activity, label='Events')
    ax.legend()

    plt.title("Artifacts removed")
    plt.show()

    [name, *suffix] = filename.rsplit(".", 1)
    destdir = os.path.join(basedir, 'cleaned')
    os.makedirs(destdir)
    outfilepath = os.path.join(destdir, f"{name}-{label}.doric")
    with h5py.File(args.file, "r") as f_in, h5py.File(outfilepath, "w") as f_out:
        f_out.create_dataset(
            "DataAcquisition/FPConsole/Signals/Series0001/LockInAOUT01/Time", data=time
        )
        f_out.create_dataset(
            f"DataAcquisition/FPConsole/Signals/Series0001/LockInAOUT01/{channel}",
            data=control,
        )
        f_out.create_dataset(
            f"DataAcquisition/FPConsole/Signals/Series0001/LockInAOUT02/{channel}",
            data=signal,
        )
        idx = 0
        for key in dios.keys():
            f_out.create_dataset(
                f"DataAcquisition/FPConsole/Signals/Series0001/DigitalIO/{key}",
                data=dios[key],
            )

    print(f"Saving to {outfilepath}")


# --- Create Buttons ---
def make_button(position, label, callback):
    ax_btn = plt.axes(position)
    btn = Button(ax_btn, label)
    btn.on_clicked(callback)
    return btn


# Define button layout positions (left, bottom, width, height)
btn_indo = make_button([0.1, 0.15, 0.2, 0.075], "Undo", undo)
btn_clear = make_button([0.4, 0.15, 0.2, 0.075], "Clear", clear)
btn_submit = make_button([0.7, 0.15, 0.2, 0.075], "Submit", submit)

# plt.legend()
plt.show()
