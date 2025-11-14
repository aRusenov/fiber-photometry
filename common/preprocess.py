import argparse
import os
from lib import (
    log,
    read_doric_file,
    run_preprocessing_pipeline,
    save_preprocessed_data,
)
from scipy import interpolate
import numpy as np


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
    "--remap", nargs="*", help="Remap a DIO key from input file to a different DIO key in output (e.g. DIO03 to DIO01)",
    default=[]
)
parser.add_argument(
    "--remove-first", dest='removeFirst', type=int, help="Remove first N seconds"
)
parser.add_argument("--name", help="Name")
parser.add_argument("--label", help="Label")
parser.add_argument(
    "--outdir",
    help="Output directory (optional). Defaults to /preprocessed in the base dir of each file",
)
args = parser.parse_args()

files = []
for file in args.file:
    for line in file.splitlines():
        files.append(line)

dio_remap = dict(tuple(remap.split(':') for remap in args.remap))

resample_to = 240
channel = args.channel
for file in files:
    basedir = os.path.dirname(file)
    filename = os.path.basename(file)
    name = args.name
    label = args.label or channel
    log(f"Processing {name} ({label})")

    time, control, signal, dios, sampling_rate = read_doric_file(
        file, channel, args.dio
    )

    # Match lengths of signal and control to old_times
    if len(signal) > len(time):
        signal = signal[:len(time)]
    elif len(signal) < len(time):
        signal = np.pad(signal, (0, len(time) - len(signal)), 'edge')

    if len(control) > len(time):
        control = control[:len(time)]
    elif len(control) < len(time):
        control = np.pad(control, (0, len(time) - len(control)), 'edge')

    for key in dios.keys():
        if len(dios[key]) > len(time):
            dios[key] = dios[key][:len(time)]
        elif len(dios[key]) < len(time):
            dios[key] = np.pad(dios[key], (0, len(time) - len(dios[key])), 'constant', constant_values=0)

    if args.removeFirst is not None:
        log(f'Removing first {args.removeFirst} seconds')
        fromIdx = int(sampling_rate * args.removeFirst)

        signal = signal[fromIdx:]
        control = control[fromIdx:]
        time = time[fromIdx:]
        for key in dios.keys():
            dios[key] = dios[key][fromIdx:]

    if resample_to != sampling_rate:
        log(f"Re-sampling to {resample_to}")
        resampling_factor = resample_to / sampling_rate

        if resampling_factor > 1:
            # Upsampling
            # Create new time points
            old_times = np.arange(len(time))
            new_times = np.linspace(0, len(time) - 1, int(len(time) * resampling_factor))

            # Interpolate signal, control and time
            f_signal = interpolate.interp1d(old_times, signal)
            f_control = interpolate.interp1d(old_times, control)
            f_time = interpolate.interp1d(old_times, time)

            signal = f_signal(new_times)
            control = f_control(new_times)
            time = f_time(new_times)

            # Upsample DIOs by repeating samples
            for key in dios.keys():
                dio = np.asarray(dios[key])
                dios[key] = np.repeat(dio, int(resampling_factor))
        else:
            # Downsampling
            signal = signal[::int(1 / resampling_factor)]
            control = control[::int(1 / resampling_factor)]
            time = time[::int(1 / resampling_factor)]
            for key in dios.keys():
                dios[key] = dios[key][::int(1 / resampling_factor)]

        sampling_rate = resample_to

    data = run_preprocessing_pipeline(signal, control, time, dios, sampling_rate, label=f'{name} ({label})')

    # Remap DIO keys
    current = list(data.dios.keys())
    for key in current:
        if key in dio_remap:
            out_key = dio_remap[key]
            log(f'Remapping {key} to {out_key}')
            data.dios[out_key] = data.dios.pop(key)

    subdir = args.outdir or os.path.join(basedir, "preprocessed")
    suffix = 1
    while True:
        outfilename = f"{name}-{label}-{suffix:02}.h2py"
        outfilepath = os.path.join(subdir, outfilename)
        if os.path.isfile(outfilepath):
            suffix += 1
        else:
            break

    if not os.path.exists(subdir):
        os.makedirs(subdir)

    save_preprocessed_data(data, outpath=outfilepath)
    log(f"Saved to {outfilepath}")
