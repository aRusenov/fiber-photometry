import argparse
import os
from lib import (
    log,
    read_doric_file,
    run_preprocessing_pipeline,
    save_preprocessed_data,
)
import re

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

channel = args.channel
for file in files:
    basedir = os.path.dirname(file)
    filename = os.path.basename(file)
    label = args.label or channel
    name = re.search(r"(\d+_\d)", filename).group(0)
    log(f"Processing {name} ({label})")

    time, control, signal, dios, sampling_rate = read_doric_file(
        file, channel, args.dio
    )
    
    if args.removeFirst is not None:
        log(f'Removing first {args.removeFirst} seconds')
        fromIdx = int(sampling_rate * args.removeFirst)
        
        signal = signal[fromIdx:]
        control = control[fromIdx:]
        time = time[fromIdx:]
        for key in dios.keys():
            dios[key] = dios[key][fromIdx:]

    data = run_preprocessing_pipeline(signal, control, time, dios)
    
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
