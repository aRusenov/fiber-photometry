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
    "--label", help="Label"
)
parser.add_argument(
    "--outdir", help="Output directory (optional). Defaults to /preprocessed in the base dir of each file"
)
args = parser.parse_args()

files = []
for file in args.file:
    for line in file.splitlines():
        files.append(line)
        
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

    data = run_preprocessing_pipeline(signal, control, time, dios)


    outfilename = f"{name}-{label}.h2py"
    subdir = args.outdir or os.path.join(basedir, "preprocessed")
    outfilepath = os.path.join(subdir, outfilename)
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    save_preprocessed_data(data, outpath=outfilepath)
    log(f'Saved to {outfilepath}')