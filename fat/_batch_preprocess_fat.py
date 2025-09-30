import os
import yaml
import subprocess

script_dir = '/Users/atanas/Documents/workspace/lab-scripts'
out_dir = '/Users/atanas/Documents/workspace/data/analysis/photometry/fat'
data_dir = '/Users/atanas/Documents/workspace/data/analysis/photometry/fat/raw'

def get_params(hem, label, defaults) -> list:
    out = []
    if type(label) is str:
        det = 1 if hem == 'left' else 2
        dios = defaults["dios"]
        files = label.split(',')
        for file in files:
            out.append((file.strip(), f"AIN0{det}", [f"DIO0{dio}" for dio in dios]))
    elif type(label) is list:
        for h in label:
            out.append((h["file"], f"AIN0{h['det']}", [f"DIO0{dio}" for dio in h['dios']]))
    else:
        raise Exception("Could not parse hemisphere")

    return out


with open("input_fat.yaml", "r") as stream:
    data = yaml.safe_load(stream)

subdir = os.path.dirname(__file__)
print(data)

defaults = data["defaults"]
labels = ['left', 'right']
for id in data.keys():
    if id == "defaults":
        continue

    sub = data[id]
    for label in labels:
        if label in sub:
            files = get_params(label, sub[label], defaults)
            for (file, det, dios) in files:
                filename = os.path.join(data_dir, file)
                print(f"{label} {filename}, {det}, {dios}")
                subprocess.run(
                    [
                        "python",
                        os.path.join(script_dir, "preprocess.py"),
                        "--channel",
                        det,
                        "--dio",
                        dios[0],
                        '--remap',
                        'DIO02:DIO01',
                        "--name",
                        id.replace('.', '_'),
                        "--label",
                        label,
                        "--outdir",
                        os.path.join(out_dir, "preprocessed"),
                        "--file",
                        filename,
                    ]
                )
