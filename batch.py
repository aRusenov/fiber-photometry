import os
import yaml
import subprocess


def get_params(label, defaults)-> list:
    out = []
    if type(label) is str:
        det = defaults["det"]
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


with open("manifest.yaml", "r") as stream:
    data = yaml.safe_load(stream)

subdir = os.path.dirname(__file__)
print(data)

script_dir='/Users/atanas/Documents/workspace/lab-scripts'
out_dir='/Users/atanas/Documents/workspace/data/5C'
data_dir='/Users/atanas/Documents/workspace/data/5C'

defaults = data["defaults"]
labels = ['left', 'right']
for id in data.keys():
    if id == "defaults":
        continue

    sub = data[id]
    for label in labels:
        if label in sub:
            files = get_params(sub[label], defaults)
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
                        dios[1],
                        '--remap',
                        'DIO03:DIO01',
                        'DIO04:DIO02',
                        "--label",
                        label,
                        "--outdir",
                        os.path.join(out_dir, "preprocessed"),
                        "--file",
                        filename,
                    ]
                )
