import os
import yaml
import subprocess
import re
from collections import defaultdict

script_dir = '/Users/atanas/Documents/workspace/lab-scripts'
out_dir = '/Users/atanas/Documents/workspace/data/analysis/photometry/fat/processed'
data_dir = '/Users/atanas/Documents/workspace/data/analysis/photometry/fat/preprocessed'


# Group by "name-hemisphere"
def get_pattern(filename):
    match = re.match(r'(\d{3}_\d-\w+)-', filename)
    return match.group(1) if match else None

# Group files by pattern
preprocessed_files = [f for f in os.listdir(data_dir) if f.endswith('.h2py')]
grouped_files = defaultdict(list)

for file in preprocessed_files:
    pattern = get_pattern(file)
    if pattern:
        grouped_files[pattern].append(file)

for id, files in grouped_files.items():
    print(f"Processing {id}")
    subprocess.run([
        'python',
        os.path.join(script_dir, 'fat', 'fat_process.py'),
        '--file',
        *[os.path.join(data_dir, f) for f in files],
        '--outdir',
        out_dir,
        '--name',
        id,
        '--label',
        'pooled'
    ])
