# Temporary smoke test that stubs heavy scientific deps and runs the CLIs with --help
import sys
import types
import runpy

# Minimal stubs for heavy dependencies used at import time

def make_module(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

# Base modules
np = make_module('numpy')
scipy = make_module('scipy')
scipy_signal = make_module('scipy.signal')
scipy_interpolate = make_module('scipy.interpolate')
scipy_optimize = make_module('scipy.optimize')
scipy_stats = make_module('scipy.stats')
h5py = make_module('h5py')
pandas = make_module('pandas')
matplotlib = make_module('matplotlib')
matplotlib_pyplot = make_module('matplotlib.pyplot')
matplotlib_ticker = make_module('matplotlib.ticker')

# Minimal attributes used by lib.py annotations/imports
# numpy
setattr(np, 'ndarray', object)
setattr(np, 'array', lambda *a, **k: a[0] if a else [])
setattr(np, 'mean', lambda x, axis=None: 0)
setattr(np, 'std', lambda x, axis=None, ddof=0: 1)
setattr(np, 'median', lambda x: 0)
setattr(np, 'abs', lambda x: x)
setattr(np, 'subtract', lambda a, b: a)
setattr(np, 'divide', lambda a, b: a)
setattr(np, 'pad', lambda arr, pad_width: arr)
setattr(np, 'logical_and', lambda a, b: a)
setattr(np, 'where', lambda cond: ([0],))
setattr(np, 'linspace', lambda a, b, num=50: [0]*num)
setattr(np, 'size', lambda x, axis=None: 1)
setattr(np, 'std', lambda x, axis=None, ddof=0: 1)
setattr(np, 'zeros', lambda shape: [0]* (shape if isinstance(shape, int) else shape[0]))
setattr(np, 'float64', float)

# scipy.interpolate
setattr(scipy_interpolate, 'make_interp_spline', lambda *a, **k: None)
setattr(scipy_interpolate, 'interp1d', lambda *a, **k: None)
# scipy.optimize
setattr(scipy_optimize, 'curve_fit', lambda *a, **k: ([1,1,1,1,1], None))
setattr(scipy_optimize, 'minimize', lambda *a, **k: None)
# scipy.stats
setattr(scipy_stats, 'linregress', lambda x,y: (0,0,0,0,0))
setattr(scipy_stats, 'kurtosis', lambda *a, **k: 0)
setattr(scipy, 'stats', scipy_stats)
# scipy.signal
setattr(scipy_signal, 'filtfilt', lambda b, a, x: x)
# h5py minimal File context manager stub
class DummyH5:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): pass
    def create_dataset(self, *a, **k): pass
    def __getitem__(self, key):
        # return minimal structures
        class Node:
            def __getitem__(self, k2):
                return []
            def keys(self):
                return []
            def __array__(self, *a, **k):
                return []
        return Node()
    def visit(self, *a, **k): pass
setattr(h5py, 'File', DummyH5)
# pandas
setattr(pandas, 'read_csv', lambda *a, **k: None)
# matplotlib
setattr(matplotlib_pyplot, 'figure', lambda *a, **k: None)
setattr(matplotlib_pyplot, 'subplots', lambda *a, **k: (None,))
setattr(matplotlib_pyplot, 'show', lambda *a, **k: None)
setattr(matplotlib_pyplot, 'plot', lambda *a, **k: None)

# Prepare argv and run scripts with --help to validate CLI wiring

def run_help(path, title):
    print(f"\n--- Running --help for {title} ---")
    sys.argv = [path, '--help']
    # Run in isolated namespace as __main__
    runpy.run_path(path, run_name='__main__')

# Execute smoke tests
run_help('artifact_clean.py', 'artifact_clean.py')
run_help('preprocess.py', 'preprocess.py')
run_help('process/5c_process.py', 'process/5c_process.py')

print('\nSmoke test completed successfully.')
