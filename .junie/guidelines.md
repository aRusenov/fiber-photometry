This is a standard python fiber photometry processing pipeline.

There are four main steps (i.e. scripts). Each step outputs a data file and subsequent steps use the output of the previous step:
- artifact_clean.py - provides a GUI for manually removing artifacts
- preprocess.py - preprocesses raw data by performing signal filtering, photobleaching correction and motion correction. Computes DFF and z-score of the entire recording.
- process/5c_process.py - extracts peri-event transients for relevant events in a 5CSRTT task.
- plot/5c_plot_group.py - plots the results of the 5c_process.py step.

There are currently no unit tests.