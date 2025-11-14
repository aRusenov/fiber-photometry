import os

files_dir = '/Users/atanas/Documents/workspace/data/backup/Data/fat-opto/PFC NAC opsin/'

ind_name = [8, 13]
ind_condition = [14, 18]

list_files = os.listdir(files_dir)  # get a list of al filesnames in directory
files = []
for i_files in range(len(list_files)):
    if list_files[i_files].startswith('!'):
        files.append(list_files[i_files])

latency_data = []  # collect all premature latencies across files

for i_files in range(len(files)):
    filename = files_dir + files[i_files]
    name = files[i_files]

    if name.startswith('!'):
        file_name = name.split(".")
        mouse = file_name[1][ind_name[0]:ind_name[1]]
        cond = file_name[1][ind_condition[0]:ind_condition[1]]

        fid = open(filename)
        lines = fid.readlines()
        fid.close()

        box = int(lines[9][5:7])
        date_start = lines[4][12:20]
        date_end = lines[5][10:18]
        total_trials = int(float(lines[13][7:-2]))

        X_variable = np.array(convert(filename, variable="X"))[0:(total_trials * 25)]

        for i_trial in range(total_trials):
            outcome = X_variable[11 + 25 * i_trial]  # trial result indicator
            if outcome == 3:  # premature
                latency = X_variable[7 + 25 * i_trial]  # premature latency
                stim_flag = int(X_variable[24 + 25 * i_trial])  # 1 = stim, 0 = no stim
                latency_data.append({
                    "mouse": mouse,
                    "condition": cond,
                    "box": box,
                    "date_start": date_start,
                    "date_end": date_end,
                    "trial": i_trial + 1,
                    "latency": latency,
                    "stim": stim_flag
                })

# Make dataframe of all raw premature latencies
latency_df = pd.DataFrame(latency_data)

# Save to Excel
latency_df.to_excel(files_dir + "premature_latencies_raw.xlsx", index=False)
