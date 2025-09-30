#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19th of June

@author: kooij_k
Preprocessing of 5choice med output data with opto experiment to excel


# description of the variables in the reversal learning medpc output
\ Overview of GLOBAL variables
\ a = counter of the total number of trials, used for array
\ b = counter of the number of started trials in current stage (to be checked with criterion)
\ c = Current stage (1 = magazine, 2 = training_1, 3 = training_2, 4 = SD_16, 5 = SD_8, 6 = SD_4, 7 = SD_2, 8 = SD_1.5, 9 = SD_1, etc SD1)
\ d = stimulus duration (sec)
\ e = counter correct responses within stage (to be checked with criterion)
\ f = Start trial after time out (f=1)
\ g = list for random stimulus duration (vSD session c = 12)
\ h = session time in hours (24h format)
\ i = session time in days
\ j = list for 3-choice random stimulus presentation
\ k = iti duration (sec)
\ l = limited hold
\ m = list for random ITI duration (vITI session c = 10)
\ n = current accuracy (to be checked with criterion)
\ o = current % omissions (to be checked with criterion)
\ p = random presentation of stimulus position (from list q)
\ q = array for randomised stimulus presentation
\ r = list for randomized pellet delivery (ITI magazine training)
\ s = session time in sec (cummulative format)
\ t = used to retrieve computer time hours
\ u = used to retrieve computer time minutes
\ v = used to retrieve computer time seconds
\ w = data array for moving average % omissions
\ x = data array for output
\ y = data array to store 24h data summaries
\ z = data array for moving average accuracy

\ x(0) = Start time of next trial; ignition of magazine light (sec)
\ x(1) = Computer time (Hours)
\ x(2) = Stage (1 = magazine, 2 = training_1, 3 = training_2, 4 = SD_16, 5 = SD_8, 6 = SD_4, 7 = SD_2, 8 = SD_1.5, 9 = SD_1, etc SD1)
\ x(3) = Stimulus location (1, 2 ,3, 4, 5)
\ x(4) = Number of np responses during eat ITI
\ x(5) = ITI during this trial
\ x(6) = Number of inappropriate magazine responses during ITI
\ x(7) = Latency of response (either premature, correct or incorrect)
\ x(8) = Magazine latency (s) after a correct response;
\ x(9) = Stimulus duration (vSD session)
\ x(10) = Indicator of trial result: 0 = other mistake, see x(11), 1 = omission, 2 = correct response
\ x(11) = Indicator of trial result: 3 = premature response, 4 = incorrect response
\ x(12) = Position of premature / correct / incorrect nosepoke (1, 2, 3, 4, 5)
\ x(13) = Variabele o; current % omissions (to be checked with criterion)
\ x(14) = Variabele n; current accuracy (to be checked with criterion)
\ x(15) = Variabele e; counter for correct responses in stage
\ x(16) = End time of TO
\ x(17) = Magazine latency (s) after a TO; time between start trial after TO and magazine response after TO
\ x(18) = Number of responses during TO
\ x(19) = Number of inappropriate magazine responses during stimulus presentation
\ x(20) = Number of compulsive responses np1 - 5
\ x(21) = Counter for omissions
\ x(22) = Counter for premature responses
\ x(23) = counter for incorrect responses
\ x(24) = opto stim (1) or not (0)
\______________________________________________

\**********************************************
\ array format 24 hour summaries
\ y(0) = Obtained food rewards in the past 24h (earned + free)
\ y(1) = Free food rewards in the past 24h
\ y(2) = Total number of trials in the past 24h (parameter a)
\ y(3) = Number of stages passed in the past 24h
\ y(4) =


"""
# INPUTS

# files_directory with medpc files
files_dir = 'L://meyelab//Ongoing//c_Project PFC_Dopaceptive//2_Research_Data//10_PFC D1r_BLA_NAC photo opto//Data//5 choice//opto stim//Analysed 29 august//Ctr mice//'

ind_name = [8, 13]
ind_condition = [14, 18]

###########################################################################3
# packages

import numpy as np
import re
import matplotlib.pyplot as plt
# import seaborn as sns
import time
import os

os.getcwd()
import pandas as pd
import matplotlib.style
import sys
# import openpyxl
from datetime import date


#################################################################################
# own functions


def convert(fileName, variable="A"):
    # this is a function to extract 1 letter of the med files
    fid = open(fileName)  # open file
    lines = fid.readlines()  # store lines in list
    fid.close()

    # extract all numbers between your variable and the next letter
    flag = variable + ":\n"  # find variable flag (eg A:) at the end of each line is a \n
    start_index = next(i for i, x in enumerate(lines) if x == flag) + 1
    lines = lines[start_index:]
    stop_index = 0

    while not (re.search('[a-zA-Z]', lines[stop_index])):  # stops at the next variable or comment
        lines[stop_index] = cleanLines(lines[stop_index])  # clean lines
        stop_index += 1
        if stop_index == len(lines):
            break

    lines = lines[:stop_index]
    # TEC = cleanLines(lines) #break lines, store time,event in list Time_Event
    TEC = [];
    #    for index in range(len(lines)):
    #        TEC.append(float(lines[index][0]))
    for i_lines in range(len(lines)):
        for i_list in range(len(lines[i_lines])):
            TEC.append(float(lines[i_lines][i_list]))
    return TEC


def cleanLines(item):
    # function for med extraction
    start = item.index(":") + 1
    item = item[start:]
    item = item.replace("\n", "")
    item = item.split()
    return item


def get_index_positions(list_of_elems, element):
    # function for med extraction
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list


######################################################################################
# start script

# get list of all files within folder
list_files = os.listdir(files_dir)  # get a list of al filesnames in directory
files = []
for i_files in range(len(list_files)):
    if list_files[i_files].startswith('!'):
        files.append(list_files[i_files])

# create empty dataphrame
colnames = ['mouse', 'condition', 'date start', 'date end', 'box', 'total days', '#trials', '#trials_stim',
            '#trials_no_stim', '#correct_stim', '#incorrect_stim', '#omissions_stim', '#premature_stim',
            '#correct_no_stim', '#incorrect_no_stim', '#omissions_no_stim', '#premature_no_stim', '%correct_stim',
            '%incorrect_stim', '%omissions_stim', '%premature_stim', '%correct_no_stim', '%incorrect_no_stim',
            '%omissions_no_stim', '%premature_no_stim', '%correct', '%incorrect', '%omissions', '%premature',
            '%correct_Remmelink_calc', '%omissions_Remmelink_calc', '%premature_Remmelink_calc']
results_day = pd.DataFrame(
    np.zeros((len(files), len(colnames))));  # make an empty panda dataframe dependent on number of files and colums
results_day = results_day.astype('object');
results_day.columns = colnames

# extract info from each file
for i_files in range(len(files)):
    filename = files_dir + files[i_files]
    name = files[i_files]

    # extract name of mouse
    if name.startswith('!'):  # only select files with ! = med files
        file_name = name.split(".")
        name = file_name[1][ind_name[0]:ind_name[1]]
        cond = file_name[1][ind_condition[0]:ind_condition[1]]

        # open document
        fid = open(filename)  # open file
        lines = fid.readlines()  # store text in list
        fid.close()

        # extract variables with 1 number
        box = int(lines[9][5:7])
        date_start = lines[4][12:20]
        date_end = lines[5][10:18]
        days = date(int(date_end[6:8]), int(date_end[0:2]), int(date_end[3:5])) - date(int(date_start[6:8]),
                                                                                       int(date_start[0:2]),
                                                                                       int(date_start[3:5]))
        days = days.days
        total_trials = int(float(lines[13][
                                     7:-2]))  # int(Y_variable[2 + 5* (days-2)])   #does not take trials in last 24hours into account (yet)

        # extract arrays
        Y_variable = np.array(convert(filename, variable="Y"))[0: (5 * days)]
        X_variable = np.array(convert(filename, variable="X"))[0:(total_trials * 25)]

        omissions_stim = 0;
        premature_stim = 0;
        incorrect_stim = 0;
        correct_stim = 0
        omissions_no_stim = 0;
        premature_no_stim = 0;
        incorrect_no_stim = 0;
        correct_no_stim = 0
        for i_trial in range(total_trials):
            if X_variable[10 + 25 * i_trial] == 1:
                if X_variable[24 + 25 * i_trial] == 1:
                    omissions_stim = omissions_stim + 1
                else:
                    omissions_no_stim = omissions_no_stim + 1
            elif X_variable[10 + 25 * i_trial] == 2:
                if X_variable[24 + 25 * i_trial] == 1:
                    correct_stim = correct_stim + 1
                else:
                    correct_no_stim = correct_no_stim + 1
            elif X_variable[11 + 25 * i_trial] == 3:
                if X_variable[24 + 25 * i_trial] == 1:
                    premature_stim = premature_stim + 1
                else:
                    premature_no_stim = premature_no_stim + 1
            elif X_variable[11 + 25 * i_trial] == 4:
                if X_variable[24 + 25 * i_trial] == 1:
                    incorrect_stim = incorrect_stim + 1
                else:
                    incorrect_no_stim = incorrect_no_stim + 1

        total_stim_trials = correct_stim + incorrect_stim + omissions_stim + premature_stim
        total_no_stim_trials = correct_no_stim + incorrect_no_stim + omissions_no_stim + premature_no_stim

        # save info in dataframa
        results_day['mouse'][i_files] = name
        results_day['condition'][i_files] = cond
        results_day['date start'][i_files] = date_start
        results_day['date end'][i_files] = date_end
        results_day['box'][i_files] = box
        results_day['total days'][i_files] = days
        results_day['#trials'][i_files] = total_trials
        results_day['#trials_stim'][i_files] = total_stim_trials
        results_day['#trials_no_stim'][i_files] = total_no_stim_trials
        results_day['#correct_stim'][i_files] = correct_stim
        results_day['#correct_no_stim'][i_files] = correct_no_stim
        results_day['#incorrect_stim'][i_files] = incorrect_stim
        results_day['#incorrect_no_stim'][i_files] = incorrect_no_stim
        results_day['#omissions_stim'][i_files] = omissions_stim
        results_day['#omissions_no_stim'][i_files] = omissions_no_stim
        results_day['#premature_stim'][i_files] = premature_stim
        results_day['#premature_no_stim'][i_files] = premature_no_stim

        # old caclulation
        results_day['%correct_stim'][i_files] = np.round(correct_stim / total_stim_trials * 100, 2)
        results_day['%correct_no_stim'][i_files] = np.round(correct_no_stim / total_no_stim_trials * 100, 2)
        results_day['%incorrect_stim'][i_files] = np.round(incorrect_stim / total_stim_trials * 100, 2)
        results_day['%incorrect_no_stim'][i_files] = np.round(incorrect_no_stim / total_no_stim_trials * 100, 2)
        results_day['%omissions_stim'][i_files] = np.round(omissions_stim / total_stim_trials * 100, 2)
        results_day['%omissions_no_stim'][i_files] = np.round(omissions_no_stim / total_no_stim_trials * 100, 2)
        results_day['%premature_stim'][i_files] = np.round(premature_stim / total_stim_trials * 100, 2)
        results_day['%premature_no_stim'][i_files] = np.round(premature_no_stim / total_no_stim_trials * 100, 2)

        results_day['%correct'][i_files] = np.round((correct_stim + correct_no_stim) / total_trials * 100, 2)
        results_day['%incorrect'][i_files] = np.round((incorrect_stim + incorrect_no_stim) / total_trials * 100, 2)
        results_day['%omissions'][i_files] = np.round((omissions_stim + omissions_no_stim) / total_trials * 100, 2)
        results_day['%premature'][i_files] = np.round((premature_stim + premature_no_stim) / total_trials * 100, 2)

        results_day['%correct_Remmelink_calc'][i_files] = np.round((correct_stim + correct_no_stim) / (
                    correct_stim + correct_no_stim + incorrect_stim + incorrect_no_stim) * 100, 2)
        results_day['%omissions_Remmelink_calc'][i_files] = np.round((omissions_stim + omissions_no_stim) / (
                    correct_stim + correct_no_stim + incorrect_stim + incorrect_no_stim + omissions_stim + omissions_no_stim) * 100,
                                                                     2)
        results_day['%premature_Remmelink_calc'][i_files] = np.round((premature_stim + premature_no_stim) / (
                    correct_stim + correct_no_stim + incorrect_stim + incorrect_no_stim + premature_stim + premature_no_stim) * 100,
                                                                     2)

# write to excel
writer = pd.ExcelWriter(files_dir + 'results_day.xlsx',
                        engine='xlsxwriter')  # Create a Pandas Excel writer using XlsxWriter as the engine.
results_day.to_excel(writer,
                     sheet_name='results_day')  # Write each dataframe to a different worksheet. you could write different string like above if you want
writer.close()  # Close the Pandas Excel writer and output the Excel file.

# def barplot_averages_1_con(df, data_col,con_col, cur_con, cur_x_label,colors_plot, titel, cur_channel, ylabel, files_dir, fontplt, fsize, indv_points,xaxis_info, y_tick_steps):
#     fig, ax = plt.subplots(figsize = (6.5,5.5))

#     bar_width = 0.35; opacity = 0.8; index = 1
#     nr_timestamps = int(len(data_col))
#     x_ticks_lab = []

#     for i_times in range(nr_timestamps):
#         subset_df = df[data_col[i_times]].loc[df[con_col] == cur_con]
#         numberofmice = int(len(subset_df))
#         ax.bar(index + (bar_width + 0.1 * bar_width) * i_times, subset_df.mean(axis=0),bar_width, yerr =  subset_df.std(axis=0)/sqrt(numberofmice) , alpha = opacity, color = colors_plot, edgecolor = 'black', capsize = 10,label = data_col[i_times],  linewidth = 3)
#         x_ticks_lab.append(xaxis_info[3][i_times] + ' to ' + xaxis_info[3][i_times+1])

#     if indv_points == 1:
#         for i_mouse in range(numberofmice):
#             ax.plot(index + (bar_width * i_times) +(bar_width * 0.1) , subset_df.iloc[i_mouse], '*', color='grey')

#     if indv_points == 2:
#         xaxis = np.arange(0, (nr_timestamps)) * (bar_width + 0.1 * bar_width) + index
#         mice = df['name'].loc[df[con_col] == cur_con].value_counts()
#         mice = list(mice.index)

#         for i_mouse in range(len(mice)):
#             yvalue = []
#             for i_con in range(nr_timestamps):
#                 subset_df = df[data_col[i_con]].loc[(df['name'] == mice[i_mouse]) & (df[con_col] == cur_con)]
#                 yvalue.append(subset_df.iloc[0])

#             plt.plot(xaxis , yvalue,  color='black', linewidth = 0.3)

#     x_ticks = xaxis
#     plots_layout(ax, titel, ylabel, cur_x_label, x_ticks, fontplt, fsize, y_tick_steps )
#     ax.set_xticklabels(x_ticks_lab)

#     #ax.legend()
#     plt.tight_layout()
#     plt.savefig(files_dir  + titel + '_' + cur_channel + '.png', transparent = True)
#     plt.savefig(files_dir  + titel + '_' + cur_channel + '.svg', transparent = True)
#     plt.show()

#     figLegend = plt.figure(figsize = (2,1.3))
#     # produce a legend for the objects in the other figure
#     plt.figlegend(*ax.get_legend_handles_labels(), frameon = False)
#     plt.axis('off')

#     # save the two figures to files
#     figLegend.savefig(files_dir + "legend_bars.png", transparent = True)
#     plt.show()
#     return plt


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sqrt


def barplot_single_column(df, data_col, group_col, title, ylabel, save_path, fontplt='Arial', fsize=18,
                          show_individual=2, colors_plot='skyblue', y_tick_steps=1):
    plt.rcParams['font.family'] = fontplt
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    unique_groups = df[group_col].unique()
    unique_groups = ['mock', 'stim']
    bar_width = 0.35
    x_positions = np.arange(len(unique_groups))

    means = []
    stds = []

    for i, group in enumerate(unique_groups):
        group_data = df[df[group_col] == group][data_col]
        means.append(group_data.mean())
        stds.append(group_data.std() / sqrt(len(group_data)))

        ax.bar(x_positions[i], means[i], yerr=stds[i], width=bar_width, color=colors_plot, edgecolor='black',
               capsize=10)

        if show_individual == 1:
            ax.plot([x_positions[i]] * len(group_data), group_data, '*', color='grey')

    if show_individual == 2:
        unique_mice = df['mouse'].unique()

        for i_mouse in range(len(unique_mice)):
            yvalue = []
            for i_con in range(len(unique_groups)):
                subset_df = df[data_col].loc[
                    (df['mouse'] == unique_mice[i_mouse]) & (df[group_col] == unique_groups[i_con])]
                yvalue.append(subset_df.iloc[0])

            plt.plot(x_positions, yvalue, color='black', linewidth=0.3)

    ax.set_title(title, fontsize=fsize + 2)
    ax.set_ylabel(ylabel, fontsize=fsize)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(list(unique_groups), fontsize=fsize)
    ax.tick_params(axis='y', labelsize=fsize)

    if y_tick_steps:
        ax.yaxis.set_major_locator(plt.MultipleLocator(y_tick_steps))

    plt.tight_layout()
    plt.savefig(f"{save_path}/{title}.png", transparent=True)
    # plt.savefig(f"{save_path}/{title}.svg", transparent=True)
    plt.show()

    return plt


# barplot_averages_1_con(results_day,'%premature_stim', 'condition', , descrip_timelock+ ' (s)', colors_gfp, 'GFP', descrip_timelock,data_type, files_dir_plt, f_font, f_size_bar-5, 2,cur_xaxis, y_lim )

barplot_single_column(df=results_day, data_col='%premature_stim', group_col='condition', title='%premature stim trials',
                      ylabel='%premature stim trials', save_path=files_dir, show_individual=2, y_tick_steps=10)
barplot_single_column(df=results_day, data_col='%premature_no_stim', group_col='condition',
                      title='%premature no stim trials', ylabel='%premature no stim trials', save_path=files_dir,
                      show_individual=2, y_tick_steps=10)

barplot_single_column(df=results_day, data_col='#trials', group_col='condition', title='#trials', ylabel='#trials',
                      save_path=files_dir, show_individual=2, y_tick_steps=50)

barplot_single_column(df=results_day, data_col='%omissions_stim', group_col='condition', title='%omissions_stim',
                      ylabel='%omissions_stim', save_path=files_dir, show_individual=2, y_tick_steps=10)
barplot_single_column(df=results_day, data_col='%omissions_no_stim', group_col='condition', title='%omissions_no_stim',
                      ylabel='%omissions_no_stim', save_path=files_dir, show_individual=2, y_tick_steps=10)

barplot_single_column(df=results_day, data_col='%incorrect_stim', group_col='condition', title='%incorrect_stim',
                      ylabel='%incorrect_stim', save_path=files_dir, show_individual=2, y_tick_steps=10)
barplot_single_column(df=results_day, data_col='%incorrect_no_stim', group_col='condition', title='%incorrect_no_stim',
                      ylabel='%incorrect_no_stim', save_path=files_dir, show_individual=2, y_tick_steps=10)

barplot_single_column(df=results_day, data_col='%correct_stim', group_col='condition', title='%correct_stim',
                      ylabel='%correct_stim', save_path=files_dir, show_individual=2, y_tick_steps=10)
barplot_single_column(df=results_day, data_col='%correct_no_stim', group_col='condition', title='%correct_no_stim',
                      ylabel='%correct_no_stim', save_path=files_dir, show_individual=2, y_tick_steps=10)

barplot_single_column(df=results_day, data_col='%premature', group_col='condition', title='%premature all trials',
                      ylabel='%premature all trials', save_path=files_dir, show_individual=2, y_tick_steps=10)
barplot_single_column(df=results_day, data_col='%omissions', group_col='condition', title='%omissions all trials',
                      ylabel='%omissions all trials', save_path=files_dir, show_individual=2, y_tick_steps=10)
barplot_single_column(df=results_day, data_col='%incorrect', group_col='condition', title='%incorrect all trials',
                      ylabel='%incorrect all trials', save_path=files_dir, show_individual=2, y_tick_steps=10)
barplot_single_column(df=results_day, data_col='%correct', group_col='condition', title='%correct all trials',
                      ylabel='%correct all trials', save_path=files_dir, show_individual=2, y_tick_steps=10)

barplot_single_column(df=results_day, data_col='%premature_Remmelink_calc', group_col='condition',
                      title='%premature Remmelink', ylabel='%premature all trials', save_path=files_dir,
                      show_individual=2, y_tick_steps=10)
barplot_single_column(df=results_day, data_col='%omissions_Remmelink_calc', group_col='condition',
                      title='%omissions Remmelink', ylabel='%omissions all trials', save_path=files_dir,
                      show_individual=2, y_tick_steps=10)
barplot_single_column(df=results_day, data_col='%correct_Remmelink_calc', group_col='condition',
                      title='%correct Remmelink', ylabel='%correct all trials', save_path=files_dir, show_individual=2,
                      y_tick_steps=10)



