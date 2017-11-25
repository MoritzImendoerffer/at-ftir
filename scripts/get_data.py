# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:14:20 2017

@author: imendoer

This function extracts the AT-FTIR data from a folder with 'data_ir' in it
and generates an excel as well as a pickled dataframe

It works.
"""

import os
import pandas as pd
import numpy as np
import re

#%%

def sort_int_nicely(l):
    '''
    This function sorts integers not like 1, 10, 11, 12, 2, 20, 21, 22
    but 1, 2, 10,11,12,20,21,22
    Because the data needs to be resorted by time after the export and import,
    this function is very handy together with the reindex function of pandas

    input: list or array
    output: human sorted list of strings
    '''
    l = list(l)
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

    l.sort(key=alphanum_key )

    l = [str(i) for i in l]
    return l

#%%

# get all folders in current directory
dirs = [entry.path for entry in os.scandir('./') if entry.is_dir()]

# etract those with 'data_ir' in folder name
dirs = [item for item in dirs if 'data_ir' in item.lower()]


#%% Loop over all folders in dirs and extract metainformation as well as data

for data_dir in dirs:
    print(data_dir)
    # the date of the experiment as to be the first item in the folder name
    # [2:] gets rid of './xxxxx'
    exp_date = data_dir.split(' ')[0][2:8]

    # get all files in folder
    files = os.listdir(data_dir)


    #%% Loop over all files, extract metadata from filenames and x,y values

    # empty list to store the data in
    data = []

    #for file in files:
    x = []
    for index, file in enumerate(files):

        # make the filepath
        file_path = data_dir + '/' + file
        # all exported files should have 'snm' in the name
        if 'snm' in file:
            with open(file_path, 'r') as f:
                # every file contains x and y values seperated by tab or comma

                y = []
                for lind, line in enumerate(f):

                    # take the metadata in filenames (date, name, time)
                    # only ones
                    if lind == 0:
                        # as decided by us the experiment name follows the date
                        name = file.split('.')[2]

                        # remove whitespaces before or after the name
                        if name[0] == ' ':
                            name = name[1:]
                        if name[-1] == ' ':
                            name = name[0:-1]

                        # the date is given in the first six entries
                        date = file[0:6]

                        # timepoints are given as ints after the point
                        time = file.split('.')[-1]

                    # gather all y values in one list
                    try:  # try to split at comma
                        y_value = float(line.strip('\n').split(',')[1])
                    except IndexError:  # otherwise split at tab
                        y_value = float(line.strip('\n').split('\t')[1])

                    # append y values to y
                    y.append(y_value)

                    '''
                    gather the wavenumber in a list. Because all exports
                    should have the same interval it should be enough
                    to do this only once. BUT THIS SHOULD BE IMPROVED WITH
                    A CHECK IF REALLY ALL FILES HAVE THE SAME X AXIS
                    '''
                    if index==0:
                        x_value = float(line.strip('\n').split(',')[0])
                        x.append(x_value)

                # get a flat list (the starred expression unpacks the y list)
                data_all = [date, name, time, *y]

                # add the data to the overall data list
                data.append(data_all)
    #%% make a pandas dataframe with wavenumbers as colums
    wave = list(np.array(x).round(0))
    df = pd.DataFrame(data, index=None,
                      columns=['date', 'name', 'time', *wave])
    a = sort_int_nicely(df.time)
    df = df.set_index('time').reindex(a)

    #%% pickle the dataframe
    df.to_pickle('{}_{}_raw.p'.format(exp_date, name))

    #%%
    writer = pd.ExcelWriter('{}_{}_raw.xlsx'.format(exp_date, name))
    df.to_excel(writer)
    writer.save()
    writer.close()


'''
The function below should be implemented in this script in order to make
a nice excel export (time sorted in human style)
'''


