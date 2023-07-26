# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:47:39 2017

@author: imendoer
This script analyses the pickled dataframe generated by get_data.py
"""

import pandas as pd
import numpy as np
from include.helpers import sort_int_nicely
import os
import re


def main():
    """"
    Loop over all files in the current folder with .p as name, extract the date
    from the filename and reads the file as dataframe. The .p files are generated
    by the get_data.py function and contain the same information as the excel
    files.
    """
    # get all files in current folder
    rel_path = './export'
    files = os.listdir(rel_path)

    # get all files which end with .p
    file = [f for f in files if '_raw.p' in f]

    for index, f in enumerate(file):
        file_path = rel_path + '/' + f
        # extract the date of the experiment
        name = f.split('_')[0]

        # read in the pickled dataframe
        df = pd.read_pickle(file_path)

        # get all blank values
        mask = df.name.str.contains('blank')

        blank = df.loc[mask, :]

        # sort time values in human form (e.g. 1,2,3..10,11) a = list of strings
        a = sort_int_nicely(blank.time)

        # set time as index (now they are strings) and resort by a
        blank=blank.set_index('time').reindex(a)

        # extract y and x values of the blank
        y_blank = blank.iloc[:, 3:]
        x_blank = blank.iloc[:, 2]

        # get all datapoints which are no blanks
        experiment = df.loc[~mask, :]

        # make sure the data is sorted in human style by the time
        a = sort_int_nicely(experiment.time)
        experiment = experiment.set_index('time').reindex(a)

        # extract y and x values of the experiment
        y_exp = experiment.iloc[:, 3:]
        x_exp = experiment.iloc[:, 2]

        # trimm the experiment to match the length of the blanks
        if len(y_blank) < len(y_exp):   # less blank points than experiment points
            y_exp = y_exp[0:len(y_blank)] # trimm the experiment
        else: # otherwise trimm the blank
            y_blank = y_blank[0:len(y_exp)]

        # subtract the blanks
        y_corrected = y_exp.values - y_blank.values

        # make a new dataframe from the corrected values with wavenumber as columns
        df_corrected = pd.DataFrame(y_corrected, columns=y_blank.columns)

        # Transpose the data --> time are now the columns, wavenumber is the index
        df_corrected = df_corrected.T.reset_index()

        # rename index column to 'x'
        df_corrected.columns.values[0] = 'wavenumber'

        # convert all timepoints to strings (for slicing with .loc)
        df_corrected.columns = [str(item) for item in df_corrected.columns]

        # save the dataframe as pickle
        df_corrected.to_pickle('{}_corrected.p'.format(name))

        # save the dataframe as excel
        writer = pd.ExcelWriter('{}_corrected.xlsx'.format(name))
        df_corrected.to_excel(writer)
        writer.save()
        writer.close()

if __name__ == '__main__':
    main()