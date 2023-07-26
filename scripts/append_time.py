#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 23:48:40 2017

@author: moritz

Script which takes the raw dataframe, and appends last row of each experiment
as many times to the data until each experiment has the same number of
timepoints

Assumes that the last measurement represents an equilibrium
"""

#%% Import Section
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

#%% Load data section

# the raw dataframe from exported data
# colums: date, name, 3497 .. 748 (wavenumbers)
# index: time as strings
raw = pd.read_pickle('../export/one_to_rule_them_all.p')

# convert time to numeric values
raw.reset_index(inplace=True)
raw.time = raw.time.apply(pd.to_numeric)



# loop over each group, extract the data, append the last row until max time
# and create a new dataframe from it

gr = raw.groupby('date')

max_time = raw.time.max()
for index, item in enumerate(gr):
    # val contains the dataframe for given experiment
    # name contains the date of the experiment
    name, val = item
    # at first iteration create a new dataframe
    if index == 0:
        # get the last entry
        last = val.iloc[-1,:].copy(deep=True)
        last_time = last.time
        
        data = val.copy(deep=True)
        # change time of last entry
        for i in range(last_time+1, max_time+1):
            last.time = i
            data = data.append(last, ignore_index=True)
        
        # data contains now the appended dataframe for first experiments
    else:
        last = val.iloc[-1,:].copy(deep=True)    
        last_time = last.time
        
        # now do the same thing again, but append this dataframe to the first one
        sub_data = val.copy(deep=True)
        for i in range(last_time+1, max_time+1):
            last.time = i
            sub_data = sub_data.append(last,ignore_index=True)
        
        data = data.append(sub_data, ignore_index=True)

#%%    
data.reset_index(inplace=True)
data.drop('index', axis=1, inplace=True)
#%% now save the dataframe
data.to_pickle('time_appended_data.p')