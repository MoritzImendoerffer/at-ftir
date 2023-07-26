#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 00:01:56 2017

@author: moritz

data analysis of AT-FTIR samples using clustermaps and igraph
"""

#%% Import section
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import igraph as igraph

#%% Data input section
raw = pd.read_pickle('time_appended_data.p')
raw.drop('index', axis=1, inplace=True)
# get date and time in the index and drop the name column
data = raw.set_index(['date', 'time'])
data.drop('name', axis=1, inplace=True)

#%% Clustermap

# slice the desired wavelengths
time = 0
wave_min = 1450
wave_max = 1700
cut = data.xs(time, level=1).T
mask = (cut.index.values < wave_max) & (cut.index.values > wave_min)
cut = cut.loc[mask,:]

# drop the first blank run from october because it is useless
#cut.drop('171017', axis=1, inplace=True)

# make correlation between runs
cor = cut.corr()

# make the clustermap
sns.set(font_scale=1.4)

plot = sns.clustermap(cor, figsize=(10,10), 
                      cmap=plt.cm.magma, metric="correlation")
file_name = '../plots/clustermaps/clustermap_t{}_{}_{}.png'\
            .format(time, wave_min, wave_max)
plt.savefig(file_name, dpi=190)

#%% Create adjacency matrix
# slice the desired wavelengths
time = 100
wave_min = 1400
wave_max = 1700
cut = data.xs(time, level=1).T
mask = (cut.index.values < wave_max) & (cut.index.values > wave_min)
cut = cut.loc[mask,:]
cor = cut.corr(method='kendall')

adj = cor.copy(deep=True).values
mask = adj > 0.95
adj[mask] = 1
adj[~mask] = 0
np.fill_diagonal(adj,0)




g = igraph.Graph(directed=False)
kwds = {'mode':'LOWER'}
graph = g.Adjacency(adj.tolist(), **kwds) 
#graph.vs['name'] = cor.index.values.tolist()

layout = graph.layout_random()
layout = graph.layout_circle()
#layout = graph.layout_reingold_tilford()
layout = graph.layout_fruchterman_reingold()

c = cor.index.values.tolist()
kwds = {'opacity':0.6, 'vertex_label':c}

plot = igraph.plot(graph.as_undirected(), 'igraph.png',layout=layout, **kwds)
plot.show()