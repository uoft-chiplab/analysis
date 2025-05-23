# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 16:32:34 2025

@author: Chip lab
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data_class import Data
from library import styles

file = '2025-04-10_H_e.dat'

xname = 'time'
ff = 0.88

run = Data(file)

# correct the data
run.data['c9'] = run.data['c9'] * ff
run.data[xname] = run.data[xname] + 26.6-11.0 # cause we shaved off 11ms

# group by mean
run.group_by_mean(xname)

# select signal and bg dfs
df = run.avg_data.loc[run.avg_data.VVA > 0]
bg_df = run.avg_data.loc[run.avg_data.VVA == 0]

# compute transfer
df['transfer'] = (df['c5'] - bg_df['c5'].mean())/df['c9']
df['em_transfer'] = df['em_c5']/df['c9'] # roughly

# plot
fig, axes = plt.subplots(1,2, figsize=(8,4))
axs = axes.flatten()

ax = axs[0]
x = df['time']
y = df['transfer']
yerr = df['em_transfer']
ax.set(xlabel='Time (ms)', ylabel='Transfer')
ax.errorbar(x, y, yerr, **styles[0])

fig.tight_layout()

plt.show()