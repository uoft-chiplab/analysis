# -*- coding: utf-8 -*-
"""
2023-10-31 (Spooky)
@author: Chip Lab

Analyzes ac dimer association field wiggle scans


CD working on this... but too brain dead
"""

from data_class import Data
from library import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os

pd.set_option('display.max_rows', 1000)

data_folder = 'data'

colors = ["blue", "red", 
		  "green", "orange", 
		  "purple", "black", "pink"]

metadata = {}

filename = '2023-10-30_E_e.dat'
param = 'wiggle time'
xlabel = 'Time (ms)'
ylabel = 'sum95'

run = Data(filename, path = data_folder, metadata=metadata)
run.subsets = 9
run.shot_num = 40
run.bg_shots_num = 8

run.data = run.data.drop(run.data[run.data['cyc'] > \
 					  run.subsets*(run.shot_num+run.bg_shots_num)].index)

# STILL WORKING ON THE BELOW, PLZ FIX LOL
for i in range(run.subsets):
	bg_start_i = 40*(i+1) + 8*i
	bg_sum95 = run.data.loc[run.data['cyc'].isin(range(bg_start_i,
			 bg_start_i+run.bg_shots_num,1))]['sum95'].mean()
	run.data.loc[run.data['cyc'].isin(range(bg_start_i,
			 bg_start_i+run.bg_shots_num,1))]['sum95'].mean()

run.group_by_mean(param)

plt.figure()
plt.errorbar(run.avg_data[param], run.avg_data['sum95'], 
			 yerr = run.avg_data['em_sum95'], fmt = 'bo', capsize = 2)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()