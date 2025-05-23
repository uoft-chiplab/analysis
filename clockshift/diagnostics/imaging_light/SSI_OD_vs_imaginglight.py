# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:14:49 2025

@author: Chip lab
"""

import numpy as np
from data_class import Data
from library import styles
import matplotlib.pyplot as plt

import pandas as pd

group_name = 'TOF'

file1 = "2025-04-11_J_e.dat"

run1 = Data(file1)
df1 = run1.data

extra_files = ["2025-04-11_L_e.dat",
			   "2025-04-11_M_e.dat"]

extra_TOFs = [0.52, 0.021]
extra_Zwires = [1.1, 1.5]

new_dfs = []

for file, TOF, Zwire in zip(extra_files, extra_TOFs, extra_Zwires):
	run = Data(file)
	new_df = run.data
	new_df['TOF'] = TOF
	new_df["Z wire"] = Zwire
	new_dfs.append(new_df)

df = pd.concat([df1, *new_dfs])

fig, axes = plt.subplots(2,2, figsize=(8,8))
axs = axes.flatten()

axs[0].set(xlabel="refmean", ylabel="ODmean")
axs[1].set(xlabel="ref - at", ylabel="ODmean")
axs[2].set(xlabel="refmean", ylabel="ODmax")
axs[3].set(xlabel="-ln(ref/at)", ylabel="ref - at", ylim=[5, 70], xlim=[-0.032, -0.01])

fig.suptitle(file1)

TOFs = np.sort(df.TOF.unique())

for i, TOF in enumerate(TOFs):
	sub_df = df.loc[df.TOF==TOF]
	
	ODmax = sub_df.loc[sub_df.refmean>1000].ODmax.mean()
	
	ax = axs[0]
	label = 'ODmax={:.2f}'.format(ODmax)
	ax.plot(sub_df['refmean'], sub_df['ODmean'], **styles[i], label=label)
	
	ax = axs[1]
	label = 'TOF='+str(TOF)+'ms'
	ax.plot(sub_df['refmean'] - sub_df['atmean'], sub_df['ODmean'], **styles[i], label=label)
	
	ax = axs[2]
	ax.plot(sub_df['refmean'], sub_df['ODmax'], **styles[i])
	
	ax = axs[3]
	ax.plot(-np.log(sub_df['refmean']/sub_df['atmean']), 
		 sub_df['refmean'] - sub_df['atmean'], **styles[i])

for ax in axs[:-1]:
	ax.legend()
fig.tight_layout()

plt.show()