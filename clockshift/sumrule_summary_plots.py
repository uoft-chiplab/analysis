# -*- coding: utf-8 -*-
"""
@author: coldatoms
"""
# paths
import os
# print(os.getcwd())
proj_path = os.path.dirname(os.path.realpath(__file__))
# print(proj_path)
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')
figfolder_path = os.path.join(proj_path, 'figures')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from library import colors, markers, tintshade, tint_shade_color, plt_settings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 9999)

# save files 
files = ['sumrule_analysis_results.xlsx',
		 'loss_sumrule_analysis_results.xlsx']

### load analysis results
# for file in files:
# 	savefile = os.path.join(proj_path, file)
# 	df = pd.read_excel(savefile, index_col=0, engine='openpyxl').reset_index()
	
full_df = pd.concat((pd.read_excel(os.path.join(proj_path,f), engine='openpyxl') \
					for f in files))
	
Runs = ['2024-09-12_E_e.dat', '2024-09-18_F_e.dat', '2024-09-23_H_e.dat',
       '2024-09-24_C_e.dat']
	
df = full_df.loc[full_df.Run.isin(Runs)]

# print df
print(df[['ToTF', 'EF', 'SR_median', 'C_median', 'C_theory', 'CoSR_median', 'Run', 'Transfer']])

### select x_axis
xname = "ToTF"
xlabel = xname

### select y_axes
labels = [[xlabel,"Sumrule"],
		   [xlabel, r"Contact $C/N$ [$k_F$]"],
		   ["Theoretical Contact", r"Contact $C/N$ [$k_F$]"],
		   [xlabel, "Clock Shift"],
		   [xlabel, r"Contact over Sumrule $C/N/SR$ [$k_F$]"],
		   ["Theoretical Contact", r"Contact over Sumrule $C/N/SR$ [$k_F$]"]]

df_names = [[xname, "SR"],
		  [xname, "C"],
		  ["C_theory", "C"],
		  [xname, "CS"],
		  [xname, "CoSR"],
		  ["C_theory", "CoSR"]]

### plots
plt.rcParams.update(plt_settings)
plt.rcParams.update({"figure.figsize": [12,8]})
fig, axes = plt.subplots(2,3)
axs = axes.flatten()

transfer_types = ['loss', 'transfer']
# flip 'em
colors = [colors[1], colors[0]]
markers = [markers[1], markers[0]]

for l, transfer in enumerate(transfer_types):
	subdf = df.loc[df['Transfer'] == transfer]
	
	# labelToTF = r"ToTF"+"={:.3f}".format(ToTF)
	color = colors[l]
	marker = markers[l]
	light_color = tint_shade_color(color, amount=1+tintshade)
	dark_color = tint_shade_color(color, amount=1-tintshade)
	plt.rcParams.update({
					 "lines.markeredgecolor": dark_color,
					 "lines.markerfacecolor": light_color,
					 "lines.color": dark_color,
					 "legend.fontsize": 14})
	
	
	# Plot vs ToTF
	j = 0 # ax counter
	for ax, label_pair, df_pair in zip(axs, labels, df_names):
		ax.set(xlabel=label_pair[0], ylabel=label_pair[1])
		error = np.array(list(zip(subdf[df_pair[1]+"_median"]-subdf[df_pair[1]+"_lower"], 
							   subdf[df_pair[1]+"_upper"]-subdf[df_pair[1]+"_median"]))).T
		ax.errorbar(subdf[df_pair[0]], subdf[df_pair[1]+"_median"], yerr=error, fmt=marker,
			  label=transfer_types[l], ecolor=dark_color)
		if j == 2 or j == 5:
			ax.plot([0.7,2], [0.7,2], 'k--')
		j += 1

axs[-2].legend()

### generate table
# axs[-1].axis('off')
# quantities = ["ToTF", "EF", "SR", "CS", "C", "C/SR"]
# values = []
# 	
# table = list(zip(quantities, *values))
# the_table = axs[-1].table(cellText=table, loc='center')
# the_table.auto_set_font_size(False)
# the_table.set_fontsize(12)
# the_table.scale(1,1.5)
			
# axs[3].legend(unique_handlesToTF,unique_labelsToTF)
# axs[1].legend(unique_handlestol, unique_labelstol)

fig.tight_layout()
plt.show()

### save figure
timestr = time.strftime("%Y%m%d-%H%M%S")
summary_plots_folder = "summary_plots"
summaryfig_path = os.path.join(proj_path, summary_plots_folder)
os.makedirs(summaryfig_path, exist_ok=True)

summaryfig_name = timestr = time.strftime("%Y%m%d-%H%M%S")+'summary.png'
summaryfig_path = os.path.join(summaryfig_path, summaryfig_name)
fig.savefig(summaryfig_path)