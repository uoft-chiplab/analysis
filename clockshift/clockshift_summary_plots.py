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

from library import styles, plt_settings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 9999)

# save files 
files = [
# 		'HFT_analysis_results.xlsx',
		 'corrected_HFT_analysis_results.xlsx',
 		 # 'corrected_loss_HFT_analysis_results.xlsx',
# 		 'acdimer_lineshape_results_c5.xlsx',
# 		 'acdimer_lineshape_results_c9.xlsx',
# 		 'acdimer_lineshape_results_sum95.xlsx',
		 ]

### load analysis results
# for file in files:
# 	savefile = os.path.join(proj_path, file)
# 	df = pd.read_excel(savefile, index_col=0, engine='openpyxl').reset_index()
	
full_df = pd.concat((pd.read_excel(os.path.join(proj_path,f), engine='openpyxl') \
					for f in files))
	
Runs = ['2024-09-12_E_e.dat', '2024-09-18_F_e.dat', '2024-09-23_H_e.dat',
       '2024-09-24_C_e.dat', '2024-09-27_B_e.dat', '2024-09-27_C_e.dat',
	   '2024-10-01_F_e.dat','2024-10-02_C_e.dat',
	      '2024-10-07_C_e.dat','2024-10-07_G_e.dat', '2024-10-08_F_e.dat',
		  ]
	
# df = full_df.loc[full_df.Run.isin(Runs)]
df = full_df

# print df
df['C_diff'] = df['C_mean']-df['C_theory']
print(df[['ToTF', 'EF', 'SW_mean', 'C_mean', 'C_theory', 'CoSW_mean', 
		  'C_diff', 'Run', 'Transfer',
		  ]])


### select x_axis
xname = "ToTF"
xlabel = xname

### select y_axes
labels = [[xlabel,"Sumrule"],
		   [xlabel, r"Contact $C/N$ [$k_F$]"],
		   ["Theoretical Contact", r"Contact $C/N$ [$k_F$]"],
		   [xlabel, "Clock Shift"],
		   [xlabel, r"Contact over SW $C/N/SW$ [$k_F$]"],
		   ["Theoretical Contact", r"Contact over SW $C/N/SW$ [$k_F$]"]]

df_names = [[xname, "SW"],
		  [xname, "C"],
		  ["C_theory", "C"],
		  [xname, "CS"],
		  [xname, "CoSW"],
		  ["C_theory", "CoSW"]]

### plots
plt.rcParams.update(plt_settings)
plt.rcParams.update({"figure.figsize": [12,8]})
fig, axes = plt.subplots(2,3)
axs = axes.flatten()

transfer_types = ['transfer', 
				  'loss',
# 				  'dimer_c9',
				  ]#,'dimer_c5','dimer_sum95']
# flip 'em
# colors = [colors[2], colors[1]]
# markers = [markers[1], markers[0]]

scale_dimer = 1

for l, transfer in enumerate(transfer_types):		
	subdf = df.loc[df['Transfer'] == transfer]
	
	sty = styles[l]
	plt.rcParams.update({"legend.fontsize": 14})
	
	
	# Plot vs ToTF
	for j, (ax, label_pair, df_pair) in enumerate(zip(axs, labels, df_names)):
		ax.set(xlabel=label_pair[0], ylabel=label_pair[1])
		if 'dimer' in transfer:
			scale_dimer = 1
			if j in [0, 1]:
				continue
			if df_pair[1] == 'CS':
				scale_dimer = 1
			if df_pair[1] == 'CoSR':
				df_pair[1] = 'C' # 'C' for dimer is actually CoSR
# 				scale_dimer=0.5
				error = np.array(list(zip(subdf[df_pair[1]+"_median"]-subdf[df_pair[1]+"_lower"], 
						   subdf[df_pair[1]+"_upper"]-subdf[df_pair[1]+"_median"]))).T*np.abs(scale_dimer)
				ax.errorbar(subdf[df_pair[0]], subdf[df_pair[1]+"_median"]*scale_dimer, 
					yerr=error, label=transfer_types[l], **sty)
		else:
			try:
				error = np.array(list(zip(subdf[df_pair[1]+"_median"]-subdf[df_pair[1]+"_lower"], 
								   subdf[df_pair[1]+"_upper"]-subdf[df_pair[1]+"_median"]))).T
				ax.errorbar(subdf[df_pair[0]], subdf[df_pair[1]+"_median"], yerr=error, 
				     label=transfer_types[l], **sty)
			except:
				continue
	
		if j == 2 or j == 5: # plotting against theoretical contact
			ax.plot([0.7,2], [0.7,2], 'k--')
		
h, l = axs[-3].get_legend_handles_labels()
axs[1].legend(h, l)

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

# fig1, ax1 = plt.subplots()

# transfer_df = df[df['Transfer'] == 'transfer']
# dimerc5_df = df[df['Transfer'] == 'dimer_c5']

# sorted_transfer_df = transfer_df.sort_values(by='ToTF')
# sorted_dimerc5_df = dimerc5_df.sort_values(by='ToTF')

# x = sorted_transfer_df['C_median']
# y = sorted_dimerc5_df['CS_median']

# sorted_transfer_df['x'] = x
# sorted_dimerc5_df['y'] = y

# print(sorted_transfer_df[['ToTF','C_median','Transfer','x']])
# print(sorted_dimerc5_df[['ToTF','CS_median','Transfer','y']])
# 	
# y = np.array(y)
# x = np.array(x)

# newy = np.array([y[0],y[1],(y[2]+y[3])/2,(y[4]+y[5])/2])
# newx = np.array([x[0],x[1],x[3],x[4]])

# newxy_df = pd.DataFrame({'New x': newx, 'New y': newy})
# print(newxy_df)

# ax1.plot(newx,newy,marker=pastamarkers.farfalle, markersize=25)

# def Linear(x,m,b):
# 	return m*x + b

# popt, pcov = curve_fit(Linear, newx,newy)
# xlist = np.linspace(newx.min(), newx.max(),100)
# ax1.plot(xlist, Linear(xlist,*popt),marker='',linestyle='-',color='b')
# print(f'y = {popt[0]:.2f}x + {popt[1]:.2f}')

# ax1.set(ylabel='Dimer c5 CS',xlabel='HFT C')