# -*- coding: utf-8 -*-
"""
@author: coldatoms
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_class import Data
from library import plt_settings, dark_colors, light_colors, markers
from cycler import Cycler

styles = Cycler([{'mec':dark_color, 'color':dark_color, 'mfc':light_color, 
				  'marker':marker} for marker, dark_color, light_color in \
					 zip(markers, dark_colors, light_colors)])
	
def spin_map(spin):
	if spin == 'c5':
		return 'b'
	else:
		return 'a'
	
# use ratio or fraction when comparing b and a, i.e. b/a or b/(a+b)
rof = 'fraction'
		
bg_uncertainty_propagate = False

files = ["2024-10-30_F_e.dat", "2024-10-30_E_e.dat", 
		 "2024-11-01_C_e.dat", "2024-11-01_G_e.dat",
		 "2024-11-01_I_e.dat"]
fields = [209,204,202.14,207,211]
dfs=[]
drop_VVAlist = [[0,1,2], [0,1], [0,1,1.5], [0,1,2],[0,3]]

spins = ['c5', 'c9']

for i, file in enumerate(files):
	drop_VVAs = drop_VVAlist[i]
	field=fields[i]
	fig, axes = plt.subplots(2, 3, figsize=(10,6))
	axs = axes.flatten()
	xname = 'VVA'
	xlabel = xname
	
	ylabels = ['counts',
			   'loss',
			   'widths',
			   'counts '+rof,
			   'loss '+rof,
			   'widths '+rof]
	
	for ax, ylabel in zip(axs, ylabels):
		ax.set(xlabel=xlabel, ylabel=ylabel)
		
	fig.suptitle(file[:-6] + " dimer spin loss, field = {} G".format(field))
	
	filename = file
	run = Data(filename)
	
	# append field
	run.data['field'] = field
	
	# make opposite fraction
	run.data['fraction59'] = 1/run.data.fraction95
	
	# compute widths as quadrature sum of h and v
	run.data['c5_s'] = np.sqrt(np.abs(run.data['two2D_sh1']*run.data['two2D_sv1']))
	run.data['c9_s'] = np.sqrt(np.abs(run.data['two2D_sh2']*run.data['two2D_sv2']))
	
	# average bg values for spins, and compute loss
	for spin in spins:
		run.data[spin+'_bg'] = run.data[(run.data['VVA'] == 0) | (run.data['VVA'] == 1)][spin].mean()
		run.data[spin+'_bg_std'] = run.data[(run.data['VVA'] == 0) | (run.data['VVA'] == 1)][spin].std()
		run.data[spin+'_loss'] = run.data[spin+'_bg'] - run.data[spin]
		
	# compute loss and width ratio
	run.data['c_ratio'] = run.data['c5']/run.data['c9']
	run.data['ratio'] = run.data['c5_loss']/run.data['c9_loss']
	run.data['s_ratio'] = run.data['c5_s']/run.data['c9_s']
	
	run.data['c_fraction'] = run.data['c5']/(run.data['c9']+run.data['c5'])
	run.data['fraction'] = run.data['c5_loss']/(run.data['c9_loss']+run.data['c5_loss'])
	run.data['s_fraction'] = run.data['c5_s']/(run.data['c9_s']+run.data['c5_s'])
	
	# average by VVA, and drop 0 VVA
	run.group_by_mean('VVA')
	for VVA_val in drop_VVAs:
		run.avg_data = run.avg_data.drop(run.avg_data.loc[(run.avg_data.VVA == VVA_val)].index)
	
		
	# correct ratio average and it's unceratinty
# 	run.avg_data['ratio'] = run.avg_data['c5_loss']/run.avg_data['c9_loss']
# 	run.avg_data['em_ratio'] = run.avg_data['ratio']*np.sqrt((run.avg_data['em_c5_loss']/run.avg_data['c5_loss'])**2 +\
#  	 									  (run.avg_data['em_c9_loss']/run.avg_data['c9_loss'])**2)
		
# 	run.avg_data['fraction'] = run.avg_data['c5_loss']/(run.avg_data['c9_loss']+run.avg_data['c5_loss'])
# 	run.avg_data['em_fraction'] = ?
	
	dfs.append(run.avg_data)
	# propagate bg uncertainty, this is not a statistical fluctuation though...
	if bg_uncertainty_propagate == True:
		for spin in spins:
			run.avg_data['em_'+spin+'_loss'] = np.sqrt(run.avg_data['em_'+spin+'_loss']**2 \
												+ run.avg_data[spin+'_bg_std'].unique()[0]**2)
	
	# plot
	x = run.avg_data[xname]	
	for spin, sty in zip(spins, styles):
		# counts
		axs[0].errorbar(x, run.avg_data[spin], 
					 run.avg_data['em_'+spin], 
					 label=spin_map(spin), **sty)
		bg = run.data[spin+'_bg'].unique()*np.ones(len(x))
		axs[0].plot(x, bg, '--')
		# loss
		axs[1].errorbar(x, run.avg_data[spin+'_loss'], 
					 run.avg_data['em_'+spin+'_loss'], 
					 label=spin_map(spin), **sty)
		# widths
		axs[2].errorbar(x, run.avg_data[spin+'_s'], 
					 run.avg_data['em_'+spin+'_s'], 
					 label=spin_map(spin), **sty)
		
	
	if rof == 'ratio':
		label = 'b/a'
		sty = list(styles)[2]
	else:
		label = 'b/(a+b)'
		sty = list(styles)[4]
	# count ratio/fraction
	axs[3].errorbar(x, run.avg_data['c_'+rof], run.avg_data['em_c_'+rof], 
					label=label, **sty)
	# loss ratio/fraction
	axs[4].errorbar(x, run.avg_data[rof], run.avg_data['em_'+rof],
					label=label, **sty)
	# width ratio/fraction
	axs[5].errorbar(x, run.avg_data['s_'+rof], run.avg_data['em_s_'+rof],
					label=label, **sty)
		
	for ax in axs:
		ax.legend()
		
	fig.tight_layout()
	plt.show()
	
	
# summary
df = pd.concat(dfs)
cols = [c for c in df.columns if not c.startswith('em_')]
df = df[cols]
mean = df.groupby(['field']).mean().reset_index()
sem = df.groupby(['field']).sem().reset_index().add_prefix("em_")
df = pd.concat([mean, sem], axis=1)
fig, ax = plt.subplots(figsize=(6,4))
ax.set(xlabel="Magnetic Field (G)", ylabel=label+"Loss "+rof)

sty = list(styles)[0]
ax.errorbar(df.field, df[rof], df['em_'+rof], **sty)
# fig.title("dimer spin loss vs. magnetic field")