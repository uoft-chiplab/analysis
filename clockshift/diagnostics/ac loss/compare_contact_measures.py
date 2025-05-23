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

files = [
# 		'2025-04-14_J_e.dat',
# 		 '2025-04-14_M_e.dat',
# 		 '2025-04-14_N_e.dat',
# 		 '2025-04-14_P_e.dat',
# 		 '2025-04-14_R_e_time=0.06.dat',
# 		 '2025-04-14_R_e_time=0.1.dat',
 		 # '2025-04-14_R_e_time=0.02.dat',
 		 '2025-04-14_U_e_sweep=46.05.dat',
 		 '2025-04-14_U_e_sweep=45.dat',
 		 '2025-04-14_U_e_sweep=48.55.dat',
 		 # '2025-04-14_X_e_sweep=46.05.dat',
 		 # '2025-04-14_X_e_sweep=45.dat',
 		 # '2025-04-14_X_e_sweep=48.55.dat',
 		 ]
		 
state_choices = [
# 				'bc',
# 				 'ab',
# 				 'bc','bc','bc','bc','bc',
# 				 'bc', 'ac', 'ab',
				 'bc', 'ac', 'ab',
				 ]

bg_choices = [
# 			  'VVA','VVA','VVA', 
# 			  'freq',
# 			  'VVA','VVA','VVA',
# 			  'VVA','VVA','VVA',
			  'VVA','VVA','VVA',
			  ]

shutter_status = [
# 				  'closed', 
# 				  'open','open', 
# 				  'closed','closed','closed','closed',
# 				  'closed','closed','closed',
				  'closed','closed','closed',
				  ]

notes = [
# 		'', '', '', '', 
# 		 '', '', '',
# 		 'Na~Nb+Nc', 'Nb~Na-Nc', 'Nc~Nb-Na',
		 'Na~Nb+Nc', 'Nb~Na-Nc', 'Nc~Nb-Na']

times_pre_jump = [
# 				  0.06, 0.06, 0.06, 0.06, 0.06,
# 				  0.1,
# 				  0.02,
# 				  0.06, 0.06, 
# 				  0.06, 0.06, 0.06,
				  0.06, 0.06, 0.06,
				  ]

# from 2025-04-14, i.e. today
ff = 0.83

# from run U when measuring a and b
pol = 1.0353813916622319 # U
# pol = 0.9855 # X
# pol = 1.0

results_list = []


for i, file in enumerate(files):
	results = {
			'filename': file,
			'run': file[11],
			'states': state_choices[i],
			'index': i,
			'notes': notes[i],
			'bg_choice': bg_choices[i],
			'shutter': shutter_status[i],
			't_pre_jump': times_pre_jump[i],
				}
	
	run = Data(file)
	
	# correct for fudge factor
	run.data['c9'] = run.data['c9'] * ff
	
	# select dfs
	if results['bg_choice'] == 'VVA':
		df = run.data.loc[run.data.VVA > 0]
		bg_df = run.data.loc[run.data.VVA == 0]
	elif results['bg_choice'] == 'freq':
		df = run.data.loc[run.data.freq > 47]
		bg_df = run.data.loc[run.data.freq < 47]
		
	print("bg c5 counts", bg_df['c5'].mean(), bg_df['c5'].std())
# 	bg_df['c5'] = 0
	
	# swept b into c9 box
	if results['states'] == 'bc':
		df['c'] = df['c5'] - bg_df['c5'].mean()
		# correct the data
		df['b'] = df['c9']
		
		# approximately
		df['a'] = (df['b'] + df['c'])*pol
		
		df['transfer'] = df['c']/(df['b'] + df['c'])
		df['transfer2'] = (bg_df['c9'].mean() - df['b'])/bg_df['c9'].mean()
		df['N'] = df['a']
		df['Nbg'] = bg_df['c9'].mean()*pol
		df['NoNbg'] = df['N']/df['Nbg']
		
	# no sweep
	elif results['states'] == 'ac':
		df['a'] = df['c9']
		df['c'] = df['c5'] - bg_df['c5'].mean()
		
		# approximately
		df['b'] = df['a']/pol - df['c']
		
		df['transfer'] = df['c']/(df['a']/pol)
		df['transfer2'] = df['c']/(bg_df['c9'].mean()/pol)
		df['N'] = df['a']
		df['Nbg'] = bg_df['c9'].mean()
		df['NoNbg'] = df['N']/df['Nbg']
		
	
	# swept b into c5 box
	elif results['states'] == 'ab':
		df['a'] = df['c9']
		df['b'] = df['c5']
		
		# approximately
		df['c'] = df['a']/pol - df['b']
		
		df['transfer'] = (df['a']/pol - df['b'])/(df['a']/pol)
		df['transfer2'] = (bg_df['c5'].mean() - df['b'])/bg_df['c5'].mean()
		df['N'] = df['a']
		df['Nbg'] = bg_df['c9'].mean()
		df['NoNbg'] = df['a']/df['Nbg']
		
		
	results['a'] = df['a'].mean()
	results['em_a'] = df['a'].sem()
	results['b'] = df['b'].mean()
	results['em_b'] = df['b'].sem()
	results['c'] = df['c'].mean()
	results['em_c'] = df['c'].sem()
	
	results['transfer'] = df['transfer'].mean()
	results['em_transfer'] = df['transfer'].sem()
	results['transfer2'] = df['transfer2'].mean()
	results['em_transfer2'] = df['transfer2'].sem()
	
	results['N'] = df['N'].mean()
	results['em_N'] = df['N'].sem()
	results['NoNbg'] = df['NoNbg'].mean()
	results['em_NoNbg'] = df['NoNbg'].sem()
	
	
	results_list.append(results)
	

df_total = pd.DataFrame(results_list)
df = df_total

# plot
fig, axes = plt.subplots(2,2, figsize=(10,8))
axs = axes.flatten()

fig.suptitle(file[0:12]+" with pol fix")
	
# transfer
ax = axs[0]
ax.set(xlabel='Run number', ylabel=r'Transfer $\alpha$')#, ylim=[0.085, 0.1])
x = df['index']
y = df['transfer']
yerr = df['em_transfer']
ax.errorbar(x, y, yerr, **styles[0], label=r'$\alpha$ from sig')
y = df['transfer2']
yerr = df['em_transfer2']
ax.errorbar(x, y, yerr, **styles[1], label=r'$\alpha$ comp. to bg')
ax.legend()

# table
ax_table = axs[1]
ax_table.axis('off')
quantities = ['#', 'run', 'states', 'bg', 'shutter', 't pre jump', 
			  'alpha','notes']
transfer_list = [ '%.3f(%.0f)' % (val, 1e3*e_val) for val, e_val in \
		 zip(list(df['transfer'].values), list(df['em_transfer'].values))]

values = [list(df['index'].values), 
		  list(df['run'].values), 
		  list(df['states'].values), 
		  list(df['bg_choice'].values),
		  list(df['shutter'].values),
		  list(df['t_pre_jump'].values),
		  transfer_list,
		  list(df['notes'].values),
		  ]
table = list(zip(quantities, values))
table = values
the_table = ax_table.table(cellText=table, loc='center', rowLabels=quantities)
the_table.auto_set_font_size(True)
# the_table.set_fontsize(16)
# the_table.scale(1,1.5)

# atom number
ax = axs[2]
ax.set(xlabel='Run number', ylabel=r'Atom number $N_\sigma$')#, ylim=[0.085, 0.1])
x = df['index']

sty = styles[0]
y = df['a']
yerr = df['em_a']
ax.errorbar(x, y, yerr, **sty, label='a')

sty = styles[1]
y = df['b']
yerr = df['em_b']
ax.errorbar(x, y, yerr, **sty, label='b')

ax.legend()

# N over N bg
ax = axs[3]
x = df['index']
y = df['NoNbg']
yerr = df['em_NoNbg']
ax.set(xlabel='Run number', ylabel=r'Atom number $N/N_{bg}$')#, ylim=[0.085, 0.1])
ax.errorbar(x, y, yerr, **styles[0])
	
fig.tight_layout()

plt.show()

# calculate pol
bg_df['r95'] = bg_df['c9']/bg_df['c5']
pol = bg_df['r95'].mean()