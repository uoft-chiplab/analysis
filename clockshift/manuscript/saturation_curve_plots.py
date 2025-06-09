# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 08:33:01 2025

@author: Chip lab
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.colors as mcolors
import matplotlib.cm
# this is a hack to access modules in the parent directory
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (clockshift)
parent_dir = os.path.dirname(current_dir)
# get the parent's parent directory (analysis)
analysis_dir = os.path.dirname(parent_dir)
# Add the parent parent directory to sys.path
if analysis_dir not in sys.path:
	sys.path.append(analysis_dir)

from library import paper_settings, generate_plt_styles
import pickle as pkl
import pandas as pd

data_path = os.path.join(analysis_dir, 'clockshift\\rf_saturation_analysis\\saturation_data')

### Fit functions
def Linear(x,m,b):
	return m*x + b

def Saturation(x, A, x0):
	return A*(1-np.exp(-x/x0))

# plotting options
# colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
# colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c']
# colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']
# colors = ['#1034A6', '#412F88', '#722B6A', '#A2264B', '#D3212D']
colors = ['#1b1044', '#812581', '#c03a76', '#f3655c', '#fde0a2']
colors = [
	colors[0],
	# colors[1],
	# colors[2],
	colors[3],
	# colors[4]
]
styles = generate_plt_styles(colors, ts=0.6)

linestyles = ['--', 
			  '--','--','--','--',
			  ':'
			  ]

### plot settings
plt.rcParams.update(paper_settings) # from library.py
font_size = paper_settings['legend.fontsize']
plt.rcParams['legend.fontsize'] = 6
fig_width = 3.4 # One-column PRL figure size in inches
subplotlabel_font = 10
		
fig, axs = plt.subplots(1, 2, figsize=(fig_width*6/5, fig_width*3/5)
						)

axs[0].set(xlabel=r'rf power $\Omega_{\mathrm{23}}^2/(2\pi)^2$ [kHz$^2$]', 
		   ylabel=r'Transfer $\alpha_\mathrm{d}$',
		   ylim=[-0.02, .22],
		#    xlim = [0,12000]
		   )

# loop over dimers
dimer_file = 'dimer_saturation_curves.pkl'

with open(os.path.join(data_path, dimer_file), 'rb') as f:
	    dimer_data = pkl.load(f)
		
# select data
dimer_data = [dimer_data[0], dimer_data[2]]
		
# chosen average saturation Rabi
OmegaRabi2 = 3272
e_OmegaRabi2 = 136

for i, data in enumerate(dimer_data):
	df = data['df']
	
	popt = data['popt_b']
	pcovd = data['pcov_b']
	perrd = np.sqrt(np.diag(pcovd))
	data['e_ToTF'] = data['ToTF']*0.03
	alpha0d = popt[0]
	# dimer
	sty = styles[i]
	color = colors[i]
	label = r'$T/T_F$ = {:.2f}({:.0f})'.format(data['ToTF'], 1e2*data['e_ToTF'])
	label = r'{:.2f}({:.0f}) $T_F$'.format(data['ToTF'], data['e_ToTF']*1e2)
	x = df['OmegaR2']
	y = df['c5_transfer']
	yerr = df['em_c5_transfer']
	
	xs = np.linspace(0, max(x), 1000)  # linspace of rf powers
	
	# rescale x-axis
	ax = axs[0]
	x = df['OmegaR2']
	
	ax.errorbar(x, y, yerr=yerr, label=label, **sty)
	ax.plot(xs, Saturation(xs, *popt), '-', color=color)
	ax.plot(xs, Linear(xs, popt[0]/popt[1], 0), linestyles[i], 
	  color=color)
	

### HFT various ToTF, same detuning ###
# save files 
files = [
		 '100kHz_saturation_curves.pkl', 
		 'near-res_saturation_curves.pkl',
		 'various_ToTF_saturation_curves.pkl',
		 ]

loaded_data = []

# grab dictionary lists from pickle files
for i, file in enumerate(files):
	with open(os.path.join(data_path, file), "rb") as input_file:
		loaded_data = loaded_data + pkl.load(input_file)
		
# turn dictionary list into dataframe
df = pd.DataFrame(loaded_data)

# ax = axs[1]

# ax.set(xlabel=r'RF Power $\Omega_{23}/\Omega_{HFT,0}$', 
# 		   ylabel=r'Transfer $\alpha_{HFT}/\alpha_{HFT,0}$',
#  		   ylim=[-0.04, 1.08]
# 		   )
sub_df = df.loc[(df.detuning == 100)]

ToTFs = sub_df.ToTF.unique()
ToTFs.sort()
ToTFs = [
	ToTFs[0],
		#   ToTFs[1],
		#  ToTFs[2], 
		#  ToTFs[3],
		  ToTFs[4]
		  ]

# chosen average saturation Rabi
OmegaRabi2 = np.mean([716.46, 962.81])

ax = axs[1]

ax.set(xlabel=r'rf power $\Omega_{\mathrm{23}}^2/(2\pi)^2$ [kHz$^2$]', 
		ylabel=r'Transfer $\alpha_{\mathrm{HFT}}$',
		ylim=[0, .8],
		xlim = [0,2200]
)
plot_data = []
for i, ToTF in enumerate(ToTFs):
	sty = styles[i]
	color = colors[i]
	e_ToTF = ToTF * 0.03
	label = r'{:.2f}({:.0f}) $T_F$'.format(ToTF, e_ToTF * 1e2)

	data = sub_df.loc[sub_df.ToTF == ToTF].squeeze()
	xs = np.linspace(0, 4.05 * OmegaRabi2, 100)
	popt = data.popt
	# pcovhft = data.pcov
	# perrhft = np.sqrt(np.diag(pcovhft))
	Gammas_Sat = Saturation(xs, *popt)
	Gammas_Lin = xs * popt[0] / popt[1]
	alpha0 = popt[0]


	# raw data plots
	ax.plot(xs , Gammas_Sat , '-', color=color)
	ax.plot(xs , Gammas_Lin , linestyles[i], color=color)
	
	x = data.df['OmegaR2']
	y = data.df['transfer'] 
	yerr = data.df['em_transfer']

	plot_data.append({
		'ToTF': ToTF,
		'label': label,
		'color': color,
		'style': sty,
		'linestyle': linestyles[i],
		'xs': xs.tolist(),
		'Gammas_Sat': Gammas_Sat.tolist(),
		'Gammas_Lin': Gammas_Lin.tolist(),
		'x': x.tolist(),
		'y': y.tolist(),
		'yerr': yerr.tolist()
	})
	ax.errorbar(x , y, yerr=yerr, **sty, label=label)


pklpath = os.path.join(parent_dir,"saturation_plot_data.pkl")

with open(pklpath, "wb") as f:
	pkl.dump(plot_data, f)	
colors_list = colors
# Save only what you need to rebuild sm
sm_config = {
    'colors': colors_list,  # list of hex or RGB colors you used
    'vmin': min(ToTFs),
    'vmax': max(ToTFs)
}

pklsmpath = os.path.join(parent_dir,"sm_config.pkl")

with open(pklsmpath, "wb") as f:
    pkl.dump(sm_config, f)
# make_transfer_plot(ax, sub_df, ToTFs, styles, colors, OmegaRabi2, linestyles)
cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors)
norm = mcolors.Normalize(min(ToTFs), max(ToTFs))
sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, 
)
cbar.set_label(r'$T/T_F$', fontsize=7)

fig.tight_layout()
subplot_labels = ['(a)', '(b)']
for n, ax in enumerate(axs):
	# ax.legend(frameon=False, handletextpad=0)
	label = subplot_labels[n]
	ax.text(-0.3, 1.05, label, 
		 transform=ax.transAxes, 
		 size=subplotlabel_font
		 )
	
plt.subplots_adjust(top=0.9)
# fig.tight_layout()
proj_path = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(proj_path, r'\manuscript\manuscript_figures')
fig.savefig(os.path.join(output_dir, 'dimer_and_HFT_saturation_curves-2025-06-09-only2.pdf'))
# plt.savefig("clockshift/manuscript/manuscript_figures/dimer_and_HFT_saturation_curves.pdf")
plt.show()
