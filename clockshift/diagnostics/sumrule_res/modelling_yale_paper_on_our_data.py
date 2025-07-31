"""
Created by Chip lab 31-07-2025

Analysis script for long time weak pulse taken on 29-07-2025 (folder was accidently called 28July2025)

"""

# paths
import sys
import os
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(os.path.dirname(os.path.dirname(proj_path)))
data_path = os.path.join(root, 'clockshift/data/sum_rule')
if root not in sys.path:
	sys.path.append(root)
from library import pi, h, hbar, mK, a0, paper_settings, generate_plt_styles
from data_class import Data
from rfcalibrations.Vpp_from_VVAfreq_July2025 import Vpp_from_VVAfreq
from fit_functions import Gaussian
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate

from warnings import filterwarnings	
filterwarnings('ignore')

# plotting options
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']
styles = generate_plt_styles(colors, ts=0.6)

### Script options
# This turns on (True) and off (False) saving the data/plots 
Save = False

### Analysis options
#Correct_ac_Loss = True

# definitions of alpha
alphas = ['transfer', 'loss']

files = ["2025-07-28_H_e",
 		 ]

tpulses = [2000,
		   ] #us

measures = ['transfer', 'loss']
# based on the UShots of that day
EF = 0.01918 # MHz 
ToTF = 0.5263
N = 52233/2
ff = 0.82
res = 47.2227 #MHz
### plot settings
plt.rcdefaults()
plt.rcParams.update(paper_settings) # from library.py
font_size = paper_settings['legend.fontsize']
fig_width = 3.4 # One-column PRL figure size in inches
subplotlabel_font = 10

### Calibrations

RabiperVpp_47MHz_July2025 = 12.13/0.452 # slightly modified from 2025 for July 2025 data
e_RabiperVpp_47MHz_2025 = 0.28 # ESTIMATE

### ac loss corrections
# these are from varying jump page results
# see diagnostics/
ToTFs = [ToTF,
		 ]
corr_cs = [1.31
		   ]
e_corr_cs = [0.08
			 ]

corr_c_interp = lambda x: np.interp(x, np.array(ToTFs), np.array(corr_cs))
e_corr_c_interp = lambda x: np.interp(x, np.array(ToTFs), np.array(e_corr_cs))
	
### constants
re = 103 * a0 # ac dimer range estimate
Eb = 3.98 # MHz # I guesstimated this from recent ac dimer spectra

### transfer functions
def GammaTilde(transfer, EF, OmegaR, trf):
	return EF/(hbar * pi * OmegaR**2 * trf) * transfer

# sinc^2 dimer lineshape functions
def sinc2(x, trf):
    t = x * trf
    result = np.ones_like(t)
    mask = t != 0
    result[mask] = (np.sin(np.pi * t[mask]) / (np.pi * t[mask]))**2
    return result

def Int2DGaussian(a, sx, sy):
	return 2*a*np.pi*sx*sy

for file, tpulse in zip(files, tpulses):
	run = Data(file + '.dat')
	run.data['OmegaR'] = 2*pi*RabiperVpp_47MHz_July2025*Vpp_from_VVAfreq(run.data['VVA'], run.data['freq'])
	run.data['OmegaR2'] = run.data['OmegaR']**2
	run.data['c9'] = run.data['c9'] * ff
	bg_c9 = run.data[run.data['VVA'] == 0]['c9'].mean()
	bg_c5 = run.data[run.data['VVA'] == 0]['c5'].mean()
	run.data['N'] = (run.data['c5']-bg_c5) + run.data['c9'] 
	run.data['alpha_transfer'] = (run.data['c5']-bg_c5) / (run.data['c5']-bg_c5 + run.data['c9'])
	run.data['alpha_loss'] = (bg_c9 - run.data['c9'])/bg_c9
	run.data['detuning_MHz'] = run.data['freq'] - res
	run.data['detuning_EF'] = run.data['detuning_MHz']/EF
	run.data['EFtohbar'] = h*EF*1e6*tpulse/1e6/hbar
	fig, ax = plt.subplots()
	# fig2, ax2 = plt.subplots()
	for measure, sty in zip(measures, styles):
		run.data['IFGR'+measure] = run.data['alpha_' + measure]*h*EF*1e6/(hbar*run.data['OmegaR2']*1e3**2*tpulse/1e6)
		run.data['scaled_alpha'+measure] = run.data['alpha_' + measure]*(h*EF*1e6/hbar*run.data['OmegaR']*1e3)**2
		run.data['MaxI'+measure] = max(run.data['IFGR'+measure])
		def alpha(tpulse):
			return tpulse**2*np.trapezoid((sinc2(run.data['detuning_EF']/2,tpulse)*run.data['alpha_' + measure])/(2*pi))
		run.data['Gamma_'+measure] = run.data['alpha_' + measure]/(tpulse/1e6)
		run.data['GammaTilde_' + measure] = GammaTilde(run.data['alpha_' + measure],
												 h*EF*1e6, run.data['OmegaR']*1e3, tpulse/1e6)
		ax.plot(run.data['detuning_EF'], run.data['GammaTilde_' + measure], label=measure, **sty)
		ax.set(title=file,
		 xlabel=r'$\hbar\omega/E_F$',
		 ylabel=r'$\widetilde{\Gamma}$')
		# ax2.plot(run.data['detuning_EF'], run.data['IFGR' + measure], label=measure, **sty)
		# ax2.plot(alpha(tpulse),linestyle='-')
		# ax2.set(title=file,
		#  xlabel=r'$\hbar\omega/E_F$',
		#  ylabel=r'$I_{FGR}$')
		# ax2[1].plot(run.data['EFtohbar'],run.data['scaled_alpha'+measure])

	### Integrate to get SW
	run.data = run.data.sort_values(by=['detuning_EF'])
	run.group_by_mean('detuning_EF')
	x = run.avg_data['detuning_EF']
	y_trans = run.avg_data['GammaTilde_transfer']
	y_loss = run.avg_data['GammaTilde_loss']
	# SW_tran = integrate.trapezoid(y=y_trans, x=x)
	# SW_loss = integrate.trapezoid(y=y_loss, x=x)

	# limit = 200
	# SW_trans = integrate.quad(lambda d: np.interp(d, x, y_trans), min(x), max(x), limit=limit)[0]
	# SW_loss = integrate.quad(lambda d: np.interp(d, x, y_loss), min(x), max(x), limit=limit)[0]

	fig, ax = plt.subplots()
	ax.errorbar(x, y_trans, yerr=run.avg_data['em_GammaTilde_transfer'], 
			 label = 'transfer', #f'SW = {SW_trans:.3f}',
			  **styles[0])
	ax.errorbar(x, y_loss, yerr=run.avg_data['em_GammaTilde_loss'], 
			 label = 'loss', #f'SW = {SW_loss:.3f}',
			  **styles[1])
	ax.set(title=file,
		 xlabel=r'$\hbar\omega/E_F$',
		 ylabel=r'$\widetilde{\Gamma}$')
	ax.legend()
	# handles, labels = ax.get_legend_handles_labels()
	# labels[0] = labels[0] + f', SW = {SW_trans:.3f}'
	# labels[1] = labels[1] + f', SW = {SW_loss:.3f}'
	# ax.legend(handles, labels)

