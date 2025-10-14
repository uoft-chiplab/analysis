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
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


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

### alpha cutoff
ALPHA_CUTOFF = False
alpha_cut = 0.1

###
COMBINE_DATA = True

# definitions of alpha
alphas = ['transfer', 'loss']

files = ["2025-07-28_H_e", # whole spectrum
		 "2025-07-28_J_e" # very fine about res
 		 ]

tpulse= 2000 # us

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
# these are from pre July 2025 so may be incorrect post-ODT2 realignment
# these are from varying jump page results
# see diagnostics/
AC_LOSS_CORRECTION = True
ToTFs = [ToTF,
		 ]
corr_cs = [1.31
		   ]
e_corr_cs = [0.08
			 ]
ToTFs = [ToTF,
		 ]
corr_cs = [1.2 #July 2025?
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

# sinc^2 lineshape functions
def sinc2(x, trf):
    t = x * trf
    result = np.ones_like(t)
    mask = t != 0
    result[mask] = (np.sin(np.pi * t[mask]) / (np.pi * t[mask]))**2
    return result

def Int2DGaussian(a, sx, sy):
	return 2*a*np.pi*sx*sy

# main spectrum data processing
run = Data(files[0] + '.dat')
run.data['OmegaR'] = 2*pi*RabiperVpp_47MHz_July2025*Vpp_from_VVAfreq(run.data['VVA'], run.data['freq'])
run.data['OmegaR2'] = run.data['OmegaR']**2
run.data['c9'] = run.data['c9'] * ff
bg_c9 = run.data[run.data['VVA'] == 0]['c9'].mean()
bg_c5 = run.data[run.data['VVA'] == 0]['c5'].mean()
run.data = run.data[run.data['VVA']!=0] # remove the bg points from df
if AC_LOSS_CORRECTION:
     run.data['c5'] = run.data['c5'] * corr_c_interp(ToTF)
run.data['N'] = (run.data['c5']-bg_c5) + run.data['c9'] 
run.data['alpha_transfer'] = (run.data['c5']-bg_c5) / (run.data['c5']-bg_c5 + run.data['c9'])
run.data['alpha_loss'] = (bg_c9 - run.data['c9'])/bg_c9
run.data['detuning_MHz'] = run.data['freq'] - res
run.data['detuning_EF'] = run.data['detuning_MHz']/EF
run.data['EFtohbar'] = h*EF*1e6*tpulse/1e6/hbar
if ALPHA_CUTOFF:
      run.data = run.data[run.data['alpha_loss']<alpha_cut]

#the data right on resonance had a poor SNR in the main dataset. Here it is stitched with another dataset
if COMBINE_DATA:
    run_res = Data(files[1] + '.dat')
    run_res.data['OmegaR'] = 2*pi*RabiperVpp_47MHz_July2025*Vpp_from_VVAfreq(run_res.data['VVA'], run_res.data['freq'])
    run_res.data['OmegaR2'] = run_res.data['OmegaR']**2
    run_res.data['c9'] = run_res.data['c9'] * ff
    bg_c9 = run_res.data[run_res.data['VVA'] == 0]['c9'].mean()
    bg_c5 = run_res.data[run_res.data['VVA'] == 0]['c5'].mean()
    run_res.data = run_res.data[run_res.data['VVA']!=0] # remove the bg points from df
    if AC_LOSS_CORRECTION:
     run.data['c5'] = run.data['c5'] * corr_c_interp(ToTF)
    run_res.data['N'] = (run_res.data['c5']-bg_c5) + run_res.data['c9'] 
    run_res.data['alpha_transfer'] = (run_res.data['c5']-bg_c5) / (run_res.data['c5']-bg_c5 + run_res.data['c9'])
    run_res.data['alpha_loss'] = (bg_c9 - run_res.data['c9'])/bg_c9
    run_res.data['detuning_MHz'] = run_res.data['freq'] - res
    run_res.data['detuning_EF'] = run_res.data['detuning_MHz']/EF
    run_res.data['EFtohbar'] = h*EF*1e6*tpulse/1e6/hbar # basically 2pi EF t with EF in linear freq
    if ALPHA_CUTOFF:
        run_res.data = run_res.data[run_res.data['alpha_loss']<alpha_cut]

    freq_mask_lower = run_res.data['freq'].min()
    freq_mask_higher = run_res.data['freq'].max()

    run.data = run.data[(run.data['freq'] < freq_mask_lower) | (run.data['freq'] > freq_mask_higher)]
    run.data = pd.concat([run.data, run_res.data])
    title = files[0] + ' combined with ' +files[1]
else:
    title = files[0]

fig, ax = plt.subplots()
fig_hist, ax_hist = plt.subplots()
for measure, sty,col in zip(measures, styles,colors):
    #run.data['IFGR_'+measure] = run.data['alpha_' + measure]*h*EF*1e6/(hbar*(run.data['OmegaR2']*1e3)**2*tpulse/1e6)
    run.data['scaled_alpha_'+measure] = run.data['alpha_' + measure]*(h*EF*1e6/hbar/(run.data['OmegaR']*1e3))**2
    #run.data['MaxI'+measure] = max(run.data['IFGR'+measure])
    run.data['Gamma_'+measure] = run.data['alpha_' + measure]/(tpulse/1e6)
    run.data['GammaTilde_' + measure] = GammaTilde(run.data['alpha_' + measure],
                                                h*EF*1e6, run.data['OmegaR']*1e3, tpulse/1e6)
    run.data['IFGR_' + measure] = run.data['GammaTilde_'+measure]*pi
    ax.plot(run.data['detuning_EF'], run.data['alpha_' + measure], label=measure, **sty)
    ax.set(title=title,
        xlabel=r'$\hbar\omega/E_F$',
        ylabel=r'$\alpha$')
    
    ax_hist.hist(run.data['alpha_'+measure], bins=10, alpha=0.5, label=measure, color=col)
    ax_hist.set(title=title,
                xlabel=r'$\alpha$')
ax_hist.legend()
ax.legend()

### Integrate to get SW
run.data = run.data.sort_values(by=['detuning_EF'])
run.group_by_mean('detuning_EF')

x = run.avg_data['detuning_EF']
y_choose = 'GammaTilde_'
y_trans = run.avg_data[y_choose + 'transfer']
y_loss = run.avg_data[y_choose + 'loss']

# apply and interpolate an SG filter to the transfer rate Gamma
SG_window = 5
SG_deg = 3
y_trans_SG = savgol_filter(y_trans, SG_window, SG_deg)
y_loss_SG = savgol_filter(y_loss, SG_window, SG_deg)
xvals = np.linspace(x.min(), x.max(), 100)
y_trans_SG_interp = interp1d(x, y_trans_SG, bounds_error=False, fill_value=0)
y_loss_SG_interp = interp1d(x, y_loss_SG,  bounds_error=False, fill_value=0)

# integrate both the raw data and the smoothed data
limit = 200
SW_trans = integrate.quad(lambda d: np.interp(d, x, y_trans), min(x), max(x), limit=limit)[0]
SW_loss = integrate.quad(lambda d: np.interp(d, x, y_loss), min(x), max(x), limit=limit)[0]
SW_trans_SG = integrate.trapezoid(y_trans_SG_interp(xvals), xvals)
SW_loss_SG = integrate.trapezoid(y_loss_SG_interp(xvals), xvals)

fig, ax = plt.subplots()
ax.errorbar(x, y_trans, yerr=run.avg_data['em_GammaTilde_transfer'], 
            label = 'transfer ' + f'SW = {SW_trans:.3f}',
            **styles[0])
ax.plot(xvals, y_trans_SG_interp(xvals),
        label = 'transfer smoothed ' + f'SW = {SW_trans_SG:.3f}',
        marker='', ls='-', lw=2, color=colors[0])
ax.errorbar(x, y_loss, yerr=run.avg_data['em_GammaTilde_loss'], 
            label = 'loss ' + f'SW = {SW_loss:.3f}',
            **styles[1])
ax.plot(xvals, y_loss_SG_interp(xvals),
        label = 'loss smoothed ' + f'SW = {SW_loss_SG:.3f}',
        marker='', ls='-', lw=2, color=colors[1])

ax.set(title=title,
        xlabel=r'$\hbar\omega/E_F$',
        ylabel=r'$\widetilde{\Gamma}$')
ax.legend()


### calculate and plot alpha from Yale apper
# modified alpha definition from Eq. (2)
# detuning and time should be in EF units
# R should ultimately have units of 1/time
# if R is given using IFGR, the output has units [EF/hbar Omega]^2 like Fig. 3
def alpha(detuning, tpulse, R_FGR_func):
    # if detuning is given in EF tpulse should also be EF units
    wpvals = np.linspace(-50, 50, 20000) # just needs to be large not inf
    Rvals = R_FGR_func(wpvals) 
    x = (wpvals - detuning)*tpulse/2
    return tpulse**2*integrate.trapezoid(\
         ((np.sin(x)/x)**2*Rvals), wpvals)/ (2*pi)

x = run.avg_data['detuning_EF']
y_choose = 'IFGR_'
y_trans = run.avg_data[y_choose + 'transfer']
y_loss = run.avg_data[y_choose + 'loss']

# apply and interpolate an SG filter 
SG_window = 5
SG_deg = 2
y_trans_SG = savgol_filter(y_trans, SG_window, SG_deg)
y_loss_SG = savgol_filter(y_loss, SG_window, SG_deg)
xvals = np.linspace(x.min(), x.max(), 100)
y_trans_SG_interp = interp1d(x, y_trans_SG,  bounds_error=False, fill_value=0)
y_loss_SG_interp = interp1d(x, y_loss_SG,  bounds_error=False, fill_value=0)

fig, ax = plt.subplots()
fig, ax = plt.subplots()
ax.errorbar(x, y_trans, yerr=run.avg_data['em_'+y_choose+'transfer'], 
            label = 'transfer',
            **styles[0])
ax.plot(xvals, y_trans_SG_interp(xvals),
        label = 'transfer smoothed ',
        marker='', ls='-', lw=2, color=colors[0])
# ax.errorbar(x, y_loss, yerr=run.avg_data['em_'+y_choose+'loss'], 
#             label = 'loss',
#             **styles[1])
# ax.plot(xvals, y_loss_SG_interp(xvals),
#         label = 'loss smoothed ',
#         marker='', ls='-', lw=2, color=colors[1])

ax.set(title=title,
        xlabel=r'$\hbar\omega/E_F$',
        ylabel=r'$I_{\mathrm{FGR}}$')
ax.legend()

times = np.arange(1, 5000, 10)/1e6 * (h*EF*1e6/hbar) # , times in us, EF in MHz
# note that 2piEF = hEF/hbar = "omega_F"
alpha_list = []
uniquad_list = []
grey_dashed = []
grey_dashed_2 = []
Z_factor_1 = 0.9
Z_factor_2 = 0.4
detuning_from_peak = 0
for time in times:
    this_alpha = alpha(detuning_from_peak, time, R_FGR_func = y_trans_SG_interp)
    #this_alpha = this_alpha *2 #?????
    alpha_list.append(this_alpha)
    uniquad_list.append(time**2/4 )
    grey_dashed.append(time**2/4 * Z_factor_1 )
    grey_dashed_2.append(time**2/4 * Z_factor_2)

alpha_list = np.array(alpha_list)
grey_dashed = np.array(grey_dashed)
grey_dashed_2=np.array(grey_dashed_2)
uniquad_list = np.array(uniquad_list)
fig, ax = plt.subplots()
ax.plot(np.array(times), np.array(alpha_list), ls='-', marker='', color='black', lw=2, label='Eq. (2)')
ax.plot(np.array(times), np.array(uniquad_list), ls='-.', marker='', color='red', lw=1, label=r'$(E_Ft/\hbar)^2/4$')
ax.plot(np.array(times), np.array(grey_dashed), ls='--', marker='', color='grey', lw=1, label =rf'$(Z \, E_Ft/\hbar)^2/4, Z={Z_factor_1:.1f}$')
ax.plot(np.array(times), np.array(grey_dashed_2), ls='--', marker='', color='brown', lw=1, label=rf'$(Z \,E_Ft/\hbar)^2/4, Z={Z_factor_2:.1f}$')
ax.plot(1.19, 0.31, **styles[4], label = 'Resonant short-time pulse, recent data')
ax.set(
    title = 'transfer alpha extrapolation',
    ylabel = r'$\alpha_(\omega ,t) \cdot (E_F / \hbar \Omega_0)^2 \quad [Eq. 2]$',
    xlabel = r'$E_F t/\hbar$',
    xscale='log',
    yscale='log',
    xlim = [0.5, 100],
    #ylim = [ 10e-2, 10e1]
    )
ax.legend()



