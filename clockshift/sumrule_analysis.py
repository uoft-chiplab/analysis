"""
Created by Chip lab 2024-06-12

Loads .dat with contact HFT scan and computes scaled transfer. Plots. Also
computes the sumrule.

To do:
	Error in x_star
	
"""
BOOTSRAP_TRAIL_NUM = 500

# paths
import os
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')
figfolder_path = os.path.join(proj_path, 'figures')

from library import pi, h, hbar, mK, a0, plt_settings, GammaTilde, tintshade, \
	 tint_shade_color, markers, colors,  chi_sq
from data_helper import check_for_col_name, bg_freq_formatter, remove_indices_formatter
from save_df_to_xlsx import save_df_row_to_xlsx
from data_class import Data
from scipy.optimize import curve_fit
from scipy.stats import sem
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
from clockshift.MonteCarloSpectraIntegration import Bootstrap_spectra_fit_trapz, \
					dist_stats, MonteCarlo_estimate_std_from_function
from contact_correlations.UFG_analysis import calc_contact
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
import time

### This turns on (True) and off (False) saving the data/plots 
Save = True

### script options
Analysis = True
Bootstrap = True
Correlations = True
Debug = False
Filter = True
Talk = True

Bg_Tracking = True
Calc_CTheory_std = True

transfer_selection = 'transfer' #'transfer' or  'loss'

### metadata
metadata_filename = 'metadata_file.xlsx'
metadata_file = os.path.join(proj_path, metadata_filename)
metadata = pd.read_excel(metadata_file)
files =  metadata.loc[metadata['exclude'] == 0]['filename'].values
if transfer_selection == 'loss':
	files =  ["2024-09-12_E_e",
			   "2024-09-18_F_e",
			   "2024-09-18_K_e"] # loss files
# files = ["2024-09-18_K_e"]

# Manual file select, comment out if exclude column should be used instead
# files = ["2024-09-12_E_e"]

# save file path
savefilename = 'sumrule_analysis_results.xlsx'
if transfer_selection == 'loss':
	savefilename = 'loss_'+savefilename
savefile = os.path.join(proj_path, savefilename)

### Vpp calibration
# VpptoOmegaR = 27.5833 # kHz/Vpp, older calibration
VpptoOmegaR = 17.05/0.703 # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
OmegaR_from_VVAfreq = lambda Vpp, freq: VpptoOmegaR * Vpp_from_VVAfreq(Vpp, freq)
	
### contants
re = 107 * a0 # ac dimer range estimate
Eb = 3.98 # MHz # I guesstimated this from recent ac dimer spectra

def a13(B):
	''' ac scattering length '''
	abg = 167.6*a0
	DeltaB = 7.2
	B0=224.2
	return abg*(1 - DeltaB/(B-B0))

def xstar(B, EF):
	return Eb/EF # hbar**2/mK/a13(B)**2 * (1-re/a13(Bfield))**(-1)

def GenerateSpectraFit(xstar, bg=False):
	def fit_func(x, A):
		xmax = xstar
		return A*x**(-3/2) / (1+x/xmax)
	if bg == True:
		return_func = lambda x: np.piecewise(x, [x<0, x>=0], [lambda z: b, 
														fit_func(z, A) + b])
		return return_func
	else:
		return fit_func

def dwSpectraFit(xi, x_star, A):
	return A*2*(1/np.sqrt(xi)-np.arctan(np.sqrt(x_star/xi))/np.sqrt(x_star))

def wdwSpectraFit(xi, x_star, A):
	return A*2*np.sqrt(x_star)*np.arctan(np.sqrt(x_star/xi))

def linear(x, a, b):
	return a*x + b

### plot settings
plt.rcParams.update(plt_settings) # from library.py
color = '#1f77b4'  # default matplotlib color (that blueish color)
color2 = '#ff7f0e'  # second default (orange)
light_color = tint_shade_color(color, amount=1+tintshade)
dark_color = tint_shade_color(color, amount=1-tintshade)
light_color2 = tint_shade_color(color2, amount=1+tintshade)
dark_color2 = tint_shade_color(color2, amount=1-tintshade)
light_red = tint_shade_color('r', amount=1+tintshade)
dark_red = tint_shade_color('r', amount=1-tintshade)
plt.rcParams.update({"figure.figsize": [12,8],
					 "font.size": 14,
					 "lines.markeredgecolor": dark_color,
					 "lines.markerfacecolor": light_color,
					 "lines.color": dark_color,
					 "lines.markeredgewidth": 2,
					 "errorbar.capsize": 0})

### loop analysis over selected datasets
save_df_index = 0
for filename in files:
	
##############################
######### Analysis ###########
##############################
	if Analysis == False:
		break # exit loop if no analysis required, this just elims one 
			  # indentation for an if statement
	
	# run params from HFT_data.py
	print("----------------")
	print("Analyzing " + filename)
	
	meta_df = metadata.loc[metadata.filename == filename].reset_index()
	if meta_df.empty:
		print("Dataframe is empty! The metadata likely needs updating." )
		continue
	
	xname = meta_df['xname'][0]
		
	# EF, it's used a lot soooo
	EF = meta_df['EF'][0] # MHz
	
	# create data structure
	filename = filename + ".dat"
	run = Data(filename, path=data_path)
	runfolder = filename 
	figpath = os.path.join(figfolder_path, runfolder)
	
	# initialize results dict to turn into df
	results = {}
	results['Run'] = filename
	results['Transfer'] = transfer_selection
	results['Pulse Time (us)'] = meta_df['trf'][0]*1e6
	results['Pulse Type'] = meta_df['pulsetype'][0]
	results['ToTF'] = meta_df['ToTF'][0]
	results['e_ToTF'] = meta_df['ToTF_sem'][0]
	results['EF'] = EF
	results['e_EF'] = meta_df['EF_sem'][0]
	results['kF'] = np.sqrt(2*mK*EF*h*1e6)/hbar
	results['barnu'] = meta_df['barnu'][0]
	results['e_barnu'] = meta_df['barnu_sem'][0]
	results['FourierWidth'] = 2/meta_df['trf'][0]/1e6
	
	# from Tilman's unitary gas harmonic trap averaging code
	results['C_theory'] = calc_contact(results['ToTF'], results['EF'], 
								results['barnu'])
	
	# sample C_theory from calibration values distributed normally to obtain std
	if Calc_CTheory_std == True:
		C_theory_mean, C_theory_std = MonteCarlo_estimate_std_from_function(calc_contact, 
				[results['ToTF'], results['EF'], results['barnu']], 
				[results['e_ToTF'], results['e_EF'], results['e_barnu']], num=200)
		print("For nominal C_theory={:.2f}".format(results['C_theory']))
		print("MC sampling of normal error gives mean={:.2f}±{:.2f}".format(
									  C_theory_mean, C_theory_std))
		results['C_theory_std'] = C_theory_dist_std
	else:	
		results['C_theory_std'] = 0.02
		
	# clock shift theory prediction from C_Theory
	results['CS_theory'] = 1/(pi*results['kF']*a13(meta_df['Bfield'][0])) \
							* results['C_theory']
	# note this is missing kF uncertainty, would have to do the above again
	results['CS_theory_std'] = 1/(pi*results['kF']*a13(meta_df['Bfield'][0])) \
							* results['C_theory_std']
	
	# check if 'vva' is in .dat file, if not, check if an alternate is, and set it equal
	check_for_col_name(run.data, 'vva', alternates=['VVA', 'amp', 'amplitude'])
	
	# remove indices if requested
	remove_indices = remove_indices_formatter(meta_df['remove_indices'][0])
	if remove_indices is not None:
		run.data.drop(remove_indices, inplace=True)
	
	# correct cloud size
	size_names = ['two2D_sv1', 'two2D_sh1', 'two2D_sv2', 'two2D_sh2']
	new_size_names = ['c5_sv', 'c5_sh', 'c9_sv', 'c9_sh']
	for new_name, name in zip(new_size_names, size_names):
		run.data[new_name] = np.abs(run.data[name])

	# average H and V sizes
	run.data['c5_s'] = (run.data['c5_sv']+run.data['c5_sh'])/2
	run.data['c9_s'] = (run.data['c9_sv']+run.data['c9_sh'])/2

	# data filtering:
	if Filter == True:
		# filter out cloud fits that are too large
		filter_indices = run.data.index[run.data['c5_s'] > 50].tolist()
		run.data.drop(filter_indices, inplace=True)
		
	# length of data set
	num = len(run.data[xname])
	
	#### compute detuning
	run.data['detuning'] = run.data[xname] - meta_df['res_freq'][0]*np.ones(num) # MHz
	
	# fudge the c9 counts using ff
	run.data['c9'] = run.data['c9'] * meta_df['ff'][0]
	
	# determine bg freq to be int, list or range
	bg_freq, bg_freq_type = bg_freq_formatter(meta_df['bg_freq'][0])
	if bg_freq_type == 'single': # select bg at one freq
		bgdf = run.data.loc[run.data['freq'] == bg_freq]
	elif bg_freq_type == 'list': # select bg at a list of freqs
		bgdf = run.data.loc[run.data['freq'].isin(bg_freq)]
	elif bg_freq_type == 'range': # select freq in ranges
		bgdf = pd.concat([run.data.loc[run.data['freq'].between(val[0], 
											  val[1])] for val in bg_freq])
	
	### compute bg values for atom numbers
	if Bg_Tracking == True: # track number drift over time
		for spin in ['c5', 'c9']:
			bg_popt, bg_pcov = curve_fit(linear, bgdf['cyc'], bgdf[spin])
			run.data['bg'+spin] = linear(run.data.cyc, *bg_popt) 
	else:
		run.data['bgc5'] = bgdf.c5.mean()
		run.data['bgc9'] = bgdf.c9.mean()
	
	# calculate numbers minus the bg
	run.data['c5mbg'] = run.data['c5'] - run.data['bgc5']
	run.data['c9mbg'] = run.data['c9'] - run.data['bgc9']
	
	# take N from c5 - bg and c9
	run.data['N'] = run.data['c5mbg']+run.data['c9']
	run.data['Nmbg'] = run.data['N'] - run.data['bgc9']
	
	# calculate number, transfer and loss
# 	run.data['transferbgN'] = (run.data['c5mbg'])/run.data['bgc9']
	run.data['transfer'] = (run.data['c5mbg'])/run.data['N']
	run.data['loss'] = (-run.data['c9mbg'])/run.data['bgc9']
	
	# determine pulse area
	if meta_df['pulsetype'][0] == 'Blackman':
		run.data['sqrt_pulse_area'] = np.sqrt(0.31) 
	elif meta_df['pulsetype'][0] == 'square':
		run.data['sqrt_pulse_area'] = 1
	else:
		ValueError("pulsetype not a known type")
	
	# map VVA and freq to OmegaR
	run.data['OmegaR'] = run.data.apply(lambda x: 2*pi*x['sqrt_pulse_area'] \
 						 * meta_df['gain'][0] * OmegaR_from_VVAfreq(x['vva'], 
													   x[xname]), axis=1)
	
	# compute scaled transfer and contact
	results['x_star'] = xstar(meta_df['Bfield'][0], EF)
	results['OmegaR_pk'] = max(run.data.loc[(run.data['detuning']>0) & (run.data['detuning']<0.1)]['OmegaR'])
	OmegaR_max = 2*pi*1*OmegaR_from_VVAfreq(10, 47)
	# here trf was in s so convert to ms, OmegaR is in kHz
	results['pulsejuice'] = results['OmegaR_pk']**2 * (meta_df['trf'][0]*1e3) / OmegaR_max 
	run.data['ScaledTransfer'] = run.data.apply(lambda x: GammaTilde(x[transfer_selection],
									h*EF*1e6, x['OmegaR']*1e3, meta_df['trf'][0]), axis=1)
	run.data['C'] = run.data.apply(lambda x: 2*np.sqrt(2)*pi**2*x['ScaledTransfer'] * \
								(np.abs(x['detuning'])/EF)**(3/2)*\
							(1+x['detuning']/EF/results['x_star']), axis=1)
		
	### now group by freq to get mean and stddev of mean
	run.group_by_mean(xname)
	
	### rough first fit of data
	# create fit function
	fit_func = GenerateSpectraFit(results['x_star'])
	
	# find fit bounds... min is roughly two Fourier widths up
	xfitmin = 2*results['FourierWidth']/EF
	# max is roughly the trap depth if using transfer, else just make it massive
	if transfer_selection == 'transfer' or transfer_selection == 'transferbgN':	
		xfitmax = meta_df['trap_depth'][0]/EF
	else:
		xfitmax = 10/EF # 10 MHz...
	xfitlims = [xfitmin, xfitmax]
	
	# fit
	xp = run.avg_data['detuning']/EF
	fp = run.avg_data['ScaledTransfer']
	yerr = run.avg_data['em_ScaledTransfer']
	fitmask = xp.between(*xfitlims)
	
	popt, pcov = curve_fit(fit_func, xp[fitmask], fp[fitmask], p0=[0.1], 
						sigma=yerr[fitmask])
	
	### calulate integrals
	xlims = [-2, max(xp)]
	xs = np.linspace(xlims[0], xlims[-1], len(fp))
	
	# interpolate scaled transfer for sumrule integration
	maxfp = max(fp)
	e_maxfp = run.avg_data.iloc[run.avg_data['ScaledTransfer'].idxmax()]['em_ScaledTransfer']
	TransferInterpFunc = lambda x: np.interp(x, np.array(xp), np.array(fp))
	
	# sumrule
	results['SR_interp'] = np.trapz(TransferInterpFunc(xs), x=xs)
	results['SR_extrap'] = dwSpectraFit(xlims[-1], results['x_star'], *popt)
	
	# first moment
	results['FM_interp'] = np.trapz(TransferInterpFunc(xs)*xs, x=xs)
	results['FM_extrap'] = wdwSpectraFit(xlims[-1], results['x_star'], *popt)
	
	results['SR'] = results['SR_interp'] + results['SR_extrap']
	results['FM'] = results['FM_interp'] + results['FM_extrap']
	
	# clock shift
	results['CS'] = results['FM']/results['SR']
	if Talk == True:
		print('raw fit C = {:.2f} ± {:.2f}'.format(pi**2*2**(3/2)*popt[0], 
							   pi**2*2**(3/2)*np.sqrt(np.diag(pcov))[0]))
		print("raw SR {:.3f}".format(results['SR']))
		print("raw FM {:.3f}".format(results['FM']))
		print("raw CS {:.2f}".format(results['CS']))
		
	# last things to add to results dict
	results['Peak Scaled Transfer'] = maxfp
	results['e_Peak Scaled Transfer'] = e_maxfp
		
##########################
##### Bootstrapping ######
##########################
	if Bootstrap == True:
		### Bootstrap resampling for integral uncertainty
		conf = 68.2689  # confidence level for CI
		
		# non-averaged data
		x = np.array(run.data['detuning']/EF)
		y = np.array(run.data['ScaledTransfer'])
		
		# sumrule, first moment and clockshift with analytic extension
		
		if Debug == True:
			print(xfitlims)
		
		SR_BS_dist, FM_BS_dist, CS_BS_dist, A_dist, SR_extrap_dist, \
			FM_extrap_dist, extrapstart = Bootstrap_spectra_fit_trapz(x, y, 
			xfitlims, results['x_star'], fit_func, trialsB=BOOTSRAP_TRAIL_NUM, 
			debug=Debug)
			
		# compute more distributions
		C_dist = A_dist*2*np.sqrt(2)*np.pi**2
		CoSR_dist = C_dist / ( 2  *SR_BS_dist) 
		FM_interp_dist = FM_BS_dist - FM_extrap_dist
			
		# list all ditributions to compure stats on
		dists = [SR_BS_dist, FM_BS_dist, CS_BS_dist, 
				  SR_extrap_dist, FM_extrap_dist, FM_interp_dist,
				  C_dist, CoSR_dist]
		names = ['SR', 'FM', 'CS', 
				  'SR_extrap', 'FM_extrap', 'FM_interp',
				  'C', 'CoSR']
		
		# update results with all stats from dists
		stats_dict = {}
		for name, dist in zip(names, dists):
			for key, value in dist_stats(dist, conf).items():
				stats_dict[name+'_'+key] = value
		results.update(stats_dict)
		
		# Clock Shift prediction from contact
		results['CS_pred_mean'] = 1/(pi*results['kF']*a13(meta_df['Bfield'][0])) \
								* results['C_mean']
		results['CS_pred_std'] = 1/(pi*results['kF']*a13(meta_df['Bfield'][0])) \
								* results['C_std']
		results['CS_pred_median'] = 1/(pi*results['kF']*a13(meta_df['Bfield'][0])) \
								* results['C_median']
		results['CS_pred_lower'] = 1/(pi*results['kF']*a13(meta_df['Bfield'][0])) \
								* results['C_upper'] 
		results['CS_pred_upper'] = 1/(pi*results['kF']*a13(meta_df['Bfield'][0])) \
								* results['C_lower']
								
								
		# Clock Shift from conatct theory 
		results['CS_theory'] = 1/(pi*results['kF']*a13(meta_df['Bfield'][0])) \
								* results['C_theory']
								
########################################
####### Confidence Interval Plot #######
########################################
		# computing CI
		numpts = 100
		alpha = 0.5
		xFit = np.linspace(xfitlims[0], 20, numpts)

		A_mean = results['C_mean']/2/np.sqrt(2)/np.pi**2
		
		print('*** Plotting Confidence Band')
		iis = range(0, len(A_dist))
		ylo = 0*xFit
		yhi = 0*xFit
		for idx, xval in enumerate(xFit):
			fvals = [fit_func(xval, Aval) for Aval in A_dist]
			ylo[idx] = np.nanpercentile(fvals, (100.-conf)/2.)
			yhi[idx] = np.nanpercentile(fvals, 100.-(100.-conf)/2.)

		# plotting CI interval fit on data
		title = f'{filename} at ' + r'$T/T_F=$'+'{:.3f}±{:.3f}'.format(results['ToTF'], results['e_ToTF']) +\
			 ' using ' + transfer_selection
		
		fig_CI, axs = plt.subplots(1,2,figsize=[10,4.5])
		xlabel = r"Detuning $\hbar\omega/E_F$"
		x = run.avg_data['detuning'][fitmask]/EF
		ylabel = r"Scaled Transfer $\tilde\Gamma$"
		y = run.avg_data['ScaledTransfer'][fitmask]
		yerr = run.avg_data['em_ScaledTransfer'][fitmask]
		
		bootstrap_chi2 = chi_sq(y, fit_func(x, A_mean), yerr, len(y)-len(popt))
		raw_chi2 = chi_sq(y, fit_func(x, *popt), yerr, len(y)-len(popt))
			
		# confidence interval plot
		ax = axs[0]
		ax.plot(xFit, fit_func(xFit, *popt), '--', color='red', label='raw fit')
		ax.plot(xFit, fit_func(xFit, A_mean), '--', color=dark_color, label='bootstrap')
		ax.fill_between(xFit, ylo, yhi, color=color, alpha=alpha, label='68% CI')
		ax.errorbar(x, y, yerr=yerr, fmt='o', color=dark_color)
		ax.set(xlabel=xlabel, ylabel=ylabel, xscale='log', yscale='log', 
			 xlim=[1, 20], ylim=[0.9*min(ylo), 1.1*max(yhi)])
		ax.legend()
		
		# residuals
		ax = axs[1]
		ylabel = r"Residuals"
		res_label = r'$\chi^2_{raw}$ = '+'{:.2f}'.format(raw_chi2)
		ax.errorbar(x, y-fit_func(x, *popt), yerr=yerr, fmt='o', label=res_label, 
			  mfc=light_red, mec=dark_red, color=dark_red)
		res_label = r'$\chi^2_{BS}$ = '+'{:.2f}'.format(bootstrap_chi2)
		ax.errorbar(x, y-fit_func(x, A_mean), yerr=yerr, fmt='o', label=res_label, 
			  color=dark_color)
		ax.plot(x, np.zeros(len(x)), 'k--')
		ax.set(xlabel=xlabel, ylabel=ylabel, xscale='log')
		ax.legend()
		
		fig_CI.suptitle(title)
		fig_CI.tight_layout()
		
		if Save == True:
			CI_figname = filename[:-6] + '_CIplot.pdf'
			fig_CI.savefig(os.path.join(figpath, CI_figname))
	
##########################
######## Plotting ########
##########################
	plt.rcParams.update({"figure.figsize": [12,10]})
	fig, axs = plt.subplots(3,3)
	
	x = run.avg_data['detuning']/EF
	label = r"trf={:.0f} us, gain={:.2f}".format(meta_df['trf'][0]*1e6, meta_df['gain'][0])
	
	###
	### (top left) plot transfer fraction
	###
	ax = axs[0,0]
	y = run.avg_data['transfer']
	y2 = run.avg_data['loss']
	yerr = run.avg_data['em_transfer']
	yerr2 = run.avg_data['em_loss']
	ylabel = r"Transfer $\Gamma \,t_{rf}$"
	
	xlims = [-0.04,max(x)]
# 	ylims = [min(min(run.data['transfer']), min(run.data['loss'])),
# 		  max(max(run.data['transfer']), max(run.data['loss']))]
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims)#, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, fmt='o', label='transfer')
	# loss 
	ax.errorbar(x, y2, yerr=yerr2, fmt='s', color=dark_color2, 
			 mfc=light_color2, mec=dark_color2, label='loss')
	ax.vlines(meta_df['trap_depth'][0]/EF, 0, max(max(y), max(y2)), 
		   linestyles='--', colors='k')
	ax.legend()
	
	###
	### (top centre) plot zoomed-in scaled transfer
	###
	ax = axs[0,1]
	y = run.avg_data['ScaledTransfer']
	yerr = run.avg_data['em_ScaledTransfer']
	ylabel = r"Scaled Transfer $\tilde\Gamma$"
	
# 	ylims = [-0.5/20, max(run.data['ScaledTransfer'])]
	xlims = [-3,3]
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims)#, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, fmt='o', label=label)
	ax.legend()
	
	###
	### (top right) plot contact
	###
	ax = axs[0,2]
	y = run.avg_data['C']
	yerr = run.avg_data['em_C']
	ylabel = r"Contact $C/N$ [$k_F$]"
	
	xlims = [-2, max(x)]
# 	ylims = [-0.1, max(run.data['C'])]
	Cdetmin = 2 
	xs = np.linspace(Cdetmin, xfitmax, num)
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims)#, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, fmt='o')
	if Bootstrap == True:
		ax.plot(xs, results['C_mean'] * np.ones(num), "--")
		ax.fill_between(xs, results['C_mean']-results['C_std'],
				 results['C_mean']+results['C_std'], color=color2, alpha=alpha)
	ax.vlines(meta_df['trap_depth'][0]/EF, 0, max(y), linestyles='--', 
		   colors='k')
	
	###
	### (middle left) C5 and C9
	###
	ax = axs[1,0]
	y = run.avg_data['c5mbg']
	y2 = run.avg_data['c9mbg']
	yerr = run.avg_data['em_c5mbg']
	yerr2 = run.avg_data['em_c9mbg']
	ylabel = r"$N_\sigma - N_{\sigma, bg}$"
	
	xlims = [-2, max(x)]
	ylims = [min(min(run.data['c5mbg']), 
			  min(run.data['c9mbg'])),
			  max(max(run.data['c5mbg']), 
			 max(run.data['c9mbg']))]
	xs = np.linspace(*xlims, num)
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims)#, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, fmt='o', label='c5')
	ax.errorbar(x, y2, yerr=yerr2, fmt='s', color=dark_color2, 
			 mfc=light_color2, mec=dark_color2, label='c9')
	ax.plot(xs, 0*np.ones(num), "--", color='k')
 	# ax.plot(xs, results['bgc9']*np.ones(num), "--", color=color2)
	ax.vlines(meta_df['trap_depth'][0]/EF, min(min(y), min(y2)), 
		   max(max(y), max(y2)), linestyles='--', colors='k')
	ax.legend()
	
	###
	### (middle centre) c5 cloud sizes
	###
	ax = axs[1,1]
	y = run.avg_data['c5_sv']
	yerr = run.avg_data['em_c5_sv']
	ylabel = r"c5 size [px]"
	
	xlims = [-2, max(x)]
	ylims = [min(y), max(y)]
	xs = np.linspace(*xlims, num)
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims)
	ax.errorbar(x, y, yerr=yerr, fmt='o', label='c5')
	ax.vlines(meta_df['trap_depth'][0]/EF, min(y), max(y), linestyles='--', colors='k')
# 	ax.legend()
		
	###
	### (middle right) c9 cloud sizes
	###
	ax = axs[1,2]
	y = run.avg_data['c9_sv']
	yerr = run.avg_data['em_c9_sv']
	ylabel = r"c9 size [px]"
	
	xlims = [-2, max(x)]
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims)
	ax.errorbar(x, y, yerr=yerr, fmt='s', color=dark_color2, 
			 mfc=light_color2, mec=dark_color2, label='c9')
	ax.vlines(meta_df['trap_depth'][0]/EF, min(y), max(y), linestyles='--', colors='k')
# 	ax.legend()
	
	###
	### (bottom left) Total atom number vs. detuning
	###
	ax = axs[2,0]
	y = run.avg_data['Nmbg']
	yerr = run.avg_data['em_Nmbg']
	ylabel = r"$N$"
	
	xlims = [-2, max(x)]
 	# ylims = [min(run.data['N']), max(run.data['N'])]
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims)#, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, fmt='o')
	ax.plot(x, np.zeros(len(x)), "--", color=color)
	ax.vlines(meta_df['trap_depth'][0]/EF, min(y), max(y), linestyles='--', colors='k')
	
	###
	### (bottom center) Total atom number vs. time
	###
	ax = axs[2,1]
	x = 31/60*run.data['cyc']  # time in ~ minutes
	y = run.data['N']
	xlabel = r"Time since scan start [~min]"
	ylabel = r"$N$"
	
	xlims = [-0.1, max(x)]
	ylims = [0.97*min(run.data['N']), 1.03*max(run.data['N'])]
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims, ylim=ylims)
	ax.plot(x, y, 'o')
	ax.plot(x, run.data['bgc9'], "--", color=color)

	###
	### (bottom right) generate table
	###
	ax = axs[2,2]
	ax.axis('off')
	quantities = ["Run", r"$T/T_F$", r"$E_F$", "barnu"]
	values = [filename[:-6],
			   "{:.2f}±{:.2f}".format(results['ToTF'], results['e_ToTF']),
			   "{:.1f}±{:.1f}kHz".format(results['EF']*1e3, results['e_EF']*1e3),
			   "{:.0f}±{:.0f}Hz".format(results['barnu'], results['e_barnu'])
			  ]
	if Bootstrap == True:
		quantities += ["SR mean", "C mean", "C theory", # "C/(2SR)",
					"CS mean", "CS theory"]
		values += [r"{:.3f}±{:.3f}".format(results['SR_mean'], results['SR_std']),
				  r"{:.2f}±{:.2f}$k_F$".format(results['C_mean'], results['C_std']),
				  r"{:.2f}±{:.2f}$k_F$".format(results['C_theory'], results['C_theory_std']),
# 				  r"{:.2f}±{:.2f}$k_F$".format(results['CoSR_mean'], results['CoSR_std']),
				   r"{:.2f}±{:.2f}".format(results['CS_mean'], results['CS_std']),
# 				   r"{:.2f}±{:.2f}".format(results['CS_pred_mean'], results['CS_pred_std']),
				   r"{:.2f}±{:.2f}".format(results['CS_theory'], results['C_theory_std'])]
	
	table = list(zip(quantities, values))
	
	the_table = ax.table(cellText=table, loc='center')
	the_table.auto_set_font_size(False)
	the_table.set_fontsize(12)
	the_table.scale(1,1.5)
	
	fig.suptitle(title)
	fig.tight_layout()
	
	###
	### save figure
	###
	if Save == True:
		os.makedirs(figpath, exist_ok=True)
		plots_figname = filename[:-6] + '_plots.pdf'
		fig.savefig(os.path.join(figpath, plots_figname))
	
###########################
####### Histograms ########
###########################

	if Bootstrap == True:
		plt.rcParams.update({"figure.figsize": [10,8]})
		fig, axs = plt.subplots(2,3)
		fig.suptitle(title)
		
		bins = 20
		
		ylabel = "Occurances"
		xlabels = ["Sum Rule", "First Moment", "Clock Shift", 
				 "Contact", "Contact over Sum Rule", "FM Interpolation"]
		dists = [SR_BS_dist, FM_BS_dist, CS_BS_dist, 
			   C_dist, CoSR_dist, FM_interp_dist]
		names = ['SR', 'FM', 'CS', 'C', 'CoSR', 'FM_interp']
		
		for ax, xlabel, dist, name in zip(axs.flatten(), xlabels, dists, names):
			ax.set(xlabel=xlabel, ylabel=ylabel)
			ax.hist(dist, bins=bins)
			ax.axvline(x=results[name+'_lower'], color='red', alpha=0.5, linestyle='--', marker='')
			ax.axvline(x=results[name+'_upper'], color='red', alpha=0.5, linestyle='--', marker='')
			ax.axvline(x=results[name+'_median'], color='red', linestyle='--', marker='')
			ax.axvline(x=results[name+'_mean'], color='k', linestyle='--', marker='')
		fig.tight_layout()	
		
		if Save == True:
			hist_figname= filename[:-6] + '_hist.pdf'
			fig.savefig(os.path.join(figpath, hist_figname))
		
#############################
####### Correlations ########
#############################
		
	if Correlations == True and Bootstrap == True:
		dists = np.vstack([C_dist, SR_BS_dist, FM_BS_dist, CS_BS_dist, FM_interp_dist])
		labels = ['Contact','Sum Rule','First Moment','Clock Shift','FM Interp']
		figure = corner.corner(dists.T, labels=labels)
		
		if Save == True:
			corner_figname = filename[:-6] + '_corner.pdf'
			figure.savefig(os.path.join(figpath, corner_figname))

###############################
####### Saving Results ########
###############################
	
	if Save == True:
		savedf = pd.DataFrame(results, index=[save_df_index])
		save_df_index += 1
		
		save_df_row_to_xlsx(savedf, savefile, filename)
		
		