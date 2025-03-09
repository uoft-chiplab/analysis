"""
Created by Chip lab 2024-11-21

Loads .dat with contact HFT scan and computes scaled transfer. Plots. Also
computes the spectral weight.

To do:
	Error in x_star
	
"""
BOOTSRAP_TRAIL_NUM = 1000

# paths
import os
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')
figfolder_path = os.path.join(proj_path, 'figures')

from library import pi, h, hbar, mK, a0, plt_settings, GammaTilde, chi_sq, \
	styles, colors, dark_colors
from data_helper import check_for_col_name, bg_freq_formatter, remove_indices_formatter
from save_df_to_xlsx import save_df_row_to_xlsx
from data_class import Data
from scipy.optimize import curve_fit
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
from clockshift.MonteCarloSpectraIntegration import Bootstrap_spectra_fit_trapz, \
					dist_stats, MonteCarlo_estimate_std_from_function
from clockshift.HFT_scaling_analysis import HFT_sat_correction, HFT_loss_sat_correction
from clockshift.resonant_transfer_scaling_model import res_sat_correction
from contact_correlations.UFG_analysis import calc_contact
# from Blackman_envelope import BlackmanFourier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner

# fitting options
fit_options = {'min': '2EF',  # 2EF or 2 Fourier Widths
			   }
	
### This turns on (True) and off (False) saving the data/plots 
Save = False
Save_run_df = False

### script options
Analysis = True
Bootstrap = False  # do bootstrap analysis
Correlations = False  # plot corner correlation plots
Debug = False
Filter_Size = True  # exclude when c5 cloud is not fit well 
Talk = True  # print statements

Saturation_Correction = True
Bg_Tracking = False  # trap bg over time and rescale bg
Calc_CTheory_std = False  # Monte Carlo estimate CTheory std

sat_correction = HFT_sat_correction
		
### metadata
metadata_filename = 'HFT_metadata_file.xlsx'
metadata_file = os.path.join(proj_path, metadata_filename)
metadata = pd.read_excel(metadata_file)

# select files
transfer_selection = 'transfer'  # 'transfer' or  'loss'

exclude_name = 'exclude_transfer'
if transfer_selection == 'loss':
	exclude_name = 'exclude_loss'
files =  metadata.loc[metadata[exclude_name] == 0]['filename'].values

# Manual file select, comment out if exclude column should be used instead
# files = ["2024-09-10_L_e"]
# files = ["2024-10-08_F_e"]
# files=["2024-09-24_C_e"]
# files = [files[0]]

# save file path
savefilename = 'HFT_analysis_results.xlsx' 

if transfer_selection == 'loss':
	savefilename = 'loss_'+savefilename
	sat_correction = HFT_loss_sat_correction
if Saturation_Correction == True:
	savefilename = 'corrected_'+savefilename
savefile = os.path.join(proj_path, savefilename)

### Vpp calibration
# VpptoOmegaR = 27.5833 # kHz/Vpp, older calibration
VpptoOmegaR = 17.05/0.728  # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
OmegaR_from_VVAfreq = lambda Vpp, freq: VpptoOmegaR * Vpp_from_VVAfreq(Vpp, freq)
	
### contants
re = 107 * a0  # ac dimer range estimate
Eb = 3.98  # MHz # I guesstimated this from recent ac dimer spectra

def a13(B):
	''' ac scattering length '''
	abg = 167.6*a0
	DeltaB = 7.2
	B0 = 224.2
	return abg*(1 - DeltaB/(B-B0))

def xstar(B, EF):
	return Eb/EF  # hbar**2/mK/a13(B)**2 * (1-re/a13(Bfield))**(-1)

def GenerateSpectraFit(xstar):
	def fit_func(x, A):
		xmax = xstar
		return A*x**(-3/2) / (1+x/xmax)
	return fit_func

def dwSpectraFit(xi, x_star, A):
	return A*2*(1/np.sqrt(xi)-np.arctan(np.sqrt(x_star/xi))/np.sqrt(x_star))

def wdwSpectraFit(xi, x_star, A):
	return A*2*np.sqrt(x_star)*np.arctan(np.sqrt(x_star/xi))

def linear(x, a, b):
	return a*x + b

### plot settings
plt.rcParams.update(plt_settings) # from library.py
plt.rcParams.update({"figure.figsize": [12,8],
					 "font.size": 14,
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
	
	xname = 'freq'
		
	# EF, it's used a lot soooo
	EF = meta_df['EF'][0] # MHz
	
	# create data structure
	filename = filename + ".dat"
	run = Data(filename, path=data_path)
	runfolder = filename 
	figpath = os.path.join(figfolder_path, runfolder)
	
	# initialize results dict to turn into df
	results = {
			'Run': filename,
			'Transfer': transfer_selection,
			'Pulse Time (us)': meta_df['trf'][0]*1e6,
			'Pulse Type': meta_df['pulsetype'][0],
			'ToTF': meta_df['ToTF'][0],
			'e_ToTF': meta_df['ToTF_sem'][0],
			'EF': EF,
			'e_EF': meta_df['EF_sem'][0],
			'kF': np.sqrt(2*mK*EF*h*1e6)/hbar,
			'barnu': meta_df['barnu'][0],
			'e_barnu': meta_df['barnu_sem'][0],
			'FourierWidth': 2/meta_df['trf'][0]/1e6, 
			}
	
	results['kFa13'] = results['kF'] * a13(meta_df['Bfield'][0])
	
	# from Tilman's unitary gas harmonic trap averaging code
	results['C_theory'], results['Ns_theory'], results['EF_theory'], \
		results['ToTF_theory'] = calc_contact(results['ToTF'], results['EF'], 
								results['barnu'])
		
	results['dimer_SW_theory'] = results['kFa13']/pi/2 * results['C_theory']
	results['SW_theory'] = 0.5 - results['dimer_SW_theory']
	
	# sample C_theory from calibration values distributed normally to obtain std
	if Calc_CTheory_std == True:
		C_theory_mean, C_theory_std = MonteCarlo_estimate_std_from_function(calc_contact, 
				[results['ToTF'], results['EF'], results['barnu']], 
				[results['e_ToTF'], results['e_EF'], results['e_barnu']], num=200)
		if Talk == True:
			print("For nominal C_theory={:.2f}".format(results['C_theory']))
			print("MC sampling of normal error gives mean={:.2f}±{:.2f}".format(
									  C_theory_mean, C_theory_std))
		results['C_theory_std'] = C_theory_std
	else:	
		results['C_theory_std'] = 0.02  # this is close to the usual std
		
	# clock shift theory prediction from C_Theory
	results['CS_theory'] = results['C_theory']/pi
	results['CS_theory_std'] = results['C_theory_std']/pi
	
	# check if 'vva' is in .dat file, if not, check if an alternate is, and set it equal
	check_for_col_name(run.data, 'vva', alternates=['VVA', 'amp', 'amplitude'])
	
	# remove indices if requested
	remove_indices = remove_indices_formatter(meta_df['remove_indices'][0])
	if remove_indices is not None:
		run.data.drop(remove_indices, inplace=True)
	
	# correct cloud size nanes
	size_names = ['two2D_sv1', 'two2D_sh1', 'two2D_sv2', 'two2D_sh2']
	new_size_names = ['c5_sv', 'c5_sh', 'c9_sv', 'c9_sh']
	for new_name, name in zip(new_size_names, size_names):
		run.data[new_name] = np.abs(run.data[name])

	# average H and V sizes as geometric mean
	run.data['c5_s'] = np.sqrt(run.data['c5_sv']**2+run.data['c5_sh']**2)
	run.data['c9_s'] = np.sqrt(run.data['c9_sv']**2+run.data['c9_sh']**2)

	# filter data by size of c5, i.e. did the 5 cloud fit?
	if Filter_Size == True:
		# filter out cloud fits that are too large
		filter_indices = run.data.index[run.data['c5_s'] > 50].tolist()
		run.data.drop(filter_indices, inplace=True)
		
	# length of data set
	num = len(run.data[xname])
	
	#### compute detuning
	run.data['detuning'] = run.data[xname] - meta_df['res_freq'][0]*np.ones(num) # MHz
	
	# fudge the c9 counts using ff
	run.data['c9'] = run.data['c9'] * meta_df['ff'][0]
	
	# determine bg freq to be int, list or range, put points in bgdf
	bg_freq, bg_freq_type = bg_freq_formatter(meta_df['bg_freq'][0])
	if bg_freq_type == 'single':  # select bg at one freq
		bgdf = run.data.loc[run.data['freq'] == bg_freq]
	elif bg_freq_type == 'list':  # select bg at a list of freqs
		bgdf = run.data.loc[run.data['freq'].isin(bg_freq)]
	elif bg_freq_type == 'range':  # select freq in ranges
		bgdf = pd.concat([run.data.loc[run.data['freq'].between(val[0], 
											  val[1])] for val in bg_freq])
	
	### compute bg values for atom numbers
	if Bg_Tracking == True:  # track number drift over time
		for spin in ['c5', 'c9']:
			bg_popt, bg_pcov = curve_fit(linear, bgdf['cyc'], bgdf[spin])
			run.data['bg'+spin] = linear(run.data.cyc, *bg_popt) 
	else:  # no tracking, just average over whole dataset
		run.data['bgc5'] = bgdf.c5.mean()
		run.data['bgc9'] = bgdf.c9.mean()
	
	# calculate numbers minus the bg
	run.data['c5mbg'] = run.data['c5'] - run.data['bgc5']
	run.data['c9mbg'] = run.data['c9'] - run.data['bgc9']
	
	# take N from c5 - bg and c9
	if meta_df['c9_state'][0] == 7:
		run.data['N'] = run.data['c5mbg'] + run.data['c9']
	# or if we had the wrong c9 state, just use that as atom number
	elif meta_df['c9_state'][0] == 9:
		run.data['N'] = run.data['c9']
	else:
		raise KeyError("No c9_state in metadata (or it's wrong!)")
		
	run.data['Nmbg'] = run.data['N'] - run.data['bgc9']
	
	# calculate number, transfer and loss
	run.data['transferbgN'] = (run.data['c5mbg'])/run.data['bgc9']
	run.data['transfer'] = (run.data['c5mbg'])/run.data['N']
	run.data['loss'] = -run.data['c9mbg']/run.data['bgc9']
	
	# find fit bounds... min is roughly two Fourier widths up
	if fit_options['min'] == 'Two Fourier Widths':
		xfitmin = 2*results['FourierWidth']/EF
	elif fit_options['min'] == '2EF':
		xfitmin = 2
	# max is roughly the trap depth if using transfer, else just make it massive
	if transfer_selection == 'transfer' or transfer_selection == 'transferbgN':	
		xfitmax = meta_df['trap_depth'][0]/EF
	else:
		xfitmax = 10/EF # 10 MHz...
	xfitlims = [xfitmin, xfitmax]
	
	# TODO add more measurements of saturation corrections
	if Saturation_Correction == True: 
		# if data is within fit bounds, apply the saturation correction factor
		run.data['in_HFT'] = (run.data['detuning']/EF > xfitlims[0]) & \
					(run.data['detuning']/EF < xfitlims[1])
		run.data[transfer_selection] = np.where(run.data['in_HFT'] == True,
                                           sat_correction(run.data[transfer_selection]),
                                          run.data[transfer_selection])
		
		# if in fourier widths, rescale like resonant transfeer
		fourier_width_lims = [-results['FourierWidth']/EF, results['FourierWidth']/EF]
		run.data['in_Fourier'] = (run.data['detuning']/EF > fourier_width_lims[0]) & \
					(run.data['detuning']/EF < fourier_width_lims[1])
		run.data[transfer_selection] = np.where(run.data['in_Fourier'] == True,
                                           res_sat_correction(run.data[transfer_selection]),
                                          run.data[transfer_selection])
	
	# determine pulse area
	if meta_df['pulsetype'][0] == 'Blackman':
		run.data['sqrt_pulse_area'] = np.sqrt(0.308218) 
	elif meta_df['pulsetype'][0] == 'square':
		run.data['sqrt_pulse_area'] = 1
	else:
		ValueError("pulsetype not a known type")
	
	# map VVA and freq to OmegaR
	run.data['OmegaR'] = run.data.apply(lambda x: 2*pi*x['sqrt_pulse_area'] \
 						 * meta_df['gain'][0] * OmegaR_from_VVAfreq(x['vva'], 
													   x[xname]), axis=1)
	
	# find largest OmegaR and compute some other things
	results['OmegaR_pk'] = max(run.data.loc[(run.data['detuning'] > 0) \
								 & (run.data['detuning'] < 0.1)]['OmegaR'])
	OmegaR_max = 2*pi*1*OmegaR_from_VVAfreq(10, 47)
	results['pulsejuice'] = results['OmegaR_pk']**2 * (meta_df['trf'][0]*1e3) / OmegaR_max 
	
	# compute scaled transfer
	# here trf was in s so convert to ms, OmegaR is in kHz
	run.data['ScaledTransfer'] = run.data.apply(lambda x: GammaTilde(x[transfer_selection],
									h*EF*1e6, x['OmegaR']*1e3, meta_df['trf'][0]), axis=1)
	
	# find x_star
	results['x_star'] = xstar(meta_df['Bfield'][0], EF)
	
	# compute contact at each detuning
	run.data['C'] = run.data.apply(lambda x: 2*np.sqrt(2)*pi**2*x['ScaledTransfer'] * \
								(np.abs(x['detuning'])/EF)**(3/2)*\
							(1 + x['detuning']/EF/results['x_star']), axis=1)
		
	### now group by freq to get mean and stddev of mean
	run.group_by_mean(xname)
	
	if Save_run_df:
		savefolder = os.path.join(proj_path, 'analyzed_data')
		run.avg_data.to_pickle(os.path.join(savefolder, filename[:-4]+'.pkl'))
	
	### rough first fit of data
	# create fit function
	fit_func = GenerateSpectraFit(results['x_star'])
	
	# fit
	xp = run.avg_data['detuning']/EF
	fp = run.avg_data['ScaledTransfer']
	yerr = run.avg_data['em_ScaledTransfer']
	fitmask = xp.between(*xfitlims)
	
	A_guess = results['C_theory']/(2*np.sqrt(2)*np.pi**2)
	popt, pcov = curve_fit(fit_func, xp[fitmask], fp[fitmask], p0=[A_guess], 
						sigma=yerr[fitmask])
	perr = np.sqrt(np.diag(pcov))
	
	### calulate integrals
	xlims = [-2, max(xp)]
	xs = np.linspace(xlims[0], xlims[-1], len(fp))
	
	# interpolate scaled transfer for spectral weight integration
	TransferInterpFunc = lambda x: np.interp(x, np.array(xp), np.array(fp))
	
	# spectral weight
	results['SW_interp'] = np.trapz(TransferInterpFunc(xs), x=xs)
	results['SW_extrap'] = dwSpectraFit(xlims[-1], results['x_star'], *popt)
	
	# first moment
	results['FM_interp'] = np.trapz(TransferInterpFunc(xs)*xs, x=xs)
	results['FM_extrap'] = wdwSpectraFit(xlims[-1], results['x_star'], *popt)
	
	results['SW'] = results['SW_interp'] + results['SW_extrap']
	results['FM'] = results['FM_interp'] + results['FM_extrap']
	
	# clock shift
	results['CS'] = results['FM']/results['SW'] * results['kFa13']
	if Talk == True:
		print('raw fit C = {:.2f} ± {:.2f}'.format(pi**2*2**(3/2)*popt[0], 
							   pi**2*2**(3/2)*perr[0]))
		print("raw SW {:.3f}".format(results['SW']))
		print("raw FM {:.3f}".format(results['FM']))
		print("raw CS {:.2f}".format(results['CS']))
		
	# Find the peak scaled transfer
	maxfp = max(fp)
	e_maxfp = run.avg_data.iloc[run.avg_data['ScaledTransfer'].idxmax()]['em_ScaledTransfer']
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
		
		# spectral weight, first moment and clockshift with analytic extension
		
		if Debug == True:
			print(xfitlims)
		
		# bootstrap fit the data
		SW_dist, FM_dist, CS_dist, A_dist, SW_extrap_dist, \
			FM_extrap_dist, extrapstart = Bootstrap_spectra_fit_trapz(x, y, 
			xfitlims, results['x_star'], fit_func, trialsB=BOOTSRAP_TRAIL_NUM, 
			debug=Debug)
			
		# fix clockshift distribution
		CS_dist = CS_dist * results['kFa13']
			
		# compute more distributions
		C_dist = A_dist*2*np.sqrt(2)*np.pi**2
		
		# spectral weight predictions, subtracting dimer SW
		SW_pred_dist = 0.5 - results['kFa13']/pi/2 * C_dist
		
		# spectral weight correction distribution
		SW_corr_dist = SW_pred_dist/SW_dist
		
		# Contact distribution corrected by SW
		CoSW_dist = C_dist * SW_corr_dist
		
		# FM distribution without extrapolated portion
		FM_interp_dist = FM_dist - FM_extrap_dist
			
		# list all ditributions to compute stats on
		dists = [SW_dist, FM_dist, CS_dist, 
				  SW_extrap_dist, FM_extrap_dist, FM_interp_dist,
				  C_dist, CoSW_dist, SW_corr_dist]
		names = ['SW', 'FM', 'CS', 
				  'SW_extrap', 'FM_extrap', 'FM_interp',
				  'C', 'CoSW', 'SW_corr']
		
		# update results with all stats from dists
		stats_dict = {}
		for name, dist in zip(names, dists):
			for key, value in dist_stats(dist, conf).items():
				stats_dict[name+'_'+key] = value
		results.update(stats_dict)
		
		# compute SW correction
		
		# Clock Shift prediction from contact
		results['CS_pred_mean'] = results['C_mean']/pi
		results['CS_pred_std'] = results['C_std']/pi
		results['CS_pred_median'] = results['C_median']/pi
		results['CS_pred_lower'] = results['C_upper'] /pi
		results['CS_pred_upper'] = results['C_lower']/pi
								
		# Clock Shift from conatct theory 
		results['CS_theory'] = results['C_theory']/pi
								
########################################
####### Confidence Interval Plot #######
########################################
		# computing CI
		numpts = 100
		alpha = 0.5
		xFit = np.linspace(*xfitlims, numpts)

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
		sty = styles[0] # blue
		sty2 = styles[3] # red
		
		ax = axs[0]
		ax.plot(xFit, fit_func(xFit, *popt), '--', color='red', label='raw fit')
		ax.plot(xFit, fit_func(xFit, A_mean), '--', color=dark_colors[0], label='bootstrap')
		ax.fill_between(xFit, ylo, yhi, color=colors[0], alpha=alpha, label='68% CI')
		ax.errorbar(x, y, yerr=yerr, **sty)
		ax.set(xlabel=xlabel, ylabel=ylabel, xscale='log', yscale='log', 
			 xlim=xfitlims, ylim=[0.9*min(ylo), 1.1*max(yhi)])
		ax.legend()
		
		# residuals
		ax = axs[1]
		ylabel = r"Residuals"
		res_label = r'$\chi^2_{raw}$ = '+'{:.2f}'.format(raw_chi2)
		ax.errorbar(x, y-fit_func(x, *popt), yerr=yerr, label=res_label, 
			  **sty2)
		res_label = r'$\chi^2_{BS}$ = '+'{:.2f}'.format(bootstrap_chi2)
		ax.errorbar(x, y-fit_func(x, A_mean), yerr=yerr, label=res_label, 
			  **sty)
		ax.plot(x, np.zeros(len(x)), 'k--')
		ax.set(xlabel=xlabel, ylabel=ylabel)
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
	title = f'{filename} at ' + r'$T/T_F=$'+'{:.3f}±{:.3f}'.format(results['ToTF'], results['e_ToTF']) +\
			 ' using ' + transfer_selection
	x = run.avg_data['detuning']/EF
	label = r"trf={:.0f} us, gain={:.2f}".format(meta_df['trf'][0]*1e6, meta_df['gain'][0])
	
	sty = styles[0]  # blue
	sty2 = styles[1]  # orange
	
	###
	### (top left) plot transfer fraction
	###
	ax = axs[0,0]
	y = run.avg_data['transfer']
	y2 = run.avg_data['loss']
	yerr = run.avg_data['em_transfer']
	yerr2 = run.avg_data['em_loss']
	xlabel = r"Detuning [EF]"
	ylabel = r"Transfer $\Gamma \,t_{rf}$"
	
	xlims = [-0.04,max(x)]
# 	ylims = [min(min(run.data['transfer']), min(run.data['loss'])),
# 		  max(max(run.data['transfer']), max(run.data['loss']))]
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims)#, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, label='transfer', **sty)
	# loss 
	ax.errorbar(x, y2, yerr=yerr2,label='loss',  **sty2)
	ax.vlines(meta_df['trap_depth'][0]/EF, 0, max(max(y), max(y2)), 
		   linestyles='--', colors='k')
	ax.legend()
	
	###
	### (top centre) plot zoomed-in scaled transfer
	###
	ax = axs[0,1]
	y = run.avg_data['ScaledTransfer']
	yerr = run.avg_data['em_ScaledTransfer']
	xlabel = r"Detuning [MHz]"
	ylabel = r"Scaled Transfer $\tilde\Gamma$"
	
# 	ylims = [-0.5/20, max(run.data['ScaledTransfer'])]
	xlims = [-2,2]
	
	x = run.avg_data['detuning']
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims)#, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, label=label, **sty)
	ax.set_yscale("log")
	#ax.legend()
	
	
	x = run.avg_data['detuning']/EF
	###
	### (top right) plot contact
	###
	ax = axs[0,2]
	y = run.avg_data['C']
	yerr = run.avg_data['em_C']
	xlabel = r"Detuning [EF]"
	ylabel = r"Contact $C/N$ [$k_F$]"
	
	xlims = [-2, max(x)]
# 	ylims = [-0.1, max(run.data['C'])]
	Cdetmin = 2 
	xs = np.linspace(Cdetmin, xfitmax, num)
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims)#, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, **sty)
	if Bootstrap == True:
		ax.plot(xs, results['C_mean'] * np.ones(num), "--", color=colors[0])
		ax.fill_between(xs, results['C_mean']-results['C_std'],
				 results['C_mean']+results['C_std'], color=colors[0], alpha=alpha)
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
	ax.errorbar(x, y, yerr=yerr, label='c5', **sty)
	ax.errorbar(x, y2, yerr=yerr2, label='c9', **sty2)
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
	ax.errorbar(x, y, yerr=yerr, label='c5', **sty)
	ax.vlines(meta_df['trap_depth'][0]/EF, min(y), max(y), linestyles='--', 
		   colors='k')
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
	ax.errorbar(x, y, yerr=yerr, label='c9', **sty2)
	ax.vlines(meta_df['trap_depth'][0]/EF, min(y), max(y), linestyles='--', 
		   colors='k')
# 	ax.legend()
	
	###
	### (bottom left) Total atom number vs. detuning
	###
	ax = axs[2,0]
	y = run.avg_data['Nmbg']
	yerr = run.avg_data['em_Nmbg']
	xlabel = r"Detuning [EF]"
	ylabel = r"$N$"
	
	xlims = [-2, max(x)]
 	# ylims = [min(run.data['N']), max(run.data['N'])]
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims)#, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, **sty)
	ax.plot(x, np.zeros(len(x)), "--", color=colors[0])
	ax.vlines(meta_df['trap_depth'][0]/EF, min(y), max(y), 
		   linestyles='--', colors='k')
	
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
	ax.plot(x, y, **sty)
	ax.plot(x, run.data['bgc9'], "--", color=colors[0])

	###
	### (bottom right) generate table
	###
	ax = axs[2,2]
	ax.axis('off')
	quantities = ["Run", r"$T/T_F$", r"$E_F$", "barnu"]
	values = [filename[:-6],
			   "{:.3f}({:.0f})".format(results['ToTF'], 1e3*results['e_ToTF']),
			   "{:.1f}({:.0f})kHz".format(results['EF']*1e3, results['e_EF']*1e4),
			   "{:.0f}({:.0f})Hz".format(results['barnu'], results['e_barnu'])
			  ]
	if Bootstrap == True:
		quantities += ["SW mean", "C mean", "C theory", "SW corr.",
					 r"C $\times$ SW corr.",
# 					"CS mean", "CS theory",
					]
		values += [
			r"{:.2f}({:.0f})".format(results['SW_mean'], 1e2*results['SW_std']),
			r"{:.2f}({:.0f})$k_F$".format(results['C_mean'], 1e2*results['C_std']),
			r"{:.2f}({:.0f})$k_F$".format(results['C_theory'], 1e2*results['C_theory_std']),
			r"{:.2f}({:.0f})".format(results['SW_corr_mean'], 1e2*results['SW_corr_std']),
			r"{:.2f}({:.0f})$k_F$".format(results['CoSW_mean'], 1e2*results['CoSW_std']),
# 			r"{:.2f}±{:.2f}".format(results['CS_mean'], results['CS_std']),
# 		r"{:.2f}±{:.2f}".format(results['CS_pred_mean'], results['CS_pred_std']),
# 		r"{:.2f}±{:.2f}".format(results['CS_theory'], results['C_theory_std']),
				   ]
	
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
		xlabels = ["Spectral Weight", "First Moment", "Clock Shift", 
				 "Contact", "Contact over SW", "FM Interpolation"]
		dists = [SW_dist, FM_dist, CS_dist, 
			   C_dist, CoSW_dist, FM_interp_dist]
		names = ['SW', 'FM', 'CS', 'C', 'CoSW', 'FM_interp']
		
		for ax, xlabel, dist, name in zip(axs.flatten(), xlabels, dists, names):
			ax.set(xlabel=xlabel, ylabel=ylabel)
			ax.hist(dist, bins=bins)
			ax.axvline(x=results[name+'_lower'], color='red', alpha=0.5, 
			  linestyle='--', marker='')
			ax.axvline(x=results[name+'_upper'], color='red', alpha=0.5, 
			  linestyle='--', marker='')
			ax.axvline(x=results[name+'_median'], color='red', 
			  linestyle='--', marker='')
			ax.axvline(x=results[name+'_mean'], color='k', 
			  linestyle='--', marker='')
		fig.tight_layout()	
		
		if Save == True:
			hist_figname= filename[:-6] + '_hist.pdf'
			fig.savefig(os.path.join(figpath, hist_figname))
		
#############################
####### Correlations ########
#############################
		
	if Correlations == True and Bootstrap == True:
		dists = np.vstack([C_dist, SW_dist, FM_dist, 
					 CS_dist, FM_interp_dist])
		labels = ['Contact','Spectral Weight','First Moment',
			'Clock Shift','FM Interp']
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
		
		