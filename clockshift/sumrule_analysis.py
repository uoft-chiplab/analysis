"""
Created by Chip lab 2024-06-12

Loads .dat with contact HFT scan and computes scaled transfer. Plots. Also
computes the sumrule.

To do:
	Calculate EF for each shot
	Check pulse area calculations
	More commenting
	Filter results when summary plotting
	....
	
"""
BOOTSRAP_TRAIL_NUM = 5000

# paths
import os
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')
figfolder_path = os.path.join(proj_path, 'figures')

from library import pi, h, hbar, mK, a0, plt_settings, GammaTilde, tintshade, \
	 tint_shade_color, markers, colors
from data_class import Data
from scipy.optimize import curve_fit
from scipy.stats import sem
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
from clockshift.MonteCarloSpectraIntegration import Bootstrap_spectra_fit_trapz
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
Talk = True
BootstrapHists = True
Correlations = True
Debug = False

transfer_selection = 'loss' #'transfer' or  'loss'

### metadata
metadata_filename = 'metadata_file.xlsx'
metadata_file = os.path.join(proj_path, metadata_filename)
metadata = pd.read_excel(metadata_file)
files =  metadata.loc[metadata['exclude'] == 0]['filename'].values
if transfer_selection == 'loss':
	files =  ["2024-09-12_E_e",
			   "2024-09-18_F_e",
			   "2024-09-18_K_e"] # loss files
files = ["2024-09-12_E_e"]

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

def GenerateSpectraFit(xstar):
	def fit_func(x, A):
		xmax = xstar
	# 	print('xstar = {:.3f}'.format(xmax))
		return A*x**(-3/2) / (1+x/xmax)
	return fit_func

def dwSpectraFit(xi, x_star, A):
	return A*2*(1/np.sqrt(xi)-np.arctan(np.sqrt(x_star/xi))/np.sqrt(x_star))

def wdwSpectraFit(xi, x_star, A):
	return A*2*np.sqrt(x_star)*np.arctan(np.sqrt(x_star/xi))

def dist_stats(dist, CI):
	""" Computes the median, upper confidence interval (CI), lower CI, mean 
		and standard deviation for a distribution named dist. Returns a dict."""
	return_dict = {
		'median': np.nanmedian(dist),
		'upper': np.nanpercentile(dist, 100-(100.0-CI)/2.),
		'lower': np.nanpercentile(dist, (100.0-CI)/2.),
		'mean': np.mean(dist),
		'std': np.std(dist)}
	return return_dict

### plot settings
plt.rcParams.update(plt_settings) # from library.py
color = '#1f77b4' # default matplotlib color (that blueish color)
light_color = tint_shade_color(color, amount=1+tintshade)
dark_color = tint_shade_color(color, amount=1-tintshade)
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
	
	# initialize results dict to turn into df
	results = {}
	results['Run'] = filename
	results['Transfer'] = transfer_selection
	results['Pulse Time (us)'] = meta_df['trf'][0]*1e6
	results['Pulse Type'] = meta_df['pulsetype'][0]
	results['ToTF'] = meta_df['ToTF'][0]
	results['EF'] = EF
	results['kF'] = np.sqrt(2*mK*EF*h*1e6)/hbar
	results['barnu'] = meta_df['barnu'][0]
	
	# from Tilman's unitary gas harmonic trap averaging code
	results['C_theory'] = calc_contact(results['ToTF'], results['EF'], 
								results['barnu'])
	# clock shift theory prediction from C_Theory
	results['CS_theory'] = 1/(pi*results['kF']*a13(meta_df['Bfield'][0])) \
							* results['C_theory']
	
	# check if 'vva' is in .dat file
	try:
		run.data['vva']
	except KeyError:
		try:
			run.data['vva'] = run.data['VVA']
		except KeyError:
			raise KeyError("No 'VVA' or 'vva' column in .dat")
	
	# remove indices if requested
	if meta_df['remove_indices'][0] == meta_df['remove_indices'][0]: # nan check
		if type(meta_df['remove_indices'][0]) != int:	
			remove_list = meta_df['remove_indices'][0].strip(' ').split(',')
			remove_indices = [int(index) for index in remove_list]
		run.data.drop(remove_indices, inplace=True)
	
	# length of data set
	num = len(run.data[xname])
	
	#### compute detuning
	run.data['detuning'] = run.data[xname] - meta_df['res_freq'][0]*np.ones(num) # MHz
	
	# fudge the c9 counts using ff
	run.data['c9'] = run.data['c9'] * meta_df['ff'][0]
	
	### compute bg c5, transfer, Rabi freq, etc.
	results['FourierWidth'] = 2/meta_df['trf'][0]/1e6
	if meta_df['bg_freq'][0] == meta_df['bg_freq'][0]: # nan check
		results['bgc5'] = run.data[run.data[xname]==meta_df['bg_freq'][0]]['c5'].mean()
		results['bgc9'] = run.data[run.data[xname]==meta_df['bg_freq'][0]]['c9'].mean()
	else: # no bg point specified, just select past Fourier width
		bg_cutoff = meta_df['res_freq'][0]-2*results['FourierWidth']
		results['bgc5'] = run.data[run.data.detuning < bg_cutoff]['c5'].mean()
		results['bgc9'] = run.data[run.data.detuning < bg_cutoff]['c9'].mean()
		
	run.data['N'] = run.data['c5']-results['bgc5']*np.ones(num)+run.data['c9']
	run.data['transfer'] = (run.data['c5'] - results['bgc5']*np.ones(num))/run.data['N']
	run.data['loss'] = (results['bgc9'] - run.data['c9'])/results['bgc9']
	
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
	results['OmegaR_pk'] = max(run.data.loc[(run.data['detuning']>0) & (run.data['detuning']<0.1)]['OmegaR'])
	OmegaR_max = 2*pi*1*OmegaR_from_VVAfreq(10, 47)
	# here trf was in s so convert to ms, OmegaR is in kHz
	results['pulsejuice'] = results['OmegaR_pk']**2 * (meta_df['trf'][0]*1e3) / OmegaR_max 
	run.data['ScaledTransfer'] = run.data.apply(lambda x: GammaTilde(x[transfer_selection],
									h*EF*1e6, x['OmegaR']*1e3, meta_df['trf'][0]), axis=1)
	run.data['C'] = run.data.apply(lambda x: 2*np.sqrt(2)*pi**2*x['ScaledTransfer'] * \
									   (np.abs(x['detuning'])/EF)**(3/2), axis=1)
		
	results['x_star'] = xstar(meta_df['Bfield'][0], EF)
	
	### now group by freq to get mean and stddev of mean
	run.group_by_mean(xname)
	
	### rough first fit of data
	# create fit function
	fit_func = GenerateSpectraFit(results['x_star'])
	
	# find fit bounds... min is roughly two Fourier widths up
	xfitmin = 2*results['FourierWidth']/EF
	# max is roughly the trap depth if using transfer, else just make it massive
	if transfer_selection == 'transfer':	
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
		print('raw fit C = {:.2f} \pm {:.2f}'.format(pi**2*2**(3/2)*popt[0], 
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
	results['CS_pred'] = 1/(pi*results['kF']*a13(meta_df['Bfield'][0])) * results['C_median']
	results['CS_pred_lower'] = 1/(pi*results['kF']*a13(meta_df['Bfield'][0])) * results['C_upper'] 
	results['CS_pred_upper'] = 1/(pi*results['kF']*a13(meta_df['Bfield'][0])) * results['C_lower']
	
##########################
######## Plotting ########
##########################
	title = f'{filename} at T/TF={meta_df["ToTF"][0]} using {transfer_selection}'
	plt.rcParams.update({"figure.figsize": [12,8]})
	fig, axs = plt.subplots(2,3)
	
	xlabel = r"Detuning $\omega_{rf}-\omega_{res}$ (MHz)"
	label = r"trf={:.0f} us, gain={:.2f}".format(meta_df['trf'][0]*1e6,meta_df['gain'][0])
	
	###
	### plot transfer fraction
	###
	ax = axs[0,0]
	x = run.avg_data['detuning']
	y = run.avg_data['transfer']
	yerr = run.avg_data['em_transfer']
	ylabel = r"Transfer $\Gamma \,t_{rf}$"
	
	xlims = [-0.04,max(x)]
	ylims = [min(run.data['transfer']),max(run.data['transfer'])]
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, fmt='o')
	
	###
	### plot scaled transfer
	###
	ax = axs[0,1]
	x = run.avg_data['detuning']/EF
	y = run.avg_data['ScaledTransfer']
	yerr = run.avg_data['em_ScaledTransfer']
	xlabel = r"Detuning $\Delta$"
	ylabel = r"Scaled Transfer $\tilde\Gamma$"
	
	xlims = [-2,max(x)]
	ylims = [-0.5/20, max(run.data['ScaledTransfer'])]
	xs = np.linspace(xlims[0], xlims[-1], len(y))
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, fmt='o', label=label)
	ax.legend()
	
	###
	### plot zoomed-in scaled transfer
	###
	ax = axs[0,2]
	xlims = [-3,3]
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, fmt='o')
	
	###
	### plot extrapolated spectra
	###
	ax = axs[1,0]
	label = r"$A \frac{\Delta^{-3/2}}{1+\Delta/\Delta^*}$"
	ax.errorbar(x, y, yerr=yerr, fmt='o')
	
	# plot the fit -3/2 power law tail
	xmax = 40/EF # 40 MHz, is ~ 3000 EF
	xxfit = np.linspace(xfitlims[0], xmax, int(1e3))
	yyfit = fit_func(xxfit, *popt)
	ax.plot(xxfit, yyfit, 'r--', label=label)
	
	mask = np.where(x < 2, True, False)
	ax.fill_between(xxfit, yyfit, alpha=0.15, color = 'b')
	ax.fill_between(x[mask], y[mask], alpha=0.15, color='b')
	ax.set(ylabel=ylabel, xlabel=xlabel, yscale='log', xscale='log')
	ax.legend()
	
	###
	### plot contact
	###
	ax = axs[1,1]
	x = run.avg_data['detuning']/EF
	y = run.avg_data['C']
	yerr = run.avg_data['em_C']
	xlabel = r"Detuning $\Delta$"
	ylabel = r"Contact $C/N$ [$k_F$]"
	
	xlims = [-2,max(x)]
	ylims = [-0.1, max(run.data['C'])]
	Cdetmin = 2 
	Cdetmax = 10 # trap depth
	xs = np.linspace(Cdetmin, Cdetmax, num)
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, fmt='o')
	ax.plot(xs, results['C_median']*np.ones(num), "--")

	###
	### generate table
	###
	ax = axs[1,2]
	ax.axis('off')
	quantities = ["Run", "ToTF"]
	values = [filename[:-6],
			   "{:.3f}".format(meta_df['ToTF'][0])
			  ]
	quantities += ["SR median", "C median", "C theory", "C/(2SR)",
				"CS median", "CS predict", "CS theory",
				'Transfer Scale', 'FM Extrap']
	values += [r"{:.3f}".format(results['SR_median']),
			  r"{:.2f}$k_F$".format(results['C_median']),
			  r"{:.2f}$k_F$".format(results['C_theory']),
			  r"{:.2f}$k_F$".format(results['CoSR_median']),
			   r"{:.2f}".format(results['CS_median']),
			   r"{:.2f}".format(results['CS_pred']),
			   r"{:.2f}".format(results['CS_theory']),
			  r"{:.4f}".format(results['pulsejuice']),
			  r"{:.2f}".format(results['FM_extrap_median'])]
	
	table = list(zip(quantities, values))
	
	the_table = ax.table(cellText=table, loc='center')
	the_table.auto_set_font_size(False)
	the_table.set_fontsize(12)
	the_table.scale(1,1.5)
	
	fig.suptitle(title)
	fig.tight_layout()
	
	### save figure
	if Save == True:
		runfolder = filename 
		figpath = os.path.join(figfolder_path, runfolder)
		os.makedirs(figpath, exist_ok=True)
		plots_figname = filename[:-6] + '_plots.pdf'
		fig.savefig(os.path.join(figpath, plots_figname))
	
###########################
####### Histograms ########
###########################

	if BootstrapHists == True:
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
		
	if Correlations == True:
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
		datatosavedf = pd.DataFrame(results, index=[save_df_index])
		save_df_index += 1
		
		try: # to open save file, if it exists
			existing_data = pd.read_excel(savefile, sheet_name='Sheet1')
			if len(datatosavedf.columns) == len(existing_data.columns) \
					and filename in existing_data['Run'].values \
					and transfer_selection in existing_data['Transfer'].values:
				print()
				print(f'{filename} has already been analyized and put into the summary .xlsx file')
				print('and columns of summary data are the same')
				print()
			elif len(datatosavedf.columns) == len(existing_data.columns):
				print('Columns of summary data are the same')
				print("There is saved data, so adding rows to file.")
				start_row = existing_data.shape[0] + 1
			 
			 # open file and write new results
				with pd.ExcelWriter(savefile, mode='a', if_sheet_exists='overlay', \
						engine='openpyxl') as writer:
					datatosavedf.to_excel(writer, index=False, header=False, 
					   sheet_name='Sheet1', startrow=start_row)
			else:
				print()
				print('Columns of summary data are different')  
				print("There is saved data, so adding rows to file.")
				start_row = existing_data.shape[0] + 1
				
				datatosavedf.columns = datatosavedf.columns.to_list()
			 # open file and write new results
				with pd.ExcelWriter(savefile, mode='a', if_sheet_exists='overlay', \
					   engine='openpyxl') as writer:
					datatosavedf.to_excel(writer, index=False, 
					   sheet_name='Sheet1', startrow=start_row)
					
		except PermissionError:
			 print()
			 print ('Is the .xlsx file open?')
			 print()
		except FileNotFoundError: # there is no save file
			 print("Save file does not exist.")
			 print("Creating file and writing header")
			 datatosavedf.to_excel(savefile, index=False, sheet_name='Sheet1')
	 
	 
