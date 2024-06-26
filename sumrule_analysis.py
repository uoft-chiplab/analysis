# -*- coding: utf-8 -*-
"""
Created by Chip lab 2024-06-12

Loads .dat with contact HFT scan and computes scaled transfer. Plots. Also
computes the sumrule.
"""
# %%
from library import pi, h, hbar, mK, a0, plt_settings, GammaTilde, tintshade, \
			MonteCarlo_trapz, MonteCarlo_interp_trapz, tint_shade_color
from data_class import Data
from HFT_data import *  # includes Vpp calibration
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

EF = 16e-3 # MHz
kF = np.sqrt(2*mK*EF*1e6*h)/hbar
Bfield = 202.1 # G
re = 107 * a0
Eb = 3.98 # MHz # I guesstimated this from recent ac dimer spectra

def a13(B):
	''' ac scattering length '''
	abg = 167.6*a0
	DeltaB = 7.2
	B0=224.2
	return abg*(1 - DeltaB/(B-B0))

xstar = Eb/EF * (1-re/a13(Bfield))**(-1)

def FixedPowerLaw(x, A):
	return A*x**(-3/2)

def FullPowerLaw(x, A):
	xmax = xstar
# 	print('xstar = {:.3f}'.format(xmax))
	return A*x**(-3/2) / (1+x/xmax)

def SR_xstar_infty(xstar):
	return 2/np.sqrt(xstar)*(1 - np.arctan(1))

def FM_xstar_infty(xstar):
	return 2*np.sqrt(xstar) * np.arctan(1)

def MonteCarlo_FM_spectra_fit_trapz(xs, ys, yserr, fitmask, xmax, num_iter=100):
	""" Computes trapz for interpolated list of data points (xs, ys+-yserr),
	which is extrapolated using FullPowerLaw out to xmax. Estimates std dev of 
	result by sampling ys and yserr from Gaussian distributions, and fitting
	to this sample, num_iter (default 1000) times."""
	fit_func = FullPowerLaw
	
	def rand_y(y, yerr, size):
		generator = np.random.default_rng()
		return generator.normal(loc=y, scale=yerr, size=num_iter)
	# array of lists of y vals, from Gaussians with centres y and widths yerr
	ys_iter = np.array([rand_y(y, yerr, num_iter) for y, 
					 yerr in zip(ys, yserr)])
	
	popts = []
	pcovs = []
	# fit to determine lineshape for each iteration
	for i in range(num_iter):
		ys_fit = ys_iter[:,i]
		popt, pcov = curve_fit(fit_func, xs[fitmask], ys_fit[fitmask])
		popts.append(popt)
		pcovs.append(pcov)
	
	num = 1000 # points to integrate over
	# interpolation array for x, num in size
	xs_interp = np.linspace(min(xs), max(xs), num)
	# extrapolation array for x, num_iter in size
	xs_extrap = np.linspace(max(xs), xmax, num)
	
	# compute interpolation array for y, num by num_iter in size
	ys_interp_iter = np.array([[np.interp(xi, xs, ys_iter[:,i]) \
							 for xi in xs_interp] for i in range(num_iter)])
	
	# integral from xstar to infty
	SR_xstar_to_infties = np.array(popts)[:,0]*SR_xstar_infty(xmax)
	FM_xstar_to_infties = np.array(popts)[:,0]*FM_xstar_infty(xmax)
	
	# for the integration, we first sum the interpolation, 
	# then the extrapolation, then we add the analytic -5/2s portion
	
	# sumrule using each set
	SR_distr = np.array([np.trapz(ys_interp_iter[i], x=xs_interp) \
			+ np.trapz(fit_func(xs_extrap, *popts[i]), x=xs_extrap) \
				+ SR_xstar_to_infties \
					for i in range(num_iter)])
	# first moment using each set	
	FM_distr = np.array([np.trapz(ys_interp_iter[i]*xs_interp, x=xs_interp) \
			+ np.trapz(fit_func(xs_extrap, *popts[i])*xs_extrap, x=xs_extrap) \
				+ FM_xstar_to_infties \
					for i in range(num_iter)])
	
	# clock shift
	# we need to do this sample by sample so we have correlated SR and FM
	CS_distr = np.array([FM/SR for FM, SR in zip(FM_distr, SR_distr)])
	
	SR_mean, e_SR = (np.mean(SR_distr), np.std(SR_distr))
	FM_mean, e_FM = (np.mean(FM_distr), np.std(FM_distr))
	CS_mean, e_CS = (np.mean(CS_distr), np.std(CS_distr))
	
	# return everything
	return SR_distr, SR_mean, e_SR, FM_distr, FM_mean, e_FM, CS_distr, \
		CS_mean, e_CS, popts, pcovs, SR_xstar_to_infties, FM_xstar_to_infties

### This turns on (True) and off (False) saving the data/plots 
Saveon = True 

### script options
Analysis = True
Summaryplots = True

### folder with relevant data
data_folder = 'SavedSumRule\\data'

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

### select datasets to analyze from HFT_data.py
HFT_datasets = [
				F_20240621,
				D_20240620,
				G_20240618,
				K_20240612
				]

HFT_removed_datasets = [
						C_20240620
						]

### loop analysis over selected datasets
for dataset in HFT_datasets:
	
	if Analysis == False:
		break # exit loop if no analysis required, this just elims one 
			  # indentation for an if statement
	
	# run params from HFT_data.py
	filename = dataset['filename']
	print("----------------")
	print("Analyzing " + filename)
	
	xname = dataset['xname']
	ff = dataset['ff']
	trf = dataset['trf']  # 200 or 400 us
	gain = dataset['gain']
	EF = dataset['EF'] #MHz
	bg_freq = dataset['bg_freq']  # chosen freq for bg, large negative detuning
	res_freq = dataset['res_freq'] # for 202.1G
	pulsetype = dataset['pulsetype']
	remove_indices = dataset['remove_indices']
	
	# determine pulse area
	if pulsetype == 'Blackman':
		pulse_area = lambda x: np.sqrt(0.31)
	elif pulsetype == 'Kaiser':
		pulse_area = lambda x: np.sqrt(0.3*0.92)
	elif pulsetype == 'square':
		pulse_area = lambda x: 1
	elif pulsetype == 'BlackmanOffset':
		pulse_area = lambda x: np.sqrt(0.31) * (x - 0.0252) + (1) * (0.0252)
	else:
		ValueError("pulsetype not a known type")
	
	# create data structure
	run = Data(filename, path=data_folder)
	# remove indices if requested
	run.data.drop(remove_indices, inplace=True)
	num = len(run.data[xname])
	
	### compute bg c5, transfer, Rabi freq, etc.
	bgc5 = run.data[run.data[xname]==bg_freq]['c5'].mean()
	run.data['N'] = run.data['c5']-bgc5*np.ones(num)+run.data['c9']*ff
	run.data['transfer'] = (run.data['c5'] - bgc5*np.ones(num))/run.data['N']
	run.data['detuning'] = run.data[xname] - res_freq*np.ones(num) # MHz
	run.data['Vpp'] = run.data['vva'].apply(VVAtoVpp)
	run.data['OmegaR'] = 2*pi*pulse_area(gain*VpptoOmegaR*run.data['Vpp']) \
								*gain*VpptoOmegaR*run.data['Vpp']
	
	run.data['ScaledTransfer'] = run.data.apply(lambda x: GammaTilde(x['transfer'],
									h*EF*1e6, x['OmegaR']*1e3, trf), axis=1)
	run.data['C'] = run.data.apply(lambda x: 2*np.sqrt(2)*pi**2*x['ScaledTransfer'] * \
									   (np.abs(x['detuning'])/EF)**(3/2), axis=1)
	# run.data = run.data[run.data.detuning != 0]
			
	### now group by freq to get mean and stddev of mean
	run.group_by_mean(xname)
	
	### interpolate scaled transfer for sumrule integration
	xp = np.array(run.avg_data['detuning'])/EF
	fp = np.array(run.avg_data['ScaledTransfer'])
	maxfp = max(fp)
	e_maxfp = run.avg_data.iloc[run.avg_data['ScaledTransfer'].idxmax()]['em_ScaledTransfer']
	TransferInterpFunc = lambda x: np.interp(x, xp, fp)

	### PLOTTING
	fig, axs = plt.subplots(2,3)
	
	xlabel = r"Detuning $\omega_{rf}-\omega_{res}$ (MHz)"
	
	### plot transfer fraction
	ax = axs[0,0]
	x = run.avg_data['detuning']
	y = run.avg_data['transfer']
	yerr = run.avg_data['em_transfer']
	ylabel = r"Transfer $\Gamma \,t_{rf}$"
	
	xlims = [-0.04,max(x)]
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims)
	ax.errorbar(x, y, yerr=yerr, fmt='o')
	
	### plot scaled transfer
	ax = axs[1,0]
	x = run.avg_data['detuning']/EF
	y = run.avg_data['ScaledTransfer']
	yerr = run.avg_data['em_ScaledTransfer']
	xlabel = r"Detuning $\Delta$"
	ylabel = r"Scaled Transfer $\tilde\Gamma$"
	
	xlims = [-2,max(x)]
	axxlims = xlims
	ylims = [min(run.data['ScaledTransfer'])-0.05,
			 max(run.data['ScaledTransfer'])]
	xs = np.linspace(xlims[0], xlims[-1], len(y))
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=axxlims, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, fmt='o')
	
	### fit and plot -3/2 power law tail
	xlowfit = 2
	xhighfit = 10
	fitmask= x.between(xlowfit, xhighfit)
	
	popt, pcov = curve_fit(FullPowerLaw, x[fitmask], y[fitmask], p0=[0.1], 
						sigma=yerr[fitmask])
	print('A = {:.3f} \pm {:.3f}'.format(popt[0], np.sqrt(np.diag(pcov))[0]))
	
	xmax = 1000000/EF
	xxfit = np.linspace(xlowfit, xmax, int(xmax*EF*10))
	yyfit = FullPowerLaw(xxfit, *popt)
	ax.plot(xxfit, yyfit, 'r--')
	
	### calulate integrals
	
	# sumrule
	SR_raw = np.trapz(y, x=x)
	print("raw data sumrule = {:.3f}".format(SR_raw))
	
	SR_raw_interp = np.trapz(TransferInterpFunc(xs), x=xs)
	print("interpolated sumrule = {:.3f}".format(SR_raw_interp))
	
	# first moment
	FM_raw = np.trapz(y*x, x=x)
	print("raw data first moment = {:.3f}".format(FM_raw))
	
	FM_raw_interp = np.trapz(TransferInterpFunc(xs)*xs, x=xs)
	print("interpolated first moment = {:.3f}".format(FM_raw_interp))
	
	# clock shift
	CS_raw = FM_raw/SR_raw
	
	### Monte-Carlo sampling for integral uncertainty
	num_iter = 1000
	
	# sumrule
# 	SR_dist, SR_mean, SR_std = trapz_w_error(x, y, yerr, num_iter)
	SR_dist, SR, e_SR = MonteCarlo_trapz(x, y, yerr, num_iter)
# 	print(r"sumrule mean = {:.3f}$\pm$ {:.3f}".format(SR_mean, SR_std))
	print(r"sumrule interp mean = {:.3f}$\pm$ {:.3f}".format(SR, e_SR))
	
	# first moment from data
# 	FM_dist, FM_mean, FM_std = trapz_w_error(x, y, yerr, num_iter)
	FM_dist, FM, e_FM = MonteCarlo_interp_trapz(x, y*x, np.abs(yerr*x), num_iter)
# 	print(r"sumrule mean = {:.3f}$\pm$ {:.3f}".format(FM_mean, FM_std))
	print(r"FM interp mean = {:.3f}$\pm$ {:.3f}".format(FM, e_FM))
	
	# clock shift
	CS = FM/SR
	
	# sumrule, first moment and clockshift with analytic extension
	FM_full_dist, FM_full, e_FM_full, SR_full_dist, SR_full, e_SR_full, \
		CS_full_dist, CS_full, e_CS_full, popts, pcovs, SR_xstar_to_infties, \
		FM_xstar_to_infties	= MonteCarlo_FM_spectra_fit_trapz(x, y, yerr, fitmask, xstar)
		
	
	### plot contact
	ax = axs[0,1]
	x = run.avg_data['detuning']/EF
	y = run.avg_data['C']
	yerr = run.avg_data['em_C']
	xlabel = r"Detuning $\Delta$"
	ylabel = r"Contact $C/N$ [$k_F$]"
	
	xlims = [-2,max(x)]
	ylims = [min(run.data['C']), max(run.data['C'])]
	Cdetmin = 3
	Cdetmax = 8
	xs = np.linspace(Cdetmin, Cdetmax, num)
	
	df = run.data[run.data.detuning/EF>Cdetmin]
	Cmean = df[df.detuning/EF<Cdetmax].C.mean()
	Csem = df[df.detuning/EF<Cdetmax].C.sem()
	
	# note this chooses the mean of the interpolation distribution as the sumrule
	C_o_SR = Cmean/(2*SR)
	e_C_o_SR = C_o_SR*np.sqrt((Csem/Cmean)**2+\
							(e_SR/SR)**2)
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, fmt='o')
	ax.plot(xs, Cmean*np.ones(num), "--")
	
	# Clock Shift from contact
	CS_pred = 1/(pi*kF*a13(Bfield)) * Cmean
	e_CS_pred = CS_pred*Csem/Cmean
	
	### plot x*Scaled transfer
	ax = axs[1,1]
	x = run.avg_data['detuning']/EF
	y = run.avg_data['ScaledTransfer'] * x
	yerr = run.avg_data['em_ScaledTransfer'] * x
	xlabel = r"Detuning $\Delta$"
	ylabel = r"$\Delta \tilde\Gamma$"
	
	xlims = [-2,max(x)]
	axxlims = xlims
	ylims = [min(run.data['ScaledTransfer']*run.data['detuning']/EF),
			 max(run.data['ScaledTransfer']*run.data['detuning']/EF)]
	xs = np.linspace(xlims[0], xlims[-1], len(y))
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=axxlims, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, fmt='o')
	xxfit = np.linspace(xlowfit, xmax, int(xmax*EF*10))
	yyfit = FullPowerLaw(xxfit, *popt)
	ax.plot(xxfit, xxfit*yyfit, 'r--')
	
	### generate table
	ax = axs[1,2]
	ax.axis('off')
	ax.axis('tight')
	quantities = ["Run","$E_F$", r"$t_{rf}$", "Gain", "Contact $C/N$", 
			    "SR raw", "SR MC interp", "C/SR"]
	values = [filename[:-6],
		   "{:.1f} kHz".format(EF*1e3), 
			  r"{:.0f} $\mu$ s".format(trf*1e6), 
			  "{:.2f}".format(gain),
			  "{:.2f}$\pm${:.2f} $k_F$".format(Cmean, Csem),
			  "{:.3f}".format(SR_raw),
			  r"{:.3f}$\pm${:.3f}".format(SR, e_SR),
			  r"{:.3f}$\pm${:.3f}".format(C_o_SR, e_C_o_SR)
			  ]
	table = list(zip(quantities, values))
	
	the_table = ax.table(cellText=table, loc='center')
	the_table.auto_set_font_size(False)
	the_table.set_fontsize(12)
	the_table.scale(1,1.5)
	
	fig.tight_layout()
	plt.show()
	
	if Saveon == True:
	 	datatosave = {'SR interp': [SR_raw_interp], 
				  'SR raw': [SR_raw],
				  'SR MC': [SR],
				  'e_SR MC': [e_SR],
				  'SR MC full': [SR_full],
				  'e_SR MC full': [e_SR_full],
				  'FM interp': [FM_raw_interp], 
				  'FM raw': [FM_raw],
				  'FM MC': [FM],
				  'e_FM MC': [e_FM],
				  'FM MC full': [FM_full],
				  'e_FM MC full': [e_FM_full],
				  'CS MC full': [FM_full],
				  'e_CS MC full': [e_FM_full],
				  'CS pred': [CS_pred],
				  'e_CS pred': [e_CS_pred],
	 			  'Gain':[gain], 
				   'Run':[filename], 
	 			  'C':[Cmean],
				   'e_C':[Csem],
	 			  'C/SR':[C_o_SR],
				   'e_C/SR':[e_C_o_SR],
	 			  'Peak Scaled Transfer':[maxfp], 
				  'e_Peak Scaled Transfer':[e_maxfp],
				   'Pulse Time (us)':[trf*1e6],
	 			  'Pulse Area':[pulse_area], 
				   'Pulse Type':[pulsetype]}
	 	datatosavedf = pd.DataFrame(datatosave)

	 	datatosave_folder = 'SavedSumRule'
	 	runfolder = filename 
	 	figpath = os.path.join(datatosave_folder,runfolder)
	 	os.makedirs(figpath, exist_ok=True)
	
	 	sumrulefig_name = 'SumRule.png'
	 	sumrulefig_path = os.path.join(figpath,sumrulefig_name)
	 	fig.savefig(sumrulefig_path)
	 	
	 	xlsxsavedfile = 'Saved_Sum_Rules.xlsx'
	 	
	 	filepath = os.path.join(datatosave_folder,xlsxsavedfile)
		 
	 	try: # to open save file, if it exists
			 existing_data = pd.read_excel(filepath, sheet_name='Sheet1')
			 print("There is saved data, so adding rows to file.")
			 start_row = existing_data.shape[0] + 1
			 
			 # open file and write new results
			 with pd.ExcelWriter(filepath, mode='a', if_sheet_exists='overlay', \
					   engine='openpyxl') as writer:
				  datatosavedf.to_excel(writer, index=False, header=False, 
					   sheet_name='Sheet1', startrow=start_row)
				  
	 	except FileNotFoundError: # there is no save file
			 print("Save file does not exist.")
			 print("Creating file and writing header")
			 datatosavedf.to_excel(filepath, index=False, sheet_name='Sheet1')

if Summaryplots == True:
	
	# load analysis results
	df = pd.read_excel(filepath, index_col=0, engine='openpyxl').reset_index()
	
	### plots
	fig, axes = plt.subplots(2,2)

	xlabel = r"Gain"
	
	# C vs. gain
	ax = axes[0,0]
	ax.errorbar(df['Gain'], df['C'], yerr=df['e_C'], fmt='o')
	ylabel = r"Contact $C/N$ [$k_F$]"
	ax.set(xlabel=xlabel, ylabel=ylabel)
	
	# sumrule vs gain
	ax = axes[0,1]
	ax.errorbar(df['Gain'], df['SR MC'], 
			 yerr=df['e_SR MC'], fmt='o')
	ylabel = "Sumrule"
	ax.set(xlabel=xlabel, ylabel=ylabel)
	
	# C/sumrule vs gain
	ax = axes[1,0]
	ax.errorbar(df['Gain'], df['C/SR'], 
			 yerr=df['e_C/SR'], fmt='o')
	ylabel = "C/SR"
	ax.set(xlabel=xlabel, ylabel=ylabel)
	
	# peak scaled transfer vs gain
	ax = axes[1,1]
	ax.errorbar(df['Gain'], df['Peak Scaled Transfer'], 
			 yerr=df['e_Peak Scaled Transfer'], fmt='o')
	ylabel = "Peak Scaled Transfer"
	ax.set(xlabel=xlabel, ylabel=ylabel)
	
	fig.tight_layout()
	plt.show()
# %%
