"""
Created by Chip lab 2024-06-12

Loads .dat with contact HFT scan and computes scaled transfer. Plots. Also
computes the sumrule.

To do:
	Calculate EF for each shot
	Check pulse area calculations
	More commenting
	Filter results when summary plotting
	Perhaps remove the MonteCarlo calculations, I think Bootstrapping is just
better
	....
	
"""

BOOTSRAP_TRAIL_NUM = 1000

# paths
import os
# print(os.getcwd())
proj_path = os.path.dirname(os.path.realpath(__file__))
# print(proj_path)
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')
figfolder_path = os.path.join(proj_path, 'figures')

# import imp 
# library = imp.load_source('library',os.path.join(root,'library.py'))
# data_class = imp.load_source('data_class',os.path.join(root,'data_class.py'))

from library import pi, h, hbar, mK, a0, plt_settings, GammaTilde, tintshade, \
				 tint_shade_color, ChipKaiser, ChipBlackman, markers, colors
from data_class import Data
from scipy.optimize import curve_fit
from scipy.stats import sem
from clockshift.MonteCarloSpectraIntegration import MonteCarlo_spectra_fit_trapz, \
												Bootstrap_spectra_fit_trapz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import time

### This turns on (True) and off (False) saving the data/plots 
Saveon = False

### script options
Analysis = True
Summaryplots = True
MonteCarlo = False
Bootstrap = True
Bootstrapplots = False
Correlations = True

### metadata
metadata_filename = 'metadata_file.xlsx'
metadata_file = os.path.join(proj_path, metadata_filename)
metadata = pd.read_excel(metadata_file)
files =  metadata.loc[metadata['exclude'] == 0]['filename'].values

# Manual file select, comment out if exclude column should be used instead
# files = ["2024-07-18_E_e"]


# save file path
savefilename = 'sumrule_analysis_results.xlsx'
savefile = os.path.join(proj_path, savefilename)

### Vpp calibration
VVAtoVppfile = os.path.join(root,"VVAtoVpp.txt") # calibration file
VVAs, Vpps = np.loadtxt(VVAtoVppfile, unpack=True)
VpptoOmegaR = 27.5833 # kHz

def VVAtoVpp(VVA):
	"""Match VVAs to calibration file values. Will get mad if the VVA is not
		also the file. """
	Vpp = 0
	for i, VVA_val in enumerate(VVAs):
		if VVA == VVA_val:
			Vpp = Vpps[i]
	if Vpp == 0: # throw a fit if VVA not in list.
		raise ValueError("VVA value {} not in VVAtoVpp.txt".format(VVA))
	return Vpp

### contants
EF = 16e-3 # MHz
kF = np.sqrt(2*mK*EF*1e6*h)/hbar
re = 107 * a0 # ac dimer range estimate
Eb = 3.98 # MHz # I guesstimated this from recent ac dimer spectra

def a13(B):
	''' ac scattering length '''
	abg = 167.6*a0
	DeltaB = 7.2
	B0=224.2
	return abg*(1 - DeltaB/(B-B0))

def xstar(B):
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
for filename in files:
	
	if Analysis == False:
		break # exit loop if no analysis required, this just elims one 
			  # indentation for an if statement
	
	# run params from HFT_data.py
	print("----------------")
	print("Analyzing " + filename)
	
	df = metadata.loc[metadata.filename == filename].reset_index()
	if df.empty:
		print("Dataframe is empty! The metadata likely needs updating." )
		continue
	
	xname = df['xname'][0]
		
	ff = df['ff'][0]
	trf = df['trf'][0]  # 200 or 400 us
	gain = df['gain'][0]
	EF = df['EF'][0] #MHz
	bg_freq = df['bg_freq'][0]  # chosen freq for bg, large negative detuning
	Bfield = df['Bfield'][0]
	res_freq = df['res_freq'][0] # for 202.1G
	pulsetype = df['pulsetype'][0]
	remove_indices = df['remove_indices'][0]
	
	# create data structure
	filename = filename + ".dat"
	run = Data(filename, path=data_path)
	
	# remove indices if requested
	if remove_indices == remove_indices: # nan check
		if type(remove_indices) != int:	
			remove_list = remove_indices.strip(' ').split(',')
			remove_indices = [int(index) for index in remove_list]
		run.data.drop(remove_indices, inplace=True)
	
	num = len(run.data[xname])
	
	#### compute detuning
	run.data['detuning'] = run.data[xname] - res_freq*np.ones(num) # MHz
	zeroptsindices = np.where(np.array(run.data['detuning']) == 0)[0]
# 	run.data['detuning'] = run.data['detuning'].drop(index=zeroptsindices)
	
	### compute bg c5, transfer, Rabi freq, etc.
	if bg_freq == bg_freq: # nan check
		bgc5 = run.data[run.data[xname]==bg_freq]['c5'].mean()
	else: # no bg point specified, just select past Fourier width
		FourierWidth = 2/trf/1e6
		bg_cutoff = res_freq-FourierWidth
		bgc5 = run.data[run.data.detuning < bg_cutoff]['c5'].mean()
		
	run.data['N'] = run.data['c5']-bgc5*np.ones(num)+run.data['c9']*ff
	run.data['transfer'] = (run.data['c5'] - bgc5*np.ones(num))/run.data['N']
	
	# map VVA to Vpp
	if pulsetype == 'KaiserOffset' or 'BlackmanOffset':
		# split pulse into offset and amplitude
		# scale offset by VVA
		run.data['offset'] = 0.0252/VVAtoVpp(10)* \
								(run.data['vva'].apply(VVAtoVpp) )
		# calculate amplitude ignoring offset
		# gain only effects this amplitude, not the offset
		run.data['Vpp'] = gain * (run.data['vva'].apply(VVAtoVpp) 
					 - run.data['offset'])
	else:
		run.data['Vpp'] = run.data['vva'].apply(VVAtoVpp)
	
	# determine pulse area
	if pulsetype == 'Blackman':
		run.data['sqrt_pulse_area'] = np.sqrt(0.31) 
	elif pulsetype == 'Kaiser':
		run.data['sqrt_pulse_area'] = np.sqrt(0.3*0.92)
	elif pulsetype == 'square':
		run.data['sqrt_pulse_area'] = 1
	elif pulsetype == 'BlackmanOffset':
		xx = np.linspace(0,1,1000)
		# integrate the square, and sqrt it
		run.data['sqrt_pulse_area'] = np.sqrt(run.data.apply(lambda x: 
		   np.trapz((ChipBlackman(xx)*(1-x['offset']/x['Vpp']) \
			   + x['offset']/x['Vpp'])**2, x=xx), axis=1))
	elif pulsetype == 'KaiserOffset':
		xx = np.linspace(0,1,1000)
		# integrate the square, and sqrt it
		run.data['sqrt_pulse_area'] = np.sqrt(run.data.apply(lambda x: 
		   np.trapz((ChipKaiser(xx)*(1-x['offset']/x['Vpp']) \
			   + x['offset']/x['Vpp'])**2, x=xx), axis=1))
	else:
		ValueError("pulsetype not a known type")

	# compute Rabi frequency, scaled transfer, and contact
	run.data['OmegaR'] = 2*pi*run.data['sqrt_pulse_area'] \
							* VpptoOmegaR * run.data['Vpp']
	OmegaR_pk = max(run.data['OmegaR'])
	OmegaR_max = 2*pi*1*VpptoOmegaR*10
	# here trf was in s so convert to ms, OmegaR is in kHz
	pulsejuice = OmegaR_pk**2 * (trf*1e3) / OmegaR_max 
	run.data['ScaledTransfer'] = run.data.apply(lambda x: GammaTilde(x['transfer'],
									h*EF*1e6, x['OmegaR']*1e3, trf), axis=1)
	run.data['C'] = run.data.apply(lambda x: 2*np.sqrt(2)*pi**2*x['ScaledTransfer'] * \
									   (np.abs(x['detuning'])/EF)**(3/2), axis=1)
	# run.data = run.data[run.data.detuning != 0]
	# need a filter like this? Transfer should be nonnegative outside of noise floor
	run.data = run.data[run.data.ScaledTransfer > -0.1] 
	# temp
	### now group by freq to get mean and stddev of mean
	run.group_by_mean(xname)
	
	### interpolate scaled transfer for sumrule integration
	xp = np.array(run.avg_data['detuning'])/EF
	fp = np.array(run.avg_data['ScaledTransfer'])
	maxfp = max(fp)
	e_maxfp = run.avg_data.iloc[run.avg_data['ScaledTransfer'].idxmax()]['em_ScaledTransfer']
	TransferInterpFunc = lambda x: np.interp(x, xp, fp)
	
	
	### ANALYSIS and PLOTTING
	plt.rcParams.update({"figure.figsize": [12,8]})
	fig, axs = plt.subplots(2,3)
	
	xlabel = r"Detuning $\omega_{rf}-\omega_{res}$ (MHz)"
	label = r"trf={:.0f} us, gain={:.2f}".format(trf*1e6,gain)
	
	### plot transfer fraction
	ax = axs[0,0]
	x = run.avg_data['detuning']
	y = run.avg_data['transfer']
	yerr = run.avg_data['em_transfer']
	ylabel = r"Transfer $\Gamma \,t_{rf}$"
	
	xlims = [-0.04,max(x)]
	ylims = [min(run.data['transfer']),max(run.data['transfer'])]
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims, ylim=ylims)
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
	ylims = [min(run.data['ScaledTransfer']),
			 max(run.data['ScaledTransfer'])]
	xs = np.linspace(xlims[0], xlims[-1], len(y))
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=axxlims, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, fmt='o', label=label)
	ax.legend()
	
	### fit and plot -3/2 power law tail
	xfitlims = [2, 10]
	fitmask = x.between(*xfitlims)
	
	x_star = xstar(Bfield)
	
	fit_func = GenerateSpectraFit(x_star)
	
	popt, pcov = curve_fit(fit_func, x[fitmask], y[fitmask], p0=[0.1], 
						sigma=yerr[fitmask])
	print('A = {:.3f} \pm {:.3f}'.format(popt[0], np.sqrt(np.diag(pcov))[0]))
	
	xmax = 1000000/EF
	xxfit = np.linspace(xfitlims[0], xmax, int(xmax*EF*10))
	yyfit = fit_func(xxfit, *popt)
	ax.plot(xxfit, yyfit, 'r--')
	
	### plot zoomed-in scaled transfer
	ax = axs[1,1]
	
	xlims = [-1,1]
	axxlims = xlims
	ylims = [(min(run.data['ScaledTransfer']))/4,
			 max(run.data['ScaledTransfer'])]
	xs = np.linspace(xlims[0], xlims[-1], len(y))
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=axxlims, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, fmt='o', label=label)
	ax.legend()
	
	### plot extrapolated spectra with -5/2 power law
	ax = axs[0,2]
	label = r"$A \frac{\Delta^{-3/2}}{1+\Delta/\Delta^*}$"
	ax.errorbar(x, y, yerr=yerr, fmt='o')
	ax.plot(xxfit, yyfit, 'r--', label=label)
	# avoid shading overlap by making new list
	mask = np.where(x < 2, True, False)
	ax.fill_between(xxfit, yyfit, alpha=0.15, color = 'b')
	ax.fill_between(x[mask], y[mask], alpha=0.15, color='b')
	ax.set(yscale='log', xscale='log', ylabel=ylabel, xlabel=xlabel)
	ax.legend()
	
	### calulate integrals
	# sumrule
	SR_interp = np.trapz(TransferInterpFunc(xs), x=xs)
	SR_extrap = dwSpectraFit(xlims[-1], x_star, *popt)
	
	# first moment
	FM_interp = np.trapz(TransferInterpFunc(xs)*xs, x=xs)
	FM_extrap = wdwSpectraFit(xlims[-1], x_star, *popt)
	
	SR = SR_interp + SR_extrap
	FM = FM_interp + FM_extrap
	
	# clock shift
	CS = FM/SR
	print("raw SR {:.3f}".format(SR))
	print("raw FM {:.3f}".format(FM))
	print("raw CS {:.2f}".format(CS))
	
	### Monte-Carlo sampling for integral uncertainty
	if MonteCarlo == True:
		num_iter = 1000
		
		# sumrule, first moment and clockshift with analytic extension
		SR_MC_dist, SR_MC, e_SR_MC, FM_MC_dist, FM_MC, e_FM_MC, \
		CS_MC_dist, CS_MC, e_CS_MC, popts, pcovs \
			= MonteCarlo_spectra_fit_trapz(x, y, yerr, fitmask, x_star, 
									 fit_func)
			
		print(r"SR MC mean = {:.3f}$\pm$ {:.3f}".format(SR_MC, e_SR_MC))
		print(r"FM MC mean = {:.3f}$\pm$ {:.3f}".format(FM_MC, e_FM_MC))
		print(r"CS MC mean = {:.2f}$\pm$ {:.2f}".format(CS_MC, e_CS_MC))
		
	### Bootstrap resampling for integral uncertainty
	if Bootstrap == True:
		num_iter = 1000
		conf = 68.2689  # confidence level for CI
		
		# non-averaged data
		x = np.array(run.data['detuning']/EF)
# 		print(x)
		y = np.array(run.data['ScaledTransfer'])
		
		# sumrule, first moment and clockshift with analytic extension
		SR_BS_dist, FM_BS_dist, CS_BS_dist, pFits, SR_extrap_dist, FM_extrap_dist = \
			Bootstrap_spectra_fit_trapz(x, y, xfitlims, x_star, fit_func, trialsB=BOOTSRAP_TRAIL_NUM)
		
		SR_BS_mean, e_SR_BS = (np.mean(SR_BS_dist), np.std(SR_BS_dist))
		FM_BS_mean, e_FM_BS = (np.mean(FM_BS_dist), np.std(FM_BS_dist))
		CS_BS_mean, e_CS_BS = (np.mean(CS_BS_dist), np.std(CS_BS_dist))
		SR_extrap_mean, e_SR_extrap = (np.mean(SR_extrap_dist), np.std(SR_extrap_dist))
		FM_extrap_mean, e_FM_extrap = (np.mean(FM_extrap_dist), np.std(FM_extrap_dist))
		CS_BS_mean, e_CS_BS = (np.mean(CS_BS_dist), sem(CS_BS_dist))
		print(r"SR BS mean = {:.3f}$\pm$ {:.3f}".format(SR_BS_mean, e_SR_BS))
		print(r"FM BS mean = {:.3f}$\pm$ {:.3f}".format(FM_BS_mean, e_FM_BS))
		print(r"CS BS mean = {:.2f}$\pm$ {:.2f}".format(CS_BS_mean, e_CS_BS))
		print(r'Extrapolation for SR = {:.4f}$\pm$ {:.4f}'.format(SR_extrap_mean, e_SR_extrap))
		print(r'Extrapolation for FM = {:.2f}$\pm$ {:.2f}'.format(FM_extrap_mean, e_FM_extrap))
		median_SR = np.nanmedian(SR_BS_dist)
		upper_SR = np.nanpercentile(SR_BS_dist, 100-(100.0-conf)/2.)
		lower_SR = np.nanpercentile(SR_BS_dist, (100.0-conf)/2.)
		
		median_FM = np.nanmedian(FM_BS_dist)
		upper_FM = np.nanpercentile(FM_BS_dist, 100-(100.0-conf)/2.)
		lower_FM = np.nanpercentile(FM_BS_dist, (100.0-conf)/2.)
		
		median_CS = np.nanmedian(CS_BS_dist)
		upper_CS = np.nanpercentile(CS_BS_dist, 100-(100.0-conf)/2.)
		lower_CS = np.nanpercentile(CS_BS_dist, (100.0-conf)/2.)
		
		Cdist = pFits*2*np.sqrt(2)*np.pi**2
		median_C = np.nanmedian(Cdist)
		upper_C = np.nanpercentile(Cdist, 100-(100.0-conf)/2.)
		lower_C = np.nanpercentile(Cdist, (100.0-conf)/2.)
		
		CoSR = Cdist / ( 2  *SR_BS_dist) 
		median_CoSR = np.nanmedian(CoSR)
		upper_CoSR = np.nanpercentile(CoSR, 100-(100.0-conf)/2.)
		lower_CoSR = np.nanpercentile(CoSR, (100.0-conf)/2.)
		
		print(r"SR BS median = {:.3f}+{:.3f}-{:.3f}".format(median_SR,
													  upper_SR-median_SR, median_SR-lower_SR))
		print(r"FM BS median = {:.3f}+{:.3f}-{:.3f}".format(median_FM, 
													  upper_FM-median_FM, median_FM-lower_FM))
		print(r"CS BS median = {:.2f}+{:.3f}-{:.3f}".format(median_CS, 
													  upper_CS-median_CS, median_CS-lower_CS))
# 		print(median_CoSR)
	
	### plot contact
	ax = axs[0,1]
	x = run.avg_data['detuning']/EF
	y = run.avg_data['C']
	yerr = run.avg_data['em_C']
	xlabel = r"Detuning $\Delta$"
	ylabel = r"Contact $C/N$ [$k_F$]"
	
	xlims = [-2,max(x)]
	ylims = [-0.1, max(run.data['C'])]
	Cdetmin = 2
	Cdetmax = 10
	xs = np.linspace(Cdetmin, Cdetmax, num)
	
	df = run.data[run.data.detuning/EF>Cdetmin]
	Cmean = df[df.detuning/EF<Cdetmax].C.mean()
	Csem = df[df.detuning/EF<Cdetmax].C.sem()
	
	
	# choose sumrule for Contact normalizing as MC, BS or raw
# 	if MonteCarlo:	
# 		C_o_SR = Cmean/(2*SR_MC)
# 		e_C_o_SR = C_o_SR*np.sqrt((Csem/Cmean)**2+(e_SR_MC/SR_MC)**2)
# 	elif Bootstrap: 
# 		C_o_SR = Cmean/(2*SR_BS_mean)
# 		e_C_o_SR = C_o_SR*np.sqrt((Csem/Cmean)**2+(e_SR_BS/SR_BS_mean)**2)
# 	else:
# 		C_o_SR = Cmean/(2*SR)
# 		e_C_o_SR = C_o_SR*Csem/Cmean
	
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims, ylim=ylims)
	ax.errorbar(x, y, yerr=yerr, fmt='o')
	ax.plot(xs, Cmean*np.ones(num), "--")
	
	# Clock Shift from contact
	CS_pred = 1/(pi*kF*a13(Bfield)) * median_C 
	e_CS_low = 1/(pi*kF*a13(Bfield)) * lower_C 
	e_CS_upper = 1/(pi*kF*a13(Bfield)) * upper_C
	 
	
	### plot x*Scaled transfer
# 	ax = axs[1,1]
# 	x = run.avg_data['detuning']/EF
# 	y = run.avg_data['ScaledTransfer'] * x
# 	yerr = np.abs(run.avg_data['em_ScaledTransfer'] * x)
# 	xlabel = r"Detuning $\Delta$"
# 	ylabel = r"$\Delta \tilde\Gamma$"
# 	
# 	xlims = [-2,max(x)]
# 	axxlims = xlims
# 	ylims = [min(run.data['ScaledTransfer']*run.data['detuning']/EF),
# 			 max(run.data['ScaledTransfer']*run.data['detuning']/EF)]
# 	xs = np.linspace(xlims[0], xlims[-1], len(y))
# 	
# 	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=axxlims, ylim=ylims)
# 	ax.errorbar(x, y, yerr=yerr, fmt='o')
# 	xxfit = np.linspace(xfitlims[0], xmax, int(xmax*EF*10))
# 	yyfit = fit_func(xxfit, *popt)
# 	ax.plot(xxfit, xxfit*yyfit, 'r--')

	### generate table
	ax = axs[1,2]
	ax.axis('off')
# 	ax.axis('tight')
	quantities = ["Run", "SR", "FM", "CS"]
	values = [filename[:-6],
			  "{:.3f}".format(SR),
			  "{:.3f}".format(FM),
			  "{:.2f}".format(CS)
			  ]
	if MonteCarlo == True:
		quantities += ["SR MC", "FM MC", "CS MC"]
		MC_values = [r"{:.3f}$\pm${:.3f}".format(SR_MC, e_SR_MC),
				   r"{:.3f}$\pm${:.3f}".format(FM_MC, e_FM_MC),
				   r"{:.2f}$\pm${:.2f}".format(CS_MC, e_CS_MC)]
		values = values + MC_values
		
	if Bootstrap == True:
		quantities += ["SR BS", "FM BS", "CS BS", 'Transfer Scale', 'FM Extrap', "Contact $C/N$"]
		BS_values = [r"{:.3f}+{:.1f}-{:.1f}".format(SR_BS_mean, median_SR - lower_SR, upper_SR - median_SR),
				   r"{:.3f}+{:.1f}-{:.1f}".format(FM_BS_mean, median_FM - lower_FM, upper_FM - median_FM),
				   r"{:.2f}+{:.1f}-{:.1f}".format(CS_BS_mean, median_CS - lower_CS, upper_CS - median_CS),
				  r"{:.4f}".format(pulsejuice),
				  r"{:.2f}$\pm${:.2f}".format(FM_extrap_mean, e_FM_extrap),
	 "{:.2f}+{:.1f}-{:.1f} $k_F$".format(median_C, median_C - lower_C, upper_C - median_C ),
]
		values = values + BS_values
		
	table = list(zip(quantities, values))
	
	the_table = ax.table(cellText=table, loc='center')
	the_table.auto_set_font_size(False)
	the_table.set_fontsize(12)
	the_table.scale(1,1.5)
	
	fig.tight_layout()
	plt.show()
	
	if Saveon == True:
		datatosave = {
				   'Run':[filename], 
	 			  'Gain':[gain], 
				   'Transfer Scale':[pulsejuice],
				   'Pulse Time (us)':[trf*1e6],
				   'Pulse Type':[pulsetype],
				   'EF':[EF],
				   'SR': [SR],
				  'FM': [FM],
				  'CS':[CS],
	 			  'C':[Cmean],
				   'e_C':[Csem],
	 			  'C/SR':[median_CoSR],
				  'CS pred': [CS_pred],
				  'lower_CS pred': [lower_CS],
				  'upper_CS pred': [upper_CS],
	 			  'Peak Scaled Transfer':[maxfp], 
				  'e_Peak Scaled Transfer':[e_maxfp]}
		 
		if MonteCarlo == True:
			datatosavePlusMC = {
				  'SR MC': [SR_MC],
				  'e_SR MC': [e_SR_MC],
				  'FM MC': [FM_MC],
				  'e_FM MC': [e_FM_MC],
				  'CS MC': [CS_MC],
				  'e_CS MC': [e_CS_MC]}
			
			datatosave.update(datatosavePlusMC)
			
		if Bootstrap == True:
			datatosavePlusBS = {
					'C/SR': [median_CoSR],
				   'lower_C/SR':[lower_CoSR-median_CoSR],
				   'upper_C/SR':[median_CoSR-upper_CoSR],
				  'SR BS mean': [SR_BS_mean],
				  'e_SR BS': [e_SR_BS],
				  'FM BS mean': [FM_BS_mean],
				  'e_FM BS': [e_FM_BS],
				  'CS BS mean': [CS_BS_mean],
				  'e_CS BS': [e_CS_BS],
				   'SR BS median': [median_SR],
				  'SR m conf': [median_SR-lower_SR],
				  'SR p conf': [upper_SR-median_SR],
				  'FM BS median': [median_FM],
				  'FM m conf': [median_FM-lower_FM],
				  'FM p conf': [upper_FM-median_FM],
				  'CS BS median': [median_CS],
				  'CS m conf': [median_CS-lower_CS],
				  'CS p conf': [upper_CS-median_CS],
				  'SR extrapolation':[SR_extrap_mean],
				  'FM extrapolation':[FM_extrap_mean],
				  'e_SR extrapolation':[e_SR_extrap],
				  'e_FM extrapolation':[e_FM_extrap]
# 				  'test':'test'
				  }
			
			datatosave.update(datatosavePlusBS)
		 
		datatosavedf = pd.DataFrame(datatosave)

		### save figure
		runfolder = filename 
		figpath = os.path.join(figfolder_path, runfolder)
		os.makedirs(figpath, exist_ok=True)
	
		sumrulefig_name = 'Analysis_Results.png'
		sumrulefig_path = os.path.join(figpath, sumrulefig_name)
		fig.savefig(sumrulefig_path)
		
		try: # to open save file, if it exists
			existing_data = pd.read_excel(savefile, sheet_name='Sheet1')
			if len(datatosave) == len(existing_data.columns) and filename in existing_data['Run'].values:
				print()
				print(f'{filename} has already been analyized and put into the summary .xlsx file')
				print('and columns of summary data are the same')
				print()
			elif len(datatosave) == len(existing_data.columns):
				print('Columns of summary data are the same')
# 				if filename in existing_data['Run'].values:
#  					print(f'{filename} has already been analyized and put into the summary .xlsx file')
#  					print()
# 				else:
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

	if (Bootstrapplots == True and Bootstrap == True):
		plt.rcParams.update({"figure.figsize": [10,8]})
		fig, axs = plt.subplots(2,3)
		fig.suptitle(filename)
		
		bins = 20
		
		# fits
		ax = axs[0,0]
		## contact over SR distribution
		xlabel = "Contact over Sum Rule"
		ylabel = "Occurances"
		ax.set(xlabel=xlabel, ylabel=ylabel)
		ax.hist(Cmean/(np.array(SR_BS_dist)*2), bins=bins)
		ax.axvline(x=Cmean/(lower_SR*2), color='red', alpha=0.5, linestyle='--', marker='')
		ax.axvline(x=Cmean/(upper_SR*2), color='red', alpha=0.5, linestyle='--', marker='')
		ax.axvline(x=Cmean/(median_SR*2), color='red', linestyle='--', marker='')
		ax.axvline(x=Cmean/(SR_BS_mean*2), color='k', linestyle='--', marker='')
		
		## scaled transfer with fit
# 		x = run.avg_data['detuning']/EF
# 		y = run.avg_data['ScaledTransfer']
# 		yerr = run.avg_data['em_ScaledTransfer']
# 		xlabel = r"Detuning $\Delta$"
# 		ylabel = r"Scaled Transfer $\tilde\Gamma$"
# 		
# 		xdata = run.data['detuning']/EF
# 		datamask = xdata.between(*xfitlims)

# 		ylims = [min(run.data.ScaledTransfer[datamask]),
# 				 max(run.data.ScaledTransfer[datamask])]
# 		
# 		plotmask = x.between(*xfitlims)
# 		xs = np.linspace(xlims[0], xlims[-1], len(y))
# 		
# 		ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xfitlims, ylim=ylims)
# 		ax.plot(xs, fit_func(xs, *popt), '--r')
# 		ax.errorbar(x[plotmask], y[plotmask], yerr=yerr[plotmask], 
# 			  fmt='o', label=label)
# 		ax.legend()
		
		# sumrule distribution
		ax = axs[0,1]
		xlabel = "Sum Rule"
		ylabel = "Occurances"
		ax.set(xlabel=xlabel, ylabel=ylabel)
		ax.hist(SR_BS_dist, bins=bins)
		ax.axvline(x=lower_SR, color='red', alpha=0.5, linestyle='--', marker='')
		ax.axvline(x=upper_SR, color='red', alpha=0.5, linestyle='--', marker='')
		ax.axvline(x=median_SR, color='red', linestyle='--', marker='')
		ax.axvline(x=SR_BS_mean, color='k', linestyle='--', marker='')
		
		# first moment distribution
		ax = axs[1,0]
		xlabel = "First Moment"
		ax.set(xlabel=xlabel, ylabel=ylabel)
		ax.hist(FM_BS_dist, bins=bins)
		ax.axvline(x=lower_FM, color='red', alpha=0.5, linestyle='--', marker='')
		ax.axvline(x=upper_FM, color='red', alpha=0.5, linestyle='--', marker='')
		ax.axvline(x=median_FM, color='red', linestyle='--', marker='')
		ax.axvline(x=FM_BS_mean, color='k', linestyle='--', marker='')
		
		# clock shift distribution
		ax = axs[1,1]
		xlabel = "Clock Shift"
		ax.set(xlabel=xlabel, ylabel=ylabel)
		ax.hist(CS_BS_dist, bins=bins)
		ax.axvline(x=lower_CS, color='red', alpha=0.5, linestyle='--', marker='')
		ax.axvline(x=upper_CS, color='red', alpha=0.5, linestyle='--', marker='')
		ax.axvline(x=median_CS, color='red', linestyle='--', marker='')
		ax.axvline(x=CS_BS_mean, color='k', linestyle='--', marker='')
		
		# SR extrapolation distribution
		ax = axs[0,2]
		xlabel = "Contact"
		ax.set(xlabel=xlabel, ylabel=ylabel)
		ax.hist(pFits*2*np.sqrt(2)*np.pi**2, bins=bins)
		
		# FM extrapolation distribution
		ax = axs[1,2]
		xlabel = "FM Extrapolation"
		ax.set(xlabel=xlabel, ylabel=ylabel)
		ax.hist(FM_extrap_dist, bins=bins)
		
		# make room for suptitle
		fig.tight_layout(rect=[0, 0.03, 1, 0.95])	
		
	if Correlations == True: 
		
		try: 
	
			x_values = [Cdist,SR_BS_dist,FM_BS_dist,CS_BS_dist,CoSR]
			x_values_names = ['C','Sum Rule','FM','Clock Shift','CoSR']
			y_values = [Cdist,SR_BS_dist,FM_BS_dist,CS_BS_dist,CoSR]
			y_values_names = ['C','Sum Rule','FM','Clock Shift','CoSR']
	
			x_values = [Cdist,SR_BS_dist,FM_BS_dist,CS_BS_dist]
			x_values_names = ['Contact','Sum Rule','First Moment','Clock Shift']
			y_values = [Cdist,SR_BS_dist,FM_BS_dist,CS_BS_dist]
			y_values_names = ['Contact','Sum Rule','First Moment','Clock Shift']
			
			
			rows = len(x_values)
			cols = rows + 1
			
			fig, axes = plt.subplots(rows, cols, figsize=(22, 15), sharex=False, sharey=False)
			axes = np.ravel(axes)
			
			medianvalues = [median_C, median_SR, median_FM, median_CS]
			uppermedianerrorvalues = [upper_C-median_C, upper_SR-median_SR, upper_FM-median_FM, upper_CS-median_CS]
			lowermedianerrorvalues = [median_C-lower_C, median_SR-lower_SR, median_FM-lower_FM, median_CS-lower_CS]
			meanvals = [np.mean(Cdist),SR_BS_mean, FM_BS_mean,CS_BS_mean]
			
			for i, (x_data, x_name) in enumerate(zip(x_values, x_values_names)):
			    for j, (y_data, y_name) in enumerate(zip(y_values, y_values_names)):
			        index = i * cols + j
					
			        if i == j :
			            ax = axes[index]
			            ax.hist(x_values[i], bins=20)
						
			            median = medianvalues[i]
			            ax.axvline(median, color='black', linestyle='--', linewidth=2, label='Median')
						
			            upper = uppermedianerrorvalues[i] + medianvalues[i]
			            ax.axvline(upper, color='r', linestyle='--', linewidth=2, label='Upper Percentile')
						
			            lower = - (lowermedianerrorvalues[i] - medianvalues[i]) 
			            ax.axvline(lower, color='r', linestyle='--', linewidth=2,label='Lower')
						
			            mean = meanvals[i]
			            ax.axvline(mean, color='g', linestyle='--', linewidth=2, label='Mean')
						
			        else: 
			            ax = axes[index]
			            ax.scatter( y_data, x_data)
	# 					
	# 		indices = [5,10,11,15,16,17]
	# 		for i in indices:
	# 		    ax = axes[i]
	# 		    ax.tick_params(axis='both', which='both', length=0)
	# 		    ax.set_xticks([])
	# 		    ax.set_yticks([])
	# 		    ax.spines['top'].set_visible(False)					
	# 		    ax.spines['bottom'].set_visible(False)				
	# 		    ax.spines['right'].set_visible(False)					
	# 		    ax.spines['left'].set_visible(False)	
	
						
			for i in range(0,4):
				axes[i].set_xlabel(f'{x_values_names[i]}')
				axes[i].xaxis.set_label_position('top')
				
			for i in range(rows):
				index = i * cols + cols - 5
				ax = axes[index]
				
				ax.set_ylabel(f'{y_values_names[i]}')
				ax.yaxis.set_label_position('left')			
	
	
			for i in range(rows):
	 			index = i * cols + cols - 1
	 			ax = axes[index]
	 			
	 			ax.axis('off')
	 			ax.text(0.5, 0.5, f'Median {x_values_names[i]} is {medianvalues[i]:.2f}+{uppermedianerrorvalues[i]:.2f}-{lowermedianerrorvalues[i]:.2f}', fontsize=12, ha='center', va='center')
			
	# 		medianlabel = Line2D([0], [0], marker='',label='Median', color='black', linestyle= '--', linewidth=3 )
	# 		upperlabel = Line2D([0], [0],marker='', label='Upper Percentile', color='r', linestyle= '--', linewidth=3 )
	# 		lowerlabel = Line2D([0], [0],marker='', label='Lower Percentile', color='r', linestyle= '--', linewidth=3 )		
	# 		meanlabel = Line2D([0], [0],marker='', label='Mean', color='g', linestyle= '--', linewidth=3 )		
	# 		
	# 		axes[5].legend(handles=[medianlabel,upperlabel,lowerlabel,meanlabel], prop={'size': 16})
	# 					
	
		except ValueError:
			print('x and y lengths do not match with some correlation plot - kp look into fixing this ')
			continue 
if Summaryplots == True:
	
# 	df = datatosavedf
	### load analysis results
		
	df = pd.read_excel(savefile, index_col=0, engine='openpyxl').reset_index()
	
# 	numeric_cols = datatosavedf.columns
# 	df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
# 	df = df[~df['Pulse Time (us)'].isin(['Pulse Time (us)'])]

	
	# get list of rf pulse times to loop over
	trflist = df['Pulse Time (us)'].unique()
	EFlist = df['EF'].unique()
	
	### plots
	plt.rcParams.update({"figure.figsize": [12,8]})
	fig, axes = plt.subplots(2,3)

# 	xlabel = r"Gain"
	xlabel = r"$\Omega_{R,pk}^2 t_{rf} / \Omega_{R, max}$"
	
	# sumrule vs gain
	ax_SR = axes[0,0]
	ylabel = "Sumrule"
	ax_SR.set(xlabel=xlabel, ylabel=ylabel)
	
	# First Moment vs gain
	ax_FM = axes[0,1]
	ylabel = "First Moment"
	ax_FM.set(xlabel=xlabel, ylabel=ylabel)
		
	# Clock Shift vs gain
	ax_CS = axes[0,2]
	ylabel = "Clock Shift"
	ax_CS.set(xlabel=xlabel, ylabel=ylabel)
	
	# C vs. gain
	ax_C = axes[1,0]
	ylabel = r"Contact $C/N$ [$k_F$]"
	ax_C.set(xlabel=xlabel, ylabel=ylabel)
	
	# C/sumrule vs gain
	ax_CoSR = axes[1,1]
	ylabel = "C/sumrule"
	ax_CoSR.set(xlabel=xlabel, ylabel=ylabel)

	# peak scaled transfer vs gain
	ax_pST = axes[1,2]
	ylabel = "Peak Scaled Transfer"
	ax_pST.set(xlabel=xlabel, ylabel=ylabel)
	

	# loop over pulse times
	
	unique_labelsEF = []
	unique_handlesEF = []
	unique_labelstrf = []
	unique_handlestrf = []
	
	for l, EF in enumerate(EFlist):
		
		subdf = df.loc[df['EF'] == EF]
		labelEF = r"EF"+"={}".format(EF)
		color = colors[l]
		light_color = tint_shade_color(color, amount=1+tintshade)
		dark_color = tint_shade_color(color, amount=1-tintshade)
		plt.rcParams.update({
						 "lines.markeredgecolor": dark_color,
						 "lines.markerfacecolor": light_color,
						 "lines.color": dark_color,
						 "legend.fontsize": 14})
		
		for i, trf in enumerate(trflist):
			sub_df = subdf.loc[subdf['Pulse Time (us)'] == trf]
			labeltrf = r"$t_{rf}$"+"={}us".format(trf)
			marker = markers[i]
		
 			
			try:
			### if MonteCarlo, select correct columns
				if MonteCarlo == True:
					SR = sub_df['SR MC']
					FM = sub_df['FM MC']
					CS = sub_df['CS MC']
					e_SR = sub_df['e_SR MC']
					e_FM = sub_df['e_FM MC']
					e_CS = sub_df['e_CS MC']
					
				elif Bootstrap == True:
					SR = sub_df['SR BS median']
					FM = sub_df['FM BS median']
					CS = sub_df['CS BS median']
					e_SR = np.array(list(zip(SR-sub_df['SR m conf'], sub_df['SR p conf']-SR))).T
					e_FM = np.array(list(zip(FM-sub_df['FM m conf'], sub_df['FM p conf']-FM))).T
					e_CS = np.array(list(zip(CS-sub_df['CS m conf'], sub_df['CS p conf']-CS))).T
				
				else: # select non-MC columns
					SR = sub_df['SR']
					FM = sub_df['FM']
					CS = sub_df['CS']
					e_SR = np.zeros(len(SR))
					e_FM = np.zeros(len(FM))
					e_CS = np.zeros(len(CS))
				
				xname = 'Transfer Scale'
				ax_C.errorbar(sub_df[xname], sub_df['C'], yerr=sub_df['e_C'], fmt=marker,ecolor = dark_color)
				plot_CoSR = ax_CoSR.errorbar(sub_df[xname], sub_df['C/SR'], yerr=[sub_df['C/SR']-sub_df['lower_C/SR']], fmt=marker,label=labeltrf,ecolor = dark_color)
				plot_pST = ax_pST.errorbar(sub_df[xname], sub_df['Peak Scaled Transfer'], 
						 yerr=sub_df['e_Peak Scaled Transfer'], fmt=marker, label=labelEF,ecolor = dark_color)
				
				ax_SR.errorbar(sub_df[xname], SR, yerr=np.abs(e_SR), fmt=marker,ecolor = dark_color)
				ax_FM.errorbar(sub_df[xname], FM, yerr=np.abs(e_FM), fmt=marker,ecolor = dark_color)
				ax_CS.errorbar(sub_df[xname], CS, yerr=np.abs(e_CS), fmt=marker,ecolor = dark_color)
				
			except TypeError:
				print()
				print('Missing a column in the .xlsx summary data file so the summary plots are being messed up')
				print('or something is wrong with the headers')
				
			if labelEF not in unique_labelsEF:
				unique_handlesEF.append(plot_pST)
				unique_labelsEF.append(labelEF)
			if labeltrf not in unique_labelstrf:
				unique_handlestrf.append(plot_CoSR)
				unique_labelstrf.append(labeltrf)
				
	leg1 = ax_pST.legend(unique_handlesEF,unique_labelsEF, loc='upper right')
	plt.gca().add_artist(leg1)
	ax_pST.legend(unique_handlestrf, unique_labelstrf, loc='center right')
	
	# add some average hlines
	CSmean = np.mean(df.CS)
	CoSRmean = np.mean(df['C/SR'])
	ax_CS.hlines(CSmean, min(df[xname]), max(df[xname]))
	ax_CoSR.hlines(CoSRmean, min(df[xname]), max(df[xname]))
# 	
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

