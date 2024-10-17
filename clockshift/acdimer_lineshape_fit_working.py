#-*- coding: utf-8 -*-

"""
AC dimer association spectra analysis script.

@author: Chip Lab
"""

# %%
import os
import sys
# this is a hack to access modules in the parent directory
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
if parent_dir not in sys.path:
	sys.path.append(parent_dir)

from data_class import Data
from save_df_to_xlsx import save_df_row_to_xlsx
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib.colors as mc
import colorsys
import pandas as pd
import numpy as np
import corner
import matplotlib.pyplot as plt
from library import GammaTilde, pi, h

from clockshift.MonteCarloSpectraIntegration import DimerBootStrapFit, dist_stats
import clockshift.pwave_fd_interp as FD # FD distribution data for interpolation functions, Ben Olsen

## paths
proj_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(proj_path, "data")
root = os.path.dirname(proj_path)
figfolder_path = os.path.join(proj_path, 'figures')

# bootstrap iterations
BOOTSRAP_TRAIL_NUM = 100

# print statements
Talk = True

## Bootstrap switches
Bootstrap = True
Bootstrapplots = True
Correlations = True

# save dataframe of convolutions
Save = True
# saveConv = False

# determines whether convolved lineshape fit to data has offset parameter
fitWithOffset = False

## plotting things
linewidth=4
def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

### lineshape functions
# various lineshape functions for fitting or modeling
def lsZY_highT(omega, Eb, TMHz, arb_scale=1):
	Gamma = arb_scale*(np.exp((omega - Eb)/TMHz)) / np.sqrt((-omega+Eb)) * np.heaviside(-omega+Eb, 1)
	Gamma = np.nan_to_num(Gamma)
	return Gamma

def lineshapefit(x, A, x0, sigma):
	ls = A*np.sqrt(-x-x0) * np.exp((x + x0)/sigma) * np.heaviside(-x-x0,1)
	ls = np.nan_to_num(ls)
	return ls

def lsMB_fixedEb(x, x0, A, sigma):
	ls = A*np.sqrt(-x+x0) * np.exp((x - x0)/sigma) * np.heaviside(-x+x0,1)
	ls = np.nan_to_num(ls)
	return ls

def lsmom3_fixedEb(x, x0, A, sigma):
	ls = A*(-x+x0) * np.exp((x-x0)/sigma) * np.heaviside(-x+x0,1)
	ls = np.nan_to_num(ls)
	return ls

def lineshape_zeroT(x, A, x0,C):
	ls = A*(2*kF**3 - 3*kF**2*np.sqrt(-x-x0) + np.sqrt(-x-x0)**3)*np.sqrt(-x-x0)/(-x-x0) + C
	ls = np.nan_to_num(ls)
	return ls

def gaussian(x, A, x0, sigma):
	return A * np.exp(-(x-x0)**2/(2*sigma**2))
def norm_gaussian(x, x0, sigma):
	return 1/(np.sqrt(2*pi*sigma**2)) * np.exp(-(x-x0)**2/(2*sigma**2))

def lsFD(x, x0, A, numDA):
	PFD = FD.distRAv[numDA] # 5 = 0.30 T/TF, 6 = 0.40 T/TF
	ls = A * PFD(np.sqrt(-x+x0))
	ls = np.nan_to_num(ls)
	return ls

def convfunc(tau, t, t0, sigma):
	 return FDinterp(tau) * norm_gaussian(t-tau, t0, sigma)
def convint(t, t0, mu):
	# the integral converges better when ranges don't go to infinity
	sliderange=10
	qrangelow = Ebfix - sliderange
	qrangehigh= Ebfix + sliderange
	return quad(convfunc, qrangelow, qrangehigh, args=(t,t0,mu))

def GenerateSpectraFit(Ebfix):
		def fit_func(x, A, sigma):
			x0 = Ebfix
			return A*np.sqrt(-x+x0) * np.exp((x - x0)/sigma) * np.heaviside(-x+x0,1)
		return fit_func

### save file name
savefile = 'acdimer_lineshape_results.xlsx'

### metadata
metadata_filename = 'metadata_dimer_file.xlsx'
metadata_file = os.path.join(proj_path, metadata_filename)
metadata = pd.read_excel(metadata_file)
# if no filename selected, code will run over all files described in metadata (TO DO)
# filenames = ['2024-10-01_F_e']
# filenames = ['2024-06-12_S_e']
# filenames=[]
filenames = False
### if the filenames list is empty, run over all available files in metadata
if not filenames:
	filenames = metadata.filename
	
## constants
kF = 1.1e7
a0 = 5.2917721092e-11 # m
re = 107 * a0
re_i = 104 * a0
re_f = 124 * a0
re = np.sqrt(re_i*re_f) # geometric mean of initial and final
def a13(B):
	abg = 167.3*a0 
	DeltaB = 7.2
	B0=224.2
	return abg*(1 - DeltaB/(B-B0))

# using 43.2 MHz VVA calibration (very minor difference at this VVA)
plot_VVAcal = False
data_file = os.path.join(data_path, 'VVAtoVpp_square_43p2MHz.txt')
cal = pd.read_csv(data_file, sep='\t', skiprows=1, names=['VVA','Vpp'])
calInterp = lambda x: np.interp(x, cal['VVA'], cal['Vpp'])
if plot_VVAcal:
	fig, ax = plt.subplots()
	xx = np.linspace(cal.VVA.min(), cal.VVA.max(),100)
	ax.plot(xx, calInterp(xx), '--')
	ax.plot(cal.VVA, cal.Vpp, 'o')
	ax.set(xlabel='VVA', ylabel='Vpp')
	
##############################
######### Analysis ###########
##############################
save_df_index = 0
for filename in filenames:
	df = metadata.loc[metadata.filename == filename].reset_index()
	if df.empty:
		print("Dataframe is empty! The metadata likely needs updating." )
	
	runfolder = filename 
	xname = df['xname'][0]
	ff = df['ff'][0]
	trf = df['trf'][0] #s
	EF = df['EF'][0] #kHz
	ToTF = df['ToTF'][0]
	VVA = df['VVA'][0] #V
	bg_freq_low = df['bg_freq_low'][0]
	bg_freq_high = df['bg_freq_high'][0]
	Bfield = df['Bfield'][0]
	res_freq = df['res_freq'][0]
	pulsetype = df['pulsetype'][0]
	remove_indices = df['remove_indices'][0]
	
	# create data structure
	filename = filename + ".dat"
	run = Data(filename, path=data_path)
	
	# define a few more constants
	T = ToTF * (EF*1000)
	VpptoOmegaR = 27.5833 # kHz 
	if pulsetype == 'square':
		pulsearea=1
	OmegaR = 2*pi*pulsearea*VpptoOmegaR*calInterp(VVA) # 1/s
	
	# remove indices if requested
	if remove_indices == remove_indices: # nan check
		if type(remove_indices) != int:	
			remove_list = remove_indices.strip(' ').split(',')
			remove_indices = [int(index) for index in remove_list]
		run.data.drop(remove_indices, inplace=True)
	
	num = len(run.data[xname])
	
	
	### process data
	run.data['detuning'] = ((run.data.freq - res_freq) * 1e3)/EF # kHz in units of EF
	bgrange = [-3.98*1e3/EF, run.data.detuning.max()]
	bgmean = np.mean(run.data[run.data['detuning'].between(bgrange[0], bgrange[1])]['sum95'])
	run.data['transfer'] = (-run.data.sum95 + bgmean) / bgmean
	run.data['ScaledTransfer'] = run.data.apply(lambda x: GammaTilde(x['transfer'],
									h*EF*1e3, OmegaR*1e3, trf), axis=1)
	run.group_by_mean('detuning')
	
	### arbitrary cutoff in case some points look strange
	cutoff = -4.04*1e3/EF
	run.avg_data['filter'] = np.where(run.avg_data['detuning'] > cutoff, 1, 0)
	
	filtdf = run.avg_data[run.avg_data['filter']==1]
	x = filtdf['detuning']
	y = filtdf['ScaledTransfer']
	yerr = filtdf['em_ScaledTransfer']
	
	nfiltdf = run.avg_data[run.avg_data['filter']==0]
	xnfilt = nfiltdf['detuning']
	ynfilt = nfiltdf['ScaledTransfer']
	yerrnfilt = nfiltdf['em_ScaledTransfer']
	
	### TEST: just choose Ebfix whever the highest datapoint is, since detuning is heavily dependent on EF....
	Ebfix = x[np.argmax(y)]
	
	### prepping evaluation ranges
	xrange=0.08*1e3/EF
	xlow = Ebfix-xrange
	xhigh = Ebfix + xrange
	xnum = 1000
	xx = np.linspace(xlow, xhigh, xnum)
	
	# list of chemical potentials/broadening parameters to iterate over
	mutrap_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) #units of EF
	
	### preparing lists to hold results
	ToTFs = []
	mus = []
	popt_FDGs = []
	perr_FDGs = []
	SRs = []
	FMs = []
	CSs = []
	Cts = []
	chi2s=[]
	FDnorms =[]
	convnorms=[]
	# prep data plot
	fig_ls, ax_ls = plt.subplots()
	fig_ls.suptitle(filename + ', {:.2f} G, EF={:.1f} kHz, T/TF={:.2f}, T={:.1f} kHz, Ebfix={:.3f} MHz'.format(Bfield, EF, ToTF, ToTF*EF, Ebfix*EF/1000))
	ax_ls.errorbar(x, y, yerr, marker='o', ls='', markersize = 12, capsize=3, mew=3, mec = adjust_lightness('tab:gray',0.2), color='tab:gray', elinewidth=3)
	ax_ls.errorbar(xnfilt, ynfilt, yerrnfilt, marker='o', ls='', markersize = 12, capsize=3, mew=3, mfc='none', color='tab:gray', elinewidth=3)
	
	arbscale=1
	qrange=xnum
	convres_list=[]
	# create FD lineshapes -- only use ToTF=0.3 for now
	FDnum=5
	if FDnum == 5:
		ttf = 0.30
	FDinterp = lambda x: np.interp(x, xx, lsFD(xx, Ebfix, arbscale, FDnum))
	FDnorm = quad(FDinterp, -qrange, qrange, points=xx, limit=2*xx.shape[0])
# 	print('FD norm: '+str(FDnorm))	
	fig_CVs, ax_CVs = plt.subplots(len(mutrap_list), figsize=(10,20))
	fig_rs, ax_rs = plt.subplots(len(mutrap_list), figsize=(10,20))
	
	for enum, i in enumerate(mutrap_list):
		mu = i
		print('mu: ' + str(mu))
		
		yyconv = []
		e_yyconv = []
		for xconv in xx:
			a, b = convint(xconv, 0, mu)
			yyconv.append(a)
			e_yyconv.append(b)
		convnorm = np.trapz(yyconv,xx)
# 		print('Conv norm: ' + str(convnorm))
		# create the convolution lineshape for current iteration
		convinterp = lambda x: np.interp(x, xx, yyconv)
		# I hate this
		if fitWithOffset:
			guess_FDG = [0.05, 0, 0]
			def convls(x, A, x0, C):
	 			return A*convinterp(x-x0) + C
		else:
			def convls(x, A, x0):
				return A*convinterp(x-x0)
			guess_FDG = [0.05, 0]
	

		popt_FDG, pcov_FDG = curve_fit(convls, x, y, sigma=yerr, p0=guess_FDG)
		perr_FDG = np.sqrt(np.diag(pcov_FDG))
# 		print(popt_FDG)
		
		# show convs explicitly
		ax_CV=ax_CVs[enum]
		labelstr = str(mu) + ' EF'
		ax_CV.plot(xx, FDinterp(xx), '-', label=labelstr)
		ax_CV.plot(xx, norm_gaussian(xx, Ebfix-3, mu), '-', label='Gaussian, width={:.2} EF'.format(mu))
		ax_CV.plot(xx, yyconv, '-', label='conv')
		ax_CV.legend()
	
	
	 	### evaluate and plot on ax_ls
		yyconvls = convls(xx, *popt_FDG)
		ax_ls.plot(xx, yyconvls, '-', linewidth=3,
				 label=labelstr)
		
		# with offset
		if fitWithOffset:
			SR_convls = np.trapz(yyconvls - popt_FDG[-1], xx)
			FM_convls = np.trapz((yyconvls - popt_FDG[-1])* xx, xx)
		else:
			SR_convls = np.trapz(yyconvls, xx)
			FM_convls = np.trapz((yyconvls)* xx, xx)
		CS_convls = FM_convls/0.5 # assumes ideal SR
		Ctilde_convls = CS_convls * (pi*kF*a13(Bfield)) / -2
	
		### calculate residuals and chi2
		ymodel = convls(x, *popt_FDG)
		convres = y - ymodel
		convres_list.append(convres)
		
		DOF = (1/(len(y) - 2))
		chi2 = DOF * np.sum((y-ymodel)**2 / yerr**2)
		
		print('chi2: ' + str(chi2))
		### residuals plots
		ylims=[-0.006, 0.006]
		ax_r = ax_rs[enum]
		ax_r.errorbar(x,convres,yerr,  marker='o', color='b', mec=adjust_lightness('b'), capsize=2, mew=2, elinewidth=2)
		ax_r.set(title=labelstr, ylim=ylims)
		
		# prepare results for output dataframe
		ToTFs.append(ttf)
		mus.append(mu)
		popt_FDGs.append(popt_FDG)
		perr_FDGs.append(perr_FDG)
		SRs.append(SR_convls)
		FMs.append(FM_convls)
		CSs.append(CS_convls)
		Cts.append(Ctilde_convls)
		chi2s.append(chi2)
		FDnorms.append(FDnorm[0])
		convnorms.append(convnorm)
		
		print('Convolution SR: ' + str(SR_convls))
		print('Convolution FM: ' + str(FM_convls))
		print('Convolution CS: ' + str(CS_convls))
		print('Convolution Ctilde: ' + str(Ctilde_convls))
		
	### time for clock shift analysis I guess
		
	# Ctilde_est = 1.44
	Ctilde_est = 1.8 # ToTF ~ 0.3
	cs_pred = -2/(pi*kF*a13(Bfield))*Ctilde_est
	print("predicted dimer clock shift [Eq. (5)]: "+ str(cs_pred))
	
	
	cstot_pred_zerorange = -1/(pi*kF*a13(Bfield)) * Ctilde_est
	print("Predicted total clock shift w/o eff. range term [Eq. (1)]: "+ str(cstot_pred_zerorange))
	csHFT_pred = 1/(pi*kF*a13(Bfield)) *Ctilde_est
	print("Predicted HFT clock shift w/o eff. range term: " + str(csHFT_pred))
	
	cstot_pred = -1/(pi*kF*a13(Bfield)) * (1- pi**2/8*re/a13(Bfield)) * Ctilde_est
	print("Predicted total clock shift w/ eff. range term [Eq. (1)]: "+ str(cstot_pred))
	csHFT_pred_corr = 1/(pi*kF*a13(Bfield))* (1/(np.sqrt(1-re/a13(Bfield)))) *Ctilde_est
	print("Predicted HFT clock shift w/ eff. range term: " + str(csHFT_pred_corr))
	kappa = 1.2594*1e8
	I_d = kF*Ctilde_est / (pi * kappa) * (1/(1+re/a13(Bfield)))
	print("Predicted dimer spectral weight [Eq. 6]: " + str(I_d))
	
	correctionfactor = 1/(kappa*a13(Bfield))*(1/(1+re/a13(Bfield)))
	print("Eff. range correction: "+ str(correctionfactor))
	
	re_range = np.linspace(re_i/a0, re_f/a0, 100)
	CS_HFT_CORR = 1/(pi*kF*a13(Bfield))* (1/(np.sqrt(1-re_range*a0/a13(Bfield)))) *Ctilde_est
	CS_TOT_CORR = -1/(pi*kF*a13(Bfield)) * (1- pi**2/8*re_range*a0/a13(Bfield)) * Ctilde_est
	CS_DIM_CORR = CS_TOT_CORR - CS_HFT_CORR
	print("CS_HFT_CORR bounds = ({:.1f}, {:.1f})".format(min(CS_HFT_CORR), max(CS_HFT_CORR)))
	print("CS_TOT_CORR bounds = ({:.1f}, {:.1f})".format(min(CS_TOT_CORR), max(CS_TOT_CORR)))
	print("CS_DIM_CORR bounds = ({:.1f}, {:.1f})".format(min(CS_DIM_CORR), max(CS_DIM_CORR)))
		
	
	### final plotting touches
	ax_ls.legend()
	xadjust=2
	ax_ls.set_xlim([xlow+xadjust, xhigh -xadjust])
	ax_ls.set_ylim([y.min() - 0.005,y.max() + 0.005])
	ax_ls.set_ylabel(r'Scaled transfer $\tilde{\Gamma}$ [arb.]')
	ax_ls.set_xlabel(r'Detuning from 12-resonance $\Delta$ [EF]')
	# how hard is it to put a second x-axis on this thing
	# Put MHz frequencies on upper x-axis
	f = lambda x: x * EF /1e3 
	g = lambda x: x * EF/1e3 #wtf
	ax2 = ax_ls.secondary_xaxis("top", functions=(f,g))
	ax2.set_xlabel("Detuning [MHz]")
	
	plt.tight_layout()
	
	results = {'filename':filename,'ToTF': ToTFs, 'mu': mus, 'popt':popt_FDGs, 'perr':perr_FDGs,
		 'SR':SRs, 'FM':FMs, 'CS':CSs, 'Ctilde':Cts, 'chi2':chi2s,'FDnorm':FDnorms, 
		 'Convnorm':convnorms}
	df = pd.DataFrame(data=results)
	
	fig_results, axs = plt.subplots(2,3)
	xplot = 'mu'
	ax_SR = axs[0,0]
	ax_SR.set(xlabel=xplot,ylabel='SR')
	ax_SR.plot(df[xplot], df.SR)
	ax_FM = axs[0,1]
	ax_FM.plot(df[xplot], df.FM)
	ax_FM.set(xlabel=xplot,ylabel='FM')
	ax_CS = axs[0,2]
	ax_CS.plot(df[xplot], df.CS)
	ax_CS.set(xlabel=xplot,ylabel='CS')
	ax_Ct = axs[1,0]
	ax_Ct.plot(df[xplot], df.Ctilde)
	ax_Ct.set(xlabel=xplot,ylabel='Ctilde')
	ax_X2 = axs[1,1]
	ax_X2.plot(df[xplot], df.chi2)
	ax_X2.set(xlabel=xplot, ylabel=r'$\chi^2$', ylim=[df.chi2.min() - 0.01, df.chi2.max()+0.01])
	ax_CV = axs[1,2]
	ax_CV.plot(df[xplot], df.mu)
	ax_CV.set(xlabel=xplot,ylabel='mu')
	fig_results.tight_layout()

##########################
##### Bootstrapping ######
##########################
	# %%
	if Bootstrap == True:
		xfitlims = [min(x), max(x)]
		fit_func = GenerateSpectraFit(Ebfix)
		
		num_iter = 1000
		conf = 68.2689  # confidence level for CI
		
		# non-averaged data
		x = np.array(run.data['detuning'])
		num = len(x)
	# 		print(x)
		y = np.array(run.data['ScaledTransfer'])
		
		# sumrule, first moment and clockshift with analytic extension
		SR_BS_dist, FM_BS_dist, CS_BS_idl_dist, CS_BS_exp_dist, pFits = \
			DimerBootStrapFit(x, y, xfitlims, Ebfix, fit_func, trialsB=BOOTSRAP_TRAIL_NUM)
			
		# list all ditributions to compure stats on
		dists = [SR_BS_dist, FM_BS_dist, CS_BS_idl_dist, CS_BS_exp_dist]
		names = ['SR', 'FM', 'CS_idl', 'CS_exp']
		
		# update results with all stats from dists
		stats_dict = {}
		for name, dist in zip(names, dists):
			for key, value in dist_stats(dist, conf).items():
				stats_dict[name+'_'+key] = value
		results.update(stats_dict)
		
		if Talk == True:
			for names in names:
				print(r"{} BS median = {:.3f}+{:.3f}-{:.3f}".format(name, 
											results[name+'_median'],
						   results[name+'_upper']-results[names+'_median'], 
						        results['SR_median']-results['SR_lower']))
	
	if (Bootstrapplots == True and Bootstrap == True):
		plt.rcParams.update({"figure.figsize": [10,8]})
		fig, axs = plt.subplots(2,2)
		fig.suptitle(filename)
		
		bins = 20
		
		ylabel = "Occurances"
		xlabels = ["Sum Rule", "First Moment", "Clock Shift Idl", 
				 "Clock Shift Exp"]
		dists = [SR_BS_dist, FM_BS_dist, CS_BS_idl_dist, CS_BS_exp_dist]
		
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
			figpath = os.path.join(figfolder_path, runfolder)
			fig.savefig(os.path.join(figpath, hist_figname))
	
		fig.tight_layout()	
		# %%	

		### generate table
		fig, axs = plt.subplots(2)
		axpred = axs[0]
		axpred.axis('off')
		axpred.axis('tight')
		quantities = [r"$\widetilde{C}$",
					  r"$r_e/a_0$",
					  r"$\Omega_d$ (zero range)",
					  r"$\Omega_+$ (zero range)", 
					  r"$\Omega_{tot}$ (zero range)", 
					  r"$\Omega_d$ (corr.)",
					  r"$\Omega_+$ (corr.)", 
					  r"$\Omega_{tot}$ (corr.)"]
		values = ["{:.1f}".format(Ctilde_est),
				  "{:.1f}".format(re/a0),
			"{:.1f}".format(cs_pred), 
				  "{:.1f}".format(csHFT_pred),
				  "{:.1f}".format(cstot_pred_zerorange),
				  "{:.1f}".format(cstot_pred - csHFT_pred_corr),
				  "{:.1f}".format(csHFT_pred_corr),
				  "{:.1f}".format(cstot_pred)]
		table = list(zip(quantities, values))
		
		the_table = axpred.table(cellText=table, loc='center')
		the_table.auto_set_font_size(False)
		the_table.set_fontsize(12)
		the_table.scale(1,1.5)
		
		
		# # %%
		axpred.set(title='Predicted clock shifts [EF]')
		
		axexp = axs[1]
		axexp.axis('off')
		axexp.axis('tight')
		quantities = [
					  r"$\widebar{\Omega_d}$ (lineshape)",
					  r"$\widebar{\Omega_d}$ (bootstrap, ideal SR)",
					  r"$\widebar{\Omega_d}$ (bootstrap, exp SR)",
					  r"$\Omega_+$", 
					  r"$\Omega_{tot}$ (lineshape)",
					  r"$\Omega_{tot}$ (bootstrap)"]
		# EXPERIMENTAL VALUES
	# 	HFT_CS_EXP = 5.77
		HFT_CS_EXP = 7.3
		DIMER_CS_EXP = -8.7
		values = [
			"{:.1f}".format(DIMER_CS_EXP),
			"{:.1f} +{:.1f}-{:.1f}".format(results['CS_idl_median'], 
						  results['CS_idl_upper']-results['CS_idl_median'], 
						  results['CS_idl_median']-results['CS_idl_lower']),
			"{:.1f} +{:.1f}-{:.1f}".format(results['CS_exp_median'], 
						  results['CS_exp_upper']-results['CS_exp_median'], 
						  results['CS_exp_median']-results['CS_exp_lower']),
			"{:.1f}".format(HFT_CS_EXP), 
			"{:.1f}".format(DIMER_CS_EXP + HFT_CS_EXP),
			"{:.1f}".format(results['CS_BS_idl_median'] + HFT_CS_EXP)]
		table = list(zip(quantities, values))
		
		the_table = axexp.table(cellText=table, loc='center')
		the_table.auto_set_font_size(False)
		the_table.set_fontsize(12)
		the_table.scale(1,1.5)
		axexp.set(title='Experimental clock shifts [EF]')
		
#############################
####### Correlations ########
#############################
		
	if Correlations == True and Bootstrap == True:
		dists = np.vstack(dists)
		fig_cor = corner.corner(dists.T, labels=xlabels)
		
###############################
####### Saving Results ########
###############################
	if Save == True:
		savedf = pd.DataFrame(results, index=[save_df_index])
		save_df_index += 1
		
		save_df_row_to_xlsx(savedf, savefile, filename)

# 	if saveConv == True:
# 		save_filename = 'acdimer_lineshape_results.csv'
# 		if os.path.isfile(save_filename):
# 			old_df = pd.read_csv(save_filename, index_col=0)
# 			new_df = pd.concat([old_df, df])
# 			new_df.to_csv(save_filename)
# 		else:
# 			df.to_csv(save_filename)
# 		
# 		file_directory = os.path.join('figures',filename)
# 		if not os.path.exists(file_directory):
# 			os.makedirs(file_directory)
# 		if fitWithOffset:
# 			str_ender = 'offset.png'
# 		else: 
# 			str_ender = 'nooffset.png'
# 			
# 		figs = [fig_ls, fig_CVs, fig_rs, fig_results, fig_cor]
# 		fig_names = ['lineshape_fits_', 'convolutions_', 'residuals_',
# 				'results_', 'correlations_']
# 		
# 		for fig, name in zip(figs, fig_names):
# 			fig_file = os.path.join(file_directory, name + str_ender)
# 			fig.savefig(os.path.join(proj_path, fig_file))	
