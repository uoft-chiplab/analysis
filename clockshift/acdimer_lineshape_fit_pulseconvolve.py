#-*- coding: utf-8 -*-

"""
Created on Thu Jun  6 20:12:51 2024

@author: coldatoms
"""

# %%
# import os
# proj_path = os.path.dirname(os.path.realpath(__file__))
# root = os.path.dirname(proj_path)
# data_path = os.path.join(proj_path, 'data')
# figfolder_path = os.path.join(proj_path, 'figures')

# import imp 
# library = imp.load_source('library',os.path.join(root,'library.py'))
# data_class = imp.load_source('data_class',os.path.join(root,'data_class.py'))
# =======
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
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib.colors as mc
import colorsys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from library import GammaTilde, pi, h

from clockshift.MonteCarloSpectraIntegration import DimerBootStrapFit
from scipy.stats import sem
import pwave_fd_interp as FD # FD distribution data for interpolation functions, Ben Olsen

## paths
proj_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(proj_path, "data")
root = os.path.dirname(proj_path)

## Diagnostic plots
plotraw = True

## Bootstrap switches
Bootstrap = True
Bootstrapplots = False

# save dataframe of convolutions
saveConv = False
saveResults = True

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

def SquareWindow(t, trf):
	window = np.piecewise(t, [np.abs(t) <= trf/2, np.abs(t) > trf/2], [1,0])
	return window


### metadata
metadata_filename = 'metadata_dimer_file.xlsx'
metadata_file = os.path.join(proj_path, metadata_filename)
metadata = pd.read_excel(metadata_file)
# if no filename selected, code will run over all files described in metadata (TO DO)
# filenames = ['2024-08-09_M_e']
# filenames = ['2024-08-11_P_e']
filenames = ['2024-07-17_J_e']
# filenames=[]
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

# %% START OF ANALYSIS LOOP
for filename in filenames:
	df = metadata.loc[metadata.filename == filename].reset_index()
	if df.empty:
		print("Dataframe is empty! The metadata likely needs updating." )
	
	xname = df['xname'][0]
	ff = df['ff'][0]
	trf = df['trf'][0] #s
	EF = df['EF'][0] #kHz
# 	EF = 1
	ToTF = df['ToTF'][0]
	VVA = df['VVA'][0] #V
	bg_freq_low = df['bg_freq_low'][0]
	bg_freq_high = df['bg_freq_high'][0]
	Bfield = df['Bfield'][0]
	res_freq = df['res_freq'][0]
	pulsetype = df['pulsetype'][0]
	gain = df['gain'][0]
	remove_indices = df['remove_indices'][0]
	Vppscope = df['Vpp'][0]
	# create data structure
	filename = filename + ".dat"
	run = Data(filename, path=data_path)
	
	# define a few more constants
	T = ToTF * (EF*1000)
	VpptoOmegaR = 27.5833 # kHz 
	if pulsetype == 'square':
		sqrtpulsearea=1
	elif pulsetype == 'blackman':
		sqrtpulsearea=np.sqrt(0.3)
		
	OmegaR = 2*pi*sqrtpulsearea*VpptoOmegaR*calInterp(VVA)*gain # 1/s
	print('OmegaR_interp: ' + str(OmegaR))
# 	OmegaR = 2*pi*sqrtpulsearea*VpptoOmegaR*Vppscope
# 	print('OmegaR_Vppscope: ' + str(OmegaR))
# 	print('OmegaR**2 * trf = ' +str(OmegaR**2 * trf) )
	
	# remove indices if requested
	if remove_indices == remove_indices: # nan check
		if np.isscalar(remove_indices):	
			run.data.drop(remove_indices, inplace=True)
		elif type(remove_indices) != int:
			remove_list = remove_indices.strip(' ').split(',')
			remove_indices = [int(index) for index in remove_list]
			run.data.drop(remove_indices, inplace=True)
	
	num = len(run.data[xname])
	
	
	### process data
	run.data['detuning'] = ((run.data.freq - res_freq) * 1e3)/EF # kHz in units of EF
	bgrange = [bg_freq_low*1e3/EF, bg_freq_high]
	bgmean = np.mean(run.data[run.data['detuning'].between(bgrange[0], bgrange[1])]['sum95'])
	run.data['transfer'] = (-run.data.sum95 + bgmean) / bgmean
	run.data['transfer'] = ((-run.data.sum95 + bgmean) / bgmean) / (2*pi) / (trf*1e3)
# 	fig, ax = plt.subplots
# 	ax.plot(run.data.detuning, run.data.transfer)
	run.data['ScaledTransfer'] = run.data.apply(lambda x: GammaTilde(x['transfer'],
									h*EF*1e3, OmegaR*1e3, trf), axis=1)
	run.group_by_mean('detuning')
	
	### plot raw data
	if plotraw:
		fig_raw, axs = plt.subplots(2,2)
		xlims = [run.avg_data.detuning.min()*EF,run.avg_data.detuning.max()*EF ]
		axs[0,0].errorbar(run.avg_data.detuning*EF, run.avg_data.c5, run.avg_data.em_c5)
		axs[0,0].set(ylabel='c5', xlim=xlims)
		axs[0,1].errorbar(run.avg_data.detuning*EF, run.avg_data.c9, run.avg_data.em_c9)
		axs[0,1].set(ylabel='c9', xlim=xlims)
		axs[1,0].errorbar(run.avg_data.detuning*EF, run.avg_data.sum95, run.avg_data.em_sum95)
		axs[1,0].set(ylabel='sum95', xlim=xlims)
		axs[1,1].errorbar(run.avg_data.detuning*EF, run.avg_data.transfer, run.avg_data.em_transfer)
		axs[1,1].set(ylabel='Transfer', xlim=xlims)
		fig_raw.tight_layout()
	
	### arbitrary cutoff in case some points look strange
	cutoff = -4.1*1e3/EF
	run.avg_data['filter'] = np.where(run.avg_data['detuning'] > cutoff, 1, 0)
	
	filtdf = run.avg_data[run.avg_data['filter']==1]
	x = filtdf['detuning']
	yparam = 'ScaledTransfer'
# 	yparam = 'transfer'
	y = filtdf[yparam]
	yerr = filtdf['em_' + yparam]
	
	nfiltdf = run.avg_data[run.avg_data['filter']==0]
	xnfilt = nfiltdf['detuning']
	ynfilt = nfiltdf[yparam]
	yerrnfilt = nfiltdf['em_' + yparam]
	
	### TEST: just choose Ebfix wherever the highest datapoint is, since detuning is heavily dependent on EF....
# 	Ebfix = x[np.argmax(y)] 
	Ebfix = -3.98 * 1e3/EF
	
	### prepping evaluation ranges
	xrange=0.15*1e3/EF
	xlow = Ebfix-xrange
	xhigh = Ebfix + xrange
	xnum = 1000
	xx = np.linspace(xlow, xhigh, xnum)

	# t in us
	def Sincw(w, t): # used Mathematica to get an exact FT
		return 0.797885 * np.sin(t/2*w)/w
	def Sincf(f,t): # kHz
		return Sincw(2*pi*f/1000, t)	
	def SincD(Delta, t): # takes dimensionless detuning
		return Sincw(2*pi*EF*Delta/1000, t) 
	
	def Blackmanw(w,t):
		match t:
			case 160:
				return (3*pi**(3/2) * (7*pi**2 - 4800*w**2) * np.sin(80*w))/ \
		(25*np.sqrt(2)*w*(pi**4 -8000*pi**2*w**2 + 10240000*w**4))
			case 80:
				return (3*pi**(3/2) * (7*pi**2 - 1200*w**2) * np.sin(40*w))/ \
		(25*np.sqrt(2)*w*(pi**4 -2000*pi**2*w**2 + 640000*w**4))
			case 40:
				return (3*pi**(3/2) * (7*pi**2 - 300*w**2) * np.sin(20*w))/ \
		(25*np.sqrt(2)*w*(pi**4 -500*pi**2*w**2 + 40000*w**4))
	
	def Blackmanf(f,t):
		return Blackmanw(2*pi*f/1000,t)
	
	def BlackmanD(Delta, t):
		return Blackmanw(2*pi*EF*Delta/1000, t)
	
	D = np.linspace(-10*1000/(trf*1e6) / EF, 10*1000/(trf*1e6) /EF, 1000)
	fig, ax=plt.subplots()
	if pulsetype=='square':
		yD = SincD(D, trf*1e6)
	elif pulsetype=='blackman':
		yD = BlackmanD(D, trf*1e6)
# 	norm = 397.932 # FROM MATHEMATICA
	norm = np.trapz(yD, D)
	ax.plot(D, yD/norm, '-')
	ax.set(xlabel='Detuning [EF]', ylabel='Magnitude')
	print('FT[pulse] norm: ' + str(norm))
	
	ToTFs = []
	popt_FDGs = []
	perr_FDGs = []
	SRs = []
	FMs = []
	CSs = []
	Cts = []
	chi2s=[]
	FDnorms =[]
	convnorms=[]
	
	arbscale=1
	qrange=xnum
	convres_list=[]
	# create FD lineshapes -- only use ToTF=0.3 for now
	FDnum=5
	if FDnum == 5:
		ttf = 0.30
	FDinterp = lambda x: np.interp(x, xx, lsFD(xx, Ebfix, arbscale, FDnum))
	FDnorm = quad(FDinterp, -qrange, qrange, points=xx, limit=2*xx.shape[0])
	
	def convfunc(tau, t):
# 		 return FDinterp(tau) * (SincD(t-tau)/norm)
		if pulsetype == 'square':
			return FDinterp(tau)/FDnorm[0] * (SincD(t-tau, trf*1e6)/norm)
		elif pulsetype=='blackman':
			return FDinterp(tau)/FDnorm[0] * (BlackmanD(t-tau, trf*1e6)/norm)
	def convint(t):
		# the integral converges better when ranges don't go to infinity
		sliderange=20
		qrangelow = Ebfix - sliderange
		qrangehigh= Ebfix + sliderange
		return quad(convfunc, qrangelow, qrangehigh, args=(t,))

	yyconv = []
	e_yyconv = []
	for xconv in xx:
		a, b = convint(xconv)
		yyconv.append(a)
		e_yyconv.append(b)
	convnorm = np.trapz(yyconv,xx)
	print('Conv norm: ' + str(convnorm))
	# create the convolution lineshape for current iteration
	convinterp = lambda x: np.interp(x, xx, yyconv)
	# I hate this
	if fitWithOffset:
		guess_FDG = [0.02, 0, 0]
		def convls(x, A, x0, C):
 			return A*convinterp(x-x0) + C
	else:
		def convls(x, A, x0):
			return A*convinterp(x-x0)
		guess_FDG = [0.02, 0]
	

	popt_FDG, pcov_FDG = curve_fit(convls, x, y, sigma=yerr, p0=guess_FDG)
	perr_FDG = np.sqrt(np.diag(pcov_FDG))
		
	# show convs explicitly
	fig_CVs, ax_CV = plt.subplots()
	ax_CV.plot(xx, FDinterp(xx), '-')
	if pulsetype=='square':
		ax_CV.plot(xx, SincD(xx-Ebfix, trf*1e6)/norm, '-', label='FT')
	elif pulsetype=='blackman':
		ax_CV.plot(xx, BlackmanD(xx-Ebfix, trf*1e6)/norm, '-', label='FT')
	ax_CV.plot(xx, yyconv, '-', label='conv')
	ax_CV.set(xlabel = 'Detuning [EF]', ylabel = 'Magnitude')
	ax_CV.legend()

 	### evaluate and plot on ax_ls
	yyconvls = convls(xx, *popt_FDG)
	fig_ls, ax_ls = plt.subplots()
	fig_ls.suptitle(filename + ', {:.2f} G, EF={:.1f} kHz, T/TF={:.2f}, T={:.1f} kHz, Ebfix={:.3f} MHz'.format(Bfield, EF, ToTF, ToTF*EF, Ebfix*EF/1000))
	ax_ls.errorbar(x, y, yerr, marker='o', ls='', markersize = 12, capsize=3, mew=3, mec = adjust_lightness('tab:gray',0.2), color='tab:gray', elinewidth=3)
	ax_ls.errorbar(xnfilt, ynfilt, yerrnfilt, marker='o', ls='', markersize = 12, capsize=3, mew=3, mfc='none', color='tab:gray', elinewidth=3)
	ax_ls.plot(xx, yyconvls, '-', linewidth=3)
	
	# with offset
	if fitWithOffset:
		SR_convls = np.trapz(yyconvls - popt_FDG[-1], xx)
		FM_convls = np.trapz((yyconvls - popt_FDG[-1])* xx, xx)
		fitdof = 3
	else:
		SR_convls = np.trapz(yyconvls, xx)
		FM_convls = np.trapz((yyconvls)* xx, xx)
		fitdof = 2
	CS_convls = FM_convls/0.5 # assumes ideal SR
	Ctilde_convls = CS_convls * (pi*kF*a13(Bfield)) / -2

	### calculate residuals and chi2
	ymodel = convls(x, *popt_FDG)
	convres = y - ymodel
	convres_list.append(convres)
	
	DOF = (1/(len(y) - fitdof))
	chi2 = DOF * np.sum((y-ymodel)**2 / yerr**2)
	
	print('chi2: ' + str(chi2))
	### residuals plots
# 	ylims=[-0.006, 0.006]
	fig_rs, ax_r = plt.subplots()
	ax_r.errorbar(x,convres,yerr,  marker='o', color='b', mec=adjust_lightness('b'), capsize=2, mew=2, elinewidth=2)
	ax_r.set(xlabel='Detuning [EF]', ylabel='Residuals', title= r'$\chi^2$ = {:.1f}'.format(chi2))
	
	# prepare results for output dataframe
	ToTFs.append(ToTF)
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
	
	### final plotting touches
	ax_ls.legend()
	xadjust = 3
	ax_ls.set_xlim([xlow+xadjust, xhigh -xadjust])
# 	ax_ls.set_ylim([y.min() - 0.005,y.max() + 0.005])
	ax_ls.set_ylabel(r'Scaled transfer $\tilde{\Gamma}$ [arb.]')
	ax_ls.set_xlabel(r'Detuning from 12-resonance $\Delta$ [EF]')
	# how hard is it to put a second x-axis on this thing
	# Put MHz frequencies on upper x-axis
	f = lambda x: x * EF /1e3 
	g = lambda x: x * EF/1e3 #wtf
	ax2 = ax_ls.secondary_xaxis("top", functions=(f,g))
	ax2.set_xlabel("Detuning [MHz]")
	
	plt.tight_layout()
	
	if saveResults == True:
		d = {'filename':filename,'ToTF': ToTFs,'popt':popt_FDGs, 'perr':perr_FDGs,
			 'SR':SRs, 'FM':FMs, 'CS':CSs, 'Ctilde':Cts, 'chi2':chi2s,'FDnorm':FDnorms, 
			 'Convnorm':convnorms}
		df = pd.DataFrame(data=d)
		
		save_filename = 'acdimer_lineshape_results.csv'
		if os.path.isfile(save_filename):
 			old_df = pd.read_csv(save_filename, index_col=0)
 			new_df = pd.concat([old_df, df])
 			new_df.to_csv(save_filename)
		else:
			df.to_csv(save_filename)
		
		file_directory = os.path.join('figures',filename)
		if not os.path.exists(file_directory):
			os.makedirs(file_directory)
		if fitWithOffset:
			str_ender = 'offset.png'
		else: 
			str_ender = 'nooffset.png'
		fig_file = os.path.join(file_directory, 'lineshape_fits_' + str_ender)
		fig_ls.savefig(os.path.join(proj_path, fig_file))
		fig_file = os.path.join(file_directory,'convolutions_' + str_ender)
		fig_CVs.savefig(os.path.join(proj_path, fig_file))
		fig_file = os.path.join(file_directory, 'residuals_' + str_ender)
		fig_rs.savefig(os.path.join(proj_path, fig_file))
		fig_file = os.path.join(file_directory, 'rawdata_' + str_ender)
		fig_raw.savefig(os.path.join(proj_path, fig_file))

	
	# %% BOOT STRAPPING
	def GenerateSpectraFit(Ebfix):
		def fit_func(x, A, sigma):
			x0 = Ebfix
			return A*np.sqrt(-x+x0) * np.exp((x - x0)/sigma) * np.heaviside(-x+x0,1)
		return fit_func
	
	# %%
	if Bootstrap == True:
		BOOTSRAP_TRAIL_NUM = 100
		xfitlims = [min(x), max(x)]
# 		fit_func = GenerateSpectraFit(Ebfix)
		fit_func = convls
		
		num_iter = 1000
		conf = 68.2689  # confidence level for CI
		
		# non-averaged data
		x = np.array(run.data['detuning'])
		num = len(x)
	# 		print(x)
		y = np.array(run.data['ScaledTransfer'])
		
		# sumrule, first moment and clockshift with analytic extension
		SR_BS_dist, FM_BS_dist, CS_BS_idl_dist, CS_BS_exp_dist, pFits, SR, FM, CS_idl, CS_exp  = \
			DimerBootStrapFit(x, y, xfitlims, Ebfix, fit_func, trialsB=BOOTSRAP_TRAIL_NUM)
		# print(SRlineshape)
		# print(SR)
		# print(FMlineshape)
		# print(FM)
		SR_BS_mean, e_SR_BS = (np.mean(SR_BS_dist), np.std(SR_BS_dist))
		FM_BS_mean, e_FM_BS = (np.mean(FM_BS_dist), np.std(FM_BS_dist))
		CS_BS_idl_mean, e_CS_BS_idl = (np.mean(CS_BS_idl_dist), np.std(CS_BS_idl_dist))
		CS_BS_exp_mean, e_CS_BS_exp = (np.mean(CS_BS_exp_dist), np.std(CS_BS_exp_dist))
		# SR_extrap_mean, e_SR_extrap = (np.mean(SR_extrap_dist), np.std(SR_extrap_dist))
		# FM_extrap_mean, e_FM_extrap = (np.mean(FM_extrap_dist), np.std(FM_extrap_dist))
		CS_BS_idl_mean, e_CS_BS_idl = (np.mean(CS_BS_idl_dist), sem(CS_BS_idl_dist))
		CS_BS_exp_mean, e_CS_BS_exp = (np.mean(CS_BS_exp_dist), sem(CS_BS_exp_dist))
		print(r"SR BS mean = {:.3f}$\pm$ {:.3f}".format(SR_BS_mean, e_SR_BS))
		print(r"FM BS mean = {:.3f}$\pm$ {:.3f}".format(FM_BS_mean, e_FM_BS))
		print(r"CS BS mean = {:.2f}$\pm$ {:.2f}".format(CS_BS_idl_mean, e_CS_BS_idl))
		print(r"CS BS mean = {:.2f}$\pm$ {:.2f}".format(CS_BS_exp_mean, e_CS_BS_exp))
		median_SR = np.nanmedian(SR_BS_dist)
		upper_SR = np.nanpercentile(SR_BS_dist, 100-(100.0-conf)/2.)
		lower_SR = np.nanpercentile(SR_BS_dist, (100.0-conf)/2.)
		
		median_FM = np.nanmedian(FM_BS_dist)
		upper_FM = np.nanpercentile(FM_BS_dist, 100-(100.0-conf)/2.)
		lower_FM = np.nanpercentile(FM_BS_dist, (100.0-conf)/2.)
		
		median_CS_idl = np.nanmedian(CS_BS_idl_dist)
		upper_CS_idl = np.nanpercentile(CS_BS_idl_dist, 100-(100.0-conf)/2.)
		lower_CS_idl = np.nanpercentile(CS_BS_idl_dist, (100.0-conf)/2.)
		median_CS_exp = np.nanmedian(CS_BS_exp_dist)
		upper_CS_exp = np.nanpercentile(CS_BS_exp_dist, 100-(100.0-conf)/2.)
		lower_CS_exp = np.nanpercentile(CS_BS_exp_dist, (100.0-conf)/2.)
		print(r"SR BS median = {:.3f}+{:.3f}-{:.3f}".format(median_SR,
													  upper_SR-SR, SR-lower_SR))
		print(r"FM BS median = {:.3f}+{:.3f}-{:.3f}".format(median_FM, 
													  upper_FM-FM, FM-lower_FM))
		print(r"CS BS idl median = {:.2f}+{:.3f}-{:.3f}".format(median_CS_idl, 
													  upper_CS_idl-median_CS_idl, median_CS_idl-lower_CS_idl))
	
	
	if (Bootstrapplots == True and Bootstrap == True):
		plt.rcParams.update({"figure.figsize": [10,8]})
		fig, axs = plt.subplots(2,2)
		fig.suptitle(filename)
		
		bins = 20
		
	# fits
		
		# sumrule distribution
		ax = axs[0,0]
		xlabel = "Sum Rule"
		ylabel = "Occurances"
		ax.set(xlabel=xlabel, ylabel=ylabel)
		ax.hist(SR_BS_dist, bins=bins)
		ax.axvline(x=lower_SR, color='red', alpha=0.5, linestyle='--', marker='')
		ax.axvline(x=upper_SR, color='red', alpha=0.5, linestyle='--', marker='')
		ax.axvline(x=median_SR, color='red', linestyle='--', marker='')
		ax.axvline(x=SR_BS_mean, color='k', linestyle='--', marker='')
		
		# first moment distribution
		ax = axs[0,1]
		xlabel = "First Moment"
		ax.set(xlabel=xlabel, ylabel=ylabel)
		ax.hist(FM_BS_dist, bins=bins)
		ax.axvline(x=lower_FM, color='red', alpha=0.5, linestyle='--', marker='')
		ax.axvline(x=upper_FM, color='red', alpha=0.5, linestyle='--', marker='')
		ax.axvline(x=median_FM, color='red', linestyle='--', marker='')
		ax.axvline(x=FM_BS_mean, color='k', linestyle='--', marker='')
		
		# clock shift distribution
		ax = axs[1,0]
		xlabel = "Clock Shift (ideal SR)"
		ax.set(xlabel=xlabel, ylabel=ylabel)
		ax.hist(CS_BS_idl_dist, bins=bins)
		ax.axvline(x=lower_CS_idl, color='red', alpha=0.5, linestyle='--', marker='')
		ax.axvline(x=upper_CS_idl, color='red', alpha=0.5, linestyle='--', marker='')
		ax.axvline(x=median_CS_idl, color='red', linestyle='--', marker='')
		ax.axvline(x=CS_BS_idl_mean, color='k', linestyle='--', marker='')
	
		ax = axs[1,1]
		xlabel = "Clock Shift (exp SR)"
		ax.set(xlabel=xlabel, ylabel=ylabel)
		ax.hist(CS_BS_exp_dist, bins=bins)
		ax.axvline(x=lower_CS_exp, color='red', alpha=0.5, linestyle='--', marker='')
		ax.axvline(x=upper_CS_exp, color='red', alpha=0.5, linestyle='--', marker='')
		ax.axvline(x=median_CS_exp, color='red', linestyle='--', marker='')
		ax.axvline(x=CS_BS_exp_mean, color='k', linestyle='--', marker='')
	
		
		# make room for suptitle
		fig.tight_layout(rect=[0, 0.03, 1, 0.95])	
		# %%	
		
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
			"{:.1f} +{:.1f}-{:.1f}".format(median_CS_idl, upper_CS_idl-median_CS_idl, median_CS_idl-lower_CS_idl),
			"{:.1f} +{:.1f}-{:.1f}".format(median_CS_exp, upper_CS_exp-median_CS_exp, median_CS_exp-lower_CS_exp),
			"{:.1f}".format(HFT_CS_EXP), 
			"{:.1f}".format(DIMER_CS_EXP + HFT_CS_EXP),
			"{:.1f}".format(CS_BS_idl_mean + HFT_CS_EXP)]
		table = list(zip(quantities, values))
		
		the_table = axexp.table(cellText=table, loc='center')
		the_table.auto_set_font_size(False)
		the_table.set_fontsize(12)
		the_table.scale(1,1.5)
		axexp.set(title='Experimental clock shifts [EF]')
		
							
