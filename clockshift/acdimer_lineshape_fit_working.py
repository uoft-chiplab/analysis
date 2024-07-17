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
import matplotlib.colors as mc
import colorsys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from library import GammaTilde, pi, h

from clockshift.MonteCarloSpectraIntegration import DimerBootStrapFit
from scipy.stats import sem

# paths
proj_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(proj_path, "data")
root = os.path.dirname(proj_path)

Bootstrap = True
Bootstrapplots = True


# plotting things
linewidth=4
def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

ToTF=0.31
EF=15.2 # kHz
kF = 1.1e7

Bfield = 202.14 # G
a0 = 5.2917721092e-11 # m
re = 107 * a0
def a13(B):
	abg = 167.3*a0 #.6 or .3?
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
pulsearea = 1 # square pulse
VpptoOmegaR = 27.5833 # kHz 
VVA = 1.4
OmegaR = 2*pi*pulsearea*VpptoOmegaR*calInterp(VVA) # 1/s
ff=1.03
trf = 640e-6

# various lineshape functions for fitting or modeling
def lsZY_highT(omega, Eb, TMHz, arb_scale=1):
	Gamma = arb_scale*(np.exp((omega - Eb)/TMHz)) / np.sqrt((-omega+Eb)) * np.heaviside(-omega+Eb, 1)
	Gamma = np.nan_to_num(Gamma)
	return Gamma

def lineshapefit(x, A, x0, sigma):
	ls = A*np.sqrt(-x-x0) * np.exp((x + x0)/sigma) * np.heaviside(-x-x0,1)
	ls = np.nan_to_num(ls)
	return ls
# Ebfix = 3.97493557
Ebfix = -3.975*1e3 /EF
# Ebfix = -3.98*1e3/EF
def lsMB_fixedEb(x, A, sigma):
	x0 = Ebfix
	ls = A*np.sqrt(-x+x0) * np.exp((x - x0)/sigma) * np.heaviside(-x+x0,1)
	ls = np.nan_to_num(ls)
	return ls

def lsmom3_fixedEb(x, A, sigma):
	x0 = Ebfix
	ls = A*(-x+x0) * np.exp((x-x0)/sigma) * np.heaviside(-x+x0,1)
	ls = np.nan_to_num(ls)
	return ls

def lineshape_zeroT(x, A, x0,C):
	ls = A*(2*kF**3 - 3*kF**2*np.sqrt(-x-x0) + np.sqrt(-x-x0)**3)*np.sqrt(-x-x0)/(-x-x0) + C
	ls = np.nan_to_num(ls)
	return ls

def gaussian(x, A, x0, sigma):
	return A * np.exp(-(x-x0)**2/(2*sigma**2))


filename='2024-06-12_S_e.dat'
run = Data(filename,path=data_path)
T = ToTF * (EF*1000)
field = 202.14
freq75 = 47.2227 # MHz, 202.14 G

# process data
run.data['detuning'] = ((run.data.freq - freq75) * 1e3)/EF # kHz in units of EF
bgrange = [-3.97*1e3/EF, run.data.detuning.max()]
bgmean = np.mean(run.data[run.data['detuning'].between(bgrange[0], bgrange[1])]['sum95'])
run.data['transfer'] = (-run.data.sum95 + bgmean) / bgmean
run.data['ScaledTransfer'] = run.data.apply(lambda x: GammaTilde(x['transfer'],
								h*EF*1e3, OmegaR*1e3, trf), axis=1)
run.group_by_mean('detuning')

# arbitrary cutoff because some points look strange
cutoff = -4.02*1e3/EF
run.avg_data['filter'] = np.where(run.avg_data['detuning'] > cutoff, 1, 0)

filtdf = run.avg_data[run.avg_data['filter']==1]
x = filtdf['detuning']
y = filtdf['ScaledTransfer']
yerr = filtdf['em_ScaledTransfer']

nfiltdf = run.avg_data[run.avg_data['filter']==0]
xnfilt = nfiltdf['detuning']
ynfilt = nfiltdf['ScaledTransfer']
yerrnfilt = nfiltdf['em_ScaledTransfer']

# modified MB
guess1 = [1, 4]
popt1,pcov1 = curve_fit(lsMB_fixedEb, x, y, sigma=yerr, p0=guess1)
perr1 = np.sqrt(np.diag(pcov1))
print('modified MB lineshape fit: ')
print(popt1)
# modified MB with extra factor of momentum
guess2 = [1, T * 1e-3/EF]
popt2,pcov2 = curve_fit(lsmom3_fixedEb, x, y, sigma=yerr, p0=guess2)
perr2 = np.sqrt(np.diag(pcov2))
print('modified MB lineshape with another relative momentum: ')
print(popt2)
# Gaussian fit
guess3 = [1, -262, T * 1e-3/EF]
popt3,pcov3 = curve_fit(gaussian, x, y, sigma=yerr, p0=guess3)
perr3 = np.sqrt(np.diag(pcov3))
print('Gaussian: ')
print(popt3)

# evaluate each of the fits
xrange=0.10*1e3/EF
xlow = Ebfix-xrange
xhigh = Ebfix + xrange
xx = np.linspace(xlow, xhigh, 400)
yy = lsMB_fixedEb(xx, *popt1)
yy2 = lsmom3_fixedEb(xx, *popt2)
yy3 = gaussian(xx, *popt3)
guessT0 = [4e-25,Ebfix,0]
yyT0 = lineshape_zeroT(xx, *guessT0)
arbscale = 1e-2/2
epsilon = 0.001 # small value to avoid divergence
xxZY = np.linspace(xlow, xhigh, 400)
yyZY = lsZY_highT(xxZY, Ebfix, T/1e3/EF, arb_scale=arbscale)

# residuals
yyres = y - lsMB_fixedEb(x, *popt1)
yyres2 = y - lsmom3_fixedEb(x, *popt2)
yyres3= y - gaussian(x, *popt3)
yyresZY = y - lsZY_highT(x, Ebfix, T/1e3/EF, arb_scale=arbscale)

# lineshape plot
fig, ax_ls = plt.subplots()
fig.suptitle('ac dimer spectrum at 202.14G, EF={:.1f} kHz, T/TF={:.2f}, T={:.1f} kHz'.format(EF, ToTF, ToTF*EF))
ax_ls.errorbar(x, y, yerr, marker='o', ls='', markersize = 12, capsize=3, mew=3, mec = adjust_lightness('tab:gray',0.2), color='tab:gray', elinewidth=3)
ax_ls.errorbar(xnfilt, ynfilt, yerrnfilt, marker='o', ls='', markersize = 12, capsize=3, mew=3, mfc='none', color='tab:gray', elinewidth=3)

fitstr = r'$A\sqrt{-\Delta-E_b}*exp(\frac{\Delta+E_b}{T}) *\Theta(-\Delta-E_b)$'
ax_ls.plot(xx, yy,'--', lw = linewidth, color='r', label='Mod. MB fit: ' + fitstr)
fitstr2 = r'$A (-\Delta-E_b)*exp(\frac{\Delta+E_b}{T}) *\Theta(-\Delta-E_b)$'
ax_ls.plot(xx, yy2, ':', lw = linewidth, color ='b', label = 'Mod. MB w/ collision mom.: ' + fitstr2)
ax_ls.plot(xx, yy3, '-.', lw=linewidth, color='k', label='Gaussian')
# T0str = r'$A(2k_F^3 - 3k_F^2 *\sqrt{-\omega - E_b} + \sqrt{-\omega - E_b}^3)\frac{\sqrt{-\omega-E_b}}{-\omega-E_b}$'
# ax_ls.plot(xx, yyT0, ls =':', color='g',label='T=0: ' + T0str)
ZYstr = r'$A * exp(\frac{\Delta + E_b}{T}) * (-\Delta - E_b)^{-1/2} *\Theta(-\Delta-E_b)$'
ax_ls.plot(xxZY, yyZY, '-', lw=linewidth, color='g', label='Eq. (49), arb. scale, ' + ZYstr)

textstr = '\n'.join((
 	r'Mod. MB fit params:',
 	r'Amplitude = {:.2f} +/- {:.2f}'.format(popt1[0], perr1[0]),
 	r'T = {:.2f} +/- {:.2f} EF'.format(popt1[1], perr1[1]),
	 r'Eb fixed at {:.1f} EF'.format(Ebfix)
 	))
ax_ls.text(xlow + 3, 0.015, textstr)

ax_ls.legend()
ax_ls.set_xlim([xlow, xhigh])
ax_ls.set_ylim([-0.01, 0.03])
ax_ls.set_xlim([-266, -259])
ax_ls.set_ylabel(r'Scaled transfer $\tilde{\Gamma}$ [arb.]')
ax_ls.set_xlabel(r'Detuning from 12-resonance $\Delta$ [EF]')

# how hard is it to put a second x-axis on this thing
# Put MHz frequencies on upper x-axis
f = lambda x: x * EF /1e3 
g = lambda x: x * EF/1e3 #wtf
ax2 = ax_ls.secondary_xaxis("top", functions=(f,g))
ax2.set_xlabel("Detuning [MHz]")

plt.tight_layout()

# residuals plots
fig, axs = plt.subplots(2,2)
ylims=[-0.015, 0.015]
ax1 = axs[0,0]
ax1.errorbar(x, yyres,yerr,  marker='o', color='r', mec=adjust_lightness('r'), capsize=2, mew=2, elinewidth=2)
ax1.set(title='Mod. MB', ylim=ylims)
ax2 = axs[0,1]
ax2.errorbar(x, yyres2, yerr, marker='o', color='b', mec=adjust_lightness('b'), capsize=2, mew=2, elinewidth=2)
ax2.set(title='Mod. k^3', ylim=ylims)
ax3 = axs[1,0]
ax3.errorbar(x, yyres3, yerr, marker='o', color='k', mec=adjust_lightness('k'), capsize=2, mew=2, elinewidth=2)
ax3.set(title='Gaussian', ylim=ylims)
ax4 = axs[1,1]
ax4.errorbar(x, yyresZY, yerr, marker='o', color='g', mec=adjust_lightness('g'), capsize=2, mew=2, elinewidth=2)
ax4.set(title='high-T', ylim=ylims)
fig.suptitle("Lineshape Residuals")
### save fig
# fig_file = os.path.join('figures', 'acdimerspectrum_fit.pdf')
# fig.savefig(os.path.join(proj_path, fig_file))

### time for clock shift analysis I guess
sumrule1 = np.trapz(lsMB_fixedEb(xx, *popt1), x=xx)
sumrule2 = np.trapz(lsmom3_fixedEb(xx, *popt2), x=xx)
sumrule3 = np.trapz(gaussian(xx, *popt3), x=xx)
print("sumrule mod. MB = {:.6f}".format(sumrule1))
print("sumrule mod. k^3 = {:.6f}".format(sumrule2))
print("sumrule gaussian = {:.6f}".format(sumrule3))

firstmoment1 = np.trapz(lsMB_fixedEb(xx, *popt1) * xx, x=xx)
firstmoment2 = np.trapz(lsmom3_fixedEb(xx, *popt2) * xx, x=xx)
firstmoment3 = np.trapz(gaussian(xx, *popt3) * xx, x=xx)
print("first moment mod. MB [EF] = {:.6f}".format(firstmoment1))
print("first moment mod. k^3 [EF] = {:.6f}".format(firstmoment2))
print("first moment gaussian [EF] = {:.6f}".format(firstmoment3))

# clock shifts
# experimental clockshift
HFTsumrule = 0.25 # approximately early July 
clockshift1 = firstmoment1/(sumrule1+HFTsumrule)
clockshift2 = firstmoment2/(sumrule2+HFTsumrule)
clockshift3 = firstmoment3/(sumrule3+HFTsumrule)
print("Clock shift mod. MB [EF]= {:.6f}".format(clockshift1))
print("Clock shift mod. k^3 [EF] = {:.6f}".format(clockshift2))
print("Clock shift gaussian [EF] = {:.6f}".format(clockshift3))

# ideal clockshift
clockshift1 = firstmoment1/0.5
clockshift2 = firstmoment2/0.5
clockshift3 = firstmoment3/0.5
print("Ideal clock shift mod. MB [EF]= {:.6f}".format(clockshift1))
print("Ideal clock shift mod. k^3 [EF] = {:.6f}".format(clockshift2))
print("Ideal clock shift gaussian [EF] = {:.6f}".format(clockshift3))

Ctilde_est = 1.44
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
	fit_func = GenerateSpectraFit(Ebfix)
	
	num_iter = 1000
	conf = 68.2689  # confidence level for CI
	
	# non-averaged data
	x = np.array(run.data['detuning'])
	num = len(x)
# 		print(x)
	y = np.array(run.data['ScaledTransfer'])
	
	# sumrule, first moment and clockshift with analytic extension
	SR_BS_dist, FM_BS_dist, CS_BS_dist, pFits, SR, FM, CS  = \
		DimerBootStrapFit(x, y, xfitlims, Ebfix, fit_func, trialsB=BOOTSRAP_TRAIL_NUM)
	# print(SRlineshape)
	# print(SR)
	# print(FMlineshape)
	# print(FM)
	SR_BS_mean, e_SR_BS = (np.mean(SR_BS_dist), np.std(SR_BS_dist))
	FM_BS_mean, e_FM_BS = (np.mean(FM_BS_dist), np.std(FM_BS_dist))
	CS_BS_mean, e_CS_BS = (np.mean(CS_BS_dist), np.std(CS_BS_dist))
	# SR_extrap_mean, e_SR_extrap = (np.mean(SR_extrap_dist), np.std(SR_extrap_dist))
	# FM_extrap_mean, e_FM_extrap = (np.mean(FM_extrap_dist), np.std(FM_extrap_dist))
	CS_BS_mean, e_CS_BS = (np.mean(CS_BS_dist), sem(CS_BS_dist))
	print(r"SR BS mean = {:.3f}$\pm$ {:.3f}".format(SR_BS_mean, e_SR_BS))
	print(r"FM BS mean = {:.3f}$\pm$ {:.3f}".format(FM_BS_mean, e_FM_BS))
	print(r"CS BS mean = {:.2f}$\pm$ {:.2f}".format(CS_BS_mean, e_CS_BS))
	median_SR = np.nanmedian(SR_BS_dist)
	upper_SR = np.nanpercentile(SR_BS_dist, 100-(100.0-conf)/2.)
	lower_SR = np.nanpercentile(SR_BS_dist, (100.0-conf)/2.)
	
	median_FM = np.nanmedian(FM_BS_dist)
	upper_FM = np.nanpercentile(FM_BS_dist, 100-(100.0-conf)/2.)
	lower_FM = np.nanpercentile(FM_BS_dist, (100.0-conf)/2.)
	
	median_CS = np.nanmedian(CS_BS_dist)
	upper_CS = np.nanpercentile(CS_BS_dist, 100-(100.0-conf)/2.)
	lower_CS = np.nanpercentile(CS_BS_dist, (100.0-conf)/2.)
	print(r"SR BS median = {:.3f}+{:.3f}-{:.3f}".format(median_SR,
												  upper_SR-SR, SR-lower_SR))
	print(r"FM BS median = {:.3f}+{:.3f}-{:.3f}".format(median_FM, 
												  upper_FM-FM, FM-lower_FM))
	print(r"CS BS median = {:.2f}+{:.3f}-{:.3f}".format(median_CS, 
												  upper_CS-CS, CS-lower_CS))


if (Bootstrapplots == True and Bootstrap == True):
	plt.rcParams.update({"figure.figsize": [10,8]})
	fig, axs = plt.subplots(2,2)
	fig.suptitle(filename)
	
	bins = 20
	
# fits
	
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

	
	# make room for suptitle
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])	
# %%	
### generate table
fig, axs = plt.subplots(2)
axpred = axs[0]
axpred.axis('off')
axpred.axis('tight')
# quantities = ["sumrule (red)", "1st moment (red)", "Clock shift (red)", \
# 			  "sumrule (blue)", "1st moment (blue)", "Clock shift (blue)", \
# 				  "sumrule (black)", "1st moment (black)", "Clock shift (black)"]
# values = ["{:.3f}".format(sumrule1),
# 		  "{:.3f}".format(firstmoment1),
# 		  "{:.3f}".format(clockshift1), 
# 		  "{:.3f}".format(sumrule2),
# 		  "{:.3f}".format(firstmoment2),
# 		  "{:.3f}".format(clockshift2),
# 		  "{:.3f}".format(sumrule3),
# 		  "{:.3f}".format(firstmoment3),
# 		  "{:.3f}".format(clockshift3)]
quantities = [r"$\Omega_d$ (zero range)",
			  r"$\Omega_+$ (zero range)", 
			  r"$\Omega_{tot}$ (zero range)", 
			  r"$\Omega_d$ (corr.)",
			  r"$\Omega_+$ (corr.)", 
			  r"$\Omega_{tot}$ (corr.)"]
values = ["{:.1f}".format(cs_pred), 
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
			  r"$\widebar{\Omega_d}$ (bootstrap)",
			  r"$\Omega_+$", 
			  r"$\Omega_{tot}$ (lineshape)",
			  r"$\Omega_{tot}$ (bootstrap)"]
# EXPERIMENTAL VALUES
HFT_CS_EXP = 5.77
HFT_CS_EXP = 4.8
mean_dimer_cs = (clockshift1 +clockshift2 + clockshift3)/3
values = [
		  "{:.1f}".format(mean_dimer_cs),
		  "{:.1f} +/- {:.1f}".format(CS_BS_mean,  e_CS_BS),
		  "{:.1f}".format(HFT_CS_EXP), 
		  "{:.1f}".format(mean_dimer_cs + HFT_CS_EXP),
		  "{:.1f}".format(CS_BS_mean + HFT_CS_EXP)]
table = list(zip(quantities, values))

the_table = axexp.table(cellText=table, loc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1,1.5)
axexp.set(title='Experimental clock shifts [EF]')

						
