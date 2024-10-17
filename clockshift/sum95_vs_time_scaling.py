# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:49:08 2024

@author: Chip Lab
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from data_class import Data
from scipy.optimize import curve_fit
from library import colors, markers, tintshade, tint_shade_color, plt_settings
from matplotlib.ticker import AutoMinorLocator


def pulsearea(x):
	return np.sqrt(0.31)*(x - 25.2) + (1)*(25.2)

pulseareanew = 1 # square pulse

def OmegaR(VVA):
	VpptoOmegaR = 27.5833 # kHz
	return  pulseareanew*VpptoOmegaR*calInterp(VVA)

def Linear(x,m,b):
	return m*x + b

def Quadratic(x, a, b):
	return a*x**2 + b

# paths
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')

# data file
data_file = os.path.join(data_path, 'VVAtoVpp_square_43p2MHz.txt')

# load data, plot
cal = pd.read_csv(data_file, sep='\t', skiprows=1, names=['VVA','Vpp'])
calInterp = lambda x: np.interp(x, cal['VVA'], cal['Vpp'])


xname = 'time'
yname = 'transfer'
cutoff_val = 0.15
cutoff_time = 55

plt.rcParams.update(plt_settings)
fig, axs = plt.subplots(2,2, figsize=[10,8])
ax = axs[0,0]
axr = axs[1,0]
axr2 = axs[1,1]

files = ['2024-09-26_F_e_VVA=2.dat',
 		 '2024-09-26_F_e_VVA=2.5.dat',
		 '2024-09-26_F_e_VVA=3.dat',
		 '2024-09-26_F_e_VVA=4.dat','2024-09-26_F_e_VVA=5.dat', 
 		 '2024-09-26_F_e_VVA=6.dat', '2024-09-26_F_e_VVA=7.dat',
 		 '2024-09-26_F_e_VVA=8.dat','2024-09-26_F_e_VVA=10.dat']
VVAs = [2,2.5,3,4,5,6,7,8,10]
slopes = []
e_slopes = []
Omega_R2 = []
chi2_lins = []
chi2_quads = []

for i, file in enumerate(files):
	
	data = Data(file)
	
	# compute OmegaR
	VVA = VVAs[i]
	data.data['OmegaR'] = OmegaR(data.data['VVA'])
	data.data['OmegaR2'] = data.data['OmegaR']**2
	Omega_R2.append(data.data['OmegaR2'].values[0])
		
	# compute bg point and transfer
	bgsum95 = data.data.loc[data.data[xname]==1]['sum95'].mean()
	data.data['transfer'] = (bgsum95 - data.data['sum95']) / bgsum95						
	data.group_by_mean(xname)
	
	# segment data into fit and not fit
	data.avg_data['cut'] = data.avg_data.apply(lambda x: \
							np.where(x[yname]<cutoff_val, 0, 1) + \
							np.where(x[xname]<cutoff_time, 0, 1), axis=1)
		
	fitdf = data.avg_data[data.avg_data.cut == 0]
	satdf = data.avg_data[data.avg_data.cut > 0]
	
	x = fitdf[xname]
	y = fitdf[yname]		
	yerr = fitdf['em_' + yname]
	xsat = satdf[xname]
	ysat = satdf[yname]
	yerrsat =satdf['em_' + yname]
	
	# for waterfall plotting
	y += 0.1*i
	ysat += 0.1*i
		
	# define colrs, and markers
	color = colors[i]
	marker = markers[i]
	light_color = tint_shade_color(color, amount=1+tintshade)
	dark_color = tint_shade_color(color, amount=1-tintshade)
	plt.rcParams.update({"lines.markeredgecolor": dark_color,
					 "lines.markerfacecolor": light_color,
					 "lines.color": dark_color})
	
	# fit data
	popt, pcov = curve_fit(Linear,x,y)
	popt2, pcov2 = curve_fit(Quadratic, x, y)
	slopes.append(popt[0])
	perr = np.sqrt(np.diag(pcov))
	e_slopes.append(perr[0])
	xlist = np.linspace(min(x),max(x),1000)
 	
	# plot fits
	ax.plot(xlist,Linear(xlist,*popt),linestyle='-',marker='', color=color)
	ax.plot(xlist, Quadratic(xlist, *popt2), linestyle='--', marker='', color=color)
	ax.set(xlim=[-2,62])
	ax.set_xlabel('Pulse duration [us]')
	ax.set_ylabel('Transfer [arb.]')

	# print results
	print(file)
	ax.hlines(i*0.1,-2,1, linestyle='--', color=color)
	print(r"m = {:.4f} ± {:.4f}, b = {:.4f} ± {:.4f}".format(popt[0], 
										  perr[0], popt[1]-0.1*i, perr[1]))
	
	# plot residuals
	yreslin = y - Linear(x, *popt)
	yresqua = y - Quadratic(x, *popt2)
	dof_lin = len(y) + 2
	dof_quad = len(y) + 2
	chi2lin = 1/dof_lin*np.sum(yreslin**2 / (yerr**2))
	chi2qua = 1/dof_quad*np.sum(yresqua**2 / (yerr**2))
	chi2_lins.append(chi2lin)
	chi2_quads.append(chi2qua)
	axr.errorbar(x, yreslin + 0.1*i, yerr=yerr)
	axr.hlines(i*0.1,0,50, linestyle='--', color=color)
	axr2.errorbar(x, yresqua + 0.1*i, yerr=yerr)
	axr2.hlines(i*0.1,0,50, linestyle='--', color=color)
	
	# make label
	label = r'$\chi_l^2$={:.2f}, $\chi_q^2$={:.2f}, VVA={}'.format(chi2lin, chi2qua, VVA)
	
	# plot data, with labels not used in legend
	ax.errorbar(x, y, yerr, label=label, ecolor=dark_color, marker=marker)
	ax.errorbar(xsat, ysat, yerrsat, ecolor=dark_color, marker=marker, mfc='white')
		
print("Average chi2 linear = {:.3f}±{:.3f}".format(np.array(chi2_lins).mean(), 
							   np.array(chi2_lins).std()/np.sqrt(np.size(chi2_lins))))
print("Average chi2 quad = {:.3f}±{:.3f}".format(np.array(chi2_quads).mean(), 
						 np.array(chi2_quads).std()/np.sqrt(np.size(chi2_quads))))

axr.set(xlabel='Pulse duration [us]', title='Residuals (linear)')
axr2.set(xlabel='Pulse duration [us]', title='Residuals (quadratic)')

axl = axs[0,1]
axl.axis('off')
h, l = ax.get_legend_handles_labels()
axl.legend(h, l)
fig.tight_layout()
plt.show()

### VVA scaling

fig, axs = plt.subplots(1,2, figsize=[8,4])
ylabel = "Transfer rate rate"
ylims = [-0.002, 1.1*max(slopes)]

ax = axs[0]
x = np.sqrt(Omega_R2)
y = slopes
ax.errorbar(x, y, yerr=e_slopes)

# wtf matplotlib
forward = lambda omega: np.interp(omega, x, VVAs)
inverse = lambda vva: np.interp(vva, VVAs, x)
secax = ax.secondary_xaxis('top', functions=(forward, inverse))
# secax.xaxis.set_minor_locator(AutoMinorLocator())
secax.set_xlabel('$VVA$')

ax.set(xlim=[0,1.1*max(x)], ylim=ylims, xlabel=r"$\Omega_R$", ylabel=ylabel)

popt, pcov = curve_fit(Linear, x, y)
perr = np.sqrt(np.diag(pcov))
print(r"m = {:.5f} ± {:.5f}, b = {:.4f} ± {:.4f}".format(popt[0], 
										  perr[0], popt[1], perr[1]))
xs = np.linspace(0, max(x), 100)
ax.plot(xs, Linear(xs, *popt), '--')

ax = axs[1]
x2 = Omega_R2
ax.errorbar(x2, y, yerr=e_slopes)


forward2 = lambda omega2: np.interp(omega2, x2, VVAs)
inverse2 = lambda vva: np.interp(vva, VVAs, x2)
secax = ax.secondary_xaxis('top', functions=(forward2, inverse2))
# secax.xaxis.set_minor_locator(AutoMinorLocator())
secax.set_xlabel('$VVA$')

ax.set(xlim=[0,1.1*max(x2)], ylim=ylims, xlabel=r"$\Omega_R^2$", ylabel=ylabel)

popt, pcov = curve_fit(Linear, x2[:-3], y[:-3])
perr = np.sqrt(np.diag(pcov))
print(r"m = {:.6f} ± {:.6f}, b = {:.4f} ± {:.4f}".format(popt[0], 
										  perr[0], popt[1], perr[1]))
xs = np.linspace(0, max(x2), 100)
ax.plot(xs, Linear(xs, *popt), '--')

fig.tight_layout()
plt.show()


