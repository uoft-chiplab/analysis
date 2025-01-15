# -*- coding: utf-8 -*-
"""
Created by Chip lab 2024-06-12

Loads .dat with contact HFT scan and computes scaled transfer. Plots. Also
computes the sumrule.
"""

from library import pi, h, plt_settings, GammaTilde
from data_class import Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq

# filename = "2024-06-12_K_e.dat"
filename = "2024-06-18_G_e.dat"
# filename = "2024-06-20_C_e.dat" # reminder; had to kill 0 detuning because of scatter
# filename = "2024-06-20_D_e.dat" # reminder; had to kill 0 detuning because of scatter
# filename = "2024-06-21_F_e.dat"

VpptoOmegaR47 = 17.05/0.703 # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
VpptoOmegaR43 = 14.44/0.656 # kHz/Vpp - 2024-09-25 calibration 
phaseO_OmegaR = lambda VVA, freq: 2*pi*VpptoOmegaR47 * Vpp_from_VVAfreq(VVA, freq)

# def VVAtoVpp(VVA):
# 	"""Match VVAs to calibration file values. Will get mad if the VVA is not
# 		also the file. """
# 	for i, VVA_val in enumerate(VVAs):
# 		if VVA == VVA_val:
# 			Vpp = Vpps[i]
# 	return Vpp

def trapz_w_error(xs, ys, yserr, num_iter=1000):
	""" Computes trapz for list of data points (xs, ys+-yserr),
	and estimates std dev of result by sampling ys and yserr from 
	Gaussian distributions, num_iter (default 1000) times."""
# 	value = np.trapz(ys, x=xs)
	def rand_y(y, yerr, size):
		generator = np.random.default_rng()
		return generator.normal(loc=y, scale=yerr, size=num_iter)
	# array of lists of y values, sampled from Gaussians with centres y and widths yerr
	ys_iter = np.array([rand_y(y, yerr, num_iter) for y, yerr in zip(ys, yserr)])
	values = np.array([np.trapz(ys_iter[:,i], x=xs) for i in range(num_iter)])
	distr_mean, distr_stdev = (np.mean(values), np.std(values))
	return values, distr_mean, distr_stdev


def trapz_interp_w_error(xs, ys, yserr, num_iter=1000):
	""" Computes trapz for interpolated list of data points (xs, ys+-yserr),
	and estimates std dev of result by sampling ys and yserr from 
	Gaussian distributions, num_iter (default 1000) times."""
# 	value = np.trapz(ys, x=xs)
	def rand_y(y, yerr, size):
		generator = np.random.default_rng()
		return generator.normal(loc=y, scale=yerr, size=num_iter)
	# array of lists of y values, sampled from Gaussians with centres y and widths yerr
	ys_iter = np.array([rand_y(y, yerr, num_iter) for y, yerr in zip(ys, yserr)])
	
	# interpolation array for x, num_iter in size
	xs_interp = np.linspace(min(xs), max(xs), num_iter)
	
	# compute interpolation array for y, num_iter by num_iter in size
	ys_interp_iter = np.array([[np.interp(xi, xs, ys_iter[:,i]) for xi in xs_interp]
					  for i in range(num_iter)])
	
	# integrals using each interpolation set
	values = np.array([np.trapz(ys_interp_iter[i], x=xs_interp) for i in range(num_iter)])
	
	distr_mean, distr_stdev = (np.mean(values), np.std(values))
	return values, distr_mean, distr_stdev

### run params
xname = 'freq'
ff = 1.03
trf = 200e-6  # 200 or 400 us
EF = 16e-3 #MHz
bg_freq = 47  # chosen freq for bg, large negative detuning
res_freq = 47.2159 # for 202.1G
pulsetype = 'Kaiser'
pulse_area = 0.3 # Blackman
pulse_area = np.sqrt(0.3*0.92) # maybe? for first 4
# pulse_area=np.sqrt(0.3) # if using real Blackman
gain = 0.2 # scales the VVA to Vpp tabulation

### create data structure
run = Data(filename)
# kill a point
run.data.drop([77], inplace=True)
num = len(run.data[xname])

### compute bg c5, transfer, Rabi freq, etc.
bgc5 = run.data[run.data[xname]==bg_freq]['c5'].mean()
run.data['N'] = run.data['c5']-bgc5*np.ones(num)+run.data['c9']*ff
run.data['transfer'] = (run.data['c5'] - bgc5*np.ones(num))/run.data['N']
run.data['detuning'] = run.data[xname] - res_freq*np.ones(num) # MHz
try:
	run.data['Vpp'] = run.data['vva'].apply(VVAtoVpp)
except KeyError:
	run.data['Vpp'] = run.data['VVA'].apply(VVAtoVpp)
run.data['OmegaR'] = 2*pi*pulse_area*gain*VpptoOmegaR*run.data['Vpp']

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
TransferInterpFunc = lambda x: np.interp(x, xp, fp)

### PLOTTING
plt.rcParams.update(plt_settings) # from library.py
plt.rcParams.update({"figure.figsize": [8,8],
					 "font.size": 14})
fig, axs = plt.subplots(2,2)

xlabel = r"Detuning $\omega_{rf}-\omega_{res}$ (MHz)"

### plot transfer fraction
ax = axs[0,0]
x = run.avg_data['detuning']
y = run.avg_data['transfer']
yerr = run.avg_data['em_transfer']
ylabel = r"Transfer $\Gamma \,t_{rf}$"

xlims = [-0.03,0.25]

ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims)
ax.errorbar(x, y, yerr=yerr, fmt='o')
# ax.plot(x, y, 'o-')

### plot scaled transfer
ax = axs[1,0]
x = run.avg_data['detuning']/EF
y = run.avg_data['ScaledTransfer']
yerr = run.avg_data['em_ScaledTransfer']
xlabel = r"Detuning $\Delta$"
ylabel = r"Scaled Transfer $\tilde\Gamma$"

xlims = [-2,16]
axxlims = [-2,16]
ylims = [run.data['ScaledTransfer'].min()*10, run.data['ScaledTransfer'].max()]
num = len(y)
num = 10000
xs = np.linspace(xlims[0], xlims[-1], num)

ax.set(xlabel=xlabel, ylabel=ylabel, xlim=axxlims, ylim=ylims)
ax.errorbar(x, y, yerr=yerr, fmt='o')
ax.plot(xs, TransferInterpFunc(xs), '-')

sumrule_interp = np.trapz(TransferInterpFunc(xs), x=xs)
sumrule_distr, sumrule_mean, sumrule_std = trapz_w_error(x, y, yerr, num)
sumrule_distr_interp, sumrule_mean_interp, sumrule_std_interp = trapz_interp_w_error(x, y, yerr, num)
sumrule = np.trapz(y, x=x)
print("interpolated sumrule = {:.3f}".format(sumrule_interp))
print("raw data sumrule = {:.3f}".format(sumrule))
print("sumrule mean = {:.3f}\pm {:.3f}".format(sumrule_mean, sumrule_std))
print("sumrule interp mean = {:.3f}\pm {:.3f}".format(sumrule_mean_interp, sumrule_std_interp))

### plot contact
ax = axs[0,1]
x = run.avg_data['detuning']/EF
y = run.avg_data['C']
yerr = run.avg_data['em_C']
xlabel = r"Detuning $\Delta$"
ylabel = r"Contact $C/N$ [$k_F$]"

xlims = [-2,16]
ylims = [run.data['C'].min(), run.data['C'].max()]
Cdetmin = 3
Cdetmax = 8
xs = np.linspace(Cdetmin, Cdetmax, num)

df = run.data[run.data.detuning/EF>Cdetmin]
Cmean = df[df.detuning/EF<Cdetmax].C.mean()

ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims, ylim=ylims)
ax.errorbar(x, y, yerr=yerr, fmt='o')
ax.plot(xs, Cmean*np.ones(num), "--")

### generate table
ax = axs[1,1]
ax.axis('off')
ax.axis('tight')
quantities = ["$E_F$", "Contact $C/N$", "sumrule"]
values = ["{:.1f} kHz".format(EF*1e3), 
		  "{:.2f} kF".format(Cmean), 
		  "{:.3f}".format(sumrule)]
table = list(zip(quantities, values))

the_table = ax.table(cellText=table, loc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1,1.5)

plt.show()
