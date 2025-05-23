# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 10:39:37 2025

@author: Chip Lab
"""

import numpy as np
from data_class import Data
from library import styles, colors
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

subtract_dark_image = False

def sigmoid(x, A, B, C):
  return A / (1 + np.exp(-(x-B)/C))

def linear(x, A, B):
	return A*x + B

def Scott(x, A, B):
	return A/(1+x/B)

xname = 'K AM'

###
### dark images
###

# dark file
dark_file = "2025-04-11_A.dat"

dark_df = Data(dark_file).data
dark_ref_m_bg = dark_df['refmean'].mean()
e_dark_ref_m_bg = dark_df['refmean'].sem()
dark_at_m_bg = dark_df['atmean'].mean()
e_dark_at_m_bg = dark_df['atmean'].sem()

print("dark ref minus bg", dark_ref_m_bg)
print("dark atom minus bg", dark_at_m_bg)

###
### scan light not atoms
###

# scan file with varying light no atoms
light_file = "2025-04-10_E_e.dat"

run = Data(light_file)
run.group_by_mean(xname)
df = run.avg_data

df['refmean_corrected'] = df["refmean"] - dark_ref_m_bg
df['em_refmean_corrected'] = np.sqrt(df['em_refmean']**2+e_dark_ref_m_bg**2)

refmean_name = 'refmean'

if subtract_dark_image:
	refmean_name = 'refmean_corrected'

# plot

fig, axes = plt.subplots(2,2, figsize=(8,8))
axs = axes.flatten()
fig.suptitle(light_file)

xp = df[xname]
fp = df[refmean_name]
e_fp = yerr=df['em_' + refmean_name]

# this is the calibration used to find K AM need for given light count
refmean_to_KAM = lambda x: np.interp(x, fp, xp)
KAM_to_refmean = lambda x: np.interp(x, xp, fp)

p0 = [3200, 0.82, 0.04]
popt, pcov = curve_fit(sigmoid, xp, fp, p0=p0)#, sigma=e_fp)

xs = np.linspace(min(xp), max(xp), 100)

# mean counts vs K AM
ax = axs[0]
ax.errorbar(xp, fp, yerr=e_fp, **styles[0], label='ref')
ax.plot(xs, sigmoid(xs, *popt), color=colors[0], marker='', linestyle='-', 
		label='sigmoid fit')
# ax.plot(xs, sigmoid(xs, *p0), color=colors[4], marker='', linestyle='-')
ax.errorbar(xp, df['atmean'], yerr=df['em_atmean'], **styles[1], label='at')
xs = np.linspace(min(xp), max(xp), 100)
ax.plot(xs, KAM_to_refmean(xs), '--', color=colors[0], label='interp')
ax.set(xlabel=xname, ylabel="mean counts")
ax.legend()

# atom - ref vs K AM
ax = axs[1]
ax.plot(xp, df['atmean']-df['refmean'], **styles[0])
ax.set(xlabel=xname, ylabel="atom - ref")

# K AM vs mean counts

ax = axs[2]
ax.errorbar(fp, xp, xerr=e_fp, **styles[0])
fs = np.linspace(min(fp), max(fp), 100)
ax.plot(fs, refmean_to_KAM(fs), '--', color=colors[0], label='interp')
ax.set(xlabel="mean counts", ylabel=xname)
ax.legend()

### Calculate scan list for fixed light intensity, variable time
times = np.linspace(0.01, 0.09, 9) # us 
max_time = 0.09

KAMs = refmean_to_KAM(max(fp)*times/max_time)

scanlist = np.array([times, KAMs]).transpose()
np.savetxt("scanlist times KAMs.txt", scanlist, delimiter=',', fmt='%f')

ax = axs[3]
ax.plot(times, KAMs, **styles[0])
ax.set(xlabel="time [ms]", ylabel='K AM')


fig.tight_layout()
plt.show()

###
### Blank file
###

# blanks file
blank_file = "2025-04-11_B_e.dat"

run = Data(blank_file)
blank_refmean = run.data.refmean.mean()
blank_ODmean = run.data.ODmean.mean()
	
###
### Sat scan with atoms
###

# file with atoms
file = "2025-04-11_D_e.dat"

run = Data(file)
run.group_by_mean('KAM')
df = run.data
df = df.loc[df['KAM'] > 0.7]

fit_df = df.loc[df.refmean>1500]

x = fit_df['refmean']
y = fit_df['ODmean']

fig, axes = plt.subplots(1,2, figsize=(8,4))
axs = axes.flatten()

fig.suptitle(file)

ax = axs[0]
ax.plot(df['refmean'], df['ODmean'], **styles[0])
ax.set(xlabel="refmean", ylabel="ODmean")#, ylim=[-0.01,0.06])

fit_func = [Scott, linear]

for i, fit in enumerate(fit_func):

	popt, pcov = curve_fit(fit, x, y)
	perr = np.sqrt(np.diag(pcov))
	xs = np.linspace(0, max(x), 100)
	
	print("full ROI correction is ", fit(0, *popt)/fit(2000, *popt))

	if fit == Scott:
		label = "I_0 = {:.0f}({:.0f})".format(popt[1], perr[1])
	else:
		label = None
	ax.plot(xs, fit(xs, *popt), '--', color=colors[i], label=label)

ax.legend()

ax = axs[1]
ax.plot(df['refmean'] - df['atmean'], df['ODmean'], **styles[0])
ax.set(xlabel="ref - at", ylabel="ODmean")

fig.tight_layout()

# -log(at - bg / ref - bg)

plt.show()





