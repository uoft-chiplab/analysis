# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:19:19 2024

@author: coldatoms
"""
# paths
import os
import sys
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
if root not in sys.path:
	sys.path.insert(0, root)

from data_class import Data
from fit_functions import Sinc2, Parabola, Sin
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize": [5,3.5]})

scan = 'FB'
FB_val = 7.05
file = "2025-08-12_I_e.dat"

fit_func = Sinc2

if scan == 'FB':
	names = ['FB', 'fraction95']
	guess = [0.5, FB_val, 0.1, 0]
elif scan == 'VVA':
	names = ['VVA', 'fraction95']
	guess = None

elif scan == 'nopulses':
	fit_func = Sin
	names = ['VVA', 'c9']
	guess = [30000, 2,0,15000]
elif scan == 'grad':
	fit_func = Parabola
	names = ['grad', 'sum95']
	guess = None

print("--------------------------")
Data(file).fit(fit_func, names, guess=guess)

def sinc2(x, A, x0, sigma, C):
		return np.abs(A)*(np.sinc((x-x0) / sigma)**2) + C
from scipy.optimize import curve_fit
from library import styles
import numpy as np
run = Data(file)
run.group_by_mean(scan)
df = run.data
dfavg = run.avg_data
fig, ax = plt.subplots(2,3, figsize=(14,5))

ax[0,0].errorbar(dfavg[scan], -dfavg['c9'], yerr = dfavg['em_c9'], **styles[0])
popt, pcov = curve_fit(sinc2, dfavg[scan], -dfavg['c9'], sigma=dfavg['em_c9'], p0= [5000, 7.045, 0.01, -df['c9'].min()])
xs=np.linspace(dfavg[scan].min(), dfavg[scan].max(), 100)
ys = sinc2(xs, *popt)
ax[0,0].plot(xs, ys, '-', color='orange')
ax[0,0].set(title=file, xlabel=scan, ylabel='-c9')

ax[1,0].plot(dfavg[scan], dfavg['sum95'],**styles[0], label='sum95')
ax[1,0].plot(dfavg[scan], dfavg['ROIsum'],**styles[1],label='ROIsum')
ax[1,0].legend()

#picking out and plotting the high scatter data based on a threshold 
    #of some error in c9 
c9errorthres = 2000
# ax[0,1].errorbar(dfavg[dfavg['em_c9']>c9errorthres][scan], 
# 				 -dfavg[dfavg['em_c9']>c9errorthres]['c9'], 
# 				yerr = dfavg[dfavg['em_c9']>c9errorthres]['em_c9'], **styles[0])
# ax[1,1].plot(dfavg[dfavg['em_c9']>c9errorthres][scan], 
# 			 dfavg[dfavg['em_c9']>c9errorthres]['sum95']
# 			 ,**styles[0], label='sum95')
# ax[1,1].plot(dfavg[dfavg['em_c9']>c9errorthres][scan], 
# 			 dfavg[dfavg['em_c9']>c9errorthres]['ROIsum']
# 			 ,**styles[1],label='ROIsum')
# ax[0,1].set(
	# title = 'averaged'
# )
#grabbing FB values associated with the high scatter points so I can plot the non
    #avg'd data
highscatter = df[df["FB"].isin(dfavg[dfavg["em_c9"] > c9errorthres]['FB'])]

# ax[0,2].plot(highscatter[scan], -highscatter['c9'], **styles[0])
# ax[1,2].plot(highscatter[scan], highscatter['sum95'],**styles[0], label='sum95')
# ax[1,2].plot(highscatter[scan], highscatter['ROIsum'],**styles[1],label='ROIsum')
# ax[0,2].set(
# 	# title = 'not averaged'
# )

yhigh = 27000 
ylow = 23000

dfcutoff = df[(df['sum95'] > ylow) & (df['sum95']<yhigh)]

ax[0,1].plot(df[scan], -df['c9'], **styles[0])
ax[0,1].plot(dfcutoff[scan], -dfcutoff['c9'], **styles[2])

ax[1,1].plot(df[scan], df['sum95'],**styles[0])
ax[1,1].plot(dfcutoff[scan], dfcutoff['sum95'],**styles[2], label='pts within cut off')
# ax[1,1].plot(df[scan], df['ROIsum'],**styles[1],label='ROIsum')
ax[1,1].legend()

dfavgcutoff = dfcutoff.groupby('FB').mean().reset_index()
dfavgcuffoff_errors = dfcutoff.groupby('FB').sem().reset_index().add_prefix('em_')
ax[0,2].errorbar(dfavgcutoff[scan], -dfavgcutoff['c9'],
				 yerr = dfavgcuffoff_errors['em_c9'], **styles[2])
popt2, pcov2 = curve_fit(sinc2, dfavgcutoff[scan], -dfavgcutoff['c9'], 
					   p0= [5000, 7.045, 0.01, -dfavgcutoff['c9'].min()])
xs=np.linspace(dfavgcutoff[scan].min(), dfavgcutoff[scan].max(), 100)
ys = sinc2(xs, *popt2)
ax[0,2].plot(xs, ys, '-', color='orange')
ax[0,2].set(
	title = 'Points within cutoff fit'
)

ax[1,2].text(0.1,.8,f'cut off range is {ylow} to {yhigh}')

fig.tight_layout()
print(popt)
print(np.sqrt(np.diag(pcov)))