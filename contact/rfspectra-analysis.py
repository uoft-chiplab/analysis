# -*- coding: utf-8 -*-
# %% 
# 2024-06-10 
# rf spectra and scaled loss analysis, 
# previously analyze_rf_spectra_2024-06-07.nb 
# author chip lab 

from library import * 
import os 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

### name and xlsx file # 
name = '2023-08-03_B.dat'
indexnum =10
upperlim = 47.2
kHz = 10**3

### files and paths
data_folder = 'rfspectra'
file_path = os.path.join(data_folder, name)
xlsxfile = 'ContactMeasurementsSummary.xlsx'

### creating data frame
df = pd.read_excel(xlsxfile, index_col=0, engine='openpyxl')

### choosing which file to grab from the xlsx file 
index = df.iloc[indexnum]
print(index) 
# %%
### grabbing specific columns from data file
metadata = pd.read_table(file_path, delimiter=',')
c9 = np.array(metadata['c9']) #loss
c5 = np.array(metadata['c5']) #probe
f95 = np.array(metadata['fraction95'])
freq = np.array(metadata['freq'])
# %%
### grabbing all variables from xlsx file 
field = index.loc['field']
ff = index.loc['ff']
rightbg = index.loc['right_bg']
pulse = index.loc['pulse']
trf = index.loc['time']
VVA = index.loc['VVA']
TShotsN = index.loc['TShotsN']
trapfreqsx = float(index.loc['trapfreqs'][1:6])/1000
trapfreqsy = float(index.loc['trapfreqs'][7:10])/1000
trapfreqsz = float(index.loc['trapfreqs'][11:14])/1000
trapfreqs = [trapfreqsx,trapfreqsy,trapfreqsz]
OmegaR = index.loc['Omega_R']

### finding the bg value of the loss 
bg =  np.average(c5, weights=freq<upperlim)

transfer = (c5 - bg )/ ((c5 - bg) + c9*ff)

if index.loc['pulse'] == 'blackman':
    pulsearea = np.sqrt(0.3)
else:
    pulsearea = 1

res= FreqMHz(field, 9/2,-5/2,9/2,-7/2)
EF = (trapfreqsy*trapfreqsx*trapfreqsz*6*TShotsN/2)**(1/3)

resfreq = (freq-res)*kHz

### constant VVA OmegaR
scale=10**3
OmegaRmax = 2*np.pi*pulsearea*OmegaR
# %%
### plotting transfer vs freq
fig, ax = plt.subplots(2,2, figsize=(18,12))

ax[0,0].set_xlabel('rf freqeuncy (kHz)')
ax[0,0].set_ylabel('Probe (c5)')
ax[0,0].plot(resfreq,c5,linestyle='',marker='.')

### plotting loss vs freq

ax[0,1].set_xlabel('rf freqeuncy (kHz)')
ax[0,1].set_ylabel('Loss (c9)')
ax[0,1].plot(resfreq,c9,linestyle='',marker='.')

### plotting transfer vs freq 

ax[1,0].set_xlabel('rf freqeuncy (kHz)')
ax[1,0].set_ylabel('Transfer')
ax[1,0].plot(resfreq,transfer,linestyle='',marker='.')
# %%
### fitting power law

### max and min where we want to fit  
minf = 30.0215 #1.5*EF
maxf = 110.022 #6*EF 
yvals = transfer

### filter mask, i.e. truncating the freq list to fit only 
### the -3/2 tail 
freqfiltermask = resfreq > minf 
newresfreq, newyvals = resfreq[freqfiltermask], yvals[freqfiltermask]
freqfiltermask = newresfreq < maxf
truncatedresfreq, truncatedyvals = newresfreq[freqfiltermask], newyvals[freqfiltermask]

### Power Law  
def fitPowerLaw(x,A,B):
    return A*x**B

### popt, pcov, perr of power law
guess = [100,-1.5]
popt, pcov = curve_fit(fitPowerLaw,truncatedresfreq,truncatedyvals,p0=guess)
perr = np.sqrt(np.diag(pcov))
# %%
### fixing the power law to -3/2 
def fixedPowerLaw(x,A):
    return A*x**(-3/2)
    
guess = [100]
poptfixed, pcovfixed = curve_fit(fixedPowerLaw,truncatedresfreq,truncatedyvals,p0=guess)

fitpowerlawresult = str(float("{:.2f}".format(popt[1])))
fitpowerlawerror = str(np.round(np.sqrt(np.diag(pcov))[1],3))
# %%
### plotting rf spectra and residuals as inset

### finding residuals for the area we chose to fit 
xlist = np.linspace(truncatedresfreq.min(),truncatedresfreq.max(),len(truncatedyvals))
residuals = truncatedyvals - fitPowerLaw(xlist,*popt)

### large plot 
fig, ax1 = plt.subplots()

ax1.plot(resfreq,yvals, linestyle='',marker='.')
ax1.plot(xlist,fitPowerLaw(xlist,*popt),linestyle='-', label='Fit Power Law: '+fitpowerlawresult+'pm'+fitpowerlawerror)
ax1.plot(xlist,fixedPowerLaw(xlist,*poptfixed),linestyle='-',label='Fixed Power Law: -1.5')
ax1.set_xlabel('rf spectra (kHz)')
ax1.set_ylabel('transfer fraction')
ax1.legend()

### insert 
###size and place of inset
#left 0 is on y axis, bottom 0 is on x axis
#width, height is size of inset 
left, bottom, width, height = [0.6, 0.5, 0.3, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.set_ylabel('residuals')
ax2.plot(xlist,residuals,linestyle='',marker='.')

print(f'A: {popt[0]}')
print(f'range fit over: {minf,maxf}')
# %% 
### plotting the data on a loglog plot
fig, ax = plt.subplots()

ax.set_xlabel('log(rf frequency)')
ax.set_ylabel('log(transfer fraction)')

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(20,150)
ax.set_ylim(.0015,.15)
ax.plot(resfreq,yvals,linestyle='',marker='.')
ax.plot(xlist,fixedPowerLaw(xlist,*poptfixed),linestyle='-',label='Fixed Power Law: -1.5')
ax.plot(xlist,fitPowerLaw(xlist,*popt),linestyle='-',label='Fit Power Law: '+fitpowerlawresult+'pm'+fitpowerlawerror)

plt.legend()
# %%
### scale data to GammaTilde
def GammaTilde(x):
    return (EF*h*scale)/(hbar*np.pi*OmegaRmax**2*trf)*x

scaledtransfer = resfreq/EF

### max and min where we want to fit  
minfscaled = 1.5
maxfscaled = 5.7
yvalsscaled = GammaTilde(transfer)

### filter mask, i.e. truncating the freq list to fit only 
### the -3/2 tail 
freqfiltermaskscaled = scaledtransfer > minfscaled 
newresfreqscaled, newyvalsscaled = scaledtransfer[freqfiltermaskscaled], yvalsscaled[freqfiltermaskscaled]
freqfiltermaskscaled = newresfreqscaled < maxfscaled
truncatedresfreqscaled, truncatedyvalsscaled = newresfreqscaled[freqfiltermaskscaled], newyvalsscaled[freqfiltermaskscaled]

### fit power law for scaled transfer
guess = [100,-1.5]
poptscaled, pcovscaled = curve_fit(fitPowerLaw,truncatedresfreqscaled,truncatedyvalsscaled,p0=guess)
perrscaled = np.sqrt(np.diag(pcovscaled))

### fixed power law scaled transfer
guess = [100]
poptfixedscaled, pcovfixedscaled = curve_fit(fixedPowerLaw,truncatedresfreqscaled,truncatedyvalsscaled,p0=guess)

fitpowerlawresultscaled = str(float("{:.2f}".format(poptscaled[1])))
fitpowerlawerrorscaled = str(np.round(np.sqrt(np.diag(pcovscaled))[1],3))

### finding residuals for the area we chose to fit 
xlistscaled = np.linspace(truncatedresfreqscaled.min(),truncatedresfreqscaled.max(),len(truncatedyvalsscaled))
residualsscaled = truncatedyvalsscaled - fitPowerLaw(xlistscaled,*poptscaled)

# %%
### plot GammaTilde
fig, ax = plt.subplots()

ax.set_xlabel('rf frequency $\Delta$ (EF)')
plt.ylabel('Scaled Transfer 'r'$\tilde{\Gamma}$/N')
ax.plot(scaledtransfer,GammaTilde(transfer),marker='.',linestyle='')
ax.plot(xlistscaled,fitPowerLaw(xlistscaled,*poptscaled),label='Fixed Power Law: -1.5')
ax.plot(xlistscaled,fixedPowerLaw(xlistscaled,*poptfixedscaled),label='Fit Power Law: '+fitpowerlawresultscaled+'pm'+fitpowerlawerrorscaled)

plt.legend()

### insert 

###size and place of inset
#left 0 is on y axis, bottom 0 is on x axis
#width, height is size of inset 
left, bottom, width, height = [0.58, 0.5, 0.3, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])

ax2.set_ylabel('residuals')
ax2.plot(xlistscaled,residualsscaled,linestyle='',marker='.')


print(f'EF is {EF:.3f}')
print(f'Range: {minfscaled,maxfscaled}')
# %% 
### finding contact 
Cscaled = GammaTilde(transfer)*np.abs(scaledtransfer)**(3/2)*np.pi**2*2**(3/2)

# %%
### plotting contact 
plt.ylabel(r'$2\sqrt{2}\pi^2\tilde{\Gamma}\Delta^{-3/2}$')
plt.xlabel('rf frequency $\Delta$ (EF)')
plt.plot(scaledtransfer,Cscaled,linestyle='',marker='.')
# %% 
### plotting scaled loss (?)
fig, ax = plt.subplots()

plt.xlabel('rf freqeuncy (kHz)')
# plt.ylabel('Scaled Loss')
plt.plot(resfreq,-c9+bg,linestyle='',marker='.',label='-loss+offset')
plt.plot(resfreq,c5,linestyle='',marker='.',label='Transfer')
plt.legend()
# %%
### putting a fit on the scaled loss 
fig, ax = plt.subplots()

plt.xlabel('rf frequency (kHz)')
plt.plot(resfreq,c5,linestyle='',marker='.',label='Transfer')
ax.plot(xlist,fitPowerLaw(xlist,*popt),linestyle='',marker='.')

### fitting scaled loss
minf = 1.5*EF
maxf = 6*EF  
scaledloss = -c9+bg

freqfiltermask = resfreq > minf 
newresfreq, newscaledloss = resfreq[freqfiltermask], scaledloss[freqfiltermask]
freqfiltermask = newresfreq < maxf
truncatedresfreq, truncatedscaledloss = newresfreq[freqfiltermask], newscaledloss[freqfiltermask]

plt.plot(resfreq,scaledloss,linestyle='',marker='.',label='-loss+offset')

guess = [100,-1.5]
poptscaledloss, pcovscaledloss = curve_fit(fitPowerLaw,truncatedresfreq,truncatedscaledloss,p0=guess)
perrscaledloss = np.sqrt(np.diag(pcovscaledloss))

xlist = np.linspace(truncatedresfreq.min(),truncatedresfreq.max(),1000)
ax.plot(xlist,fitPowerLaw(xlist,*poptscaledloss),linestyle='',marker='.')

plt.plot([],[],linestyle='',label='1.5EF to 6EF')

plt.legend()

print(f'Power law for transfer is {popt[1]:.4f}\pm{perr[1]:.3}')
print(f'Power law for scaled loss is {poptscaledloss[1]:.4f}\pm{perrscaledloss[1]:.3}')
# %%
