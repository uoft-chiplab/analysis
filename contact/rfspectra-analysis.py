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
import ast

### name and xlsx file # 
name = '2023-08-08_F.dat'
indexnum = 12
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
TShotsN = index.loc['TShotsN']
trapfreqsx = float(index['trapfreqs'].strip('{}').split(',')[0])/1000
trapfreqsy = float(index['trapfreqs'].strip('{}').split(',')[1])/1000
trapfreqsz = float(index['trapfreqs'].strip('{}').split(',')[2])/1000
trapfreqs = [trapfreqsx,trapfreqsy,trapfreqsz]

### pulse area Blackman or square 
if index.loc['pulse'] == 'blackman':
    pulsearea = np.sqrt(0.3)
else:
    pulsearea = 1

### varying vva or not
if index.loc['VVA'] == 'varying':
    VVArule = index['VVArule'].strip('{}').strip('.').split(',') 
    VVAdata = np.array(metadata['vva'])
    OmegaR = dict(zip(VVArule[::2], VVArule[1::2]))
    OmegaR_cleaned = {key.strip() + '.0' if key.strip() in ['2', '3', '4', '5', '8', '10'] else key.strip(): float(value.strip()) for key, value in OmegaR.items()}
    OmegaRlist = []
    for x in VVAdata:
        OmegaRlist.append(2*np.pi*pulsearea*OmegaR_cleaned[str(x)])
        OmegaRmax = (OmegaRlist)
else:
    VVA = index.loc['VVA']
    OmegaR = index.loc['Omega_R']
    OmegaRmax = 2*np.pi*pulsearea*OmegaR

### finding the bg value of the loss 
bg =  np.average(c5, weights=freq<upperlim)
bgloss = np.average(c9, weights=freq<upperlim)

### transfer fraction 
transfer = (c5 - bg )/ ((c5 - bg) + c9*ff)

res= FreqMHz(field, 9/2,-5/2,9/2,-7/2)
EF = (trapfreqsy*trapfreqsx*trapfreqsz*6*TShotsN/2)**(1/3)
resfreq = (freq-res)*kHz
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

ax[1,1].text(0.5,0.5,name,fontsize=18)
# %%
### fitting power law

### max and min where we want to fit  
minf = 1.5*EF
maxf = 6*EF 
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
fig, (ax1,ax3) = plt.subplots(1,2,figsize=(20,10))

ax1.plot(resfreq,yvals, linestyle='',marker='.')
ax1.plot(xlist,fitPowerLaw(xlist,*popt),linestyle='-', label='Fit Power Law: '+fitpowerlawresult+'pm'+fitpowerlawerror)
ax1.plot(xlist,fixedPowerLaw(xlist,*poptfixed),linestyle='-',label='Fixed Power Law: -1.5')
ax1.set_xlabel('rf spectra (kHz)')
ax1.set_ylabel('transfer fraction')
ax1.legend()

ax3.plot(resfreq,yvals,linestyle='',marker='.')
ax3.set_xlabel('log')
ax3.set_ylabel('log')
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.set_xlim(20,150)
ax3.set_ylim(.0015,.15)
ax3.plot(xlist,fixedPowerLaw(xlist,*poptfixed),linestyle='-',label='Fixed Power Law: -1.5')
ax3.plot(xlist,fitPowerLaw(xlist,*popt),linestyle='-',label='Fit Power Law: '+fitpowerlawresult+'pm'+fitpowerlawerror)
plt.legend()

### insert 
###size and place of inset
#left 0 is on y axis, bottom 0 is on x axis
#width, height is size of inset 
left, bottom, width, height = [0.31, 0.6, 0.15, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.set_ylabel('residuals')
ax2.plot(xlist,residuals,linestyle='',marker='.')

print(f'A: {popt[0]}')
print(f'range fit over: {minf,maxf}')
# %%
### scale data to GammaTilde
def GammaTilde(x):
        if index.loc['VVA'] == 'varying':
            return (EF*h*kHz)/(hbar*np.pi*np.array(OmegaRmax)[:,]**2*trf)*x
        else:
            return (EF*h*kHz)/(hbar*np.pi*OmegaRmax**2*trf)*x

scaledtransfer = resfreq/EF
#%%
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

# ### fit power law for scaled transfer
guess = [100,-1.5]
poptscaled, pcovscaled = curve_fit(fitPowerLaw,truncatedresfreqscaled,truncatedyvalsscaled,p0=guess)
perrscaled = np.sqrt(np.diag(pcovscaled))

### fixed power law scaled transfer
guess = [100]
poptfixedscaled, pcovfixedscaled = curve_fit(fixedPowerLaw,truncatedresfreqscaled,truncatedyvalsscaled,p0=guess)

fitpowerlawresultscaled = str(float("{:.2f}".format(poptscaled[1])))
fitpowerlawerrorscaled = str(np.round(perrscaled[1],3))

### finding residuals for the area we chose to fit 
xlistscaled = np.linspace(truncatedresfreqscaled.min(),truncatedresfreqscaled.max(),len(truncatedyvalsscaled))
residualsscaled = truncatedyvalsscaled - fitPowerLaw(xlistscaled,*poptscaled)

# %%
### plot GammaTilde
fig, (ax1,ax3) = plt.subplots(1,2,figsize=(20,10))

ax1.set_xlabel('rf frequency $\Delta$ (EF)')
ax1.set_ylabel('Scaled Transfer 'r'$\tilde{\Gamma}$/N')
ax1.plot(scaledtransfer,GammaTilde(transfer),marker='.',linestyle='')
ax1.plot(xlistscaled,fitPowerLaw(xlistscaled,*poptscaled),label='Fixed Power Law: -1.5')
ax1.plot(xlistscaled,fixedPowerLaw(xlistscaled,*poptfixedscaled),label='Fit Power Law: '+fitpowerlawresultscaled+'pm'+fitpowerlawerrorscaled)
ax1.legend()

ax3.set_xlabel('log')
ax3.set_ylabel('log')
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.set_xlim(min((xlistscaled)),max((xlistscaled)))
ax3.set_ylim(min((fitPowerLaw(xlistscaled,*poptscaled))),max((fitPowerLaw(xlistscaled,*poptscaled))))
ax3.plot(scaledtransfer,GammaTilde(transfer),marker='.',linestyle='')
ax3.plot(xlistscaled,fitPowerLaw(xlistscaled,*poptscaled),label='Fixed Power Law: -1.5')
ax3.plot(xlistscaled,fixedPowerLaw(xlistscaled,*poptfixedscaled),label='Fit Power Law: '+fitpowerlawresultscaled+'pm'+fitpowerlawerrorscaled)
plt.legend()

### insert 

###size and place of inset
#left 0 is on y axis, bottom 0 is on x axis
#width, height is size of inset 
left, bottom, width, height = [0.31, 0.6, 0.15, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])

ax2.set_ylabel('residuals')
ax2.plot(xlistscaled,residualsscaled,linestyle='',marker='.')


print(f'EF is {EF:.3f}')
print(f'Range: {minfscaled,maxfscaled}')
# %% 
### finding contact 
Cscaled = GammaTilde(transfer)*np.abs(scaledtransfer)**(3/2)*np.pi**2*2**(3/2)

# %%
### fitting contact 
### max and min where we want to fit  
minfC = 1.5
maxfC = 5.7
yvalsC = Cscaled

### filter mask
freqfiltermaskscaled = scaledtransfer > minfscaled 
newresfreqscaled, newyvalsC = scaledtransfer[freqfiltermaskscaled], yvalsC[freqfiltermaskscaled]
freqfiltermaskscaled = newresfreqscaled < maxfscaled
truncatedresfreqscaled, truncatedyvalsC = newresfreqscaled[freqfiltermaskscaled], newyvalsC[freqfiltermaskscaled]

### fit Contact
def Contactfit(x,B,A):
    return B*x**A/(x**(-3/2))

guess = [1,1]
poptC, pcovC = curve_fit(Contactfit,truncatedresfreqscaled,truncatedyvalsC,p0=guess)
perrC = np.sqrt(np.diag(pcovC))
fitContact = str(float("{:.2f}".format(poptC[1])))
fitContacterror = str(np.round(perrC[1],3))

### fixed contact
xlistfixedC = np.linspace(poptC[0],poptC[0],len(truncatedyvalsC))


### finding residuals for the area we chose to fit 
xlistC = np.linspace(truncatedresfreqscaled.min(),truncatedresfreqscaled.max(),len(truncatedyvalsC))
residualsscaled = truncatedyvalsscaled - fitPowerLaw(xlistscaled,*poptscaled)

# %%
### plotting contact 
fig, ax = plt.subplots()

ax.set_ylabel(r'$2\sqrt{2}\pi^2\tilde{\Gamma}\Delta^{-3/2}$')
ax.set_xlabel('rf frequency $\Delta$ (EF)')
ax.plot(scaledtransfer,Cscaled,linestyle='',marker='.')
ax.plot(xlistC,Contactfit(xlistC,*poptC),label='fit A: '+fitContact+'pm'+fitContacterror)
ax.plot(xlistC,xlistfixedC,label='fixed A')
plt.legend()
# %%
### max and min scaled loss 
minf = 2*EF
maxf = 110.022 #6*EF 
yvals1 = (-c9+bgloss)

### filter mask
freqfiltermask = resfreq > minf 
newresfreq1, newyvals1 = resfreq[freqfiltermask], yvals1[freqfiltermask]
freqfiltermask = newresfreq1 < maxf
truncatedresfreq1, truncatedyvals1 = newresfreq1[freqfiltermask], newyvals1[freqfiltermask]

yvals2 = c5

### filter mask
freqfiltermask = resfreq > minf 
newresfreq2, newyvals2 = resfreq[freqfiltermask], yvals2[freqfiltermask]
freqfiltermask = newresfreq2 < maxf
truncatedresfreq2, truncatedyvals2 = newresfreq2[freqfiltermask], newyvals2[freqfiltermask]

### popt, pcov, perr of power law
guess = [100,-1.5]
popt1, pcov1 = curve_fit(fitPowerLaw,truncatedresfreq1,truncatedyvals1,p0=guess)
perr1 = np.sqrt(np.diag(pcov1))

popt2, pcov2 = curve_fit(fitPowerLaw,truncatedresfreq2,truncatedyvals2,p0=guess)
perr2 = np.sqrt(np.diag(pcov2))


xlist1 = np.linspace(truncatedresfreq1.min(),truncatedresfreq1.max(),len(truncatedyvals2))
xlist2 = np.linspace(truncatedresfreq2.min(),truncatedresfreq2.max(),len(truncatedyvals2))

# %% 
### plotting scaled loss (?)
fig, ax = plt.subplots()

transfernew = 1 - (c9 )/(c5 + ff*(c9))

ax.set_xlabel('rf freqeuncy (kHz)')
# plt.ylabel('Scaled Loss')
ax.set_ylim(-1000,3000)
ax.set_xlim(EF,110)
ax.plot(resfreq,(-c9+bgloss),linestyle='',marker='.',label='-c9+offset')
ax.plot(xlist1,fitPowerLaw(xlist1,*popt1),linestyle='-',label='-c9+offset')

ax.plot(resfreq,c5,linestyle='',marker='.',label='c5')
ax.plot(xlist2,fitPowerLaw(xlist2,*popt2),linestyle='-',label='c5')

# ax.plot(resfreq,transfernew,linestyle='',marker='.',label='')

# plt.plot([],[],linestyle='',label='2EF to 110')

plt.legend()

print(f'Power law for -c9+offset is {popt1[1]:.4f}\pm{perr1[1]:.3}')
print(f'Power law for c5 is {popt2[1]:.4f}\pm{perr2[1]:.3}')
# %%
### max and min where we want to fit  
minf = 2*EF
maxf = 110.022 #6*EF 
yvalsf95 = f95

### filter mask
freqfiltermask = resfreq > minf 
newresfreqf95, newyvalsf95 = resfreq[freqfiltermask], yvalsf95[freqfiltermask]
freqfiltermask = newresfreqf95 < maxf
truncatedresfreqf95, truncatedyvalsf95 = newresfreqf95[freqfiltermask], newyvalsf95[freqfiltermask]

yvals2 = c5

### filter mask
freqfiltermask = resfreq > minf 
newresfreq2, newyvals2 = resfreq[freqfiltermask], yvals2[freqfiltermask]
freqfiltermask = newresfreq2 < maxf
truncatedresfreq2, truncatedyvals2 = newresfreq2[freqfiltermask], newyvals2[freqfiltermask]

### popt, pcov, perr of power law
guess = [100,-1.5]
poptf95, pcovf95 = curve_fit(fitPowerLaw,truncatedresfreqf95,truncatedyvalsf95,p0=guess)
perrf95 = np.sqrt(np.diag(pcovf95))

popt2, pcov2 = curve_fit(fitPowerLaw,truncatedresfreq,truncatedyvals2,p0=guess)
perr2 = np.sqrt(np.diag(pcov2))


xlistf95 = np.linspace(truncatedresfreqf95.min(),truncatedresfreqf95.max(),len(truncatedyvalsf95))
xlist2 = np.linspace(truncatedresfreq2.min(),truncatedresfreq2.max(),len(truncatedyvals2))

# %%
### f95 
fig, ax = plt.subplots()

ax.set_ylim(0,.15)
ax.set_xlim(EF,110)
ax.plot(resfreq,f95,linestyle='',marker='.',label='f95')
ax.plot(xlistf95,fitPowerLaw(xlistf95,*poptf95),linestyle='-',label='f95')

ax.legend()

print(f'Power law for f95 is {poptf95[1]:.4f}\pm{perrf95[1]:.3}')
# %%
fig, ax = plt.subplots()
ax.plot(resfreq,transfernew,linestyle='',marker='.',label='')


# %%
