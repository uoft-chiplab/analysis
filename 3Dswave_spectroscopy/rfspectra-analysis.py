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
plt.rcParams.update({'font.size': 16})

Avg = False
SaveOn = True

### name and xlsx file # 
name = '2023-08-03_B.dat'
indexnum = 10
upperlim = 47.2
kHz = 10**3


### files and paths
data_folder = 'rfspectra'
file_path = os.path.join(data_folder, name)
xlsxfile = 'ContactMeasurementsSummary.xlsx'

### reading excel file 
df = pd.read_excel(xlsxfile, index_col=0, engine='openpyxl')

### choosing which file to grab from the xlsx file 
index = df.iloc[indexnum]
print(index) 
# %%
### grabbing specific columns from data file
metadata = pd.read_table(file_path, delimiter=',')
dataframe = pd.DataFrame(metadata)
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
losstransfer = (bgloss - c9)/bgloss

res= FreqMHz(field, 9/2,-5/2,9/2,-7/2)
EF = (trapfreqsy*trapfreqsx*trapfreqsz*6*TShotsN/2)**(1/3)
resfreq = (freq-res)*kHz

minf = 1.5*EF
maxf = 6*EF
# %%
### average data 
avgfreq = dataframe.groupby('freq').mean().reset_index()['freq']
avgresfreq = np.array((avgfreq-res)*kHz)
avgc9 = np.array(dataframe.groupby('freq').mean()['c9'])
avgc5 = np.array(dataframe.groupby('freq').mean()['c5'])
# %%
### plotting transfer vs freq
c5_c9_transfer_losstrans_vs_resfreq, ax = plt.subplots(2,2, figsize=(18,12))

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

ax[1,1].set_ylabel('loss transfer (bg - c9)')
ax[1,1].plot(resfreq,losstransfer,linestyle='',marker='.')
ax[1,1].text(-10,max(losstransfer),name,fontsize=18,ha='right',va='top')
# %%
### filter mask function 
### i.e. truncating the freq list to fit only 
### the -3/2 tail 

### max and min where we want to fit  
def filtermask(yvals,resfreq,minf,maxf):
    freqfiltermask = resfreq > minf 
    newresfreq, newyvals = resfreq[freqfiltermask], yvals[freqfiltermask]
    freqfiltermask = newresfreq < maxf
    truncatedresfreq, truncatedyvals = newresfreq[freqfiltermask], newyvals[freqfiltermask]

    return truncatedresfreq, truncatedyvals
# %%
### fitting Power Law  
def fitPowerLaw(x,A,B):
    return A*x**B

### popt, pcov, perr of power law

def fitPowerLawfit(transfer,resfreq,minf,maxf):
    guess = [100,-1.5]
    popt, pcov = curve_fit(fitPowerLaw,filtermask(transfer,resfreq,minf,maxf)[0],filtermask(transfer,resfreq,minf,maxf)[1],p0=guess)
    perr = np.sqrt(np.diag(pcov))
    
    return popt,pcov,perr
# %%
### fixing the power law to -3/2 
def fixedPowerLaw(x,A):
    return A*x**(-3/2)
    
### popt, pcov, perr of power law

def fixedPowerLawfit(transfer,resfreq,minf,maxf):
    guess = [100]
    popt, pcov = curve_fit(fixedPowerLaw,filtermask(transfer,resfreq,minf,maxf)[0],filtermask(transfer,resfreq,minf,maxf)[1],p0=guess)
    perr = np.sqrt(np.diag(pcov))
    
    return popt,pcov,perr 
# %%
### plotting rf spectra and residuals as inset
yvals = transfer

### finding residuals for the area we chose to fit 
xlist = np.linspace(filtermask(transfer,resfreq,minf,maxf)[0].min(),filtermask(transfer,resfreq,minf,maxf)[0].max(),len(filtermask(transfer,resfreq,minf,maxf)[1]))
residuals = filtermask(transfer,resfreq,minf,maxf)[1] - fitPowerLaw(xlist,*fitPowerLawfit(transfer,resfreq,minf,maxf)[0])

### large plot 
transfrac_w_resid_and_loglog_plot, (ax1,ax3) = plt.subplots(1,2,figsize=(20,10))

ax1.plot(resfreq,yvals,linestyle='',marker='.')
ax1.plot(xlist,fitPowerLaw(xlist,*fitPowerLawfit(transfer,resfreq,minf,maxf)[0]),linestyle='-', label='Fit Power Law: '+str(float("{:.2f}".format(fitPowerLawfit(transfer,resfreq,minf,maxf)[0][1])))+'pm'+str(np.round(np.sqrt(np.diag(fitPowerLawfit(transfer,resfreq,minf,maxf)[1]))[1],3)))
ax1.plot(xlist,fixedPowerLaw(xlist,*fixedPowerLawfit(transfer,resfreq,minf,maxf)[0]),linestyle='-',label='Fixed Power Law: -1.5')
ax1.set_xlabel('rf spectra (kHz)')
ax1.set_ylabel('transfer fraction')
ax1.legend()

ax3.plot(resfreq,yvals,linestyle='',marker='.')
ax3.set_xlabel('log')
ax3.set_ylabel('log')
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.set_xlim(min(xlist),max(xlist))
ax3.set_ylim(min(fixedPowerLaw(xlist,*fixedPowerLawfit(transfer,resfreq,minf,maxf)[0])),max(fixedPowerLaw(xlist,*fixedPowerLawfit(transfer,resfreq,minf,maxf)[0])))
ax3.plot(xlist,fixedPowerLaw(xlist,*fixedPowerLawfit(transfer,resfreq,minf,maxf)[0]),linestyle='-',label='Fixed Power Law: -1.5')
ax3.plot(xlist,fitPowerLaw(xlist,*fitPowerLawfit(transfer,resfreq,minf,maxf)[0]),linestyle='-',label='Fit Power Law: '+ str(float("{:.2f}".format(fitPowerLawfit(transfer,resfreq,minf,maxf)[0][1])))+'pm'+str(np.round(np.sqrt(np.diag(fitPowerLawfit(transfer,resfreq,minf,maxf)[1]))[1],3)))
ax3.plot([],[],label=name,linestyle='')
ax3.legend()

### insert 
###size and place of inset
#left 0 is on y axis, bottom 0 is on x axis
#width, height is size of inset 
left, bottom, width, height = [0.31, 0.6, 0.15, 0.2]
ax2 = transfrac_w_resid_and_loglog_plot.add_axes([left, bottom, width, height])
ax2.set_ylabel('residuals')
ax2.plot(xlist,residuals,linestyle='',marker='.')

print(f'A: {fitPowerLawfit(transfer,resfreq,minf,maxf)[0][0]}')
print(f'range fit over: {minf/EF,maxf/EF} EF')
print(f'EF is {EF}')
# %%
### scale data to GammaTilde
def GammaTilde(x):
        if index.loc['VVA'] == 'varying':
            return (EF*h*kHz)/(hbar*np.pi*np.array(OmegaRmax)[:,]**2*trf)*x
        else:
            return (EF*h*kHz)/(hbar*np.pi*OmegaRmax**2*trf)*x

detuning = resfreq/EF #ok this is  bad name really scaled freq
#%%
### max and min where we want to fit 
minf = 1.5
maxf = 5.8
yvalsscaled = GammaTilde(transfer)

### finding residuals for the area we chose to fit 
xlistscaled = np.linspace(min(filtermask(yvalsscaled,detuning,minf,maxf)[0]),max(filtermask(yvalsscaled,detuning,minf,maxf)[0]),len(filtermask(yvalsscaled,detuning,minf,maxf)[1]))
residualsscaled = filtermask(yvalsscaled,detuning,minf,maxf)[1] - fitPowerLaw(xlistscaled,*fitPowerLawfit(yvalsscaled,detuning,minf,maxf)[0])

# %%
### plot GammaTilde
GammaTilde_w_resid_and_loglog, (ax1,ax3) = plt.subplots(1,2,figsize=(20,10))

ax1.set_xlabel('rf frequency $\Delta$ (EF)')
ax1.set_ylabel('Scaled Transfer 'r'$\tilde{\Gamma}$/N')
ax1.plot(detuning,yvalsscaled,marker='.',linestyle='')
ax1.plot(xlistscaled,fitPowerLaw(xlistscaled,*fitPowerLawfit(yvalsscaled,detuning,minf,maxf)[0]),label='Fit Power Law: '+ str(float("{:.2f}".format(fitPowerLawfit(yvalsscaled,detuning,minf,maxf)[0][1])))+'pm'+str(np.round(np.sqrt(np.diag(fitPowerLawfit(yvalsscaled,detuning,minf,maxf)[1]))[1],3)))
ax1.plot(xlistscaled,fixedPowerLaw(xlistscaled,*fixedPowerLawfit(yvalsscaled,detuning,minf,maxf)[0]),label='Fixed Power Law: -1.5')
ax1.legend()

ax3.set_xlabel('log')
ax3.set_ylabel('log')
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.set_xlim(min((xlistscaled)),max((xlistscaled)))
ax3.set_ylim(min((fitPowerLaw(xlistscaled,*fitPowerLawfit(yvalsscaled,detuning,minf,maxf)[0]))),max((fitPowerLaw(xlistscaled,*fitPowerLawfit(yvalsscaled,detuning,minf,maxf)[0]))))
ax3.plot(detuning,yvalsscaled,marker='.',linestyle='')
ax3.plot(xlistscaled,fitPowerLaw(xlistscaled,*fitPowerLawfit(yvalsscaled,detuning,minf,maxf)[0]),label='Fixed Power Law: -1.5')
ax3.plot(xlistscaled,fixedPowerLaw(xlistscaled,*fixedPowerLawfit(yvalsscaled,detuning,minf,maxf)[0]),label='Fit Power Law: '+ str(float("{:.2f}".format(fitPowerLawfit(yvalsscaled,detuning,minf,maxf)[0][1])))+'pm'+str(np.round(np.sqrt(np.diag(fitPowerLawfit(yvalsscaled,detuning,minf,maxf)[1]))[1],3)))
ax3.plot([],[],label=name)

plt.legend()

### insert 

###size and place of inset
#left 0 is on y axis, bottom 0 is on x axis
#width, height is size of inset 
left, bottom, width, height = [0.31, 0.6, 0.15, 0.2]
ax2 = GammaTilde_w_resid_and_loglog.add_axes([left, bottom, width, height])

ax2.set_ylabel('residuals')
ax2.plot(xlistscaled,residualsscaled,linestyle='',marker='.')

print(f'EF is {EF:.3f}')
print(f'Range: {minf,maxf}')
# %%
### sum rule 

#this creates a function that you put x pts into
#and it gives you y pts 
xlims = [-2,16]
num = len(transfer)

xs = np.linspace(xlims[0], xlims[-1], num)

TransferInterpFunc = lambda x: np.interp(x, detuning, GammaTilde(x))
sumrule = np.trapz(GammaTilde(transfer), x=xs)

print(sumrule)
# %% 
### finding contact 
Cscaled = GammaTilde(transfer)*np.abs(scaledtransfer)**(3/2)*np.pi**2*2**(3/2)
# %%
### fitting contact 
### max and min where we want to fit  
minfC = 1.5
maxfC = 5.7
yvalsC = Cscaled

### fit Contact
def Contactfit(x,B,A):
    return B*x**A/(x**(-3/2))

### fixed contact
xlistfixedC = np.linspace(fitPowerLawfit(yvalsC,detuning,minf,maxf)[0][0],fitPowerLawfit(yvalsC,detuning,minf,maxf)[0][0],len(filtermask(yvalsC,detuning,minfC,maxfC)[1]))

### finding residuals for the area we chose to fit 
xlistC = np.linspace(filtermask(yvalsC,detuning,minfC,maxfC)[0].min(),filtermask(yvalsC,detuning,minfC,maxfC)[0].max(),len(filtermask(yvalsC,detuning,minfC,maxfC)[1]))
residualsscaled = filtermask(yvalsC,detuning,minfC,maxfC)[1] - fitPowerLaw(xlistfixedC,*fitPowerLawfit(yvalsC,detuning,minf,maxf)[0])

# %%
### plotting contact 
contact, ax = plt.subplots()

ax.set_ylabel(r'$2\sqrt{2}\pi^2\tilde{\Gamma}\Delta^{-3/2}$')
ax.set_xlabel('rf frequency $\Delta$ (EF)')
ax.plot(detuning,Cscaled,linestyle='',marker='.')
ax.plot(xlistC,Contactfit(xlistC,*fitPowerLawfit(yvalsC,detuning,minf,maxf)[0]),label='fit A: '+fitContact+'pm'+fitContacterror)
ax.plot(xlistC,xlistfixedC,label='fixed A')
plt.legend()
# %%
### save data in xlsx file and plots 

#put what you want to save into a dictionary then 
#into a data frame 
if SaveOn == True:
    datatosave = {'Name': [name], 'GammaTilde Power Law fit': str(float("{:.2f}".format(fitPowerLawfit(yvalsscaled,detuning,minf,maxf)[0][1]))), 
              'Error':str(np.round(np.sqrt(np.diag(fitPowerLawfit(yvalsscaled,detuning,minf,maxf)[1]))[1],3)),
              'hi':'hi','test':'test'}

    datatosavedf = pd.DataFrame(datatosave)

    datatosave_folder = 'SavedData'
    runfolder = name
    figpath = os.path.join(datatosave_folder,name)
    os.makedirs(figpath,exist_ok=True)

    xlsxsavedfile = 'Saved_Data.xlsx'

    file_path = os.path.join(datatosave_folder, xlsxsavedfile)

    datatosavedf.to_excel(file_path,index=False)

    fig1_name = 'c5_c9_transfer_losstrans_vs_resfreq.png'
    fig1_path = os.path.join(figpath, fig1_name)
    c5_c9_transfer_losstrans_vs_resfreq.savefig(fig1_path)

    fig2_name = 'transfrac_w_resid_and_loglog_plot.png'
    fig2_path = os.path.join(figpath, fig2_name)
    transfrac_w_resid_and_loglog_plot.savefig(fig2_path)

    fig3_name = 'GammaTilde_w_resid_and_loglog.png'
    fig3_path = os.path.join(figpath, fig3_name)
    GammaTilde_w_resid_and_loglog.savefig(fig3_path)

    fig4_name = 'contact.png'
    fig4_path = os.path.join(figpath, fig4_name)
    contact.savefig(fig4_path)
# %%
### max and min scaled loss 
minf = 1.2
maxf = 6
yvals1 = GammaTilde(losstransfer)

yvals2 = GammaTilde(transfer)

xlist1 = np.linspace(filtermask(yvals1,detuning,minf,maxf)[0].min(),filtermask(yvals1,resfreq,minf,maxf)[0].max(),len(filtermask(yvals1,scaledtransfer,minf,maxf)[1]))
xlist2 = np.linspace(filtermask(yvals2,detuning,minf,maxf)[0].min(),filtermask(yvals2,resfreq,minf,maxf)[0].max(),len(filtermask(yvals2,scaledtransfer,minf,maxf)[1]))
# %% 
### plotting scaled loss (?)
fig, ax = plt.subplots(1,2,figsize=(20,10))

ax[0].set_xlabel('rf freqeuncy (kHz)')
# plt.ylabel('Scaled Loss')
# ax.set_ylim(-1000,3000)
# ax.set_xlim(EF,110)
# ax[0].plot(resfreq,(-c9+bgloss),linestyle='',marker='.',label='-c9+offset')
# ax[0].plot(xlist1,fitPowerLaw(xlist1,*popt1),linestyle='-',label='-c9+offset')

# ax[0].plot(resfreq,c5,linestyle='',marker='.',label='c5')
# ax[0].plot(xlist2,fitPowerLaw(xlist2,*popt2),linestyle='-',label='c5')

# ax[1].plot(resfreq,losstransfer,linestyle='',marker='.',label='c9bg-c9')

ax[0].plot(detuning,GammaTilde(losstransfer),linestyle='',marker='.',label='scaled loss transfer')
ax[0].plot(detuning,GammaTilde(transfer),linestyle='',marker='.',label='scaled transfer')
ax[0].plot(xlist1,fitPowerLaw(xlist1,*fitPowerLawfit(yvals1,scaledtransfer,minf,maxf)[0]))
ax[0].plot(xlist2,fitPowerLaw(xlist2,*fitPowerLawfit(yvals2,scaledtransfer,minf,maxf)[0]))

# ax3.plot(scaledtransfer,yvalsscaled,marker='.',linestyle='')
# ax3.plot(xlistscaled,fitPowerLaw(xlistscaled,*poptscaled),label='Fixed Power Law: -1.5')

ax[1].plot([],[],linestyle='',label=name)
ax[1].set_xlabel('log')
ax[1].set_ylabel('log')
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_xlim(min((xlist1)),max((xlist1)))
ax[1].set_ylim(min((fitPowerLaw(xlist1,*fitPowerLawfit(yvals1,detuning,minf,maxf)[0]))),max((fitPowerLaw(xlist1,*fitPowerLawfit(yvals1,detuning,minf,maxf)[0]))))
ax[1].plot(detuning,GammaTilde(losstransfer),linestyle='',marker='.',label='scaled loss transfer')
ax[1].plot(detuning,GammaTilde(transfer),linestyle='',marker='.',label='scaled transfer')
ax[1].plot(xlist1,fitPowerLaw(xlist1,*fitPowerLawfit(yvals1,detuning,minf,maxf)[0]))
ax[1].plot(xlist2,fitPowerLaw(xlist2,*fitPowerLawfit(yvals2,detuning,minf,maxf)[0]))

ax[0].legend()
ax[1].legend()
# ax[2].legend()

print(f'Plot range is {minf,maxf} EF')
print(f'Power law for loss transfer is {fitPowerLawfit(yvals1,detuning,minf,maxf)[0][1]:.4f}\pm{fitPowerLawfit(yvals1,detuning,minf,maxf)[2][1]:.3}')
print(f'Power law for transfer is {fitPowerLawfit(yvals2,detuning,minf,maxf)[0][1]:.4f}\pm{fitPowerLawfit(yvals1,detuning,minf,maxf)[2][1]:.3}')
# %%
### looking for plataeu

plateau = losstransfer*(resfreq)**(3/2)

fig,ax = plt.subplots()

ax.set_xlabel('resfreq')
ax.set_ylabel('loss transfer * (detuning)**3/2')
ax.plot(resfreq,plateau,linestyle='',marker='.',label=name)
ax.legend()
# %%
