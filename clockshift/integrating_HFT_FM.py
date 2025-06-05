#from HFT_dimer_bg_analysis import getDataFrame
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
# paths
import sys
import os
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')
if root not in sys.path:
	sys.path.append(root)

from library import pi, h, hbar, mK, a0, paper_settings, generate_plt_styles

# HFT lines with and without final state effect (xd)
def HFTtailFSE(x, A, xd):
    return A*x**(-3/2) / (1+x/xd)
def HFTtail(x, A):
	return A*x**(-3/2)

def a13(B):
	''' ac scattering length '''
	abg = 167.6*a0
	DeltaB = 7.2
	B0 = 224.2
	return abg*(1 - DeltaB/(B-B0))


#df = getDataFrame()
# print(df)
df = pd.read_excel(os.path.join(data_path, 'HFT_dimer_bg_analysis_results.xlsx'))
idx = 11 # an index where C ~ 2.5
C = df['C'][idx] # Ctilde
scale = 2.5/C
#scale = 1
C = C *scale # forced to be 2.5
e_C = df['e_C'][idx] * scale
EF = df['EF'][idx]
e_EF=df['e_EF'][idx]
detuning = 0.1 # MHz
prefactor = 1/(2**(3/2)*pi**2)
GammaTilde = prefactor * C * (detuning/EF)**(-3/2)
e_GammaTilde =  GammaTilde * e_C / C # this is because the total uncertainty was calculated inside e_C already

xstar = df['x_star'][idx] # EF
# FOR ZERO RANGE THEORY, NEED ZERO RANGE XSTAR OR OMEGA_A
omega_a = hbar/mK/(a13(202.14)**2) /2/pi/ 1e6/ EF 

lowest_bound =False
if lowest_bound:
	C = C - e_C
	GammaTilde = GammaTilde - e_GammaTilde

# estimated HFT tails based on single measurement
xi = 1 # 1 EF for min cutoff, complete guess rn
xf = 1000
xx = np.linspace(xi, xf, xf*2)
yyFSE = HFTtailFSE(xx, prefactor*C, omega_a)
yy = HFTtail(xx, prefactor*C)

# find x-value where line intersects with noisefloor to find cutoff
noisefloor=1e-5
omegamax = (prefactor*C/noisefloor)**(2/3)
def func(x):
	return noisefloor*x**(3/2)*(1+x/omega_a) - prefactor*C
omegamaxFSE = fsolve(func, x0=200)[0]
print(f'omegamax = {omegamax}')
print(f'omegamaxFSE = {omegamaxFSE}')

# calculate FM up to cutoff
cut = [xx<omegamax]
cut_FSE = [xx<omegamaxFSE]
FM_FSE = np.trapz(xx[cut_FSE]*yyFSE[cut_FSE], xx[cut_FSE])
FM = np.trapz(xx[cut]*yy[cut], xx[cut])
print('FM up to a cutoff:')
print(f'FM = {FM:.2f} EF')
print(f'FM FSE = {FM_FSE:.2f} EF') 

# calculate FM up to infinity
xi = 0.5 # 1 EF for min cutoff, complete guess rn
xf = 1000000
xx = np.linspace(xi, xf, xf*2)
yyFSE = HFTtailFSE(xx, prefactor*C, omega_a)
yy = HFTtail(xx, prefactor*C)
# calculate FM up to cutoff
cut = [xx<xf]
cut_FSE = [xx<xf]
FM_FSE = np.trapz(xx[cut_FSE]*yyFSE[cut_FSE], xx[cut_FSE])
FM = np.trapz(xx[cut]*yy[cut], xx[cut])
print('FM out to infinity:')
print(f'FM = {FM:.2f} EF')
print(f'FM FSE = {FM_FSE:.2f} EF') 

# also check how much FM responds to change in initial cutoff

fig, ax = plt.subplots()
FM_list = []
FM_FSE_list = []
xi_list = []
for i in range(0, 5):
 	cut_i = xx[i]
 	print(f'Initial cutoff = {cut_i} :')
 	cut = [(cut_i < xx) & (xx < omegamax)]
 	cut_FSE = [(cut_i < xx) & (xx < omegamaxFSE)]
 	FM = np.trapz(xx[cut]*yy[cut], xx[cut])
 	FM_FSE = np.trapz(xx[cut_FSE]*yyFSE[cut_FSE], xx[cut_FSE])
 	print(f'FM = {FM:.2f} EF')
 	print(f'FM FSE = {FM_FSE:.2f} EF') 
 	FM_list.append(FM)
 	FM_FSE_list.append(FM_FSE)
 	xi_list.append(xx[i])
ax.plot(xi_list, FM_list, 'b', ls='--')
ax.plot(xi_list, FM_FSE_list, 'b', ls='-')
ax.set(ylabel='FM', xlabel='initial xi [EF]')


# plotting
fig, ax = plt.subplots()
ax.plot(xx, yyFSE, '-', color='blue')
ax.plot(xx, yy, '--', color='blue')
ax.errorbar(detuning/EF, GammaTilde, yerr= e_GammaTilde, color='r', marker='o')
ax.hlines(noisefloor, xx.min(), xx.max(), color='k', ls='--')
ax.plot(omegamax, noisefloor, 'ro')
ax.plot(omegamaxFSE, noisefloor, 'ro')
ax.set(yscale='log',
	   xscale='log',
	   ylabel=r'$\widetilde{\Gamma}$',
	   xlabel=r'$\tilde{\omega} [E_F]$')


