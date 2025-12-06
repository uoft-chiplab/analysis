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
#idx = 11 # an index where C ~ 2.5
idx =10 # an index where C ~ 2.2
C = df['C'][idx] # Ctilde
#scale = 2.5/C
scale = 1
C = C *scale # if on, force to be 2.5
e_C = df['e_C'][idx] * scale
EF = df['EF'][idx]
e_EF=df['e_EF'][idx]
kF = df['kF'][idx]
a13kF = df['a13kF'][idx]
SW = df['SW_c5'][idx]
detuning = 0.1 # MHz
prefactor = 1/(2**(3/2)*pi**2)
xstar = df['x_star'][idx] # EF
# FOR ZERO RANGE THEORY, NEED ZERO RANGE XSTAR OR OMEGA_A
omega_a = hbar/mK/(a13(202.14)**2) /2/pi/ 1e6/ EF 

GammaTilde = prefactor * C * (detuning/EF)**(-3/2) / (1+detuning/EF/omega_a)
e_GammaTilde =  GammaTilde * e_C / C # this is because the total uncertainty was calculated inside e_C already

# choice of SR value on denominator of CS calc
sumrule = 0.5 # ideal
#sumrule = 0.25 # empirical


lowest_bound =False
if lowest_bound:
	C = C - e_C
	GammaTilde = GammaTilde - e_GammaTilde

# estimated HFT tails based on single measurement
xi_nr =0
xf_nr = 4
xi = 4 # min cutoff of HFT, complete guess rn
xf = 400
xx = np.linspace(0, xf, xf*10) # units of EF
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

# calculate CS up to cutoff
cut = [(xi<xx) & (xx<omegamax)][0]
cut_FSE = [(xi < xx) & (xx<omegamaxFSE)][0]
cut_nr = [(xi_nr < xx) & (xx<xf_nr)][0]
CS_nr = np.trapz(xx[cut_nr] * yyFSE[cut_nr], xx[cut_nr] ) / sumrule
CS_FSE = np.trapz(xx[cut_FSE]*yyFSE[cut_FSE], xx[cut_FSE]) / sumrule
CS = np.trapz(xx[cut]*yy[cut], xx[cut]) / sumrule
print('CS up to a cutoff:')
print(f'CS = {CS:.2f} EF')
print(f'CS FSE = {CS_FSE:.2f} EF') 
print(f'CS near-resonant = {CS_nr:.2f} EF')

# also check how much CS responds to change in near res and HFT cutoff

CS_list = []
CS_FSE_list = []
cutoff_list = []
CS_NR_list = []
SW_NR_list = []
SW_FSE_list = []
for i in np.arange(5, 120, 1):
	cutoff = i/1000/EF
	cut = [(cutoff < xx) & (xx < omegamax)][0]
	cut_FSE = [(cutoff < xx) & (xx < omegamaxFSE)][0]
	cut_nr = [(0 < xx)&(xx<cutoff)][0]
	CS_nr = np.trapz(xx[cut_nr] * yyFSE[cut_nr], xx[cut_nr] ) / sumrule
	CS = np.trapz(xx[cut]*yy[cut], xx[cut]) / sumrule
	CS_FSE = np.trapz(xx[cut_FSE]*yyFSE[cut_FSE], xx[cut_FSE]) / sumrule
	SW_nr = np.trapz(yyFSE[cut_nr], xx[cut_nr])
	SW_FSE = np.trapz(yyFSE[cut_FSE], xx[cut_FSE])
	# print(f'CS = {CS:.2f} EF')
	# print(f'CS FSE = {CS_FSE:.2f} EF') 
	# print(f'CS near-resonant = {CS_nr:.2f} EF')
	CS_list.append(CS)
	CS_FSE_list.append(CS_FSE)
	CS_NR_list.append(CS_nr)
	cutoff_list.append(cutoff)
	SW_NR_list.append(SW_nr)
	SW_FSE_list.append(SW_FSE)
	if i == 50:
		print(SW_FSE)
cutoff_list = np.array(cutoff_list)
CS_FSE_list = np.array(CS_FSE_list)
CS_NR_list = np.array(CS_NR_list)
SW_NR_list = np.array(SW_NR_list)
SW_FSE_list = np.array(SW_FSE_list)
fig, axs = plt.subplots(2, figsize=(8,6))
axs[0].plot(cutoff_list * EF * 1000, CS_FSE_list, 'b', ls='-')
axs[0].vlines(50, min(CS_FSE_list), max(CS_FSE_list), 'b', ls='--')
axs[1].plot(cutoff_list * EF * 1000, CS_NR_list, 'g', ls= '-')
axs[1].vlines(50, min(CS_NR_list), max(CS_NR_list), 'g', ls='--')
axs[0].set(ylabel='CS HFT', xlabel='near res-HFT cutoff  [kHz]')
axs[1].set(ylabel='CS NR', xlabel='near res-HFT cutoff [kHz]')
fig.suptitle(fr'$\hbar \Delta /E_F$ estimates for C={C:.1f} N $k_F$, assumes $\int \widetilde\Gamma d \tilde\omega = 0.5$, $E_F \approx 14$ kHz')
fig.tight_layout()
fig, axs = plt.subplots(2, figsize=(8,6))
axs[0].plot(cutoff_list * EF * 1000, SW_FSE_list, 'b', ls='-')
axs[0].vlines(50, min(SW_FSE_list), max(SW_FSE_list), 'b', ls='--')
axs[1].plot(cutoff_list * EF * 1000, SW_NR_list, 'g', ls= '-')
axs[1].vlines(50, min(SW_NR_list), max(SW_NR_list), 'g', ls='--')
axs[0].set(ylabel='SW HFT', xlabel='near res-HFT cutoff  [kHz]')
axs[1].set(ylabel='SW NR', xlabel='near res-HFT cutoff [kHz]')
fig.suptitle(fr'SW estimates for C={C:.1f} N $k_F$, $E_F \approx 14$ kHz')
fig.tight_layout()

# estimated HFT tails based on single measurement
# fix the lower bound, and scan over the final bound
fig, axs = plt.subplots(1,2, figsize=(8,4))
plt.rcParams.update({'font.size': 12}) # Adjust 
xi = 50 / 1000 / EF # 50 kHz converted to EF units
xf = 400
xx = np.linspace(0, xf, xf*10) # units of EF
yyFSE = HFTtailFSE(xx, prefactor*C, omega_a)
yy = HFTtail(xx, prefactor*C)

# also check how much CS responds to change in near res and HFT cutoff

CS_list = []
CS_FSE_list = []
cutoff_list = []
CS_NR_list = []
SW_NR_list = []
SW_FSE_list = []
for i in np.arange(60, 5000, 100):
	cutoff = i/1000/EF
	cut = [(xx > xi) & (xx < cutoff)][0]
	cut_FSE = [(xx > xi) & (xx < cutoff)][0]
	CS = np.trapz(xx[cut]*yy[cut], xx[cut]) / sumrule
	CS_FSE = np.trapz(xx[cut_FSE]*yyFSE[cut_FSE], xx[cut_FSE]) / sumrule
	SW_FSE = np.trapz(yyFSE[cut_FSE], xx[cut_FSE])
	CS_FSE_list.append(CS_FSE)
	cutoff_list.append(cutoff)
	SW_FSE_list.append(SW_FSE)
	
cutoff_list = np.array(cutoff_list)
CS_FSE_list = np.array(CS_FSE_list)
CS_NR_list = np.array(CS_NR_list)
SW_NR_list = np.array(SW_NR_list)
SW_FSE_list = np.array(SW_FSE_list)
axs[0].plot(cutoff_list * EF, CS_FSE_list, 'orange', ls='-')
axs[0].vlines(omegamaxFSE*EF, min(CS_FSE_list), max(CS_FSE_list), 'black', ls='--', label='Noise floor cutoff')
axs[0].set(ylabel=r'$\Delta_\mathrm{HFT}$', xlabel='Upper integration bound [MHz]')
axs[0].legend()
# fig.suptitle(fr'$\hbar \Delta /E_F$ estimates for C={C:.1f} N $k_F$, assumes $\int \widetilde\Gamma d \tilde\omega = 0.5$, $E_F \approx 14$ kHz')
# fig.suptitle(fr'$\hbar \Delta /E_F$ estimates for C={C:.1f} N $k_F$')

# plotting gammatilde vs omegatilde
plt.rcParams.update({'font.size': 12}) # Adjust the size as needed
axs[1].plot(xx, yyFSE, '-', color='blue', label='Eqn. (1)')
axs[1].plot(xx, yy, '--', color='blue', label=r'Eqn. (1), $\omega_a \to \infty$')
axs[1].plot(detuning/EF, GammaTilde, ls='', color='r', marker='o', label=r'$\omega/\omega_a \approx 0.05$')
axs[1].axhspan(0, noisefloor, alpha=0.2, color='red', label='Noise floor')
axs[1].annotate('Noise floor',
			xy=(xx.min()+0.2, noisefloor-5e-4),
			xytext = (xx.min()+0.2, noisefloor-5e-4),
			# arrowprops = dict(arrowstyle='->', color='black'),
			fontsize=12
)
# ax.plot(omegamax, noisefloor, 'ro')
# ax.plot(omegamaxFSE, noisefloor, 'ro')
axs[1].set(yscale='log',
	   xscale='log',
	   ylabel=r'$\widetilde{\Gamma}$',
	   xlabel=r'$\tilde{\omega} [E_F]$')
axs[1].legend()
fig.tight_layout()
plt.savefig('figures/refereeA9_visual.png', dpi=300)

# yoffset = 0.01
# xoffset = -5
# label = (r'$\omega/(2\pi) = 100$' + ' kHz\n' + 
# 		 r'$\omega/\omega_a \approx 0.05$')
# ax.annotate(label,
# 			xy=(detuning/EF, GammaTilde),
# 			xytext = (detuning/EF + xoffset, GammaTilde+yoffset),
# 			arrowprops = dict(arrowstyle='->', color='black'),
# 			fontsize=12
# )

# Create inset
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
# axins = fig.add_axes([0.25, 0.35, 0.2, 0.2])
# plt.rcParams.update({'font.size': 12}) 
# axins.plot(xx, yyFSE, '-', color='blue')
# axins.plot(xx, yy, '--', color='blue')

# # Zoom into a specific region
# x1, x2, y1, y2 = 6.5, 7.7, 3.5e-3, 5e-3
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# axins.errorbar(detuning/EF, GammaTilde, yerr= e_GammaTilde, color='r', marker='o')
# # Draw box and connecting lines
# mark_inset(ax, axins, loc1=1, loc2=1, fc="none", ec="0.5")
# mark_inset(ax, axins, loc1=4, loc2=4, fc="none", ec="0.5")



