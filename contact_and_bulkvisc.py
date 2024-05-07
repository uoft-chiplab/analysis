# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:31:19 2024

@author: coldatoms
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bulkvisctrap_class import BulkViscUniform, BulkViscTrap

def zeta_from_contact(contact, freq):
	return 2**(3/2) *contact/(36*np.pi) * freq**(-3/2)

def calc_zeta(f, rate, Ei):
	"""This is a little weird. Ask CD."""
	return 2/9*rate/1e3/ f * Ei/f

TilmanPRL0p25ToTF_filename = "zetaomega_T0.25.txt"
TilmanPRL0p58ToTF_filename = "zetaomega_T0.58.txt"

TilmanFiles = [TilmanPRL0p25ToTF_filename, TilmanPRL0p58ToTF_filename]

trap_to_uni_ratio = 0.12474304953514184 # determined by compared tabulated uni 
										# data and trap-averaged at one freq. 
										# This is not precise

# change matplotlib options
plt.rcParams.update({"figure.figsize": [6,9]})
fig, axs = plt.subplots(2,1)
num = 500
font = {'size'   : 12}
plt.rc('font', **font)
legend_font_size = 10 # makes legend smaller, so plot is visible

ax = axs[0]
xlabel=r'Frequency $\omega/E_F$'
ylabel=r'Contact Correlation $a^2 \zeta$ [dimless]'
ax.set(xscale='log', yscale='log', xlabel=xlabel, ylabel=ylabel 
	   #xlim=[1,10], ylim=[7e-4, 0.1])
	   )

ax_res = axs[1]
ylabel=r'Ratio of $a^2\zeta$ [dimless]'
ax_res.set(xlabel=xlabel, ylabel=ylabel, xscale='log')

### Add contact line(s)
CToTF0p25 = 2.65
CToTF0p58 = 2.55

Cvals = [CToTF0p25, CToTF0p58]
num = 100
xx = np.linspace(1, 10, num)

### Add tabulated Tilman ToTF lines from PRL
# have to divide by twelve because Tilman scaled by 12 for some reason

# ToTF = 0.58
xtilman, ytilman = np.loadtxt(TilmanPRL0p58ToTF_filename, unpack=True, delimiter=' ')


# ToTF = 0.25
# xtilman, ytilman = np.loadtxt(TilmanPRL0p25ToTF_filename, unpack=True, delimiter=' ')
# ax.plot(xtilman, ytilman/12, 'b', 
# 			 label=r'L-W calculation:  $T/T_F=0.25$')
# ax_res.plot(xtilman, ytilman/12/zeta_from_contact(CToTF0p25, xtilman), 'b', 
# 			 label=r'L-W calculation/$\zeta(C)$:  $T/T_F=0.25$')


Thetas = [0.25, 0.40, 0.58, 0.75, 1.40, 2.00]
Ts = [4.8e3, 7.6e3, 11e3, 14.25e3, 32.6e3, 46.5e3] # Hz
barnus = [306, 306, 306, 306, 306*np.sqrt(1.5), 306*np.sqrt(1.5)] # mean trap freq in Hz
mutraps = [9825, 5050, -3800, -14.88e3, -91.8e3, -180e3] # harmonic trap chemical potential
mubulks = [7520, 5450, 1500, -3250, -34400, -70.7e3] # uniform trap chemical potential
BVT_colors = ['blue', 'teal', 'red', 'orange', 'brown', 'r']

params = list(zip(Ts, barnus, mubulks))
params_trap = list(zip(Ts, barnus, mutraps))
theta_indices = [0, 2]

for i, file, C in zip(theta_indices, TilmanFiles, Cvals):
	
	xtilman, ytilman = np.loadtxt(file, unpack=True, delimiter=' ')
	nus = xtilman*18965.5
	
	BVU = BulkViscUniform(*params[i], nus)
	BVT = BulkViscTrap(*params_trap[i], nus)
	BVT.zetas = calc_zeta(BVT.nus/1e3, BVT.Edottraps/(2*BVT.Ns)/1e3, np.ones(len(BVT.nus)))
	
	ax.plot(xx, zeta_from_contact(C, xx),'--', color=BVT_colors[i],
		 label='$a^2\zeta(C)$: $T/T_F={:.2f}$'.format(Thetas[i]))
	ax.plot(xtilman, ytilman/12, color=BVT_colors[i], 
			 label=r'L-W calculation:  $T/T_F={:.2f}$'.format(Thetas[i]))
	ax.plot(xtilman, BVU.zetas, ':', color=BVT_colors[i], 
		 label="Drude form $T/T_F={:.2f}$".format(Thetas[i]))
	
	ax.plot(xtilman, BVT.zetas, ':', color=BVT_colors[i], 
		 label="Drude form trap-avg $T/T_F={:.2f}$".format(Thetas[i]))
	
	ax_res.plot(xtilman, ytilman/12/zeta_from_contact(C, xtilman), color=BVT_colors[i], 
			 label=r'$\zeta(C)$/L-W: $T/T_F={:.2f}$'.format(Thetas[i]))
	ax_res.plot(xtilman, BVU.zetas/(ytilman/12), '--', color=BVT_colors[i], 
			 label=r'Drude/L-W:  $T/T_F={:.2f}$'.format(Thetas[i]))
	ax_res.plot(xtilman, BVT.zetas/BVU.zetas, ':', color=BVT_colors[i], 
			 label=r'Drude trap/Drude uni:  $T/T_F={:.2f}$'.format(Thetas[i]))
	
	print(np.mean(BVT.zetas/BVU.zetas))
	
	if i==0:
		ax.legend()
		ax_res.legend()
	

# ax_res.relim()
# ax_res.autoscale()	
	
plt.tight_layout()
plt.show()
	