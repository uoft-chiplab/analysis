# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:12:19 2024

@author: coldatoms
"""

import numpy as np
import matplotlib.pyplot as plt

f = 0.51

def rho_aa(VVA):
	return f - f*np.sin(VVA * np.pi/4)**2 + (1-f)*np.sin(VVA * np.pi/4)**2 

def rho_bb(VVA):
	return 1-f - (1-f)*np.sin(VVA * np.pi/4)**2 + f*np.sin(VVA * np.pi/4)**2 

def f95(VVA):
	return rho_aa(VVA)/(rho_aa(VVA)+rho_bb(VVA))

VVAs = np.linspace(0, 4, 100)

VVAs = np.append(VVAs,[0])

plt.figure(figsize=(6,4))
plt.xlabel('VVA')
plt.ylabel(r"$\rho_{aa}$")
plt.plot(VVAs, rho_aa(VVAs))
plt.show()

plt.figure(figsize=(6,4))
plt.xlim(-0.2,4.2)
plt.xlabel('VVA')
plt.ylabel(r"f95")
plt.plot(VVAs, f95(VVAs))
plt.show()