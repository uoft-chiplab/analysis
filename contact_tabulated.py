# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:31:36 2024

@author: coldatoms

Tabulated by eye from Vale paper and 
T. Enss, R.Haussmann, W. Zwerger,
Ann.of Phys. 326, 3,2011,770-796,
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

test_contact_plot = False

# Contact density estimate from Vale paper and Tilman paper
# Crappy eye-balled data
T = np.array([0, 0.05, 0.1, .15, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
C1 = np.array([3.025, 3.02, 2.98, 2.85, 2.65, 2.58, 2.51, 2.40, 2.29])
C2 = np.array([0.061, 0.046, 0.038, 0.032, 0.017])	 

C = np.concatenate((C1, C2*(3*np.pi**2)))

ContactInterpolation = lambda x: np.interp(x, T, C)
xs = np.arange(0, 10, 0.1)

if test_contact_plot:

	plt.figure()
	plt.plot(T, C, 'ro')
	plt.plot(xs, ContactInterpolation(xs), 'r-')
	plt.xlabel(r"$T (T_F)$")
	plt.ylabel(r'Contact Density $\mathcal{C}/(n k_F)')
	# plt.yscale('log')
	# plt.xscale('log')
	plt.xlim(0,1)
	plt.ylim(2, 3.1)
	plt.show()