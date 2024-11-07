# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:49:33 2024

@author: coldatoms
"""

import numpy as np
import matplotlib.pyplot as plt
from contact_correlations.UFG_analysis import calc_contact

# ToTFs = np.array([0.2  , 0.225, 0.25 , 0.275, 0.3  , 0.325, 0.35 , 0.375, 0.4,
#         0.425, 0.45 , 0.475, 0.5  , 0.525, 0.55 , 0.575, 0.6  ])

ToTFs = np.linspace(0.2, 1.2, 29)

Cs = np.array([2.1783244 , 2.02855915, 1.90539411, 1.78810748, 1.67342239,
       1.56085269, 1.45088512, 1.34443262, 1.24180172, 1.14444157,
       1.05270978, 0.96701767, 0.88750127, 0.81417138, 0.74688656,
       0.68537897, 0.62934843, 0.57842509, 0.53227651, 0.49047896,
       0.45264838, 0.41839089, 0.38739359, 0.35932629, 0.33388573,
       0.31080807, 0.28985089, 0.27077148, 0.25339308])

contact_interpolation = lambda x: np.interp(x, ToTFs, Cs)

if __name__ == "__main__":
	
	EF = 14e3 # Hz
	barnu = 320 # Hz
	Cs = np.array([calc_contact(ToTF, EF, barnu)[0] for ToTF in ToTFs])
	
	plt.figure(figsize=(6,4))
	plt.xlabel(r"$T/T_F$")
	plt.ylabel(r"Contact, $C$ [$k_F/N$]")

	plt.plot(ToTFs, Cs)
	
	contact_interpolation = lambda x: np.interp(x, ToTFs, Cs)
	plt.plot(ToTFs, contact_interpolation(ToTFs), '--')
		
	plt.show()