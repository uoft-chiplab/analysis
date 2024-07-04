# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:51:13 2024

@author: coldatoms
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

time1 = np.array([45 + 50*x for x in range(0,7)])
time2 = np.array([415 + 50*x for x in range(0,9)])
time = np.concatenate((time1, time2))

B = np.array([202.106, 202.125, 202.141, 202.146, 202.148, 202.137, 202.123, 202.108, 202.120, 202.132, 202.139, 202.150, 202.139,202.135, 202.127, 202.111])
B_err = 0.001* np.array([8,6,6,4,7,4,4,3,4,6,4,4,4,4,4,3])

def FixedSin2p5kHz(t, A, p, C):
	omega = 0.0025 * 2 * np.pi # 2.5 kHz
	return A*np.sin(omega*t - p) + C

popt,pcov = curve_fit(FixedSin2p5kHz, time, B, sigma=B_err)

t=np.linspace(0, time[-1], 50)
fitB=FixedSin2p5kHz(t, *popt)
plt.errorbar(time, B, yerr=B_err, fmt='bo')
plt.plot(t, fitB, 'r--')
plt.ylim(202.09, 202.16)
plt.ylabel('B [G]')
plt.xlabel('time [us]')