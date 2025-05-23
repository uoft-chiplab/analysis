# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 11:53:55 2025

@author: coldatoms
"""

import numpy as np
import matplotlib.pyplot as plt

P = 31.8e-3 # mW
pi = np.pi
w = 954e-4 # cm

ypx = 138
wpx = 250

I_0 = 2*P/(pi*w**2)

print(I_0)

I_cloud = I_0 * np.exp(-ypx**2/wpx**2)

print(I_cloud)

print(2800/5300*I_0*0.2/0.24)

s = 1/2

rho_ee = 1/2 * s/(1+s)

rho_gg = 1-rho_ee

correction = 1/rho_gg

print(correction)

plt.figure()
xs = np.linspace(0.01, 1.0, 100)
ys = (xs+2)/(1+xs)/2
ys2 = 1/(1+xs/3)

plt.plot(xs, ys, '--')
plt.plot(xs, ys2, '--')
plt.show()