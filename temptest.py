# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 14:00:03 2025

@author: coldatoms
"""

import numpy as np
import matplotlib.pyplot as plt

# sinc^2 dimer lineshape functions
def sinc2(x, trf):
	"""sinc^2 normalized to sinc^2(0) = 1"""
	t = x*trf
	return np.piecewise(t, [t==0, t!=0], [lambda t: 1, 
					   lambda t: (np.sin(np.pi*t/2)/(np.pi*t/2))**2])

def sinc2alt(x, trf):
	"""sinc^2 where the zeros are at 1/t"""
	t = x*trf
	return np.piecewise(t, [t==0, t!=0], [lambda t: 1, 
					   lambda t: (np.sin(np.pi*t)/(np.pi*t))**2])


xx = np.linspace(-3000, 3000, 3000)
trf = 0.001
yy = sinc2(xx, trf)
yy2 = sinc2alt(xx, trf)
int1 = np.trapz(yy, xx)
int2 = np.trapz(yy2, xx)
fig, ax = plt.subplots()
ax.plot(xx, yy, '-')
ax.plot(xx, yy2, '--')
print(f'int1={int1}')
print(f'int2={int2}')
print(f'int1/int2 = {int1/int2}')

