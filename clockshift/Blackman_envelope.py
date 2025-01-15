# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:52:12 2024

@author: coldatoms
"""

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

a0 = 0.42659
a1 = 0.49656
a2 = 0.076849

# a0 = 21/50
# a1 = 25/50
# a2 = 4/50

def Blackman_envelope(x, a0=a0, a1=a1, a2=a2):
	return np.piecewise(x, [x<0, x>1, (x>=0) & (x<=1)], [lambda x: 0, 
		lambda x: 0, lambda x: a0 - a1*np.cos(2*pi*x) + a2*np.cos(4*pi*x)])

b0 = 1060.9629
b1 = -3.520967
b2 = 0.0027443

c0 = 6234.181826
c1 = -197.392088
c2 = 1.0

def Blackman_Fourier(x, b0=b0, b1=b1, b2=b2, c0=c0, c1=c1, c2=c2):
	return np.sin(x/2)*2*(b0 + b1*x**2 + b2*x**4)/(c0*x+c1*x**3+c0*x**3)
	

Blackman = Blackman_envelope

xs = np.linspace(-0.5, 1.5, 1000)

fig, axs = plt.subplots(2,1, figsize=(8,5))

ax = axs[0]
ax.plot(xs, Blackman(xs), '-')
ax.set(xlabel="Time (arb.)", ylabel="Amplitude (arb.)",
			   ylim=[-0.005,0.01]
			   )


xs = np.linspace(-2*pi, 2*pi, 1000)

ax = axs[1]
ax.plot(xs, np.abs(Blackman_Fourier(xs))**2, '-')
ax.set(xlabel="Time (arb.)", ylabel="Fourier Amplitude$^2$ (arb.)")

fig.tight_layout()

plt.show()
