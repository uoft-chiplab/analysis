# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:32:15 2024

@author: coldatoms
"""

import numpy as np
from library import *

### constants
c = 299792458

# 40K D2 line params
Gamma = 2*pi*6.035e6
lamda_0 = 766.7e-9
omega_0 = 2*pi*c/lamda_0

# ODT params
lamda = 1064e-9
omega = 2*pi*c/lamda
Delta = omega-omega_0
w1 = 25e-6
w2 = 70e-6
P1 = 0.05 # in W
P2 = 0.5

def Gaussian2D(x, y, A, sigma):
	return A/sigma**2 * np.exp(-2*(x**2+y**2)/sigma**2)

def I1(x, y ,z):
	A = 2*P1/pi
	sigma = np.sqrt(w1**2+(x*lamda/pi/w1)**2)
	return Gaussian2D(y, z, A, sigma)

def I2(x, y ,z):
	A = 2*P2/pi
	sigma = np.sqrt(w2**2+(z*lamda/pi/w2)**2)
	return Gaussian2D(x, y, A, sigma)

def U1(x, y, z):
	return 3*pi*c**2*Gamma/(2*w0**3*Delta) * I1(x,y,z)

def U2(x, y, z):
	return 3*pi*c**2*Gamma/(2*w0**3*Delta) * I2(x,y,z)

