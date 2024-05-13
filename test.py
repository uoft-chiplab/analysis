# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:43:27 2024

@author: coldatoms
"""

from library import *
from field_wiggle_calibration import Bamp_from_Vpp
import numpy as np
import matplotlib.pyplot as plt

Vpp = 1.8
B = 202.1
EF = 17e3
freq = 5

CaoC0 = 0.23
C202p1 = 0.78

dCdkFa = 1.69

Bamp, e_Bamp = Bamp_from_Vpp(Vpp, freq)

kF = np.sqrt(2*mK*EF*h)/hbar
num = 100
Blist = np.linspace(B-Bamp, B+Bamp, num)
def invkFa(B):
	return 1/kF/a97(B-Bamp)

Cmax = (invkFa(Blist[0])-invkFa(B)) * dCdkFa + C202p1
Cmin = (invkFa(Blist[-1])-invkFa(B)) * dCdkFa + C202p1

Camp = (invkFa(Blist[-1])-invkFa(B)) * dCdkFa

A = 1/kF/a97(B+Bamp)

phi = 0.4

Edot = EF/4/pi * A * Camp * np.sin(phi) * 2 * pi * freq*1e3



plt.figure()
plt.plot(Blist, (invkFa(Blist)-invkFa(B)) * dCdkFa + C202p1)


rate = 22.3
A = 0.474
Ei = 18.47e3

Edotheating = rate*A**2 * Ei

rate = 47.66
A = 0.217
Ei = 22.078e3

Edotheating2 = rate*A**2 * Ei

