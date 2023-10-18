# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:40:59 2023

@author: coldatoms
"""
import numpy as np 
import matplotlib.pyplot as plt
from data import * 

#tilmans numbers from txt file from email 
xtilman, ytilman = np.loadtxt("zetaomega_T0.58.txt", unpack=True, delimiter=' ')
xtilman2, ytilman2 = np.loadtxt("zetaomega_T0.25.txt", unpack=True, delimiter=' ')

#dE/dT/A^2 = 9/2 omega^2 bulkvisc eq'n
E = 9 * 3.14 * xtilman**2 * ytilman /12 # for 0.58 ToTF
E2 = 9 * 3.14 * xtilman2**2 * ytilman2 /12 # for 0.25 ToTF 

#experiment numbers 
ToTFi = 0.53 #same for oct 10 and oct 17 from tshots 
EF = 19 # kHz
A = 0.131 # 1/kFa

#oct 10 run E 10ms, scan freq
namexE, nameyE, xE, yE = data("2023-10-10_E_e.dat", names=['freq','ToTFcalc'])

trfE = 10 # ms

arrayyE = [yE-ToTFi for yE in yE]
arrayyE = np.array(arrayyE)*(1/EF/trfE) /A**2
arrayxE = np.array(xE)/EF

#oct 17 run I 5ms, scan freq
namexI, nameyI, xI, yI = data("2023-10-17_I_e.dat", names=['freq (kHz)','ToTFcalc'])

trfI = 5 #ms

arrayyI = [yI-ToTFi for yI in yI]
arrayyI = np.array(arrayyI)*(1/EF/trfI) /A**2
arrayxI = np.array(xI)/EF

#oct 17 run I 5ms, scan freq
namexK, nameyK, xK, yK = data("2023-10-17_K_e.dat", names=['freq (kHz)','ToTFcalc'])

trfK = 2 #ms

arrayyK = [yK-ToTFi for yK in yK]
arrayyK = np.array(arrayyK)*(1/EF/trfK) /A**2
arrayxK = np.array(xK)/EF

# plotting vs freq

plt.figure(0)
plt.xlabel('freq (EF)')
plt.ylabel('dE/dt (EF**2) / A**2')
# plt.plot(xtilman, E, label='0.58')
# plt.plot(xtilman2, E2, label='0.25')
plt.plot(arrayxE,arrayyE, 'ro', label='oct 10 run E, 10ms')
plt.plot(arrayxI,arrayyI, 'go', label='oct 17 run I, 5ms')
plt.plot(arrayxK,arrayyK, 'bo', label='oct 17 run K, 2ms')

plt.legend()

#######################################################

#oct 17 run J 5ms, scan amp
namexJ, nameyJ, xJ, yJ = data("2023-10-17_J_e.dat", names=['amp (Vpp)','ToTFcalc'])

trfJ = 5 # ms
omegaJ = 30 #kHz
EF = 19
A = 0.13

arrayyJ = [yJ-ToTFi for yJ in yJ]
arrayyJ = np.array(arrayyJ)*(1/EF/trfJ) 
arrayxJ = np.array(xJ)/2*A # scale to (1/kF a) from Vpp, where 2Vpp = A

#dE/dT = 9/2 A^2 omega^2 bulkvisc eq'n
bulk30kHz = 9.9377384570e-01/12
EJ = 9 * arrayxJ**2 * 3.14 * (omegaJ/EF)**2 * bulk30kHz # for 0.58 ToTF
EJ2 = 9 * arrayxJ**2 * 3.14 * (omegaJ/EF)**2 * bulk30kHz # for 0.25 ToTF 

plt.figure(1)
plt.xlabel('A (1/akF)')
plt.ylabel('dE/dt')
# plt.plot(arrayxJ, EJ, label='0.58')
# plt.plot(arrayxJ, EJ2, label='0.25')
plt.plot(arrayxJ,arrayyJ, 'ro', label='oct 17 run J')
plt.legend()
