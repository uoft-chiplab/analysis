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

trapavg = 1/1.76
trapavg2 = 1/2.31
#dE/dT/A^2 = 9/2 omega^2 bulkvisc eq'n
E = 9 * np.pi * xtilman**2 * ytilman /12 * trapavg  # for 0.58 ToTF
E2 = 9 * np.pi * xtilman2**2 * ytilman2 /12 * trapavg2 # for 0.25 ToTF 

# E =  xtilman**2 * ytilman /12 # for 0.58 ToTF
# E2 = xtilman2**2 * ytilman2 /12 # for 0.25 ToTF 

#experiment numbers 
ToTFi = 0.53 #same for oct 10 and oct 17 from tshots 
EF = 19 # kHz
A = 0.131 # 1/kFa
uatom = 1.66053906660e-27 #kg
mK = 39.96399848*uatom 
h = 6.62607015e-34
hbar = h/2/np.pi

kF = np.sqrt(2*mK*EF*h*1000)/hbar

nHO = kF**3/6/np.pi 

#oct 10 run E 10ms, scan freq
namexE, nameyE, xE, yE = data("2023-10-10_E_e.dat", names=['freq','ToTFcalc'])

trfE = 10 # ms

arrayyE = [yE-ToTFi for yE in yE]
arrayyEdEdt = np.array(arrayyE)*(1/EF/trfE) /A**2
arrayxE = np.array(xE)/EF

# print(arrayyEdEdt * (np.sqrt(2/arrayxE)))

#oct 17 run I 5ms, scan freq
namexI, nameyI, xI, yI = data("2023-10-17_I_e.dat", names=['freq (kHz)','ToTFcalc'])

trfI = 5 #ms

arrayyI = [yI-ToTFi for yI in yI]
arrayyIdEdt = np.array(arrayyI)*(1/EF/trfI) /A**2
arrayxI = np.array(xI)/EF

#oct 17 run K 2ms, scan freq
namexK, nameyK, xK, yK = data("2023-10-17_K_e.dat", names=['freq (kHz)','ToTFcalc'])

trfK = 2 #ms

arrayyK = [yK-ToTFi for yK in yK]
arrayyKdEdt = np.array(arrayyK)*(1/EF/trfK) /A**2
arrayxK = np.array(xK)/EF

#oct 18 run H 1ms, 1Vpp, scan freq
ToTFi = 0.52
namexH, nameyH, xH, yH = data("2023-10-18_H_e.dat", names=['freq (kHz)','ToTFcalc'])

trfH = 1 #ms
A = 0.13/2

arrayyH = [y-ToTFi for y in yH]
arrayyHdEdt = np.array(arrayyH)*(1/EF/trfH) /A**2
arrayxH = np.array(xH)/EF

#oct 18 run I 1ms, 2Vpp, scan freq
ToTFi = 0.52
namexI2, nameyI2, xI2, yI2 = data("2023-10-18_I_e.dat", names=['freq (kHz)','ToTFcalc'])

trfI2 = 1 #ms
A = 0.13

arrayyI2 = [y-ToTFi for y in yI2]
arrayyI2dEdt = np.array(arrayyI2)*(1/EF/trfI2) /A**2
arrayxI2 = np.array(xI2)/EF

# plotting vs freq

plt.figure(0)
plt.xlabel('freq (EF)')
plt.ylabel('dE/dt/A**2 (EF/tF)')
plt.plot(xtilman, E, label='0.58')
plt.plot(xtilman2, E2, label='0.25')
plt.plot(arrayxE,arrayyEdEdt, 'ro', label='oct 10 run E, 10ms')
plt.plot(arrayxI,arrayyIdEdt, 'go', label='oct 17 run I, 5ms')
plt.plot(arrayxK,arrayyKdEdt, 'bo', label='oct 17 run K, 2ms')
plt.plot(arrayxH,arrayyHdEdt, 'o', color = 'purple', label='oct 18 run H, 1ms A/2')
plt.plot(arrayxI2,arrayyI2dEdt, 'o', color = 'orange', label='oct 18 run I, 1ms A')

plt.legend()

############################################################

# plt.figure(2)
# arrayyET = np.array(arrayyE)
# arrayyIT = np.array(arrayyI)
# arrayyKT = np.array(arrayyK)



# plt.xlabel('freq (EF)')
# plt.ylabel('ToTF')
# plt.plot(arrayxE,arrayyET, 'ro', label='oct 10 run E, 10ms')
# plt.plot(arrayxI,arrayyIT, 'go', label='oct 17 run I, 5ms')
# plt.plot(arrayxK,arrayyKT, 'bo', label='oct 17 run K, 2ms')

# plt.legend()
# #######################################################

# #oct 17 run J 5ms, 30kHz, scan amp
# namexJ, nameyJ, xJ, yJ = data("2023-10-17_J_e.dat", names=['amp (Vpp)','ToTFcalc'])

# trfJ = 5 # ms
# omegaJ = 30 #kHz
# EF = 19
# A = 0.13

# arrayyJ = [yJ-ToTFi for yJ in yJ]
# arrayyJ = np.array(arrayyJ)/trfJ
# arrayxJ = np.array(xJ)/2*A # scale to (1/kF a) from Vpp, where 2Vpp = A

# #dE/dT = 9/2 A^2 omega^2 bulkvisc eq'n
# bulk30kHz = 3.4144771219e-01/12
# alist = np.linspace(0,.13,100)
# EJ = 9 * alist**2 * 3.14 * (omegaJ)**2/EF * bulk30kHz # for 0.58 ToTF
# EJ2 = 9 * alist**2 * 3.14 * (omegaJ)**2/EF * bulk30kHz # for 0.25 ToTF 

# plt.figure(1)
# plt.xlabel('A (1/akF)')
# plt.ylabel('dE/dt')
# # plt.plot(alist, EJ, label='0.58')
# # plt.plot(alist, EJ2, label='0.25')
# plt.plot(arrayxJ,arrayyJ, 'ro', label='oct 17 run J')
# plt.legend()
