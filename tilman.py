# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:40:59 2023

@author: coldatoms
"""
import numpy as np 
import matplotlib.pyplot as plt
from data import * 

xtilman, ytilman = np.loadtxt("zetaomega_T0.58.txt", unpack=True, delimiter=' ')
 
E = 9 * 3.14 * xtilman**2 * ytilman *0.13**2 /12

namex, namey, x, y = data("2023-10-10_E_e.dat", names=['freq','ToTFcalc'])
# print(x, y)
arrayy = [y-0.5 for y in y]
arrayy = np.array(arrayy)*(0.1/19) *20


arrayx = np.array(x)/19



plt.xlabel('freq (EF)')
plt.ylabel('dE/dt (EF**2)')
plt.plot(xtilman, E)
plt.plot(arrayx,arrayy, 'ro')