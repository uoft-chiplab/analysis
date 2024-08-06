# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:30:49 2024

@author: coldatoms
"""

from data_class import Data
from fit_functions import *

file = '2024-04-25_C_e.dat'
data = Data(file)


# fixed sin fit
names = ['time','fraction95']
guess = [0.006,1.6,0.074]

data.fit(FixedSin1kHz, names=names, guess=guess)
data.group_by_mean(names[0])

fit_func, param_names, my_guess = FixedSin1kHz(data.data[names])

data.plot(names)
num = 100
xlist = np.linspace(np.min(data.data[names[0]]), np.max(data.data[names[0]]), num)
data.ax.plot(xlist, fit_func(xlist, *data.popt), '-', color='orange')

# free fit
# names = ['time','fraction95']
# guess = [0.01,2*3.14,1.6,0.07]

# data.fit(Sin, names=names, guess=guess)
# data.group_by_mean(names[0])

# fit_func, param_names, my_guess = Sin(data.data[names], guess=guess)

# data.plot(names)
# num = 100
# xlist = np.linspace(np.min(data.data[names[0]]), np.max(data.data[names[0]]), num)
# data.ax.plot(xlist, fit_func(xlist, *data.popt), '-', color='orange')