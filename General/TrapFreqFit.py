# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:19:19 2024

@author: coldatoms
"""
# paths
import os
import sys
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
if root not in sys.path:
	sys.path.insert(0, root)

import numpy as np
from data_class import Data
from fit_functions import TrapFreq2, RabiFreq

file_name ='2025-02-06_D_e'
piezos = [30, 50, 120, 140]
files = [file_name + "_piezo_f=" + str(piezo) + '.dat' for piezo in piezos]
# path = "E:\Data\2024\09 September2024\06September2024\F_z1_trapfreq"
xname = 'time'

trap_freq = True

if trap_freq:
	fit_func = TrapFreq2
	names = [xname, 'G_ctr_x']
	guess = [4, 10, 2*np.pi*0.14, 0, 133]
	
else: # Rabi freq?
	fit_func = RabiFreq
	names = [xname, 'fraction95']
	guess = [1.1,11,0,0]

for file in files:
	label = file
	print("--------------------------")
	print("Fitting {}".format(label))
	Data(file).fit(fit_func, names, guess=guess)
