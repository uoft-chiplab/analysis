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

files = ['2024-09-25_J_e.dat']
# path = "E:\Data\2024\09 September2024\06September2024\F_z1_trapfreq"
xname = 'time'

trap_freq = False

if trap_freq:
	fit_func = TrapFreq2
	names = [xname, 'G_ctr_z']
	guess = [8, 7, 2*3.14*0.5, 0, 95]
	
else: # Rabi freq?
	fit_func = RabiFreq
	names = [xname, 'fraction95']
	guess = [1.1,11,0,0]

for file in files:
	label = file
	print("--------------------------")
	print("Fitting {}".format(label))
	Data(file).fit(fit_func, names, guess=guess)
