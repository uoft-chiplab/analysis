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
from glob import glob

import numpy as np
from data_class import Data
from fit_functions import TrapFreq2, RabiFreq
root_project = os.path.dirname(os.getcwd())
# Carmen_Santiago\Analysis Scripts
root_analysis = os.path.dirname(root_project)
# Carmen_Santiago\
root = os.path.dirname(root_analysis)
# Fast-Modulation-Contact-Correlation-Project\contact_correlations\phaseshift
analysis_folder = os.path.join(root_project, r"contact_correlations\phaseshift")
# Carmen_Santiago\\Data
root_data = os.path.join(root, "Data")

run = '2025-12-08_H'
y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
datfiles = glob(f"{runpath}*=*.dat")
if datfiles == None:
	datfiles = datfiles
else:	
	datfiles =glob(f"{runpath}*.dat")
path = runpath.split("\\")[-1]
runname = datfiles[0].split("\\")[-2].lower() #

# file_name ='2025-10-22_J_e'
# file_name = '2025-10-23_C'
piezos = [30, 50, 120, 140]
# files = [run + '_e.dat']#+ "_piezo_f=" + str(piezo) + '.dat' for piezo in piezos]
# path = r"E:\Data\2025\10 October2025\23October2025\C_Y1piezo_trapfreq_scanpiezo"
xname = 'time'

trap_freq = True

if trap_freq:
	fit_func = TrapFreq2
	names = [xname, 'G_ctr_y']
	guess = [15, 2, 2, 2, 98]

#y names are G_ctr_x , G_ctr_y, and fCtr1 
	
else: # Rabi freq?
	fit_func = RabiFreq
	names = [xname, 'fraction95']

	guess = [1,10,10,0.5]

for files in datfiles:
	print(files)
	label = files
	file_name = files.split("\\")[-1]
	print("--------------------------")
	print("Fitting {}".format(label))
	run = Data(file_name, path=path)
	run.fit(fit_func, names, guess=guess)
