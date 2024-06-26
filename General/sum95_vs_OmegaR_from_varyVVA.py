# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:03:54 2024

@author: coldatoms
"""
from library import *
from data_class import Data
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fit_functions import *

# Construct the full file path
file_path = os.path.join(current_dir, 'VVAtoVpp.txt')

Vpp = {}
with open(file_path) as f:
	next(f)
	for line in f:
		tok = line.split()
		Vpp[tok[0]] = tok[1]

Vpp = {key : float(value) for key, value in Vpp.items()}

def pulsearea(x):
	return np.sqrt(0.31)*(x - 25.2) + (1)*(25.2)

pulseareanew = np.sqrt(3)

def OmegaR(VVA):
 	return  2*np.pi*pulseareanew*Vpp[str(VVA)]
 
# List of VVA values
VVA_list = [float(num) for num in Data("2024-06-25_J_e.dat",average_by='VVA').avg_data['VVA']]

# Initialize an empty list to store results
results = []

# Iterate through each VVA value
for vva in VVA_list:
    # Call OmegaR function and append the result to the list
    results.append(OmegaR(vva))
	
fig, ax = plt.subplots()

y = Data("2024-06-25_J_e.dat",average_by='VVA').avg_data['sum95']

ax.plot(results, y)

def Linear(x,m,b):
	return m*x + b

popt, pcov = curve_fit(Linear,results,y)

xlist = np.linspace(min(results),max(results),1000)

ax.plot(xlist,Linear(xlist,*popt),linestyle='-',marker='')

ax.set_xlabel('OmegaR')
ax.set_ylabel('sum95')
print(f'm={popt[0]},b={popt[1]}')