# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:14:52 2024

@author: coldatoms
"""
from library import *
import numpy as np 
import matplotlib.pyplot as plt
from data_class import *
from scipy.interpolate import interp1d

textfile = 'spectrascanlist.txt'

dir_path = os.path.join(current_dir, 'clockshift\\HFT_scan_list')
file_path = os.path.join(dir_path, textfile)

number, detuning, VVA = np.loadtxt(file_path, unpack=True)

VVApertlistrun = Data('2024-09-17_C_e.dat')

xname = 'VVA' # is in MHz
ff = 1.02
ODTscale = 1.5
ToTF = 0.45
EF = 16e3 # Hz
EFMHz = EF/1e6
trf = 0.2e-3 # s
FourierWidth = 2/(trf*1e6) # MHz
res75= 47.2227
num = len(VVApertlistrun.data[xname])

VVApertlistrun.data['N'] = VVApertlistrun.data['c5']-bgc5*np.ones(num)+VVApertlistrun.data['c9']
bgc5 = VVApertlistrun.data[(VVApertlistrun.data[xname]-res75)< -3*FourierWidth]['c5'].mean()
VVApertlistrun.data['transfer'] = (VVApertlistrun.data['c5'] - bgc5*np.ones(num))/VVApertlistrun.data['N']

VVApertlistrungrouped = VVApertlistrun.data.groupby(xname).mean().reset_index()

plt.plot(VVApertlistrungrouped['VVA'],VVApertlistrungrouped['transfer'])

VVApertinterpfcn = interp1d(VVApertlistrungrouped['VVA'],VVApertlistrungrouped['transfer'], bounds_error=False,
    fill_value='extrapolate')
# print(type(VVApertinterpfcn))

def scalingthetransfertail(w):
	if np.any(w >= 0.02):
		newtransfer = 0.1*w**(-3/2)/0.100**(-3/2) 
		return newtransfer, VVApertinterpfcn(newtransfer)
		
print(scalingthetransfertail(detuning))	
	
# plt.plot(VVApertlistrungrouped['VVA'],VVApertinterpfcn,marker='',linestyle='-')

