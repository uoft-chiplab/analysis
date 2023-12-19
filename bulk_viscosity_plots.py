# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:29:36 2023

@author: coldatoms
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
import scipy.integrate as integrate
from scipy.special import kn
import scipy.special as special

bulkT58 = np.loadtxt("zetaomega_T0.58.txt", comments="#", delimiter=" ", unpack=False)

bulkT25 = np.loadtxt("zetaomega_T0.25.txt", comments="#", delimiter=" ", unpack=False)

bulkT2p0 = np.array([[0.01, 0.014],[0.4,0.014],[1,0.0135],[1.4,0.012],
					 [2, 0.01],[4,0.005],[7,0.003],[10,0.0014]])

data_folder = "data\\heating\\"

# bulkmeasshort = [0.00014,0.000131]
# bulkshorterror = [0.00018,0.00011]
# omegaoEFshort = [15/19,50/19]

# bulkmeas203 = [0.000099,0.00013,0.00000567,0.00001]
# bulk203error = [0.000042,0.0000025,0.0000004,0.0000007]
# bulkmeas203 = [1.65e-4,5.8e-5,1.03e-5,1.58e-5]
# bulk203error = [3.7e-5,2.2e-6,4e-7,6.3e-7]
# omegaoEF203 = [5/19,15/19,50/19,150/19]

file = data_folder + "Nov09_ToTF0p65_203G.txt"
omegaoEF203, bulkmeas203, bulk203error = np.loadtxt(file,
								   delimiter=',', unpack=True)

# bulkmeas202p1 = [0.00264,0.00121,0.00171,0.000364,0.00745,0.0022]
# bulkerror202p1 = [0.00043,0.00053,0.00019,0.000038,0.0018,0.00019]
# omegaoEF202p1 = [15/19,5/17,50/19,150/19,10/19,30/19]

file = data_folder + "Nov07_ToTF0p6_202p1G.txt"
omegaoEF202p1, bulkmeas202p1, bulkerror202p1 = np.loadtxt(file,
								  delimiter=',', unpack=True)

file = data_folder + "Nov18_bg202p1G.txt"
omegaoEFbg202p1, bulkmeasbg202p1, bulkerrorbg202p1 = np.loadtxt(file,
								  delimiter=',', unpack=True)

EF = 23
# bulkhot = [.000011,.00000133,.000000952,.000000142,-.00000000654]
# bulkhoter = [.0000046,.0000025,.00000043,.00000017,.000000092]
# bulkhot = [1.91e-3,5.45e-4,2.50e-4,9.8e-5,6.78e-5]
# bulkhoter = [2.7e-4,1.2e-4,3.0e-5,1.5e-5,7.3e-6]
# omegaoEFhot = [10/23.53,20/23.88,50/23.79,100/23.81,150/23.64]

file = data_folder + "Nov15_ToTF1p4_202p1G.txt"
omegaoEFhot, bulkhot, bulkhoter = np.loadtxt(file,
								   delimiter=',', unpack=True)

def bulkplot():
	fig = plt.figure(4, figsize=(8,6))
	

	ax = fig.add_subplot(111)

	ax.set_xlabel('omega/EF')
	ax.set_ylabel('Dynamical Bulk Viscosity')
	ax.loglog(bulkT58[:,0],bulkT58[:,1]*12,linestyle='-',label='T=0.58', color='r')
# 	ax.loglog(bulkT58[:,0],bulkT58[:,1],marker='.')
# 	ax.loglog(bulkT25[:,0],bulkT25[:,1]/12,linestyle='-',label='T=0.25', color='teal')
# 	ax.loglog(bulkT2p0[:,0],bulkT2p0[:,1],linestyle='-',label='T=2.00', color='brown')
# 	ax.loglog(bulkT25[:,0],bulkT25[:,1],marker='d')
	ax.errorbar(omegaoEF203,bulkmeas203,yerr = bulk203error,marker='o',
			 linestyle='None',label='203G, ToTF ~ 0.65')
	ax.errorbar(omegaoEF202p1,bulkmeas202p1,yerr = bulkerror202p1,marker='o',
			 linestyle='None',label='202.1G, ToTF ~ 0.60', color='r')
	ax.errorbar(omegaoEFbg202p1,bulkmeasbg202p1,yerr = bulkerrorbg202p1,marker='o',
			 linestyle='None',label='bg 202.1G, ToTF ~ 0.60', color='k')
	ax.errorbar(omegaoEFhot,bulkhot,yerr = bulkhoter,marker='o',
			 linestyle='None',label='202.1G, ToTF = 1.4', color = 'brown')

	ax.legend(loc = 'lower left')
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.set(xlim=[0.01, 10])
	return fig

bulkplot()

# fig.savefig('test.png')