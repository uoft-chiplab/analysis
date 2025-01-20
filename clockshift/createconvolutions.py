#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:26:31 2024

@author: kevinxie
"""
# this is a hack to access modules in the parent directory
# Get the current script's directory
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
if parent_dir not in sys.path:
	sys.path.append(parent_dir)
## This script is intended to create and save lineshapes that convolve an atom-to-dimer transfer spectrum with a FD distribution at some T/TF,
## with the square of the FT of the transfer pulse shape.
import numpy as np
import matplotlib.pyplot as plt
plt.ion() # turn on interactive mode so plots won't block execution
import pandas as pd
import clockshift.pwave_fd_interp as FD # FD distribution data for interpolation functions, Ben Olsen
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d
from library import pi, h

plot_convs=False
save_pickle=True
load_pickle=False

pickle_file = "./clockshift/convolutions_EFs_640us.pkl"

# transfer lineshape w/ FD distribution
def lsFD(x, A, numDA):
	PFD = FD.distRAv[numDA]  # 5 = 0.30 T/TF, 6 = 0.40 T/TF
	ls = A * PFD(np.sqrt(-x))
	ls = np.nan_to_num(ls)
	return ls

# t in us
def Sincw(w, t): # used Mathematica to get an exact FT
	return 0.797885 * np.sin(t/2*w)/w
def Sincf(f,t): # kHz
	return Sincw(2*pi*f/1000, t)
def Sinc2D(Delta, t, EF): # takes dimensionless detuning
	return Sincw(2*pi*Delta*EF/1000, t)**2

# evaluation range
xrange = 300 # kHz
xnum = 1000

arbscale = 1
# these are all the TTFs BAO's lookup table works for, but we don't need all of them
TTFs = np.array([0.20, 0.25, 0.30, 0.40, 0.50])
# trfs = np.array([10, 20, 30, 40, 50, 100, 150, 200]) # testing
#trfs = np.array([10])
trfs = np.array([640])
EFs = np.array([12,14,16])
# trfs = np.arange(10,400,10)

resultsTTF = []
resultstrf = []
resultsEF = []
resultsls = []

for EF in EFs:
	# generate sample range using EF
	xx = np.linspace(-xrange/EF, xrange/EF, xnum)
	
	for idx, TTF in enumerate(TTFs):
	# 	if idx < 4 or idx > 9: continue
		print(str(idx), str(TTF), str(EF))
		
		# transfer spectrum lineshape with FD dist at some TTF
		if plot_convs:
			fig, ax = plt.subplots()
			ax.plot(xx, lsFD(xx, arbscale, idx))
			ax.set(title='FD dist transfer spectrum, TTF={:.2f}'.format(TTF), xlabel='Detuning [EF]')
			plt.show()
		FDinterp = lambda x: np.interp(x, xx, lsFD(xx, arbscale, idx))
		FDnorm = quad(FDinterp, -xnum, xnum, points=xx, limit=2*xx.shape[0]) # had to do wacky things to integrate properly
		print('FDNORM: ' + str(FDnorm))
		
		for trf in trfs:
			# FT of the pulse shape
			# evaluate and get the norm of the Sinc2 function
			D = np.linspace(-10*1e3/trf / EF, 10*1e3/trf /EF, xnum)
			yD = Sinc2D(D, trf, EF)
			FTnorm = np.trapz(yD, D)
			print('FTNORM: ' + str(FTnorm))
			if plot_convs:
				fig, ax=plt.subplots()
				ax.plot(D, yD)
				ax.set(title='FT^2 square pulse, trf={:.1f}'.format(trf), xlabel='Freq [EF]')
				plt.show()
			
			# the convolution function is just a product of two normalized
			# functions, slid across one another through the parameter t, and integrated
			# at every slide position t
			def convfunc(tau, t):
				return FDinterp(tau)/FDnorm[0] * Sinc2D(t-tau, trf, EF)/FTnorm
			def convint(t):
				# the integral converges better when ranges don't go to infinity
				sliderange = 20 
				qrangelow = - sliderange
				qrangehigh=  sliderange
				return quad(convfunc, qrangelow, qrangehigh, args=(t,))
		
			yyconv = []
			e_yyconv = []
			for xconv in xx:
				a, b = convint(xconv)
				yyconv.append(a)
				e_yyconv.append(b)
			convnorm = np.trapz(yyconv,xx)
			print('Conv norm: ' + str(convnorm))
			# create the convolution lineshape for current iteration
			convinterp = interp1d(xx, yyconv, bounds_error=False, fill_value=0)
	
			# show convs explicitly
			if plot_convs:
				fig_CVs, ax_CV = plt.subplots()
				ax_CV.plot(xx, FDinterp(xx)/FDnorm[0], '-', label='GammaFD/norm')
				ax_CV.plot(xx, Sinc2D(xx, trf, EF)/FTnorm, '-', label='pulseFT^2/norm')
				ax_CV.plot(xx, yyconv/convnorm, '-', label='conv/norm')
				ax_CV.set(xlabel = 'Detuning [EF]', ylabel = 'Magnitude')
				ax_CV.legend()
				plt.show()
				
			
			resultsTTF.append(TTF)
			resultsEF.append(EF)
			resultstrf.append(trf)
			resultsls.append(convinterp)
			
results = {'TTF':resultsTTF, 'EF':resultsEF, 'TRF':resultstrf, 'LS':resultsls}

df = pd.DataFrame(data=results)

if load_pickle:
	loaded_df = pd.read_pickle(pickle_file)
	df = pd.concat([loaded_df, df], ignore_index=True, sort=False)

if save_pickle:
	df.to_pickle(pickle_file)
			
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	