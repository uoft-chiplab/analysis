#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:26:31 2024

@author: kevinxie
"""

## This script is intended to create and save lineshapes that convolve an atom-to-dimer transfer spectrum with a FD distribution at some T/TF,
## with the square of the FT of the transfer pulse shape.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pwave_fd_interp as FD # FD distribution data for interpolation functions, Ben Olsen
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d
from library import pi, h

plot_convs=True
save_pickle=True
EF = 16 # [kHz], stand-in value

# transfer lineshape w/ FD distribution
def lsFD(x, x0, A, numDA):
	PFD = FD.distRAv[numDA]  # 5 = 0.30 T/TF, 6 = 0.40 T/TF
	ls = A * PFD(np.sqrt(-x+x0))
	ls = np.nan_to_num(ls)
	return ls

# t in us
def Sincw(w, t): # used Mathematica to get an exact FT
	return 0.797885 * np.sin(t/2*w)/w
def Sincf(f,t): # kHz
	return Sincw(2*pi*f/1000, t)
def Sinc2D(Delta, t): # takes dimensionless detuning
	return Sincw(2*pi*EF*Delta/1000, t)**2

# evaluation range
# Ebguess = -3980 / EF # ~ -249 EF
Ebguess=0
xrange = 200 / EF # ~ 19 EF
xnum = 1000
xx = np.linspace(Ebguess-xrange, Ebguess+xrange, xnum)

arbscale=1
# these are all the TTFs BAO's lookup table works for, but we don't need all of them
TTFs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.50]
# trfs = np.arange(10, 400, 10)
trfs = np.array([10, 20, 30, 40, 100, 200]) # testing
# trfs = np.arange(10,400,10)

resultsTTF = []
resultstrf = []
resultsls = []

for idx, TTF in enumerate(TTFs):
	if idx < 5 or idx > 8: continue
	print(str(idx), str(TTF))
	
	# transfer spectrum lineshape with FD dist at some TTF
	fig, ax = plt.subplots()
	ax.plot(xx, lsFD(xx, Ebguess, arbscale, idx))
	ax.set(title='FD dist transfer spectrum, TTF={:.2f}'.format(TTF), xlabel='Detuning [EF]')
	plt.show()
	FDinterp = lambda x: np.interp(x, xx, lsFD(xx, Ebguess, arbscale, idx))
	FDnorm = quad(FDinterp, -xnum, xnum, points=xx, limit=2*xx.shape[0]) # had to do wacky things to integrate properly
	print('FDNORM: ' + str(FDnorm))
	
	for trf in trfs:
		
		# FT of the pulse shape
		# evaluate and get the norm of the Sinc2 function
		D = np.linspace(-10*1e3/trf / EF, 10*1e3/trf /EF, 1000)
		yD = Sinc2D(D, trf)
		FTnorm = np.trapz(yD, D)
		print('FTNORM: ' + str(FTnorm))
		fig, ax=plt.subplots()
		ax.plot(D, yD)
		ax.set(title='FT^2 square pulse, trf={:.1f}'.format(trf), xlabel='Freq [EF]')
		plt.show()
		
		# the convolution function is just a product of two normalized
		# functions, slid across one another through the parameter t, and integrated
		# at every slide position t
		def convfunc(tau, t):
			return FDinterp(tau)/FDnorm[0] * Sinc2D(t-tau, trf)/FTnorm
		def convint(t):
			# the integral converges better when ranges don't go to infinity
			sliderange=20 
			qrangelow = Ebguess - sliderange
			qrangehigh= Ebguess + sliderange
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
		convinterp = interp1d(xx, yyconv)
# 		convinterp = lambda x: np.interp(x, xx, yyconv)

		# show convs explicitly
		if plot_convs:
			fig_CVs, ax_CV = plt.subplots()
			ax_CV.plot(xx, FDinterp(xx)/FDnorm[0], '-', label='GammaFD/norm')
			ax_CV.plot(xx, Sinc2D(xx-Ebguess, trf)/FTnorm, '-', label='pulseFT^2/norm')
			ax_CV.plot(xx, yyconv/convnorm, '-', label='conv/norm')
			ax_CV.set(xlabel = 'Detuning [EF]', ylabel = 'Magnitude')
			ax_CV.legend()
			plt.show()
			
		
		resultsTTF.append(TTF)
		resultstrf.append(trf)
		resultsls.append(convinterp)
		
	results = {'TTF':resultsTTF, 'TRF':resultstrf, 'LS':resultsls}
	df = pd.DataFrame(data=results)
	if save_pickle:
		df.to_pickle("convolutions.pkl")
		
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	