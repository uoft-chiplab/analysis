# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:04:26 2024
@author: Chip Lab
"""

import numpy as np
from scipy.optimize import curve_fit
import time
import matplotlib.pyplot as plt

		
def Bootstrap_spectra_fit_trapz(xs, ys, xfitlims, fit_func, trialsB=1000, 
								wave='p', pGuess=[1], debug=False):
	""" """
	if wave == 'p':
		def dwSpectra(xi, x_star):
			return 2*(1/np.sqrt(xi)-np.arctan(np.sqrt(x_star/xi))/np.sqrt(x_star))
	
	if debug == True:
		trialsB = 10
	
	num = 5000 # points to integrate over
	
	print("** Bootstrap resampling")
	trialB = 0 # trials counter
	fails = 0 # failed fits counter
	nData = len(xs)
	nChoose = nData # number of points to choose
	pFitB = np.zeros([trialsB, len(pGuess)]) # array of fit params
	SR_distr = []
	A_distr = []
	
	while (trialB < trialsB) and (fails < trialsB):
		if (0 == trialB % (trialsB / 5)):
 			print('   %d of %d @ %s' % (trialB, trialsB, 
						time.strftime("%H:%M:%S", time.localtime())))
		 
		inds = np.random.choice(np.arange(0, nData), nChoose, replace=True)
		# we need to make sure there are no duplicate x values or the fit
		# will fail to converge. do this by adding random offsets...
		xTrial = np.random.normal(np.take(xs, inds), 0.0001)
		
		yTrial = np.take(ys, inds)
		p = xTrial.argsort()
		xTrial = xTrial[p]
		yTrial = yTrial[p]
		
		fitpoints = np.array([[xfit, yfit] for xfit, yfit in zip(xTrial, yTrial) \
						if (xfit > xfitlims[0] and xfit < xfitlims[-1])])

		if debug == True:
			print([min(fitpoints[:,0]),max(fitpoints[:,0])])

		try:
			# pylint: disable=unbalanced-tuple-unpacking
			pFit, cov = curve_fit(fit_func, fitpoints[:,0], 
						 fitpoints[:,1], pGuess)
		except Exception:
			print("Failed to converge")
			fails += 1
			continue
		
		if np.sum(np.isinf(trialB)) or pFit[0]<0:
			print('Fit params out of bounds')
			print(pFit)
			continue
		else:
			pFitB[trialB, :] = pFit
			trialB += 1
	
		# extrapolation starting point
		xi = max(xTrial)
	
		# select interpolation points to not have funny business
		interp_points = np.array([[x, y] for x, y in zip(xTrial, yTrial) if (x > -2)])
		
		# interpolation array for x, num in size
		x_interp = np.linspace(min(interp_points[:,0]), xi, num)
	
		# compute interpolation array for y, num by num_iter in size
		y_interp = np.array([np.interp(x, interp_points[:,0], 
								 interp_points[:,1]) for x in x_interp])
	
		# for the integration, we first sum the interpolation, 
		# then the extrapolation, then we add the analytic -5/2s portion
		
		# sumrule using each set
		SR = np.trapz(y_interp, x=x_interp)
		
		if SR<0 or pFit[0]<0 or pFit[0]>100:
			print("Integration out of bounds")
			continue # don't append trial to lists
	
		SR_distr.append(SR)
		A_distr.append(pFit[0])
			
	# return everything
	return np.array(SR_distr), np.array(A_distr)
