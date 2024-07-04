# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:04:26 2024
@author: Chip Lab
"""

import numpy as np
from scipy.optimize import curve_fit

def MonteCarlo_spectra_fit_trapz(xs, ys, yserr, fitmask, xstar, fit_func, 
									num_iter=1000):
	""" Computes trapz for interpolated list of data points (xs, ys+-yserr),
	which is extrapolated using fit_func out to max(xs). Estimates std dev of 
	result by sampling ys and yserr from Gaussian distributions, and fitting
	to this sample, num_iter (default 1000) times."""
	
	def dwSpectra(xi, x_star):
		return 2*(1/np.sqrt(xi)-np.arctan(np.sqrt(x_star/xi))/np.sqrt(x_star))

	def wdwSpectra(xi, x_star):
		return 2*np.sqrt(x_star)*np.arctan(np.sqrt(x_star/xi))
	
	def rand_y(y, yerr, size):
		generator = np.random.default_rng()
		return generator.normal(loc=y, scale=yerr, size=num_iter)
	# array of lists of y vals, from Gaussians with centres y and widths yerr
	ys_iter = np.array([rand_y(y, yerr, num_iter) for y, 
					 yerr in zip(ys, yserr)])
	
	popts = []
	pcovs = []
	# fit to determine lineshape for each iteration
	for i in range(num_iter):
		ys_fit = ys_iter[:,i]
		popt, pcov = curve_fit(fit_func, xs[fitmask], ys_fit[fitmask])
		popts.append(popt)
		pcovs.append(pcov)
	
	# extrapolation starting point
	xi = max(xs)
	
	# interpolation array for x, num in size
	num = 1000 # points to integrate over
	xs_interp = np.linspace(min(xs), xi, num)
	
	# compute interpolation array for y, num by num_iter in size
	ys_interp_iter = np.array([[np.interp(xi, xs, ys_iter[:,i]) \
							 for xi in xs_interp] for i in range(num_iter)])
	
	# integral from xi to infty
	SR_extrapolations = np.array(popts)[:,0]*dwSpectra(xi, xstar)
	FM_extrapolations = np.array(popts)[:,0]*wdwSpectra(xi, xstar)
	
	# for the integration, we first sum the interpolation, 
	# then the extrapolation, then we add the analytic -5/2s portion
	
	# sumrule using each set
	SR_distr = np.array([np.trapz(ys_interp_iter[i], x=xs_interp) \
			+ SR_extrapolations[i] for i in range(num_iter)])
	# first moment using each set	
	FM_distr = np.array([np.trapz(ys_interp_iter[i]*xs_interp, x=xs_interp) \
			+ FM_extrapolations[i] for i in range(num_iter)])
	
	# clock shift
	# we need to do this sample by sample so we have correlated SR and FM
	CS_distr = np.array([FM/SR for FM, SR in zip(FM_distr, SR_distr)])
	
	SR_mean, e_SR = (np.mean(SR_distr), np.std(SR_distr))
	FM_mean, e_FM = (np.mean(FM_distr), np.std(FM_distr))
	CS_mean, e_CS = (np.mean(CS_distr), np.std(CS_distr))
	
	# return everything
	return SR_distr, SR_mean, e_SR, FM_distr, FM_mean, e_FM, CS_distr, \
		CS_mean, e_CS, popts, pcovs

# def MonteCarlo_trapz(xs, ys, yserr, num_iter=1000):
# 	""" Computes trapz for list of data points (xs, ys+-yserr),
# 	and estimates std dev of result by sampling ys and yserr from 
# 	Gaussian distributions, num_iter (default 1000) times."""
# # 	value = np.trapz(ys, x=xs)
# 	def rand_y(y, yerr, size):
# 		generator = np.random.default_rng()
# 		return generator.normal(loc=y, scale=yerr, size=num_iter)
# 	# array of lists of y values, sampled from Gaussians with centres y and widths yerr
# 	ys_iter = np.array([rand_y(y, yerr, num_iter) for y, yerr in zip(ys, yserr)])
# 	values = np.array([np.trapz(ys_iter[:,i], x=xs) for i in range(num_iter)])
# 	distr_mean, distr_stdev = (np.mean(values), np.std(values))
# 	return values, distr_mean, distr_stdev


# def MonteCarlo_interp_trapz(xs, ys, yserr, num_iter=1000):
# 	""" Computes trapz for interpolated list of data points (xs, ys+-yserr),
# 	and estimates std dev of result by sampling ys and yserr from 
# 	Gaussian distributions, num_iter (default 1000) times."""
# # 	value = np.trapz(ys, x=xs)
# 	def rand_y(y, yerr, size):
# 		generator = np.random.default_rng()
# 		return generator.normal(loc=y, scale=yerr, size=num_iter)
# 	# array of lists of y values, sampled from Gaussians with centres y and widths yerr
# 	ys_iter = np.array([rand_y(y, yerr, num_iter) for y, yerr in zip(ys, yserr)])
# 	
# 	# interpolation array for x, num_iter in size
# 	xs_interp = np.linspace(min(xs), max(xs), num_iter)
# 	
# 	# compute interpolation array for y, num_iter by num_iter in size
# 	ys_interp_iter = np.array([[np.interp(xi, xs, ys_iter[:,i]) for xi in xs_interp]
# 					  for i in range(num_iter)])
# 	
# 	# integrals using each interpolation set
# 	values = np.array([np.trapz(ys_interp_iter[i], x=xs_interp) for i in range(num_iter)])
# 	
# 	distr_mean, distr_stdev = (np.mean(values), np.std(values))
# 	return values, distr_mean, distr_stdev

# def MonteCarlo_interp_extrap_trapz(xs, ys, yserr, xmax, 
# 								   fit_func, num_iter=1000):
# 	""" Computes trapz for interpolated list of data points (xs, ys+-yserr),
# 	which is extrapolated using fit_func out to xmax. Estimates std dev of 
# 	result by sampling ys and yserr from Gaussian distributions, and fitting
# 	to this sample, num_iter (default 1000) times."""
# 	def rand_y(y, yerr, size):
# 		generator = np.random.default_rng()
# 		return generator.normal(loc=y, scale=yerr, size=num_iter)
# 	# array of lists of y vals, from Gaussians with centres y and widths yerr
# 	ys_iter = np.array([rand_y(y, yerr, num_iter) for y, 
# 					 yerr in zip(ys, yserr)])
# 	
# 	fits = np.array([curve_fit(fit_func, xs, ys_iter[:,i]) \
# 					for i in range(num_iter)])
# 		
# 	popts = fits[:,0]
# 	pcovs = fits[:,1]
# 	
# 	# interpolation array for x, num_iter in size
# 	xs_interp = np.linspace(min(xs), max(xs), num_iter)
# 	# extrapolation array for x, num_iter in size
# 	xs_extrap = np.linspace(max(xs), xmax, num_iter)
# 	
# 	# compute interpolation array for y, num_iter by num_iter in size
# 	ys_interp_iter = np.array([[np.interp(xi, xs, ys_iter[:,i]) \
# 							 for xi in xs_interp] for i in range(num_iter)])
# 	
# 	# integrals using each interpolation set
# 	values = np.array([np.trapz(ys_interp_iter[i], x=xs_interp) \
# 					+ np.trapz(fit_func(xs_extrap, *popts[i]), x=xs_extrap) \
# 					for i in range(num_iter)])
# 	
# 	distr_mean, distr_stdev = (np.mean(values), np.std(values))
# 	return values, distr_mean, distr_stdev, popts, pcovs