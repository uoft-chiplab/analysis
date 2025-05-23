# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:04:26 2024
@author: Chip Lab
"""
import numpy as np
from scipy.optimize import curve_fit
import time

def MonteCarlo_estimate_std_from_function(func, inputs, input_errors, num=100, **kwargs):
	""" Sample output of function from calibration values distributed normally 
	to obtain std"""
	# sample output of function from calibration values distributed normally to obtain std
	dist = []
	i = 0
	while i < num:
		dist.append(func(*[np.random.normal(val, err) for val, err \
					 in zip(inputs, input_errors)], **kwargs))
		i += 1
	return np.array(dist).mean(), np.array(dist).std()

def Multivariate_MonteCarlo_estimate_std_from_function(func, means, cov, num=100, **kwargs):
	""" Sample output of function from calibration values distributed normally
		with covariance cov about their means to obtain std."""
	dist = []
	i = 0
	while i < num:
		dist.append(func(*np.random.multivariate_normal(means, cov), **kwargs))
		i += 1
	return np.array(dist).mean(), np.array(dist).std()


def dist_stats(dist, CI):
	""" Computes the median, upper confidence interval (CI), lower CI, mean 
		and standard deviation for a distribution named dist. Returns a dict."""
	return_dict = {
		'median': np.nanmedian(dist),
		'upper': np.nanpercentile(dist, 100-(100.0-CI)/2.),
		'lower': np.nanpercentile(dist, (100.0-CI)/2.),
		'mean': np.mean(dist),
		'std': np.std(dist)}
	return return_dict

def Bootstrap_spectra_fit_trapz(xs, ys, xfitlims, xstar, fit_func, 
									trialsB=100, pGuess=[1], debug=False):
	""" """
	def dwSpectra(xi, x_star):
		return 2*(1/np.sqrt(xi)-np.arctan(np.sqrt(x_star/xi))/np.sqrt(x_star))

	def wdwSpectra(xi, x_star):	
		return 2*np.sqrt(x_star)*np.arctan(np.sqrt(x_star/xi))
	
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
	FM_distr = []
	CS_distr = []
	SR_extrap_distr = []
	FM_extrap_distr = []
	A_distr = []
	extrapstart = []
	
	while (trialB < trialsB) and (fails < trialsB):
		if (0 == trialB % (trialsB / 5)):
 			print('   %d of %d @ %s' % (trialB, trialsB, time.strftime("%H:%M:%S", time.localtime()))
		  )
		 
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
	
		# integral from xi to infty
		SR_extrapolation = pFit[0]*dwSpectra(xi, xstar)
		FM_extrapolation = pFit[0]*wdwSpectra(xi, xstar)
	
		# for the integration, we first sum the interpolation, 
		# then the extrapolation, then we add the analytic -5/2s portion
		
		# sumrule using each set
		SR = np.trapz(y_interp, x=x_interp) + SR_extrapolation
		
		
		# first moment using each set	
		FM = np.trapz(y_interp*x_interp, x=x_interp) + FM_extrapolation
	
		# clock shift
		# we need to do this sample by sample so we have correlated SR and FM
		CS = FM/SR
		
		if SR<0 or CS<0 or CS>100:
			print("Integration out of bounds")
			continue # don't append trial to lists
		
		extrapstart.append(xi)
	
		SR_extrap_distr.append(SR_extrapolation)
		FM_extrap_distr.append(FM_extrapolation)
		
		CS_distr.append(CS)
		FM_distr.append(FM)
		SR_distr.append(SR)
		
		A_distr.append(pFit[0])
			
	# return everything
	return np.array(SR_distr), np.array(FM_distr), np.array(CS_distr), \
				np.array(A_distr), np.array(SR_extrap_distr), np.array(FM_extrap_distr), \
					np.array(extrapstart)

def DimerBootStrapFit(xs, ys, int_bounds, fit_func, trialsB=100, 
					  pGuess=[0.02, -260], interp_data=False):
	""""""
	
	num = 5000 # points to integrate over
	
	print("** Bootstrap resampling")
	trialB = 0 # trials counter
	fails = 0 # failed fits counter
	nData = len(xs)
	nChoose = nData # number of points to choose
	SR_distr = []
	FM_distr = []
	CS_distr = []
	
	while (trialB < trialsB) and (fails < trialsB):
		if (0 == trialB % (trialsB / 5)):
 			print('   %d of %d @ %s' % (trialB, trialsB, 
							time.strftime("%H:%M:%S", time.localtime())))
		inds = np.random.choice(np.arange(0, nData), nChoose, replace=True)
		xTrial = np.random.normal(np.take(xs, inds), 0.0001)
		# we need to make sure there are no duplicate x values or the fit
		# will fail to converge
		yTrial = np.take(ys, inds)
		p = xTrial.argsort()
		xTrial = xTrial[p]
		yTrial = yTrial[p]
	
		# ignore fit and integrate interpolated data, check if results in bounds
		if interp_data == True:
			# interpolation array for x, num in size
			x_interp = np.linspace(min(xTrial), max(xTrial), num)
			# compute interpolation array for y, num by num_iter in size
			y_interp = np.array([np.interp(x, xTrial, yTrial) for x in x_interp])
			# sumrule using each set
			SR = np.trapz(y_interp, x=x_interp) 
			# first moment using each set	
			FM = np.trapz(y_interp*x_interp, x=x_interp) 
			# clock shift
			CS = FM / 0.5
			
			if SR < 0:
				print("SR out of bounds", SR)
				fails += 1
				continue
			else:
				trialB += 1
			
		# fit data and check fit params within bounds
		else:
			try:
				popt_BS, pcov_BS = curve_fit(fit_func, xTrial, 
							 yTrial, pGuess)
			except Exception:
				print("Failed to converge")
				fails += 1
				continue
			
			if np.sum(np.isinf(trialB)) or popt_BS[0]<0:
				print('Fit params out of bounds', popt_BS)
				continue
			
			else:
				# compute SR, FM and CS
				xeval = np.linspace(*int_bounds, num)
				SR = np.trapz(fit_func(xeval, *popt_BS), x=xeval)
				FM = np.trapz(xeval*fit_func(xeval, *popt_BS), x=xeval)
				CS = FM/0.5
				
				trialB += 1
				
		
		CS_distr.append(CS)
		FM_distr.append(FM)
		SR_distr.append(SR)
	
	# return everything
	return SR_distr, FM_distr, CS_distr

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