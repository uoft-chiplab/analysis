# -*- coding: utf-8 -*-
"""
@author: Colin Dale
2025-04-09
"""
import numpy as np
import time
from scipy.optimize import curve_fit
from scipy._lib._util import getfullargspec_no_self as _getfullargspec

def bootstrap_fit(fit_func, xs, ys, p0=None, pbounds=None, trials=100, 
				  x_shift=None, conf=68.2689, talk=True):
	"""Bootstrap resamples xs and ys lists of points, fitting with 
	scipy.optimize.curve_fit

    Parameters
    ----------
    fit_func : function
        A function with first argument x, and all other arguments are fit
		parameters.
	xs : list
		A list of x data points to sample for fit. Indices match ys.
	ys : list
		A list of y data points to sample for fit. 
	
	Optional Arguments
	------------------
	p0 : list or None
		A list of guess starting parameters to pass to curve_fit. Default
		is None.
	pbounds : list or None
		A list of bounds for optimal parameters. Default is None. Format as
		[[p1_min, p1_max], ... [pn_min, pn_max]] to bound all parameters or
		if any param should be unbounded, replace min max pair with None, e.g.
		[None, [p2_min, p2_max], ...]. These bounds are not passed to 
		curve_fit, but the popts of curev_fit are checked after fitting. 
		Bootstrap fitting fails if too many params are out of bounds.
	trials : int
		Number of bootstrapping trials.
	x_shift : float
		Width of normal distribution sampled to shift x points by s.t. there 
		are no duplicate x values so the fit will converge. Should be much 
		smaller than the point spacing. Default is a crude estimate of min 
		difference between x values divided by 10,000.
	conf : float
		Perentile limit for upper and lower confidence intervals. Default
		is the usual one sigma, or 68.2689%.
	talk : If true, function will print some timing information.

    Returns
    -------
    popts : list of lists
		A list of best-fit params from successful fits.
	popt_stats_dicts : list of dicts
		A list of dictionaries, one for each fit param, with keys for stats
		results for the param distributions:
			median - the median 
			upper - the upper confidence interval edge
			lower - the lower confidence interval edge
			mean - the mean
			std - the standard deviation

    """
	
	if talk:
		print("** Bootstrap resampling")
	trial = 0 # trials counter
	fails = 0 # failed fits counter
	OOBfails = 0 # number of out of bound fit params failures
	nData = len(xs)
	nChoose = nData # number of points to choose
	
	if x_shift is None:
		x_shift = np.abs(min(np.diff(xs)))*1e-5 # small change in x, see later
	
	popts = [] # initialize best-fit params list
	OOBpopts = [] # out of bounds best-fit params list, for debugging
	
	# loop through trials, but stop if fails = trials
	while (trial < trials) and (fails < trials) and (OOBfails < trials):
		if (0 == trial % (trials / 10)) and talk: # show time every once in a while
 			print('   %d of %d @ %s' % (trial, trials, 
							time.strftime("%H:%M:%S", time.localtime())))
		
		# choose indices with replacement
		inds = np.random.choice(np.arange(0, nData), nChoose, replace=True)
		
		# we need to make sure there are no duplicate x values or the fit
		# will fail to converge, fix this in a dumb way by shifting x values 
		# randomly by a tiny bit
		xTrial = np.random.normal(np.take(xs, inds), x_shift)
		
		yTrial = np.take(ys, inds)
		p = xTrial.argsort()
		xTrial = xTrial[p]
		yTrial = yTrial[p]
			
		# try to fit data and see if best-fit params converge
		try:
			popt, pcov = curve_fit(fit_func, xTrial, yTrial, p0=p0)
		except Exception:
			if talk:
				print("Failed to converge")
			fails += 1
			continue
		
		# check fit params within defined bounds
		if pbounds is not None:
			# loop over params, check if in bounds
			for p, pbound in zip(popt, pbounds):
				if pbound is not None:
					# robust check agianst misordering of min and max
					if not min(*pbound) < p < max(*pbound):
						OOBfails += 1 # out of bounds fail
						OOBpopts.append(popt)
						
		
		trial += 1 # trial successful
		popts.append(popt) # add best-fit params to list
		
	# throw error if too many fails
	if fails >= trials:
		raise RuntimeError("""RuntimeError: Optimal parameters not found: 
					 Number of failed bootstrap fits has reached the number
					 of trials: """, fails)
					 
    # throw error if too many out of bound fails
	if OOBfails >= trials:
		raise RuntimeError("""RuntimeError: Too many optimal fit params 
					 outside of bounds. Check pbounds input. Here are the
					 out of bounds popts: """, OOBpopts)
					 
	# get distributions of best-fit params by transposing popts
	popt_dists = np.transpose(popts)
	
	# make list of dicts with stats results
	popt_stats_dicts = [dist_stats(popt_dist, conf=conf) for popt_dist in popt_dists]
	
	# return everything
	return popts, popt_stats_dicts

def dist_stats(dist, conf=68.2689):
	""" Computes the median, upper confidence interval (conf), lower conf, mean 
		and standard deviation for a distribution named dist. Returns a dict."""
		
	return_dict = {
		'median': np.nanmedian(dist),
		'upper': np.nanpercentile(dist, 100-(100.0-conf)/2.),
		'lower': np.nanpercentile(dist, (100.0-conf)/2.),
		'mean': np.mean(dist),
		'std': np.std(dist)
		}
	return return_dict

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

# not used, but could be implimented for more debugging checks
def count_fit_params(fit_func):
	""" counts fit params for function, taken from curve_fit definition """
	# determine number of parameters by inspecting the function
	sig = _getfullargspec(fit_func)
	args = sig.args
	if len(args) < 2:
		raise ValueError("Unable to determine number of fit parameters.")
	num_p = len(args) - 1
	return num_p