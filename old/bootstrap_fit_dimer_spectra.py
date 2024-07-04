# -*- coding: utf-8 -*-
"""
Nov 15 2023
@author: Colin Dale
	Editted heavily from code originally made by Ben Olsen.
"""

# Imports ######################################################################
import io
import time
import sys
import numpy as np
import os # for dealing with files/directories
import csv # for summary files containting fit results
import warnings # for suppressing warnings due to poorly-converging fits
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

from scipy.optimize import curve_fit
from math import sqrt, exp, sin, cos, atan
from data_class import Data  # experimental data pandas structure
from library import *
from fit_functions import NegGaussian
from tabulate import tabulate

# Initialization ###############################################################

do_profile = False # profile code execution (for optimizing runtime)
if do_profile:
    import cProfile, pstats
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()

#warnings.filterwarnings("ignore",category=RuntimeWarning)
warnings.filterwarnings("ignore")

def colorfunc(data_type): 
    # Choose the color of the plot/text/etc. based on the fit type
    cmapH = cm.get_cmap("hot")
    cmapC = cm.get_cmap("cool")
    if data_type == 'calibration': 
        return cmapH(2/7.)
    else: # wiggle measure
        return cmapC(5/11.)

### TESTING FLAGS ###
do_debug = True
# an initial debugging plot to make sure fit guesses are reasonable

## FITTING FLAGS ###
do_append = False
# append new fits to the old file
do_stats = True
# output results to summary csv files
do_plots = True
# output fit plots and histograms
trialsB = 100
# total Monte Carlo fits to perform

PARAM_INDEX = 1
WIDTH_LIMIT = 1
OFFSET_LIMIT = 1e5
PARAM_MIN = 42
PARAM_MAX = 45
CONFIDENCE_INTERVAL = 68.2689

dataset = "2023-11-14_F"
data_folder = "data\\" + dataset
bootstrap_folder = "bootstrap\\"
data_type = "dimer"

delay_times = np.linspace(0.05, 0.57, 14)
wiggle_freq = 2.5
field = 202.1

fit_func = NegGaussian
num = 500
file_prefix = dataset+"_e"

guess = None
if data_type == 'dimer':
	x_name = "freq"
	y_name = "sum95"
	xlabel = "Frequency (MHz)"
	ylabel = "Atom Number (9+7)"
	
elif data_type == 'calibration':
	x_name = "freq"
	y_name = "fraction95"
	xlabel = "Frequency (MHz)"
	ylabel = "Fractional Atom No. (7/7+9)"

timestr = time.strftime("%Y%m%d-%H%M%S")
folder = bootstrap_folder + dataset + "_" + timestr + '/'

# Load data #######################
try:
	os.remove(dataset + timestr + '/' + fit_func + '_' + dataset + '.csv')
except:
	print('Nothing to delete')
        
for delay in delay_times:
	delay_name = str(int(delay*1e3))+"us"
	print('**** Analyzing ' + dataset + " delay = " + delay_name + " ****")
	delay_tag = "_delay={:.2f}".format(delay)
	file = file_prefix + delay_tag + ".dat"
	run = Data(file, path=data_folder)
	run.B = field
	run.omega = wiggle_freq
	run.name = delay_name
    
	fit_data = np.array(run.data[[x_name, y_name]])
	nData = len(fit_data)
	func, auto_guess, fit_params = fit_func(fit_data)
	num_params = len(fit_params)
	
	xx = np.linspace(np.min(fit_data[:,0]),
				   np.max(fit_data[:,0]), num)
	
	if guess is None:
		pGuess = auto_guess
	else:
		pGuess = guess
	try:
		run.popt, run.pcov = curve_fit(func, fit_data[:,0],
									 fit_data[:,1], p0=pGuess)
		popt = run.popt
		run.perr = np.sqrt(np.diag(run.pcov))	
		perr = run.perr
	except RuntimeError: # guess params sucked, so plot and skip
		print("Unable to fit {:.2f} delay scan".format(delay))
		plt.plot(xx, func(xx, *pGuess), "--")
		plt.plot(fit_data[:,0], fit_data[:,1], '+r')
		plt.title("Guess fit for {:.2f} delay scan".format(delay))
		continue
	
	if do_debug:
		plt.plot(fit_data[:,0], fit_data[:,1], '+r')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		print("Guess:")
		print(pGuess)
		plt.plot(xx, (func(xx, *pGuess)), '--',color='purple', label="guess")
		print("Fit:")
		print(popt)
		print(perr)
		plt.plot(xx, (func(xx, *popt)), '-', color=colorfunc(data_type), 
			   label="fit")
		parameter_table = tabulate([['Guesses', *pGuess],
			  ['Values', *popt], ['Errors', *perr]], headers=fit_params)
		print(parameter_table)
		plt.legend()

	subfolder = folder + "delay_" + run.name + '/'

	try:
		os.makedirs(subfolder)
	except Exception:
		pass
        
# Bootstrap fitting ####################################################
        
	print("** Bootstrap resampling")
	xx = fit_data[:,0]
	yy = fit_data[:,1]
                    
	if os.path.isfile(subfolder + '/' + 'param_fits.txt'):
		pFitB = np.loadtxt(subfolder + '/' + 'param_fits.txt', delimiter=', ')
# 		print(np.shape(pFitB))
		old_trialsB = np.shape(pFitB)[0]
		# number of old trials loaded in
		print("   Loading from " + dataset + " file: %i old trials" % old_trialsB)
		if do_append:
			trialB = old_trialsB # starting position for new fits
			trialsB = trialsB-old_trialsB # update index
			if trialsB > 0:
				pFitB_old = pFitB.copy()
				pFitB = np.concatenate((pFitB,np.zeros([trialsB, num_params])))
				new_trialsB = True
			else:
				new_trialsB = False
		else:
			new_trialsB = False
	elif os.path.isfile(subfolder + '/bs_params.npy'):
		pFitB = np.load(subfolder + '/bs_params.npy')
		old_trialsB = np.shape(pFitB)[0]
		print("   Loading from npy file: %i old trials" % old_trialsB)
		if do_append:
			trialB = old_trialsB # starting position for new fits
			trialsB = trialsB-old_trialsB # update index
			if trialsB > 0:
				pFitB_old = pFitB.copy()
				pFitB = np.concatenate((pFitB,np.zeros([trialsB, 6, 3*num_params])))
				new_trialsB = True
			else:
				new_trialsB = False
		else:
			new_trialsB = False
		#print(pFitB)
# 		print(np.shape(pFitB))
# 		print(pFitB[0,:num_params,0])
# 		print(pFitB[0,:num_params,1])
# 		print(pFitB[0,:num_params,2])
# 		print(pFitB[0,:num_params,3])
# 		print(pFitB[0,:num_params,4])
# 		print(pFitB[0,:num_params,5])
# 		print(pFitB[0,:num_params,6])
		np.savetxt(subfolder + '/param_fits.txt', pFitB[:,0:num_params,0], delimiter= ', ')
	else:
		old_trialsB = 0

		pFitB = np.zeros([trialsB, num_params])
		trialB = 0 # starting position for new fits
		new_trialsB = True

            
	if new_trialsB:
		print('   Running %i new trials to reach %i total' % (trialsB, old_trialsB+trialsB))

		fracChoose = 1.0
		# fraction of original number of data points for bootstrap resampling
		nChoose = int(np.floor(fracChoose*nData))
		# make into an integer number of points to select
		fails = 0
		# counter for failed fits
	
		trialB = 0
		fails = 0
            
		while (trialB < trialsB) and (fails < trialsB):
			if (trialsB <= 100) or (0 == trialB % (trialsB / 5)):
				print('   %d of %d @ %s' % (trialB, trialsB, time.strftime("%H:%M:%S", time.localtime())))
                    
			inds = np.random.choice(np.arange(0, nData), nChoose, replace=True)
			xTrial = np.random.normal(np.take(xx, inds), 1)
			# we need to make sure there are no duplicate x values or the fit
			# will fail to converge
			yTrial = np.take(yy, inds)
			p = xTrial.argsort()
                
			xTrial = xTrial[p]
			yTrial = yTrial[p]

			try:
				# pylint: disable=unbalanced-tuple-unpacking
				pFit, cov = curve_fit(func, xTrial, yTrial, pGuess)
			except Exception:
				print("Failed to converge")
				fails += 1
				continue
                
			if np.sum(np.isinf(trialB)) or abs(pFit[0]) > abs(pFit[3]) \
										or PARAM_MIN > pFit[1] > PARAM_MAX \
										or abs(pFit[2]) > WIDTH_LIMIT \
										or pFit[3] > OFFSET_LIMIT or pFit[3] < 0:
											
				print('Fit params out of bounds')
				print(pFit)
				continue
			else:
				pFitB[trialB, :] = pFit
				trialB += 1

# 	print(pFitB[:,0:num_params])
	np.savetxt(subfolder + '/' + 'param_fits.txt', pFitB[:,0:num_params], delimiter= ', ')

	conf = CONFIDENCE_INTERVAL  # confidence level for CI

	if do_stats:
		print("** Stats")
		fits = pFitB
            # for fit in range(np.size(fits,axis=0)):
                # if (fits[fit,3] < mincut) or (fits[fit,3] > maxcut):
                #     fits[fit] = np.nan
            
		median_param = np.nanmedian(fits[:, PARAM_INDEX])
		upper_param = np.nanpercentile(fits[:, PARAM_INDEX], 100-(100.0-conf)/2.)
		lower_param = np.nanpercentile(fits[:, PARAM_INDEX], (100.0-conf)/2.)
            
		csvPath = folder + dataset + '.csv'
		print(csvPath)
		if not os.path.isfile(csvPath):
			with open(csvPath, 'w', newline='') as csvFile:
				csvWriter = csv.writer(csvFile, delimiter=',')
				csvWriter.writerow(['Run           ', 'B (G)       ', 'omega (kHz)     ',
								  'Amplitude   ', 'f0 (MHz)     ', 'Width      ', 'Offset      ', 
								  'median    ', 'lower     ', 'upper     ',
								  'sigma_M    ', 'sigma_P    ',
								  'N_fits      '])
				csvWriter.writerow(['{:<14}'.format(run.name),
                                        '{:12.2f}'.format(run.B),
                                        '{:12.4f}'.format(run.omega),
                                        '{:12.2f}'.format(np.nanmedian(fits[:, 0])),
                                        '{:12.4f}'.format(np.nanmedian(fits[:, 1])),
                                        '{:12.2f}'.format(np.nanmedian(fits[:, 2])),
                                        '{:12.2f}'.format(np.nanmedian(fits[:, 3])),
                                        '{:12.2f}'.format(median_param),
                                        '{:12.2f}'.format(lower_param),
                                        '{:12.2f}'.format(upper_param),
                                        '{:12.2f}'.format(median_param - lower_param),
                                        '{:12.2f}'.format(upper_param - median_param),
                                        '{:12}'.format(np.size(fits,axis=0))
                                        ])
		else:
			with open(csvPath, 'a', newline='') as csvFile:
				csvWriter = csv.writer(csvFile, delimiter=',')
				csvWriter.writerow(['{:<14}'.format(run.name),
                                        '{:12.2f}'.format(run.B),
                                        '{:12.4f}'.format(run.omega),
                                        '{:12.2f}'.format(np.nanmedian(fits[:, 0])),
                                        '{:12.4f}'.format(np.nanmedian(fits[:, 1])),
                                        '{:12.2f}'.format(np.nanmedian(fits[:, 2])),
                                        '{:12.2f}'.format(np.nanmedian(fits[:, 3])),
                                        '{:12.2f}'.format(median_param),
                                        '{:12.2f}'.format(lower_param),
                                        '{:12.2f}'.format(upper_param),
                                        '{:12.2f}'.format(median_param - lower_param),
                                        '{:12.2f}'.format(upper_param - median_param),
                                        '{:12}'.format(np.size(fits,axis=0))
                                        ])

# Plotting #####################################################################

		if do_plots:
			print('** Plots')
            # display vector for plotting
			numpts = 1000
			xFit = np.linspace(np.min(xx), np.max(xx), numpts)
                
			fig_size = plt.rcParams["figure.figsize"]
			fig_size[0] = 10
			fig_size[1] = 8
			plt.rcParams["figure.figsize"] = fig_size

			fig = plt.figure()
			fits = pFitB

			pMed = np.zeros(num_params)
			for index in range(len(pMed)):
				pMed[index] = np.nanmedian(fits[:, index])
# 			print(pMed)

			plt.plot(xFit, func(xFit, *pMed), color=colorfunc(dataset), alpha=.2)
            
            # plt.axvline(x = np.nanmedian(fits[:, 3]), color=colorfunc(dataset), alpha=0.5)
            # plt.axvline(x = np.nanpercentile(fits[:, 3], (100.-conf)/2.), color=colorfunc(dataset), alpha=0.5)
            # plt.axvline(x = np.nanpercentile(fits[:, 3], 100.-(100.-conf)/2.), color=colorfunc(dataset), alpha=0.5)
        
			do_plotCI = True
			if do_plotCI:
				print('*** Plotting Confidence Band')
				iis = range(0,np.shape(fits)[0])
				ylo1 = 0*xFit
				yhi1 = 0*xFit
				for idx, xval in enumerate(xFit):
# 					print('    %i of %i @ %s' % (idx, numpts,time.strftime("%H:%M:%S", time.localtime())))
                    #print(xval)
                    #print([fits[ii,0],fits[ii,1],fits[ii,2],-1*qb_factor*fits[ii,3]])
                    #FDW_qb, FCQB_FD_int
					yvals = [func(xval,*fits[ii,:num_params]) for ii in iis]
                    #fvals = [func(xval,fits[ii,0],fits[ii,1],fits[ii,2],fits[ii,3]) for ii in iis] # change for bound (ii,:-2)/ QB (ii,:)
                    #print(fvals)
					ylo1[idx] = np.nanpercentile(yvals,(100.-conf)/2.)
					yhi1[idx] = np.nanpercentile(yvals,100.-(100.-conf)/2.)
    
                    #print(np.percentile(func(xval,*pFitB[:,:-2]),(100.-conf)/2.))
                    
                #print(np.shape(np.percentile([func(xFit,*pFitB[ii,:-2]) for ii in range(0,np.shape(pFitB)[0])])))

				plt.plot(xFit, ylo1, color=colorfunc(dataset))
				plt.plot(xFit, yhi1, color=colorfunc(dataset))
            
			plt.plot(xx, yy, '.')
            #plt.plot(xFit, func(xFit, *pGuess), 'k--')
			plt.xlim([np.nanmin(xFit),np.nanmax(xFit)])
            
            #plt.ylim([0,np.nanmax(yy)*1.1])
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.suptitle('[' + dataset + '/' + run.name + "] " + dataset + 
                         " Bootstrap resampling ({:.2f} % confidence), {} trails".format(conf, trialsB))
            #plt.show()
			print(subfolder + dataset + '_Fit.pdf')
			plt.savefig(subfolder + dataset + '_Fit.pdf')
			plt.close()

			fig2 = plt.figure()
            
			fits = pFitB[:,:]
            #print(fits)

            # for fit in range(np.size(fits,axis=0)):
            #     if (fits[fit,3] < mincut) or (fits[fit,3] > maxcut):#(fits[fit,3] < 10) or (fits[fit,3] > 1000):
            #         fits[fit] = np.nan

			for i in range(0, num_params): 
				ax1 = plt.subplot(2, 2, i+1)

				median_p = np.nanmedian(fits[:, i])
				#print(median_f)
				upper_p = np.nanpercentile(fits[:, i], 100-(100.0-conf)/2.)
				lower_p = np.nanpercentile(fits[:, i], (100.0-conf)/2.)

				array = fits[:, i]
				finite = array [~np.isnan(array)]
                # print(finite)
				ax1.hist(finite, bins=50, alpha=0.5, color=colorfunc(dataset),
                        range=(np.nanpercentile(finite,1.0),np.nanpercentile(finite,99.0)))
				ax1.axvline(x=median_p, c=colorfunc(dataset),
                            label='BS: %.4f-%.4f+%.4f' % (median_p, median_p - lower_p, upper_p - median_p))
				ax1.axvline(x=upper_p, c=colorfunc(dataset),alpha=0.5)
				ax1.axvline(x=lower_p, c=colorfunc(dataset),alpha=0.5)
                #ax1.axvline(x=median+upperB, c='b',alpha=0.5)
                #ax1.axvline(x=median-lowerB, c='b',alpha=0.5)
                #ax1.axvline(x=median+upper95, c='r',alpha=0.5)
                #ax1.axvline(x=median-lower95, c='r',alpha=0.5)

				ax1.legend(
                    bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                    mode="expand", borderaxespad=0.)

				ax1.set_xlabel(fit_params[i])

			fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
			fig2.suptitle('[' + dataset + '/' + run.name + 
                          '] param fit parameter distributions ({:.2f} % confidence), {} trials'.format(conf, trialsB))

			fig2.savefig(subfolder + dataset + '_Hists.pdf')
			plt.close()

			fig3 = plt.figure()
			for i in range(0, num_params):
				for j in range(i+1, num_params):
					ax3 = plt.subplot(num_params-1, num_params-1, (num_params-1)*i+j)
					ax3.scatter(pFitB[:, j], pFitB[:, i], color=colorfunc(dataset), s=10,
                                edgecolor='none', alpha=0.25)

					ax3.set_xlabel(fit_params[j])
					ax3.set_ylabel(fit_params[i])
			fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
			fig3.suptitle('[' + dataset + '/' + run.name + "] Fit parameter correlations")

			fig3.savefig(subfolder + dataset + '_Correlations.pdf')
			plt.close()