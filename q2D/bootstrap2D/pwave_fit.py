# -*- coding: utf-8 -*-
"""
July 14 2023
@author: Colin Dale
	Code originally made by Ben Olsen.
"""

# Imports ######################################################################
import io
import time
import sys
import numpy as np

from scipy import optimize # nonlinear fitting
from scipy import constants # CODATA constants for physical parameters
from scipy import integrate
from math import sqrt, exp, sin, cos, atan 
import os # for dealing with files/directories
import csv # for summary files containting fit results
import warnings # for suppressing warnings due to poorly-converging fits

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import dimer_data  # experimental data stored in another file

# Initialization ###############################################################

do_profile = False # profile code execution (for optimizing runtime)
if do_profile:
	import cProfile, pstats
	from pstats import SortKey
	pr = cProfile.Profile()
	pr.enable()

warnings.filterwarnings("ignore")

# use CODATA18 for physical constant in SI units
h = constants.value('Planck constant')  # value of Planck constant in J s
hbar = h/(2*np.pi)  # value of reduced Planck constant in J s
uB = constants.value('Bohr magneton')  # value of Planck constant in J s
aB = constants.value('Bohr radius')  # value of Bohr radius in m
u = constants.value('atomic mass constant')  # value of Planck constant in J s
mK40 = 39.96399848 * u  # mass K-40 in kg
dmu = .134*uB  # differential mag. moment between -9/2, -7/2 K40 states (J/T)
kB = constants.value('Boltzmann constant')  # value of Planck constant in J s
pi = np.pi
sqrt12 = np.sqrt(0.5)

# FR parameters
Rvdw = 65.02*aB  # Faulke 2008
B0xy = 0.01983  # xy resonance location in T
DBxy = -.002455  # xy width in T
vbgxy = -905505 * aB**3  # xy bg scattering volume
B0z = .01988  # z resonance location in T
DBz = -0.002195  # z resonance width in T
vbgz = -1040850 * aB**3  # z bg scattering volume


def colorfunc(fitType): 
	# Choose the color of the plot/text/etc. based on the fit type
	cmapH = cm.get_cmap("hot")
	cmapC = cm.get_cmap("cool")
	if fitType == 'MB': # Maxwell-Boltzmann
		return cmapH(2/7.)
	else: # Fermi-Dirac
		return cmapC(5/11.)

# Fit Functions ################################################################

def Reffxy(B): 
	# Effective range for the xy resonance
	return -vbgxy*mK40*(-B+B0xy+DBxy)**2*dmu/(DBxy*hbar**2)


def Reffz(B): 
	# Effective Range for the z resonance
	return -vbgz*mK40*(-B+B0z+DBz)**2*dmu/(DBz*hbar**2)

Reff = Reffz(200) # choose a junk Reff to prevent odd errors

def FC0(k, ka, Reff): 
	# Franck-Condon overlaps
	return mK40*abs(Reff)/(np.pi*hbar**2) * k**3/((k**2+ka**2)**2)


def fit_gauss(f, Ni, A, width, fb):
	# gaussian centered around -fb (sign chosen to agree with other fits)
	return Ni*(1 - .1*A*np.exp(-(f-fb)**2/2/width**2))


def kk(f, fb): 
	# scattering wave vector using the sign convention that fb is negative
	return np.sqrt(h*mK40*abs(f+fb)*1000/hbar/hbar)


def ka(fb): 
	# resonance pole value kappa
	return np.sqrt(h*mK40*abs(fb)*1000/hbar**2)


def fit_MB_bound(f, Ni, A, T, fb): 
	# fit using Maxwell-Boltzmann statistics
	return np.piecewise(
		f, [f <= -fb],
		[lambda f: Ni,
		# when f < -fb
			lambda f: Ni*(1 -
						   1e-32 * A * kk(f,fb)**2/(4*np.pi**3) *
						   np.sqrt(np.pi*hbar**2/(mK40*kB*T*1e-9)) *
						   np.exp(-kk(f,fb)**2*hbar**2/(mK40*kB*T*1e-9)) *
						   FC0(kk(f,fb), ka(fb), Reff))
			]
		)

def fit_pwave_lineshape(f, Ni, A, T, fb): 
	# fit using Maxwell-Boltzmann statistics
	return np.piecewise(
		f, [f >= fb],
		[lambda f: Ni,
		# when f < fb
			lambda f: Ni*(1-A*(fb-f)**(3/2)/fb**2 * np.exp(-(fb-f)/T))
			]
		)

### TESTING FLAGS ###
do_debug = False
# an initial debugging plot to make sure fit guesses are reasonable
test_fit = True
# display an initial guess plot
group_by_mean = True
# average the data when plotting

## FITTING FLAGS ###
do_filter = True
# remove data points in filter ranges
do_append = False
# append new fits to the old file
do_stats = True
# output results to summary csv files
do_plots = True
# output fit plots and histograms
trialsB = 1000
# total Monte Carlo fits to perform

datalist = ['swave']
timestr = time.strftime("%Y%m%d-%H%M%S")

# Load data #######################
for dataType in datalist:

	if dataType == 'pwave':
		runs = ['dimers2d_2022-08-12_F_LAT2_40ER_198G_FM',
				'dimers2d_2022-08-12_G_LAT2_40ER_198p3G_FM_y',
				'dimers2d_2022-08-16_E_LAT1_80ER_198p3G_FM_y',
				'dimers2d_2022-08-16_E_LAT1_80ER_198p3G_FM_z',
				'dimers2d_2022-08-17_N_LAT1_80ER_198p3G_FM_z',
				'dimers2d_2022-08-18_E_LAT1_80ER_198p4G_FM_z',
				'dimers2d_2022-08-18_F_LAT1_80ER_198p2G_FM_y'
# 				'dimers2d_2022-08-18_F_LAT1_80ER_198p2G_FM_z'
# 				,
# 				'dimers2d_2022-11-30_K_LAT1_80ER_198p5G_FM_y',
# 				'dimers2d_2022-11-30_K_LAT1_80ER_198p5G_FM_z'
				]
		
		runs = ['dimers2d_2022-08-18_F_LAT1_80ER_198p2G_FM_z']
		
		fitType = 'pwave_lineshape'
		
	elif dataType == 'swave':
		runs =  [#'dimers2d_2022-08-12_G_LAT2_40ER_198P3G_FM_z',
			     'dimers2d_2022-08-16_E_LAT1_80ER_198p3G_FM_x',
				 'dimers2d_2022-08-17_N_LAT1_80ER_198p3G_FM_x',
				 'dimers2d_2022-08-18_E_LAT1_80ER_198p4G_FM_x',
				 'dimers2d_2022-08-18_F_LAT1_80ER_198p2G_FM_x',
# 				 'dimers2d_2022-08-18_G_LAT1_80ER_198p6G_FM',
				 'dimers2d_2022-08-18_I_LAT1_80ER_199p1G_FM',
				 'dimers2d_2022-08-22_P_LAT2_40ER_198p4G_FM',
				 'dimers2d_2022-08-22_Q_LAT2_40ER_198p4G_FM_evenwaveonly',
				 'dimers2d_2022-08-23_G_LAT2_40ER_199p4G_FM',
				 'dimers2d_2022-08-24_J_LAT2_40ER_198G_FM',
				 'dimers2d_2022-08-24_K_LAT2_40ER_199p5G_FM',
				 'dimers2d_2023-03-12_B_LAT2_120ER_199p5G_FM',
				 'dimers2d_2023-03-12_C_LAT2_120ER_199p7G_FM',
				 'dimers2d_2023-03-13_G_LAT2_120ER_199p8G_FM',
				 'dimers2d_2023-03-14_H_LAT2_120ER_198p5G_FM97kHz',
				 'dimers2d_2023-03-14_H_LAT2_120ER_198p5G_FM92kHz',
				 'dimers2d_2023-05-03_B_LAT2_120ER_200G_FM_blackman_8ms5p5V'
# 				 'dimers2d_2023-05-01_O_LAT2_120ER_200p1G_FM_8msblackman'
				 ]
		
		runs = ['dimers2d_2023-03-12_C_LAT2_120ER_199p7G_FM',
			  'dimers2d_2023-05-01_H_LAT2_120ER_200G_FM_8msblackman',
			  'dimers2d_2022-08-23_G_LAT2_40ER_199p4G_FM']
# 		
		
		fitType = 'Gaussian'

	try:
		os.remove(dataType + timestr + '/' + fitType + '_' + dataType + '.csv')
	except:
		print('Nothing to delete')
		
	for run, run_no in zip(runs, range(len(runs))):
		print('**** Analyzing ' + run + " ****")

		myData = dimer_data.loadData(run)
		
		if myData.exclude:
			print('Excluded')
			#continue

		specData = myData.data
		# the atom loss spectrum data
		specData.sort()
# 		print(myData.B, myData.freq97)
	
		if group_by_mean == True:
			# group data by x value and take the mean
			Y = specData.copy()
			groups = Y[:,0].copy()
			Y = np.delete(Y, 0, axis=1)
		
			_ndx = np.argsort(groups)
			xx, _pos, g_count  = np.unique(groups[_ndx], 
											return_index=True, 
											return_counts=True)
			
			g_sum = np.add.reduceat(Y[_ndx], _pos, axis=0)
			yy = g_sum / g_count[:,None]
			yy = yy.flatten()
		
		else:
			# in the case of rf spectra, convert rf frequencies in MHz to detunings in kHz (CHECK THE SIGN)
			# xx = np.array([1000*(s[0]-myData.freq97) for s in specData])
			xx = np.array([s[0] for s in specData])
			yy = np.array([s[1] for s in specData])
			# the remaining atom number vector
			
		# filter data by removing ranges [[x1, x2], [x3, x4] , ...]
		if do_filter:
			excludeList = myData.excludeRanges
			uxx = xx
			uyy = yy
			
			for pair in excludeList:
				xxp = np.array([s[0] for s in zip(xx, yy) if (s[0] <= pair[0] or s[0] >= pair[1])])
				yyp = np.array([s[1] for s in zip(xx, yy) if (s[0] <= pair[0] or s[0] >= pair[1])])
				xx = xxp
				yy = yyp
		# sort the data by the frequency
		nData = len(xx)
		# count how many data points
			
		if test_fit:
			xx = 1000*(xx - np.ones(len(xx))*myData.freq97)
			if do_filter:
				uxx = 1000*(uxx - np.ones(len(uxx))*myData.freq97)
			
		if fitType =='Gaussian':
			plabels = ["Ni", "A", "w", "fB", 'chisq', 'b-g']
			func = fit_gauss
			pGuess = myData.pGuessG
		else:
			plabels = ["Ni", "A", "T", "fB", 'chisq', 'b-g']
			func = fit_pwave_lineshape
			pGuess = myData.pGuess	
		
		if do_debug:
			plt.figure(run_no, figsize=(7,5))
			plt.title(run)
			if do_filter:
				plt.plot(uxx,uyy, '+b')
			plt.plot(xx, yy, '+r')
			plt.xlabel("Freq (MHz)")
			plt.ylabel("Atom no.")
			num = 1000
			xFit = np.linspace(np.min(xx), np.max(xx), num)

			if test_fit:
				print("Guess:")
				print(pGuess)
				print(func(pGuess[3], *pGuess))
				plt.plot(xFit, (func(xFit, *pGuess)), '--',color=colorfunc(fitType))
				plt.axvline(pGuess[3], color=colorfunc(fitType), linestyle='--')
				# pylint: disable=unbalanced-tuple-unpacking
				pFit, cov = optimize.curve_fit(func, xx, yy, pGuess)
				
				plt.plot(xFit, (func(xFit, *pFit)), '-', color='black')
				plt.axvline(pFit[3], color='black', linestyle='--')
				print("Fit:")
				print(pFit)

			if do_profile:
				pr.disable()
				s = io.StringIO()
				sortby = SortKey.TIME
				ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
				ps.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
				print(s.getvalue())
			plt.show()
			continue
			# don't do the rest of the fitting


		folder = dataType + timestr + '/' + run + '/'

		try:
			os.makedirs(folder)
		except Exception:
			pass
		
		# Bootstrap fitting ####################################################
		
		print("** Bootstrap resampling")
					
		if os.path.isfile(folder + '/' + fitType + '_fits.txt'):
			pFitB = np.loadtxt(folder + '/' + fitType + '_fits.txt', delimiter=', ')
			print(np.shape(pFitB))
			old_trialsB = np.shape(pFitB)[0]
			# number of old trials loaded in
			print("   Loading from " + fitType + " file: %i old trials" % old_trialsB)
			if do_append:
				trialB = old_trialsB # starting position for new fits
				trialsB = trialsB-old_trialsB # update index
				if trialsB > 0:
					pFitB_old = pFitB.copy()
					pFitB = np.concatenate((pFitB,np.zeros([trialsB, 4])))
					new_trialsB = True
				else:
					new_trialsB = False
			else:
				new_trialsB = False
		elif os.path.isfile(folder + '/bs_params.npy'):
			pFitB = np.load(folder + '/bs_params.npy')
			old_trialsB = np.shape(pFitB)[0]
			print("   Loading from npy file: %i old trials" % old_trialsB)
			if do_append:
				trialB = old_trialsB # starting position for new fits
				trialsB = trialsB-old_trialsB # update index
				if trialsB > 0:
					pFitB_old = pFitB.copy()
					pFitB = np.concatenate((pFitB,np.zeros([trialsB, 6,12])))
					new_trialsB = True
				else:
					new_trialsB = False
			else:
				new_trialsB = False
				#print(pFitB)
# 			print(np.shape(pFitB))
# 			print(pFitB[0,0:4,0])
# 			print(pFitB[0,0:4,1])
# 			print(pFitB[0,0:4,2])
# 			print(pFitB[0,:4,3])
# 			print(pFitB[0,:4,4])
# 			print(pFitB[0,:4,5])
# 			print(pFitB[0,:4,6])
			np.savetxt(folder + '/MB_fits.txt', pFitB[:,0:4,0], delimiter= ', ')
		else:
			old_trialsB = 0

			pFitB = np.zeros([trialsB, 4])
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
					pFit, cov = optimize.curve_fit(func, xTrial, yTrial, pGuess)
				except Exception:
					print("Failed to converge")
					fails += 1
					continue
				
				if np.sum(np.isinf(trialB)) or abs(pFit[3] - pGuess[3]) > 200. or pFit[1] > 2e9 or pFit[1] <= 0.:
					print('Fit params out of bounds')
					print(pFit)
					continue
				else:
					pFitB[trialB, :] = pFit
					trialB += 1

		print(pFitB[:,0:4])
		np.savetxt(folder + '/' + fitType + '_fits.txt', pFitB[:,0:4], delimiter= ', ')
			
		if do_profile:
			pr.disable()
			s = io.StringIO()
			sortby = SortKey.TIME
			ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
			ps.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
			print(s.getvalue())

		conf = 68.2689  # confidence level for CI
		
		#print(xx)
		#print(yy)
		mincut = min(np.max(xx),np.min(xx))
		maxcut = max(np.max(xx),np.min(xx))
		
		print(mincut)
		print(maxcut)

		if do_stats:
			print("** Stats")
			fits = pFitB
			for fit in range(np.size(fits,axis=0)):
				if (fits[fit,3] < mincut) or (fits[fit,3] > maxcut):
					fits[fit] = np.nan
			
			median_f = np.nanmedian(fits[:, 3])
			upper_f = np.nanpercentile(fits[:, 3], 100-(100.0-conf)/2.)
			lower_f = np.nanpercentile(fits[:, 3], (100.0-conf)/2.)
			
			csvPath = dataType + timestr + '/' + fitType + '_' + dataType + '.csv'
			print(csvPath)
			if not os.path.isfile(csvPath):
				with open(csvPath, 'w', newline='') as csvFile:
					csvWriter = csv.writer(csvFile, delimiter=',')
					csvWriter.writerow(['Run		   ', 'B (G)	   ','Exclude	 ',
										'Offset	  ', 'Amplitude   ',
										'T (kHz)	 ', 'fB (kHz)	',
										'median_f	', 'lower_f	 ', 'upper_f	 ',
										'sigma_fM	', 'sigma_fP	',
										'N_fits	  '])
					csvWriter.writerow(['{:<14}'.format(run),
										'{:12.2f}'.format(myData.B),
										'{:12}'.format(myData.exclude),
										'{:12.2f}'.format(np.nanmedian(fits[:, 0])),
										'{:12.4f}'.format(np.nanmedian(fits[:, 1])),
										'{:12.2f}'.format(np.nanmedian(fits[:, 2])),
										'{:12.2f}'.format(np.nanmedian(fits[:, 3])),
										'{:12.2f}'.format(median_f),
										'{:12.2f}'.format(lower_f),
										'{:12.2f}'.format(upper_f),
										'{:12.2f}'.format(median_f - lower_f),
										'{:12.2f}'.format(upper_f - median_f),
										'{:12}'.format(np.size(fits,axis=0))
										])
			else:
				with open(csvPath, 'a', newline='') as csvFile:
					csvWriter = csv.writer(csvFile, delimiter=',')
					csvWriter.writerow(['{:<14}'.format(run),
										'{:12.2f}'.format(myData.B),
										'{:12}'.format(myData.exclude),
										'{:12.2f}'.format(np.nanmedian(fits[:, 0])),
										'{:12.4f}'.format(np.nanmedian(fits[:, 1])),
										'{:12.2f}'.format(np.nanmedian(fits[:, 2])),
										'{:12.2f}'.format(np.nanmedian(fits[:, 3])),
										'{:12.2f}'.format(median_f),
										'{:12.2f}'.format(lower_f),
										'{:12.2f}'.format(upper_f),
										'{:12.2f}'.format(median_f - lower_f),
										'{:12.2f}'.format(upper_f - median_f),
										'{:12}'.format(np.size(fits,axis=0))
										])

# Plotting #####################################################################

		if do_plots:
			print('** Plots')
			# display vector for plotting
			numpts = 100
			xFit = np.linspace(np.min(xx), np.max(xx), numpts)
				
			fig_size = plt.rcParams["figure.figsize"]
			fig_size[0] = 10
			fig_size[1] = 8
			plt.rcParams["figure.figsize"] = fig_size

			fig = plt.figure()
			#for i in range(0, np.shape(pFitB)[0]-1):

			#print(pFitB)
			fits = pFitB
			#print(k)
			#print(numDA)
			#print(np.shape(fits))
			#print(fits)
			for fit in range(np.size(fits,axis=0)):
				if (fits[fit,3] < mincut) or (fits[fit,3] > maxcut):#(fits[fit,3] < 10) or (fits[fit,3] > 1000):
					fits[fit] = np.nan

			pMed = np.array([0.,0.,0.,0.])
# 			pMed[0] = np.nanmedian(fits[:, 0])
# 			pMed[1] = np.nanmedian(fits[:, 1])
# 			pMed[2] = np.nanmedian(fits[:, 2])
# 			pMed[3] = np.nanmedian(fits[:, 3])
# 			print(pMed)

			plt.plot(xFit, func(xFit, *pMed), color=colorfunc(fitType), alpha=.2)
			
			plt.axvline(x = np.nanmedian(fits[:, 3]), color=colorfunc(fitType), alpha=0.5)
			plt.axvline(x = np.nanpercentile(fits[:, 3], (100.-conf)/2.), color=colorfunc(fitType), alpha=0.5)
			plt.axvline(x = np.nanpercentile(fits[:, 3], 100.-(100.-conf)/2.), color=colorfunc(fitType), alpha=0.5)
		
			do_plotCI = True
			if do_plotCI:
				print('*** Plotting Confidence Band')
				iis = range(0,np.shape(fits)[0])
				ylo1 = 0*xFit
				yhi1 = 0*xFit
				for idx, xval in enumerate(xFit):
					print('	%i of %i @ %s' % (idx, numpts,time.strftime("%H:%M:%S", time.localtime())))
					#print(xval)
					#print([fits[ii,0],fits[ii,1],fits[ii,2],-1*qb_factor*fits[ii,3]])
					#FDW_qb, FCQB_FD_int
					fvals = [func(xval,*fits[ii,:4]) for ii in iis]
					#fvals = [func(xval,fits[ii,0],fits[ii,1],fits[ii,2],fits[ii,3]) for ii in iis] # change for bound (ii,:-2)/ QB (ii,:)
					#print(fvals)
					ylo1[idx] = np.nanpercentile(fvals,(100.-conf)/2.)
					yhi1[idx] = np.nanpercentile(fvals,100.-(100.-conf)/2.)
	
					#print(np.percentile(func(xval,*pFitB[:,:-2]),(100.-conf)/2.))
					
				#print(np.shape(np.percentile([func(xFit,*pFitB[ii,:-2]) for ii in range(0,np.shape(pFitB)[0])])))

				plt.plot(xFit, ylo1, color=colorfunc(fitType))
				plt.plot(xFit, yhi1, color=colorfunc(fitType))
			
			plt.plot(xx, yy, '.')
			#plt.plot(xFit, func(xFit, *pGuess), 'k--')
			plt.xlim([np.nanmin(xFit),np.nanmax(xFit)])
			
			#plt.ylim([0,np.nanmax(yy)*1.1])
			plt.xlabel(" freq (kHz)")
			plt.ylabel("Atoms remaining")
			plt.suptitle('[' + dataType + '/' + run + "] " + fitType + " Bootstrap resampling")
			#plt.show()
			print(folder + fitType + '_Fit.pdf')
			plt.savefig(folder + fitType + '_Fit.pdf')
			plt.close()

			fig2 = plt.figure()
			
			fits = pFitB[:,:]
			#print(fits)

			for fit in range(np.size(fits,axis=0)):
				if (fits[fit,3] < mincut) or (fits[fit,3] > maxcut):#(fits[fit,3] < 10) or (fits[fit,3] > 1000):
					fits[fit] = np.nan

			for i in range(0, 4): # change for B nvar+1/QB nvar+0
				ax1 = plt.subplot(2, 2, i+1)

				median_f = np.nanmedian(fits[:, i])
				#print(median_f)
				upper_f = np.nanpercentile(fits[:, i], 100-(100.0-conf)/2.)
				lower_f = np.nanpercentile(fits[:, i], (100.0-conf)/2.)

				array = fits[:, i]
				finite = array [~np.isnan(array)]
# 				print(finite)
				ax1.hist(finite, bins=50, alpha=0.5, color=colorfunc(fitType),
						range=(np.nanpercentile(finite,1.0),np.nanpercentile(finite,99.0)))
				ax1.axvline(x=median_f, c=colorfunc(fitType),
							label='BS: %.2f-%.2f+%.2f' % (median_f, median_f - lower_f, upper_f - median_f))
				ax1.axvline(x=upper_f, c=colorfunc(fitType),alpha=0.5)
				ax1.axvline(x=lower_f, c=colorfunc(fitType),alpha=0.5)
				#ax1.axvline(x=median+upperB, c='b',alpha=0.5)
				#ax1.axvline(x=median-lowerB, c='b',alpha=0.5)
				#ax1.axvline(x=median+upper95, c='r',alpha=0.5)
				#ax1.axvline(x=median-lower95, c='r',alpha=0.5)

				ax1.legend(
					bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
					mode="expand", borderaxespad=0.)

				ax1.set_xlabel(plabels[i])

			fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
			fig2.suptitle('[' + dataType + '/' + run + '] MB Fit parameter distributions (%.2f %% confidence)' % (conf))

			fig2.savefig(folder + fitType + '_Hists.pdf')
			plt.close()

			fig3 = plt.figure()
			for i in range(0, 4):
				for j in range(i+1, 4):
					ax3 = plt.subplot(4-1, 4-1, (4-1)*i+j)
					ax3.scatter(pFitB[:, j], pFitB[:, i], color=colorfunc(fitType), s=10,
								edgecolor='none', alpha=0.12)

					ax3.set_xlabel(plabels[j])
					ax3.set_ylabel(plabels[i])
			fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
			fig3.suptitle('[' + dataType + '/' + run + "] Fit parameter correlations")

			fig3.savefig(folder + fitType + '_Correlations.png')
			plt.close()