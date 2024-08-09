# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Calculations below assume errors are not correlated. I'm not so sure how to get
around that. Perhaps we need to do some like Monte Carlo or bootstrapping thing?
Or maybe we need to stop computing quantities that depend on so many related things.


"""
import os
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)

from data_class import Data
from library import deBroglie, deBroglie_kHz, a97, chi_sq, mK, hbar, h, kB, pi, \
	plt_settings, colors, markers, tintshade, tint_shade_color
from scipy.optimize import curve_fit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle

from fit_functions import Linear

# field wiggle calibration fit
from contact_correlations.field_wiggle_calibration import Bamp_from_Vpp 

### paths
data_path = os.path.join(proj_path, 'data')
data_path = os.path.join(proj_path, 'data/reanalyzed')
figfolder_path = os.path.join(proj_path, 'figures')

### flags
Analysis = True
Debug = True

### metadata
metadata_filename = 'heating_metadata.xlsx'
metadata_file = os.path.join(proj_path, metadata_filename)
metadata = pd.read_excel(metadata_file)	
files =  metadata.loc[metadata['exclude'] == 0]['filename'].values

### save file path
savefilename = 'heating_rates_results.xlsx'
savefile = os.path.join(proj_path, savefilename)

### plot settings
plt.rcParams.update(plt_settings) # from library.py
color = '#1f77b4' # default matplotlib color (that blueish color)
light_color = tint_shade_color(color, amount=1+tintshade)
dark_color = tint_shade_color(color, amount=1-tintshade)
plt.rcParams.update({"figure.figsize": [12,8],
					 "font.size": 14,
					 "lines.markeredgecolor": dark_color,
					 "lines.markerfacecolor": light_color,
					 "lines.color": dark_color,
					 "lines.markeredgewidth": 2,
					 "errorbar.capsize": 0})

##################
#### Analysis ####
##################

def calc_A(B0, T, Bamp):#, e_T, e_Bamp
	''' Returns A ''' #and e_A 
	A = deBroglie_kHz(T)/a97(B0-Bamp)
# 	e_a97 = np.abs(a97(B0-Bamp) - a97(B0-Bamp+e_Bamp)) # assumes symmetric
# 	e_dB = np.abs(deBroglie(T) - deBroglie(T+e_T)) # same
# 	e_A = A*np.sqrt((e_a97/a97(B0-Bamp))**2 + (e_dB/deBroglie(T))**2)
	return A
 # , e_A

def compute_slope_midpoint(df, x_name, y_name):
	""" Computes the rise over the run and the midpoint from a dataframe
		given x and y column names. Assumes there are only two points. 
		If there are more than two points, use fit_slope_midpoint. """
	if len(df[x_name].unique()) > 2:
		raise ValueError("Too many values to compute slope.")
		
	if len(df[x_name].unique()) < 2:
		print(df)
		print(df[y_name])
		print(df[x_name])
		raise ValueError("""Too few values to compute slope. Something strange 
				   with the data.""")
	
	deltay = df[y_name][1] - df[y_name][0]
	deltax = df[x_name][1] - df[x_name][0]
	
	midpoint = (df[y_name][1] + df[y_name][0])/2
	e_midpoint = np.sqrt(df[y_name][1]**2 + df[y_name][0]**2)
	
	slope = deltay/deltax
	e_slope = slope * e_midpoint/deltay
		
	return slope, e_slope, midpoint, e_midpoint

def linear(x, a, b):
	return a*x + b

def fit_slope_midpoint(df, x_name, y_name):
	""" Fits slope and computes midpoint given x and y column names and
		dataframe. """
	if len(df[x_name].unique()) <= 2:
		raise ValueError("Too few values to fit slope.")
		
	popt, pcov = curve_fit(linear, df[x_name], df[y_name], sigma=df['em_'+y_name])
	perr = np.sqrt(np.diag(pcov))
	
	midpoint = linear((min(df[x_name])+max(df[x_name]))/2, *popt)
	# this is wrong, assumes no covariance
	e_midpoint = perr[0]*(min(df[x_name])+max(df[x_name]))/2 * perr[1]
		
	return popt[0], perr[0], midpoint, e_midpoint
	
slope_variables = ['TkHz', 'EFkHz', 'ToTF', 'N', 'EkHz', 'EkHz2', 'SNkB', 
				   'smomEkHz', 'smomEoEFEst', 'smomEoEF']

time_name = 'time'

### loop analysis over selected datasets
for filename in files:
	
	if Analysis == False:
		break # exit loop if no analysis required, this just elims one 
			  # indentation for an if statement
			  
    # run params from HFT_data.py
	print("----------------")
	print("Analyzing " + filename)
	
	metadf = metadata.loc[metadata.filename == filename].reset_index()
	if metadf.empty:
		print("Dataframe is empty! The metadata likely needs updating.")
		continue
	
	# create data structure
	filename = filename + ".dat"
	run = Data(filename, path=data_path)
	df = run.data
	
	# convert T to kHz
	df['TkHz'] = df['T']*kB/h/1e3
	
	# compute Bamp
	df['Bamp'] = df.apply(lambda x: Bamp_from_Vpp(x['amplitude'], x['freq'])[0], axis=1)
	


	# groupby time, averaging and computing sem
	df = pd.concat([df.groupby([time_name, 'freq']).mean().reset_index(),
			 df.groupby([time_name, 'freq']).sem().reset_index().add_prefix('em_')], axis=1)
	
	# loop over freq in multiscan
	for freq in df['freq'].unique():
		# subdf for the loop frequency
		subdf = df.loc[df['freq'] == freq].reset_index()
		
# 		Debug = True
# 		if Debug == True:
# 			break
		
		# initialize arrays to hold computed slopes
		slopes = []
		e_slopes = []
		midpoints = []
		e_midpoints = []
		
		# compute slopes and midpoints for requested values
		for name in slope_variables:
			if len(subdf[time_name].unique()) <= 2: # rise over run
				slope, e_slope, midpoint, e_midpoint = compute_slope_midpoint(subdf, 
														 time_name, name)
			else: # fit to linear function
				slope, e_slope, midpoint, e_midpoint = fit_slope_midpoint(subdf, 
															   time_name, name)
			# slopes are in kHz cause time is in ms
			slopes.append(slope)
			e_slopes.append(e_slope)
			midpoints.append(midpoint)
			e_midpoints.append(e_midpoint)
		
		# fill results dict with slopes and midpoints
		results = dict(zip([name+'_rate' for name in slope_variables], 
						  slopes))
		results.update(dict(zip(['e_'+name+'_rate' for name in slope_variables], 
						  e_slopes)))
		results.update(dict(zip(slope_variables, midpoints)))
		results.update(dict(zip(['e_'+name for name in slope_variables], 
						  e_midpoints)))
		
		# add from df to results dict
		results['time'] = max(subdf.time) - min(subdf.time)
		results['Bamp'] = subdf.Bamp
		results['freq'] = subdf.freq[0]
		
		# add from metadata to results dict
		results.update(metadf.to_dict(orient='records')[0])
		
		# compute A and lambda_db
		results['A'] = calc_A(metadf.B, results['TkHz'], subdf.Bamp[0])
		results['lambda'] = deBroglie_kHz(results['TkHz'])
		results['kF'] = np.sqrt(2*mK*h*1e3*results['EFkHz'])/hbar
		results['e_kF'] = results['kF']*(results['e_EFkHz']/results['EFkHz'])/2
		
		# compute heating rate, zeta and C
		results['heating'] = results['EkHz_rate']/results['EFkHz']**2/results['A']**2
		results['e_heating'] = results['heating'] * np.sqrt( 
			# 2x here is because it's EF**2 
			(2*results['e_EFkHz']/results['EFkHz'])**2 + \
				(results['e_EkHz_rate']/results['EkHz_rate'])**2)
		
		results['zeta'] = (results['kF']*results['lambda'])**2/(9*pi*results['freq']**2) * \
			results['EkHz_rate']/results['A']**2
		results['e_zeta'] = results['zeta'] * np.sqrt(
			# kF**2 is like EF, so
			(results['e_EFkHz']/results['EFkHz'])**2 + \
			(results['e_TkHz']/results['TkHz'])**2 + \
				(results['e_EkHz_rate']/results['EkHz_rate'])**2)
			
		pi_factors = 2/9*36*pi*(2*pi)**(3/2)
		fandT_factors = (results['freq']/results['TkHz'])**(3/2)/ \
			(results['lambda']*results['kF'])/(2*pi*results['freq']**2)
		results['C'] = pi_factors*fandT_factors*results['EkHz_rate']/results['A']**2
		# C propto Edot/kF, so.., but there is also a factor of temperature. Hmm
		results['e_C'] = results['C']*np.sqrt((results['e_kF']/results['kF'])**2 \
							+(results['e_EkHz_rate']/results['EkHz_rate'])**2 \
								+(results['e_TkHz']/results['TkHz'])**2)
		
			
