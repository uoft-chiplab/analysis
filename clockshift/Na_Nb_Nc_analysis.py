# -*- coding: utf-8 -*-
"""
@author: Chip Lab

This script plots results from 100kHz HFT runs 
where we alternated between imaging b+c and a+c.

"""

from data_class import Data 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from library import styles, GammaTilde, h

from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
from contact_correlations.contact_interpolation import contact_interpolation as C_interp

plotting = True
# KX TESTING
correct_ac_loss = False
datasets = [
# 			'2025-02-26_C_e.dat',
			'2025-02-28_C_e_ODT2=1.dat',
			'2025-02-28_C_e_ODT2=2.dat',
			'2025-02-28_C_e_ODT2=3.dat',
			'2025-02-28_C_e_ODT2=4.dat',
			'2025-02-28_C_e_ODT2=6.dat',
			]

reduced_imaging_light = [
						'2025-02-28_F_e.dat', # reduced imaging light overall
						]

equalized_light = [
				'2025-02-28_H_e.dat', # tried to equalize light on img2 and 3
				]

blanks_rescaled = [
				'2025-03-04_F.dat',	  # rescale with blanks
				]


low_OD = [
		'2025-03-04_H_e.dat', # hot and low atom num
		'2025-03-04_K_e.dat', # coldish and low atom num
		]

drop_data = [
			'2025-03-04_O_e_drop time=0.8.dat',
			'2025-03-04_O_e_drop time=1.2.dat',
			]

labels = [' ',
		  'less light',
		  'equal light',
		  'blanks',
		  'low OD',
		  'hot data',
		  ]

included_data = [
				datasets, 
				reduced_imaging_light, 
				equalized_light,
				blanks_rescaled,
				low_OD,
				drop_data,
				]


filenames = []

for data_list in included_data:
	filenames = filenames + data_list

ToTFs = [
# 		0.358,
		0.355, 
		0.412, 
		0.446, 
		0.533,
		0.63,
		0.355,
		0.355,
		0.32,
		1.20,
		0.445,
		0.822,
		1.05,
		]

EFs = [
# 	   17.7,
	   13.67,
	   14.63,
	   14.95,
	   15.38,
	   15.61,
	   13.67,
	   13.67,
	   15.6,
	   14.3,
	   11.0,
	   15.7,
	   15.6,
	   ]

pulse_times = [
# 			   200,
			   200,
			   200,
			   200,
			   200,
			   200,
			   200,
			   200,
			   200,
			   400,
			   200,
			   200,
			   200,
			   ]

ff = 0.88

detuning = 100

# phaseO calibraiton
VpptoOmegaR47 = 12.01/0.452 # kHz/Vpp - 2025-02-12 calibration 
phaseO_OmegaR = lambda VVA, freq: 2*np.pi*VpptoOmegaR47 * Vpp_from_VVAfreq(VVA, freq)

# omega^2 [kHz^2] 1e saturation average fit value from various ToTF
HFT_x0_hot = 897.441
HFT_x0_cold = 737
HFT_e_x0 = 35

def saturation_scale(x, x0):
	""" x is OmegaR^2 and x0 is fit 1/e Omega_R^2 """
	return x/x0*1/(1-np.exp(-x/x0))

dfs = []

i = 0
for filename, ToTF, EF, pulse_time in zip(filenames, ToTFs, EFs, pulse_times):
	run = Data(filename)
	
	# fudge the c9 number
	run.data['c9'] = run.data['c9'] * ff
	
	# calculate OmegaR
	OmegaR = phaseO_OmegaR(max(run.data.VVA.values), 47.3227) * np.sqrt(0.31)
	
	# saturation correction
	if ToTF > 0.35:
		sat_scale = saturation_scale(OmegaR**2/(2*np.pi)**2, HFT_x0_hot)
		corr_c = 0.66 if correct_ac_loss else 1
		corr_a = 0.98 if correct_ac_loss else 1
	else:
		sat_scale = saturation_scale(OmegaR**2/(2*np.pi)**2, HFT_x0_cold)
		corr_c = 0.76 if correct_ac_loss else 1
		corr_a = 0.99 if correct_ac_loss else 1
		
# 	sat_scale = 1
	print("Saturation scale is {:.2f}".format(sat_scale))
	
	# acquire bg dataframe
	bgs = run.data.iloc[(run.data['VVA'] == 0).values]
	
	bcbgs = bgs.iloc[(bgs['sweep'] == 46.05).values]
	acbgs = bgs.iloc[(bgs['sweep'] == 45.90).values]
	
	# counts
	acbgc9 = acbgs['c9'].reset_index(drop=True)/corr_a
	acbgc5 = acbgs['c5'].reset_index(drop=True)/corr_c
	
	bcbgc9 = bcbgs['c9'].reset_index(drop=True)/corr_a
	bcbgc5 = bcbgs['c5'].reset_index(drop=True)/corr_c
	
	# count weighted counts
# 	acbgc9cwc = acbgs['c9cwc'].reset_index(drop=True)
# 	acbgc5cwc = acbgs['c5cwc'].reset_index(drop=True)
# 	
# 	bcbgc9cwc = bcbgs['c9cwc'].reset_index(drop=True)
# 	bcbgc5cwc = bcbgs['c5cwc'].reset_index(drop=True)
# 	
# 	# count weighted counts
# 	acbgc9cwc = acbgs['c9cwc'].reset_index(drop=True)
# 	acbgc5cwc = acbgs['c5cwc'].reset_index(drop=True)
# 	
# 	bcbgc9cwc = bcbgs['c9cwc'].reset_index(drop=True)
# 	bcbgc5cwc = bcbgs['c5cwc'].reset_index(drop=True)
		
	# acquire signal dataframe
	datapts = run.data.iloc[((run.data['VVA'] > 0)).values]
	
	bc = datapts.iloc[(datapts['sweep'] == 46.05).values]	
	ac = datapts.iloc[(datapts['sweep'] == 45.90).values]	
	
	# counts
	acc9 = ac['c9'].reset_index(drop=True)/corr_a
	acc5 = ac['c5'].reset_index(drop=True)/corr_c
	
	bcc9 = bc['c9'].reset_index(drop=True)/corr_a
	bcc5 = bc['c5'].reset_index(drop=True)/corr_c
	
	# count weighted counts
# 	acc9cwc = ac['c9cwc'].reset_index(drop=True)
# 	acc5cwc = ac['c5cwc'].reset_index(drop=True)
# 	
# 	bcc9cwc = bc['c9cwc'].reset_index(drop=True)
# 	bcc5cwc = bc['c5cwc'].reset_index(drop=True)
	
	# now average and std dev lists
	data_lists = [acbgc9, acbgc5, bcbgc9, bcbgc5,
				  acc9, acc5, bcc9, bcc5,
# 				  acc9cwc, acc5cwc, bcc9cwc, bcc5cwc,
# 				  acbgc9cwc, acbgc5cwc, bcbgc9cwc, bcbgc5cwc,
				  ]
	
	# names of quantities
	keys = ['Na_bg', 'Nc_bg', 'Nb_bg', 'Nc2_bg',
			'Na', 'Nc', 'Nb', 'Nc2',
# 			'Na_bg_cwc', 'Nc_bg_cwc', 'Nb_bg_cwc', 'Nc2_bg_cwc',
# 			'Na_cwc', 'Nc_cwc', 'Nb_cwc', 'Nc2_cwc',
			]
	
	values = []
	
	for q in data_lists:
		values.append(q.mean())
		values.append(q.sem())
		
	# names of std dev of quantities
	e_keys = ['e_'+key for key in keys]
	
	# keys for final df, need to interleave e_keys
	df_keys = []
	for key_pair in zip(keys, e_keys):
		df_keys.extend(key_pair)
	
	# make dataframe of averaged values
	df = pd.DataFrame([dict(zip(df_keys, values))])
	
	df['Nb+Nc'] = df['Nb'] + df['Nc2']
	df['e_Nb+Nc'] = np.sqrt(df['e_Nb']**2+df['e_Nc2']**2)
	
	df['transfer'] = sat_scale * (df['Nc2']- df['Nc2_bg'])/(df['Nb'] + df['Nc2'] - df['Nc2_bg'])
	
	# This is an estimate
	df['e_transfer'] = df['transfer'] * np.sqrt(df['e_Nc2']**2+df['e_Nc_bg']**2)/ \
								(df['Nc2'] - df['e_Nc2'])
	
	df['loss'] = sat_scale * (df['Nb_bg'] - df['Nb'])/df['Nb_bg']
	df['e_loss'] = df['Nb']/df['Nb_bg']*np.sqrt((df['e_Nb_bg']/df['Nb_bg'])**2+\
						   (df['e_Nb']/df['Nb'])**2)
		
	df['scaledtransfer'] = GammaTilde(df['transfer'], h*EF*1e3, OmegaR*1e3, 
							  pulse_time/1e6)
	
	df['e_scaledtransfer'] = df['e_transfer']/df['transfer'] * df['scaledtransfer']
	
	df['scaledloss'] = GammaTilde(df['loss'], h*EF*1e3, OmegaR*1e3, 
							  pulse_time/1e6)
	df['e_scaledloss'] = df['e_loss']/df['loss'] * df['scaledloss']
	
	df['C'] = 2*np.sqrt(2)*np.pi**2*df['scaledtransfer']*(detuning/EF)**(3/2)
	df['e_C'] = df['e_transfer']/df['transfer'] * df['C']
	
	df['Closs'] = 2*np.sqrt(2)*np.pi**2*df['scaledloss']*(detuning/EF)**(3/2)
	df['e_Closs'] = df['e_loss']/df['loss'] * df['Closs']
		
	df['ratio'] = df['transfer']/df['loss']
	df['e_ratio'] = df['ratio']*np.sqrt((df['e_transfer']/df['transfer'])**2+\
						   (df['e_loss']/df['loss'])**2)
		
	df['Na_diff'] = df['Na_bg'] - df['Na']
	df['e_Na_diff'] = np.sqrt(df['e_Na_bg']**2 + df['e_Na']**2)
	
	df['Nb+Nc_diff'] = (df['Nb_bg'] + df['Nc2_bg']) - (df['Nb'] + df['Nc2'])
	df['e_Nb+Nc_diff'] = np.sqrt(df['e_Nb']**2+df['e_Nc2']**2+df['e_Nc2_bg']**2+df['e_Nb_bg']**2)
	
	df['anomalous_loss'] = sat_scale * (df['Nb+Nc_diff'])/df['Nb_bg']
	df['e_anomalous_loss'] = np.abs(df['anomalous_loss'])*\
							np.sqrt((df['e_Nb+Nc_diff']/df['Nb+Nc_diff'])**2 \
									   + (df['e_Nb_bg']/df['Nb_bg'])**2)
	
	keys.extend(['Nb+Nc'])
	# print results
	for key in keys:
		print(key + " = {:.0f}+-{:.0f}".format(df[key].values[0], df['e_'+key].values[0]))
	
	key = 'transfer'
	print(key + " = {:.2f}+-{:.2f}".format(df[key].values[0], df['e_'+key].values[0]))
	key = 'loss'
	print(key + " = {:.2f}+-{:.2f}".format(df[key].values[0], df['e_'+key].values[0]))
	key = 'ratio'
	print(key + " = {:.2f}+-{:.2f}".format(df[key].values[0], df['e_'+key].values[0]))
	
	df['ToTF'] = ToTF
	df['EF'] = EF
	df['filename'] = filename
	df['number'] = i
	
	dfs.append(df)
	i += 1
	
df = pd.concat(dfs)

# plot all data
if plotting == True:
	fig, axes = plt.subplots(3,3, figsize=(12,10.5))
	axs = axes.flatten()
	
	ylabels = [
				'transfer/loss',
				'Nb+Nc bg - signal',
				'Na bg - signal',
				'Contact',
				'Contact from Loss',
				'Na',
				'Nc',
				'Anomalous loss',
				'Anomalous loss',
				]
	
	xname = 'ToTF'
	
	# contact theory
	xs = np.linspace(min(df[xname]), max(df[xname]), 100)
	axs[3].plot(xs, C_interp(xs), ':', color='k', label='Tilman Theory')
	axs[4].plot(xs, C_interp(xs), ':', color='k', label='Tilman Theory')
	
	# set axes labels
	for i in range(len(ylabels)):
		axs[i].set(xlabel=xname, ylabel=ylabels[i])
		
	# correct the x labels
	axs[7].set(xlabel='transfer')
	axs[8].set(xlabel='Nb - Nb_bg')
	
	# loop over sets of data
	for j, file_list in enumerate(included_data):
		subdf = df.loc[df.filename.isin(file_list)]
		label = labels[j]
		sty = styles[j]
		
	
		# transfer vs. loss
		ax = axs[0]
		ax.errorbar(subdf[xname], subdf['ratio'], yerr=subdf['e_ratio'], 
				 **sty, label=label)
		
		# Nb + Nc bg - signal
		ax = axs[1]
		ax.errorbar(subdf[xname], subdf['Nb+Nc_diff'], yerr=subdf['e_Nb+Nc_diff'], 
				 **sty, label=label)
		
		# Na  bg - signal
		ax = axs[2]
		ax.errorbar(subdf[xname], subdf['Na_diff'], yerr=subdf['e_Na_diff'], 
				 **sty, label=label)
		
		
		# Contact
		ax = axs[3]
		ax.errorbar(subdf[xname], subdf['C'], yerr=subdf['e_C'], 
				 **sty, label=label)
		
		# Contact from loss
		ax = axs[4]
		ax.errorbar(subdf[xname], subdf['Closs'], yerr=subdf['e_Closs'], 
				 **sty, label=label)
		
		# count weighted counts
		ax = axs[5]
# 		ax.errorbar(subdf[xname], subdf['Na_cwc']/subdf['Na'], 
# 			  yerr=subdf['e_Na_cwc']/subdf['Na'], **sty, label=label)
		ax.plot(subdf[xname], subdf['Na'], **sty, label=label)
		
		# count weighted counts
		ax = axs[6]
# 		ax.errorbar(subdf[xname], subdf['Nb_cwc']/subdf['Nb'], 
# 			  yerr=subdf['e_Nb_cwc']//subdf['Nb'], **sty, label=label)
		ax.errorbar(subdf[xname], subdf['Nc'], **sty, label=label)
		
		# transfer vs anomalous loss
		ax = axs[7]
		ax.errorbar(subdf['transfer'], subdf['anomalous_loss'], yerr=subdf['e_anomalous_loss'], 
				 **sty, label=label)
		
		# Nb - Nb_bg vs anomalous loss
		ax = axs[8]
		ax.errorbar(subdf['loss'], subdf['anomalous_loss'], yerr=subdf['e_anomalous_loss'], 
				 **sty, label=label)
		
	
	for i in range(len(ylabels)):
		axs[i].legend()
		
# 	xname = 'EF'
# 	
# 	# set axes labels
# 	for i in [3,4,5]:
# 		axs[i].set(xlabel=xname, ylabel=ylabels[i-3])
# 	
# 	# loop over sets of data
# 	for j, file_list in enumerate(included_data):
# 		subdf = df.loc[df.filename.isin(file_list)]
# 		label = labels[j]
# 		sty = styles[j]
# 	
# 		# transfer vs. loss
# 		ax = axs[3]
# 		ax.errorbar(subdf[xname], subdf['ratio'], yerr=subdf['e_ratio'], 
# 				 **sty, label=label)
# 		
# 		# Nb + Nc bg - signal
# 		ax = axs[4]
# 		ax.errorbar(subdf[xname], subdf['Nb+Nc_diff'], yerr=subdf['e_Nb+Nc_diff'], 
# 				 **sty, label=label)
# 		
# 		# Na  bg - signal
# 		ax = axs[5]
# 		ax.errorbar(subdf[xname], subdf['Na_diff'], yerr=subdf['e_Na_diff'], 
# 				 **sty, label=label)
# 	
# 	for i in [3,4,5,]:
# 		axs[i].legend()
	
	fig.tight_layout()
	