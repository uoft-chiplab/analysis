# -*- coding: utf-8 -*-
"""
2023-10-19
@author: Chip Lab

Data class for loading, fitting, and plotting .dat
Matlab output files

Relies on fit_functions.py
"""
# %%
import os 
from glob import glob
import sys
from library import *
from fit_functions import *
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from matplotlib.ticker import MaxNLocator
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq

#to use fit example:
	#Data("filename").fit(fit_func=One you want, names=['x','y'])
	
#to use multiplot ex:
	#Data(“filename”).multiplot(fit_func, names=[‘x’,’y’],avg=’x’)
#to use avg data 
	#Data("filename",average_by='x')
#to exclude by a certain x value 	
	# Data("filename",exclude_range=#,exclude_range_x='x')
	
#Holy moly thanks for writing this ^

# file = "2024-09-05_X_e.dat"
# guess = [8, 7, 2*3.14*0.4, 0, 95]
drive = '\\\\UNOBTAINIUM\\E_Carmen_Santiago' 
plt.rcParams.update(plt_settings)

class Data:
	def __init__(self, filename, path=None, analysis=None, column_names=None, 
			  exclude_list=None, average_by=None, metadata=None,
			  exclude_range=None, exclude_range_x=None):
		self.filename = filename
		if metadata is not None:
			self.__dict__.update(metadata)  # Store all the extra variables
		
		if path:
			self.file = os.path.join(path, filename) # making manual path for the filename
		else:
			print(drive + '\\Data\\' + filename[:4] + '\\*\\*\\*\\' + filename)
			self.file = glob(drive + '\\Data\\' + filename[:4] + '\\*\\*\\*\\' + filename)[0] # EXTREMELY greedy ; for Fermium
			
		self.data = pd.read_table(self.file, delimiter=',') # making dataframe of chosen data
		
		if column_names:
			self.data = self.data[column_names]
		if exclude_list is not None:
			self.exclude(exclude_list)
		if exclude_range is not None:
			self.excluderange(filename, exclude_range, exclude_range_x)
		if average_by:
			self.group_by_mean(average_by)

	def analysis(self, EF=12, VVA=0, trf=10, nobg=False, track_bg=False):
		'''
		Use Data('filename').analysis().data to run this. This fcn is designed to output a dataframe
		that has columns that are needed for any analysis (e.g. transfer, c5bg, etc)
		'''
		###putting EF, trf into the data frame
		if EF == 12:
			print(f'❗❗Default value of EF = 12, trf = 10 used in analysis❗❗')
		self.data['EF'] = EF
		self.data['trf'] = trf

		###grabbing OmegaR 
		RabiperVpp47 = 13.05 / 0.500 # kHz/Vpp on scope 2025-10-21
		e_RabiperVpp47 = 0.22
		phaseO_OmegaR = lambda VVA, freq: 2*np.pi*RabiperVpp47 * Vpp_from_VVAfreq(VVA, freq)
		self.data['OmegaR'] = phaseO_OmegaR(VVA, self.data['freq'])

		###find the background by either tracking the bg pts across the scan and fitting it or 
		###using a VVA value default to 0
		if track_bg:
			# fit to line, calc it, append to df
			a=1
		else:
			self.data["c5bg"] = self.data[self.data['VVA'] == VVA]['c5'].mean()
			self.data['c9bg'] =  self.data[self.data['VVA'] == VVA]['c9'].mean()
			self.data["c5bg_sem"] = 1
			self.data["c5bg_std"] = 1
			self.data["c9bg_sem"] = 1
			self.data["c9bg_std"] = 1
		###if using a VVA value to find the bg then removing those points from the rest of the dataset
			self.data = self.data[self.data['VVA'] != VVA]
		###option to have no background at all 
		if nobg:
			self.data['c5bg'] = 0
			self.data['c9bg'] = 0
		###finding the transfer 
		self.data['dimertransfer'] = (1 - self.data['c5']/self.data['c5bg'])/2
		self.data['HFTtransfer'] = (self.data['c5'] - self.data['c5bg'])/(self.data['c5'] - self.data['c5bg'] + self.data['c9'])
		self.data['losstranfser'] = np.ones(len(self.data['c9']))-self.data['c9']/self.data['c9bg']
		###finding the scaled tranfser 
		self.data['scaledtransfer'] = self.data['EF']/(hbar * np.pi * self.data['OmegaR']**2 * self.data['trf']) * self.data['HFTtransfer']
		self.data['scaledtransfer_dimer'] = self.data['EF']/(hbar * np.pi * self.data['OmegaR']**2 * self.data['trf']) * self.data['dimertransfer']

		return self
		
	# exclude list of points
	def exclude(self, exclude_list):
		self.data = self.data.drop(index=exclude_list)
		
	# exclude list of points based on x values 
	def excluderange(self, filename, exclude_range, exclude_range_x):
		data_values = np.array(Data(f"{filename}").data[exclude_range_x])
		indices = np.where(data_values > exclude_range)[0]
		self.data = self.data.drop(index=indices)
		
	# group by scan name, compute mean 
	def group_by_mean(self, scan_name):
		mean = self.data.groupby([scan_name]).mean().reset_index()
		sem = self.data.groupby([scan_name]).sem().reset_index().add_prefix("em_")
		std = self.data.groupby([scan_name]).std().reset_index().add_prefix("e_")
		self.avg_data = pd.concat([mean, std, sem], axis=1)
		
	# group by scan name, compute mendian 
	def group_by_median(self, scan_name):
		med = self.data.groupby([scan_name]).median().reset_index()
		upper = self.data.groupby([scan_name]).quantile(0.68).reset_index().add_prefix("upper_")
		lower = self.data.groupby([scan_name]).quantile(0.32).reset_index().add_prefix("lower_")
		self.med_data = pd.concat([med, upper, lower], axis=1)
				
# plot raw data or average
	def plot(self, names, label=None, axes_labels=None):
		self.fig = plt.figure()
		self.ax = plt.subplot()
		if label==None:
			label = self.filename

		if hasattr(self, 'avg_data'): # check for averaging
			self.ax.errorbar(self.avg_data[f"{names[0]}"], self.avg_data[f"{names[1]}"], 
				yerr=self.avg_data[f"em_{names[1]}"], **styles[0],
				# capsize=2, marker='o', ls='',
				label=label)
		else:
			self.ax.plot(self.data[f"{names[0]}"], self.data[f"{names[1]}"], **styles[0],
				label = label)
			
		if axes_labels == None:
			axes_labels = [f"{names[0]}", f"{names[1]}"]
			
		self.ax.set_xlabel(axes_labels[0])
		self.ax.set_ylabel(axes_labels[1])
		self.ax.legend()
		
#residuals 
	def plot_residuals(self, fit_func, names, guess=None, label=None, axes_labels=None):
		self.fig = plt.figure()
		self.ax = plt.subplot()
		fit_data = np.array(self.data[names])
		func, default_guess, param_names = fit_func(fit_data)
		if guess is None:	
			guess = default_guess
		if hasattr(self, 'avg_data'): # check for averaging
			popt, pcov = curve_fit(func, self.avg_data[f"{names[0]}"], 
						  self.avg_data[f"{names[1]}"],p0=guess, 
						  sigma=self.avg_data[f"em_{names[1]}"])
		else:
			popt, pcov = curve_fit(func, self.data[f"{names[0]}"], 
						  self.data[f"{names[1]}"],p0=guess)
		residuals = self.data[f"{names[1]}"] - func( self.data[f"{names[0]}"],*popt)

		if label==None:
			label = self.filename
		self.ax.plot(self.data[f"{names[0]}"], self.data[f"{names[0]}"]*0, 
			   **styles[0],
			#    linestyle='-'
			   )
		if hasattr(self, 'avg_data'): # check for averaging
			self.ax.errorbar(self.avg_data[f"{names[0]}"], self.avg_data[f"{names[1]}"]- func( self.avg_data[f"{names[0]}"],*popt), 
				yerr=self.avg_data[f"em_{names[1]}"], 
				**styles[0],
				# capsize=2, marker='o', ls='',
				label=label)
		else:
			self.ax.plot(self.data[f"{names[0]}"], residuals,
				**styles[0],
				label = label)
			
		if axes_labels == None:
			axes_labels = [f"{names[0]}", f"{names[1]}"]
			
		self.ax.set_xlabel(axes_labels[0])
		self.ax.set_ylabel(axes_labels[1])
		self.ax.legend()
	
# plotting fit and residuals on subplots 

	def subplot_fit_and_residuals(self, fit_func, names, guess=None, label=None, axes_labels=None):

		self.fig, self.ax = plt.subplots(2,1)

		fit_data = np.array(self.data[names])
		func, default_guess, param_names = fit_func(fit_data)
			
		if guess is None:	
			guess = default_guess
		if hasattr(self, 'avg_data'): # check for averaging
			self.popt, self.pcov = curve_fit(func, self.avg_data[f"{names[0]}"], 
						  self.avg_data[f"{names[1]}"],p0=guess, 
						  sigma=self.avg_data[f"em_{names[1]}"])
		else:
			self.popt, self.pcov = curve_fit(func, self.data[f"{names[0]}"], 
						  self.data[f"{names[1]}"],p0=guess)
		self.perr = np.sqrt(np.diag(self.pcov))
		residuals = self.data[f"{names[1]}"] - func( self.data[f"{names[0]}"]**2,*self.popt)

		self.parameter_table = tabulate([['Values', *self.popt], ['Errors', *self.perr]], 
								 headers=param_names)
		print(self.parameter_table)
		
		if label==None:
			label = self.filename
 			
 #plotting data pts 
		if hasattr(self, 'avg_data'): # check for averaging
			self.ax[0].errorbar(self.avg_data[f"{names[0]}"], self.avg_data[f"{names[1]}"], 
				yerr=self.avg_data[f"em_{names[1]}"], capsize=2, marker='o', ls='',
				label=label)
		else:
			self.ax[0].plot(self.data[f"{names[0]}"], self.data[f"{names[1]}"], 'o',
				label = label)
			
		if axes_labels == None:
			axes_labels = [f"{names[0]}", f"{names[1]}"]
			
	#plotting fit
		if hasattr(self, 'ax'): # check for plot
			num = 500
			xlist = np.linspace(self.data[f"{names[0]}"].min(), 
					   self.data[f"{names[0]}"].max(), num)
			self.ax[0].plot(xlist, func(xlist, *self.popt))
			
		self.ax[1].plot(self.data[f"{names[0]}"], self.data[f"{names[0]}"]*0, linestyle='-')
		if hasattr(self, 'avg_data'): # check for averaging
			self.ax[1].errorbar(self.avg_data[f"{names[0]}"], self.avg_data[f"{names[1]}"]- func( self.avg_data[f"{names[0]}"],*self.popt), 
				yerr=self.avg_data[f"em_{names[1]}"], capsize=2, marker='o', ls='',
				label=label)
		else:
			self.ax[1].plot(self.data[f"{names[0]}"], residuals,
				label = label,marker='o')
 			
		if axes_labels == None:
			axes_labels = [f"{names[0]}", f"{names[1]}"]
 			
		self.ax[0].set_yscale('log')
		self.ax[0].set_xscale('log')
		self.ax[0].set_title(fit_func)
		self.ax[1].set_xlabel(axes_labels[0])
		self.ax[0].set_ylabel(axes_labels[1])
		self.ax[0].legend()


		
# fit data to fit_func and plot if Data has a figure
	def fit(self, fit_func, names, guess=None, label=None, exclude_list=None):
#included an exclusion list here just for the fit so you can plot the whole data set but just fit a portion 
		if exclude_list is None:
			fit_data = np.array(self.data[names])
			self.plot(names, label=label, axes_labels=None)
		else:
			fit_data = np.array(self.data[names])
			self.plot(names, label=label, axes_labels=None)
			
			self.data = self.data.drop(index=exclude_list)
			fit_data = np.array(self.data[names])
		
		func, default_guess, param_names = fit_func(fit_data)
		
		if guess is None:	
			guess = default_guess
			
		if hasattr(self, 'avg_data'): # check for averaging
			self.popt, self.pcov = curve_fit(func, self.avg_data[f"{names[0]}"], 
						  self.avg_data[f"{names[1]}"],p0=guess, 
						  sigma=self.avg_data[f"em_{names[1]}"])
		else:
			self.popt, self.pcov = curve_fit(func, self.data[f"{names[0]}"], 
						  self.data[f"{names[1]}"],p0=guess)
		self.perr = np.sqrt(np.diag(self.pcov))
		
		self.parameter_table = tabulate([['Values', *self.popt], ['Errors', *self.perr]], 
								 headers=param_names)
		print(self.parameter_table)
# 		
		if fit_func == TrapFreq2:
			freq = self.popt[2]*10**3/2/np.pi
			er = self.perr[2]*10**3/2/np.pi
			ot = self.popt[1]*self.popt[2]
			print('The trap frequency is {:.6f} +/-{:.2}'.format(freq,er))
			print('omega*tau is',ot)
 							
		if hasattr(self, 'ax'): # check for plot
			num = 500
			xlist = np.linspace(self.data[f"{names[0]}"].min(), 
					   self.data[f"{names[0]}"].max(), num)
			self.ax.plot(xlist, func(xlist, *self.popt),
			linestyle='-',marker='', color='orange',
			)
			self.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

		
# fit data to fit_func and plot if Data has a figure
	def fitnoplots(self, fit_func, names, guess=None):
		fit_data = np.array(self.data[names])
		func, default_guess, param_names = fit_func(fit_data)
 		
		if guess is None:	
			guess = default_guess
			
		if hasattr(self, 'avg_data'): # check for averaging
			self.popt, self.pcov = curve_fit(func, self.avg_data[f"{names[0]}"], 
						  self.avg_data[f"{names[1]}"],p0=guess, 
						  sigma=self.avg_data[f"em_{names[1]}"])
		else:
			self.popt, self.pcov = curve_fit(func, self.data[f"{names[0]}"], 
						  self.data[f"{names[1]}"],p0=guess)
		self.perr = np.sqrt(np.diag(self.pcov))
		
# plot multiple plots at once (i.e. from a multi scan where we group by one param)				
	def multiplot(self, fit_func, names, guess=None, avg=None, axes_labels=None):
		listi = []
		mvalues = []
		for i in os.listdir(os.path.dirname(self.file)):
			if "=" in i:
				listi.append(i)
		for l in listi:
	#### getting data ####
			newfile = glob(drive + '\\Data\\2024\\*\\*\\*\\' + l)[0]
			newdata = pd.read_table(newfile, delimiter=',')
			
			
	#### avg data ####	
			if avg is None:
				x = newdata[f"{names[0]}"]
				y = newdata[f"{names[1]}"]
			else:
				mean = newdata.groupby([avg]).mean().reset_index()
				sem = newdata.groupby([avg]).sem().reset_index().add_prefix("em_")
				std = newdata.groupby([avg]).std().reset_index().add_prefix("e_")
				avg_data = pd.concat([mean, std, sem], axis=1)
				x = avg_data[f"{names[0]}"]
				y = avg_data[f"{names[1]}"]
				
				    

	#### fitting ####
			fit_data = np.array(x,y)
			func, default_guess, param_names = fit_func(fit_data)
			if guess is None:	
				guess = default_guess
				
			if avg is None:	
				popt, pcov = curve_fit(func, x, y,p0=guess)
			else:
				popt, pcov = curve_fit(func, avg_data[f"{names[0]}"], 
						  avg_data[f"{names[1]}"],p0=guess, 
						  sigma=avg_data[f"em_{names[1]}"])								
			residuals = y - func( x,*popt)
			perr = np.sqrt(np.diag(pcov))
			
			mvalues.append(popt[0])
			
	#### plotting ####
			plt.xlabel(f"{names[0]}")
			plt.ylabel(f"{names[1]}")
			
			if avg is None:
				plt.plot(x,y,marker='d',linestyle='',label=l)
			else: 
				plt.errorbar(avg_data[f"{names[0]}"], avg_data[f"{names[1]}"], yerr=avg_data[f"em_{names[1]}"], capsize=2, marker='o', ls='',
				label=l)
						
			num = 500
			xlist = np.linspace(x.min(), x.max(), num)
			plt.plot(xlist, func(xlist, *popt))
			plt.legend()
# 		print(valueslist)
		
			parameter_table = tabulate([['Values', *popt], ['Errors', *perr]], 
 								 headers=param_names)	
			print(l)
			print(parameter_table)
		print()

# mvals = [-88.99076798864414, -91.2429345878061, -111.3760984309834, -96.48405841019768, -91.21568223696836, -118.73570896749739, -107.22430452927524, -93.46532124367161]
# odtratio = [0.8,0.9,1.1,1.2,1.3,1.4,1.5,1]	
# plt.xlabel('ODT ratio')
# plt.ylabel('Slope')
# plt.plot(odtratio,mvals,'.')
		

# %%
