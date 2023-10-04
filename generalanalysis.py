# -*- coding: utf-8 -*-
"""
2023-09-25
@author: Chip Lab

Fits script

"""
from analysisfunctions import * # includes numpy and constants
import matplotlib.pyplot as plt
import os 
import scipy.optimize as curve_fit
from get_data import *
from tabulate import tabulate # pip install tabulate
import statsmodels.api as sm
from statsmodels.formula.api import ols
###

# importing data 

def data(filename):
	names = ["time (us)", "sum95"] #choosing x , y columns from .dat 
	path = os.getcwd() #getting the path 
	parent = os.path.dirname(path) #getting the directory name 
	parentparent = os.path.dirname(parent)
	file = os.path.join(parentparent, "Data", "2023", "10 October2023", "04October2023", "D_dimer_rabi_osc_9VVAscantime", filename) #making path for the filename
	data = data_from_dat(file, names) #making array of chosen data
	x = data[:,0] 
	y = data[:,1]
	
	return names[0], names[1], x, y 

#exclude below certain threshold 

def data_exclude(filename):
	names = ["time", "fCtr1"] #choosing x , y columns from .dat 
	path = os.getcwd() #getting the path 
	parent = os.path.dirname(path) #getting the directory name 
	file = os.path.join(parent, "Data", "2023", "02 February2023", "01February2023", "F_Ztrapfreq_LAT1_80ER", filename) #making path for the filename
	data = data_from_dat(file, names) #making array of chosen data
	x = data[:,0]
	y = data[:,1]
	mymin = np.where(y < 70)[0] # indecies for where y < 80in this case
	x2 = np.delete(x, mymin)
	y2 = np.delete(y, mymin)
	
	return names[0], names[1], x2, y2

# average data 




#plotting raw data with gaussian 
#guess=['Amplitude', 'Frequency','Width','Background']
def plotgaussian(filename, guess=None):
	plt.title(f"Gaussian fit for {filename}")
	plt.xlabel(f"{data(filename)[0]}")
	plt.ylabel(f"{data(filename)[1]}")
	plt.plot(data(filename)[2],data(filename)[3],'go')
	if guess is None:	
		guess = [-(max(data(filename)[3])-min(data(filename)[3])), data(filename)[2][data(filename)[3].argmin()],0.04,np.mean(data(filename)[3])]
	popt, pcov = curve_fit.curve_fit(Gaussian, data(filename)[2], data(filename)[3],p0=guess)
	ym = Gaussian(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),*popt)
	plt.plot(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',popt[0],popt[1],popt[2],popt[3]], ['Errors',errors[0],errors[1],errors[2],errors[3]]], headers=['Amplitude', 'Frequency','Width','Background']))


#plotting raw data with linear function 
#guess=['Slope', 'Offset']
def plotlinear(filename, guess=None):
	plt.title(f"Linear fit for {filename}")
	plt.xlabel(f"{data(filename)[0]}")
	plt.ylabel(f"{data(filename)[1]}")
	plt.plot(data(filename)[2],data(filename)[3],'go')
	if guess is None:
		guess = [(max(data(filename)[3])-min(data(filename)[3]))/(max(data(filename)[2])-min(data(filename)[2])), data(filename)[2][data(filename)[3].argmin()]]
	popt, pcov = curve_fit.curve_fit(Linear, data(filename)[2], data(filename)[3],p0=guess)
	ym = Linear(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),*popt)
	plt.plot(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',popt[0],popt[1]], ['Errors',errors[0],errors[1]]], headers=['Slope', 'Offset']))


#plotting raw data with Lorentzian function 
#guess=['Amplitude', 'b**2' ,'Frequency', 'Width', 'Background']
def plotlorentzian(filename, guess=None):
	plt.title(f"Lorentzian fit for {filename}")
	plt.xlabel(f"{data(filename)[0]}")
	plt.ylabel(f"{data(filename)[1]}")
	plt.plot(data(filename)[2],data(filename)[3],'bo')
	if guess is None:
		guess = [(max(data(filename)[3])-(sorted(set(data(filename)[3]))[2])), 1, data(filename)[2][data(filename)[3].argmin()], 0.04, np.mean(data(filename)[3])]
	popt, pcov = curve_fit.curve_fit(Lorentzian, data(filename)[2], data(filename)[3],p0=guess)
	ym = Lorentzian(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),*popt)
	plt.plot(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',popt[0],popt[1],popt[2],popt[3],popt[4]], ['Errors',errors[0],errors[1],errors[2],errors[3],errors[4]]], headers=['Amplitude', 'b**2' ,'Frequency', 'Width', 'Background']))


#plotting raw data with Sinc function 
#guess=['Amplitude', 'Frequency', 'Width', 'Background']
def plotsinc(filename, guess=None):
	plt.title(f"Sinc fit for {filename}")
	plt.xlabel(f"{data(filename)[0]}")
	plt.ylabel(f"{data(filename)[1]}")
	plt.plot(data(filename)[2],data(filename)[3],'go')
	if guess is None:
		guess = [(max(data(filename)[3])-(sorted(set(data(filename)[3]))[2])), (sorted(set(data(filename)[3]))[3]+sorted(set(data(filename)[3]))[2]), 4, np.mean(data(filename)[3])]
	popt, pcov = curve_fit.curve_fit(Sinc, data(filename)[2], data(filename)[3],p0=guess)
	ym = Sinc(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),*popt)
	plt.plot(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',popt[0],popt[1],popt[2],popt[3]], ['Errors',errors[0],errors[1],errors[2],errors[3]]], headers=['Amplitude', 'Frequency', 'Width', 'Background']))


#plotting raw data with Sinc**2 function 
#guess=['Amplitude', 'Frequency', 'Width', 'Background']
def plotsinc2(filename, guess=None):
	plt.title(f"Sinc**2 fit for {filename}")
	plt.xlabel(f"{data(filename)[0]}")
	plt.ylabel(f"{data(filename)[1]}")
	plt.plot(data(filename)[2],data(filename)[3],'go')
	if guess is None:
		guess = [(max(data(filename)[3])-(sorted(set(data(filename)[3]))[0])), (sorted(set(data(filename)[3]))[1]+sorted(set(data(filename)[3]))[0]), 4, np.mean(data(filename)[3])]
	popt, pcov = curve_fit.curve_fit(Sinc2, data(filename)[2], data(filename)[3],p0=guess)
	ym = Sinc2(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),*popt)
	plt.plot(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',popt[0],popt[1],popt[2],popt[3]], ['Errors',errors[0],errors[1],errors[2],errors[3]]], headers=['Amplitude', 'Frequency', 'Width', 'Background']))



#plotting raw data with Trap Freq function 
#guess=['Amplitude', 'tau', 'f', 'fc', 's', 'C']
def plottrapfreq(filename, guess=None):
	plt.title(f"Trap Freq fit for {filename}")
	plt.xlabel(f"{data(filename)[0]}")
	plt.ylabel(f"{data(filename)[1]}")
# 	plt.ylim(90, 110) # sets y axes
	plt.plot(data(filename)[2],data(filename)[3],'go')
	if guess is None:
		guess = [6000, 0.25, 2  ,-2 , 100, -0.1] # 'Amplitude', 'tau', 'f', 'fc', 'C' # extra slope to account for loss over time ?? 
	popt, pcov = curve_fit.curve_fit(TrapFreq, data(filename)[2], data(filename)[3],p0=guess)
	ym = TrapFreq(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),*popt)
	plt.plot(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]], ['Errors',errors[0],errors[1],errors[2],errors[3],errors[4],errors[5]]], headers=['Amplitude', 'tau', 'omega', 'phase', 'Offset', 'Slope']))



#plotting raw data with Rabi Freq function  
#guess=['Amplitude', 'b', 'x0', 'C']
#being weird?? 

def plotrabifreq(filename, guess=None):
	plt.title(f"Rabi Freq fit for {filename}")
	plt.xlabel(f"{data(filename)[0]}")
	plt.ylabel(f"{data(filename)[1]}")
	plt.plot(data(filename)[2],data(filename)[3],'go')
	if guess is None:
	# 	guess = [max(data(filename)[3])-min(data(filename)[3]), 1, min(data(filename)[3]), 0]
		guess = [1,1,1,0]
	popt, pcov = curve_fit.curve_fit(RabiFreq, data(filename)[2], data(filename)[3],p0=guess)
	ym = RabiFreq(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),*popt)
	plt.plot(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',popt[0],popt[1],popt[2],popt[3]], ['Errors',errors[0],errors[1],errors[2],errors[3]]], headers=['Amplitude', 'b', 'x0', 'C']))


#plotting raw data with Parabola function 
#guess=['A', 'x0', 'Offset']
def plotparabola(filename, guess=None):
	plt.title(f"Parabolic fit for {filename}")
	plt.xlabel(f"{data(filename)[0]}")
	plt.ylabel(f"{data(filename)[1]}")
	plt.plot(data(filename)[2],data(filename)[3],'go')
	if guess is None:
		guess = [1, 1, 1]
	popt, pcov = curve_fit.curve_fit(Parabola, data(filename)[2], data(filename)[3],p0=guess)
	ym = Parabola(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),*popt)
	plt.plot(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',popt[0],popt[1],popt[2]], ['Errors',errors[0],errors[1],errors[2]]], headers=['A', 'x0', 'Offset']))


#plotting raw data with exponential function 
#guess=['Amplitude', 'sigma']
def plotexp(filename, guess=None):
	plt.title(f"Exponential fit for {filename}")
	plt.xlabel(f"{data(filename)[0]}")
	plt.ylabel(f"{data(filename)[1]}")
	plt.plot(data(filename)[2],data(filename)[3],'go')
	if guess is None:
		guess = [max(data(filename)[3])-min(data(filename)[3]), 1]
	popt, pcov = curve_fit.curve_fit(Expontial, data(filename)[2], data(filename)[3],p0=guess)
	ym = Expontial(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),*popt)
	plt.plot(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',popt[0],popt[1]], ['Errors',errors[0],errors[1]]], headers=['Amplitude', 'sigma']))


#plotting raw data with Rabiline function 
#['b', 'l', 'm','A','s','j','k','p'] ????
def plotrabiline(filename, guess=None):
	plt.title(f"Rabi Line fit for {filename}")
	plt.xlabel(f"{data(filename)[0]}")
	plt.ylabel(f"{data(filename)[1]}")
	plt.plot(data(filename)[2],data(filename)[3],'go')
	if guess is None:
		guess = [1, 1, 1, 1, 1, 1, 1, 0]
	popt, pcov = curve_fit.curve_fit(RabiLine, data(filename)[2], data(filename)[3],p0=guess)
	ym = RabiLine(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),*popt)
	plt.plot(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7]], ['Errors',errors[0],errors[1],errors[2],errors[3],errors[4],errors[5],errors[6],errors[7]]], headers=['b', 'l', 'm','A','s','j','k','p']))


#plotting raw data with Erfc function 
#guess=['Amplitude', 'x0', 'sigma', 'Offset']
def ploterfc(filename, guess=None):
	plt.title(f"Erfc fit for {filename}")
	plt.xlabel(f"{data(filename)[0]}")
	plt.ylabel(f"{data(filename)[1]}")
	plt.plot(data(filename)[2],data(filename)[3],'go')
	if guess is None:
		guess = [1, 1, 1, 0]
	popt, pcov = curve_fit.curve_fit(ErfcFit, data(filename)[2], data(filename)[3],p0=guess)
	ym = ErfcFit(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),*popt)
	plt.plot(np.linspace(max(data(filename)[2]),min(data(filename)[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',popt[0],popt[1],popt[2],popt[3]], ['Errors',errors[0],errors[1],errors[2],errors[3]]], headers=['Amplitude', 'x0', 'sigma', 'Offset']))

