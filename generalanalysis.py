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
import seaborn as sns
###

# importing data 

def data(filename):
	names = ["time (us)", "sum95"] #choosing x , y columns from .dat 
	path = os.getcwd() #getting the path 
	parent = os.path.dirname(path) #getting the directory name 
	parentparent = os.path.dirname(parent)
	file = os.path.join(parentparent, "Data", "2023", "10 October2023", 
					 "04October2023", "D_dimer_rabi_osc_9VVAscantime", filename) #making path for the filename
	data = data_from_dat(file, names) #making array of chosen data
	x = data[:,0] 
	y = data[:,1]
	
	return names[0], names[1], x, y 

#exclude below certain threshold 

def data_exclude(filename):
	names = ["time", "fCtr1"] #choosing x , y columns from .dat 
	path = os.getcwd() #getting the path 
	parent = os.path.dirname(path) #getting the directory name 
	file = os.path.join(parent, "Data", "2023", "02 February2023", 
					 "01February2023", "F_Ztrapfreq_LAT1_80ER", filename) #making path for the filename
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
	fitdata = data(filename)
	plt.title(f"Gaussian fit for {filename}")
	plt.xlabel(f"{fitdata[0]}")
	plt.ylabel(f"{fitdata[1]}")
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:	
		guess = [-(max(fitdata[3])-min(fitdata[3])), 
		   fitdata[2][fitdata[3].argmin()],0.04,np.mean(fitdata[3])]
	popt, pcov = curve_fit.curve_fit(Gaussian, fitdata[2], fitdata[3],p0=guess)
	ym = Gaussian(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values', *popt], ['Errors', *errors]], 
				headers=['Amplitude', 'Frequency','Width','Background']))


#plotting raw data with linear function 
#guess=['Slope', 'Offset']
def plotlinear(filename, guess=None):
	fitdata = data(filename)
	plt.title(f"Linear fit for {filename}")
	plt.xlabel(f"{fitdata[0]}")
	plt.ylabel(f"{fitdata[1]}")
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [(max(fitdata[3])-min(fitdata[3]))/(max(fitdata[2])-min(fitdata[2])), 
		   fitdata[2][fitdata[3].argmin()]]
	popt, pcov = curve_fit.curve_fit(Linear, fitdata[2], fitdata[3],p0=guess)
	ym = Linear(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Slope', 'Offset']))


#plotting raw data with Lorentzian function 
#guess=['Amplitude', 'b**2' ,'Frequency', 'Width', 'Background']
def plotlorentzian(filename, guess=None):
	fitdata = data(filename)
	plt.title(f"Lorentzian fit for {filename}")
	plt.xlabel(f"{fitdata[0]}")
	plt.ylabel(f"{fitdata[1]}")
	plt.plot(fitdata[2],fitdata[3],'bo')
	if guess is None:
		guess = [(max(fitdata[3])-(sorted(set(fitdata[3]))[2])), 1, 
		   fitdata[2][fitdata[3].argmin()], 0.04, np.mean(fitdata[3])]
	popt, pcov = curve_fit.curve_fit(Lorentzian, fitdata[2], fitdata[3],p0=guess)
	ym = Lorentzian(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'b**2' ,'Frequency', 'Width', 'Background']))


#plotting raw data with Sinc function 
#guess=['Amplitude', 'Frequency', 'Width', 'Background']
def plotsinc(filename, guess=None):
	fitdata = data(filename)
	plt.title(f"Sinc fit for {filename}")
	plt.xlabel(f"{fitdata[0]}")
	plt.ylabel(f"{fitdata[1]}")
	plt.plot(data(filename)[2],data(filename)[3],'go')
	if guess is None:
		guess = [(max(fitdata[3])-(sorted(set(fitdata[3]))[2])), 
		   (sorted(set(fitdata[3]))[3]+sorted(set(fitdata[3]))[2]), 4, np.mean(fitdata[3])]
	popt, pcov = curve_fit.curve_fit(Sinc, fitdata[2], fitdata[3],p0=guess)
	ym = Sinc(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'phase', 'Width', 'Background']))


#plotting raw data with Sinc**2 function 
#guess=['Amplitude', 'Frequency', 'Width', 'Background']
def plotsinc2(filename, guess=None):
	fitdata = data(filename)
	plt.title(f"Sinc**2 fit for {filename}")
	plt.xlabel(f"{fitdata[0]}")
	plt.ylabel(f"{fitdata[1]}")
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [(max(fitdata[3])-(sorted(set(fitdata[3]))[0])), 
		   (sorted(set(fitdata[3]))[1]+sorted(set(fitdata[3]))[0]), 4, np.mean(fitdata[3])]
	popt, pcov = curve_fit.curve_fit(Sinc2, fitdata[2], fitdata[3],p0=guess)
	ym = Sinc2(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'phase', 'Width', 'Background']))



# plotting raw data with Trap Freq function 
# guess=['Amplitude', 'tau', 'f', 'fc', 's', 'C']
def plottrapfreq(filename, guess=None):
	fitdata = data(filename)
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	# plot data
	fig1 = plt.figure(0)
	plt.title(f"Trap Freq fit for {filename}")
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
# 	plt.ylim(90, 110) # sets y axis limits
	# fit data
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [6000, 0.25, 2  ,-2 , 100, -0.1] # 'Amplitude', 'tau', 'f', 'phase', 'C', 'm'
					# where m is the slope of linear term 
	popt, pcov = curve_fit.curve_fit(TrapFreq, fitdata[2], fitdata[3],p0=guess)
	num = 200
	ym = TrapFreq(np.linspace(max(fitdata[2]),min(fitdata[2]),num=num),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=num),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'tau', 'omega', 'phase', 'Offset', 'Slope']))
	# plot residuals
	residuals = fitdata[3] - TrapFreq(fitdata[2],*popt)
	fig2 = plt.figure(1)
	plt.plot(fitdata[2],fitdata[3]*0,'-')
	plt.plot(fitdata[2], residuals,'g+')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel +" Residuals")
	plt.show(fig1,fig2)


## 
#plotting raw data with Rabi Freq function  
#guess=['Amplitude', 'b', 'x0', 'C']
#being weird?? 

def plotrabifreq(filename, guess=None):
	fitdata = data(filename)
	plt.title(f"Rabi Freq fit for {filename}")
	plt.xlabel(f"{fitdata[0]}")
	plt.ylabel(f"{fitdata[1]}")
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
	# 	guess = [max(fitdata[3])-min(fitdata[3]), 1, min(fitdata[3]), 0]
		guess = [1,1,1,0]
	popt, pcov = curve_fit.curve_fit(RabiFreq, fitdata[2], fitdata[3],p0=guess)
	ym = RabiFreq(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'b', 'x0', 'C']))


#plotting raw data with Parabola function 
#guess=['A', 'x0', 'Offset']
def plotparabola(filename, guess=None):
	fitdata = date(filename)
	plt.title(f"Parabolic fit for {filename}")
	plt.xlabel(f"{fitdata[0]}")
	plt.ylabel(f"{fitdata[1]}")
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [1, 1, 1]
	popt, pcov = curve_fit.curve_fit(Parabola, fitdata[2], fitdata[3],p0=guess)
	ym = Parabola(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values', *popt], ['Errors', *errors[0]]], 
				headers=['A', 'x0', 'Offset']))


#plotting raw data with exponential function 
#guess=['Amplitude', 'sigma']
def plotexp(filename, guess=None):
	fitdata = data(filename)
	plt.title(f"Exponential fit for {filename}")
	plt.xlabel(f"{fitdata[0]}")
	plt.ylabel(f"{fitdata[1]}")
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [max(fitdata[3])-min(fitdata[3]), 1]
	popt, pcov = curve_fit.curve_fit(Expontial, fitdata[2], fitdata[3],p0=guess)
	ym = Expontial(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], headers=['Amplitude', 'sigma']))


#plotting raw data with Rabiline function 
#['b', 'l', 'm','A','s','j','k','p'] ????
def plotrabiline(filename, guess=None):
	fitdata = data(filename)
	plt.title(f"Rabi Line fit for {filename}")
	plt.xlabel(f"{fitdata[0]}")
	plt.ylabel(f"{fitdata[1]}")
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [1, 1, 1, 1, 1, 1, 1, 0]
	popt, pcov = curve_fit.curve_fit(RabiLine, fitdata[2], fitdata[3],p0=guess)
	ym = RabiLine(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['b', 'l', 'm','A','s','j','k','p']))


#plotting raw data with Erfc function 
#guess=['Amplitude', 'x0', 'sigma', 'Offset']
def ploterfc(filename, guess=None):
	fitdata = data(filename)
	plt.title(f"Erfc fit for {filename}")
	plt.xlabel(f"{fitdata[0]}")
	plt.ylabel(f"{fitdata[1]}")
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [1, 1, 1, 0]
	popt, pcov = curve_fit.curve_fit(ErfcFit, fitdata[2], fitdata[3],p0=guess)
	ym = ErfcFit(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'x0', 'sigma', 'Offset']))

