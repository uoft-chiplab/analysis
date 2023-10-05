# -*- coding: utf-8 -*-
"""
2023-09-25
@author: Chip Lab

Fits script

"""
from analysisfunctions import * # includes numpy and constants
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import os 
import scipy.optimize as curve_fit
from get_data import *
from tabulate import tabulate # pip install tabulate
from collections import defaultdict
# import seaborn as sns
###

# importing data 

def data(filename):
	names = ["delaytime", "field"] #choosing x , y columns from .dat 
	path = os.getcwd() #getting the path 
	parent = os.path.dirname(path) #getting the directory name 
	parentparent = os.path.dirname(parent) # i had to go back another folder since putting this code on github
	file = os.path.join(parentparent, "Data", "2023", "10 October2023", 
					 "04October2023", "Summary", filename) #making path for the filename
	data = data_from_dat(file, names) #making array of chosen data
	x = data[:,0] 
	y = data[:,1]
	return *names, x, y

#exclude below certain threshold 

def data_exclude(filename):
	names = ["delaytime", "field"] #choosing x , y columns from .dat 
	x = data(filename)[2]
	y = data(filename)[3]
	mymin = np.where(y < 202.00)[0] # indecies for where y < 80in this case
	x2 = np.delete(x, mymin)
	y2 = np.delete(y, mymin)
	
	return names[0], names[1], x2, y2


def list_duplicates(filename):
	List = data(filename)[2].tolist()	
	d1 = {item:List.count(item) for item in List}  # item and their counts
	elems = list(filter(lambda x: d1[x] > 3, d1))  # get duplicate elements
	d2 = dict(zip(range(0, len(List)), List))  # each item and their indices
	# item and their list of duplicate indices in a dictionary 
	dictonary = {item: list(filter(lambda x: d2[x] == item, d2)) for item in elems}
	dups_list = list(dictonary.values())
	print(dictonary)
	
	return dups_list


def data_exclude_points(filename):
	x = data(filename)[2]
	y = data(filename)[3]	
# 	print("Duplicate elements in given array are: "); 
# 	for i in range(0, len(x)):    
# 		   for j in range(i+1, len(x)):    
# 			         if(x[i] == x[j]):    
# 						       print(x[j])
	xduplicate = list_duplicates(filename)[0]
	xduplicate_but1 = xduplicate.pop(0) # getting rid of the first element of the duplicated list so that one of the points stays in the data set
	
	
	x2 = np.delete(x, xduplicate)
	y2 = np.delete(y, xduplicate)
	return x2, y2
	


# average data 


#plotting raw data with cos 
#guess=['Amplitude', 'Frequency','Width','Background']
def plotcos(filename, guess=None, residuals=False, datatype='exclude'):
	fig1 = plt.figure(0)
	if datatype == 'raw':
		fitdata = data(filename)
	else:
		if datatype == 'exclude':
			fitdata = data_exclude(filename)
		else:
			fitdata = data_exclude_points(filename)
	plt.title(f"Cos fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2], fitdata[3], 'go')
	if guess is None:	
		guess = [-0.2, 0, 10, 202]
	popt, pcov = curve_fit.curve_fit(Cos, fitdata[2], fitdata[3],p0=guess)
	ym = Cos(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	freq = popt[1]/2/3.14
	period = 1/freq
	delay = popt[2] % (3.141592654) /popt[1]
	values = list([*popt, freq, period, delay])
	errors = np.concatenate((errors, [errors[1]/2/3.14, period * errors[1]/popt[1], delay * errors[2]/popt[2]]))
	print(tabulate([['Values', *values], ['Errors', *errors]], headers=['Amplitude','omega','phase','offset', 'freq', 'period', 'delay']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Cos(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals, 'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)


#plotting raw data with sin 
#guess=['Amplitude', 'Frequency','Width','Background']
def plotsin(filename, guess=None, errors=False, residuals=False):
	fig1 = plt.figure(0)
	fitdata = data(filename)
	plt.title(f"Sin fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:	
		guess = [-0.2, 
		   0,10,202]
	popt, pcov = curve_fit.curve_fit(Sin, fitdata[2], fitdata[3],p0=guess)
	ym = Sin(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	freq = popt[1]/2/3.14
	period = 1/freq
	delay = popt[2] % (3.141592654) /popt[1]
	values = list([*popt, freq, period, delay])
	errors = np.concatenate((errors, [errors[1]/2/3.14, period * errors[1]/popt[1], delay * errors[2]/popt[2]]))
	print(tabulate([['Values', *values], ['Errors', *errors]], headers=['Amplitude','omega','phase','offset', 'freq', 'period', 'delay']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Sin(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)


#plotting raw data with gaussian 
#guess=['Amplitude', 'Frequency','Width','Background']
def plotgaussian(filename, guess=None, residuals=False):
	fig1 = plt.figure(0)
	fitdata = data(filename)
	plt.title(f"Gaussian fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:	
		guess = [-(max(fitdata[3])-min(fitdata[3])), 
		   fitdata[2][fitdata[3].argmin()],0.04,np.mean(fitdata[3])]
	popt, pcov = curve_fit.curve_fit(Gaussian, fitdata[2], fitdata[3],p0=guess)
	ym = Gaussian(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values', *popt], ['Errors', *errors]], 
				headers=['Amplitude','Frequency','Width','Background']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)

#plotting raw data with linear function 
#guess=['Slope', 'Offset']
def plotlinear(filename, guess=None, residuals=False):
	fig1 = plt.figure(0)
	fitdata = data(filename)
	plt.title(f"Linear fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
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
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)

#plotting raw data with Lorentzian function 
#guess=['Amplitude', 'b**2' ,'Frequency', 'Width', 'Background']
def plotlorentzian(filename, guess=None, residuals=False):
	fig1 = plt.figure(0)
	fitdata = data(filename)
	plt.title(f"Lorentzian fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
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
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)


#plotting raw data with Sinc function 
#guess=['Amplitude', 'Frequency', 'Width', 'Background']
def plotsinc(filename, guess=None, residuals=False):
	fig1 = plt.figure(0)
	fitdata = data(filename)
	plt.title(f"Sinc fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [(max(fitdata[3])-(sorted(set(fitdata[3]))[2])), 
		   (sorted(set(fitdata[2]))[1]+sorted(set(fitdata[2]))[-1])/2, (sorted(set(fitdata[2]))[1]-sorted(set(fitdata[2]))[-1])/2, np.mean(fitdata[3])]
		print(guess)
	popt, pcov = curve_fit.curve_fit(Sinc, fitdata[2], fitdata[3],p0=guess)
	ym = Sinc(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'center', 'Width', 'offset']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)
	

#plotting raw data with Sinc**2 function 
#guess=['Amplitude', 'Frequency', 'Width', 'Background']
def plotsinc2(filename, guess=None, residuals=False):
	fig1 = plt.figure(0)
	fitdata = data(filename)
	plt.title(f"Sinc**2 fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
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
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)


# plotting raw data with Trap Freq function 
# guess=['Amplitude', 'tau', 'f', 'fc', 's', 'C']
def plottrapfreq(filename, guess=None, residuals=False):
	fig1 = plt.figure(0)
	fitdata = data(filename)
	# plot data
	fig1 = plt.figure(0)
	plt.title(f"Trap Freq fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.xlim(-0.01, 0.2) # sets y axis limits
	# fit data
	for i in range(len(fitdata[2])) :
		if fitdata[2][i] == 0.7 :
 			fitdata[2][i] = 0 
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [10000, 0.05, 20  ,-2 , 100, -0.1] # 'Amplitude', 'tau', 'omega', 'phase', 'C', 'm'
					# where m is the slope of linear term 
	popt, pcov = curve_fit.curve_fit(TrapFreq, fitdata[2], fitdata[3],p0=guess)
	num = 200
	ym = TrapFreq(np.linspace(max(fitdata[2]),min(fitdata[2]),num=num),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=num),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'tau', 'omega', 'phase', 'Offset', 'Slope']))
	figures = [fig1]
	# plot residuals
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)
	
	

# plotting raw data with Trap Freq function 
# guess=['Amplitude', 'tau', 'f', 'fc', 's', 'C']
def plottrapfreq2(filename, guess=None, residuals=False):
	fig1 = plt.figure(0)
	fitdata = data(filename)
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	# plot data
	fig1 = plt.figure(0)
	plt.title(f"Trap Freq fit no linear term for {filename}")
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.xlim(-0.01, 0.2) # sets y axis limits
	# fit data
	for i in range(len(fitdata[2])) :
		if fitdata[2][i] == 0.7 :
 			fitdata[2][i] = 0 
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [10000, 0.05, 20  ,-2 , 100] # 'Amplitude', 'tau', 'omega', 'phase', 'C', 'm'
					# where m is the slope of linear term 
	popt, pcov = curve_fit.curve_fit(TrapFreq2, fitdata[2], fitdata[3],p0=guess)
	num = 200
	ym = TrapFreq2(np.linspace(max(fitdata[2]),min(fitdata[2]),num=num),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=num),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'tau', 'omega', 'phase', 'Offset']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)
	
## 
#plotting raw data with Rabi Freq function  
#guess=['Amplitude', 'b', 'x0', 'C']
#being weird?? 

def plotrabifreq(filename, guess=None, residuals=False):
	fig1 = plt.figure(0)
	fitdata = data(filename)
	plt.title(f"Rabi Freq fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
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
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)


#plotting raw data with Parabola function 
#guess=['A', 'x0', 'Offset']
def plotparabola(filename, guess=None, residuals=False):
	fig1 = plt.figure(0)
	fitdata = data(filename)
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.title(f"Parabolic fit for {filename}")
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [-3000, 44.82, 3000]
	popt, pcov = curve_fit.curve_fit(Parabola, fitdata[2], fitdata[3],p0=guess)
	ym = Parabola(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values', *popt], ['Errors', *errors]], 
				headers=['A', 'center', 'Offset']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)

	

#plotting raw data with exponential function 
#guess=['Amplitude', 'sigma']
def plotexp(filename, guess=None, residuals=False):
	fig1 = plt.figure(0)
	fitdata = data(filename)
	plt.title(f"Exponential fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [max(fitdata[3])-min(fitdata[3]), 1]
	popt, pcov = curve_fit.curve_fit(Expontial, fitdata[2], fitdata[3],p0=guess)
	ym = Expontial(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'sigma']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)

#plotting raw data with Rabiline function 
#['b', 'l', 'm','A','s','j','k','p'] ????
def plotrabiline(filename, guess=None, residuals=False):
	fig1 = plt.figure(0)
	fitdata = data(filename)
	plt.title(f"Rabi Line fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [1, 1, 1, 1, 1, 1, 1, 0]
	popt, pcov = curve_fit.curve_fit(RabiLine, fitdata[2], fitdata[3],p0=guess)
	ym = RabiLine(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['b', 'l', 'm','A','s','j','k','p']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)

#plotting raw data with Erfc function 
#guess=['Amplitude', 'x0', 'sigma', 'Offset']
def ploterfc(filename, guess=None, residuals=False):
	fig1 = plt.figure(0)
	fitdata = data(filename)
	plt.title(f"Erfc fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [1, 1, 1, 0]
	popt, pcov = curve_fit.curve_fit(ErfcFit, fitdata[2], fitdata[3],p0=guess)
	ym = ErfcFit(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'x0', 'sigma', 'Offset']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)
