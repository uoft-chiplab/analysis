# -*- coding: utf-8 -*-
"""
2023-10-05
@author: Chip Lab

Data functions
"""

import os 
from glob import glob
from get_data import *
from library import *
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as curve_fit

# importing data 
Bfield = 201.4 #G 
res = FreqMHz(Bfield, 9/2, -5/2, 9/2, -7/2)

def data(filename, names=['freq','sum95'], autofind=True):
	"""
	Inputs: filename, header names, autofind file or manually input it
	
	Returns: header names used for axes labels, x values, y values 
	"""
	drive = '\\\\UNOBTAINIUM\\E_Carmen_Santiago' # when using Fermium
	if autofind:
		file = glob(drive + '\\Data\\2023\\*\\*\\*\\' + filename)[0] # EXTREMELY greedy
	else :
		file = os.path.join(drive, "Data", "2023", "10 October2023", 
					 "12October2023", "B_ac_dimer_201p2G_scanfreq", filename) #making manual path for the filename
	data = data_from_dat(file, names) #making array of chosen data
	x = data[:,0] - res
# 	x = [x+5 for x in x] #added 5 to every x value
	y = data[:,1]
	return *names, x, y


#kiera playing around at home

# def data(filename, names=['freq','fraction95'], autofind=False):
# 	drive = os.getcwd()[0:3] #'E:\\'
# 	if autofind:
# 		file = glob(drive + '\\Documents\\Github\\\\*\\*\\' + filename)[0] # EXTREMELY greedy
# 	else :
# 		file = os.path.join(\\kierapnd\\Documents\\Graduate\\python\\trials\\, filename) #making manual path for the filename
# 	data = data_from_dat(file, names) #making array of chosen data
# 	x = data[:,0]
# 	#x = [x+5 for x in x]
# 	y = data[:,1]
# 	return *names, x, y


#exclude below certain threshold 

def data_exclude(filename, names=['freq','sum95']):
	"""
	Inputs: filename
	
	Returns: header names used for axes labels, x values, y values 
	"""
	names = [data(filename, names)[0], data(filename, names)[1]]#choosing x , y columns from .dat 
	x = data(filename, names)[2]
	y = data(filename, names)[3]
	mymin = np.where(y < 202.00)[0] # indecies for where y < 80in this case
	x2 = np.delete(x, mymin)
	y2 = np.delete(y, mymin)
	
	return *names, x2, y2

#exclude the repeated point at the end

def list_duplicates(filename, names=['freq','sum95']):
	"""
	Returns: list of indicies of points duplicated more than 3 times 
	"""
	List = data(filename, names)[2].tolist()	
	d1 = {item:List.count(item) for item in List}  # item and their counts
	elems = list(filter(lambda x: d1[x] > 3, d1))  # get duplicate elements
	d2 = dict(zip(range(0, len(List)), List))  # each item and their indices
	# item and their list of duplicate indices in a dictionary 
	dictonary = {item: list(filter(lambda x: d2[x] == item, d2)) for item in elems}
	dups_list = list(dictonary.values())
	print(dictonary)
	
	return dups_list


def data_exclude_points(filename, names=['freq','sum95']):
	"""
	Returns: header names from data, x and y values excluding the duplicated points  
	"""
	names = [data(filename, names)[0], data(filename, names)[1]] #choosing x , y columns from .dat 
	x = data(filename, names)[2]
	y = data(filename, names)[3]	
	xduplicate = list_duplicates(filename, names)[0]
	xduplicate_but1 = xduplicate.pop(0) # getting rid of the first element of the duplicated list so that one of the points stays in the data set
	x2 = np.delete(x, xduplicate)
	y2 = np.delete(y, xduplicate)
	
	return *names, x2, y2

def justdata(filename, names=['freq','sum95'], autofind=True):
	"""
	Inputs: filename, header names, autofind file or manually input it
	
	Returns: header names used for axes labels, x values, y values 
	"""
	drive = '\\\\UNOBTAINIUM\\E_Carmen_Santiago' # when using Fermium
	if autofind:
		file = glob(drive + '\\Data\\2023\\*\\*\\*\\' + filename)[0] # EXTREMELY greedy
	else :
		file = os.path.join(drive, "Data", "2023", "10 October2023", 
					 "12October2023", "B_ac_dimer_201p2G_scanfreq", filename) #making manual path for the filename
	data = data_from_dat(file, names) #making array of chosen data
	x = data[:,0]
# 	x = [x+5 for x in x] #added 5 to every x value
	y = data[:,1]
	return  x, y

def avgdata(filename, names, fittype='Gaussian'):
    fitdata = data(filename, names, fittype)
    # plt.title(f"{fittype} fit for {filename}")
    # xlabel = f"{fitdata[0]}"
    # ylabel = f"{fitdata[1]}"
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.plot(fitdata[2], fitdata[3], 'go')
    
    fig = plt.figure(1)
    namex = data(filename, names)[0] 
    namey = data(filename, names)[1] #choosing x , y columns from .dat 
    x = data(filename, names)[2]
    y = data(filename, names)[3]
    data2 = pd.DataFrame({namex: x, namey: y}) 
    
    # data2.set_index('x2',inplace=True)
    
    avgdatay = data2.groupby([namex])[namey].mean().apply(np.array).tolist()
    avgdatay = data2.groupby([namex])[namey].apply(np.mean)
    # plt.subplot(2,2,1)
    
    avgdata = data2.groupby([namex])[namey].mean()

    
    avgdata.plot( marker = '.', linestyle = 'none')
    
    guess = [-2800,-4.1,0.05,27221]
    popt, pcov = curve_fit.curve_fit(Gaussian, avgdata, avgdata ,p0=guess, maxfev=5000)
    ym = Gaussian(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
    
    print(popt)
    errors = np.sqrt(np.diag(pcov))
    # freq = 0.01
# 	period = 1/freq
# 	delay = popt[1] % (3.141592654) /freq
    values = list([*popt])
	#errors = np.concatenate((errors, [errors[1]/2/3.14, period * errors[1]/popt[1], delay * errors[2]/popt[2]]))
    # print(tabulate([['Values', *values], ['Errors', *errors]], headers=['Amplitude','phase','offset', 'freq', 'period', 'delay']))
	
    plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	
    return fig
            
	

def data_choice(datatype, filename, names=['freq','sum95']):
	"""
	Choose the data type you want  
	"""
	if datatype == 'raw':
		fitdata = data(filename, names)
	else:
		if datatype == 'exclude':
			fitdata = data_exclude(filename, names)
		else:
			if datatype == 'exclude multiple points':
				fitdata = data_exclude_points(filename, names)
				
				
				
