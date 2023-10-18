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
import scipy.optimize as curve_fit
import pandas as pd



# importing data 
Bfield = 201 #G 
res = FreqMHz(Bfield, 9/2, -5/2, 9/2, -7/2)

def data(filename, datatype='raw',  names=['freq','sum95'],  autofind=True):
	"""
	Inputs: filename, header names, autofind file or manually input it
	
	Returns: header names used for axes labels, x values, y values 
	"""
	drive = '\\\\UNOBTAINIUM\\E_Carmen_Santiago' # when using Fermium
	if autofind:
		file = glob(drive + '\\Data\\2023\\*\\*\\*\\' + filename)[0] # EXTREMELY greedy
	else :
		file = os.path.join(drive, "Data", "2023", "09 September2023", 
					 "29September2023", "E_ac_dimer_201G_scanfreq", filename) #making manual path for the filename
	data = data_from_dat(file, names) #making array of chosen data
	x = data[:,0]
# 	x = [x-0.5 for x in x] #added 5 to every x value
	y = data[:,1]
# y = [y-0.5 for y in y] #subtracted 5 to every x value
	return *names, x, y


#KP playing around at home

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

def data_exclude(filename, datatype='raw',  names=['freq','sum95']):
	"""
	Inputs: filename
	
	Returns: header names used for axes labels, x values, y values 
	"""
	names = [data(filename, datatype, names)[0], data(filename, datatype, names)[1]]#choosing x , y columns from .dat 
	x = data(filename, datatype, names)[2]
	y = data(filename, datatype, names)[3]
	mymin = np.where(x >50)[0] # indecies for where y < 80in this case
	x2 = np.delete(x, mymin)
	y2 = np.delete(y, mymin)
	
	return *names, x2, y2

#exclude the repeated point at the end

def list_duplicates(filename ,datatype='raw', names=['freq','sum95']):
	"""
	Returns: list of indicies of points duplicated more than 3 times 
	"""
	List = data(filename, names)[2].tolist()	
	d1 = {item:List.count(item) for item in List}  # item and their counts
	elems = list(filter(lambda x: d1[x] > 5, d1))  # get duplicate elements
	d2 = dict(zip(range(0, len(List)), List))  # each item and their indices
	# item and their list of duplicate indices in a dictionary 
	dictonary = {item: list(filter(lambda x: d2[x] == item, d2)) for item in elems}
	dups_list = list(dictonary.values())
	print(dictonary)
	
	return dups_list


def data_exclude_points(filename, datatype='raw', names=['freq','sum95']):
	"""
	Returns: header names from data, x and y values excluding the duplicated points  
	"""
	names = [data(filename, datatype, names)[0], data(filename, datatype, names)[1]] #choosing x , y columns from .dat 
	x = data(filename, datatype, names)[2]
	y = data(filename, datatype, names)[3]	
	xduplicate = list_duplicates(filename, datatype, names)[0]
	xduplicate_but1 = xduplicate.pop(0) # getting rid of the first element of the duplicated list so that one of the points stays in the data set
	x2 = np.delete(x, xduplicate)
	y2 = np.delete(y, xduplicate)
	
	return *names, x2, y2

#grouping data by x value then taking the mean 

def avgdata_data(filename, datatype, names, avg=False, fittype='Gaussian', guess=None):
	fitdata = data(filename, datatype, names, fittype)

	namex = data(filename, datatype, names)[0] 
	namey = data(filename, datatype, names)[1] #choosing x , y columns from .dat 
	x = data(filename, datatype, names)[2]
	y = data(filename, datatype, names)[3]

	data2 = pd.DataFrame({namex: x, namey: y}) 
	
	avgdatagroup = data2.groupby([namex])[namey].mean()
	
	avgdatagroup = avgdatagroup.reset_index()
	
	avgdatagroup = np.array(avgdatagroup)
	
# 	avgdata = avgdatagroup(namex)
	
	return namex, namey, avgdatagroup[:,0], avgdatagroup[:,1]		


#choosing data 

def choose_data(filename, datatype, names, avg=False, fittype='Null', guess=None):
	if avg is True:
		fitdata = avgdata_data(filename, datatype, names)
	else:
		if datatype == 'raw':
			fitdata = data(filename, datatype, names)
		elif datatype == 'exclude':
			fitdata = data_exclude(filename, datatype, names)
		elif datatype == 'exclude multiple points':
			fitdata = data_exclude_points(filename, datatype, names)
		else:
			fitdata = 'nothing'
			
	return fitdata 