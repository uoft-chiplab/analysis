# -*- coding: utf-8 -*-
"""
2023-10-05
@author: Chip Lab

Data functions
"""

import os 
from glob import glob
from get_data import *

# importing data 

def data(filename, names=['freq','fraction95'], autofind=True):
	drive = os.getcwd()[0:3] #'E:\\'
	if autofind:
		file = glob(drive + '\\Data\\2023\\*\\*\\*\\' + filename)[0] # EXTREMELY greedy
	else :
		file = os.path.join(drive, "Data", "2023", "10 October2023", 
					 "05October2023", "Summary", filename) #making manual path for the filename
	data = data_from_dat(file, names) #making array of chosen data
	x = data[:,0]
	#x = [x+5 for x in x]
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

#exclude the repeated point at the end

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
	names = ["delaytime", "field"] #choosing x , y columns from .dat 
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
	
	return *names, x2, y2
	

