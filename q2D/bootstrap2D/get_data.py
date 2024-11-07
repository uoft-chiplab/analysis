# -*- coding: utf-8 -*-
"""
July 13 2023
@author: Colin Dale
	The purpose of this code is to take the data from a 
	.dat file for use in Bootstrap fitting.
"""

import numpy as np
debug = False

def data_from_dat(datname, names_of_interest):
	"""
	filename : str
		name of .dat file
	names_of_interest : list of str
		list of column names in .dat to pull

	Returns
	-------
	data : np.array of floats
		transposed array of data
	"""
	datadat = np.loadtxt(datname, delimiter=',',dtype=str)
	data = []
	for name in names_of_interest:
		index = np.argwhere(datadat == name)[0][1]
		data.append(datadat[1:,index])
		
	# transpose data and convert to float
	data = np.array([np.array(i) for i in zip(*data)]).astype(np.float)
	return data

def from_csv(csv_name, run_name, column_name):
	"""
	csv_name : str
		name of .csv file
	run_name : str
		name of .dat file/run
	column_name : str
		name of column in .csv file to pull value from

	Returns
	-------
	value : str
		str value from csv
	"""
	datacsv = np.loadtxt(csv_name, delimiter=',',dtype=str)
	col_index = np.argwhere(datacsv == column_name)[0][1]
	row_index = np.argwhere(datacsv == run_name)[0][0]
	value = datacsv[row_index, col_index]
	return value


###### DEBUGGING ######

if debug == True:
	file = "dimers2d_2022-08-12_F_LAT2_40ER_198G_FM.dat"
	names = ["freq", "sum95"]
	
	print(data_from_dat(file, names))
	
	column_name = 'B'
	csv_name = '2Ddimer_parameters.csv'
	run_name = 'dimers2d_2022-08-16_E_LAT1_80ER_198p3G_FM'
		
	datacsv = np.loadtxt(csv_name, delimiter=',',dtype=str)
	col_index = np.argwhere(datacsv == column_name)[0][1]
	row_index = np.argwhere(datacsv == run_name)[0][0]
	Bfield = datacsv[row_index, col_index]
	print(Bfield)
