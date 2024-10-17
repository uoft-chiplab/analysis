# -*- coding: utf-8 -*-
"""
@author: Chip Lab
"""

import ast
import itertools
import numpy as np

def check_for_col_name(df, col_name, alternates=[]):
	""" Checks for a column name in a dataframe. If the name does not exist, 
		checks for names in list alternates. If no name there exists either, 
		throws an error. """
	# check if col_name is in df
	try: 
		df[col_name]
		return None # if it works, return
	except KeyError: # if not, try alternates
		for alt_name in alternates:
			try: 
				df[col_name] = df[alt_name]
				return None # if it works, return
			except KeyError: # if it doesn't work, continue
				continue
		# if nothing works, raise error 
		raise KeyError("No {name} column in .dat, nor any of {alternates}")

def remove_indices_formatter(remove_indices):
	if remove_indices == remove_indices: # nan check
		if type(remove_indices) == int:	
			return remove_indices # already formatted
		elif type(remove_indices) == str: 
			if remove_indices.count('[') == 1: # convert str to list of integers
				remove_list = ast.literal_eval(remove_indices)
			else: # multiple lists, we'll use those as ranges
				remove_ranges = ast.literal_eval(remove_indices)
				remove_list = list(itertools.chain(*[range(val[0], val[1]+1) \
									for val in remove_ranges]))
		else: 
			raise ValueError("remove_indices is not formatted correctly: {remove_indices}")
		return remove_list
	else: # no removal
		return None
	
def bg_freq_formatter(bg_freq):
	""" Formats a metadata file cell entry in one of three ways:
		- if integer, pass it back along with 'int'
		- if list, pass it back as a list along with 'list'
		- if list of lists, pass it back along with 'ranges'"""
	if type(bg_freq) == str: 
		# convert str to list of integers
		if bg_freq.count('[') == 1: # only just a list
			input_type = 'list'
		else: # multiple lists, we'll use those as ranges
			input_type = 'range'
		bg_return = ast.literal_eval(bg_freq)
	elif type(bg_freq) == int or type(bg_freq) == float or \
												type(bg_freq) == np.float64:
		input_type = 'single'
		bg_return = bg_freq
	else: 
		raise ValueError("bg_freq is not formatted correctly:" + bg_freq)
	return bg_return, input_type 


