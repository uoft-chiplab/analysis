# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:18:06 2024

@author: coldatoms
"""

import pandas as pd

def save_df_row_to_xlsx(savedf, savefile, filename):
	'''Saves savedf row to an xlsx savefile, checking file for filename to overwrite.'''
	try: # to open save file, if it exists
		existing_data = pd.read_excel(savefile, sheet_name='Sheet1')
		if len(savedf.columns) == len(existing_data.columns) \
				and filename in existing_data['Run'].values:
			print()
			print(f'{filename} has already been analyzed and put into the summary .xlsx file')
			print('and columns of summary data are the same')
			print()
		elif len(savedf.columns) == len(existing_data.columns):
			print('Columns of summary data are the same')
			print("There is saved data, so adding rows to file.")
			start_row = existing_data.shape[0] + 1
		 
		 # open file and write new results
			with pd.ExcelWriter(savefile, mode='a', if_sheet_exists='overlay', \
					engine='openpyxl') as writer:
				savedf.to_excel(writer, index=False, header=False, 
				   sheet_name='Sheet1', startrow=start_row)
		else:
			print()
			print('Columns of summary data are different')  
			print("There is saved data, so adding rows to file.")
			start_row = existing_data.shape[0] + 1
			
			savedf.columns = savedf.columns.to_list()
		 # open file and write new results
			with pd.ExcelWriter(savefile, mode='a', if_sheet_exists='overlay', \
				   engine='openpyxl') as writer:
				savedf.to_excel(writer, index=False, 
				   sheet_name='Sheet1', startrow=start_row)
				
	except PermissionError:
		 print()
		 print ('Is the .xlsx file open?')
		 print()
	except FileNotFoundError: # there is no save file
		 print("Save file does not exist.")
		 print("Creating file and writing header")
		 savedf.to_excel(savefile, index=False, sheet_name='Sheet1')