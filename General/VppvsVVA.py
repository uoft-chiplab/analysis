# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:12:14 2024

@author: coldatoms
"""
from library import *
import matplotlib.pyplot as plt

file_path = os.path.join(current_dir, 'VVAtoVpp.txt')
square = os.path.join(current_dir, 'VVAtoVpp_square_43p2MHz.txt')

delimiter = '\t'  # Assuming columns are separated by spaces

### Initialize empty lists to store columns
BMVVA = []
BMVpp = []

sqVVA = []
sqVpp = []

### Open the file
try:
	with open(file_path, 'r') as file:
		next(file)
        # Read each line
		for line in file:
            # Strip whitespace and split by delimiter
			parts = line.strip().split(delimiter)
            
            # Assuming there are three columns separated by spaces
			if len(parts) == 2:
				
				col1, col2 = parts
				BMVVA.append(col1)
				BMVpp.append(col2)
			else:
				print(f"Ignoring line with unexpected format: {line}")
				
	with open(square, 'r') as file:
		next(file)
        # Read each line
		for line in file:
            # Strip whitespace and split by delimiter
			parts = line.strip().split(delimiter)
            
            # Assuming there are three columns separated by spaces
			if len(parts) == 2:
				
				col1, col2 = parts
				sqVVA.append(col1)
				sqVpp.append(col2)
			else:
				print(f"Ignoring line with unexpected format: {line}")
except FileNotFoundError:
    print(f"The file {file_path} does not exist.")
except IOError as e:
    print(f"Error reading the file: {e}")
	
BMVpp = [float(val) for val in BMVpp]
BMVVA = [float(val) for val in BMVVA]

sqVpp = [float(val) for val in sqVpp]
sqVVA = [float(val) for val in sqVVA]

fig, ax = plt.subplots()

ax.set_xlabel('VVA')
ax.set_ylabel('Vpp')
ax.plot(BMVVA,BMVpp,label='BM 47MHz')
ax.scatter(sqVVA,sqVpp,marker='d',color='orange',edgecolor='black',label='sq 43.2MHz')
ax.scatter([10],[3.6],color='green',marker='+',label='sq 47MHz')

ax.legend()