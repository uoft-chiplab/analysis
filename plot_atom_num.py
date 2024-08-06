# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:56:29 2024

Plots atom number and EF from Ushots across specified date range

@author: Mendelevium
"""
from datetime import datetime
import glob
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib as mpl


clist = ["hotpink", "cornflowerblue", "yellowgreen"]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=clist) 

# filter for ushots data
july_ushots = glob.glob(r'Z:\Data\2024\07 July2024\*\*UShots\*UHfit.dat')
june_ushots = glob.glob(r'Z:\Data\2024\06 June2024\*\*UShots\*UHfit.dat')

ushots = june_ushots + july_ushots
# get list of days
dates = np.unique([fpath.split('\\')[-1].split('_')[0] for fpath in ushots])

# list of attributes to save
attributes = ['ToTF', 'N', 'EFkHz']
eattributes = ['e'+x for x in attributes]

# initialize dataframe and columns
data = pd.DataFrame()
data['date'] = [0]*len(dates)
for x in attributes:
	data[x] = [0.0]*len(dates) # mean
for x in eattributes:
	data[x] = [0.0]*len(dates) # std

# iterate through all ushot files, save based on date taken
for i in range(len(dates)):
	date_str = dates[i]
	vals = []
	# first get list of all ushots taken on same day
	for fpath in ushots:
		if date_str in fpath:
			fit_dat = pd.read_csv(fpath)
			vals.extend(fit_dat[attributes].values)
	# calculate mean, std, and save to df
	vals = np.array(vals)
	data.loc[i, 'date'] = datetime.strptime(date_str, "%Y-%m-%d")
	data.loc[i, attributes] = vals.mean(axis=0) # mean
	data.loc[i,  eattributes] = vals.std(axis=0) # std

fig, ax1 = plt.subplots()

ax1.set_xlabel('date')
ax1.set_ylabel('EF (kHz)', color=clist[0])
ax1.scatter(data['date'], data['EFkHz'], color=clist[0])
ax1.tick_params(axis='y', labelcolor=clist[0])

ax2 = ax1.twinx()

ax2.set_ylabel('N', color=clist[1])
ax2.scatter(data['date'], data['N'], color=clist[1])
ax2.tick_params(axis='y', labelcolor=clist[1])

ax1.axvline(datetime.strptime("2024-07-03", "%Y-%m-%d"), color="lightgrey", 
 			linestyle="--", label="July 3, 2024")

ax1.legend()

fig.tight_layout()
plt.show()
