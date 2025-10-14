# -*- coding: utf-8 -*-
"""
field_wiggle.py
2023-11-14
@author: Chip Lab

Analysis field wiggle scans where the frequency of transfer is 
varied, and the delay time is varied. Files are organized by
delay time due to how MatLab outputs the .dat files.
"""
import sys
module_folder = 'E:\\Analysis Scripts\\analysis'
if module_folder not in sys.path:
	sys.path.insert(0, 'E:\\Analysis Scripts\\analysis')
from data_class import Data
from fit_functions import Sinc2, Lorentzian, Gaussian, \
							Parabola, Sin, FixedSin
from scipy.optimize import curve_fit
from library import *
from tabulate import tabulate

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os

data_folder = os.getcwd()

# run metadata
run = os.path.split(data_folder)[-1] #'yyyy-mm-dd_A
wiggle_freq = 6 # kHz
wiggle_amp = 1.8 # Vpp
pulse_length = 10 # us
note = ''

# parameters relevent for analysis
x_name = "freq (MHz)"
y_name = "fraction95"
fit_func = Sinc2
num = 500
save_final_plot = True

# Fixed sinusoidal function depending on given wiggle freq
def FixedSinkHz(t, A, p, C):
	omega = wiggle_freq/1000.0 * 2 * np.pi # kHz
	return A*np.sin(omega*t - p) + C

# initalize no guesses, but fill them in if needed
guess = [0.035, 2, 202.16]

times_weird = [ 0.4, 0.48,  0.56, 0.57]
times_notweird = [0.34, 0.36, 0.38, 0.42, 0.44, 0.46, 0.5, 0.52,0.54]

time_list1 = times_weird + times_notweird #np.array([0.34, 0.36, 0.38, 0.42, 0.44, 0.46, 0.5, 0.52, 0.4, 0.48, 0.54, 0.56, 0.57])
# time_list2 = np.array([0.4, 0.48, 0.54, 0.56, 0.57])
letter_list = ['D', 'E', 'F']
fn_list = [
    f"2025-09-16_{letter}_e_wiggle time pre={time:.2f}.dat"
    for time in time_list1
    for letter in letter_list
    if os.path.exists(f"2025-09-16_{letter}_e_wiggle time pre={time:.2f}.dat")
]

# fn_list = [f"2025-09-16_{letter}_e_wiggle time pre={time:.2f}.dat" for time, letter in zip(time_list1, letter_list)]
# fn_list2 = [f"2025-09-16_{letter}_e_wiggle time pre={time:.2f}.dat" for time, letter in time_list1]
# fn_list3 = [f"2025-09-16_{letter}_e_wiggle time pre={time:.2f}.dat" for time in time_list2]

# time_list = np.concat([time_list1, time_list2])
# time_list = time_list*1000 + pulse_length/2.0

file_list = fn_list 

data_df = pd.DataFrame({})

for file in file_list:
	data = Data(file, path=data_folder)
	data_df = pd.concat([data_df, data.data], ignore_index=True)
	
popt_list = []
perr_list = []
B_list = []
e_B_list = []

data_df.time = data_df['wiggle time pre']*1000 + pulse_length/2.0

### Fit 97 transfer scans as function of frequency
for i, time in enumerate(data_df.time.unique()):
	
	this_df = data_df[data_df.time == time]
	
    # This is dumb, we're making a Data class using the last file, but just overwriting the data
	data = Data(file, path=data_folder)
	data.data = this_df
	data.filename = f'delay time = {time} us'
	
	# fit fraction95 vs. freq with Sinc2
	data.fit(fit_func, names = [x_name, y_name])
	popt_list.append(data.popt)
	perr_list.append(data.perr)
	
	# convert peak freq to B field
	data.B = B_from_FreqMHz(data.popt[1])
	data.Berr = np.abs(data.B - B_from_FreqMHz(data.popt[1] + data.perr[1]))
	B_list.append(data.B)
	e_B_list.append(data.Berr)
	
# create summary dataframe of analysis and export
data_dict = {'time':np.array(data_df.time.unique()), 'popt':popt_list, 'perr':perr_list, 'B':B_list, 'e_B':e_B_list}	
df = pd.DataFrame.from_dict(data_dict)

# Extract elements of popt and perr lists into their own columns
split_df1 = pd.DataFrame(df['popt'].tolist(), columns=['A','x0','sigma','C'])
split_df2 = pd.DataFrame(df['perr'].tolist(), columns = ['e_A','e_x0','e_sigma','e_C'])
df = pd.concat([df, split_df1], axis=1)
df = pd.concat([df, split_df2], axis=1)
df.drop(['popt','perr'],axis=1, inplace=True)
df.to_csv(run+'_analysis_summary.csv')

print('fitting B vs. time....')
# fit B vs. time
func=FixedSinkHz
param_bounds = [[0, 0, -np.inf],[np.inf, 2*np.pi, np.inf]]
popt, pcov = curve_fit(func, df['time'], df['B'], 
					   sigma=df['e_B'], 
					   bounds=param_bounds, p0=guess)
perr = np.sqrt(np.diag(pcov))

fit_params=['Amplitude','Phase','Offset']
parameter_table = tabulate([['Values', *popt], 
								 ['Errors', *perr]], 
								 headers=fit_params)
print("Field calibration:")
print("")
print(parameter_table)
	
# stuff result into summary csv file
fcal_dict = {'run':run, 'wiggle_freq':wiggle_freq, 'wiggle_amp':wiggle_amp, 'B_amp':popt[0], 'B_phase':popt[1], 'B_offset':popt[2],\
			 'e_B_amp':perr[0], 'e_B_phase':perr[1], 'e_B_offset':perr[2], 'note':note}
fcal_df = pd.DataFrame.from_dict([fcal_dict])

summ_path = 'E:\\Analysis Scripts\\analysis\\contact_correlations\\FieldWiggleCal\\field_cal_summary.csv'
summ_df = pd.read_csv(summ_path)

if summ_df['run'].str.contains(run).any():
	idx = summ_df.index[summ_df['run']==run]
	summ_df.drop(idx,inplace=True)
	update_df = pd.concat([summ_df, fcal_df])
else :
	update_df = pd.concat([summ_df, fcal_df])

pd.DataFrame.to_csv(update_df, summ_path, index=False)

# plot B vs. time
plt.figure()
ax = plt.subplot()
# ax.set(ylim=[202.11, 202.2])
ax.yaxis.get_major_formatter().set_useOffset(False)
xx = np.linspace(np.min(df['time']),
				   np.max(df['time']), num)
ax.plot(xx, func(xx, *popt), "--", label=f'fit [{popt[0]:.3f}({perr[0]*1e3:.0f}), {popt[1]:.2f}({perr[1]*1e2:.0f}), {popt[2]:.3f}({perr[2]*1e3:.0f})]')
ax.plot(xx, func(xx, *[popt[0], popt[1]+0.8, popt[2]]), "-.", color='orange', label=f'exp response (0.8 rad delay)')
ax.errorbar(df['time'],df['B'], 
			 yerr=df['e_B'], fmt='go')
ax.set(title=run[0:10]+ ' ' + str(wiggle_freq) + " kHz, " + str(wiggle_amp) + " Vpp field wiggle cal")
ax.set(xlabel='Time [us]', ylabel='Field [G]')
if save_final_plot:
	plt.savefig(run + ' ' + str(wiggle_freq) + ' kHz, ' + str(wiggle_amp) + ' Vpp field wiggle cal.png')
	
# ax.legend()
plt.show()