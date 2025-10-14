# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:19:19 2024

@author: coldatoms
"""
# paths
import os
import sys
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
if root not in sys.path:
	sys.path.insert(0, root)

from data_class import Data
from fit_functions import Sinc2
import matplotlib.pyplot as plt
import numpy as np 
from scipy import stats

plt.rcParams.update({"figure.figsize": [5,3.5]})

files = [
	"2025-08-12_I_e.dat",
	"2025-08-12_K_e.dat",
    "2025-08-12_R_e.dat",
		 "2025-08-12_Q_e.dat",
		 "2025-08-13_C_e.dat",
		 "2025-08-13_D_e.dat",
		 "2025-08-13_E_e.dat",
		 "2025-08-14_B_e.dat"
         ]

pulse_length_used = [
	200,
    1000,
	2000,
	1000,
    200,
	200,
	200,
	200
]

fit_func = Sinc2

upperField = 209
upperFB = 7.04852
upperVVA = 1.977
lowerField = 202.14
lowerFB = 3.24728
lowerVVA = 2.01108

def Field_to_FB(field):
	return (field-lowerField)*((upperFB-lowerFB)/(upperField-lowerField))+lowerFB
#to find FB value (V) to Gauss I am using the Breit Rabi
    #values of the FB for 209, 209.1, 209.2, 209.3, 209.4 
	#subtracting the respecitve 0.1G apart ones and avg'ing them 
FB_to_field_0p1G = ((7.147-7.091) + (7.202-7.147) + (7.258-7.202)
					+ (7.313-7.258))/4
#to find the field from the sigma found by fitting 
def FB_to_Field(FB):
	return np.abs(FB)/FB_to_field_0p1G*0.1
#similarily for field 
field_to_freq_0p1G = ((45.8989-45.8836) + (45.9142-45.8989) + (45.9294-45.9142)
					  + (45.9447-45.9294))/4
field_to_freq_0p1G_list = [(45.8989-45.8836), (45.9142-45.8989), 
						   (45.9294-45.9142), (45.9447-45.9294)]
field_to_freq_0p1G_sem = stats.sem(field_to_freq_0p1G_list)
def freq(fb):
	return FB_to_Field(np.abs(fb))/0.1*field_to_freq_0p1G
def freq_field(field):
    return np.abs(field)/0.1*field_to_freq_0p1G
#to find the field from the FB V value inputted into the sequencer
#I just inversed the field to fb eq'n from the mathemtica notebook 
def fb_to_field(FB):
	return (FB-lowerFB)/((upperFB-lowerFB)/(upperField-lowerField)) + lowerField

fields = []
fields_errors =[]
freqs = []
freqs_errors_list = []
pulse_lenghts = []
pulse_lenghts_errors = []
for file in files:
    run = Data(file)
    df = run.data 
    df['Field'] = fb_to_field(df['FB'])
    # names = ['FB', 'fraction95']
    # peak_guess = 7.05
    # guess = [0.5, peak_guess, 0.05, 0]
    # run.fit(fit_func, names, guess=guess)
    # print(f'The width of the peak is {FB_to_Field(run.popt[2]):.4f} G, which is {freq(run.popt[2])*1000:.4f} kHz')
    # print(f'Which is equivalent to a pulse length of {(1/freq(run.popt[2])):.4f} us')

    names = ['Field', 'fraction95']
    peak_guess = 209
    guess = [0.5, peak_guess, 0.05, 0]
    run.fit(fit_func, names, guess=guess)
    freqs_errors = np.sqrt((run.perr[2]/run.popt[2])**2 + (float(stats.sem(field_to_freq_0p1G_list))/field_to_freq_0p1G)**2)*(freq_field(run.popt[2])*1000)
    print(f'The width of the peak is {np.abs(run.popt[2]):.3f}({run.perr[2]:.3f}) G, which is {freq_field(run.popt[2])*1000:.4f} kHz')
    print(f'Which is equivalent to a pulse length of {(1/freq_field(run.popt[2])):.4f} us')
    print()
    fields.append(np.abs(run.popt[2]))
    fields_errors.append(run.perr[2])
    freqs.append(freq_field(run.popt[2])*1000)
    freqs_errors_list.append(freqs_errors)
    pulse_lenghts.append(1/freq_field(run.popt[2]))
    pulse_lenghts_errors.append(1/freqs_errors)
	
print('In Summary:')
for i, file in enumerate(files):
    print(f'For {file} the width of the sinc^2 was {fields[i]:.3f}({fields_errors[i]*1000:.0f})G, or {freqs[i]*1000:.1f}({freqs_errors_list[i]*10:.0f})kHz, which corresponds to a pulse length of {pulse_lenghts[i]:.0f}({pulse_lenghts_errors[i]:.0f})us compared to a pulse length of {pulse_length_used[i]} us used.')
	