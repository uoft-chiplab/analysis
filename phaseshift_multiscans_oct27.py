#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Analyse wiggle phase shift measurements from Oct 27 2023, where binding energies and loss amplitude are fit for each time point in the wiggle.

"""

from data_class import Data
from library import *

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data_folder = 'data/oct27phaseshift'
summary_df = pd.read_csv(data_folder + '/Oct27summary.csv')
summary_df.head()

def wigglefield_from_time(t):
    omega=2*np.pi*5000
    amp=0.0695243 #+/- 0.0025798
    phase=2.17306 #+/- 0.0474719
    offset=202.072 #+/- 0.00203875
    return amp * np.sin(omega * t - phase) + offset

def Eb_from_field(B, whichfit='linear'):
    '''
    Eb cal from Oct 30
    Eb in MHz, field input in G
    '''
    if whichfit is 'linear':
        a=-0.05124 #+/- 0.006695
        b=14.35 #+/- 1.353
        return a*B + b
    
    
    

# binding energy fits
# Linear fit
#  ------------
# Fit parameters are -0.05124, 14.35
# with errors 0.006695, 1.353
# Fit chi^2 is 0.2588
# Parabolic fit
#  ------------
# Fit parameters are -0.05124, 14.35
# with errors 0.006695, 1.353
# Fit chi^2 is 0.203

# 5 kHz wiggle cal
#   	Amplitude  	phase    	offset
# ------  -----------  ---------  ------------
# Values	0.0695243  2.17306	202.072
# Errors	0.0025798  0.0474719	0.00203875

