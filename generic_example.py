# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:45:22 2024

This file contains generic examples for plotting and fitting using the Data class.

@author: coldatoms
"""
from data_class import *

# FB calibration
file = '2024-03-01_G_e.dat'
Data(file).fit(Sinc2, names= ['FB','fraction95'])

# VVA calibration
file = '2024-03-01_H_e.dat'
Data(file).fit(Parabola, names=['VVA','fraction95'])

# Trap freq

