# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:01:18 2024

@author: coldatoms
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from data_class import Data
from library import FreqMHz, hbar, h, pi

def FermiEnergy(n, w):
	return hbar * w * (6 * n)**(1/3)

{wx, wy, wz}
mean_w = (wx*wy*wz)**(1/3)

ws = [int(w) for w in w_list]

run_metadata['mean_w'] = run.metadata['trapfreqs']
run_metadata['EF'] = FermiEnergy(run_metadata['TShotsN']/2,
								 run_metadata['mean_w']) 

run.data['Delta'] = run.data['detuning']/run_metadata['EF'].values[0]

c9_bg = mean(run.data['c9'].loc[run.data['Delta'] < run_metadata['EF'].values[0]])






