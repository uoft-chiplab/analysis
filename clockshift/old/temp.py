import os
import sys
# Get the current script's directory
analysis_dir = 'E:\\Analysis Scripts\\analysis'
# Add the parent directory to sys.path
if analysis_dir not in sys.path:
	sys.path.append(analysis_dir)
from library import pi, h, hbar, mK, a0, plt_settings, styles, colors
from data_class import Data
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

file = '2023-10-27_S_e.dat'
run =  Data(file, path=r'E:\Data\2023\10 October2023\27October2023\S_5kHzwiggle_0usdelay_eb_10uspulse10VVA')
run.group_by_mean('freq (MHz)')
fig, ax= plt.subplots()
ax.errorbar(run.avg_data['freq (MHz)'], run.avg_data['sum95'],
			run.avg_data['em_sum95'])

run.avg_data.to_excel('cmonman.xlsx')