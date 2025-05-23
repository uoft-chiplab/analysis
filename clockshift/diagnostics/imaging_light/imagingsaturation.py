# paths
import os

import sys

#analysis_path = 'E:\\Analysis Scripts\\analysis'
analysis_path = '\\\\UNOBTAINIUM\\E_Carmen_Santiago\\Analysis Scripts\\analysis'
if analysis_path not in sys.path:
    sys.path.append(analysis_path)
data_path = analysis_path + '\\clockshift\\data\\imaging_saturation'
from library import pi, h, hbar, mK, a0, plt_settings, styles, colors
from data_helper import remove_indices_formatter
from save_df_to_xlsx import save_df_row_to_xlsx
from data_class import Data
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
from clockshift.MonteCarloSpectraIntegration import MonteCarlo_estimate_std_from_function
from contact_correlations.UFG_analysis import calc_contact
from contact_correlations.contact_interpolation import contact_interpolation as C_interp
from scipy.optimize import curve_fit
from warnings import catch_warnings, simplefilter
from cycler import Cycler, cycler
from scipy.ndimage import gaussian_filter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

def saturation_func(x, A, xsat):
    return A/(1+x/xsat)  

files = ['2025-01-24_D_e.dat']


run = Data(files[0], path=data_path)

df = run.data
xname = df['refmean']
yname = df['ODmean']
popt, pcov = curve_fit(saturation_func, xname, yname, p0=[0.02, 9000])
perr = np.sqrt(np.diag(pcov))
xs = np.linspace(xname.min(), xname.max(), 100)
ys= saturation_func(xs, *popt)
fig, ax = plt.subplots()
ax.plot(df['refmean'], df['ODmean'])
ax.plot(xs, ys, '--', marker='', label=f'Isat={popt[1]:.1f}({perr[1]:.1f})')
ax.set(xlabel='Mean Ref Counts [px]',
       ylabel='meanOD')
ax.legend()

print(f'Fit parameters: {popt[0]:.3f} +/- {perr[0]:.3f}, {popt[1]:.2f} +/- {perr[1]:.2f}')