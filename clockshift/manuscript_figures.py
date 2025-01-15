# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:34:41 2025

These are a collection of select figures for drafting the clock shift manuscript.

@author: coldatoms
"""

import os
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

# Import CMasher to register colormaps
import cmasher as cmr
# Access rainforest colormap through CMasher or MPL
cmap = cmr.rainforest                   # CMasher
#cmap = plt.get_cmap('cmr.rainforest')   # MPL

Save = False
Talk = True

correct_spinloss = True
saturation_correction = True

### metadata
HFT_metadata_filename = 'HFT_metadata_file.xlsx'
HFT_metadata_file = os.path.join(proj_path, HFT_metadata_filename)
HFT_metadata = pd.read_excel(HFT_metadata_file)
# select files
transfer_selection = 'transfer'  # 'transfer' or  'loss'
exclude_name = 'exclude_transfer'
if transfer_selection == 'loss':
	exclude_name = 'exclude_loss'
HFT_files = ["2024-09-10_P_e"] # manual file select

### save file path

### Vpp calibration
VpptoOmegaR47 = 17.05/0.703 # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
VpptoOmegaR43 = 14.44/0.656 # kHz/Vpp - 2024-09-25 calibration 
phaseO_OmegaR = lambda VVA, freq: 2*pi*VpptoOmegaR47 * Vpp_from_VVAfreq(VVA, freq)


def spin_map(spin):
	if spin == 'c5':
		return 'b'
	elif spin == 'c9':
		return 'a'
	elif spin == 'sum95':
		return 'a+b'
	elif spin == 'ratio95' or spin == 'dimer':
		return 'a/b'
	else:
		return ''