# -*- coding: utf-8 -*-
"""
@author: Chip Lab
"""

import numpy as np

### Vpp calibration
VVAtoVppfile = "VVAtoVpp.txt" # calibration file
VVAs, Vpps = np.loadtxt(VVAtoVppfile, unpack=True)
VpptoOmegaR = 27.5833 # kHz

def VVAtoVpp(VVA):
	"""Match VVAs to calibration file values. Will get mad if the VVA is not
		also the file. """
	for i, VVA_val in enumerate(VVAs):
		if VVA == VVA_val:
			Vpp = Vpps[i]
	return Vpp

### Datasets

# first real BM pulse 
F_20240621 = {'filename': "2024-06-21_F_e.dat", 
			  'xname': 'freq',
			  'ff': 1.03,
			  'trf': 400e-6,  # 200 or 400 us
			  'gain': 0.05,
			  'EF': 16e-3, #MHz
			  'bg_freq': 47,  # chosen freq for bg, large negative detuning
			  'res_freq': 47.2159, # for 202.1G
			  'pulsetype': 'Blackman',
			  'remove_indices': []}

# below this are all accidentally Kaiser pulses
D_20240620 = {'filename': "2024-06-20_D_e.dat", 
			  'xname': 'freq',
			  'ff': 1.03,
			  'trf': 200e-6,  # 200 or 400 us
			  'gain': 0.3,
			  'EF': 16e-3, #MHz
			  'bg_freq': 47,  # chosen freq for bg, large negative detuning
			  'res_freq': 47.2159, # for 202.1G
			  'pulsetype': 'Kaiser',
			  'remove_indices': []}

C_20240620 = {'filename': "2024-06-20_C_e.dat", 
			  'xname': 'freq',
			  'ff': 1.03,
			  'trf': 200e-6,  # 200 or 400 us
			  'gain': 0.05,
			  'EF': 16e-3, #MHz
			  'bg_freq': 47,  # chosen freq for bg, large negative detuning
			  'res_freq': 47.2159, # for 202.1G
			  'pulsetype': 'Kaiser',
			  'remove_indices': [32, 37, 77, 82, 122] # this is all 0 detuning points and another detuning...
			  }

G_20240618 = {'filename': "2024-06-18_G_e.dat", 
			  'xname': 'freq',
			  'ff': 1.03,
			  'trf': 200e-6,  # 200 or 400 us
			  'gain': 0.1,
			  'EF': 16e-3, #MHz
			  'bg_freq': 47,  # chosen freq for bg, large negative detuning
			  'res_freq': 47.2159, # for 202.1G
			  'pulsetype': 'Kaiser',
			  'remove_indices': []}

K_20240612 = {'filename': "2024-06-12_K_e.dat", 
			  'xname': 'freq',
			  'ff': 1.03,
			  'trf': 200e-6,  # 200 or 400 us
			  'gain': 0.2,
			  'EF': 16e-3, #MHz
			  'bg_freq': 47,  # chosen freq for bg, large negative detuning
			  'res_freq': 47.2159, # for 202.1G
			  'pulsetype': 'Kaiser',
			  'remove_indices': []}
