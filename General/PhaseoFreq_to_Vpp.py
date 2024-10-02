# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:14:19 2024

@author: coldatoms
"""
from library import *
import numpy as np 
import matplotlib.pyplot as plt

textfile = "phaseofreq_to_Vpp_VVA_2p3_square.txt"
textfile_VVA10 = "phaseofreq_to_Vpp_VVA_10_square.txt"

file_path = os.path.join(current_dir, 'rfcalibrations')
filepath_textfile = os.path.join(file_path, textfile) # calibration file
filepath_textfile_VVA10 = os.path.join(file_path, textfile_VVA10) # calibration file
freqs, Vpps = np.loadtxt(filepath_textfile, unpack=True)
relVpps = Vpps/np.max(Vpps)
freqs_VVA10, Vpps_VVA10 = np.loadtxt(filepath_textfile_VVA10, unpack=True)
relVpps_VVA10 = Vpps_VVA10/np.max(Vpps_VVA10)

interpfcn = np.interp(freqs, freqs, relVpps)
interpfcn_VVA10 = np.interp(freqs_VVA10, freqs_VVA10, relVpps_VVA10)

fig, ax = plt.subplots()
ax.plot(freqs, interpfcn, linestyle = '-', color='red', mfc='red', label='VVA=2.3V')
ax.plot(freqs_VVA10, interpfcn_VVA10, linestyle = '-', color='blue', mfc='blue', label='VVA=10V')
ax.set(xlabel='Freq (MHz)', ylabel= 'relative amplitude on scope')
ax.legend()

# fig_VVA2p3, ax_VVA2p3 = plt.subplots()
# ax_VVA2p3.plot(freqs, relVpps)
# ax_VVA2p3.plot(freqs, interpfcn, linestyle = '-')
# ax_VVA2p3.set_xlabel('Freqeuncy (MHz)')
# ax_VVA2p3.set_ylabel('Scope Reading (mVpp)')
# ax_VVA2p3.set_title('PhaseO Scope vs Freq for VVA=2.3')

# fig_VVA10, ax_VVA10 = plt.subplots()
# ax_VVA10.plot(freqs_VVA10, relVpps_VVA10)
# ax_VVA10.plot(freqs_VVA10, interpfcn_VVA10, linestyle = '-')
# ax_VVA10.set_xlabel('Freqeuncy (MHz)')
# ax_VVA10.set_ylabel('Scope Reading (Vpp)')
# ax_VVA10.set_title('PhaseO Scope vs Freq for VVA=10')
