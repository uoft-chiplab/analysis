# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:26:57 2024

@author: coldatoms
"""

import re
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from library import FreqMHz
from data_class import Data
import matplotlib.colors as mc
import colorsys

summary = pd.read_excel('./contact_correlations/phaseshift/phaseshift_summary.xlsx')
summary = summary.dropna(subset=['ps', 'e_ps', 'Amp'])

fig, ax = plt.subplots()
ax
ax.errorbar(summary['freq'], summary['ps'], yerr=summary['e_ps'])
ax.set(ylim=[0, 1], xlabel='Drive freq [kHz]', ylabel = 'Phase shift [rad]')

# ax=axs[1]
# ax.plot(summary['freq'], summary['Amp'])