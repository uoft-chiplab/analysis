# -*- coding: utf-8 -*-
"""
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
from library import colors, tintshade, tint_shade_color, plt_settings

import numpy as np
import matplotlib.pyplot as plt

files = ["2024-09-10_E_e.dat",
 		 "2024-09-10_F_e.dat",
 		 "2024-09-10_H_e.dat"
		]
res_freqs = [47.2227, 45.8835, 47.2227, 48.3759]
labels = ['202.14G, 9/7 to 5, square',
		  '209G, 9 to 7, square',
		  '202.14G, 9/7 to 5, Blackman',
		  '209G, 9/7 to 5, square']
markers = ['s', 's', 'o', 's']

names = ['freq', 'fraction95']

sinc2 = Sinc2

### plot settings
plt.rcParams.update(plt_settings) # from library.py
plt.rcParams.update({"figure.figsize": [6,4],
					 "font.size": 14,
					 "lines.markeredgewidth": 2,
					 "errorbar.capsize": 0})
plt.figure()
plt.xlabel("Detuning (MHz)")
plt.ylabel("Transfer")
plt.xlim(-1,-0.1)
plt.ylim(-0.02,0.2)

for i, file in enumerate(files):
	light_color = tint_shade_color(colors[i], amount=1+tintshade)
	dark_color = tint_shade_color(colors[i], amount=1-tintshade)
	print("--------------------------")
	data = Data(file).data
	data['detuning'] = data[names[0]] - res_freqs[i]
	if i == 2:
		data['fraction95']
	plt.plot(data['detuning'], data[names[1]], '', marker=markers[i], 
			  markeredgecolor=dark_color, markerfacecolor=light_color, 
			  label=labels[i])
	
plt.legend()
	
	
