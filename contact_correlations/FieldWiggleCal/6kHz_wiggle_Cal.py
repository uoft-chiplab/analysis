import os
import sys

home = os.path.abspath(__file__ + 6 * '/..')
data_file = os.path.join(home, 'Data\\2025\\09 September2025\\16September2025\\D_6kHz_wiggle_calibration_10us_9to7_202p14G_1p8Vpp')
analysis_path = os.path.join(home, 'Analysis Scripts\\analysis')
if analysis_path not in sys.path:
	sys.path.insert(0, analysis_path)
from data_class import Data
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np
from scipy.optimize import curve_fit
import re

filename = '2025-09-16_D'

grabbing_times = glob.glob(os.path.join(data_file, f'{filename}_e_wiggle time pre=*.dat'))
times = [float(re.search(r'time=([0-9.]+)(?=\.dat)', f).group(1)) for f in grabbing_times]

filename = '2025-09-16_D'

names = ['freq','fraction95']
guess = [0.5, 43.248, 0.1, 0 ]

for time in times:
	df = Data(filename).fit(Sinc2, names, guess=guess)
