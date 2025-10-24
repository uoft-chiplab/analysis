#Fitting Rabi Freq
import os 
import sys
general_path = os.path.dirname(os.path.realpath('data_class.py'))
root = os.path.dirname(general_path)
if root not in sys.path:
	sys.path.insert(0, root)
	
from data_class import Data
from fit_functions import *

filename = "2025-07-17_B_e.dat"

df = Data(filename)

df.fit(RabiFreq, names=['pulse time (ms)','c5'], guess = [20000, 12, 0.01,0.01]) 
# df.fit(RabiFreqDecay, names=['pulse time (ms)','c5'], guess = [20000, 12, 1,0.01,0.01]) 