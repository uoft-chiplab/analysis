# paths
import os
import sys
import numpy as np
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
if root not in sys.path:
	sys.path.insert(0, root)
	
from fit_functions import ImageLightSaturation

import numpy as np
from data_class import Data
import matplotlib.pyplot as plt
file = "F_KAM_scan_imaging_sat_factor_e_data.csv"
data = Data(file, path="E:\\Data\\2025\\10 October2025\\14October2025\\F_KAM_scan_imaging_sat_factor")

# had inf OD
# data.data = data.data.drop(34)

# too low light, OD not reliable
data.data = data.data[data.data['K AM'] > 0.72]

names = ['roi.ref_mean', 'roi.OD_mean']
guess = [0.03, 4500]

data.fit(ImageLightSaturation, names, guess=guess)
