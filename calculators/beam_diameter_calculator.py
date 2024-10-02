# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:17:17 2024

@author: coldatoms
"""
import numpy as np
pi = np.pi


def beam_diameter(f, MFD, lamda):
	return 4 * lamda * f /pi/MFD

