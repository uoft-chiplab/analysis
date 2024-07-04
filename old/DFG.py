# -*- coding: utf-8 -*-
"""
2023-11-02

@author: Chip Lab

DFG
"""
from data_class import Data
from library import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os

import mpmath as mp
from scipy.optimize import fsolve

T = 300 # (nK)
mean_trapfreq = 2*pi*(151.6*429*442)**(1/3)
omega = mean_trapfreq
mu_est = 900 # (nK)

# use ploylog from mp math, but make it a numpy ufunc so it can take arrays
# then turn back to float for use with numpy... sigh
polylog = lambda s, z: np.array(np.frompyfunc(mp.polylog, 2, 1)(s,z))

def deBroglie(T):
	return np.sqrt(2*pi*hbar**2/(mK*kB*T/1e9))

def n_peak(mu, T):
	return -1/deBroglie(T)**3 * polylog(3/2,-np.exp(mu/T))

def N_HO(mu, T, omega):
	return -(T/1e9*kB/hbar/omega)**3 * polylog(3, -np.exp(mu/T))

def mu_HO(N, T, omega, mu_est=mu_est):
	def solve_mu(mu, N, T, omega):
		return np.array(N_HO(mu, T, omega) - N)
	return fsolve(solve_mu, mu_est, args=(N, T, omega))

def E_HO(mu, T, omega):
	return -3*T*(T/1e9*kB/hbar/omega)**3 * polylog(4, -np.exp(-mu/T))


