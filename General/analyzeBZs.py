# paths
import os
import sys
import numpy as np
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
if root not in sys.path:
	sys.path.insert(0, root)

from data_class import Data
from fit_functions import Sinc2, Parabola, Sin, Linear
import matplotlib.pyplot as plt
from scipy.stats import sem

plt.rcParams.update({"figure.figsize": [5,3.5]})

file = "2026-08-10_B_e.csv"

df = Data(file).data

df['Bztotal'] = df['Bz0'] + df['Bz1L'] + df['Bz1R'] + df['Bz2L'] + df['Bz2R']

df['Bz0frac'] = df['Bz0']/df['Bztotal']
df['Bz1frac'] = (df['Bz1L'] + df['Bz1R'])/df['Bztotal']
df['Bz2frac'] = (df['Bz2L'] + df['Bz2R'])/df['Bztotal']

fig, ax = plt.subplots()
ax.errorbar(df['freq'], df['Bz0frac'], yerr=sem(df['Bz0frac']), fmt='o', label='Bz0')
ax.errorbar(df['freq'], df['Bz1frac'], yerr=sem(df['Bz1frac']), fmt='o', label='Bz1')
ax.errorbar(df['freq'], df['Bz2frac'], yerr=sem(df['Bz2frac']), fmt='o', label='Bz2')
ax.hlines(df['Bz0frac'].mean(), df['freq'].min(), df['freq'].max(), colors='C0', linestyles='dashed')
ax.hlines(df['Bz1frac'].mean(), df['freq'].min(), df['freq'].max(), colors='C1', linestyles='dashed')
ax.hlines(df['Bz2frac'].mean(), df['freq'].min(), df['freq'].max(), colors='C2', linestyles='dashed')
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('Fraction of Total Bz')
ax.legend()

