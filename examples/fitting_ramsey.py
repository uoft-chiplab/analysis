import os
os.chdir('../')

from data_class import Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

files = [
"2026-10-27_F_e.csv"
       ]

dfs = []

for file in files:
    data = Data(file)
    dfs.append(data.data)

def ramseyfit(x,A,T,f,phi,B):
    return A*np.exp(-x/T)*np.cos(2*np.pi*f*x - phi) + B

result = pd.concat(dfs, ignore_index=True)

dflessthan1ms = result[result['time (ms)'] < 1.5]
time = dflessthan1ms['time (ms)']
f95 = dflessthan1ms['fraction95']

guess = [0.5, 0.5, 4, 0, 0.5]
xs = np.linspace(min(time), max(time), 100)

popt, pcov = curve_fit(ramseyfit, time, f95, p0=guess)
perr = np.sqrt(np.diag(pcov))

fig, ax = plt.subplots(1, figsize=(8,4))

ax.plot(time, f95, color='darksalmon')
ax.plot(xs, ramseyfit(xs,*popt), linestyle='-',marker='',color='maroon', label=f"{popt[1]:.2f}$\pm${perr[1]:.2f} ms decay time")

ax.set_xlabel("wait time (ms)")
ax.set_ylabel("c5/c9")
ax.legend()

print(popt)