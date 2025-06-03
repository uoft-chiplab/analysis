from HFT_dimer_bg_analysis import getDataFrame
import matplotlib.pyplot as plt
import numpy as np

df = getDataFrame()
# print(df)

# maxC = df['C'].max()

def fitfunc(x, A):
    xstar = df['x_star'][11]
    xmax = xstar
    return A*x**(-3/2) / (1+x/xmax)

f0 = 100  # MHz
A_at100 = df['raw_transfer_fraction'][11]

# Frequency range for the fit
freqs = np.linspace(10, 200, num=200)  

fig, ax = plt.subplots(1)

ax.plot(f0, A_at100, 'ro')
ax.plot(fitfunc(freqs, 1000), linestyle='-')

