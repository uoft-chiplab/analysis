import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def saturation_fit(t, N0, tau):
    return N0 * (1 - np.exp(-t / tau))


filename = '2026-02-26_Rb_noLN2.txt'
data = pd.read_csv(filename, delimiter='\t')

filename2 = '2026-02-26_Rb_withLN2.txt'
data2 = pd.read_csv(filename2, delimiter='\t')

filename3 = '2026-02-26_Rb_withLN2_nodisp.txt'
data3 = pd.read_csv(filename3, delimiter='\t')

filename4 = '2026-02-26_Rb_withLN2_1p5disp.txt'
data4 = pd.read_csv(filename4, delimiter='\t')

# data = data[data['Time (us)'] < 1000]
# data2 = data2[data2['Time (us)'] < 1000]
# data3 = data3[data3['Time (us)'] < 1000]
# data4 = data4[data4['Time (us)'] < 1000]


data.columns = ['Time (us)', 'N (arb)']
data['N (arb)'] = data['N (arb)'] / 1e8
data['Time (s)'] = data['Time (us)'] / 1e3

data2.columns = ['Time (us)', 'N (arb)']
data2['N (arb)'] = data2['N (arb)'] / 1e8

data3.columns = ['Time (us)', 'N (arb)']
data3['N (arb)'] = data3['N (arb)'] / 1e8
data3['Time (s)'] = data3['Time (us)'] / 1e3

data4.columns = ['Time (us)', 'N (arb)']
data4['N (arb)'] = data4['N (arb)'] / 1e8
data4['Time (s)'] = data4['Time (us)'] / 1e3

p0 = [max(data['N (arb)']), 1]  # Initial guess for N0 and tau
popt, pcov = curve_fit(saturation_fit, data['Time (s)'], data['N (arb)'], p0=p0)
xx = np.linspace(0, max(data['Time (s)']), 100)
yy = saturation_fit(xx, *popt)

p02 = [max(data2['N (arb)']), 1]  # Initial guess for N0 and tau
popt2, pcov2 = curve_fit(saturation_fit, data['Time (s)'], data2['N (arb)'], p0=p02)
yy2 = saturation_fit(xx, *popt2)

p03 = [max(data3['N (arb)']), 1]  # Initial guess for N0 and tau
popt3, pcov3 = curve_fit(saturation_fit, data3['Time (s)'], data3['N (arb)'], p0=p03)
xx3 = np.linspace(0, max(data3['Time (s)']), 100)
yy3 = saturation_fit(xx3, *popt3)

p04 = [max(data4['N (arb)']), 1]  # Initial guess for N0 and tau
popt4, pcov4 = curve_fit(saturation_fit, data4['Time (s)'], data4['N (arb)'], p0=p04)
xx4 = np.linspace(0, max(data4['Time (s)']), 100)
yy4 = saturation_fit(xx4, *popt4)

fig, ax = plt.subplots()
ax.plot(data['Time (s)'], data['N (arb)'], 'o', color='coral')
ax.plot(xx, yy, color='darkorange', label='Fit: N0={:.2f}, tau={:.2f}, w/o ln2'.format(*popt))
ax.set(xlabel = 'Time (s)', ylabel='N (arb) [10^8]', title='Rb MOT Loading Curve')

ax.plot(data['Time (s)'], data2['N (arb)'], 'o', color='pink')
ax.plot(xx, yy2, color='darkmagenta', label='Fit: N0={:.2f}, tau={:.2f}, w/ 3.5 V disp ln2'.format(*popt2))

ax.plot(data3['Time (s)'], data3['N (arb)'], 'o', color='deepskyblue')
ax.plot(xx3, yy3, color='royalblue', label='Fit: N0={:.2f}, tau={:.2f}, w/ ln2 no disp'.format(*popt3))

ax.plot(data4['Time (s)'], data4['N (arb)'], 'o', color='palegreen')
ax.plot(xx4, yy4, color='darkgreen', label='Fit: N0={:.2f}, tau={:.2f}, w/ ln2 1.5 V disp'.format(*popt4))

ax.legend()
