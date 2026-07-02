import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

save_flag = False

x = np.array([19.5,12.2,8.2,5.4,3.3,2.9,2.36,2.3])
# x = np.array([2.3,2.36,2.9,3.3,5.4,8.2,12.2,19.5])

interpolation = np.interp(np.arange(0, len(x)), np.arange(0, len(x)), x)

def expfun(x, a, b, c):
    return a * np.exp(-b * x) + c

popt, pcov = curve_fit(expfun, np.arange(0, len(x)), x, p0=(20, 0.3, 1)) 

fig, ax = plt.subplots()
ax.plot(x, '.')
ax.plot(expfun(np.arange(0, len(x)), *popt), '-', color='lightblue',
        label = 'current starting evap value')

ax.set( 
    xlabel='Index',
    ylabel = 'Chip Evap Freq Value'
)

endpoint = 4
amplitude = 30

def freq_list(amp):
    return expfun(np.arange(0, len(x)), amp, popt[1], endpoint)

ax.plot(freq_list(amplitude), '.-', color='pink',
        label = 'increased starting evap value')

ax.legend()

if save_flag:
        
    folder = '\\\\UNOBTAINIUM\\AnalysisDrive\\LocalCode\\chip_evap_lists'
    np.savetxt(folder + '\\chip_evap_list_amplitude_starts_at_' + str(freq_list(amplitude)[0]) + 'MHz.txt', expfun(np.arange(0, len(x)), amplitude, popt[1], endpoint), fmt='%.2f')

print(f'Frequency list of scaled amplitude {amplitude}: {freq_list(amplitude)}')