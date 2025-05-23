### Plotting a blackman pulse + FT 
#%%
import numpy as np
from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

# fs = 1_000_000  # Sampling rate = 1 MHz → 1 sample = 1 µs
# pulse_duration_us = 200  # Duration of pulse in microseconds
# pulse_samples = int(pulse_duration_us * fs / 1_000_000)  # Samples in pulse

window = signal.windows.blackman(100) #100 is essentially us

fig, ax = plt.subplots(2,2, figsize = (8,5))
ax = ax.flatten()
ax[0].plot(window)
ax[0].set(ylabel = 'Amplitude', 
          xlabel = 'Samples', 
        #   title = "Blackman window"
        )

#Fourier Transform the BM with the FT scipy fcn fft
A = fft(window, 2048) #/ (len(window)/2.0) #2048 bc fft likes powers of 2, and ex: if u zero-pad to 2048, get 1024 positive frequencies, making the spectrum smoother.
freq = np.linspace(-0.5, 0.5, len(A))
#fftshift shifts the zero-frequency component to the center of the spectrum 
response = np.abs(fftshift(A / abs(A).max()))

ax[1].plot(freq, response)
ax[1].set(ylabel = 'Amplitude', 
          xlabel = 'Normalized frequency [cycles per sample]', 
        #   title = "Frequency response of the Blackman window",
        #   xlim = [-0.5, 0.5], 
          )

response = np.log10(np.maximum(response, 1e-10))
ax[2].plot(freq, response)
ax[2].set(ylabel = 'Log Amp',
          xlabel = 'Freq',
          ylim = [ -6, 0],
          xlim = [-.2,.2]
          )

fig.tight_layout()

FTintlog = np.trapz(response)
# %%
