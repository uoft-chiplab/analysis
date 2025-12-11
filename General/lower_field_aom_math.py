"""This script does some AOM frequency math to help determine how to implement a "lower field" imaging setpoint."""
# Dec 2025

import numpy as np
import matplotlib.pyplot as plt

mu = -1.4 # diferential magnetic moment b/w |9/2,-9/2> and |11/2, -11/2> (HF imaging cycling transition) in MHz/G

def field_to_AOM_freq(Bfield, ref_field=209, ref_AOM_freq=-154):
    """ Given a reference field and AOM frequency, compute the AOM frequency
        needed to image at a different field.
        
        Bfield : desired magnetic field in Gauss
        ref_field : reference magnetic field in Gauss
        ref_AOM_freq : reference AOM frequency in MHz
    """
    delta_B = Bfield - ref_field
    delta_freq = mu * delta_B # in MHz
    new_AOM_freq = ref_AOM_freq + delta_freq/2# 1/2 b/c double pass AOM
    return new_AOM_freq

Bfields = np.linspace(140, 235, 20) # G
AOM_freqs = field_to_AOM_freq(Bfields)

special_fields = [209, 202.1, 180, 235, 160]
special_AOM_freqs = [field_to_AOM_freq(f) for f in special_fields]
labels = ['Nominal HF','Unitarity', 'Lower limit of HF stabilization', 'Proposed higher field', 'Proposed lower field']

fig, ax = plt.subplots()
ax.plot(Bfields, AOM_freqs, '-')
for B, f, label in zip(special_fields, special_AOM_freqs, labels):
    ax.plot(B, f, 'o', label=label)
ax.hlines(-170, Bfields.min(), Bfields[3], color='black', ls='--', label='AOM Center Freq')
ax.vlines(140, AOM_freqs.min(), AOM_freqs.max(), color='green', ls='--', label='gg p-wave resonance')
ax.vlines(224, AOM_freqs.min(), AOM_freqs.max(), color='green', ls='--', label='ac s-wave resonance')
ax.set(xlabel='Magnetic Field (G)',
       ylabel='HF AOM Freq double-pass (MHz)')

ax.legend()

switchAOM_freq = 80 # MHz
ax2 = ax.twinx()
ymin, ymax = ax.get_ylim()
ax2.set_ylim(ymin - switchAOM_freq/2, ymax - switchAOM_freq/2)
ax2.set_ylabel(f'w/ +{switchAOM_freq} MHz single-pass shift (MHz)')