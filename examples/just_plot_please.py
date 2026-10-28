import matplotlib.pyplot as plt
import numpy as np

SSI_TOF = [1.97,
           2.35,
           3.45]

c5_sum = [12.5828,
          12.4932,
          13.0078]

e_c5_sum = [0.184579,
            0.146309,
            0.398395]

fig, ax = plt.subplots()
ax.errorbar(SSI_TOF, c5_sum, yerr=e_c5_sum, ls='', marker='o')
ax.set(xlabel='SSI TOF [ms]',
       ylabel='c5 box count sum [1e3]')