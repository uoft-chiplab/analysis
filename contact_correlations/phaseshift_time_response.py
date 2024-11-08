# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:24:20 2024

@author: Chip Lab
"""

from contact_correlations.UFG_analysis import BulkViscTrap
import numpy as np
import matplotlib.pyplot as plt
import pickle

# pickle
pickle_file = 'time_delay_BVTs.pkl'

load = True

# parameters
ToTFs = [#0.2, 
		 0.3, 0.4, 0.5, 0.6]
EFs = [#13e3, 
	   14e3, 15e3, 16e3, 18e3]
barnu = 377
num = 100

def contact_time_delay(phi, period):
	""" Computes the time delay of the contact response given the oscillation
		period and the phase shift.
	"""
	return phi/(2*np.pi) * period


# load pickle
if load == True:
	with open(pickle_file, 'rb') as f:
	    BVTs = pickle.load(f)
	
	analysis = False
		
else: 
	# compute BVT and save to pickle
	BVTs = []
	analysis = True

# analysis loop
for i in range(len(ToTFs)):
	ToTF = ToTFs[i]
	EF = EFs[i]
	
	if analysis == False:
		break
	
	T = ToTF*EF
	nus = T*np.logspace(-2, 1, num)
	# compute trap averaged quantities using Tilman's code
	BVT = BulkViscTrap(ToTF, EF, barnu, nus)
	
	# compute time delays
	BVT.time_delay = contact_time_delay(BVT.phaseshiftsQcrit, 1/BVT.nus)
	
	BVTs.append(BVT)
	
with open(pickle_file, 'wb') as f:
    pickle.dump(BVTs, f)


# plot phase shift and time delay of contact response
subfigs = 2

fig, axes = plt.subplots(1, subfigs, figsize=(5*subfigs, 4))
axs = axes.flatten()

# phase shift
ax = axs[0]
ax.set(xlabel=r"Frequency, $h\nu/k_BT$", 
	   ylabel=r"Phase Shift, $\phi$ [rad]", xscale='log')

# time delay
ax = axs[1]
ax.set(xlabel=r"Frequency, $h\nu/k_BT$", 
	   ylabel=r"Response Delay, $t_{delay}$ [us]", xscale='log')


for BVT in BVTs:
	if BVTs[0]:

		pickletime = False
		# compute BVT and save to pickle
		BVTs = []
		analysis = True
		pickle_file = 'time_delay_ToTF_0p2.pkl'
	# analysis loop
		EFs = [15,16,17,18]
		if pickletime == True:
			for i in range(len(EFs)):
				ToTF = 0.2
				EF = EFs[i]
				
				if analysis == False:
					break
				
				T = ToTF*EF
				nus = T*np.logspace(-2, 1, num)
				# compute trap averaged quantities using Tilman's code
				BVT1 = BulkViscTrap(ToTF, EF, barnu, nus)
				
				# compute time delays
				BVT1.time_delay = contact_time_delay(BVT1.phaseshiftsQcrit, 1/BVT1.nus)
				
				BVTs.append(BVT1)
			
			with open(pickle_file, 'wb') as f:
			    pickle.dump(BVTs, f)
		else:
			with open(pickle_file, 'rb') as f:
			    BVTs = pickle.load(f)
			
		label = r"T={:.0f} kHz".format(BVT.T)
		label2 = f'ToTF={BVT.ToTF}, EF={(BVT.T/BVT.ToTF)/10e2:.0f} kHz'
		axs[1].plot(BVT.nus/BVT.T, BVT.time_delay*1e6, '-', label=label)
		axs[0].plot(BVT.nus/BVT.T, BVT.phaseshiftsQcrit, '-', label=label2)
		axs[0].legend()
	
	else:	
		break
# 		label = r"T={:.0f} kHz".format(BVT.T)
# 		label2 = f'ToTF={BVT.ToTF}, EF={(BVT.T/BVT.ToTF)/10e2:.0f} kHz'
# 		axs[1].plot(BVT.nus/BVT.T, BVT.time_delay*1e6, '-', label=label)
# 		axs[0].plot(BVT.nus/BVT.T, BVT.phaseshiftsQcrit, '-', label=label2)
# 		axs[0].legend()
# 		axs[1].legend()

plt.tight_layout()
