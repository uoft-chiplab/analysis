# -*- coding: utf-8 -*-
"""
@author: Chip Lab
"""

from contact_correlations.UFG_analysis import calc_contact
import matplotlib.pyplot as plt
import numpy as np

ToTF = 0.3
EF = 16e3
barnu = 306

contact = calc_contact(ToTF, EF, barnu)

print("Contact = ", contact)

ToTFs = np.linspace(0.2, 0.6, 17)
# EFs = np.array([10, 12, 14, 16, 18, 20])*1e3
EF = 14e3
barnus = np.array([200, 230, 260, 290, 320, 350, 380])

plt.figure()
plt.xlabel(r"$T/T_F$")
plt.ylabel(r"Contact, $C$ [$k_F/N$]")

results = []
for barnu in barnus:
	
	for ToTF in ToTFs:
		results.append(calc_contact(ToTF, EF, barnu))
	
	plt.plot(ToTFs, np.array(results)[:,0], label=barnu)
	
plt.legend()
plt.show()