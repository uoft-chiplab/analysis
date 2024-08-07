# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:50:55 2024

@author: coldatoms
"""

import matplotlib.pyplot as plt 

Diamopen = [4,3.52,2.54,1.27,0.6]

Rbcounts = [5.11e7,2.225e7,2.01e7,1.685e5,0]
Rbx = [0.735,0.687,0.617,1.415,0]
Rby = [0.916,0.883,0.775,1.601,0]

KD1counts = [3.878e7,1.49e7,5.348e6,4.913e2,0]
KD1x = [0.718,0.639,0.63,0.022,0]
KD1y = [0.937,0.814,0.82,0.018,0]

KnoD1counts = [3.227e7,1.149e7,5.612e6,2.54e1,0]
KnoD1x = [0.856,0.683,0.712,0.002,0]
KnoD1y = [1.079,0.839,0.866,0.005,0]

fig, axcounts = plt.subplots()

axcounts.scatter(Diamopen, Rbcounts, label = 'Rb')
axcounts.scatter(Diamopen, KD1counts, label ='K with D1')
axcounts.scatter(Diamopen, KnoD1counts, label = 'K no D1')

axcounts.set_xlabel('Diameter of Iris (cm)')
axcounts.set_ylabel('Counts')

axcounts.set_ylim(0,1.8e5)

axcounts.legend()

fig, axx = plt.subplots()

axx.scatter(Diamopen, Rbx, label = 'Rb')
axx.scatter(Diamopen, KD1x, label ='K with D1')
axx.scatter(Diamopen, KnoD1x, label = 'K no D1')

axx.set_xlabel('Diameter of Iris (cm)')
axx.set_ylabel('Width of Cloud in x (mm)')

axx.legend()

fig, axy = plt.subplots()

axy.scatter(Diamopen, Rby, label = 'Rb')
axy.scatter(Diamopen, KD1y, label ='K with D1')
axy.scatter(Diamopen, KnoD1y, label = 'K no D1')

axy.set_xlabel('Diameter of Iris (cm)')
axy.set_ylabel('Width of Cloud in y (mm)')

axy.legend()