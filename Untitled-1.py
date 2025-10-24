# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 14:19:19 2025

@author: coldatoms
"""
# paths
import os
import sys

home = os.path.abspath(__file__ + 6 * '/..')
analysis_path = os.path.join(home, 'Analysis Scripts\\analysis')
if analysis_path not in sys.path:
	sys.path.insert(0, analysis_path)
from data_class import Data
from scipy.optimize import curve_fit

def doublesinc2(x,A,x0,p,B,x1,q,C):
      return A*np.sinc((x-x0)/p)**2 + B*np.sinc((x-x1)/q)**2 + C

def sinc2(x, A, x0, p, C):
      return A*np.sinc((x-x0)/p)**2 + C

time_list = [0.8, 
             0.82, 
             0.84
             ]

for time in time_list:
    run = Data(f'2025-09-13_C_e_time={time}.dat')
    df = run.data

    bg = df[df['vva'] == 0]
    data = df[df['vva'] != 0 ]
    # data['N'] = data['c5'] - bg['c5'].mean() + data['c9']
    # data['transfer'] = (data['c5'] - bg['c5'].mean())/(data['N'])
    avgdata = df.groupby('freq').mean().reset_index()
    bg = avgdata[avgdata['vva'] != 4]
    avgdata = avgdata[avgdata['vva'] == 4]
    avgdata['N'] = avgdata['c5'] - bg['c5'].mean() + avgdata['c9']
    avgdata['transfer'] = (avgdata['c5'] - bg['c5'].mean())/(avgdata['N'])
    avgdata['loss'] = (bg['c9'].mean() - avgdata['c9'])/bg['c9'].mean()

    yname = 'transfer'
    xname = 'freq'

    #finding where the dips are to try and fit them generically 
    ymin1 = avgdata[avgdata[xname] < 43.2][yname].nsmallest(1).index[0]
    ymin2 = avgdata[avgdata[xname] > 43.2][yname].nsmallest(1).index[0]

    xmin1 = avgdata.loc[ymin1, xname]
    xmin2 = avgdata.loc[ymin2, xname]
    
    #amplitudes 
    amp = avgdata[yname].max() - avgdata[yname].min()
    #offset 
    offset = avgdata[yname].max()

    popt, pcov = curve_fit(doublesinc2, avgdata[xname], avgdata[yname], p0=[-amp,xmin1,0.1,-amp,xmin2,.1,offset])
    try:
         popt2, pcov2 = curve_fit(sinc2, avgdata[avgdata[xname] < 43.1][xname], avgdata[avgdata[xname] < 43.1][yname], p0=[-amp,xmin1,0.100, offset])
    except RuntimeError:
          xmin1 = 43.1
    xs = np.linspace(avgdata[xname].min(), avgdata[xname].max(), 100)
    xs2 = np.linspace(avgdata[avgdata[xname] < 43.2][xname].min(), avgdata[avgdata[xname] < 43.2][xname].max(), 100)
    
    fig, ax = plt.subplots()

    ax.plot(avgdata[xname], avgdata[yname])
    ax.plot(xs, doublesinc2(xs, *popt), marker='', linestyle='-', color='blueviolet')
    ax.plot(xs2, sinc2(xs2, *popt2), marker='', linestyle='-', color='mediumvioletred')
    ax.set(
          ylabel = yname,
          xlabel = 'Frequency (MHz)',
          title = f'{time*1000} us'
    )

