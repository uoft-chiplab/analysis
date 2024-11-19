# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:11:38 2024

@author: coldatoms
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from library import colors
from cycler import Cycler

pi = np.pi

def Sincw(w, t): # used Mathematica to get an exact FT
	return 0.797885 * np.sin(t/2*w)/w
def Sincf(f,t): # kHz
	return Sincw(2*pi*f, t)
def Sinc2D(Delta, t, EF): # takes dimensionless detuning
	return Sincw(2*pi*Delta*EF/1000, t)**2

def sinc2(x, trf):
	"""sinc^2 normalized to sinc^2(0) = 1"""
	t = x*trf
	return np.piecewise(t, [t==0, t!=0], [lambda t: 1, 
					   lambda t: (np.sin(np.pi*t)/(np.pi*t))**2])

trf = 10
EF = 20
x = np.linspace(-10, 10, 1000)
plt.figure()
plt.plot(x, sinc2(x*EF/1000, 10), '-')
plt.plot(x, Sinc2D(x, 10, EF)/Sinc2D(0.0001*x, 10, EF), '--')

plt.plot(x, Sincf(x*EF/1000, 10)**2/Sincf(0.0001*x*EF/1000, 10)**2, ':')
plt.xlabel("frequency EF")
plt.show()

# use **styles[i] in errorbar input to cycle through
styles = Cycler([{'color':color, 'marker':''} for color in colors])

# load lineshape
df_ls = pd.read_pickle('./clockshift/convolutions_EFs.pkl')

trf = 10

df_trf = df_ls.loc[df_ls.TRF == trf]

A = 1
x0 = 0
params = [A, x0]

xs = np.linspace(-1, 0.5, 1000)


for EF in df_trf.EF.unique():
	fig, ax = plt.subplots(figsize=(6,4))
	ax.set(xlabel=r'$\hbar f/E_F$', ylabel='Weight')
	
	fig.suptitle(f"EF = {EF} kHz")
	
	df = df_trf.loc[df_trf.EF == EF]
	
	
	for TTF, lineshape, sty in zip(df.TTF.unique(), df.LS.unique(), styles):
		
		# turn convolution lineshape into function
		def convls(x, A, x0):
			return A*lineshape(x-x0)
		
		conv_norm = max(convls(xs, *params))
		
		ax.plot(xs, convls(xs, *params)/conv_norm, label=f"{TTF} T/TF", linestyle='-', **sty)
		
		ax.plot(xs, sinc2(xs*EF/1000, 10), linestyle='--', **sty)
	fig.tight_layout()
	
	ax.legend()
	plt.show()