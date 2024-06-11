# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:08:48 2024

@author: coldatoms
"""

from data_class import *

#pixis 2
plt.ylabel('G_ctr_x/mean(G_sigma_x)')

#pixis 1
# plt.ylabel('fCtr1/mean(fWidth1)')
plt.xlabel('time')

df = Data('2024-05-28_F_e.dat').data

x=df['time']
y=df['G_ctr_x']/np.mean(df['G_sigma_x'])
# plt.ylabel('G_ctr_y')
# y = df['G_ctr_y']



plt.plot(x,y,linestyle='',marker='.')

guess=[1,1,2,0,3.25,0]

#for I 2024-05-27
# guess=[10,30,1,0,112,0]

def TrapFreq(x, A, b, l, x0, C, D):
		return A*np.exp(-x/b)*(np.sin(l * x - x0)) +  C + D*x
	
popt, pcov = curve_fit(TrapFreq  ,x,y,p0=guess)


perr = np.sqrt(np.diag(pcov))

param_names = ['Amplitude','tau','omega','Center','Offset','Linear Slope']
		
parameter_table = tabulate([['Values', *popt], ['Errors', *perr]], 
								 headers=param_names)
print(parameter_table)
		
print(popt[2]*10**3/2/np.pi)
print(popt[2]*popt[1])

num = 500
xlist = np.linspace(x.min(), x.max(), num)

plt.plot(xlist, TrapFreq(xlist, *popt))
			