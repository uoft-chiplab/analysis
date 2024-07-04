# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 20:12:51 2024

@author: coldatoms
"""

from data_class import *
pi=np.pi
uatom = 1.66054e-27
mK = 39.96399848*uatom
ToTF = 0.6
kB =1.3806488e-23
EF = 16000 # Hz
T = ToTF*EF # Hz
h=6.62606957e-34
hbar = h/(2*pi)

def dimerlineshape(omega, Eb, TMHz, fudge=0, arb_scale=1):
	# everything in MHz
	print(TMHz)
	Gamma = (np.sqrt(-omega-Eb-fudge) * np.exp((omega+Eb+fudge)/TMHz)* np.heaviside(-omega - Eb-fudge, 1))* arb_scale
# 	Gamma = np.nan_to_num(Gamma)
	return Gamma

# data = Data('2023-10-02_H_e.dat', path='E:\\Data\\2023\\10 October2023\\02October2023\\H_202p1G_acdimer_1p8VVA320us', average_by='freq')
# data = Data('2023-10-02_I_e.dat', path='E:\\Data\\2023\\10 October2023\\02October2023\\I_202p1G_acdimer_1p5VVA640us', average_by='freq')
# data = Data('2024-06-12_R_e.dat', average_by='freq')
data = Data('2024-06-12_S_e.dat', average_by='freq')
# data =Data('2024-06-12_T_e.dat', average_by='freq')
# field=202.1
field = 202.14
df = data.avg_data
freq75 = 47.2159 # MHz, 202.1G
# freq75 = 47.2227 # MHz, 202.14 G
# freq75 = 47.1989 # Mhz, 202 G
df['detuning'] = df.freq - freq75
yfudge = 200
df['transfer'] = (-df.sum95 + np.max(df.sum95) - yfudge) / np.max(df.sum95)
df['em_transfer'] = df.em_sum95 / np.max(df.sum95)

fig, ax = plt.subplots()
ax.errorbar(df.detuning, df.transfer, yerr=df.em_transfer, marker='o', ls='',  capsize=2, mew=2, color='b')

xlow = -4.1
xhigh = -3.9
xx = np.linspace(xlow, xhigh, 500)
Ebpred= 4.04 #MHz but for 202.14 G...
# Ebpred = 4.01749 # MHz for 202.1 G using JT's Eq. (26)
# Ebpred = 4.027 # 202 G

# arbscale=0.0008
arbscale=5
yy = dimerlineshape(xx, Ebpred, T/10e6, arb_scale=arbscale)
ax.plot(xx, yy, '--',  color='r',label='Lineshape, Eb=-{:.2f} MHz @ {:.2f} G, T={:.1f} T_F, arb. scale'.format(Ebpred, field, ToTF))

ToTFfudge = 4
Tfudge = ToTFfudge*EF
Ebfudge = -0.065
# arbscale = 0.0028
arbscale=2
yyfudge = dimerlineshape(xx, Ebpred, Tfudge/10e6, fudge = Ebfudge, arb_scale=arbscale)
ax.plot(xx, yyfudge, '--',  color='g',label='Fudged Lineshape, Eb={:.2f} MHz, T={:.1f} T_F, arb. scale'.format(-Ebpred - Ebfudge, ToTFfudge))

ToTFfudge = 2
Tfudge = ToTFfudge*EF
Ebfudge = -0.065
# arbscale = 0.0028
arbscale=3
yyfudge = dimerlineshape(xx, Ebpred, Tfudge/10e6, fudge = Ebfudge, arb_scale=arbscale)
ax.plot(xx, yyfudge, '--',  color='y',label='Fudged Lineshape, Eb={:.2f} MHz, T={:.1f} T_F, arb. scale'.format(-Ebpred - Ebfudge, ToTFfudge))


ax.set_xlim([xlow, xhigh])
# ax.set_xlim([-4.07, -4.02])
ax.set_ylabel(r'$\Gamma$ [arb.]')
ax.set_xlabel(r'Detuning from 12-resonance [MHz]')
ax.set_title('2024-06-12 S ac dimer spectrum at 202.14G')
ax.legend()
plt.locator_params(nbins=10)
plt.tight_layout()