# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 20:12:51 2024

@author: coldatoms
"""
import sys
data_class_dir = 'E:\\Analysis Scripts\\analysis'
if data_class_dir not in sys.path:
	sys.path.append(data_class_dir)
	print(sys.path)
from data_class import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import quad

pi=np.pi
uatom = 1.66054e-27
mK = 39.96399848*uatom
ToTF = 0.31
kB =1.3806488e-23
EF = 16000 # Hz
T = ToTF*EF # Hz
h=6.62606957e-34
hbar = h/(2*pi)

VVAtoVppfile = data_class_dir + "\\VVAtoVpp.txt" # calibration file
VVAs, Vpps = np.loadtxt(VVAtoVppfile, unpack=True)
VpptoOmegaR = 27.5833 # kHz
def VVAtoVpp(VVA):
 	"""Match VVAs to calibration file values. Will get mad if the VVA is not
		also the file. """
 	for i, VVA_val in enumerate(VVAs):
		 if VVA == VVA_val:
 			Vpp = Vpps[i]
 	return Vpp

Vpp = VVAtoVpp(1.4)
ff=1.03
trf = 640e-6
pulseArea=1
OmegaR = 2*pi*pulseArea*VpptoOmegaR*Vpp # 1/s

def dimerlineshape(omega, Eb, TMHz, fudge=0, arb_scale=1):
	# everything in MHz
	Gamma = (np.sqrt(-omega-Eb-fudge) * np.exp((omega+Eb+fudge)/TMHz)* np.heaviside(-omega - Eb-fudge, 1))* arb_scale
# 	Gamma = np.nan_to_num(Gamma)
	return Gamma

def dimerlineshape2(omega, Eb, TMHz, arb_scale=1):
	Gamma = arb_scale* mK/2 * (np.exp(omega + Eb)/TMHz) / np.sqrt(mK*(-omega-Eb))
	return Gamma

def lineshapefit(x, A, x0, sigma):
	ls = A*np.sqrt(-x-x0) * np.exp((x + x0)/sigma) * np.heaviside(-x-x0,1)
	ls = np.nan_to_num(ls)
	return ls
# Ebfix = 3.97493557
Ebfix = 3.97
def lineshapefit_fixedEb(x, A, sigma):
	x0 = Ebfix
	ls = A*np.sqrt(-x-x0) * np.exp((x + x0)/sigma) * np.heaviside(-x-x0,1)
	ls = np.nan_to_num(ls)
	return ls

def lineshape_zeroT(x, A, x0,C):
	kF = 1.1e7
	ls = A*(2*kF**3 - 3*kF**2*np.sqrt(-x-x0) + np.sqrt(-x-x0)**3)*np.sqrt(-x-x0)/(-x-x0) + C
	ls = np.nan_to_num(ls)
	return ls

# data = Data('2023-10-02_H_e.dat', path='E:\\Data\\2023\\10 October2023\\02October2023\\H_202p1G_acdimer_1p8VVA320us', average_by='freq')
# data = Data('2023-10-02_I_e.dat', path='E:\\Data\\2023\\10 October2023\\02October2023\\I_202p1G_acdimer_1p5VVA640us', average_by='freq')
# data = Data('2024-06-12_R_e.dat', average_by='freq')
data = Data(filename='2024-06-12_S_e.dat',path = 'E:\\Analysis Scripts\\analysis\\acdimer\\data', average_by='freq')
# data =Data('2024-06-12_T_e.dat', average_by='freq')
# field=202.1
ToTF=0.31
EF=15.2 # kHz
field = 202.14
df = data.avg_data

freq75 = 47.2159 # MHz, 202.1G
# freq75 = 47.2227 # MHz, 202.14 G
# freq75 = 47.1989 # Mhz, 202 G
# this needs to be redone with c5 + ff c9 rather than sum95 I think?...
df['detuning'] = df.freq - freq75
bgrange = [-3.96, df.detuning.max()]
bgmean = np.mean(df[df['detuning'].between(bgrange[0], bgrange[1])]['sum95'])
df['transfer'] = (-df.sum95 + bgmean) / bgmean
df['em_transfer'] = df.em_sum95 / np.max(df.sum95)
df['ScaledTransfer'] = df.apply(lambda x: GammaTilde(x['transfer'],
								h*EF*1e3, OmegaR*1e3, trf), axis=1)

cutoff = -4.005
df['filter'] = np.where(df['detuning'] > cutoff, 1, 0)
fitdf = df[df['filter']==1]

guess = [2, 0.004]
popt,pcov = curve_fit(lineshapefit_fixedEb, fitdf.detuning, fitdf.transfer, sigma=fitdf.em_transfer, p0=guess)
perr = np.sqrt(np.diag(pcov))
print(popt)
xlow = -4.30
xhigh = -3.9
xx = np.linspace(xlow, xhigh, 500)
yy = lineshapefit_fixedEb(xx, *popt)

guessT0 = [4e-25,Ebfix,0]
yyT0 = lineshape_zeroT(xx, *guessT0)

fig, axs = plt.subplots(2)
fig.suptitle('ac dimer spectrum at 202.14G, EF={:.1f} kHz, T/TF={:.2f}, T={:.1f} kHz'.format(EF, ToTF, ToTF*EF))
ax_loss = axs[0]
ax_loss.errorbar(df[df['filter']==1]['detuning'],df[df['filter']==1]['sum95'], yerr=df[df['filter']==1]['em_sum95'], marker='o', ls='', markersize = 10, capsize=2, mew=2, color='b')
ax_loss.errorbar(df[df['filter']==0]['detuning'],df[df['filter']==0]['sum95'], yerr=df[df['filter']==0]['em_sum95'], marker='o', ls='', markersize = 10, capsize=2, mew=2, mfc='none',color='b')
ax_loss.set_ylabel('Counts [arb.]')
ax_loss.hlines(bgmean, bgrange[0], bgrange[1], label='background')
ax_loss.set_xlim([xlow, xhigh])
ax_loss.legend()

ax_ls = axs[1]
ax_ls.errorbar(df[df['filter']==1].detuning, df[df['filter']==1].transfer, yerr=df[df['filter']==1].em_transfer, marker='o', ls='', markersize = 10, capsize=2, mew=2, color='b', label='data')
ax_ls.errorbar(df[df['filter']==0].detuning, df[df['filter']==0].transfer, yerr=df[df['filter']==0].em_transfer, marker='o', ls='', markersize = 10, capsize=2, mew=2, mfc='none', color='b', label='excluded from fit')
fitstr = r'$A\sqrt{-\omega-E_b}*exp(\frac{\omega+E_b}{T}) *\Theta(-\omega-E_b)$'
ax_ls.plot(xx, yy, ls='--', color='r', label='High-T fit: ' + fitstr)
T0str = r'$A(2k_F^3 - 3k_F^2 *\sqrt{-\omega - E_b} + \sqrt{-\omega - E_b}^3)\frac{\sqrt{-\omega-E_b}}{-\omega-E_b}$'
# ax_ls.plot(xx, yyT0, ls =':', color='g',label='T=0: ' + T0str)
ax_ls.set_xlim([xlow, xhigh])
ax_ls.set_ylabel(r'$\Gamma$ [arb.]')
ax_ls.set_xlabel(r'Detuning from 12-resonance [MHz]')
# ax_ls.legend()

# textstr = '\n'.join((
#  	r'High-T fit params:',
#  	r'Amplitude = {:.2f} +/- {:.1f}'.format(popt[0], perr[0]),
#  	r'T = {:.2f} +/- {:.1f} kHz'.format(popt[1]*1000, perr[1] *1000),
# 	 r'Eb fixed at {:.3f} MHz'.format(Ebfix)
#  	))
# ax_ls.text(xlow + 0.005, 0.045, textstr)

plt.tight_layout()

# fig.savefig('figures/acdimerspectrum_fit.pdf')

# fig, ax = plt.subplots()
# ax.scatter(df.detuning, df.ScaledTransfer)
# # interpolate and integrate
# f_interp = interp1d(fitdf.detuning, fitdf.transfer)
# xx = np.linspace(fitdf.detuning.min(), fitdf.detuning.max(), 100)
# ax_ls.plot(xx, f_interp(xx))

# q = quad(f_interp, fitdf.detuning.min(), fitdf.detuning.max(), points = fitdf.detuning)
# print('q: ' + str(q))


xx = np.linspace(xlow, -Ebfix, 100)
yy = dimerlineshape2(xx, Ebfix, T/10e6, arb_scale=10e6)
ax_ls.plot(xx, yy, 'm--')


