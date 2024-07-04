# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:12:38 2024

@author: coldatoms
"""

# compute trap averaged bulk viscosity of unitary Fermi gas
# given \mu/T, trap \bar\omega/T and the drive frequency \omega/T
# (all quantities in units of T or thermal length lambda)
#
# (c) Tilman Enss 2024
#

import numpy as np
from baryrat import BarycentricRational
from mpmath import polylog
import matplotlib.pyplot as plt
from scipy.integrate import quad

eosfit = {'nodes': np.array([5.45981500e+01, 3.35462628e-04, 1.21824940e+01, 2.11700002e+00, 2.00855369e+01, 9.48773584e+00]), 
          'values': np.array([4.09430522, 1.00045107, 3.075465  , 2.28944495, 3.52625831, 2.95847522]), 
          'weights': np.array([ 0.65950347, -0.00352991,  0.41685159,  0.07509359, -0.26560626, -0.56133033])}
r = BarycentricRational(eosfit['nodes'],eosfit['values'],eosfit['weights'])

def eos_ufg(betamu):
    """EOS of unitary gas: f_n/f_n_ideal(z) for a single spin component"""
    z = np.exp(betamu)
    f_n_ideal = np.vectorize(lambda Z: float(-polylog(1.5,-Z).real))(z)
    factor = np.where(betamu<-8,1+np.sqrt(2)*z,r(z))
    return factor*f_n_ideal

sumrulefit = {'nodes': np.array([1.22144641e+01, 8.33717634e-03, 3.05244000e+00, 3.48110474e-01]), 
              'values': np.array([1.46259386e+00, 1.24595501e-04, 5.81003381e-01, 5.49151117e-02]), 
              'weights': np.array([ 0.33160786, -0.30343046, -0.66528124,  0.59612671])}
rs = BarycentricRational(sumrulefit['nodes'],sumrulefit['values'],sumrulefit['weights'])

def zeta(betamu,nuT):
    """shear viscosity of unitary gas: zeta(beta*mu,beta*omega)"""
    z = np.exp(betamu)
    sumruleT = np.where(betamu<-4.8,0.36*z**(5/3),rs(z))
    gammaT = 1.739-0.0892*z+0.00156*z**2
    return sumruleT*gammaT/(nuT**2+gammaT**2)

def weight(v,barnu):
    """area of equipotential surface of potential value V/T=v=0...inf"""
    return 2/(barnu**3)*np.sqrt(v/np.pi)

def number_per_spin(mu,barnu):
    """compute number of particles per spin state for trapped unitary gas:
       N_sigma = int_0^infty dv w(v) n_sigma*lambda^3(mu-v)"""
    N,Nerr = quad(lambda v: weight(v,barnu)*eos_ufg(mu-v),0,np.inf,epsrel=1e-4)
    print("number:",N,Nerr)
    return N

def Etrap(mu,barnu):
    """compute trapping potential energy:
       E_trap = int_0^infty dv w(v) 2*n_sigma*lambda^3(mu-v) v"""
    E,Eerr = quad(lambda v: weight(v,barnu)*2*eos_ufg(mu-v)*v,0,np.inf,epsrel=1e-4)
    print("trap energy:",E,Eerr)
    return E

def thermo(T,mu0,mubulk0,barnu0=305):
    """compute thermodynamics of trapped gas (energies E=h*nu=hbar*omega in Hz)"""
    mu = mu0/T
    mubulk = mubulk0/T
    barnu = barnu0/T
    Ns = number_per_spin(mu,barnu)
    Etrap0 = T*Etrap(mu,barnu) # in Hz
    EF = barnu0*(6*Ns)**(1/3) # in Hz
    Theta = T/EF
    f_n = 2*eos_ufg(mubulk)
    f_p,f_p_err = quad(lambda v: 2*eos_ufg(mubulk-v),0,np.inf,epsrel=1e-4)
    Ebulk = (3/2)*f_p*T # internal energy is 3/2 times pressure, for two spin components
    return Ns,Etrap0,Ebulk,EF,Theta,f_n,f_p

def visc(T,mu0,mubulk0,nu0,A=1,barnu0=305):
    """compute bulk viscosity in bulk and averaged over the trap"""
    mu = mu0/T
    mubulk = mubulk0/T
    barnu = barnu0/T
    nu = nu0/T
    Z1,Z1err = quad(lambda v: weight(v,barnu)*(2*eos_ufg(mu-v))**(1/3)*zeta(mu-v,nu),0,np.inf,epsrel=1e-4)
    Z = 9/2*2*np.pi*nu0**2*A**2/(3*np.pi**2)**(2/3)*Z1
    Z1bulk = (2*eos_ufg(mubulk))**(1/3)*zeta(mubulk,nu)
    print("Z1bulk",Z1bulk,mubulk,nu,zeta(mubulk,nu))
    Zbulk = 9/2*2*np.pi*nu0**2*A**2/(3*np.pi**2)**(2/3)*Z1bulk
    return Z,Zbulk

Ns,Etrap0,Ebulk,EF,Theta,f_n,f_p = thermo(11000,-3800,1500)
print(Ns,Etrap0,Ebulk,EF,Theta,f_n,f_p)
Z,Zbulk = visc(11000,-3800,1500,50000)
Etot0 = 2*Etrap0 # valid at unitarity
print("trap:",Z,Z/Etot0) # relative rate of change in Hz
print("homogeneous:",Zbulk,Zbulk/Ebulk)

nu0s = np.linspace(0.,150000,25)
Zs = np.zeros_like(nu0s)
Zbulks = np.zeros_like(nu0s)
for i,nu0 in enumerate(nu0s):
    Zs[i],Zbulks[i] = visc(11000,-3800,1500,nu0)
plt.title(r'Unitary gas with $a^{-1}(t)=\lambda^{-1}\sin(2\pi\nu t)$ at $kT/h=11$kHz')
# plt.plot(nu0s,Zbulks/Ebulk,'bo-',label=r'uniform with local $\varepsilon_F=19$kHz')
plt.plot(nu0s,Zs/Etot0,'ro-',label=r'trap $\bar\nu=305$Hz, global $E_F=19$kHz')
plt.legend()
plt.xlabel(r'frequency $\nu$ [Hz]')
plt.ylabel(r'heating rate $\partial_t E/E$ [Hz]')
#plt.savefig('heatingrate_fastdrive.pdf')
# plt.savefig('heatingrate_slowdrive.pdf')

Elist = [10.92078224808712,10.528362790503234,13.90724158248126,11.488624540498774,14.750032315213994]
nulist = [5*1000,10*1000,30*1000,50*1000,150*1000]
plt.plot(nulist,Elist,'o')
plt.show()