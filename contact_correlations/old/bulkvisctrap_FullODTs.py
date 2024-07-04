# compute trap averaged bulk viscosity of unitary Fermi gas
# given \mu/T, trap \bar\omega/T and the drive frequency \omega/T
# (all quantities E/h in units of Hz or lengths in units of the thermal length lambda_T)
#
# (c) Tilman Enss 2024
#

# Colin Dale modifications 2024-04-03

import numpy as np
from baryrat import BarycentricRational
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar

# print results
print_results = False

#
# properties of homogeneous (bulk) gas: EOS, bulk viscosity, and bulk thermodynamics
#

eosfit = {'nodes': np.array([5.45981500e+01, 3.35462628e-04, 4.48168907e+00, 1.28402542e+00]), 
          'values': np.array([2.66603452e+01, 3.35574145e-04, 5.63725236e+00, 1.91237718e+00]), 
          'weights': np.array([ 0.52786226, -0.10489219, -0.69208542,  0.48101646])}
eosrat = BarycentricRational(eosfit['nodes'],eosfit['values'],eosfit['weights'])

def eos_ufg(betamu):
    """EOS of unitary gas: phase space density f_n(beta*mu) for both spin components (Zwierlein data)"""
    z = np.exp(betamu)
    f_n = 2*np.where(betamu<-8,z,eosrat(z)) # approximant is for a single spin component, so multiply by 2
    return f_n

sumrulefit = {'nodes': np.array([1.22144641e+01, 8.33717634e-03, 3.05244000e+00, 3.48110474e-01]), 
              'values': np.array([1.46259386e+00, 1.24595501e-04, 5.81003381e-01, 5.49151117e-02]), 
              'weights': np.array([ 0.33160786, -0.30343046, -0.66528124,  0.59612671])}
sumrat = BarycentricRational(sumrulefit['nodes'],sumrulefit['values'],sumrulefit['weights'])

def zeta(betamu,betaomega):
    """dimensionless bulk viscosity of unitary gas: zeta-tilde(beta*mu,beta*omega)"""
    z = np.exp(betamu)
    sumruleT = np.where(betamu<-4.8,0.36*z**(5/3),sumrat(z)) # area under viscosity peak in units of T
    gammaT = 1.739-0.0892*z+0.00156*z**2 # width of viscosity peak in units of T
    return sumruleT*gammaT/(betaomega**2+gammaT**2)

def phaseshift_arg(betamu,betaomega):
    """ arctan(omega zeta/sum_rule) """
    z = np.exp(betamu)
    gammaT = 1.739-0.0892*z+0.00156*z**2 # width of viscosity peak in units of T
    return betaomega * gammaT/(betaomega**2+gammaT**2)

#
# trapped gas
#

def weight(v,betabaromega):
    """area of equipotential surface of potential value V/T=v=0...inf"""
    return 2/(betabaromega**3)*np.sqrt(v/np.pi)

def weight_Gaussian(v,betabaromega):
    """area of equipotential surface of potential value V/T=v=0...inf"""
    return 2/(betabaromega**3)*np.sqrt(v/np.pi)

def number_per_spin(betamu,betabaromega):
    """compute number of particles per spin state for trapped unitary gas:
       N_sigma = int_0^infty dv w(v) f_n_sigma*lambda^3(mu-v)"""
    N_sigma,Nerr = quad(lambda v: weight(v,betabaromega)*eos_ufg(betamu-v)/2,0,np.inf,epsrel=1e-4)
    return N_sigma

def Epot_trap(betamu,betabaromega):
    """compute trapping potential energy (in units of T):
       E_trap = int_0^infty dv w(v) f_n*lambda^3(mu-v) v"""
    Epot,Eerr = quad(lambda v: weight(v,betabaromega)*eos_ufg(betamu-v)*v,0,np.inf,epsrel=1e-4)
    return Epot

def thermo_trap(T,betamu,betabaromega):
    """compute thermodynamics of trapped gas"""
    Ns = number_per_spin(betamu,betabaromega)
    EF = T*betabaromega*(6*Ns)**(1/3) # in Hz, without 2pi
    Theta = T/EF
    Epot = T*Epot_trap(betamu,betabaromega) # in Hz, without 2pi
    return Ns,EF,Theta,Epot

def heating_trap(T,betamu,betaomega,betabaromega):
    """compute viscous heating rate E-dot averaged over the trap"""
    Ztrap,Ztraperr = quad(lambda v: weight(v,
			   betabaromega)*eos_ufg(betamu-v)**(1/3)*zeta(betamu-v,betaomega),0,np.inf,epsrel=1e-4)
    # Ztrap_norm,Ztraperr_norm = quad(lambda v: weight(v,betabaromega)*eos_ufg(betamu-v)**(1/3),0,np.inf,epsrel=1e-4)
    Edot = 9*np.pi*(T*betaomega)**2/(3*np.pi**2)**(2/3)*Ztrap
    return Edot #, Ztrap/Ztrap_norm # modified to return trap avged zeta

def phaseshift_arg_trap(betamu,betaomega,betabaromega):
    """compute viscous heating rate E-dot averaged over the trap"""
    argtrap,argtraperr = quad(lambda v: weight(v,
		   betabaromega)*eos_ufg(betamu-v)**(1/3)*phaseshift_arg(betamu-v,betaomega),0,np.inf,epsrel=1e-4)
    argtrap_norm,argtraperr_norm = quad(lambda v: weight(v,betabaromega)*eos_ufg(betamu-v)**(1/3),0,
										np.inf,epsrel=1e-4)
    return argtrap/argtrap_norm #, Ztrap/Ztrap_norm # modified to return trap avged zeta

def find_betamu(T, EF, betabaromega, guess=None):
	"""solves for betamu that matches T, EF and betabaromega of trap"""
	sol = root_scalar(lambda x: EF - T*betabaromega*(6*number_per_spin(x, 
				 betabaromega))**(1/3), bracket=[20e3/T, -300e3/T], x0=guess)
	return sol.root, sol.iterations

class BulkViscTrapToTF:
	def __init__(self, Theta, EF, barnu, mutrap_guess=None, nu_max=120e3):
		#
		# initialize params
		# *all* energies E=h*nu=hbar*omega are given as nu in Hz (without the 2pi factor,
		# even if some of the variables are called omega, like the driving frequency or the trap frequency!)
		#
		hbar = 1.05e-34
		m = 40*1.67e-27 # potassium
		T = EF*Theta
		lambda_T = np.sqrt(hbar/(m*T)) # thermal wavelength (unit of length, in meters)
		a0 = lambda_T # put actual amplitude of scattering length drive, in meters
		self.A = lambda_T/a0 # dimensionless amplitude of drive
		self.nus = np.linspace(1e3,nu_max,2*int(nu_max/1e3)-1) # choose frequency grid for drive frequency nu (in Hz, without the 2pi)
# 		self.nus = np.insert(self.nus, 0, 1)

		#
		# unitless parameters
		#
		betaomegas = self.nus/T # all frequencies, temperatures and energies are without the 2pi
		betamutrap_guess = mutrap_guess/T
		betabaromega = barnu/T # all frequencies, temperatures and energies are without the 2pi
		
		#
		# find betamutrap that produces correct EF given T, EF and betabaromega
		#
		betamutrap, no_iter = find_betamu(T, EF, betabaromega, guess=betamutrap_guess)
		if print_results == True:
			print("Found betamutrap={:.2f} in {} iterations".format(betamutrap, no_iter))
			print("From initial guess {:.2f}".format(betamutrap_guess))
		
		#
		# compute trap properties 
		#
		self.Ns,self.EF,self.Theta,Epot = thermo_trap(T,betamutrap,betabaromega)
		self.Etotal = 2*Epot # virial theorem valid at unitarity, 
		# but we have to decide if we want to normalize the trap heating rate by the total or by the internal energy
		self.Edottraps = self.A**2*np.array([heating_trap(T,betamutrap,betaomega,betabaromega) for betaomega in betaomegas])
		
		#
		# labels for plots
		# 
		self.title = r'Unitary gas with $a^{-1}(t)=\lambda^{-1}\sin(2\pi\nu t)$ at $kT/h=$'+'{:.1f}'.format(T/1e3)+'kHz'
		self.label_trap = r'trap $\bar\nu=$'+'{:.0f}'.format(barnu)+'Hz, $E_F='+'{:.1f}'.format(self.EF/1e3)+'$kHz, $T/T_F=$'+'{:.2f}'.format(self.Theta)
		
		#
		# print results
		#
		if print_results == True:
			print("drive parameters: amplitude 1/a0=%g 1/m, lambda_T=%g m, dimensionless A=%f" % (1/a0,lambda_T,self.A))
			print("trapped system: N_sigma=%f, EF=%f, global T/TF=%f, total energy %f" % (self.Ns,self.EF,self.Theta,self.Etotal))

#
# Debugging
#
bulkvisctrapToTF_debug = False
if bulkvisctrapToTF_debug == True:
	
	Thetas = [0.25, 0.40, 0.58, 0.75, 1.40, 2.00] # ToTF
	Ts = [4.8e3, 7.6e3, 11e3, 14.25e3, 32.6e3, 46.5e3] # Hz
	EFs = [19.0e3, 19.0e3, 19.0e3, 19.0e3, 19.0e3*np.sqrt(1.5) ,19.0e3*np.sqrt(1.5)]
	barnus = [306, 306, 306, 306, 306*np.sqrt(1.5), 306*np.sqrt(1.5)] # mean trap freq in Hz
	mutraps = [9825, 5050, -3800, -14.88e3, -91.8e3, -180e3] # harmonic trap chemical potential
	
	params = list(zip(Ts, barnus, mutraps))
	guess_params = list(zip(Thetas, EFs, barnus))
	
	index = 0

	BVTguess = BulkViscTrapToTF(*guess_params[index], mutrap_guess=mutraps[index], nu_max=120e3)
	
	plt.rcParams.update({"figure.figsize": [6,4]})
 	
	plt.figure()
# 	plt.ylim(0, 0.5)
	plt.xlabel(r'frequency $\nu$ (kHz)')
	plt.ylabel(r"Scaled Heating Rate $\partial_t E/E/A^2$ (Hz)")
	labelguess = "Guess result $T/T_F=${:.2f} trap".format(BVTguess.Theta)
		
	plt.plot(BVTguess.nus/1e3,BVTguess.Edottraps/BVTguess.Etotal,'--',color='blue',label=labelguess)
	plt.tight_layout()
	plt.legend()
	plt.show()