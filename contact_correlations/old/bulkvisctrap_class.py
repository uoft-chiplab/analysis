# compute trap averaged bulk viscosity of unitary Fermi gas
# given \mu/T, trap \bar\omega/T and the drive frequency \omega/T
# (all quantities E/h in units of Hz or lengths in units of the thermal length lambda_T)
#
# (c) Tilman Enss 2024
#

# Colin Dale modifications 2024-03-24
# Split traped and uniform gases into two different classes that store
# computed results and parameters. Added another trap gas class that can find
# the chemical potential for given ToTF, barnu, and EF

import numpy as np
from baryrat import BarycentricRational
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar

pi = np.pi

# print results
print_results = False

#
# Contact density estimate from Vale paper
# Crappy eye-balled piecewise function
def contact_density(Theta):
	# ToTF < 0.18
	def func1(Theta):
		return 3
	# ToTF >= 0.18
	def func2(Theta):
		# linearly interpolate two points: (ToTF, C): {(0.2, 2.65), (1.0, 2.3)}
		m = (2.65-2.3)/(0.2-1.0)
		b = 2.65 - m*0.2
		return m*Theta + b
	return np.piecewise(Theta, [Theta<0.18, Theta>=0.18], [func1, func2])

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

def Theta(betamu):
	return (4*pi)/((3*pi**2)* eos_ufg(betamu))**(2/3)

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

def thermo_bulk(betamu, T): # Had to add T as an argument here - CD
    """compute thermodynamics of homogeneous gas (energies E=h*nu=hbar*omega given as nu in Hz)"""
    f_n = eos_ufg(betamu) # phase space density
    theta = 4*np.pi/(3*np.pi**2*f_n)**(2/3)
    f_p,f_p_err = quad(lambda v: eos_ufg(betamu-v),0,np.inf,epsrel=1e-4) # pressure by integrating density over mu
    Ebulk = (3/2)*f_p*T # internal energy density of UFG is 3/2 times pressure, for two spin components (in units of lambda^-3)
    return f_n,theta,f_p,Ebulk

def heating_bulk(T,betamu,betaomega):
    """compute viscous heating rate E-dot in homogeneous system"""
    Zbulk = eos_ufg(betamu)**(1/3)*zeta(betamu,betaomega)
    Edot = 9*np.pi*(T*betaomega)**2/(3*np.pi**2)**(2/3)*Zbulk
    return Edot

def heating_from_zeta(T,betamu,betaomega,zeta):
    """compute viscous heating rate E-dot in homogeneous system"""
    Zbulk = eos_ufg(betamu)**(1/3)*zeta
    Edot = 9*np.pi*(T*betaomega)**2/(3*np.pi**2)**(2/3)*Zbulk
    return Edot

def C_bulk(betamu):
    """compute Contact Density for bulk gas"""
    Cbulk = eos_ufg(betamu)**(4/3)*contact_density(Theta(betamu))
    return Cbulk

def heating_C(T, betaomega, C):
    """compute heating rate at high frequency from contact density"""
    pifactors = (3*pi**2)**(1/3)/(36*pi*(2*pi)**(3/2))
    Edot_C = 9*pi*(T*betaomega)**2/(betaomega)**(3/2)*pifactors*C
    return Edot_C

def zeta_C(betamu, betaomega):
    """compute heating rate at high frequency from contact density"""
    pifactors = 3*pi**2/(36*pi*(2*pi)**(3/2))
    zetaC = pifactors*eos_ufg(betamu) * contact_density(Theta(betamu)) / (betaomega)**(3/2)
    return zetaC

class BulkViscUniform:
	def __init__(self, T, mubulk, nus):
		#
		# initialize params
		# *all* energies E=h*nu=hbar*omega are given as nu in Hz (without the 2pi factor,
		# even if some of the variables are called omega, like the driving frequency or the trap frequency!)
		#
		self.T = T
		self.mubulk = mubulk
		hbar = 1.05e-34
		m = 40*1.67e-27 # potassium
		lambda_T = np.sqrt(hbar/(m*T)) # thermal wavelength (unit of length, in meters)
		a0 = lambda_T # put actual amplitude of scattering length drive, in meters
		self.A = lambda_T/a0 # dimensionless amplitude of drive
		self.nus = nus
# 		self.nus = np.linspace(0.,nu_max,num+1) # choose frequency grid for drive frequency nu (in Hz, without the 2pi)
		
		#
		# compute bulk properties
		#
		
		betamubulk = self.mubulk/self.T
		betaomegas = self.nus/self.T # all frequencies, temperatures and energies are without the 2pi
		f_n,self.Theta,f_p,self.Ebulk = thermo_bulk(betamubulk, self.T)
		self.EF = self.T/self.Theta
		self.Edotbulks = self.A**2*np.array([heating_bulk(self.T,betamubulk,
										betaomega) for betaomega in betaomegas])
		
		self.Cbulk = C_bulk(betamubulk)
		self.C = contact_density(Theta(betamubulk))
		self.EdotbulksC = self.A**2*np.array([heating_C(self.T,betaomega,
										   self.Cbulk) for betaomega in betaomegas])
		
		self.zetas = np.array([zeta(betamubulk,betaomega) for betaomega in betaomegas])
		self.zeta_Cs = np.array([zeta_C(betamubulk, betaomega) for betaomega in betaomegas])  # added for comparison - CD
		
		self.phaseshifts = np.array([np.arctan(phaseshift_arg(betamubulk,
									betaomega)) for betaomega in betaomegas])
		
		self.betamubulk = betamubulk
		
		#
		# labels for plots
		# 
		
		self.title = r'Unitary gas with $a^{-1}(t)=\lambda^{-1}\sin(2\pi\nu t)$ at $kT/h=$'+'{:.1f}'.format(self.T/1e3)+'kHz'
		self.label_uni = r'uniform with local $\varepsilon_F=$'+'{:.1f}'.format(self.T/1e3/self.Theta)+'kHz'
		
		#
		# print results
		#
		
		if print_results == True:
			print("drive parameters: amplitude 1/a0=%g 1/m, lambda_T=%g m, dimensionless A=%f" % (1/a0,lambda_T,self.A))
			print("homogeneous system: phase space density %f, local T/TF=%f, pressure %f, energy density %f" % (f_n,self.Theta,f_p,self.Ebulk))

#
# trapped gas
#

def weight(v,betabaromega):
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

def C_trap(betamu, betabaromega):
    """compute Contact Density averaged over the trap"""
    Ctrap,Ctraperr = quad(lambda v: weight(v,
			   betabaromega)*eos_ufg(betamu-v)**(4/3)*contact_density(Theta(betamu-v)),0,np.inf,epsrel=1e-4)
    return Ctrap

class BulkViscTrap:
	def __init__(self, T, barnu, mutrap, nus):
		#
		# initialize params
		# *all* energies E=h*nu=hbar*omega are given as nu in Hz (without the 2pi factor,
		# even if some of the variables are called omega, like the driving frequency or the trap frequency!)
		#
		self.T = T
		self.barnu = barnu
		self.mutrap = mutrap
		hbar = 1.05e-34
		m = 40*1.67e-27 # potassium
		self.lambda_T = np.sqrt(hbar/(m*T)) # thermal wavelength (unit of length, in meters)
		a0 = self.lambda_T # put actual amplitude of scattering length drive, in meters
		self.A = self.lambda_T/a0 # dimensionless amplitude of drive
# 		self.nus = np.linspace(0.,nu_max,int(nu_max/1e3)+1) # choose frequency grid for drive frequency nu (in Hz, without the 2pi)
		self.nus = nus
		#
		# compute trap properties
		#
		
		betaomegas = self.nus/self.T # all frequencies, temperatures and energies are without the 2pi
		betamutrap = self.mutrap/self.T
		betabaromega = barnu/T # all frequencies, temperatures and energies are without the 2pi
		self.Ns,self.EF,self.Theta,Epot = thermo_trap(self.T,betamutrap,betabaromega)
		self.kF = np.sqrt(4*pi*m*self.EF/hbar)
		self.Etotal = 2*Epot # virial theorem valid at unitarity, 
		# but we have to decide if we want to normalize the trap heating rate by the total or by the internal energy
		self.Edottraps = self.A**2*np.array([heating_trap(self.T,betamutrap,
						betaomega,betabaromega) for betaomega in betaomegas])
	
		self.Ctrap =  C_trap(betamutrap, betabaromega)
		self.EdottrapsC = self.A**2*np.array([heating_C(self.T,betaomega,
										   self.Ctrap) for betaomega in betaomegas])
		#
		# labels for plots
		# 
		self.title = r'Unitary gas with $a^{-1}(t)=\lambda^{-1}\sin(2\pi\nu t)$ at $kT/h=$'+'{:.1f}'.format(self.T/1e3)+'kHz'
		self.label_trap = r'trap $\bar\nu=$'+'{:.0f}'.format(self.barnu)+'Hz, $E_F='+'{:.1f}'.format(self.EF/1e3)+'$kHz, $T/T_F=$'+'{:.2f}'.format(self.Theta)
		
		#
		# print results
		#
		if print_results == True:
			print("drive parameters: amplitude 1/a0=%g 1/m, lambda_T=%g m, dimensionless A=%f" % (1/a0,self.lambda_T,self.A))
			print("trapped system: N_sigma=%f, EF=%f, global T/TF=%f, total energy %f" % (self.Ns,self.EF,self.Theta,self.Etotal))

class BulkViscTrapToTF:
	def __init__(self, Theta, EF, barnu, nus, mutrap_guess=None):
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
		self.nus = nus # choose frequency grid for drive frequency nu (in Hz, without the 2pi)

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

###
### Plotting
###

### Uniform Density System 
bulk_plot = True

if bulk_plot == True:
	titlebulk = 'Uniform Density Gas'
	
	Thetas = [0.25, 0.58]
	Ts = [4.8e3, 11e3] # Hz
	barnus = [306, 306] # mean trap freq in Hz
	mubulks = [7520, 1500] # uniform trap chemical potential
	BV_colors = ['teal', 'r']
	theta_indices = [0, 1]
	params_bulk = list(zip(Ts, mubulks))
	
	# load Tilman L-W data from PRL
	xTilman = []
	yTilman = []
	TilmanFiles = ["zetaomega_T0.25.txt", "zetaomega_T0.58.txt"]
	for file in TilmanFiles:
		xx, yy = np.loadtxt(file, unpack=True, delimiter=' ')
		yy = yy/12 	# funny arbitrary scaling
		xTilman.append(xx)
		yTilman.append(yy)
	
	# set plotting parameters
	plt.rcParams.update({"figure.figsize": [6,8]})
	xlabel=r'Frequency $\omega/E_F$'
	font = {'size'   : 12}
	plt.rc('font', **font)
	legend_font_size = 10 # makes legend smaller, so plot is visible
	fig, axs = plt.subplots(2,1)
	
	# make frequency array
	nu_max = 200000
	num = int(nu_max/1e3)+1
	nus = np.linspace(0.,nu_max,int(nu_max/1e3)+1)
	
	# set axes attributes
	ax = axs[0]
	ylabel = r'Heating Rate $\dot{E}/E$ (Hz)'
	ax.set(xlabel=xlabel, ylabel=ylabel, title=titlebulk)
	
	ax_zeta = axs[1]
	ylabel = r'Contact Correlation $z$'
	ax_zeta.set(xlabel=xlabel, ylabel=ylabel, yscale='log', xscale='log')
	
	label_Drude = 'Drude'
	label_LW = 'L-W'
	label_C = 'Contact'

	# loop over ToTF
	BVUs = []
	for i in theta_indices:
		color = BV_colors[i]
		BVU = BulkViscUniform(*params_bulk[i], nus)
		label = "$T/T_F=${:.2f}".format(Thetas[i])
		
		# find nus for nu/EF > 1 to use for high_freq
		nu_small = 0
		for nu in nus:
			if nu < (BVU.T/BVU.Theta):
				nu_small += 1
# 		nu_small = 0
		
		TilmanEdotbulk = BVU.A**2*np.array([heating_from_zeta(BVU.T, BVU.betamubulk, 
			betaomega, zeta) for betaomega, zeta in zip(xTilman[i]/BVU.Theta, yTilman[i])])
		
		ax.plot(BVU.nus/BVU.EF, BVU.Edotbulks/BVU.Ebulk, '-.', label=label, 
		  color=color)
		ax.plot(xTilman[i], TilmanEdotbulk/BVU.Ebulk, '-', color=color)
		ax.plot(BVU.nus[nu_small:]/BVU.EF, BVU.EdotbulksC[nu_small:]/BVU.Ebulk, '--', color=color)
		
		ax_zeta.plot(BVU.nus/BVU.EF, BVU.zetas, '-.', label=label_Drude, color=color)
		ax_zeta.plot(xTilman[i], yTilman[i], '-', label=label_LW, color=color)
		ax_zeta.plot(BVU.nus[nu_small:]/BVU.EF, BVU.zeta_Cs[nu_small:], '--', 
			   label=label_C, color='k')
		
		if i == 0:
			ax_zeta.legend(prop={'size':legend_font_size})
		
		BVUs.append(BVU)
	
	ax.legend(prop={'size':legend_font_size})
	plt.tight_layout()
	plt.show()


### Harmonic Trap
trap_plot = True

if trap_plot == True:
	titlebulk = 'Harmonically Trapped Gas'
	
	Thetas = [0.25, 0.58]
	Ts = [4.8e3, 11e3] # Hz
	barnus = [306, 306] # mean trap freq in Hz
	mutraps = [9825, -3800] # harmonic trap chemical potential
	BV_colors = ['teal', 'r']
	theta_indices = [0, 1]
	params_trap = list(zip(Ts, barnus, mutraps))
	
	# load Tilman L-W data from PRL
	xTilman = []
	yTilman = []
	TilmanFiles = ["zetaomega_T0.25.txt", "zetaomega_T0.58.txt"]
	for file in TilmanFiles:
		xx, yy = np.loadtxt(file, unpack=True, delimiter=' ')
		yy = yy/12 	# funny arbitrary scaling
		xTilman.append(xx)
		yTilman.append(yy)
	
	# set plotting parameters
	plt.rcParams.update({"figure.figsize": [6,8]})
	xlabel=r'Frequency $\omega/E_F$'
	font = {'size'   : 12}
	plt.rc('font', **font)
	legend_font_size = 10 # makes legend smaller, so plot is visible
	fig, axs = plt.subplots(2,1)
	
	# make frequency array
	nu_max = 200000
	num = int(nu_max/1e3)+1
	nus = np.linspace(0.,nu_max,int(nu_max/1e3)+1)
	
	# set axes attributes
	ax = axs[0]
	ylabel = r'Heating Rate $\dot{E}/E$ (Hz)'
	ax.set(xlabel=xlabel, ylabel=ylabel, title=titlebulk)
	
	ax_zeta = axs[1]
	ylabel = r'Contact Correlation $z$'
	ax_zeta.set(xlabel=xlabel, ylabel=ylabel, yscale='log', xscale='log')
	
	label_Drude = 'Drude'
	label_LW = 'L-W'
	label_C = 'Contact'

	# loop over ToTF
	BVUs = []
	for i in theta_indices:
		color = BV_colors[i]
		BVU = BulkViscUniform(*params_bulk[i], nus)
		label = "$T/T_F=${:.2f}".format(Thetas[i])
		
		# find nus for nu/EF > 1 to use for high_freq
		nu_small = 0
		for nu in nus:
			if nu < (BVU.T/BVU.Theta):
				nu_small += 1
# 		nu_small = 0
		
		TilmanEdotbulk = BVU.A**2*np.array([heating_from_zeta(BVU.T, BVU.betamubulk, 
			betaomega, zeta) for betaomega, zeta in zip(xTilman[i]/BVU.Theta, yTilman[i])])
		
		ax.plot(BVU.nus/BVU.EF, BVU.Edotbulks/BVU.Ebulk, '-.', label=label, 
		  color=color)
		ax.plot(xTilman[i], TilmanEdotbulk/BVU.Ebulk, '-', color=color)
		ax.plot(BVU.nus[nu_small:]/BVU.EF, BVU.EdotbulksC[nu_small:]/BVU.Ebulk, '--', color=color)
		
		ax_zeta.plot(BVU.nus/BVU.EF, BVU.zetas, '-.', label=label_Drude, color=color)
		ax_zeta.plot(xTilman[i], yTilman[i], '-', label=label_LW, color=color)
		ax_zeta.plot(BVU.nus[nu_small:]/BVU.EF, BVU.zeta_Cs[nu_small:], '--', 
			   label=label_C, color='k')
		
		if i == 0:
			ax_zeta.legend(prop={'size':legend_font_size})
		
		BVUs.append(BVU)
	
	ax.legend(prop={'size':legend_font_size})
	plt.tight_layout()
	plt.show()