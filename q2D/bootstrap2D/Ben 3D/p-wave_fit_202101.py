# p-wave_fit.py
# 
# 2020/12/22
#
# Written by Ben A Olsen
# 
# Perform fits of loss spectra of ultracold atomic clouds near a p-wave Feshbach
# resonance (specifically K-40 near 198.8 G)
# The fitting used is bootstrap resampling since we don;t have a good model for
# the error on individual data points.

# Imports ######################################################################
import io
import time
import sys
import numpy as np

from scipy import optimize # nonlinear fitting
from scipy import constants # CODATA constants for physical parameters
from scipy import integrate
#from scipy.stats import linregress
#from scipy.stats import chi2
#from scipy.stats import moment, kurtosis, skew
from math import sqrt, exp, sin, cos, atan 
# faster functions when we can't vectorize

import os # for dealing with files/directories
import csv # for summary files containting fit results
import warnings # for suppressing warnings due to poorly-converging fits

import matplotlib.pyplot as plt
#import matplotlib.colors as col
import matplotlib.cm as cm

import pwave_data  # experimental data stored in another file
import pwave_fd_interp as FD # FD distribution data for interpolation functions

# Initialization ###############################################################

do_profile = False # profile code execution (for optimizing runtime)
if do_profile:
    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()

#warnings.filterwarnings("ignore",category=RuntimeWarning)
warnings.filterwarnings("ignore")

# use CODATA18 for physical constant in SI units
h = constants.value('Planck constant')  # value of Planck constant in J s
hbar = h/(2*np.pi)  # value of reduced Planck constant in J s
uB = constants.value('Bohr magneton')  # value of Planck constant in J s
aB = constants.value('Bohr radius')  # value of Bohr radius in m
u = constants.value('atomic mass constant')  # value of Planck constant in J s
mK40 = 39.96399848 * u  # mass K-40 in kg
dmu = .134*uB  # differential mag. moment between -9/2, -7/2 K40 states (J/T)
kB = constants.value('Boltzmann constant')  # value of Planck constant in J s
pi = np.pi
sqrt12 = np.sqrt(0.5)

# FR parameters
Rvdw = 65.02*aB  # Faulke 2008
B0xy = 0.01983  # xy resonance location in T
DBxy = -.002455  # xy width in T
vbgxy = -905505 * aB**3  # xy bg scattering volume
B0z = .01988  # z resonance location in T
DBz = -0.002195  # z resonance width in T
vbgz = -1040850 * aB**3  # z bg scattering volume


def colorfunc(fitType): 
    # Choose the color of the plot/text/etc. based on the fit type
    cmapH = cm.get_cmap("hot")
    cmapC = cm.get_cmap("cool")
    if fitType == 'MB': # Maxwell-Boltzmann
        return cmapH(2/7.)
    else: # Fermi-Dirac
        return cmapC(5/11.)

# Fit Functions ################################################################

def Reffxy(B): 
    # Effective range for the xy resonance
    return -vbgxy*mK40*(-B+B0xy+DBxy)**2*dmu/(DBxy*hbar**2)


def Reffz(B): 
    # Effective Range for the z resonance
    return -vbgz*mK40*(-B+B0z+DBz)**2*dmu/(DBz*hbar**2)


Reff = Reffz(200) # choose a junk Reff to prevent odd errors


def FC0(k, ka, Reff): 
    # Franck-Condon overlaps
    return mK40*abs(Reff)/(np.pi*hbar**2) * k**3/((k**2+ka**2)**2)


def fit_gauss(f, Ni, A, width, fb):
    # gaussian centered around -fb (sign chosen to agree with other fits)
    return Ni*(1 - .1*A/np.sqrt(2*pi)/width*np.exp(-(f+fb)**2/2/width**2))


def kk(f, fb): 
    # scattering wave vector using the sign convention that fb is negative
    return np.sqrt(h*mK40*abs(f+fb)*1000/hbar/hbar)


def ka(fb): 
    # resonance pole value kappa
    return np.sqrt(h*mK40*abs(fb)*1000/hbar**2)


def fit_MB_bound(f, Ni, A, T, fb): 
    # fit using Maxwell-Boltzmann statistics
    return np.piecewise(
        f, [f <= -fb],
        [lambda f: Ni,
         # when f < -fb
            lambda f: Ni*(1 -
                          1e-32 * A * kk(f,fb)**2/(4*np.pi**3) *
                          np.sqrt(np.pi*hbar**2/(mK40*kB*T*1e-9)) *
                          np.exp(-kk(f,fb)**2*hbar**2/(mK40*kB*T*1e-9)) *
                          FC0(kk(f,fb), ka(fb), Reff))
            ]
        )

# Fermi-Dirac distribution fitting functions #########

def DFGpow(power, E, EF, func):
    # auxiliary function in calculation of F-D distribution function
    return (E/EF)**power * func(np.sqrt(E/EF/2.))


def DFG0p5(EEF2, func):
    # special case of auxiliary function
    return EEF2 * func(EEF2*sqrt12)


def DFGfitpre(f, EF, power, func, Reff, fb):
    # prefactor in F-D fitting
    return (
        DFGpow(power, h*abs(fb+f)*1000., EF, func) *
        FC0(kk(f,fb), ka(fb), Reff)
    )
    
def DFGfitscale(f, EF, power, func, Reff, fb):
    # scaling factor in F-D fitting
    return DFGfitpre(-fb+0.5*EF/h/1000., EF, power, func, Reff, fb)

numDA = 5 # choose which T/TF interpolated function to choose [0-11]

def fit_FD_bound(f, Ni, Amp, TF, fb):
    # Fermi-Dirac statistics for boud-state lineshapes
    return np.piecewise(
        f, [f <= -fb],
        [lambda f: Ni,
            lambda f: Ni * (1 -
            Amp * DFGfitpre(f, TF*kB*1e-9, 0.5, FD.distRAv[numDA], Reff, fb)/
            DFGfitscale(f, TF*kB*1e-9, 0.5, FD.distRAv[numDA], Reff, fb))]
    )

# Quasibound fitting functions #####################

def oldFCQ0(fp, f, fb):
    kp = np.sqrt(mK40*h*abs(fp-f)*1000)/hbar
    kq = np.sqrt(mK40*h*fp*1000)/hbar
    #deltaq = np.arctan2(1/(-1/(kq**3*nu)+1/(kq*Reffz(B))))
    Er = hbar**2 / (mK40 * Reff**2)
    sigma = 4*np.pi*hbar**2/mK40 * (h*fp*1000)**2/((h*fp*1000)**3 + ((h*(fp - fb)*1000)**2*Er))
    np.sinsquared = kq**2 /(4*np.pi) * sigma
    integral = (kp/kq)**(1.5)/(kp**2-kq**2)*mK40/(np.pi*hbar**2)
    return np.sinsquared * abs(integral)**2


def FCQ0(fq, f, fb):
    kq = np.sqrt(h*mK40*abs(fq)*1000/hbar**2)
    Er = hbar**2 / (mK40 * Reff**2)
    sigma = 4*np.pi*hbar**2/mK40 * (h*fb*1000)**2/((h*fb*1000)**3 + ((h*(fq - fb)*1000)**2*Er))
    sindel = kq**2 /(4*np.pi) * sigma
    return mK40**2 * kk(f,fb)**3 * sindel / (np.pi**2 * hbar**4 * kq**3 *
                                             (kk(f, fb)**2 - kq**2))


def newFCQ0(fq, k, fb):
    kq = sqrt(h*mK40*fq*1000/hbar/hbar)
    return mK40 * k*k*k * fb*fb / (fb*fb*fb*h*1000 + (fq - fb)*(fq-fb)*hbar*hbar / (mK40 * Reff*Reff)) / (pi*pi * hbar*hbar * kq * (k*k - kq*kq))


def NewThermal3D(fp, f, T):
    xT = kB*T*1e-9
    return hbar*h*(fp-f)*1000/sqrt(mK40*pi*pi*pi*xT*xT*xT)*exp(-h*(fp-f)*1000/xT)


def FCQB_int(f, Off, Amp, T, fb):
    kval = kk(f,fb)
    xT = abs(kB*T*1e-9)
    def integrand(fp):  # here fp is the integration variable
        kp = sqrt(h*mK40*abs(fp)*1000/hbar/hbar)
        return (
            sqrt(mK40*h*(fp-f)*1000)*
            h*1000/sqrt(mK40*pi*pi*pi*xT*xT*xT)*(abs(fp-f)*exp(-h*abs(fp-f)*1000/xT)-
             fp*exp(-h*abs(fp)*1000/xT))*
            #FCQ0(fp, f, fb)
            #newFCQ0(fp, kval, fb)
            mK40 * kval*kval*kval * fb*fb / (fb*fb*fb*h*1000 + (fp - fb)*(fp-fb)*hbar*hbar / (mK40 * Reff*Reff)) / (pi*pi * hbar*hbar * kp * (kval*kval - kp*kp))
        )
    
    return Off - 3e-39*Amp*h*integrate.quad(integrand,f,f+200)[0]

fit_MB_qb = np.vectorize(FCQB_int)


def FCQB_FD_int(f, Off, Amp, TF, fb):
    kval = kk(f,fb)
    func = FD.distRA[numDA]
    xT = abs(h*1000./(TF*kB*1e-9))
    #@lru_cache(maxsize = 10000)
    def integrand(fp):  # here fp is the integration variable
        if fp <= 0:
            return 0.0
        kp = sqrt(h*mK40*abs(fp)*1000/hbar/hbar)
        return (
            #(DFGpow(0.5, h*df*1000., TF*kB*1e-9, FD.distRA[numDA])-
            # DFGpow(0.5, h*fp*1000., TF*kB*1e-9, FD.distRA[numDA]))*
            #(DFG0p5(np.sqrt(h*(fp-f)*1000./(TF*kB*1e-9)), func)-
            # DFG0p5(np.sqrt(h*fp*1000./(TF*kB*1e-9)), func))*
            #newFCQ0(fp, kval, fb)
            (sqrt((fp-f)*xT) * func(sqrt((fp-f)*xT)*sqrt12) -
            sqrt(fp*xT) * func(sqrt(fp*xT)*sqrt12)) / 
            (fb*fb*fb*h*1000 + (fp - fb)*(fp-fb)*hbar*hbar / (mK40 * Reff*Reff)) / 
            (pi*pi * hbar*hbar * kp * (kval*kval - kp*kp))
        )
    
    return Off - 2e-68*Amp * mK40 * kval*kval*kval * fb*fb* integrate.quad(integrand,f,f+200, epsabs = 1e-3, limit=50)[0]#np.logspace(0,3,4))[0]

fit_FD_qb = np.vectorize(FCQB_FD_int)


def MBW_qb(f, Off, Amp, T, fb):
    # for wiggling data

    nu = hbar*hbar*Reff/(mK40*fb*1000*h)
    R = Rvdw*2**(3/4)
    eps = np.finfo(float).eps
    xT = kB*abs(T)*1e-9
    
    def FCQBFull(df, ff):
        kp = sqrt(mK40*df*1000*h)/hbar
        kq = sqrt(mK40*abs(ff)*1000*h)/hbar
        
        deltaq = atan(1/(-1/(kq*kq*kq*nu+eps)+1/(kq*Reff+eps)))
        deltap = atan(1/(-1/(kp*kp*kp*nu)+1/(kp*Reff)))
        
        cq = cos(deltaq)
        sq = sin(deltaq)
        ckq = cos(kq*R)
        skq = sin(kq*R)
        
        cp = cos(deltap)
        sp = sin(deltap)
        ckp = cos(kp*R)
        skp = sin(kp*R)
        
        int01  = 0.5*pi*cp*cq
        int02  = 0.5*pi*sp*sq
        int1   = (-kp*kp*kq*R*ckq*skp+(kp*kq*kq*R*ckp+(kp*kp-kq*kq)* skp)*skq)*cp*cq
        int2   = (-kp*kq*kq*R*ckq*skp+(kp*kp*kq*R*skq+(kp*kp-kq*kq)* ckq)*ckp)*sp*sq
        int3   =  (kp*kq*kq*R*ckq*ckp+(kp*kp*kq*R*skq+(kp*kp-kq*kq)* ckq)*skp)*cp*sq
        int4   = -(kp*kq*kq*R*skq*skp+(kp*kp*kq*R*ckq+(-kp*kp+kq*kq)*skq)*ckp)*sp*cq

        return mK40*mK40/(pi*pi*hbar*hbar*hbar*hbar*(kp*kq+eps))/(kp*kq*(kp*kp-kq*kq)*R)**2 * (int01+int02+int1+int2+int3+int4 if kp == kq else int1+int2+int3+int4)**2

    def integrand(fp):  # here fp is the integration variable
        df = abs(fp-f)
        return (
            sqrt(mK40*h*df*1000)*
            h*1000/sqrt(mK40*pi*pi*pi*xT*xT*xT)*(df*exp(-h*df*1000/xT)- fp*exp(-h*fp*1000/xT))*
            FCQBFull(df,fp)
        )

    return Off - 1e-22*Amp*h*integrate.quad(integrand,f,f+200)[0]

fit_MBW_qb = np.vectorize(MBW_qb)


def FDW_qb(f, Off, Amp, TF, fb):
    # for wiggling data

    func = FD.distRA[numDA]
    nu = hbar*hbar*Reff/(mK40*fb*1000.0*h)
    R = Rvdw*2.0**(3.0/4.0)
    eps = np.finfo(float).eps
    xT = abs(h*1000./(TF*kB*1e-9))
    
    

    def FCQBFull(df, ff):
        kp = sqrt(mK40*df*1000*h)/hbar
        kq = sqrt(mK40*abs(ff)*1000*h)/hbar
        
        deltaq = np.arctan(1/(-1/(kq*kq*kq*nu+eps)+1/(kq*Reff+eps)))
        deltap = np.arctan(1/(-1/(kp*kp*kp*nu)+1/(kp*Reff)))
        
        cq = cos(deltaq)
        sq = sin(deltaq)
        ckq = cos(kq*R)
        skq = sin(kq*R)
        
        cp = cos(deltap)
        sp = sin(deltap)
        ckp = cos(kp*R)
        skp = sin(kp*R)
        
        int01  = 0.5*pi*cp*cq
        int02  = 0.5*pi*sp*sq
        int1   = (-kp*kp*kq*R*ckq*skp+(kp*kq*kq*R*ckp+(kp*kp-kq*kq)* skp)*skq)*cp*cq
        int2   = (-kp*kq*kq*R*ckq*skp+(kp*kp*kq*R*skq+(kp*kp-kq*kq)* ckq)*ckp)*sp*sq
        int3   =  (kp*kq*kq*R*ckq*ckp+(kp*kp*kq*R*skq+(kp*kp-kq*kq)* ckq)*skp)*cp*sq
        int4   = -(kp*kq*kq*R*skq*skp+(kp*kp*kq*R*ckq+(-kp*kp+kq*kq)*skq)*ckp)*sp*cq

        return mK40*mK40/(pi*pi*hbar*hbar*hbar*hbar*(kp*kq+eps))/(kp*kq*(kp*kp-kq*kq)*R)**2 * (int01+int02+int1+int2+int3+int4 if kp == kq else int1+int2+int3+int4)**2

    def integrand(fp):
        # here fp is the integration variable
        if fp <= 0: return 0.0
        df = abs(fp-f)
        return (
            sqrt(mK40*h*(fp-f)*1000)*
            (sqrt((fp-f)*xT) * func(sqrt((fp-f)*xT)*sqrt12) -
            sqrt(fp*xT) * func(sqrt(fp*xT)*sqrt12))*
            FCQBFull(df,fp)
        )

    return Off - 2e9*Amp*h*integrate.quad(integrand,f,f+200)[0]

fit_FDW_qb = np.vectorize(FDW_qb)

# Load data ####################################################################

fitType = 'FD'
# fit functional form
# Options are 'MB' for Maxwell-Boltzmann, or 'FD' for Fermi-Dirac
for dataType in ['bound_z_w']:
#['bound_xy_w', 'bound_xy_rf', 'bound_z_w', 'bound_z_rf']:
#['qb_xy_w','qb_xy_rf','qb_z_w','qb_z_rf']

    if dataType == 'bound_xy_w':
        runs = ['Nov7runC', 'Nov7runH', 'Nov7runI', 'Nov8runE', 'Nov8runF',
                'Nov8runG', 'Nov8runH', 'Nov9runJ', 'Nov9runK', 'Nov9runL',
                'Nov19runK_15', 'Nov19runK_25', 'Nov19runK_45', 'Nov19runK_5',
                'Nov19runT_5', 'Nov19runT_10', 'Nov19runT_2', 'Nov19runT_15',
                'Dec5runD_0p35', 'Dec5runD_0p18', 'Dec5runD_0p14',
                'Dec5runD_0p10', 'Dec5runD_0p06', 'Dec5runE_0p10',
                'Dec5runE_0p14', 'Dec5runE_0p18', 'Dec5runE_0p35',
                'Dec5runE_0p06', 'Dec5runH_0p10', 'Dec5runH_0p20',
                'Dec5runH_0p30', 'Jan25runE', 'Feb7runB_1V', 'Feb7runB_0p1V',
                'Feb7runC', 'Feb7runD', 'Feb14runB', 'JinData']

    elif dataType == 'bound_xy_rf':
        runs = ['Feb19RunI', 'Feb19RunJ', 'Feb19RunK', 'Feb20RunB', 'Feb20RunF',
                'Feb20RunC', 'Feb21RunD', 'Feb22RunG', 'Feb26RunD',
                'Feb26RunE', 'Feb27RunC', 'Feb27RunE', 'Feb27RunF', 'Feb28RunB',
                'Feb28RunC', 'Feb28RunD', 'Feb28RunE', 'Aug02RunE', 'Aug06RunE',
                'Aug06RunH', 'Aug07RunD', 'Aug07RunG', '2020Jan28RunF',
                '2020Jan29RunE', '2020Jan30RunF', '2020Feb03RunE',
                '2020Feb04RunG', '2020Feb27RunS', '2020Feb27RunP',
                '2020Mar11RunD', '2020Mar11RunE', '2020Mar11RunF',
                '2020Mar11RunH', '2020Mar12RunD', '2020Mar12RunJ',
                '2020Mar12RunL', '2020Mar13RunF', '2020Mar16runD',
                '2020Mar16runI', '2020Mar16runK']

    elif dataType == 'bound_z_w':
        runs = ['Nov7runCz', 'Nov9runJz', 'Nov8runHz', 'Nov8runGz', 'Nov8runFz',
                'Nov8runEz', 'Nov7runHz', 'Nov7runIz']
        runs = ['Nov7runCz']

    elif dataType == 'bound_z_rf':
        runs = ['Feb20runDz', 'Feb19runIz', 'Feb19runJz', 'Feb19runKz',
                'Feb20runBz', 'Feb20runCz', 'Feb21runGz', 'Feb27runEz',
                'Feb27runFz', 'Aug02runEz','Aug06runEz', '2020Jan28runFz',
                '2020Jan29runFz', '2020Jan30runFz', '2020Feb03runEz',
                '2020Feb04runGz', '2020Feb27runSz', '2020Feb27runPz']

    elif dataType == 'qb_xy_w':
        runs = ['Nov07runDxy', 'Nov09runDxy','Nov09runHxy']
        runs = ['Nov09runDxy']

    elif dataType == 'qb_xy_rf':
        runs = ['Feb06runHxy', 'Mar02runJ', 'Mar03runIxy', 'Mar06runExy',
                'Mar17runDxy', 'Mar17runGxy']

    elif dataType == 'qb_z_w':
        runs = ['Nov07runD', 'Nov09runD', 'Nov09runG', 'Nov09runH', 'Nov09runI',
                'Nov19runJ']

    elif dataType == 'qb_z_rf':
        runs = ['Feb06runH', 'Feb22runD','Mar03runJ','Mar03runI', 'Mar06runE',
                'Mar17runD', 'Mar17runG']

    try:
        os.remove(dataType + '/' + fitType + '_' + dataType + '.csv')
    except:
        print('Nothing to delete')
        
    for run in runs:
        print('**** Analyzing ' + run + " ****")

        myData = pwave_data.loadData(run)
        
        if myData.exclude:
            print('Excluded')
            #continue

        specData = myData.data
        # the atom loss spectrum data
        specData.sort(key=lambda x: x[0])
        # sort the data by the frequency
        nData = len(specData)
        # count how many data points
        
        qb_factor = (-1.0 if myData.state == 'bound' else 1.0)

        if myData.method == 'wiggle':
            xx = np.array([s[0] for s in specData]) 
            # in the case of field wiggling data, no conversion necessary
        else:
            xx = np.array([qb_factor*1000*(s[0]-myData.freq97) for s in specData])
            # in the case of rf spectra, convert rf frequencies in MHz to detunings in kHz (CHECK THE SIGN)
            
        yy = [s[1] for s in specData]
        # the remaining atom number vector

        B = myData.B/1e4
        # convert bias magnetic field from gauss to tesla
        
        if myData.resonance == 'xy':
            # choose the correct effective range expression
            Reff = Reffxy(B)
        else:
            Reff = Reffz(B)
            
        if fitType =='MB':
            plabels = ["Ni", "A", "T", "fB", 'chisq', 'b-g']
        else:
            plabels = ["Ni", "A", "TF", "fB", 'chisq', 'b-g']

        if hasattr(myData, 'pGuess'):
            pGuessMB = np.array(myData.pGuess, dtype=float)
            
        if hasattr(myData,'pGuessFD'):
            pGuessFD = np.array(myData.pGuessFD, dtype=float)
        else:
            pGuessFD = FD.pGuessFD

        pGuess=np.ndarray([4],dtype=float)
        
        def guess_func(state, method):
            
            if fitType =='MB':
                if state == 'quasibound':
                    if method == 'wiggle':
                        func = fit_MBW_qb
                        #pGuess[1] = 50*pGuessFile[1]
                    else:
                        func = fit_MB_qb
                else: # the case of free - bound transition
                    func = fit_MB_bound
                pGuess = pGuessMB
            else:
                if state == 'quasibound':
                    if method == 'wiggle':
                        func = fit_FDW_qb
                    else:
                        func = fit_FD_qb
                else:# the case of free - bound transition
                    func = fit_FD_bound
                pGuess = pGuessFD
                
            return pGuess, func
                    

        do_debug = False
        # an initial debugging plot to make sure fit guesses are reasonable
        do_guesses = True
        # display an initial guess plot
        do_fits = True
        # display an initial fit result
        if do_debug:
            plt.plot(xx, yy, '+r')

            xFit = np.linspace(np.min(xx)-10, np.max(xx)+10, 40)

            pGuess, func = guess_func(myData.state, myData.method)
            if do_guesses:
                print("Guess:")
                print(pGuess)
                print(func(100., *pGuess))
                plt.plot(xFit, (func(xFit, *pGuess)), '--',color=colorfunc(fitType))
            
            if do_fits:
                pGuess, func = guess_func(myData.state, myData.method)
                # pylint: disable=unbalanced-tuple-unpacking
                pFit, cov = optimize.curve_fit(func, xx, yy, pGuess)
                
                plt.plot(xFit, (func(xFit, *pFit)), '-', color='black')
                plt.axvline(qb_factor*pFit[3], color='black')
                print("Fit:")
                print(pFit)


            if do_profile:
                pr.disable()
                s = io.StringIO()
                sortby = SortKey.TIME
                ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                ps.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
                print(s.getvalue())
            plt.show()
            continue
            # don't do the rest of the fitting


        folder = dataType + '/' + myData.run + '/'

        try:
            os.makedirs(folder)
        except Exception:
            pass
        
        # Bootstrap fitting ####################################################
        
        do_append = False
        # append new fits to the old file
        do_stats = True
        # output results to summary csv files
        do_plots = True
        # output fit plots and histograms
        trialsB = 1000
        # total Monte Carlo fits to perform

        print("** Bootstrap resampling")
                    
        if os.path.isfile(folder + '/' + fitType + '_fits.txt'):
            pFitB = np.loadtxt(folder + '/' + fitType + '_fits.txt', delimiter=', ')
            print(np.shape(pFitB))
            old_trialsB = np.shape(pFitB)[0]
            # number of old trials loaded in
            print("   Loading from " + fitType + " file: %i old trials" % old_trialsB)
            if do_append:
                trialB = old_trialsB # starting position for new fits
                trialsB = trialsB-old_trialsB # update index
                if trialsB > 0:
                    pFitB_old = pFitB.copy()
                    pFitB = np.concatenate((pFitB,np.zeros([trialsB, 4])))
                    new_trialsB = True
                else:
                    new_trialsB = False
            else:
                new_trialsB = False
        elif os.path.isfile(folder + '/bs_params.npy'):
            pFitB = np.load(folder + '/bs_params.npy')
            old_trialsB = np.shape(pFitB)[0]
            print("   Loading from npy file: %i old trials" % old_trialsB)
            if do_append:
                trialB = old_trialsB # starting position for new fits
                trialsB = trialsB-old_trialsB # update index
                if trialsB > 0:
                    pFitB_old = pFitB.copy()
                    pFitB = np.concatenate((pFitB,np.zeros([trialsB, 6,12])))
                    new_trialsB = True
                else:
                    new_trialsB = False
            else:
                new_trialsB = False
                #print(pFitB)
            print(np.shape(pFitB))
            print(pFitB[0,0:4,0])
            print(pFitB[0,0:4,1])
            print(pFitB[0,0:4,2])
            print(pFitB[0,:4,3])
            print(pFitB[0,:4,4])
            print(pFitB[0,:4,5])
            print(pFitB[0,:4,6])
            np.savetxt(folder + '/MB_fits.txt', pFitB[:,0:4,0], delimiter= ', ')
        else:
            old_trialsB = 0

            pFitB = np.zeros([trialsB, 4])
            trialB = 0 # starting position for new fits
            new_trialsB = True

            
        if new_trialsB:
            print('   Running %i new trials to reach %i total' % (trialsB, old_trialsB+trialsB))

            fracChoose = 1.0
            # fraction of original number of data points for bootstrap resampling
            nChoose = int(np.floor(fracChoose*nData))
            # make into an integer number of points to select
            fails = 0
            # counter for failed fits
            

            trialB = 0
            fails = 0
            
            pGuess, func = guess_func(myData.state, myData.method)

            
            while (trialB < trialsB) and (fails < trialsB):
                if (trialsB <= 100) or (0 == trialB % (trialsB / 5)):
                    print('   %d of %d @ %s' % (trialB, trialsB, time.strftime("%H:%M:%S", time.localtime())))
                    
                inds = np.random.choice(np.arange(0, nData), nChoose, replace=True)
                xTrial = np.random.normal(np.take(xx, inds), 1)
                # we need to make sure there are no duplicate x values or the fit
                # will fail to converge
                yTrial = np.take(yy, inds)
                p = xTrial.argsort()
                
                xTrial = xTrial[p]
                yTrial = yTrial[p]

                try:
                    # pylint: disable=unbalanced-tuple-unpacking
                    pFit, cov = optimize.curve_fit(func, xTrial, yTrial, pGuess)
                except Exception:
                    print("Failed to converge")
                    fails += 1
                    continue
                
                if np.sum(np.isinf(trialB)) or abs(pFit[3] - pGuess[3]) > 200. or pFit[1] > 2e9 or pFit[1] <= 0.:
                    print('Fit params out of bounds')
                    print(pFit)
                    continue
                else:
                    pFitB[trialB, :] = pFit
                    trialB += 1

        print(pFitB[:,0:4])
        np.savetxt(folder + '/' + fitType + '_fits.txt', pFitB[:,0:4], delimiter= ', ')
            
        if do_profile:
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.TIME
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
            print(s.getvalue())

        conf = 68.2689  # confidence level for CI
        
        #print(xx)
        #print(yy)
        mincut = min(np.max(xx),np.min(xx))
        maxcut = max(np.max(xx),np.min(xx))
        
        print(mincut)
        print(maxcut)

        if do_stats:
            print("** Stats")
            fits = pFitB
            for fit in range(np.size(fits,axis=0)):
                if (-fits[fit,3] < mincut) or (-fits[fit,3] > maxcut):
                    fits[fit] = np.nan
            
            median_f = np.nanmedian(qb_factor*fits[:, 3])
            upper_f = np.nanpercentile(qb_factor*fits[:, 3], 100-(100.0-conf)/2.)
            lower_f = np.nanpercentile(qb_factor*fits[:, 3], (100.0-conf)/2.)
            
            csvPath = dataType + '/' + fitType + '_' + dataType + '.csv'
            print(csvPath)
            if not os.path.isfile(csvPath):
                with open(csvPath, 'w', newline='') as csvFile:
                    csvWriter = csv.writer(csvFile, delimiter=',')
                    csvWriter.writerow(['Run           ', 'B (G)       ','Exclude     ',
                                        'Offset      ', 'Amplitude   ',
                                        'EF (nK)     ', 'fB (kHz)    ',
                                        'median_f    ', 'lower_f     ', 'upper_f     ',
                                        'sigma_fM    ', 'sigma_fP    ',
                                        'T/TF_fit    ',
                                        'N_fits      '])
                    csvWriter.writerow(['{:<14}'.format(myData.run),
                                        '{:12.2f}'.format(myData.B),
                                        '{:12}'.format(myData.exclude),
                                        '{:12.2f}'.format(np.nanmedian(fits[:, 0])),
                                        '{:12.4f}'.format(np.nanmedian(fits[:, 1])),
                                        '{:12.2f}'.format(np.nanmedian(fits[:, 2])),
                                        '{:12.2f}'.format(np.nanmedian(qb_factor*fits[:, 3])),
                                        '{:12.2f}'.format(median_f),
                                        '{:12.2f}'.format(lower_f),
                                        '{:12.2f}'.format(upper_f),
                                        '{:12.2f}'.format(median_f - lower_f),
                                        '{:12.2f}'.format(upper_f - median_f),
                                        '{:12.2f}'.format(FD.TTFs[numDA] if numDA >= 0 else numDA),
                                        '{:12}'.format(np.size(fits,axis=0))
                                        ])
            else:
                with open(csvPath, 'a', newline='') as csvFile:
                    csvWriter = csv.writer(csvFile, delimiter=',')
                    csvWriter.writerow(['{:<14}'.format(myData.run),
                                        '{:12.2f}'.format(myData.B),
                                        '{:12}'.format(myData.exclude),
                                        '{:12.2f}'.format(np.nanmedian(fits[:, 0])),
                                        '{:12.4f}'.format(np.nanmedian(fits[:, 1])),
                                        '{:12.2f}'.format(np.nanmedian(fits[:, 2])),
                                        '{:12.2f}'.format(np.nanmedian(qb_factor*fits[:, 3])),
                                        '{:12.2f}'.format(median_f),
                                        '{:12.2f}'.format(lower_f),
                                        '{:12.2f}'.format(upper_f),
                                        '{:12.2f}'.format(median_f - lower_f),
                                        '{:12.2f}'.format(upper_f - median_f),
                                        '{:12.2f}'.format(FD.TTFs[numDA] if numDA >= 0 else numDA),
                                        '{:12}'.format(np.size(fits,axis=0))
                                        ])

# Plotting #####################################################################

        if do_plots:
            print('** Plots')
            # display vector for plotting
            numpts = 100
            xFit = np.linspace(np.min(xx), np.max(xx), numpts)
                
            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = 10
            fig_size[1] = 8
            plt.rcParams["figure.figsize"] = fig_size

            fig = plt.figure()
            #for i in range(0, np.shape(pFitB)[0]-1):

            #print(pFitB)
            fits = pFitB
            #print(k)
            #print(numDA)
            #print(np.shape(fits))
            #print(fits)
            for fit in range(np.size(fits,axis=0)):
                if (qb_factor*fits[fit,3] < mincut) or (qb_factor*fits[fit,3] > maxcut):#(fits[fit,3] < 10) or (fits[fit,3] > 1000):
                    fits[fit] = np.nan

            pGuess, func = guess_func(myData.state, myData.method)
            pMed = np.array([0.,0.,0.,0.])
            pMed[0] = np.nanmedian(fits[:, 0])
            pMed[1] = np.nanmedian(fits[:, 1])
            pMed[2] = np.nanmedian(fits[:, 2])
            pMed[3] = np.nanmedian(fits[:, 3])
            print(pMed)

            plt.plot(xFit, func(xFit, *pMed), color=colorfunc(fitType), alpha=.2)
            
            plt.axvline(x = np.nanmedian(qb_factor*fits[:, 3]), color=colorfunc(fitType), alpha=0.5)
            plt.axvline(x = np.nanpercentile(qb_factor*fits[:, 3], (100.-conf)/2.), color=colorfunc(fitType), alpha=0.5)
            plt.axvline(x = np.nanpercentile(qb_factor*fits[:, 3], 100.-(100.-conf)/2.), color=colorfunc(fitType), alpha=0.5)
        
            do_plotCI = True
            if do_plotCI:
                print('*** Plotting Confidence Band')
                iis = range(0,np.shape(fits)[0])
                ylo1 = 0*xFit
                yhi1 = 0*xFit
                for idx, xval in enumerate(xFit):
                    print('    %i of %i @ %s' % (idx, numpts,time.strftime("%H:%M:%S", time.localtime())))
                    #print(xval)
                    #print([fits[ii,0],fits[ii,1],fits[ii,2],-1*qb_factor*fits[ii,3]])
                    #FDW_qb, FCQB_FD_int
                    fvals = [func(xval,*fits[ii,:4]) for ii in iis]
                    #fvals = [func(xval,fits[ii,0],fits[ii,1],fits[ii,2],fits[ii,3]) for ii in iis] # change for bound (ii,:-2)/ QB (ii,:)
                    #print(fvals)
                    ylo1[idx] = np.nanpercentile(fvals,(100.-conf)/2.)
                    yhi1[idx] = np.nanpercentile(fvals,100.-(100.-conf)/2.)
    
                    #print(np.percentile(func(xval,*pFitB[:,:-2]),(100.-conf)/2.))
                    
                #print(np.shape(np.percentile([func(xFit,*pFitB[ii,:-2]) for ii in range(0,np.shape(pFitB)[0])])))

                plt.plot(xFit, ylo1, color=colorfunc(fitType))
                plt.plot(xFit, yhi1, color=colorfunc(fitType))
            
            plt.plot(xx, yy, '.')
            #plt.plot(xFit, func(xFit, *pGuess), 'k--')
            plt.xlim([np.nanmin(xFit),np.nanmax(xFit)])
            
            #plt.ylim([0,np.nanmax(yy)*1.1])
            plt.xlabel(myData.method + " freq (kHz)")
            plt.ylabel("Atoms remaining")
            plt.suptitle('[' + dataType + '/' + myData.run + "] " + fitType + " Bootstrap resampling")
            #plt.show()
            print(folder + fitType + '_Fit.pdf')
            plt.savefig(folder + fitType + '_Fit.pdf')
            plt.close()


            fig2 = plt.figure()
            
            fits = pFitB[:,:]
            #print(fits)

            for fit in range(np.size(fits,axis=0)):
                if (qb_factor*fits[fit,3] < mincut) or (qb_factor*fits[fit,3] > maxcut):#(fits[fit,3] < 10) or (fits[fit,3] > 1000):
                    fits[fit] = np.nan

            for i in range(0, 4): # change for B nvar+1/QB nvar+0
                ax1 = plt.subplot(2, 2, i+1)

                median_f = np.nanmedian(fits[:, i])
                #print(median_f)
                upper_f = np.nanpercentile(fits[:, i], 100-(100.0-conf)/2.)
                lower_f = np.nanpercentile(fits[:, i], (100.0-conf)/2.)

                array = fits[:, i]
                finite = array [~np.isnan(array)]
                print(finite)
                ax1.hist(finite, bins=50, alpha=0.5, color=colorfunc(fitType),
                        range=(np.nanpercentile(finite,1.0),np.nanpercentile(finite,99.0)))
                ax1.axvline(x=median_f, c=colorfunc(fitType),
                            label='BS: %.2f-%.2f+%.2f' % (median_f, median_f - lower_f, upper_f - median_f))
                ax1.axvline(x=upper_f, c=colorfunc(fitType),alpha=0.5)
                ax1.axvline(x=lower_f, c=colorfunc(fitType),alpha=0.5)
                #ax1.axvline(x=median+upperB, c='b',alpha=0.5)
                #ax1.axvline(x=median-lowerB, c='b',alpha=0.5)
                #ax1.axvline(x=median+upper95, c='r',alpha=0.5)
                #ax1.axvline(x=median-lower95, c='r',alpha=0.5)


                ax1.legend(
                    bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                    mode="expand", borderaxespad=0.)

                ax1.set_xlabel(plabels[i])

            fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
            if fitType == 'FD':
                fig2.suptitle('[' + dataType + '/' + myData.run + '] T/TF = %.2f Fit parameter distributions (%.2f %% confidence)' % (FD.TTFs[numDA] if numDA >= 0 else numDA, conf))
            else:
                fig2.suptitle('[' + dataType + '/' + myData.run + '] MB Fit parameter distributions (%.2f %% confidence)' % (conf))

            fig2.savefig(folder + fitType + '_Hists.pdf')
            plt.close()

            fig3 = plt.figure()
            for i in range(0, 4):
                for j in range(i+1, 4):
                    ax3 = plt.subplot(4-1, 4-1, (4-1)*i+j)
                    ax3.scatter(pFitB[:, j], pFitB[:, i], color=colorfunc(fitType), s=10,
                                edgecolor='none', alpha=0.12)

                    ax3.set_xlabel(plabels[j])
                    ax3.set_ylabel(plabels[i])
            fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig3.suptitle('[' + dataType + '/' + myData.run + "] Fit parameter correlations")

            fig3.savefig(folder + fitType + '_Correlations.png')
            plt.close()