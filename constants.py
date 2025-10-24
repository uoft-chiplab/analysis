from scipy.constants import c, pi, h, hbar, k as kB, m_e, mu_0, e, epsilon_0
from scipy.constants import physical_constants as pc

from math import gamma

#
# Fundamental constants
#

uatom = pc["atomic mass constant"][0]  # Atomic mass unit (kg)
a_0 = pc["Bohr radius"][0]  # Bohr radius (m)
gS = -pc["electron g factor"][0]  # Electron g-factor, NOTE THE MINUS SIGN
mu_B = pc["Bohr magneton"][0]  # Bohr magneton (J/T)

#
# Potassium-40 specific constants
#

mK = 39.96399848 * uatom  # Mass of potassium-40 (kg)
I = 4  # Nuclear spin of potassium-40

ahf = -h * 285.7308e6  # For groundstate
gI = 0.000176490  # Total nuclear g-factor  
# gJ = 2.00229421  # For groundstate measured value 
gJ = gS  # For theoretical value 

# D1 and D2 line parameters
LambdaD1 = 770.108136507e-9  # (m)
LambdaD2 = 766.700674872e-9  # (m)
NuD1, NuD2 = (c / LambdaD1, c / LambdaD2)
kD1, kD2 = (2 * pi / LambdaD1, 2 * pi / LambdaD2)
GammaD1 = 2 * pi * 6.035e6  # (Hz)
GammaD2 = 2 * pi * 6.035e6  # (Hz)

# Feshbach resonance parameters
abg = 167 * a_0 
DeltaB97 = 6.9  # (G)
DeltaB95 = 7.2  # (G)
B097 = 202.10  # (G)
B097zero = 209.115  # (G)
B095 = 224.2  # (G)

#
# Van der Waals parameters
#

C6 = 3897
r0 = 1/8**(1/2) * gamma(3/4)/gamma(5/4) * ((mK * C6)/hbar**2)**(1/4)
r0 = 60 * a_0
