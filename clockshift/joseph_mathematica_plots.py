##### Created 2025-06-09 by the chip lab
##### This file was created to recreate the theory plots Joseph T. made in mathematica for the 
##### fast spectroscopy/CS manuscript 

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from scipy.special import gamma
import sys
import os
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')
if root not in sys.path:
	sys.path.append(root)
from library import pi, h, hbar, mK, a0, paper_settings, generate_plt_styles
colors = ['#c70808','#66a61e','#000000','#1c89cb','#4b13b4']

styles = generate_plt_styles(colors, ts=0.6)

# ### plot settings
plt.rcdefaults()
plt.rcParams.update(paper_settings) # from library.py
font_size = paper_settings['legend.fontsize']
fig_width = 3.4 # One-column PRL figure size in inches
subplotlabel_font = 10

# Constants
hbar = 1.0545718e-34  # J*s
amu = 1.66053873e-27
massK40 = 39.96399848*amu   # amu
a0 = 0.52917720859e-10      # meters
aB = a0
h = 2*np.pi*hbar   

# Feshbach resonance parameters
B095 = 224.2            
B097 = 202.15           
aBG95 = 167.3 * a0        
Delta95 = 7.2             
sresAC = 1.9        
a95atB097 = 221.46296*aB
re95atB097 = 103*aB    

# Derived quantities
r0K40 = 65.02231404 * aB
abar40K = 4 * np.pi / gamma(0.25)**2 * r0K40
re0 = gamma(0.25)**4 * abar40K / (6 * np.pi**2)
# re0 = 9.59808e-9
# Energy and pole formulas
def poleER(re, aS):
    argument = np.where(aS - 2 * re > 0, aS - 2 * re, np.nan)
    valid_aS = np.where(aS > 0, aS, np.nan)
    return (1 / re - np.sqrt(argument) / (np.sqrt(valid_aS) * re))


def EdimerER(re, aS):
    return hbar**2 * poleER(re, aS)**2 / massK40

def poleTmatrix(re, aS):
    def equation(kappa):
        return 1/(aS * kappa) - (1 - (2/np.pi) * np.arctan(np.pi * kappa * re / 4))
    result = root_scalar(equation, x0=1/aS, 
                        #  x1=2.0, method='secant',
                         method='newton'
                        #  bracket=[1e-10, 10/aS]
                         )
    return result.root

def EdimerTmatrix(re, aS):
    kappa = poleTmatrix(re / aB, aS / aB)
    return hbar**2 * kappa**2 / aB**2 / massK40

def poleQuartic(re, aS):
    def equation(kappa):
        return 1/(kappa * aS) + (re * kappa)/2 - (1/(96 * np.pi**2)) * re**3 * kappa**3 - 1
    result = root_scalar(equation, x0=1/aS, 
                         x1=2.0, method='secant',
                        #  method='newton'
                        #  bracket=[1e-10, 10/aS]
                         )
    return result.root

def EdimerQuartic(re, aS):
    kappa = poleQuartic(re / aB, aS / aB)
    return hbar**2 * kappa**2 / aB**2 / massK40

# Phase shift functions
def eta_unitary(n):
    return (n + 0.5) * np.pi

myx = 1.0798111488013815

def eta_x(j, x):
    return eta_unitary(j) + x / eta_unitary(j)

def aSSqWxOverr0(j, x):
    eta = eta_x(j, x)
    return 1 - np.tan(eta) / eta

def reffSqWxOverr0(j, x):
    eta = eta_x(j, x)
    return 1 - eta**2 / (3 * (eta - np.tan(eta))**2) + 1 / (-eta**2 + eta * np.tan(eta))

def eta_f0(nf):
    return (nf + 0.5) * np.pi

def eta_f_approx(nf, ktilde):
    eta = eta_f0(nf)
    return eta + ktilde / eta + (1/(2 * eta) - 1/eta**3) * ktilde**3

def dimerkappaSWapprox(nf, eta_f):
    def eq(ktilde):
        return eta_f_approx(nf, ktilde) - eta_f
    result = root_scalar(eq, x0=1.0, 
                        #  x1=2.0, method='secant',
                         method='newton',
                        #  bracket=[1e-5, 5.0]
                         )
    return result.root

def dimerkappaSW(nf, eta_f):
    def eq(kappa_t):
        term = np.sqrt(eta_f**2 - kappa_t**2)
        return term / np.tan(term) + kappa_t
    init = dimerkappaSWapprox(nf, eta_f)
    result = root_scalar(eq, x0=init, 
                        #  x1=2.0, method='secant',
                         method='newton',
                        #  bracket=[1e-5, eta_f - 1e-5]
                        )
    return result.root

# Functions
def x_B(dB, abg, Delta):
    denom = (1 - Delta / dB)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = 1 / abg / denom
    # Optionally mask or warn where denom is near zero
    if np.any(np.isclose(denom, 0.0, atol=1e-8)):
        print("Warning: denom near zero in x_B()")
    return result

def revdW(xval):
    return re0* (1 - 2 * abar40K * xval + 2 * abar40K**2 * xval**2)

def reRes(xval, sres, abg):
    return -2 * abar40K / sres * (1 - abg * xval)**2

def re_full(dB, abg, Delta, sres):
    xval = x_B(dB, abg, Delta)
    return revdW(xval) + reRes(xval, sres, abg)

def aS13(B):
    return 1 / x_B(B - B095, aBG95, Delta95)

def re13(B):
    return re_full(B - B095, aBG95, Delta95, sresAC)

def BfieldFromAs(aS):
    return Delta95 / (1 - aS/aBG95) + B095

# Energy from kappa
def energy_from_kappa(kappa, re):
    return -hbar**2 / (massK40) * (kappa/(115*a0))**2 / h / 1e6  # MHz

# Precompute B ↔ aS13 mapping range
B_min, B_max = B095 - 10, B095 + 10
B_check = np.linspace(B_min, B_max, 500)
aS_check = np.array([aS13(B) for B in B_check])
aS_min, aS_max = np.min(aS_check), np.max(aS_check)

print(f"Valid aS13 range: {aS_min/a0:.2f} a0 to {aS_max/a0:.2f} a0")

# Generate SqW parametric data
j = 100
x_vals = np.linspace(0, 2, 200)
B_vals_sqW = []
E_vals_sqW = []

for x in x_vals:
    try:
        # Step 1: Compute effective scattering length
        aS_eff = aSSqWxOverr0(j, x)*115*a0
        if not np.isfinite(aS_eff):
            continue

        # Step 2: Invert aS13 to get B
        B = BfieldFromAs(aS_eff)
        if not np.isfinite(B):
            continue

        # Step 3: Get eta and kappa
        eta = eta_x(j, x)
        kappa = dimerkappaSW(j, eta)
        if not np.isfinite(kappa):
            continue

        # Step 4: Compute dimer energy
        E = energy_from_kappa(kappa, re0)
        if np.isfinite(E):
            B_vals_sqW.append(B)
            E_vals_sqW.append(E)
        else:
            print(f"[Loop] Energy not finite at x={x:.2f}, kappa={kappa:.4f}")
    except Exception as e:
        print(f"[Loop] Failed at x={x:.2f}: {e}")
        continue



# Dimer energy vs B for three pole models
B_range = np.linspace(B097 , B095-1, 300)
E_ER, E_TM, E_QP = [], [], []

for B in B_range:
    aS = aS13(B)
    re = re13(B)
    try:
        E_ER.append(-EdimerER(re, aS) / h / 1e6)       # MHz
        E_TM.append(-EdimerTmatrix(re, aS) / h / 1e6)
        E_QP.append(-EdimerQuartic(re, aS) / h / 1e6)
    except Exception:
        E_ER.append(np.nan)
        E_TM.append(np.nan)
        E_QP.append(np.nan)


# Plotting
myaS = 221.5 * a0
j = 100

x_vals = np.linspace(0.01, 10, 300)
reff_over_aS = [reffSqWxOverr0(j, x)/aSSqWxOverr0(j, x) for x in x_vals]
aS_kappa_vals = [aSSqWxOverr0(j, x) * dimerkappaSW(j, eta_x(j, x)) for x in x_vals]

fig_width = 3.4 # One-column PRL figure size in inches

fig, ax = plt.subplots(1,3, figsize=[fig_width*2, fig_width])
ax[0].plot(reff_over_aS, aS_kappa_vals, 'r--', label='SqW')

# Compare with poles
reOveraS_vals = np.linspace(0.01, 0.6, 200)
kappa_ER = [myaS * poleER(r * myaS, myaS) for r in reOveraS_vals]
kappa_Tmatrix = [myaS * poleTmatrix(r * myaS, myaS) for r in reOveraS_vals]
kappa_Quartic = [myaS * poleQuartic(r * myaS, myaS) for r in reOveraS_vals]

# ax.plot(reOveraS_vals, kappa_ER, label='Effective Range pole')
ax[0].plot(reOveraS_vals, kappa_Tmatrix, label='T-matrix pole')
# ax.plot(reOveraS_vals, kappa_Quartic, label='Quartic phase shift pole')

# CCC point
re_ratio = 105 / 221.5
E_dimer = 4.021e6 * 6.62607015e-34  # MHz to J
kappa_CCC = np.sqrt(massK40 * E_dimer / hbar**2) * 221.5 * a0
ax[0].plot([re_ratio], [kappa_CCC], 'ko', label='CCC')

ax[0].set_xlabel(r'$r_e / a_S$')
ax[0].set_ylabel(r'$a_S \kappa$')
# ax.set_aspect('equal', adjustable='box')
ax[0].legend()

fig, ax = plt.subplots(3,1)

# Blist = np.linspace(202, 222, 100)
ax[0].plot(B_range, aS13(B_range)/a0)
ax[0].set_ylim(0,800)

ax[1].plot(B_range, re13(B_range)/a0)
ax[1].set(ylabel = 're13/a0')

ax[2].plot(B_range, re13(B_range)/aS13(B_range))
ax[2].set(ylabel = 're13/a13')

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(B_vals_sqW, E_vals_sqW, 'r--', label='SqW')
ax.plot(B_range, E_ER, linestyle='-', label='Effective Range pole')
ax.plot(B_range, E_TM, linestyle='-.', label='T-matrix pole')
# ax.plot(B_range, E_QP, linestyle='--', label='Quartic phase shift pole')
ax.plot([B097], [-4.021], 'ko', label='CCC')
ax.set_xlabel('Magnetic Field B (G)')
ax.set_ylabel('Dimer Binding Energy (MHz)')
ax.set(
    ylim = [-4.2,0],
    xlim = [200,224]
)
# ax.set_xlim(B097 - 1, B095)
ax.legend()
plt.tight_layout()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Create figure
fig = plt.figure(figsize=(fig_width*2, fig_width*3/5))  # Wider figure for 3 columns

# Define GridSpec: 3 columns, but manually handle rows
gs = gridspec.GridSpec(3, 3, width_ratios=[1.3, 1.3, 1])  # 3x3 grid

# Plot 1: Top-left (row 0–2, col 0)
ax1 = fig.add_subplot(gs[:, 0])
ax1.plot(reff_over_aS, aS_kappa_vals, linestyle='--', color=colors[0], label='SqW')
ax1.plot(reOveraS_vals, kappa_Tmatrix, linestyle='-' , color=colors[1], label='T-matrix pole')
ax1.plot([re_ratio], [kappa_CCC], marker='.', color=colors[2], label='CC')
ax1.set(
    xlabel = r'$r_e/a_s$',
    ylabel = r'$\kappa a_s$',
    xlim = [0, 0.6],
    ylim = [0.8, 2]
)

# ax1.set_title("Plot 1")

# Plot 2: Top-middle (row 0–2, col 1)
ax2 = fig.add_subplot(gs[:, 1])
sqw, = ax2.plot(B_vals_sqW, E_vals_sqW, linestyle='--', color=colors[0], label='SqW')
t, = ax2.plot(B_range, E_TM, linestyle='-', color=colors[1], label='T-matrix pole')
zerorange, = ax2.plot(B_range, E_ER, linestyle='-.', color=colors[4], label='Zero Range pole')
cc, = ax2.plot([B097], [-4.021], marker='.', color=colors[2], label='CC')
ax2.set(
    ylim = [-4.5,0],
    xlim = [201, 224],
    xlabel = 'Magnetic Field (G)',
    ylabel = 'Energy (MHz)'
)
ax1.legend(frameon=False, handles=[sqw, t, zerorange, cc])

sqw_dict = {
    'Energy (MHz)': E_vals_sqW,
    'Magnetic Field (G)': B_vals_sqW, 
}


# Create a DataFrame
df = pd.DataFrame(sqw_dict)

# Save to Excel file
df.to_excel('sqw_theory_line.xlsx', index=False)

# Plot 3a, 3b, 3c: stacked in column 3
ax3a = fig.add_subplot(gs[0, 2])
ax3a.plot(B_range, aS13(B_range)/a0,linestyle='-', color=colors[3])
ax3a.set(
    ylabel = r'$a_{13}/a_0$',
    ylim = [150,800],
    yticks = [200, 600],
    xlim = [202, 222]
)
# ax3a.set_title("Plot 3a")

ax3b = fig.add_subplot(gs[1, 2],sharex = ax3a)
ax3b.plot(B_range, re13(B_range)/a0,linestyle='-',color=colors[3])
ax3b.set(
    ylabel = r'$r_{e,13}/a_0$',
    ylim = [103, 117]
)
# ax3b.set_title("Plot 3b")

ax3c = fig.add_subplot(gs[2, 2],sharex = ax3a)
ax3c.plot(B_range, re13(B_range)/aS13(B_range),linestyle='-',color=colors[3])
ax3c.set_xlabel("Magnetic Field (G)")
ax3c.set(
    ylabel = r'$r_{e.13}/a_{13}$',
    ylim = [0,0.5],
    yticks = [0, 0.4]
)
for ax in [ax3a, ax3b]:
    ax.tick_params(labelbottom=False)

labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
axes = [ax1, ax2, ax3a, ax3b, ax3c]

for i, ax in enumerate(axes):
    # Place label just outside the top-left of each subplot:
    ax.text(
        -0.05, 1.01, labels[i],  # x < 0 moves left outside, y > 1 moves above top
        transform=ax.transAxes,
        fontsize=10,
        # fontweight='bold',
        va='bottom',
        ha='right'
    )

output_dir = os.path.join(proj_path, '\manuscript\manuscript_figures')
os.makedirs(output_dir, exist_ok=True)
fig.tight_layout()

# Now save the figure
fig.savefig('mathematica_to_python.pdf',bbox_inches='tight', dpi=300)
# Layout adjustment

