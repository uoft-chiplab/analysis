#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1, jv
#%%
# Generate input array
x = np.linspace(0, 20, 500)

# Calculate Bessel functions
y0 = j0(x)
y1 = j1(x)

# Plot results
plt.plot(x, y0, label='$J_0(x)$')
plt.plot(x, y1, label='$J_1(x)$')
plt.grid(True)
plt.legend()
plt.show()
# %%
#Under the Raman-Nath approximation, the periodic potential is directly imprinted on the phase of the atomic wave function and the population of atoms in n th diffraction order yields
m = 6.636e-26 # 1.44316e-25 # kg for rb 6.636e-26 #kg for k #40 #amu 
k_L = 2 * np.pi / 1064e-9  # Wave number for 1064 nm light
hbar = 1.0545718e-34  # Reduced Planck's constant in J*s
E_r = hbar**2 * k_L**2 / (2 * m)  # Recoil energy

Deltat_0 = 1e-3  # shortest pulse duration in seconds to scatter all atoms into high diffraction orders
V_L = 4.81*hbar/Deltat_0

def P_n(n, V_0_recoil, omega_r, Deltat_0):
    """
    Calculates diffraction population.
    V_0_recoil is the potential depth scaled in units of recoil energy E_r.
    """
    # beta = (V_0 * E_r / hbar) * delta_t = V_0 * omega_r * delta_t
    beta = V_0_recoil * omega_r * Deltat_0
    return jv(n, beta / 2)**2

# %%
# 1. Experimental Parameters (Example: Rb-87 D2 line)
Deltat_0 = 4e-6   # pulse time (s)
omega_r =  E_r / hbar   # Recoil frequency in rad/s 

# 2. Define the x-axis range in units of Recoil Energy (0 to 60 E_r)
V_0_range = np.linspace(0, 60, 500)

# 3. Generate the plot
plt.figure(figsize=(8, 5))
plt.plot(V_0_range, P_n(0, V_0_range, omega_r, Deltat_0), label='$n=0$', lw=2)
plt.plot(V_0_range, P_n(1, V_0_range, omega_r, Deltat_0), label='$n=1$', lw=2)
plt.plot(V_0_range, P_n(2, V_0_range, omega_r, Deltat_0), label='$n=2$', lw=2)

# 4. Graph Formatting
plt.xlabel(f'Lattice Depth $V_0$ ($E_r$)')
plt.ylabel(f'Population $P_n$')
plt.title(f'Raman-Nath Diffraction Populations ($\Delta t_0 = {Deltat_0*1e6:.1f}\,\mu$s)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
alpha = E_r / hbar * Deltat_0

def de(n, c_n_minus, c_n, c_n_plus,V_0_recoil):
    beta = V_0_recoil * omega_r * Deltat_0
    return alpha*n**2/Deltat_0 *c_n + beta/4/Deltat_0 * (c_n_minus + 2*c_n + c_n_plus)
# %%
