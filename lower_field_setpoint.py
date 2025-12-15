import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Vout(Vin, Vc, R1, R2, R3):
    return (Vin*R2*R3 + Vc*R1*R2)/(R1*R2 + R1*R3 + R2*R3)

def R1(Vin, Vout, R2, R3):
    return (R2*(Vin/Vout*R3 - R3)/(R2 + R3))

R3 = 75000
Vin = 2.5
Vc = [10,0,-10]
Voutmeas = [0.675, 0.616, 0.557] # Measured from long ago
VoutPerVc = np.diff(Voutmeas).mean()/np.diff(Vc).mean()
print(f'VcperVout = {VoutPerVc} V/V')

Bfields = [215, 195, 175] # G
def line(x, m, b):
    return m*x + b

popt, pcov = curve_fit(line, Bfields, Voutmeas)
def Vout_for_B(B):
    return popt[0] * B  + popt[1]

fig, ax = plt.subplots()
Bs = np.linspace(140, 235, 100)
special_Bs = np.array([140, 160,195])
ax.plot(Bs, Vout_for_B(Bs), '-')
for B in special_Bs:
    Vo = Vout_for_B(B)
    ax.plot(B, Vo, 'o', label = f'Vo = {Vo:.3f} V at B={B} G')
ax.set(xlabel='Magnetic Field (G)', ylabel='Vout (V)', title='Estimated Vout vs Bfield')
ax.legend()

chosen_Bs = [140, 150, 160]
fig, ax = plt.subplots()
for B in chosen_Bs:
    chosen_Vout = Vout_for_B(B)
    print(f'At B={B} G, estimated Vout={chosen_Vout:.3f} V')
    R2_vals = np.arange(10, 2000, 100)
    R1_vals = R1(Vin, chosen_Vout, R2_vals, R3)
    chosen_R2s = np.array([10,
                           50,
                           75,
                           93,
                           100,
                           130,
                           138.5, 
                           250,
                           350,
                           500,
                           1000,
                           1200,
                           1270,
                           1820,
                           2500,
                           5000,
                            9000,
                            10000])
                        
    chosen_R1s = R1(Vin, chosen_Vout, chosen_R2s, R3)
    
    for r2, r1 in zip(chosen_R2s, chosen_R1s):
        print(f'For R2={r2} Ohms, R1={r1:.1f} Ohms')

    ax.plot(R2_vals, R1_vals, label=f'B={B}G')
    ax.plot(chosen_R2s, chosen_R1s, 'o')
    ax.set(xlabel='R2 (Ohms)', ylabel='R1 (Ohms)', 
        title=f'R1 vs R2')
ax.legend()

# R2_check = 130
# # double check with Vout function and particular R2, R1 values
# R1_check = R1(Vin, Vout_225G, R2_check, R3)
# print(f'For R2={R2_check} Ohms, calculated R1={R1_check} Ohms for225 G center')

# R2_vals = np.arange(100, 2000, 100)
# R1_vals = R1(Vin, Vout_220G, R2_vals, R3)

# fig, ax = plt.subplots()
# ax.plot(R2_vals, R1_vals, color='green')
# ax.set(xlabel='R2 (Ohms)', ylabel='R1 (Ohms)', title='R1 vs R2 for Vout=0.690V with Vin=2.5V and R3=75kOhm')

# # double check with Vo# double check with Vout function and particular R2, R1 values
# R1_check = R1(Vin, Vout_220G, R2_check, R3)
# print(f'For R2={R2_check} Ohms, calculated R1={R1_check} Ohms for 220 G center')

# R2_vals = np.arange(100, 2000, 100)
# R1_vals = R1(Vin, Vout_215G, R2_vals, R3)

# fig, ax = plt.subplots()
# ax.plot(R2_vals, R1_vals, color='green')
# ax.set(xlabel='R2 (Ohms)', ylabel='R1 (Ohms)', title='R1 vs R2 for Vout=0.675V with Vin=2.5V and R3=75kOhm')
# # double check with Vout function and particular R2, R1 values
# R1_check = R1(Vin, Vout_215G, R2_check, R3)
# print(f'For R2={R2_check} Ohms, calculated R1={R1_check} Ohms for 215 G center')