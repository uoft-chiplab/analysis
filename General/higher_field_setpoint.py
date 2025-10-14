import numpy as np
import matplotlib.pyplot as plt

def Vout(Vin, Vc, R1, R2, R3):
    return (Vin*R2*R3 + Vc*R1*R2)/(R1*R2 + R1*R3 + R2*R3)

def R1(Vin, Vout, R2, R3):
    return (R2*(Vin/Vout*R3 - R3)/(R2 + R3))

R3 = 75000
Vin = 2.5
Vout_225G = 0.705 # estimated Vout for 225 G centering
Vout_220G = 0.690
Vout_215G = 0.675
R2_vals = np.arange(100, 2000, 100)
R1_vals = R1(Vin, Vout_225G, R2_vals, R3)

fig, ax = plt.subplots()
ax.plot(R2_vals, R1_vals, color='green')
ax.set(xlabel='R2 (Ohms)', ylabel='R1 (Ohms)', title='R1 vs R2 for Vout=0.705V with Vin=2.5V and R3=75kOhm')

R2_check = 130
# double check with Vout function and particular R2, R1 values
R1_check = R1(Vin, Vout_225G, R2_check, R3)
print(f'For R2={R2_check} Ohms, calculated R1={R1_check} Ohms for225 G center')

R2_vals = np.arange(100, 2000, 100)
R1_vals = R1(Vin, Vout_220G, R2_vals, R3)

fig, ax = plt.subplots()
ax.plot(R2_vals, R1_vals, color='green')
ax.set(xlabel='R2 (Ohms)', ylabel='R1 (Ohms)', title='R1 vs R2 for Vout=0.690V with Vin=2.5V and R3=75kOhm')

# double check with Vo# double check with Vout function and particular R2, R1 values
R1_check = R1(Vin, Vout_220G, R2_check, R3)
print(f'For R2={R2_check} Ohms, calculated R1={R1_check} Ohms for 220 G center')

R2_vals = np.arange(100, 2000, 100)
R1_vals = R1(Vin, Vout_215G, R2_vals, R3)

fig, ax = plt.subplots()
ax.plot(R2_vals, R1_vals, color='green')
ax.set(xlabel='R2 (Ohms)', ylabel='R1 (Ohms)', title='R1 vs R2 for Vout=0.675V with Vin=2.5V and R3=75kOhm')
# double check with Vout function and particular R2, R1 values
R1_check = R1(Vin, Vout_215G, R2_check, R3)
print(f'For R2={R2_check} Ohms, calculated R1={R1_check} Ohms for 215 G center')