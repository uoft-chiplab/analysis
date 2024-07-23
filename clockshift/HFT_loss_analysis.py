
import numpy as np 
import matplotlib.pyplot as plt 
import os 
os.chdir("/Users/maggie/Documents/ChipAnalysis/")
import pandas as pd 
from scipy.integrate import trapezoid, simpson, cumulative_simpson
from scipy.optimize import curve_fit
from data_class import Data
from library import FreqMHz, FermiEnergy, OmegaRcalibration, hbar, h, pi

import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["hotpink", "cornflowerblue", "yellowgreen"]) 

# define constants 
gain = 1
SPA = {"blackman":0.31, "square":1}

# b
F1 = 9/2
mF1 = -7/2
# c
F2 = 9/2
mF2 = -5/2

# fit starting EF
EFmin = 2
EFmax = 10

# prints and displays fit results if True
# mostly turned off for debugging purposes
verb = 0
# saves figures to folder if True
save_figs = 0
# file to save text to
save_file = None


# get interpolated function returning Rabi frequency, given VVA
OmegaR_interp = OmegaRcalibration()

data_folder = 'clockshift/old_data'
metadata_file = os.path.join(data_folder, "ContactMeasurementsSummary.xlsx")
metadata = pd.read_excel(metadata_file)

if save_file:
    save_file =  open(os.path.join(data_folder, save_file), "w") 

columns1 = ["mean_w", "EF", "res_freq", "Delta_cutoff", "c9_bg"]
columns2 = ["sum_rule", "a", "e_a", "contact"]
metadata[columns1] = [None]*len(columns1)
metadata[columns2] = [None]*len(columns2)

def get_metadata(filename):
    date = filename.split("_")[0]
    letter = filename.split("_")[1].split(".")[0]

    run_metadata = metadata.loc[metadata.date==date].loc[metadata.letter==letter]
    i = run_metadata.index[0]

    return i

def scale_x(run, i):
    run_metadata = metadata.loc[i]
    B = run_metadata["field"]
    N = run_metadata["TShotsN"]/2 # divide by 2 to get N/spin
    t_pulse = run_metadata["time"] # us
    pulse_type = run_metadata["pulse"]

    VVA = run_metadata['VVA']

    if VVA == "varying":
        try:
            VVA = run.data['vva']
        except:
            VVA = run.data['VVA']

        if np.min(VVA) < 1.1 or np.max(VVA) > 10:
            print("VVA out of interpolation range")
            return

            # # make dict of VVA to Vpps from VVArule
            # VVA_rule_str = np.fromstring(run_metadata['VVArule'][1:-1], sep=",")
            # VVA_keys = VVA_rule_str[0::2]
            # VVA_entries = VVA_rule_str[1::2]

            # VVA_rule = {}
            # for i in range(len(VVA_keys)):
            #     VVA_rule[VVA_keys[i]] = VVA_entries[i]

            # # convert Vpp to OmegaR with calibration value from OmegaR_calibration in library.py
            # Vpps = np.array([VVA_rule[vva] for vva in VVA])
            # OmegaR = SPA[pulse_type] * gain * 27.5833*Vpps / 1e3
    
    OmegaR = SPA[pulse_type] * gain * OmegaR_interp(VVA) / 1e3  # kHz- > MHz
                                               
    # calculate values
    ws = np.fromstring(run_metadata["trapfreqs"][1:-1], sep=",") # wx, wy, wz in MHz
    mean_w = 2*pi*np.prod(ws)**(1/3) # mean trapping freq in MHz
    EF = FermiEnergy(N, mean_w)/(h*1E6) # J -> MHz
    res_freq = abs(FreqMHz(B, F1, mF1, F2, mF2)) # MHz

    # rescale x axis
    try:
        freq = run.data["freq"]
    except:
        try:
            freq = run.data["#2"]
        except:
            freq = run.data["dummy"]

    run.data["Delta"] = (freq - res_freq)/EF

    # calculate background value
    c9 = run.data['c9']
    Delta_cutoff = -2/t_pulse * (1 + int(pulse_type == "blackman")) # extra factor of 2 in Fourier transform width if blackman, 1 if square
    c9_bg  = np.mean(c9.loc[run.data['Delta'] < Delta_cutoff])
   
    run.data["Transfer"] = (c9_bg - c9)/(t_pulse*c9_bg)
    run.data["Scaled transfer"] = run.data["Transfer"] * (EF*h)/(hbar * pi * OmegaR**2) # EF in MJ, Omega_R in MHz, transfer in MHz
    run.data['OmegaR'] = OmegaR

    # save calculated metadata values
    metadata.loc[i, ["mean_w", "EF", "res_freq", "Delta_cutoff", "c9_bg"]] = [mean_w, EF, res_freq, Delta_cutoff, c9_bg]

    # also calcualte average y values
    run.group_by_mean("Delta")
    x, y, ey = np.array(run.avg_data['Delta']), np.array(run.avg_data["Scaled transfer"]), np.array(run.avg_data["e_Scaled transfer"])

    return x, y, ey

def f(x, a):
    return a* x**(-3/2)

def sum_rule(x, y):
    i = 0 # data point at lower bound of integration  

    if verb:
        print(f"trapezoidal= {trapezoid(y=y, x=x):.3f}\n"+
            f"simpson: {simpson(y=y, x=x):.3f}")
    if save_file:
        save_file.write(f"Integration results\ntrapezoidal:\t\t{trapezoid(y=y, x=x):.3f}\n"+
            f"simpson=\t\t{simpson(y=y, x=x):.3f}")

    area = trapezoid(y=y[i:], x=x[i:])
    return area

def fit(x, y, ey, p0=[1]):
    imin = np.argmin(np.abs(x-EFmin))
    imax = np.argmin(np.abs(x-EFmax))
    popt, pcov = curve_fit(f, x[imin:imax], y[imin:imax], p0, sigma=ey[imin:imax], bounds=[0,np.inf])
    perr = np.sqrt(np.diag(pcov))
    a, e_a = popt[0], perr[0]
    C  = a*pi**2 * np.sqrt(2)

    if verb: 
        print(f"\nfit results (with bound)\nA={a:.3f} \pm {e_a:.3f}")
        print(f"\ncontact={C:.2f}")
    if save_file:
        save_file.write(f"\nfit results (with bound)\nA=\t\t\t{a:.3f} \pm {e_a:.3f}\ncontact=\t\t{C:.2f}")
    return a, e_a, C, imin, imax

def plot(x, y, ey, a, i, imin, imax, filename):
    fig, axs = plt.subplots(2, 1, sharex=True, height_ratios=[2,1])

    # plot data
    axs[0].errorbar(x, y, ey)
    axs[0].axvline(metadata.loc[i, "Delta_cutoff"], linestyle="--", color="lightgrey",label=r"$\Delta$ cutoff")
    # plot fit
    xfit = np.linspace(EFmin, EFmax,1000)
    axs[0].plot(xfit, f(xfit, a), linestyle="-", marker="", label="fit", zorder=5)
    # plot residuals
    axs[1].axhline(0, linestyle="--", color="lightgrey")
    axs[1].errorbar(x[imin:imax], f(x[imin:imax], a)-y[imin:imax])
    
    # label axes
    axs[1].set_xlabel(r"$\Delta$ [EF]")
    axs[0].set_ylabel(r'$\tilde\Gamma$')
    axs[1].set_ylabel("residuals")
    # draw table of attributes
    col_labels = [filename]
    row_labels = ['Omega_R','time','EF','sum_rule','a','contact', 'Delta_cutoff']
    table_vals = [[f"{(metadata.loc[i, row_label]):.2f}"] for row_label in row_labels]
    row_labels += ["pulse", "VVA"] # add pulse type separately since dtype is str
    table_vals += [[metadata.loc[i, "pulse"]], [metadata.loc[i, "VVA"]]]

    #plotting
    axs[0].table(cellText=table_vals,
                colWidths=[0.35],
                rowLabels=row_labels,
                colLabels=col_labels,
                bbox = [1.28, 0.1, 0.3, 0.1*len(table_vals)])

    # also add a legend
    axs[0].legend()

    if save_figs:
        plt.savefig(f"{data_folder}/{filename[:-4]}.jpeg", bbox_inches="tight")

fit_err_dat = []

exclude = ['2023-08-16_J.dat', '2023-08-16_F.dat', # varying time
           '2023-08-21_E.dat', '2023-08-30_J.dat', '2023-08-30_I.dat', '2023-08-16_J.dat', '2023-08-22_J.dat', '2023-08-16_F.dat', '2023-08-23_H.dat', '2023-08-23_H.dat', # varying time
           '2023-08-03_C.dat', '2023-09-08_B.dat', # looks funny
           '2023-09-14_L.dat', '2023-09-13_N2.dat', '2023-08-08_F2.dat', '2023-09-14_J.dat', '2023-09-18_E.dat', '2023-09-13_N.dat', # no metadata
           '2023-09-13_M.dat', '2023-09-13_G.dat', '2023-09-18_E2.dat', '2023-09-13_G3.dat', '2023-09-13_G2.dat', '2024-06-12_K_e.dat'] # no metadata


for filename in os.listdir("clockshift/old_data"):
    if filename in exclude or ".dat" not in filename:
        continue 

    print(f"\n---------------------------{filename}---------------------------")
    if save_file:
        save_file.write(f"\n---------------------------{filename}---------------------------\n")
    
    # open file
    try:
        run = Data(filename, path=data_folder)
        i = get_metadata(filename)
    except:
        print("metadata not found")
        continue

    if metadata.loc[i, "time"] == "varying":
        print("varying time")
        continue

    # scale detuning & calculate transfer
    x, y, ey = scale_x(run, i) # x = Delta, y = avg scaled transfer
    if x[-1] <= 2:
        print("dataset out of fit range")
        continue

    # calculate sum rule and fit
    try:
        area = sum_rule(x, y)
        a, e_a, C, imin, imax = fit(x, y, ey)

        metadata.loc[i, columns2] = [area, a, e_a, C]
    
    except:
        print("fit error")
        continue

    # plot
    if verb or save_figs:
        plot(x, y, ey, a, i, imin, imax, filename)

if save_file:
    save_file.close()

plt.figure()
A = np.array(metadata["sum_rule"])
A = A[A != None]
plt.plot(A)
plt.ylabel("Sum rule")
plt.yscale("log")
plt.axhline(0.5, linestyle="--")

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:01:18 2024

@author: coldatoms
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from data_class import Data
from library import FreqMHz, hbar, h, pi

def FermiEnergy(n, w):
	return hbar * w * (6 * n)**(1/3)

{wx, wy, wz}
mean_w = (wx*wy*wz)**(1/3)

ws = [int(w) for w in w_list]

run_metadata['mean_w'] = run.metadata['trapfreqs']
run_metadata['EF'] = FermiEnergy(run_metadata['TShotsN']/2,
								 run_metadata['mean_w']) 

run.data['Delta'] = run.data['detuning']/run_metadata['EF'].values[0]

c9_bg = mean(run.data['c9'].loc[run.data['Delta'] < run_metadata['EF'].values[0]])







