import numpy as np 
import matplotlib.pyplot as plt 
import os 
os.chdir("/Users/maggie/Documents/ChipAnalysis/")
import pandas as pd 
from scipy.integrate import trapezoid, simpson
from scipy.optimize import curve_fit
from data_class import Data
from library import FreqMHz, FermiEnergy, OmegaRcalibration, hbar, h, pi

import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["hotpink", "cornflowerblue", "yellowgreen"]) 

# define constants 

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
verb = 1
# saves figures to folder if True
save_figs = 0

data_folder = 'clockshift/old_data'
metadata_file = os.path.join(data_folder, "ContactMeasurementsSummary.xlsx")
metadata = pd.read_excel(metadata_file)

columns1 = ["mean_w", "EF", "res_freq", "Delta_cutoff", "c9_bg"]
columns2 = ["sum_rule", "a", "e_a", "contact"]
metadata[columns1] = [pd.Series]*len(columns1)
metadata[columns2] = [pd.Series]*len(columns2)

def get_metadata(filename):
    date = filename.split("_")[0]
    letter = filename.split("_")[1].split(".")[0]

    run_metadata = metadata.loc[metadata.date==date].loc[metadata.letter==letter]
    i = run_metadata.index[0]

    if np.isnan(run_metadata["Omega_R"].values[0]):
        return None

    return i

def scale_x(run, i):
    run_metadata = metadata.loc[i]
    B = run_metadata["field"]
    N = run_metadata["TShotsN"]/2 # divide by 2 to get N/spin
    t_pulse = run_metadata["time"] # us
    pulse_type = run_metadata["pulse"]
    Omega_R = run_metadata["Omega_R"] / 1e3 # kHz -> MHz

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
    Delta_cutoff = -2/t_pulse * (1 + int(pulse_type == "blackman")) # extra factor of 2 in Fourier transform width if blackman, 1 if square
    c9_bg  = np.mean(run.data['c9'].loc[run.data['Delta'] < Delta_cutoff])

    run.data["Transfer"] = (c9_bg - run.data['c9'])/(t_pulse*c9_bg)
    run.data["Scaled transfer"] = run.data["Transfer"] * (EF*h)/(hbar * pi * Omega_R**2) # EF in MJ, Omega_R in MHz, transfer in MHz

    # save calculated metadata values
    metadata.loc[i, ["mean_w", "EF", "res_freq", "Delta_cutoff", "c9_bg"]] = [mean_w, EF, res_freq, Delta_cutoff, c9_bg]

    # also calcualte average y values
    run.group_by_mean("Delta")
    x, y, ey = np.array(run.avg_data['Delta']), np.array(run.avg_data["Scaled transfer"]), np.array(run.avg_data["e_Scaled transfer"])

    return x, y, ey

def f(x, a):
    return a* x**(-3/2)

def sum_rule(x, y):
    i = 0
    if verb:
        print("all points")
        print(f"trapezoidal: {trapezoid(y, x):.3f}\n"+
            f"simpson: {simpson(y, x):.3f}")

        print("\nwith initial point removed")
        print(f"trapezoidal: {trapezoid(y[i:], x[i:]):.3f}\n"+
            f"simpson: {simpson(y[i:], x[i:]):.3f}")
    
    area = simpson(y[i:], x[i:])
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

        xfit = np.linspace(EFmin, EFmax,1000)

        plt.plot(xfit, f(xfit, *popt), linestyle="-", marker="", label="fit")
        plt.xlabel(r"$\Delta$ [EF]")
        plt.ylabel(r'$\tilde\Gamma$')

    return a, e_a, C


fit_err_dat = []

# '2023-08-16_J.dat', '2023-08-16_F.dat' -- no frequency column?
# '2023-08-03_C.dat', '2023-09-08_B.dat' -- looks funny...

exclude = ['2023-08-16_J.dat', '2023-08-16_F.dat', '2023-08-03_C.dat', '2023-09-08_B.dat']

for filename in os.listdir("clockshift/old_data"): #["2023-07-28_F.dat"]
    print(f"\n---------------------------{filename}---------------------------")

    try:
        run = Data(filename, path=data_folder)
        i = get_metadata(filename)
    except:
        print("metadata not found")
        continue

    if i == None:
        print("skipping varying VVA")
        continue

    try:
        x, y, ey = scale_x(run, i) # x = Delta, y = avg scaled transfer
        if x[-1] <= 3:
            print("dataset out of fit range")
            continue
        if verb:
            plt.figure()
            plt.errorbar(x, y, ey)
            plt.axvline(metadata.loc[i, "Delta_cutoff"], linestyle="--", color="lightgrey",label=r"$\Delta$ cutoff")

    except:
        print("error reading file")
        continue

    try:
        area = sum_rule(x, y)
        a, e_a, C = fit(x, y, ey)

        metadata.loc[i, columns2] = [area, a, e_a, C]
    except:
        plt.savefig(data_folder+"/filename[:-3]"+".jpeg")
        fit_err_dat.append(filename)
        print("fit error")
        continue

    if verb:
        # draw table of attributes
        col_labels = [filename]
        row_labels = ['Omega_R','time','EF','sum_rule','a','contact', 'Delta_cutoff']
        table_vals = [[f"{(metadata.loc[i, row_label]):.2f}"] for row_label in row_labels]

        row_labels += ["pulse"]
        table_vals += [[metadata.loc[i, "pulse"]]]

        #plotting
        my_table = plt.table(cellText=table_vals,
                            colWidths=[0.35],
                            rowLabels=row_labels,
                            colLabels=col_labels,
                            bbox = [1.28, 0.1, 0.3, 0.1*len(table_vals)])

        # also add a legend
        plt.legend()

    if save_figs:
        plt.savefig(f"{data_folder}/{filename[:-4]}.jpeg", bbox_inches="tight")


