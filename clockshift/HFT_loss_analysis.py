
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd 
from scipy.integrate import trapezoid, simpson, cumulative_simpson
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["hotpink", "cornflowerblue", "yellowgreen"]) # make plot colors pretty
os.chdir("/Users/maggie/Documents/ChipAnalysis/") # change this to analysis/ path
from data_class import Data
from library import FreqMHz, FermiEnergy, OmegaRcalibration, ChipKaiser, ChipBlackman
from library import hbar, h, pi

# define constants 

# b
F1 = 9/2
mF1 = -7/2
# c
F2 = 9/2
mF2 = -5/2

# fit EF range
EFmin = 2
EFmax = 8

# ended up setting this to constant
Delta_cutoff = -1.5

# fit function
f = lambda x, a: a* x**(-3/2)

# saves figures to folder if True
save_figs = True
# filename to save results to. If None, no file is saved. Save path is clockshift/data/HFT_loss_results/
save_file = "HFT_loss_results.csv"
# folder to read data from
data_folder = 'clockshift/old_data'
# specify when data was taken, as the file format differs
old_data = "old" in data_folder

try:
    os.mkdir(data_folder+"/HFT_loss_results")
except:
    print('overwriting previous results')

# headers to be used
if old_data:
    metadata_file = os.path.join(data_folder, "ContactMeasurementsSummary.xlsx")
    new_columns = ["mean_w", "EF", "res_freq", "Delta_cutoff", "c9_bg", "sum_rule", "a", "e_a", "contact"]
else:
    metadata_file = os.path.join('clockshift', "metadata_file.xlsx")
    new_columns = ["Delta_cutoff", "c9_bg", "sum_rule", "a", "e_a", "contact"]

metadata = pd.read_excel(metadata_file)
metadata[new_columns] = [None]*len(new_columns)

VVAs, Vpps = np.loadtxt("VVAtoVpp.txt", unpack=True)
VpptoOmegaR = 27.5833 # kHz
VVAtoVpps = lambda x: np.interp(x, VVAs, Vpps)

def VVAtoVpp(VVA):
    """ adapted from from sumrule_analysis
    """
    try:
        Vpp = Vpps[np.argwhere(5.2==VVAs)]
    except:
        raise ValueError("VVA value {} not in VVAtoVpp.txt".format(VVA))
    return float(Vpp)

def SPA(pulsetype, run, gain):
    """from sumrule_analysis
    """
    if pulsetype == 'KaiserOffset' or pulsetype == 'BlackmanOffset':
        run.data['offset'] = 0.0252/VVAtoVpp(10)* \
                                (run.data['vva'].apply(VVAtoVpp) )
        run.data['Vpp'] = gain * (run.data['vva'].apply(VVAtoVpp) 
                        - run.data['offset'])

    if pulsetype == 'Blackman' or pulsetype == 'blackman':
        return np.sqrt(0.31) 
    elif pulsetype == 'Kaiser':
        return np.sqrt(0.3*0.92)
    elif pulsetype == 'square':
        return 1
    elif pulsetype == 'BlackmanOffset':
        xx = np.linspace(0,1,1000)
        # integrate the square, and sqrt it
        return np.sqrt(run.data.apply(lambda x: 
            np.trapz((ChipBlackman(xx)*(1-x['offset']/x['Vpp']) \
                + x['offset']/x['Vpp'])**2, x=xx), axis=1))
    elif pulsetype == 'KaiserOffset':
        xx = np.linspace(0,1,1000)
        # integrate the square, and sqrt it
        return np.sqrt(run.data.apply(lambda x: 
            np.trapz((ChipKaiser(xx)*(1-x['offset']/x['Vpp']) \
                + x['offset']/x['Vpp'])**2, x=xx), axis=1))
    else:
        ValueError("pulsetype not a known type")
        return None

def get_metadata(filename):
    """ returns index of filename in metadata dataframe"""
    if old_data:
        # index by date and letter separately
        date = filename.split("_")[0]
        letter = filename.split("_")[1].split(".")[0]
        run_metadata = metadata.loc[metadata.date==date].loc[metadata.letter==letter]

    else:
        # index by filename, with .dat removed
        run_metadata = metadata.loc[metadata.filename==filename.split(".")[0]] 

    i = run_metadata.index[0]

    return i

def scale_x_old(run, i):
    # get metadata of run
    run_metadata = metadata.loc[i]

    B = run_metadata["field"] # G
    N = run_metadata["TShotsN"]/2 # divide by 2 to get N/spin
    t_pulse = run_metadata["time"] # us
    pulse_type = run_metadata["pulse"]
    gain = 1

    VVA = run_metadata['VVA']

    if VVA == "varying":
        try:
            VVA = run.data['vva']
        except:
            VVA = run.data['VVA']

        # make dictionary from VVA rule
        VVA_rule_str = np.fromstring(run_metadata['VVArule'][1:-1], sep=",")
        VVA_keys = VVA_rule_str[0::2]
        VVA_entries = VVA_rule_str[1::2]

        VVA_rule = {}
        for i in range(len(VVA_keys)):
            VVA_rule[VVA_keys[i]] = VVA_entries[i]

        # convert Vpp to OmegaR with calibration value from OmegaR_calibration in library.py
        run.data['Vpps'] = [VVA_rule[vva] for vva in VVA]
        run_SPA = SPA(pulse_type, run, gain)

        OmegaR = 2*pi* run_SPA*gain* VpptoOmegaR *run.data['Vpps']/1e3 # kHz- > MHz
    else:
        OmegaR = run_metadata["Omega_R"]/1e3  # kHz- > MHz

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

    # calculate detuning
    run.data['OmegaR'] = OmegaR
    run.data["Delta"] = (freq - res_freq)/EF

    # calculate transfer
    c9 = run.data['c9']
    c9_bg  = np.mean(c9.loc[run.data['Delta'] < Delta_cutoff])
    run.data["Transfer"] = (c9_bg - c9)/(t_pulse*c9_bg)
    run.data["Scaled transfer"] = run.data["Transfer"] * (EF*h)/(hbar * pi * OmegaR**2) # EF in MJ, Omega_R in MHz, transfer in MHz
    
    # calculate average y values
    run.group_by_mean("Delta")
    x, y, ey = np.array(run.avg_data['Delta']), np.array(run.avg_data["Scaled transfer"]), np.array(run.avg_data["e_Scaled transfer"])
    
    # list contains values to save to metadata df
    return x, y, ey, [mean_w, EF, res_freq, Delta_cutoff, c9_bg]

def scale_x(run, i):
    # get metadata of run
    run_metadata = metadata.loc[i]

    EF = run_metadata["EF"] # MHz
    t_pulse = run_metadata["trf"]*1e6 # us
    pulse_type = run_metadata["pulsetype"] 
    res_freq = run_metadata["res_freq"] #MHz
    gain = run_metadata['gain']
    bad_indices = run_metadata['remove_indices']

    # remove indicies if list is not empty
    if not np.sum(np.isnan(bad_indices)):
        run.data.drop(bad_indices)

    # calcualte rabi frequency
    run_VVAs = run.data['vva']
    run_SPA = SPA(pulse_type, run, gain)
    run_Vpps = VVAtoVpps(run_VVAs)
    OmegaR = 2*pi* VpptoOmegaR * run_Vpps * run_SPA * gain / 1e3  # kHz- > MHz

    # calculate detuning
    freq = run.data["freq"]
    run.data['OmegaR'] = OmegaR
    run.data["Delta"] = (freq - res_freq)/EF

    # calculate transfer
    c9 = run.data['c9']
    c9_bg  = np.mean(c9.loc[run.data['Delta'] < Delta_cutoff])
    run.data["Transfer"] = (c9_bg - c9)/(t_pulse*c9_bg)
    run.data["Scaled transfer"] = run.data["Transfer"] * (EF*h)/(hbar * pi * OmegaR**2) # EF in MJ, Omega_R in MHz, transfer in MHz
    
    # take average
    run.group_by_mean("Delta")
    x, y, ey = np.array(run.avg_data['Delta']), np.array(run.avg_data["Scaled transfer"]), np.array(run.avg_data["e_Scaled transfer"])

    # list contains values to save to metadata df
    return x, y, ey, [Delta_cutoff, c9_bg]

def fit(x, y, ey, p0=[0.1]):
    """fits data to f(x,a) """
    imin = np.argmin(np.abs(x-EFmin))
    imax = np.argmin(np.abs(x-EFmax))
    popt, pcov = curve_fit(f, x[imin:imax], y[imin:imax], p0, sigma=ey[imin:imax], bounds=[0,np.inf])
    perr = np.sqrt(np.diag(pcov))

    a, e_a = popt[0], perr[0]

    # also return imin, imax for plotting purposes
    return a, e_a, imin, imax

# runs in old_data to exclude 
exclude = ['2023-08-16_J.dat', '2023-08-16_F.dat', # varying time
           '2023-08-21_E.dat', '2023-08-30_J.dat', '2023-08-30_I.dat', '2023-08-16_J.dat', '2023-08-22_J.dat', '2023-08-16_F.dat', '2023-08-23_H.dat', '2023-08-23_H.dat', # varying time
           '2023-08-03_C.dat', '2023-09-08_B.dat', '2023-08-24_B.dat', '2023-07-23_E.dat', '2023-07-27_E.dat', # looks funny
           '2023-09-14_L.dat', '2023-09-13_N2.dat', '2023-08-08_F2.dat', '2023-09-14_J.dat', '2023-09-18_E.dat', '2023-09-13_N.dat', # no metadata
           '2023-09-13_M.dat', '2023-09-13_G.dat', '2023-09-18_E2.dat', '2023-09-13_G3.dat', '2023-09-13_G2.dat', '2024-06-12_K_e.dat', # no metadata
           '2023-09-07_ZA2.dat', '2023-09-07_ZA.dat', '2023-08-02_C.dat', '2023-08-02_B.dat', '2023-08-02_E.dat', '2023-08-02_H.dat'] # not enough points at neg. detuning

for filename in os.listdir(data_folder):
    if filename in exclude or ".dat" not in filename:
        continue 

    # open file
    try:
        run = Data(filename, path=data_folder)
        i = get_metadata(filename)
    except Exception as e:
        print("metadata not found")
        continue

    if not old_data and (metadata.loc[i, "exclude"] == 1):
        continue

    # scale detuning & calculate transfer
    try:
        if old_data:
            x, y, ey, calc_vals = scale_x_old(run, i) # x = Delta, y = avg scaled transfer
        else:
            x, y, ey, calc_vals = scale_x(run, i)
    except Exception as e:
        print("error reading data:", e)
        continue
    
    if x[-1] <= EFmin:
        print("dataset out of fit range")
        continue
    
    # calculate sum rule and contact
    try:
        area = trapezoid(y=y, x=x)
        a, e_a, imin, imax = fit(x, y, ey)
        C  = a*pi**2 * np.sqrt(2)
    except Exception as e:
        print("fit error:", e)
        continue
    
    # save calculated values
    metadata.loc[i, new_columns] = calc_vals + [area, a, e_a, C]

    # plot
    if save_figs:
        fig, axs = plt.subplots(2, 2, height_ratios=[2,1], width_ratios=[2,1], figsize=(15,12))

        # plot data
        axs[0,0].errorbar(x, y, ey)
        axs[0,0].axvline(metadata.loc[i, "Delta_cutoff"], linestyle="--", color="lightgrey",label=r"$\Delta$ cutoff")
        # plot fit
        xfit = np.linspace(EFmin, EFmax, 1000)
        xfit_full = np.linspace(EFmin, x[-1], 1000)
        axs[0,0].plot(xfit_full, f(xfit_full, a), linestyle="--", marker="", label="", color="cornflowerblue", zorder=5)
        axs[0,0].plot(xfit, f(xfit, a), linestyle="-", marker="", label="fit", zorder=5, color="cornflowerblue")
        axs[0,0].legend()
        axs[0,0].set_xlabel(r"$\Delta$ [EF]")
        axs[0,0].set_ylabel(r'$\tilde\Gamma$')
        axs[0,0].set_title(filename)
        axs[0,0].set_xlim([-1, x[-1]])
        # plot residuals
        axs[1,0].axhline(0, linestyle="--", color="lightgrey")
        axs[1,0].errorbar(x[imin:], f(x[imin:], a)-y[imin:])
        axs[1,0].set_xlabel(r"$\Delta$ [EF]")
        axs[1,0].set_ylabel("fit-data")
        axs[1,0].set_title("residuals")
        axs[1,0].set_xlim([-1, x[-1]])     

        # plot closeup on peak
        axs[0,1].errorbar(x, y, ey)
        axs[0,1].set_xlabel(r"$\Delta$ [EF]")
        axs[0,1].set_ylabel(r'$\tilde\Gamma$')
        axs[0,1].set_title("peak")
        axs[0,1].set_xlim(-0.5, 1)

        # plot closeup on tail
        axs[1,1].errorbar(x, y, ey)
        axs[1,1].set_xlim(2, 10)
        axs[1,1].set_ylim(-0.03, 0.03)
        axs[1,1].set_title("tail")
        axs[1,1].axhline(0, linestyle="--", color="lightgrey")
        axs[1,1].set_xlabel(r"$\Delta$ [EF]")
        axs[1,1].set_ylabel(r'$\tilde\Gamma$')

        plt.savefig(f"{data_folder}/HFT_loss_results/{filename[:-4]}.jpeg", bbox_inches="tight")

# write results to csv
if save_file:
    save_file_path = os.path.join(data_folder, 'HFT_loss_results', save_file)

    if old_data:
        metadata[metadata['a']>=0].to_csv(save_file_path, index=False, columns=["date", "letter", "pulse", "time", "Omega_R"]+new_columns)
    else:
        metadata[metadata['exclude']==0].to_csv(save_file_path, index=False, columns=["filename", "EF", "pulsetype", "trf"]+new_columns)

