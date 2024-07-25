import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd 
from scipy.integrate import trapezoid, simpson, cumulative_simpson
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["hotpink", "cornflowerblue", "yellowgreen"]) 
os.chdir("/Users/maggie/Documents/ChipAnalysis/")
from data_class import Data
from library import FreqMHz, FermiEnergy, OmegaRcalibration, GammaTilde, ChipKaiser, ChipBlackman
from library import hbar, h, pi

# define constants 
gain = 1

# b
F1 = 9/2
mF1 = -7/2
# c
F2 = 9/2
mF2 = -5/2

# fit starting EF
EFmin = 2
EFmax = 8

# saves figures to folder if True
save_figs = 1
# file to save text to
save_file = "HFT_loss_results.csv"

data_folder = 'clockshift/old_data'
old_data = "old" in data_folder

try:
    os.mkdir(data_folder+"/HFT_loss_results")
except:
    print('overwriting previous results')

if old_data:
    metadata_file = os.path.join(data_folder, "ContactMeasurementsSummary.xlsx")
    columns1 = ["mean_w", "EF", "res_freq", "Delta_cutoff", "c9_bg"]
else:
    metadata_file = os.path.join('clockshift', "metadata_file.xlsx")
    columns1 = ["Delta_cutoff", "c9_bg"]

columns2 = ["sum_rule", "a", "e_a", "contact"]

metadata = pd.read_excel(metadata_file)
metadata[columns1] = [None]*len(columns1)
metadata[columns2] = [None]*len(columns2)

VVAtoVppfile = os.path.join("VVAtoVpp.txt")
VVAs, Vpps = np.loadtxt(VVAtoVppfile, unpack=True)
VpptoOmegaR = 27.5833 # kHz
VVAtoVpps = lambda x: np.interp(x, VVAs, Vpps)

OmegaR_interp = OmegaRcalibration()

def VVAtoVpp(VVA):
	"""from sumrule_analysis """
	Vpp = 0
	for i, VVA_val in enumerate(VVAs):
		if VVA == VVA_val:
			Vpp = Vpps[i]
	if Vpp == 0: # throw a fit if VVA not in list.
		raise ValueError("VVA value {} not in VVAtoVpp.txt".format(VVA))
	return Vpp

def SPA(pulsetype, run):
	# from sumrule_analysis
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
    if old_data:
        date = filename.split("_")[0]
        letter = filename.split("_")[1].split(".")[0]
        run_metadata = metadata.loc[metadata.date==date].loc[metadata.letter==letter]

    else:
        run_metadata = metadata.loc[metadata.filename==filename.split(".")[0]] # remove .dat from filename

    i = run_metadata.index[0]

    return i

def scale_x_old(run, i):
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

        VVA_rule_str = np.fromstring(run_metadata['VVArule'][1:-1], sep=",")
        VVA_keys = VVA_rule_str[0::2]
        VVA_entries = VVA_rule_str[1::2]

        VVA_rule = {}
        for i in range(len(VVA_keys)):
            VVA_rule[VVA_keys[i]] = VVA_entries[i]

        # convert Vpp to OmegaR with calibration value from OmegaR_calibration in library.py
        run.data['Vpps'] = [VVA_rule[vva] for vva in VVA]
        run_SPA = SPA(pulse_type, run)

        OmegaR = 2*pi* run_SPA*gain* 27.5833 *run.data['Vpps']/1e3 # kHz- > MHz
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

    run.data["Delta"] = (freq - res_freq)/EF

    # calculate background value
    c9 = run.data['c9']
    Delta_cutoff = -1.5 #-2/t_pulse * (1 + int(pulse_type == "blackman")) # extra factor of 2 in Fourier transform width if blackman, 1 if square
    c9_bg  = np.mean(c9.loc[run.data['Delta'] < Delta_cutoff])
    
    run.data["Transfer"] = (c9_bg - c9)/(t_pulse*c9_bg)
    run.data["Scaled transfer"] = run.data["Transfer"] * (EF*h)/(hbar * pi * OmegaR**2) # EF in MJ, Omega_R in MHz, transfer in MHz
    run.data['OmegaR'] = OmegaR

    # save calculated metadata values
    metadata.loc[i, columns1] = [mean_w, EF, res_freq, Delta_cutoff, c9_bg]

    # also calcualte average y values
    run.group_by_mean("Delta")
    x, y, ey = np.array(run.avg_data['Delta']), np.array(run.avg_data["Scaled transfer"]), np.array(run.avg_data["e_Scaled transfer"])
    
    return x, y, ey

def scale_x(run, i):
    run_metadata = metadata.loc[i]
    EF = run_metadata["EF"] # MHz
    t_pulse = run_metadata["trf"]*1e6 # us
    pulse_type = run_metadata["pulsetype"]
    res_freq = run_metadata["res_freq"]
    gain = run_metadata['gain']
    bad_indices = run_metadata['remove_indices']

    run_VVAs = run.data['vva']
    if np.min(run_VVAs) < 1.1 or np.max(run_VVAs) > 10:
        print("VVA out of interpolation range")
        return
    
    run_SPA = SPA(pulse_type, run)
    run_Vpps = VVAtoVpps(run_VVAs)
    OmegaR = 2*pi* VpptoOmegaR * run_Vpps * run_SPA * gain / 1e3  # kHz- > MHz

    # rescale x axis
    freq = run.data["freq"]
    run.data["Delta"] = (freq - res_freq)/EF

    # calculate background value
    c9 = run.data['c9']
    Delta_cutoff = -1.5 #-2/t_pulse * (1 + int(pulse_type == "blackman")) # extra factor of 2 in Fourier transform width if blackman, 1 if square
    c9_bg  = np.mean(c9.loc[run.data['Delta'] < Delta_cutoff])
    
    run.data["Transfer"] = (c9_bg - c9)/(t_pulse*c9_bg)
    run.data["Scaled transfer"] = run.data["Transfer"] * (EF*h)/(hbar * pi * OmegaR**2) # EF in MJ, Omega_R in MHz, transfer in MHz
    run.data['OmegaR'] = OmegaR

    # save calculated metadata values
    metadata.loc[i, columns1] = [Delta_cutoff, c9_bg]

    # remove indicies if required
    if not np.sum(np.isnan(bad_indices)):
        run.data.drop(bad_indices)

    # also calcualte average y values
    run.group_by_mean("Delta")
    x, y, ey = np.array(run.avg_data['Delta']), np.array(run.avg_data["Scaled transfer"]), np.array(run.avg_data["e_Scaled transfer"])

    return x, y, ey

def f(x, a):
    return a* x**(-3/2)

def sum_rule(x, y):
    area = trapezoid(y=y, x=x)
    return area

def fit(x, y, ey, p0=[0.1]):
    imin = np.argmin(np.abs(x-EFmin))
    imax = np.argmin(np.abs(x-EFmax))
    popt, pcov = curve_fit(f, x[imin:imax], y[imin:imax], p0, sigma=ey[imin:imax], bounds=[0,np.inf])
    perr = np.sqrt(np.diag(pcov))
    a, e_a = popt[0], perr[0]
    C  = a*pi**2 * np.sqrt(2)

    return a, e_a, C, imin, imax

def plot(x, y, ey, a, i, imin, imax, filename):
    print(i)
    fig, axs = plt.subplots(2, 2, height_ratios=[2,1], width_ratios=[2,1], figsize=(15,12))

    # plot data
    axs[0,0].errorbar(x, y, ey)
    axs[0,0].axvline(metadata.loc[i, "Delta_cutoff"], linestyle="--", color="lightgrey",label=r"$\Delta$ cutoff")
    # plot fit
    xfit = np.linspace(EFmin, EFmax, 1000)
    xfit_full = np.linspace(EFmin, x[-1], 1000)
    axs[0,0].plot(xfit_full, f(xfit_full, a), linestyle="--", marker="", label="", color="cornflowerblue", zorder=5)
    axs[0,0].plot(xfit, f(xfit, a), linestyle="-", marker="", label="fit", zorder=5, color="cornflowerblue")
    # plot residuals
    axs[1,0].axhline(0, linestyle="--", color="lightgrey")
    axs[1,0].errorbar(x[imin:], f(x[imin:], a)-y[imin:])
    
    # label axes
    axs[0,0].set_xlabel(r"$\Delta$ [EF]")
    axs[1,0].set_xlabel(r"$\Delta$ [EF]")
    axs[0,1].set_xlabel(r"$\Delta$ [EF]")
    axs[1,1].set_xlabel(r"$\Delta$ [EF]")
    axs[0,0].set_ylabel(r'$\tilde\Gamma$')
    axs[0,1].set_ylabel(r'$\tilde\Gamma$')
    axs[1,1].set_ylabel(r'$\tilde\Gamma$')
    axs[1,0].set_ylabel("fit-data")
    # draw table of attributes

    axs[0,0].set_title(filename)
    axs[1,0].set_title("residuals")
    axs[0,1].set_title("peak")
    axs[1,1].set_title("tail")

    axs[0,1].errorbar(x, y, ey)
    axs[0,1].set_xlim(-0.5, 1)
    axs[1,1].errorbar(x, y, ey)
    axs[1,1].set_xlim(2, 10)
    axs[1,1].set_ylim(-0.05, 0.05)
    axs[1,1].axhline(0, linestyle="--", color="lightgrey")

    axs[0,0].set_xlim([-1, x[-1]])
    axs[1,0].set_xlim([-1, x[-1]])

    # also add a legend
    axs[0,0].legend()

    if save_figs:
        plt.savefig(f"{data_folder}/HFT_loss_results/{filename[:-4]}.jpeg", bbox_inches="tight")

exclude = ['2023-08-16_J.dat', '2023-08-16_F.dat', # varying time
           '2023-08-21_E.dat', '2023-08-30_J.dat', '2023-08-30_I.dat', '2023-08-16_J.dat', '2023-08-22_J.dat', '2023-08-16_F.dat', '2023-08-23_H.dat', '2023-08-23_H.dat', # varying time
           '2023-08-03_C.dat', '2023-09-08_B.dat', '2023-08-24_B.dat', '2023-07-23_E.dat', # looks funny
           '2023-09-14_L.dat', '2023-09-13_N2.dat', '2023-08-08_F2.dat', '2023-09-14_J.dat', '2023-09-18_E.dat', '2023-09-13_N.dat', # no metadata
           '2023-09-13_M.dat', '2023-09-13_G.dat', '2023-09-18_E2.dat', '2023-09-13_G3.dat', '2023-09-13_G2.dat', '2024-06-12_K_e.dat', # no metadata
           '2023-09-07_ZA2.dat', '2023-09-07_ZA.dat', '2023-08-02_C.dat', '2023-08-02_B.dat', '2023-08-02_E.dat', '2023-08-02_H.dat'] # not enough points at neg. detuning

for filename in os.listdir(data_folder):
    if filename in exclude or ".dat" not in filename:
        continue 

    print(f"\n---------------------------{filename}---------------------------")

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
            x, y, ey = scale_x_old(run, i) # x = Delta, y = avg scaled transfer
        else:
            x, y, ey = scale_x(run, i)
    except Exception as e:
        print("error reading data:", e)
        continue

    if x[-1] <= EFmin:
        print("dataset out of fit range")
        continue
    
    # calculate sum rule and fit
    try:
        area = sum_rule(x, y)
        a, e_a, C, imin, imax = fit(x, y, ey)
        metadata.loc[i, columns2] = [area, a, e_a, C]
    except Exception as e:
        print("fit error:", e)
        continue

    # plot
    if save_figs:
        plot(x, y, ey, a, i, imin, imax, filename)

if save_file:
    save_file_path = os.path.join(data_folder, 'HFT_loss_results', save_file)

    if old_data:
        metadata[metadata['a']>=0].to_csv(save_file_path, index=False, columns=["date", "letter", "pulse", "time", "Omega_R"]+columns1+columns2)
    else:
        metadata[metadata['exclude']==0].to_csv(save_file_path, index=False, columns=["filename", "EF", "pulsetype", "trf"]+columns1+columns2)

    # pd.read_csv(save_file_path)
