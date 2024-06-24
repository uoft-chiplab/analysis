# -*- coding: utf-8 -*-
"""
Created by Chip lab 2024-06-12

Loads .dat with contact HFT scan and computes scaled transfer. Plots. Also
computes the sumrule.
"""

from library import pi, h, plt_settings, GammaTilde
from data_class import Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# filename = "2024-06-12_K_e.dat"
filename = "2024-06-18_G_e.dat"
# filename = "2024-06-20_C_e.dat" # reminder; had to kill 0 detuning because of scatter
# filename = "2024-06-20_D_e.dat" # reminder; had to kill 0 detuning because of scatter
# filename = "2024-06-21_F_e.dat"
VVAtoVppfile = "VVAtoVpp.txt" # calibration file
VVAs, Vpps = np.loadtxt(VVAtoVppfile, unpack=True)
VpptoOmegaR = 27.5833 # kHz

def VVAtoVpp(VVA):
	"""Match VVAs to calibration file values. Will get mad if the VVA is not
		also the file. """
	for i, VVA_val in enumerate(VVAs):
		if VVA == VVA_val:
			Vpp = Vpps[i]
	return Vpp

### run params
xname = 'freq'
ff = 1.03
trf = 200e-6  # 200 or 400 us
EF = 16e-3 #MHz
bg_freq = 47  # chosen freq for bg, large negative detuning
res_freq = 47.2159 # for 202.1G
pulsetype = 'Kaiser'
pulse_area = 0.3 # Blackman
pulse_area = np.sqrt(0.3*0.92) # maybe? for first 4
# pulse_area=np.sqrt(0.3) # if using real Blackman
gain = 0.1 # scales the VVA to Vpp tabulation

### create data structure
run = Data(filename)
# kill a point
run.data.drop([77], inplace=True)
num = len(run.data[xname])

### compute bg c5, transfer, Rabi freq, etc.
bgc5 = run.data[run.data[xname]==bg_freq]['c5'].mean()
run.data['N'] = run.data['c5']-bgc5*np.ones(num)+run.data['c9']*ff
run.data['transfer'] = (run.data['c5'] - bgc5*np.ones(num))/run.data['N']
run.data['detuning'] = run.data[xname] - res_freq*np.ones(num) # MHz
run.data['Vpp'] = run.data['vva'].apply(VVAtoVpp)
run.data['OmegaR'] = 2*pi*pulse_area*gain*VpptoOmegaR*run.data['Vpp']

run.data['ScaledTransfer'] = run.data.apply(lambda x: GammaTilde(x['transfer'],
								h*EF*1e6, x['OmegaR']*1e3, trf), axis=1)
run.data['C'] = run.data.apply(lambda x: 2*np.sqrt(2)*pi**2*x['ScaledTransfer'] * \
								   (x['detuning']/EF)**(3/2), axis=1)
# run.data = run.data[run.data.detuning != 0]

	
### now group by freq to get mean and stddev of mean
run.group_by_mean(xname)

### interpolate scaled transfer for sumrule integration
xp = np.array(run.avg_data['detuning'])/EF
fp = np.array(run.avg_data['ScaledTransfer'])
maxfp = max(fp)
TransferInterpFunc = lambda x: np.interp(x, xp, fp)

### PLOTTING
plt.rcParams.update(plt_settings) # from library.py
plt.rcParams.update({"figure.figsize": [8,8],
					 "font.size": 14})
fig, axs = plt.subplots(2,2)

xlabel = r"Detuning $\omega_{rf}-\omega_{res}$ (MHz)"

### plot transfer fraction
ax = axs[0,0]
x = run.avg_data['detuning']
y = run.avg_data['transfer']
yerr = run.avg_data['em_transfer']
ylabel = r"Transfer $\Gamma \,t_{rf}$"

xlims = [-0.03,0.25]

ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims)
ax.errorbar(x, y, yerr=yerr, fmt='o')
# ax.plot(x, y, 'o-')

### plot scaled transfer
ax = axs[1,0]
x = run.avg_data['detuning']/EF
y = run.avg_data['ScaledTransfer']
yerr = run.avg_data['em_ScaledTransfer']
xlabel = r"Detuning $\Delta$"
ylabel = r"Scaled Transfer $\tilde\Gamma$"

xlims = [-2,16]
axxlims = [-2,16]
ylims = [-0.5,1]
xs = np.linspace(xlims[0], xlims[-1], len(y))

ax.set(xlabel=xlabel, ylabel=ylabel, xlim=axxlims, ylim=ylims)
ax.errorbar(x, y, yerr=yerr, fmt='o')
ax.plot(x, y, '-')

# sumrule = np.trapz(TransferInterpFunc(xs), x=xs)
sumrule = np.trapz(y, x=x)
print("sumrule = {:.3f}".format(sumrule))

### plot contact
ax = axs[0,1]
x = run.avg_data['detuning']/EF
y = run.avg_data['C']
yerr = run.avg_data['em_C']
xlabel = r"Detuning $\Delta$"
ylabel = r"Contact $C/N$ [$k_F$]"

xlims = [-2,16]
ylims = [-2, 5]
Cdetmin = 3
Cdetmax = 8
xs = np.linspace(Cdetmin, Cdetmax, num)

df = run.data[run.data.detuning/EF>Cdetmin]
Cmean = df[df.detuning/EF<Cdetmax].C.mean()

ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims, ylim=ylims)
ax.errorbar(x, y, yerr=yerr, fmt='o')
ax.plot(xs, Cmean*np.ones(num), "--")

### generate table
ax = axs[1,1]
ax.axis('off')
ax.axis('tight')
quantities = ["$E_F$", "Contact $C/N$", "sumrule"]
values = ["{:.1f} kHz".format(EF*1e3), 
		  "{:.2f} kF".format(Cmean), 
		  "{:.3f}".format(sumrule)]
table = list(zip(quantities, values))

the_table = ax.table(cellText=table, loc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1,1.5)

plt.show()

datatosave = {'SumRule': [sumrule], 'Gain':[gain], 'Run':[filename], 
			  'C/N':[Cmean],
			  'Max Scaled Transfer':[maxfp], 'Pulse Time (us)':[trf*1e6],
			  'Pulse Area':[pulse_area], 'Pulse Type':[pulsetype]}
datatosavedf = pd.DataFrame(datatosave)

datatosave_folder = 'SavedSumRule'
runfolder = filename 
figpath = os.path.join(datatosave_folder,runfolder)
os.makedirs(figpath, exist_ok=True)

sumrulefig_name = 'SumRule.png'
sumrulefig_path = os.path.join(figpath,sumrulefig_name)
fig.savefig(sumrulefig_path)

xlsxsavedfile = 'Saved_Sum_Rules.xlsx'

filepath = os.path.join(datatosave_folder,xlsxsavedfile)

### if you want to change the headers you need this on then it makes 2 lines so delete the duplicate
# and turn off after
# datatosavedf.to_excel(filepath,index=False)

try:
    existing_data = pd.read_excel(filepath, sheet_name='Sheet1')
except FileNotFoundError:
    existing_data = None
new_data = datatosavedf

with pd.ExcelWriter(filepath, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
    if existing_data is not None:
        # Calculate the starting row for writing new_data
        start_row = existing_data.shape[0] + 1  # Start after existing_data
        new_data.to_excel(writer, index=False, header=False, sheet_name='Sheet1', startrow=start_row)
    else:
        # Write new_data starting from the first row if no existing data
        new_data.to_excel(writer, index=False, sheet_name='Sheet1')

fig, ax2 = plt.subplots()

sumruleexceldf = pd.read_excel(filepath, index_col=0, engine='openpyxl').reset_index()
sumrules = np.array([.353,.336,.373,.443])
# sumrules = sumruleexceldf['SumRule']

gain = np.array([.3,.2,.1,.05])
# gain = sumruleexceldf['Gain']

CoN = np.array([.85,.98,1.08,1.44])

C = CoN/(sumrules/0.5)

ax2.set_xlabel('gain')
ax2.set_ylabel('C')
ax2.plot(gain, C,linestyle='',marker='.')