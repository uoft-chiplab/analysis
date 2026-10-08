import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime as dt
import matplotlib.dates as mdates
from glob import glob

# get aug and sept files
root_path = r"D:\logs\MOTcam_10875"
files = glob("2026-08-*_gaussian_fit.csv", root_dir=root_path)+glob("2026-09-*_gaussian_fit.csv", root_dir=root_path)

date_data = []

metadata = pd.DataFrame(columns=["Timestamp", "Amp", "Ntot"])

atom = "Rb"

for f in files:
    if "08-20" in f:
        continue
    # load data

    data = pd.read_csv(os.path.join(root_path, f))
    if data["Timestamp"].dtype == int:
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], format="%Y%m%d%H%M%S")
    else:
        data["Timestamp"] = pd.to_datetime(data["Timestamp"])

    if atom == "Rb":
        data = data[data['Gain_dB'] == 24.0]
        # drop bogus fits
        data.drop(data[data["Ntot"] > 1e8].index, inplace=True)
        # drop fits with poor atom number
        data.drop(data[data["Ntot"] < 9e6].index, inplace=True)
        data.drop(data[data["Amp"] < 700].index, inplace=True)
    elif atom == "K":
        data = data[data['Gain_dB'] == 24.0]
        # drop bogus fits
        data.drop(data[data["Ntot"] > 9e6].index, inplace=True)
        # drop fits with poor atom number
        data.drop(data[data["Ntot"] < 1e5].index, inplace=True)
        data.drop(data[data["Amp"] < 70].index, inplace=True)

    data.reset_index(drop=True, inplace=True)

    if len(data)< 150:
        continue

    metadata = pd.concat([metadata, data[["Timestamp", "Amp", "Ntot"]]], ignore_index=True)

    date_data.append(data[["Timestamp", "Amp", "Ntot"]])

plot_date = dt.datetime(2026,1,1,1,1,1)
fig, axs = plt.subplots(len(date_data), 1, figsize=(10, 1*len(date_data)), sharex=True)

for i, d in enumerate(date_data):
    # reset time to plot on same axis
    time_adjusted_data = d.copy()

    time = time_adjusted_data["Timestamp"][0].to_pydatetime()
    time_adjusted_data["Timestamp"] = time_adjusted_data["Timestamp"] - pd.Timedelta(days=(time - plot_date).days)

    factor = 7 if atom == "Rb" else 5

    axs[i].plot(time_adjusted_data["Timestamp"], time_adjusted_data["Ntot"]/(10**factor), label=f"{time.date()}")
    axs[i].set_ylabel(f"N ($10^{factor}$)")
    axs[i].set_xlabel("Time")
    axs[i].legend(loc=1)

axs[0].set_title(f"Atom number {atom} MOT")
fig.tight_layout(h_pad=0.05)

time_form = mdates.DateFormatter('%H:%M')
axs[-1].xaxis.set_major_formatter(time_form)

# amplitude plot

fig, axs = plt.subplots(len(date_data), 1, figsize=(10, 1*len(date_data)), sharex=True)

for i, d in enumerate(date_data):
    # reset time to plot on same axis
    time_adjusted_data = d.copy()

    time = time_adjusted_data["Timestamp"][0].to_pydatetime()
    time_adjusted_data["Timestamp"] = time_adjusted_data["Timestamp"] - pd.Timedelta(days=(time - plot_date).days)

    axs[i].plot(time_adjusted_data["Timestamp"], time_adjusted_data["Amp"], label=f"{time.date()}")
    axs[i].set_ylabel("A")
    axs[i].set_xlabel("Time")
    axs[i].legend(loc=1)

axs[0].set_title(f"{atom} MOT Gaussian fit amplitude")
fig.tight_layout(h_pad=0.05)

time_form = mdates.DateFormatter('%H:%M')
axs[-1].xaxis.set_major_formatter(time_form)