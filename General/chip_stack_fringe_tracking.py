from scipy.io import loadmat
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np
# Load the .mat file
from glob import glob
time_form = mdates.DateFormatter('%H:%M')
from pathlib import Path
from datetime import datetime
np.seterr(all="ignore")

import sys
sys.path.append(r"D:\LocalCode\pythermometry")
from helpers.gaussfit import gauss1dFit
from helpers.gaussians import gauss1d

imgs = glob(r"F:\Data\2026\09 September2026\22September2026\*\imgs\A_*[0-9][0-9][0-9][0-9].mat")
# n = 738 #where atoms where turned on on the 21st
# imgs = imgs[738+32:-5] # skip first image due to camera calibrating
imgs = imgs[0:]
ROI = [50, -1, 500, 550] # for fringe tracking
atom_ROI = [90, 105, 480, 750] # crop out dark fringes, influences fit
npeaks=4 # number of peaks to track
peaknames = [f"peak{i}" for i in range(npeaks)]

def calculateOD(path, ROI):

    imgs = loadmat(path)
    # logger.info(imgs)
    # get image data from imgs file
    atom_img = imgs[f"image2"]
    ref_img = imgs[f"image3"]
    # logger.info(f'ref img: {ref_img}')
    bg_img = imgs[f"image4"]

    atom_img -= bg_img
    ref_img -= bg_img

    # ignore saturation since we don't care about absolute atom number here
    OD_img = np.real(-np.emath.log(atom_img / ref_img))
    return OD_img[ROI[0]:ROI[1], ROI[2]:ROI[3]]

df = pd.DataFrame(columns=peaknames)
dfnoatoms = pd.DataFrame(columns=peaknames)
times = []
timesnoatoms = []
atoms_on = True 
for i, imgpath in enumerate(imgs):
    OD_atoms = calculateOD(imgpath, atom_ROI)
    OD = calculateOD(imgpath, ROI)
    stat = Path(imgpath).stat()
    if hasattr(stat, 'st_birthtime'):
        timestamp = pd.to_datetime(datetime.fromtimestamp(stat.st_birthtime))
    else:
        continue 
    # check if atoms present in images
    if atoms_on and np.max(OD_atoms>0.1):
        # crop atoms out from fringes
        OD = OD[(atom_ROI[1]+10)-ROI[0]:, :]
        
        # fit atoms
        OD_atoms_sum = np.sum(OD_atoms, axis=0)
        try:
            popts, __ = gauss1dFit(np.arange(len(OD_atoms_sum)), OD_atoms_sum)
            gdx, gx0, gA, gb = popts
            fit_success = True

            # if gA>10:
            #     plt.figure()
            #     plt.plot(OD_atoms_sum)
        except:
            fit_success = False

    elif atoms_on and np.max(OD_atoms<0.1):
        ODnoatoms = OD
        y_profile_noatoms = np.sum(ODnoatoms, axis=1)
        peaksnotaoms, peak_properties = find_peaks(-y_profile_noatoms, width=(3,), height=(5,), prominence=0.5)
        dfnoatoms.loc[i, peaknames] = peaksnotaoms[:npeaks]
        # dfnoatoms.insert(loc=4, column='OD max', value=np.max(OD_atoms), allow_duplicates=True)
        # print(np.max(OD_atoms))
        timesnoatoms.append(timestamp)
        # print("adjust box, no atoms found")
        continue
    
    # skip bad shots
    if (-np.inf in OD) or (not fit_success):
        print(f"skipping image {i} due to bad shot, fit result {fit_success}")
        continue

    y_profile = np.sum(OD, axis=1)

    

    # track fringes
    peaks, peak_properties = find_peaks(-y_profile, width=(3,), height=(5,), prominence=0.5)
    
    try:
        if atoms_on:
            df.loc[i, ["gA", "gx0"]] = [gA, gx0]
        df.loc[i, peaknames] = peaks[:npeaks]
        times.append(timestamp)
    except:
        pass
    # plt.plot(y_profile)
    # plt.plot(peaks, y_profile[peaks], marker="o", ls="")

df["times"] = times
dfnoatoms["times"] = timesnoatoms
df.dropna(inplace=True)
# df = df[:646]
# times = times[:646]

###imgs without atoms
if atoms_on:
    droppedshots = dfnoatoms.index[dfnoatoms['peak0'] > 40].tolist()
    dfnoatoms = dfnoatoms.drop(dfnoatoms.index[dfnoatoms['peak0'] > 40].tolist())
    fig, ax = plt.subplots(4,1, figsize=(20, 5), sharex=True)
    colors=["hotpink", "cornflowerblue", "yellowgreen", "mediumpurple"]

    for i in range(4):
        ax[i].plot(dfnoatoms["times"] , dfnoatoms[f"peak{i}"], color=colors[i], label=f"peak {i+1}")
        ax[i].set_ylabel("position") 
        ax[i].legend()

    ax[-1].set_xlabel("Time")
    ax[-1].xaxis.set_major_formatter(time_form)
    ax[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    fig.suptitle(f'Dropped shots: {droppedshots}')

    fig = plt.figure()
    gs = GridSpec(2, 2, width_ratios=[1,1], height_ratios=[1,3])
    # ax1 = fig.add_subplot(gs[0, :])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    # ax1.imshow(OD, vmin=-0.2, vmax=0.4, cmap="RdPu_r")
    ax3.imshow(ODnoatoms, vmin=-0.2, vmax=0.4, cmap="RdPu_r")
    ax4.yaxis.set_inverted(True) 
    ax4.plot(y_profile_noatoms, range(len(y_profile_noatoms)), color="plum")
    ax4.plot(y_profile_noatoms[peaksnotaoms], peaksnotaoms, marker="*", ls="", color="hotpink", label="peaks")
    ax4.legend(loc=3)
    ax3.set_ylabel("OD image")
    ax4.set_xlabel("x")
    ax4.set_xlabel("OD sum in x")
    fig.tight_layout()

###imgs with atoms

fig, ax = plt.subplots(4,1, figsize=(20, 5), sharex=True)
colors=["hotpink", "cornflowerblue", "yellowgreen", "mediumpurple"]

for i in range(4):
    if i == 0 and "gA" in df.columns:
        ax[i].plot(df["times"] , df["gA"], color=colors[i], label="1D gaussian fit in x")
        ax[i].set_ylabel("Amplitude") 
    elif i == 1 and "gx0" in df.columns:
        ax[i].plot(df["times"] , df["gx0"], color=colors[i], label="1D gaussian fit in x")
        ax[i].set_ylabel("centre") 
    else:
        ax[i].plot(df["times"] , df[f"peak{i}"], color=colors[i], label=f"peak {i+1}")
        ax[i].set_ylabel("position") 
    ax[i].legend()

ax[-1].set_xlabel("Time")
ax[-1].xaxis.set_major_formatter(time_form)
ax[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# sample OD img
# fig, ax = plt.subplots(1, 2, figsize=(3.5, 5), sharey=True)
# ax[0].imshow(OD, vmin=-0.2, vmax=0.4, cmap="RdPu_r")
# ax[1].plot(y_profile, range(len(y_profile)), color="plum")
# ax[1].plot(y_profile[peaks], peaks, marker="*", ls="", color="hotpink", label="peaks")
# ax[1].legend(loc=3)
# ax[0].set_ylabel("OD image")
# ax[0].set_xlabel("x")
# ax[1].set_xlabel("OD sum in x")
# fig.tight_layout()

fig = plt.figure()
gs = GridSpec(2, 2, width_ratios=[1,1], height_ratios=[1,3])
ax1 = fig.add_subplot(gs[0, :])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3])

ax1.imshow(OD_atoms, vmin=-0.2, vmax=0.4, cmap="RdPu_r")
ax3.imshow(OD, vmin=-0.2, vmax=0.4, cmap="RdPu_r")
ax4.yaxis.set_inverted(True) 
ax4.plot(y_profile, range(len(y_profile)), color="plum")
ax4.plot(y_profile[peaks], peaks, marker="*", ls="", color="hotpink", label="peaks")
ax4.legend(loc=3)
ax3.set_ylabel("OD image")
ax4.set_xlabel("x")
ax4.set_xlabel("OD sum in x")
fig.tight_layout()