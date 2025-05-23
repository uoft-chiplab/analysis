# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:19:24 2025

@author: coldatoms
"""

# correlate2d finds the part of img1 which img2 is a subsection of

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import correlate2d, fftconvolve
from scipy.ndimage import rotate

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mode = "fft"
imgtype = "pixis"

img1_path = r"\\HAFNIUM\Save pictures\2023-02-02\imaging light thorcam.tif"
img2_path = r"\\HAFNIUM\Save pictures\2025-02-03\HF_fringe.png"

if logging.DEBUG >= logger.getEffectiveLevel():
    # test images
    img1 = np.copy(plt.imread(img2_path)[:,:,0])
    x, y, w, h = 100, 250, 100, 300
    # x, y , w, h = 0, 0, np.shape(img1)[1], np.shape(img1)[0]
    img2 = np.copy(img1[y:y+h, x:x+w])

elif imgtype != "pixis":
    # for img1 from thorcam 
    img1 = np.copy(plt.imread(img1_path)[:,:,0])
    img1 = np.rot90(img1, 1)
    img1 = img1/255
    # renormalize img to between 0, 1 
    img2_rotated = (img1 - img1.min()) / (img1.max() - img1.min())

    img2_full = plt.imread(img2_path)[:,:,0] 
    h, w = img2_full.shape # final dimensions of img2
    h = int(h/1.25)
    w = int(w/1.25)
    img2_rotated = rotate(img2_full, angle=-4, reshape="false", mode="nearest") # rotate so the razor angle matches in both images

    # Get the new rotated image size
    h_rot, w_rot = img2_rotated.shape
    # Calculate the x and y coords of the crop point, based on equal margins from centre
    x = int((w_rot - w) / 2)
    y = int((h_rot - h) / 2)

    # Crop the rotated image back to the original size 
    img2 = np.copy(img2_rotated[y:y+h, x:x+w])
    # renormalize img to between 0, 1 
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())

else: 
    img1 = np.copy(plt.imread(img1_path)[:,:,0])
    img2 = np.copy(plt.imread(img2_path)[:,:,0])

# process images
img1 - img1.mean()
img2 - img2.mean()

# get max to plot on same color scale
vmin = np.min([img1.min(), img2.min()])
vmax = np.max([img1.max(), img2.max()])

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
axs[0].imshow(img1, cmap="RdPu_r", vmin=vmin, vmax=vmax)

if logging.DEBUG >= logger.getEffectiveLevel():
    # test
    axs[1].imshow(img2, cmap="RdPu_r", vmin=vmin, vmax=vmax)
    rect = patches.Rectangle([x, y], w, h, linewidth=1, edgecolor="white", facecolor='none', label="original crop")
    axs[0].add_patch(rect)
    
    axs[0].set_title("full image")
    axs[1].set_title("cropped section")
    axs[2].set_title("cross correlations")

elif imgtype != "pixis":
    axs[1].imshow(img2_rotated, cmap="RdPu_r", vmin=vmin, vmax=vmax)
    rect = patches.Rectangle([x, y], w, h, linewidth=1, edgecolor="skyblue", facecolor='none', label="crop region")
    axs[1].add_patch(rect)

    axs[0].set_title("thorcam img")
    axs[1].set_title("cropped pixis img")
    axs[2].set_title("cross correlations")

    axs[1].legend()

else:
    axs[1].imshow(img2, cmap="RdPu_r", vmin=vmin, vmax=vmax)

    axs[0].set_title("img1")
    axs[1].set_title("img2")
    axs[2].set_title("cross correlations")


if mode != "fft":
    # choose output size to be same as the input
    corr2d = correlate2d(img1, img2, mode='same', boundary='fill', fillvalue=0)

    if imgtype != "pixis":
        # find point match where subimg is centered
        y, x = np.unravel_index(np.argmax(corr2d), corr2d.shape)
        # plot matched section from cross correlation
        rect = patches.Rectangle([x-w/2, y-h/2], w, h, linewidth=1, edgecolor="skyblue", facecolor='none')
        axs[0].add_patch(rect)

    axs[2].imshow(corr2d)

else:
    # Perform cross correlation using FFT -- seems to be faster than correlate 2d
    corr2d = fftconvolve(img1, np.flip(img2))
    y, x = np.unravel_index(np.argmax(corr2d), corr2d.shape)

    if imgtype != "pixis":
        # not matching areas, just looking at similarities

        # plot matched section from cross correlation
        rect = patches.Rectangle([x-w, y-h], w, h, linewidth=1, edgecolor="skyblue", facecolor='none', label="conv. match")
        axs[0].add_patch(rect)
        axs[0].legend()

    axs[2].imshow(corr2d)
