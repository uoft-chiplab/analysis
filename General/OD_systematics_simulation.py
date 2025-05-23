# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:00:35 2025

@author: Chip Lab

OD systematics simulation script.

Goal of this script is to test how differences in bg counts on imgs 2-4 
affect OD estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
from library import styles, colors

def Gaussian(x, A, mu, sigma):
	return A * np.exp(-((x-mu)/sigma)**2/2)

def TwoDGaussian(x, y, A, mu_x, mu_y, sigma):
	return Gaussian(x, A, mu_x, sigma) * Gaussian(y, 1, mu_y, sigma)

def OD(img2, img3, img4):
	return -np.log((img2-img4)/(img3-img4))

# generate random image
N = int(1000/2)
X = np.linspace(0, N-1, N)
Y = np.linspace(0, N-1, N)
XX, YY  = np.meshgrid(X, Y)

# add random noise to grid as bg
bg = 600 # bg offset
noise = np.sqrt(bg)
# noise = 0.01
img2 = np.random.normal(bg, noise, XX.shape)
img3 = np.random.normal(bg, noise, XX.shape)
img4 = np.random.normal(bg, noise, XX.shape)

# add imaging light
light = 2000
light_pattern = np.random.normal(light, np.sqrt(light), XX.shape)
img2 += light_pattern
img3 += light_pattern

# add Gaussian cloud to img
A = -light/4  # because atoms absorb light
mu = N/2
sigma = N/8
atoms_pattern = TwoDGaussian(XX, YY, A, mu, mu, sigma)
img2 += atoms_pattern

# calculate OD
ODimg = OD(img2, img3, img4)

# put images in list for plotting
titles = ['img2', 'img3', 'img4', 'OD']
imgs = [img2, img3, img4, ODimg]

# plot 
fig, axes = plt.subplots(2,2, figsize=(12,12))
axs = axes.flatten()
for ax, img, title in zip(axs, imgs, titles):
	ax.set(xlabel='X (px)', ylabel='Y (px)', title=title)
	ax.imshow(img)

fig.tight_layout()
plt.show()

# calculate max OD
max_OD = max(ODimg.flatten())
print("Max OD is {:.2f}".format(max_OD))

# plot OD vs. img4 bg difference
fig = plt.figure(figsize = (6,4))
x = np.linspace(-100, 100, N)
plt.plot(x, OD(light*3/4+bg, light+bg, x+bg), '-')
plt.xlabel("Delta img4")
plt.ylabel("OD")
plt.show()

# plot OD vs. light counts for difference img4 bg differences
fig = plt.figure(figsize = (6,4))
x = np.linspace(0, light, N)

OD_nominal = OD(x*3/4+bg, x+bg, bg)

diffs = [-100, -40, -20, 0,]# 20, 40]

for diff, color in zip(diffs, colors):
	label = 'img4 + '+ str(diff)
	plt.plot(x, OD(x*3/4+bg, x+bg, bg+diff)/OD_nominal, '-', color=color, label=label)	
plt.xlabel("Light counts")
plt.ylabel("OD/OD_correct")
plt.legend()
plt.show()