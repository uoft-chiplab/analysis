# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:22:21 2024

@author: coldatoms
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('field_cal_summary.csv')

# Fix wiggle_amp at 0.9 Vpp, plot B amp vs. wiggle freq
fig, ax = plt.subplots(1,1)
sub_df = df[df['wiggle_amp']==0.9]
ax.errorbar(sub_df['wiggle_freq'], sub_df['B_amp'], sub_df['e_B_amp'],fmt='bo')
ax.set_xlabel('drive freq (kHz)')
ax.set_ylabel('B field fit amplitude (G)')
ax.set_title('Fixed 0.9 Vpp drive amplitude')

# Fix wiggle_amp at 0.9 Vpp, plot B phase vs. wiggle freq
fig, ax = plt.subplots(1,1)
sub_df = df[df['wiggle_amp']==0.9]
ax.errorbar(sub_df['wiggle_freq'], sub_df['B_phase'], sub_df['e_B_phase'],fmt='bo')
ax.set_xlabel('drive freq (kHz)')
ax.set_ylabel('B field fit phase (rad)')
ax.set_title('Fixed 0.9 Vpp drive amplitude')

# Fix wiggle_freq at 10 kHz, plot B amp vs. wiggle amp
fig, ax = plt.subplots(1,1)
sub_df = df[df['wiggle_freq']==10.0]
ax.errorbar(sub_df['wiggle_amp'], sub_df['B_amp'], sub_df['e_B_amp'],fmt='bo')
ax.set_xlabel('drive amp (Vpp)')
ax.set_ylabel('B field fit amplitude (G)')
ax.set_title('Fixed 10 kHz drive freq')

# contour plot fun
fig, ax=plt.subplots(1,1)
hm_df = df.pivot(index='wiggle_amp', columns='wiggle_freq', values = 'B_amp')
sns.heatmap(hm_df,annot=True, cmap="viridis")
ax.set_title('B-field fit amplitude (G)')
ax.set_xlabel('drive freq (kHz)')
ax.set_ylabel('drive amp (Vpp)')