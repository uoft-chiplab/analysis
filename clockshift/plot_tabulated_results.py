# -*- coding: utf-8 -*-
"""
Created on Fri May 16 02:14:55 2025
KX made this as a check to ensure the output tabulated results csv files were correct
@author: coldatoms
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a_data_df = pd.read_csv('clockshift/tabulated_results/subplot_a_data.csv')
a_theory_df = pd.read_csv('clockshift/tabulated_results/subplot_a_theory.csv')
b_data_df = pd.read_csv('clockshift/tabulated_results/subplot_b_data.csv')
b_theory_df = pd.read_csv('clockshift/tabulated_results/subplot_b_theory.csv')
c_data_df = pd.read_csv('clockshift/tabulated_results/subplot_c_data.csv')
c_theory_df = pd.read_csv('clockshift/tabulated_results/subplot_c_theory.csv')

fig, ax = plt.subplots()
ax.errorbar(a_data_df['TTF'], a_data_df['C'], yerr=a_data_df['e_C'], xerr=a_data_df['e_TTF'], ls='')
ax.plot(a_theory_df['TTF'], a_theory_df['C'])
ax.fill_between(a_theory_df['TTF'], a_theory_df['C']*(1-a_theory_df['e_C']),  a_theory_df['C']*(1+a_theory_df['e_C']))

fig, ax = plt.subplots()
ax.errorbar(b_data_df['C'], b_data_df['SW'], yerr=b_data_df['e_SW'], xerr=b_data_df['e_C'], ls='')
ax.plot(b_theory_df['C'], b_theory_df['SW_ZR'])
ax.plot(b_theory_df['C'], b_theory_df['SW_CCC'])
ax.plot(b_theory_df['C'], b_theory_df['SW_SqW'])
ax.fill_between(b_theory_df['C'], b_theory_df['SW_SqW']*(1-b_theory_df['e_SW_SqW']),\
				b_theory_df['SW_SqW']*(1-b_theory_df['e_SW_SqW']))
	
fig, ax= plt.subplots()
ax.errorbar(c_data_df['C'], c_data_df['CS_d'], yerr=c_data_df['e_CS_d'], xerr=c_data_df['e_C'], ls='')
ax.errorbar(c_data_df['C'], c_data_df['CS_HFT'], yerr=c_data_df['e_CS_HFT'], xerr=c_data_df['e_C'], ls='')
ax.errorbar(c_data_df['C'], c_data_df['CS_tot'], yerr=c_data_df['e_CS_tot'], xerr=c_data_df['e_C'], ls='')
ax.plot(c_theory_df['C'], c_theory_df['CS_HFT'], '-')
ax.plot(c_theory_df['C'], c_theory_df['CS_tot'],'-')
ax.plot(c_theory_df['C'], c_theory_df['CS_d'],'-')