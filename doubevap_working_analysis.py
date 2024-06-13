# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:47:08 2024

@author: coldatoms
"""

from data_class import *

# df = Data('2024-06-06_L_UHfit.dat').data
# totf = df.groupby(['field','evaptime']).agg({'ToTF':['mean','std']}).reset_index()
# Ns =df.groupby(['field','evaptime']).agg({'N':['mean','std']}).reset_index()

# fig, axs = plt.subplots(2,1)
# fields = df.field.unique()
# axT = axs[0]
# axN = axs[1]

# for field in fields:
# 	totfsub = totf[totf['field']==field]
# 	Nsub = Ns[Ns['field']==field]
# 	axT.errorbar(totfsub.evaptime, totfsub['ToTF','mean'], yerr=totfsub['ToTF','std'], capsize=2, label=str(field), marker='o', ls='')
# 	axN.errorbar(Nsub.evaptime, Nsub['N','mean'], yerr=Nsub['N','std'], capsize=2, label=str(field), marker='o', ls='')
# axT.legend()
# axN.legend()
# axT.set_ylabel('ToTF')
# axN.set_ylabel('N')
# axT.set_xlabel('evap time [ms]')
# axN.set_xlabel('evap time [ms]')


# df = Data('2024-06-06_M_UHfit.dat').data
# totf = df.groupby(['ODT1','evaptime']).agg({'ToTF':['mean','std']}).reset_index()
# Ns =df.groupby(['ODT1','evaptime']).agg({'N':['mean','std']}).reset_index()

# fig, axs = plt.subplots(2,1)
# odts = df.ODT1.unique()
# axT = axs[0]
# axN = axs[1]

# for odt in odts:
# 	totfsub = totf[totf['ODT1']==odt]
# 	Nsub = Ns[Ns['ODT1']==odt]
# 	axT.errorbar(totfsub.evaptime, totfsub['ToTF','mean'], yerr=totfsub['ToTF','std'], capsize=2, label=str(odt), marker='o', ls='')
# 	axN.errorbar(Nsub.evaptime, Nsub['N','mean'], yerr=Nsub['N','std'], capsize=2, label=str(odt), marker='o', ls='')
# axT.legend()
# axN.legend()
# axT.set_ylabel('ToTF')
# axN.set_ylabel('N')
# axT.set_xlabel('evap time [ms]')
# axN.set_xlabel('evap time [ms]')


# df = Data('2024-06-07_D_UHfit.dat').data
# totf = df.groupby(['ODT1','ODT2','evaptime']).agg({'ToTF':['mean','std']}).reset_index()
# Ns =df.groupby(['ODT1','ODT2','evaptime']).agg({'N':['mean','std']}).reset_index()

# fig, axs = plt.subplots(2,1)
# odts = (df.ODT1.unique(), df.ODT2.unique())
# axT = axs[0]
# axN = axs[1]

# for i, odt1 in enumerate(odts[0]):
# 	if odt1==0.06:
# 		continue
# 	totfsub = totf[(totf['ODT1']==odts[0][i])]
# 	Nsub = Ns[(Ns['ODT1']==odts[0][i])]
# 	labelstr = 'odt1=' + str(odts[0][i]) + ', odt2=' + str(odts[1][i])
# 	axT.errorbar(totfsub.evaptime, totfsub['ToTF','mean'], yerr=totfsub['ToTF','std'], capsize=2, label=labelstr, marker='o', ls='')
# 	axN.errorbar(Nsub.evaptime, Nsub['N','mean'], yerr=Nsub['N','std'], capsize=2, label=labelstr, marker='o', ls='')
# axT.legend()
# axN.legend()
# axT.set_ylabel('ToTF')
# axN.set_ylabel('N')
# axT.set_xlabel('evap time [ms]')
# axN.set_xlabel('evap time [ms]')


# df = Data('2024-06-10_E_UHfit.dat').data
# totf = df.groupby(['odt1','odt2','time']).agg({'ToTF':['mean','std']}).reset_index()
# Ns =df.groupby(['odt1','odt2','time']).agg({'N':['mean','std']}).reset_index()

# fig, axs = plt.subplots(2,1)
# odts = (df.odt1.unique(), df.odt2.unique())
# axT = axs[0]
# axN = axs[1]

# for i, odt1 in enumerate(odts[0]):
# 	if odt1==0.06:
# 		continue
# 	totfsub = totf[(totf['odt1']==odts[0][i])]
# 	Nsub = Ns[(Ns['odt1']==odts[0][i])]
# 	labelstr = 'odt1=' + str(odts[0][i]) + ', odt2=' + str(odts[1][i])
# 	axT.errorbar(totfsub.time, totfsub['ToTF','mean'], yerr=totfsub['ToTF','std'], capsize=2, label=labelstr, marker='o', ls='')
# 	axN.errorbar(Nsub.time, Nsub['N','mean'], yerr=Nsub['N','std'], capsize=2, label=labelstr, marker='o', ls='')
# axT.legend()
# axN.legend()
# axT.set_ylabel('ToTF')
# axN.set_ylabel('N')
# axT.set_xlabel('evap time [ms]')
# axN.set_xlabel('evap time [ms]')


df = Data('2024-06-12_ZB_UHfit.dat').data
totf = df.groupby(['FB','time']).agg({'ToTF':['mean','std']}).reset_index()
Ns =df.groupby(['FB','time']).agg({'N':['mean','std']}).reset_index()

fig, axs = plt.subplots(2,1)
fbs = df.FB.unique()
axT = axs[0]
axN = axs[1]

for i, fb in enumerate(fbs):
	totfsub = totf[(totf['FB']==fb)]
	Nsub = Ns[(Ns['FB']==fb)]
	labelstr = 'FB=' + str(fb)
	axT.errorbar(totfsub.time, totfsub['ToTF','mean'], yerr=totfsub['ToTF','std'], capsize=2, label=labelstr, marker='o', ls='')
	axN.errorbar(Nsub.time, Nsub['N','mean'], yerr=Nsub['N','std'], capsize=2, label=labelstr, marker='o', ls='')
axT.legend()
axN.legend()
axT.set_ylabel('ToTF')
axN.set_ylabel('N')
axT.set_xlabel('evap time [ms]')
axN.set_xlabel('evap time [ms]')



