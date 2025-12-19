
# settings for directories, standard packages...
from preamble import *
from library import colors, kB
import os

runs = {"data": "2025-12-15_O"}#, "bg":"2025-12-15_N"}
	

filename = r'e:\Data\2025\12 December2025\15December2025\L_dimer_spinmix_scan\2025-12-15_O_e.dat'
bg_filename = r'e:\Data\2025\12 December2025\15December2025\N_spinmix_bg_scan\2025-12-15_N_e.dat'

data = Data(filename.split("\\")[-1], path=filename)

# rename VVA columns on first page to spin_mix_VVA
data.data['spin_mix_VVA'] = data.data['VVA'].values

# add new VVA col for dimer pulse VVA
data.data['VVA'] = 9

# data.data['trf'] = 10e-6 # s
# data.data['freq'] = 43.240e6 # Hz
# data.data['EF'] = 13.94e3 # Hz

# df_spinmix = pd.DataFrame({'VVA':np.unique(data.data['spin_mix_VVA'])})
    
# for VVA in np.unique(data.data['spin_mix_VVA']):
#     # dumb way to do this
#     datVVA = Data(filename.split("\\")[-1], path=filename)
#     datVVA.data = data.data[data.data['spin_mix_VVA'] == VVA].copy(deep=True)

#     datVVA.analysis(bgVVA = 0, pulse_type="square")
#     datVVA.group_by_mean('spin_mix_VVA')
#     df = datVVA.avg_data

#     df_spinmix.loc[df_spinmix['VVA'] == VVA, ['fraction95', 'contact_dimer', 'em_contact_dimer', 'scaledtransfer_dimer', 'em_scaledtransfer_dimer']] = \
#     df[['fraction95', 'contact_dimer', 'em_contact_dimer', 'scaledtransfer_dimer', 'em_scaledtransfer_dimer']].iloc[0].values

    
# df_spinmix['norm_sig'] = df_spinmix['scaledtransfer_dimer'] / df_spinmix['scaledtransfer_dimer'].max()
# df_spinmix['em_norm_sig'] = df_spinmix['em_scaledtransfer_dimer'] / df_spinmix['scaledtransfer_dimer'].max()
# df_spinmix['norm_C'] = df_spinmix['contact_dimer'] / df_spinmix['contact_dimer'].max()
# df_spinmix['em_norm_C'] = df_spinmix['em_contact_dimer'] / df_spinmix['contact_dimer'].max()

# plt.errorbar(df_spinmix['fraction95'], df_spinmix['contact_dimer'], df_spinmix['em_contact_dimer'])
# plt.xlabel("fraction95")
# plt.ylabel("single shot dimer measured C")