import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Load the .mat file
from glob import glob
time_form = mdates.DateFormatter('%H:%M')

# get latest path
mpath = glob(r'F:\Data\2026\09 September2026\17September2026\A_Rb_offchip_insitu\imgs\A_Rb_offchip_insitu_history_2026_9_17_11_*.mat')[0]
df = pd.DataFrame(scipy.io.loadmat(mpath, simplify_cells=True))
# # Convert to a Pandas DataFrame and save as CSV
# df.to_csv('output_file.csv', index=False)

mot_path = r"D:\logs\MOTcam_10875\2026-09-17_gaussian_fit.csv"
mot_data_full = pd.read_csv(mot_path)
mot_data = mot_data_full[mot_data_full['Timestamp'] <= 20260917114500] 
mot_data["Timestamp"] = pd.to_datetime(mot_data["Timestamp"], format="%Y%m%d%H%M%S")
mot_data.reset_index(drop=True, inplace=True)

temp_path = r"D:\logs\tsp01_temp/tsp_log_sept2_2026.txt"
temp_path_full = pd.read_csv(temp_path, encoding='latin-1', delimiter='\t', skiprows=17)
temp_df_sept17 = temp_path_full[temp_path_full['Date'] == 'Sep 17 2026']
                           
time_strings = temp_df_sept17["Time"].astype(str).str.split().str[-1]
temp_df_sept17["Plot_Time"] = pd.to_datetime(time_strings, format='%H:%M:%S')

# filter temp data for noon -- Rb MOT after molasses
df_filtered = temp_df_sept17.set_index("Plot_Time").sort_index()
start_time, end_time = '12:00:00', '13:00:00'
df_filtered = df_filtered.between_time(start_time, end_time)

# filter temp data for morning data
df_filtered0 = temp_df_sept17.set_index("Plot_Time").sort_index()
start_time0, end_time0 = '09:30:00', '11:45:00'
df_filtered0 = df_filtered0.between_time(start_time0, end_time0)

# plot
fig, ax = plt.subplots(5, 3, figsize=(19, 8)
                       , sharex='col'
                       )

df.drop(index=range(50), inplace=True)  # Drop the first 50 rows
mot_data.drop(index=range((len(mot_data)-len(df))), inplace=True)  # Drop the first 50 rows

time = mot_data["Timestamp"]

ax[0,0].plot(time, df['N_fit'], label='N fit', color='hotpink', linewidth=2)
ax[0,1].plot(time, df['x_centre'], label='x center', color='hotpink', linewidth=2)
ax[0,2].plot(time, df['y_centre'], label='y center', color='hotpink', linewidth=2)

ax[1,0].plot(time, df['RMSE'], label='RMSE', color='cornflowerblue', linewidth=2)
ax[1,1].plot(time, df['sigmaX'], label='sigmaX', color='cornflowerblue', linewidth=2)
ax[1,2].plot(time, df['sigmaY'], label='sigmaY', color='cornflowerblue', linewidth=2)

ax[2,0].plot(time, df['psd'], label='psd', color='yellowgreen', linewidth=2)
ax[2,1].plot(time, df['Tx']*1e9, label='Tx (nK)', color='yellowgreen', linewidth=2)
ax[2,2].plot(time, df['Ty']*1e9, label='Ty (nK)', color='yellowgreen', linewidth=2)

ax[3,0].plot(time, mot_data['Ntot']/1e7, label='N MOT (1e7)', color='mediumpurple', linewidth=2)
ax[3,1].plot(time, mot_data['Amp'], label='MOT Amp', color='mediumpurple', linewidth=2)

ax[4,0].plot(time, mot_data['xC'], label='MOT x0', color='goldenrod', linewidth=2)
ax[4,1].plot(time, mot_data['yC'], label='MOT y0', color='goldenrod', linewidth=2)

# plot temperature data
step = len(df_filtered0) // len(time)
# Slice the Y data to match the length of time
room_temp = df_filtered0['Temperature[°C]'].iloc[::step][:len(time)]
humidity = df_filtered0['Humidity[%]'].iloc[::step][:len(time)]
ax[3,2].plot(time, room_temp, label='Room Temp [°C]', color='mediumpurple', linewidth=2)
ax[4,2].plot(time, humidity, label='Humidity [%]', color='goldenrod', linewidth=2)

# format time on x axis
for i in range(3):
    ax[-1,i].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax[-1,i].xaxis.set_major_formatter(time_form)
    ax[-1,i].set_xlabel("Time")

for a in ax.flat:
    a.legend(loc=2)

ax[0,0].set_title("Rb Off-Chip Insitu Imaging with K present")
fig.tight_layout()

# Rb after molasses
mot_data = mot_data_full[mot_data_full['Timestamp'] > 20260917120000] 
mot_data = mot_data[mot_data['Timestamp'] < 20260917130000] 
mot_data["Timestamp"] = pd.to_datetime(mot_data["Timestamp"], format="%Y%m%d%H%M%S")
mot_data.reset_index(drop=True, inplace=True)

fig, ax = plt.subplots(4, 2, figsize=(10, 8), sharex='col')

ax[0,0].plot(mot_data["Timestamp"], mot_data["Ntot"]/(10**8), label=f"N (1e8)", color='cornflowerblue')
ax[1,0].plot(mot_data["Timestamp"], mot_data["xC"], label=f"x0", color='hotpink')
ax[2,0].plot(mot_data["Timestamp"], mot_data["yC"], label=f"y0", color='yellowgreen')
ax[3,0].plot(mot_data["Timestamp"], mot_data["Amp"], label=f"Amp", color='mediumpurple')
# ax[4].plot(mot_data["Timestamp"], mot_data["yC"], label=f"y0", color='yellowgreen')
ax[0,1].plot(df_filtered.index, df_filtered["Temperature[°C]"], label=f"Temperature[°C]", color='orange')
ax[3,0].set_xlabel("Time")
ax[3,0].xaxis.set_major_formatter(time_form)
ax[3,1].xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
ax[3,1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax[0,0].set_title("Rb MOT after molasses with K present")
for a in ax.flat:
    a.legend(loc=1)


