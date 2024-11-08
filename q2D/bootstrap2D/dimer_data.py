# -*- coding: utf-8 -*-
"""
July 13  2023
@author: Colin Dale
	Class loadData of q2D dimer data to bootstrap fit. This file mimics
	Ben Olsen's pwave_data.py file that was used for the 3D dimer
	bootstrap fitting. Instead of having all data and metadata in this
	file, I am just pulling it from .dat files and the 
	2Ddimer_parameters.csv metadata file.
	
To do: 
	- Only load csv once, perhaps make a dictionary.
	- Load .dat data from saved python file, have flag option to reload.
	- Add all data.
	- Add data range selection
"""
from get_data import data_from_dat, from_csv

names_of_interest = ['freq','c9'] 	# column names to pull from .dat
csv_name = "2Ddimer_parameters.csv" 	# meta data file
data_folder = "data//"

debug = False

class loadData:
	def __init__(self, run: str):
		self.run = run
		names_of_interest = ['freq','c9'] 
#  Test file ###########################

		if run == 'test':
			self.LAT = 2
			self.dimer = 'y'
			self.B = 198.0
			self.exclude = False
			self.saturating = False
			self.pGuess = [21383, 4, 10.7,-107]
			self.pGuessG = [21383, 4, 10.7,-120]
			self.freq97 = 44.1723
			self.excludeRanges = [[44.1, 44.13]]
			self.data = data_from_dat(data_folder+'dimers2d_2022-08-12_F_LAT2_40ER_198G_FM'+'.dat', names_of_interest)
			
#  p-wave dimers ###########################
#2
		elif run == 'dimers2d_2022-08-12_F_LAT2_40ER_198G_FM':
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [21383, 4, 10.7,-107]
			self.pGuessG = [21383, 4, 10.7,-120]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.1, 44.14]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#3	
		elif run == 'dimers2d_2022-08-12_G_LAT2_40ER_198p3G_FM_y':
			self.run = 'dimers2d_2022-08-12_G_LAT2_40ER_198p3G_FM'
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [19000, 12, 12,-49]
			self.pGuessG = [21383, 12, 10.7,-59]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[43,43.98],[44.176,44.23]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#4
		elif run == 'dimers2d_2022-08-16_E_LAT1_80ER_198p3G_FM_y':
			self.run = 'dimers2d_2022-08-16_E_LAT1_80ER_198p3G_FM'
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [19000, 12, 12,-67]
			self.pGuessG = [21383, 12, 10.7,-59]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[43.98,44.05],[44.18,44.2]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#4 z
		elif run == 'dimers2d_2022-08-16_E_LAT1_80ER_198p3G_FM_z':
			self.run = 'dimers2d_2022-08-16_E_LAT1_80ER_198p3G_FM'
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer2')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [19000, 120, 12,-165]
			self.pGuessG = [21383, 12, 10.7,-59]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.1,44.1555],[44.18,44.2]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#8 z
		elif run == 'dimers2d_2022-08-17_N_LAT1_80ER_198p3G_FM_z':
			self.run = 'dimers2d_2022-08-17_N_LAT1_80ER_198p3G_FM'
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer2')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [9000, 120, 12,-165]
			self.pGuessG = [21383, 12, 10.7,-59]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[43.99,44.015]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#9 z
		elif run == 'dimers2d_2022-08-18_E_LAT1_80ER_198p4G_FM_z':
			self.run = 'dimers2d_2022-08-18_E_LAT1_80ER_198p4G_FM'
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer2')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [19000, 120, 12,-144]
			self.pGuessG = [21383, 12, 10.7,-59]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.02,44.05],[44.11,45]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#10 y
		elif run == 'dimers2d_2022-08-18_F_LAT1_80ER_198p2G_FM_y':
			self.run = 'dimers2d_2022-08-18_F_LAT1_80ER_198p2G_FM'
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer2')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [19000, 120, 12,-85]
			self.pGuessG = [21383, 12, 10.7,-59]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[43,44.04]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#10 z
		elif run == 'dimers2d_2022-08-18_F_LAT1_80ER_198p2G_FM_z':
			self.run = 'dimers2d_2022-08-18_F_LAT1_80ER_198p2G_FM'
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer2')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [19000, 120, 12,-165]
			self.pGuessG = [21383, 12, 10.7,-59]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[43.955,43.985],[44.07,45]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#23 y
		elif run == 'dimers2d_2022-11-30_K_LAT1_80ER_198p5G_FM_y':
			self.run = 'dimers2d_2022-11-30_K_LAT1_80ER_198p5G_FM'
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer2')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [19000, 120, 12,-40]
			self.pGuessG = [21383, 12, 10.7,-59]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.09,44.127],[44.23,44.27]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#23 z
		elif run == 'dimers2d_2022-11-30_K_LAT1_80ER_198p5G_FM_z':
			self.run = 'dimers2d_2022-11-30_K_LAT1_80ER_198p5G_FM'
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer2')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [19000, 120, 12,-170]
			self.pGuessG = [21383, 12, 10.7,-59]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.15, 44.27]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
            
#  s-wave dimers            
#3 z    
		elif run == 'dimers2d_2022-08-12_G_LAT2_40ER_198p3G_FM_z':
			self.run = 'dimers2d_2022-08-12_G_LAT2_40ER_198p3G_FM'
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [24000, 2, 10.7,-230]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.11,44.23]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#4 x    
		elif run == 'dimers2d_2022-08-16_E_LAT1_80ER_198p3G_FM_x':
			self.run = 'dimers2d_2022-08-16_E_LAT1_80ER_198p3G_FM'
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [17000, 2, 10.7,-190]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.11,44.21]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#8 x   
		elif run == 'dimers2d_2022-08-17_N_LAT1_80ER_198p3G_FM_x':
			self.run = 'dimers2d_2022-08-17_N_LAT1_80ER_198p3G_FM'
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [7000, 2, 10.7,-214]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.03,44.05]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#9 x   
		elif run == 'dimers2d_2022-08-18_E_LAT1_80ER_198p4G_FM_x':
			self.run = 'dimers2d_2022-08-18_E_LAT1_80ER_198p4G_FM'
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [18000, 2, 10.7,-200]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.06, 44.095],[44.125,45]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#10 x   
		elif run == 'dimers2d_2022-08-18_F_LAT1_80ER_198p2G_FM_x':
			self.run = 'dimers2d_2022-08-18_F_LAT1_80ER_198p2G_FM'
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [24000, 2, 10.7,-230]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[43.995,44.015],[44.075,45]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#11 x   
		elif run == 'dimers2d_2022-08-18_G_LAT1_80ER_198p6G_FM':
			self.run = 'dimers2d_2022-08-18_G_LAT1_80ER_198p6G_FM'
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [24000, 2, 10.7,-230]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.34,44.42]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#12 x   
		elif run == 'dimers2d_2022-08-18_I_LAT1_80ER_199p1G_FM':
			self.run = 'dimers2d_2022-08-18_I_LAT1_80ER_199p1G_FM'
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [10000, 2, 10.7,-70]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.293,44.425]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#13 z
		elif run == 'dimers2d_2022-08-22_P_LAT2_40ER_198p4G_FM':
			self.LAT = from_csv(csv_name, run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [24000, 2, 10.7,-230]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.115,45]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#14 z
		elif run == 'dimers2d_2022-08-22_Q_LAT2_40ER_198p4G_FM_evenwaveonly':
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [24000, 2, 10.7,-230]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[0,0]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#16 z
		elif run == 'dimers2d_2022-08-23_G_LAT2_40ER_199p4G_FM':
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [24000, 2, 10.7,-40]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.38,44.58]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#21 z
		elif run == 'dimers2d_2022-08-24_J_LAT2_40ER_198G_FM':
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [30000, 2, 10.7,-330]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[43.97,44.11]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#22 z
		elif run == 'dimers2d_2022-08-24_K_LAT2_40ER_199p5G_FM':
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [24000, 2, 10.7,-50]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[0,0]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)

#26 z
		elif run == 'dimers2d_2023-03-12_B_LAT2_120ER_199p5G_FM':
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [6000, 2, 10.7,-150]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.33,44.4],[44.44,45]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#27 z
		elif run == 'dimers2d_2023-03-12_C_LAT2_120ER_199p7G_FM':
			names_of_interest = ['freq', 'c9']
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [6000, 2, 10.7,-100]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.37,45]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#28 z
		elif run == 'dimers2d_2023-03-13_G_LAT2_120ER_199p8G_FM':
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [7000, 2, 10.7,-100]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[0,0]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)		
#30 z
		elif run == 'dimers2d_2023-03-14_H_LAT2_120ER_198p5G_FM97kHz':
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [7000, 2, 10.7,-300]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44,45]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#31 z
		elif run == 'dimers2d_2023-03-14_H_LAT2_120ER_198p5G_FM92kHz':
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-234]
			self.pGuessG = [7000, 2, 10.7,-300]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44,45]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#33 z
		elif run == 'dimers2d_2023-05-01_H_LAT2_120ER_200G_FM_8msblackman':
			names_of_interest = ['freq', 'c9']
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-100]
			self.pGuessG = [8000, 2, 10.7,-60]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.45, 45]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#34 z
		elif run == 'dimers2d_2023-05-01_O_LAT2_120ER_200p1G_FM_8msblackman':
			names_of_interest = ['freq', 'sum95']
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-100]
			self.pGuessG = [12000, 2, 10.7,-50]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.485,44.53]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
#35 z
		elif run == 'dimers2d_2023-05-03_B_LAT2_120ER_200G_FM_blackman_8ms5p5V':
			self.LAT = from_csv(csv_name, self.run, 'LAT')
# 			self.dimer = from_csv(csv_name, self.run, 'dimer')
			self.B = float(from_csv(csv_name, self.run, 'B'))
			self.exclude = False
			self.saturating = False
			self.pGuess = [24000, 12, 13,-70]
			self.pGuessG = [13000, 2, 10.7,-70]
			self.freq97 = float(from_csv(csv_name, self.run, 'freq97'))
			self.excludeRanges = [[44.45,45]]
			self.data = data_from_dat(data_folder+self.run+'.dat', names_of_interest)
			
###### DEBUGGING ######			
if debug == True:
	test_run = 'test'
	data = loadData(test_run).data
	
	test_run = 'dimers2d_2022-08-12_G_LAT2_40ER_198p3G_FM'
	data = loadData(test_run)
	
	column_name = 'B'
	run_name = 'dimers2d_2022-08-16_E_LAT1_80ER_198p3G_FM'
	B = float(from_csv(csv_name, run_name, column_name))
	
	
	
