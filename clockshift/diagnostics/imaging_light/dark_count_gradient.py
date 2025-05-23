# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 11:22:39 2025

@author: Chip Lab
"""

import numpy as np

c9_y = 190 - 10
c5_y = 137 - 10

margin_y = [0, 245]

img4_margins = [4.555, 4.498]
img4_slope = (img4_margins[1] - img4_margins[0])/(margin_y[1] - margin_y[0])
img3_margins = [4.641, 4.559]
img3_slope = (img3_margins[1] - img3_margins[0])/(margin_y[1] - margin_y[0])
img2_margins = [4.780, 4.656]
img2_slope = (img2_margins[1] - img2_margins[0])/(margin_y[1] - margin_y[0])

assumed_nominal_margin = img3_slope*c9_y + img3_margins[0]

def calc_OD(e_img2, e_img3, e_img4, absorption):
	# img 4
	no_light_counts = 700
	img4 = no_light_counts * e_img4
		
	# img 3
	light_counts = 3000
	img3 = (light_counts + no_light_counts * e_img3)
	
	# img 2
	atom_counts = (1 + absorption) * light_counts
	img2 = atom_counts + no_light_counts * e_img2
	
	return -np.log((img2 - img4)/(img3 - img4))

for box, c_y, ab in zip(['c9', 'c5'], [c9_y, c5_y], [-0.6, -0.2]):

	e_img2 = (img2_slope*c_y + img2_margins[0])/assumed_nominal_margin
	e_img3 = (img3_slope*c_y + img3_margins[0])/assumed_nominal_margin
	e_img4 = (img4_slope*c_y + img4_margins[0])/assumed_nominal_margin
	
	correct_OD = calc_OD(1, 1, 1, ab)
	real_OD = calc_OD(e_img2, e_img3, e_img4, ab)
	
	print("For box = ", box)
	print("The correct OD is", correct_OD)
	print("The real OD is", real_OD)
	
	print("The error is ", (correct_OD-real_OD)/correct_OD)