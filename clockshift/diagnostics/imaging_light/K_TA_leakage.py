# -*- coding: utf-8 -*-
"""
2025-04-08
@author: Chip Lab
"""

import numpy as np
# from clockshift.diagnostics.dark_count_gradient import assumed_nominal_margin

# shift pixels by 50 for easy slope calculation
y_pixel_shift = 50
c9_y = 190 - y_pixel_shift
c5_y = 137 - y_pixel_shift

margin_y = [50-y_pixel_shift, 250-y_pixel_shift]

img4_margins = [4.77, 4.60]
img4_slope = (img4_margins[1] - img4_margins[0])/(margin_y[1] - margin_y[0])
img3_margins = [5.00, 4.83]
img3_slope = (img3_margins[1] - img3_margins[0])/(margin_y[1] - margin_y[0])
img2_margins = [4.94, 4.80]
img2_slope = (img2_margins[1] - img2_margins[0])/(margin_y[1] - margin_y[0])

assumed_nominal_margin = 4.580755102040817  # from dark_count_gradient

def calc_OD(e_img2, e_img3, e_img4):
	# img 4
	no_light_counts = 700
	img4 = no_light_counts * e_img4
		
	# img 3
	light_counts = 3000
	img3 = (light_counts + no_light_counts * e_img3)
	
	# img 2
	absorption = -0.6
	atom_counts = (1 + absorption) * light_counts
	img2 = atom_counts + no_light_counts * e_img2
	
	return -np.log((img2 - img4)/(img3 - img4))

for box, c_y in zip(['c9', 'c5'], [c9_y, c5_y]):

	e_img2 = (img2_slope*c_y + img2_margins[0])/assumed_nominal_margin
	e_img3 = (img3_slope*c_y + img3_margins[0])/assumed_nominal_margin
	e_img4 = (img4_slope*c_y + img4_margins[0])/assumed_nominal_margin
	
	correct_OD = calc_OD(1, 1, 1)
	real_OD = calc_OD(e_img2, e_img3, e_img4)
	
	print("For box = ", box)
	print("The correct OD is", correct_OD)
	print("The real OD is", real_OD)
	
	print("The error is ", (correct_OD-real_OD)/correct_OD)