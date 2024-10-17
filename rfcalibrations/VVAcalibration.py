# -*- coding: utf-8 -*-
"""
Author: ChipLab
"""


### Vpp calibration
VVAtoVppfilename = "VVAtoVpp.txt"
VVAtoVppfile = os.path.join(root, VVAtoVppfilename) # calibration file
VVAs, Vpps = np.loadtxt(VVAtoVppfile, unpack=True)
VpptoOmegaR = 27.5833 # kHz

def VVAtoVpp(VVA, calibration_file):
	"""Match VVAs to calibration file values. Will get mad if the VVA is not
		also in the file. """
	Vpp = 0
	for i, VVA_val in enumerate(VVAs):
		if VVA == VVA_val:
			Vpp = Vpps[i]
	if Vpp == 0: # throw a fit if VVA not in list.
		raise ValueError("VVA value {} not in VVAtoVpp.txt".format(VVA))
	return Vpp