import pandas as pd 
from data_class import Data
import numpy as np

minus1MHz = Data('2025-04-14_O_e.dat')
minus1MHzzero = minus1MHz.data[minus1MHz.data['Frequency'] == 46.2227] #-1 MHz detuned @ 3VVA

zeroVVA = Data('2025-04-14_N_e.dat')
zeroVVA = zeroVVA.data[zeroVVA.data['VVA'] == 0]

zeroVVA_closedshutter = Data('2025-04-14_J_e.dat')
zeroVVA_closedshutter = zeroVVA_closedshutter.data[zeroVVA_closedshutter.data['VVA'] == 0]
