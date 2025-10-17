from .constants import ahf, gI, gJ, mu_B, h, I
import numpy as np


def EhfFieldInTesla(B, F, mF):
    """Computes the hyperfine energy in Joules for a given magnetic field B (in Tesla),
    total angular momentum F, and its projection mF using the Breit-Rabi formula."""
    x = (gJ - gI) * mu_B * B / (ahf * (I + 1/2))
    return -ahf/4 + gI * mu_B * mF * B + (-1)**(F - 1/2) * (ahf * (I + 1/2))/2 * np.sqrt( 
        1 + (4 * mF)/(2*I + 1) * x + x**2)


def Ehf(B, F, mF):
    """Computes the hyperfine energy in Joules for a given magnetic field B (in Gauss),
    total angular momentum F, and its projection mF using the Breit-Rabi formula."""
    return EhfFieldInTesla(1e-4 * B, F, mF)  # Field input given in Gauss


def FreqMHz(B, F1, mF1, F2, mF2):
    """Computes the transition frequency in MHz between two hyperfine states
    (F1, mF1) and (F2, mF2) at a given magnetic field B (in Gauss)."""
    return 1e-6/h * (Ehf(B, F1, mF1) - Ehf(B, F2, mF2))