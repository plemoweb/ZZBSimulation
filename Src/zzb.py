"""
zzb.py

Bhattacharyya-relaxed ZZB
Equation (40) of the paper.
"""

import numpy as np


def compute_zzb(
    h_values,
    pe_values,
    prior_width,
):
    """
    Compute the Bhattacharyya-relaxed ZZB.

    Equation (40):

        ZZB = ∫ h(1-h/T)Pe(h) dh
    """

    h_values = np.asarray(h_values, dtype=float)

    pe_values = np.asarray(pe_values, dtype=float)

    weighting = 1.0 - h_values / prior_width

    weighting = np.maximum(weighting, 0.0)

    integrand = h_values * weighting * pe_values

    zzb = np.trapezoid(
        integrand,
        h_values,
    )

    return float(zzb)