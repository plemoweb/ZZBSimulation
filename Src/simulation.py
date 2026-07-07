"""
simulation.py

High-level simulation pipeline for ZZB.

This module combines all low-level modules in src/
to perform one complete simulation.

Current stage:
    Waveform
        ↓
    Signal Matrix
        ↓
    Gram Matrix
        ↓
    Bhattacharyya Distance

Later:
    Probability
        ↓
    ZZB
"""

import numpy as np

from .signal_matrix import build_signal_matrix
from .gram import gram_matrix
from .bhattacharyya import bhattacharyya_distance
from .probability import pe_bhattacharyya, similarity

def compute_three_grams(
    waveform,
    taus,
    target_index,
    h,
):
    """
    Construct

        G0
        G1
        Gavg

    corresponding to

        R(tau)

        R(tau+h)

        Rav
    """

    K = len(taus)

    # -------------------------
    # Original hypothesis
    # -------------------------

    S0 = build_signal_matrix(
        waveform,
        taus,
    )

    G0 = gram_matrix(S0)

    # -------------------------
    # Shifted hypothesis
    # -------------------------

    taus_shift = taus.copy()

    taus_shift[target_index] += h

    S1 = build_signal_matrix(
        waveform,
        taus_shift,
    )

    G1 = gram_matrix(S1)

    # -------------------------
    # Average hypothesis
    # -------------------------

    columns = []

    for i in range(K):

        if i != target_index:

            columns.append(S0[:, i])

    columns.append(S0[:, target_index])

    columns.append(S1[:, target_index])

    U = np.column_stack(columns)

    C = np.eye(K + 1)

    C[-2, -2] = 0.5
    C[-1, -1] = 0.5

    sqrtC = np.sqrt(C)

    Gavg = sqrtC @ U.conj().T @ U @ sqrtC

    return G0, G1, Gavg

def simulate_once(
    waveform,
    taus,
    target_index,
    h,
    sigma_alpha=1.0,
    snr_db=10.0,
):
    """
    Perform one complete simulation.

    Returns
    -------

    dict
    """



    G0, G1, Gavg = compute_three_grams(
        waveform,
        taus,
        target_index,
        h,
    )

    BD = bhattacharyya_distance(
        G0,
        G1,
        Gavg,
        sigma_alpha,
        snr_db,
    )

    Pe = pe_bhattacharyya(BD)

    return {

        "waveform": waveform,

        "taus": taus,

        "shifted_taus": np.where(
            np.arange(len(taus)) == target_index,
            taus + h * (np.arange(len(taus)) == target_index),
            taus,
        ),

        "G0": G0,

        "G1": G1,

        "Gavg": Gavg,

        "BD": BD,

        "Pe": Pe
    }