"""
monte_carlo.py

Monte Carlo simulation utilities.

This module repeatedly calls simulate_once()
to estimate the expectation with respect to
random target locations.

Mathematically,

    E_tau [ Pe ]

is approximated by

    (1/N) * sum Pe_i

Author:
"""

import numpy as np

from .simulation import simulate_once
from .random_targets import generate_random_targets


def monte_carlo_pe(
    *,
    num_trials: int,
    fs: float,
    T: float,
    B: float,
    K: int,
    target_index: int,
    h: int,
    max_delay: int,
    min_spacing: int,
    sigma_alpha: float = 1.0,
    snr_db: float = 10.0,
):
    """
    Monte Carlo estimation of Pe(h).

    Parameters
    ----------
    num_trials
        Number of Monte Carlo trials.

    fs
        Sampling frequency.

    T
        Pulse duration.

    B
        Chirp bandwidth.

    K
        Number of targets.

    target_index
        Target whose delay is perturbed.

    h
        Delay perturbation.

    max_delay
        Maximum allowable delay.

    min_spacing
        Minimum spacing between targets.

    sigma_alpha
        Target amplitude standard deviation.

    snr_db
        Signal-to-noise ratio.

    Returns
    -------
    dict
    """

    pe_all = []
    bd_all = []

    last_result = None

    for _ in range(num_trials):

        # ----------------------------------
        # Random target generation
        # ----------------------------------

        taus = generate_random_targets(
            K=K,
            max_delay=max_delay,
            min_spacing=min_spacing,
        )

        # ----------------------------------
        # One complete simulation
        # ----------------------------------

        result = simulate_once(
            fs=fs,
            T=T,
            B=B,
            taus=taus,
            target_index=target_index,
            h=h,
            sigma_alpha=sigma_alpha,
            snr_db=snr_db,
        )

        pe_all.append(result["Pe"])
        bd_all.append(result["BD"])

        last_result = result

    pe_all = np.asarray(pe_all)
    bd_all = np.asarray(bd_all)

    return {

        # -------------------------
        # Monte Carlo averages
        # -------------------------

        "mean_pe": float(np.mean(pe_all)),

        "mean_bd": float(np.mean(bd_all)),

        # -------------------------
        # Monte Carlo statistics
        # -------------------------

        "std_pe": float(np.std(pe_all)),

        "std_bd": float(np.std(bd_all)),

        # -------------------------
        # Raw samples
        # -------------------------

        "pe_all": pe_all,

        "bd_all": bd_all,

        # -------------------------
        # Number of trials
        # -------------------------

        "num_trials": num_trials,

        # -------------------------
        # Last realization
        # (useful for visualization)
        # -------------------------

        "last_result": last_result,
    }