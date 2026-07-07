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
from .scenario import generate_scenario
from .waveform import generate_lfm

from .config_class import SimulationConfig

def monte_carlo_pe(
    *,
    config: SimulationConfig,
    h: int,
):
    """
    Monte Carlo estimation of Pe(h).

    Parameters
    ----------
    config
        Simulation configuration.

    h
        Delay perturbation.

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

    _, waveform = generate_lfm(
    config.fs,
    config.T,
    config.B,
    )

    for _ in range(config.num_trials):

        # ----------------------------------
        # Random target generation
        # ----------------------------------

        taus = generate_scenario(
            K=config.K,
            max_delay=config.max_delay,
            min_spacing=config.min_spacing,
            mode="random",
        )

        # ----------------------------------
        # One complete simulation
        # ----------------------------------

        result = simulate_once(

            waveform=waveform,

            taus=taus,

            target_index=config.target_index,

            h=h,

            sigma_alpha=config.sigma_alpha,

            snr_db=config.snr_db,
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

        "num_trials": config.num_trials,

        # -------------------------
        # Last realization
        # (useful for visualization)
        # -------------------------

        "last_result": last_result,
    }