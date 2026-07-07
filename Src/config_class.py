"""
config_class.py

Simulation configuration.
"""

from dataclasses import dataclass


@dataclass
class SimulationConfig:

    # -------------------------------------------------
    # Waveform
    # -------------------------------------------------

    fs: float
    T: float
    B: float

    # -------------------------------------------------
    # Scenario
    # -------------------------------------------------

    K: int

    target_index: int

    min_spacing: int

    max_delay: int

    # -------------------------------------------------
    # Monte Carlo
    # -------------------------------------------------

    num_trials: int

    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------

    sigma_alpha: float = 1.0

    snr_db: float = 10.0

from .config import *

def default_config():

    return SimulationConfig(

        fs=FS,

        T=T,

        B=B,

        K=K,

        target_index=TARGET_INDEX,

        min_spacing=MIN_SPACING,

        max_delay=MAX_DELAY,

        num_trials=N_TRIALS,

        sigma_alpha=SIGMA_ALPHA,

        snr_db=SNR_DB,
    )