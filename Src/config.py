"""
config.py

Default simulation parameters.

This file stores the default parameters used
throughout the project.

Changing values here changes the default
simulation configuration.
"""

# ==========================================================
# Waveform Parameters
# ==========================================================

# Sampling frequency (Hz)
FS = 20e6

# Pulse duration (s)
T = 20e-6

# LFM bandwidth (Hz)
B = 5e6


# ==========================================================
# Target Scenario
# ==========================================================

# Number of targets
K = 3

# Target to perturb
TARGET_INDEX = 1

# Minimum spacing between adjacent targets (samples)
MIN_SPACING = 20

# Maximum delay (samples)
MAX_DELAY = int(FS * T)


# ==========================================================
# Monte Carlo
# ==========================================================

# Monte Carlo trials
N_TRIALS = 500


# ==========================================================
# Signal Model
# ==========================================================

# Target amplitude standard deviation
SIGMA_ALPHA = 1.0

# Signal-to-noise ratio (dB)
SNR_DB = 10.0


# ==========================================================
# ZZB Integration
# ==========================================================

# Maximum perturbation
H_MAX = 40

# Perturbation values
H_VALUES = range(H_MAX + 1)

# ==========================================================
# Experiment Parameters
# ==========================================================

# SNR sweep (dB)
SNR_LIST = [-10, -5, 0, 5, 10, 15, 20]

# Bandwidth sweep (Hz)
BANDWIDTH_LIST = [
    1e6,
    2e6,
    5e6,
    10e6,
    20e6,
]