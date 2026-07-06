"""
Experiment 00

Visualize the transmitted LFM waveform.
"""

from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from Src.config import *
from Src.waveform import generate_lfm


# ============================================================
# Generate waveform
# ============================================================

t, waveform = generate_lfm(
    fs=FS,
    T=T,
    B=B,
)



# ============================================================
# Magnitude
# ============================================================

magnitude = np.abs(waveform)

# ============================================================
# Phase
# ============================================================

phase = np.unwrap(np.angle(waveform))

# ============================================================
# Instantaneous Frequency
# ============================================================

dt = 1 / FS

inst_freq = np.diff(phase) / (2 * np.pi * dt)

t_freq = t[:-1]

# ============================================================
# Figure Directory
# ============================================================

figure_dir = PROJECT_ROOT / "Figures"
figure_dir.mkdir(exist_ok=True)

# ============================================================
# Figure 1
# ============================================================

plt.figure(figsize=(8,4))

plt.plot(t * 1e6, np.real(waveform))

plt.grid(True)

plt.xlabel("Time (μs)")

plt.ylabel("Amplitude")

plt.title("Real Part of LFM Waveform")

plt.tight_layout()

plt.savefig(
    figure_dir / "waveform_real.png",
    dpi=300,
)

# ============================================================
# Figure 2
# ============================================================

plt.figure(figsize=(8,4))

plt.plot(t * 1e6, np.imag(waveform))

plt.grid(True)

plt.xlabel("Time (μs)")

plt.ylabel("Amplitude")

plt.title("Imaginary Part of LFM Waveform")

plt.tight_layout()

plt.savefig(
    figure_dir / "waveform_imag.png",
    dpi=300,
)

# ============================================================
# Figure 3
# ============================================================

plt.figure(figsize=(8,4))

plt.plot(t * 1e6, magnitude)

plt.grid(True)

plt.xlabel("Time (μs)")

plt.ylabel("Magnitude")

plt.title("Magnitude of LFM Waveform")

plt.tight_layout()

plt.savefig(
    figure_dir / "waveform_magnitude.png",
    dpi=300,
)

# ============================================================
# Figure 4
# ============================================================

plt.figure(figsize=(8,4))

plt.plot(t_freq * 1e6, inst_freq / 1e6)

plt.grid(True)

plt.xlabel("Time (μs)")

plt.ylabel("Frequency (MHz)")

plt.title("Instantaneous Frequency")

plt.tight_layout()

plt.savefig(
    figure_dir / "waveform_frequency.png",
    dpi=300,
)

plt.show()