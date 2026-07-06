"""
Experiment 02

Bhattacharyya Distance versus delay perturbation h
"""

from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from Src.simulation import simulate_once
from Src.config import *


# -----------------------------
# Fixed scenario
# -----------------------------

taus = np.array([60, 120, 200])

target_index = 1

# -----------------------------
# Scan h
# -----------------------------

h_values = np.arange(0, 51)

BD_values = []

for h in h_values:

    result = simulate_once(
        fs=FS,
        T=T,
        B=B,
        taus=taus,
        target_index=target_index,
        h=h,
    )

    BD_values.append(result["BD"])


# -----------------------------
# Plot
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(
    h_values,
    BD_values,
    linewidth=2
)

plt.grid(True)

plt.xlabel("Delay perturbation h (samples)")

plt.ylabel("Bhattacharyya Distance")

plt.title("Bhattacharyya Distance vs Delay Perturbation")

plt.tight_layout()
fig_dir = PROJECT_ROOT / "figures"
fig_dir.mkdir(exist_ok=True)

plt.savefig(fig_dir / "exp02_bd_vs_h.png", dpi=300, bbox_inches="tight")
plt.show()



