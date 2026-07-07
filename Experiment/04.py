"""
Experiment 04

Bhattacharyya-relaxed Ziv-Zakai Bound (Equation 40)

Workflow
--------
Random Targets
    ↓
Simulation
    ↓
Bhattacharyya Distance
    ↓
Bhattacharyya Error Probability
    ↓
Monte Carlo Average
    ↓
Equation (40)
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
from Src.simulation import simulate_once
from Src.random_targets import generate_random_targets
from Src.zzb import compute_zzb
from Src.monte_carlo import monte_carlo_pe


# ============================================================
# Experiment Parameters
# ============================================================

TARGET_INDEX = 1

H_VALUES = np.arange(0, 41)

N_TRIALS = 500

snr_db = -10.0
# ============================================================
# Monte Carlo
# ============================================================

average_pe = []

print("=" * 60)
print("Monte Carlo Simulation")
print("=" * 60)

for h in H_VALUES:

    result = monte_carlo_pe(
        h=h,
        num_trials=N_TRIALS,
        snr_db=snr_db,
        fs=FS,
        T=T,
        B=B,
        K=K,
        target_index=TARGET_INDEX,
        max_delay=int(T * FS),
        min_spacing=MIN_SPACING,
    )

    average_pe.append(result["mean_pe"])

average_pe = np.array(average_pe)


# ============================================================
# Compute Integrand
# ============================================================

prior_width = int(T * FS)

weight = 1.0 - H_VALUES / prior_width

weight = np.maximum(weight, 0)

integrand = H_VALUES * weight * average_pe


# ============================================================
# Compute ZZB
# ============================================================

zzb = compute_zzb(
    h_values=H_VALUES,
    pe_values=average_pe,
    prior_width=prior_width,
)


# ============================================================
# Output
# ============================================================

print()
print("=" * 60)
print("Bhattacharyya-relaxed ZZB")
print("=" * 60)
print(f"Prior Width : {prior_width}")
print(f"Monte Carlo : {N_TRIALS}")
print(f"Maximum h   : {H_VALUES[-1]}")
print()
print(f"Estimated ZZB = {zzb:.6f}")
print("=" * 60)


# ============================================================
# Save Figures
# ============================================================

figure_dir = PROJECT_ROOT / "Figures"
figure_dir.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Figure 1
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    H_VALUES,
    average_pe,
    linewidth=2,
)

plt.grid(True)

plt.xlabel("Delay Perturbation h (samples)")
plt.ylabel(r"$P_{e,B}^{(k)}(h)$")
plt.title("Average Bhattacharyya Error Probability")

plt.tight_layout()

plt.savefig(
    figure_dir / "exp04_average_pe.png",
    dpi=300,
)


# ------------------------------------------------------------
# Figure 2
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    H_VALUES,
    integrand,
    linewidth=2,
)

plt.grid(True)

plt.xlabel("Delay Perturbation h (samples)")
plt.ylabel(r"$h(1-h/T)P_e(h)$")
plt.title("Integrand of Equation (40)")

plt.tight_layout()

plt.savefig(
    figure_dir / "exp04_integrand.png",
    dpi=300,
)


plt.show()