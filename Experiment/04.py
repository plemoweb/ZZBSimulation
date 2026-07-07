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


# ============================================================
# Experiment Parameters
# ============================================================

TARGET_INDEX = 1

H_VALUES = np.arange(0, 41)

N_TRIALS = 500


# ============================================================
# Monte Carlo
# ============================================================

average_pe = []

print("=" * 60)
print("Monte Carlo Simulation")
print("=" * 60)

for h in H_VALUES:

    pe_sum = 0.0

    max_delay = int(T * FS) - int(h)

    for _ in range(N_TRIALS):

        taus = generate_random_targets(
            K=K,
            max_delay=max_delay,
            min_spacing=MIN_SPACING,
        )

        result = simulate_once(
            fs=FS,
            T=T,
            B=B,
            taus=taus,
            target_index=TARGET_INDEX,
            h=int(h),
            snr_db=-10,
        )

        pe_sum += result["Pe"]

    pe_mean = pe_sum / N_TRIALS

    average_pe.append(pe_mean)

    print(f"h = {h:2d}    Pe = {pe_mean:.6e}")

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