"""
Experiment 06

Bandwidth Sweep

Evaluate the Bhattacharyya-relaxed ZZB
under different bandwidth values.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from Src.config import *
from Src.config_class import default_config

from Src.monte_carlo import monte_carlo_pe

from Src.valley_filling import valley_filling
from Src.zzb import compute_zzb


# ==========================================================
# Configuration
# ==========================================================

config = default_config()

# ==========================================================
# Results
# ==========================================================

zzb_all = []

pe_curves = []

# ==========================================================
# Bandwidth Sweep
# ==========================================================

for i, bw in enumerate(BANDWIDTH_LIST):

    print("=" * 60)
    print(f"Bandwidth = {bw:.1f} MHz ({i+1}/{len(BANDWIDTH_LIST)})")
    print("=" * 60)

    config.B = bw

    pe_curve = []

    for h in H_VALUES:

        result = monte_carlo_pe(

            config=config,

            h=h,

        )

        pe_curve.append(result["mean_pe"])

        print(
            f"h={h:2d}    Pe={result['mean_pe']:.6e}"
        )

    pe_curve = np.asarray(pe_curve)

    pe_curves.append(pe_curve)

    # --------------------------------------
    # Valley Filling
    # --------------------------------------

    pe_vf = valley_filling(pe_curve)

    # --------------------------------------
    # ZZB
    # --------------------------------------

    zzb = compute_zzb(

        h_values=H_VALUES,

        pe_values=pe_vf,

        prior_width=config.max_delay,

    )

    zzb_all.append(zzb)

    print()

    print(f"Estimated ZZB = {zzb:.6f}")

    print()

# ==========================================================
# Convert
# ==========================================================

zzb_all = np.asarray(zzb_all)

pe_curves = np.asarray(pe_curves)

# ==========================================================
# Save
# ==========================================================

result_dir = PROJECT_ROOT / "Results" /"Experiment_06_Bandwidth_Sweep"

result_dir.mkdir(exist_ok=True)

np.save(result_dir / "bandwidth_values.npy", BANDWIDTH_LIST)

np.save(result_dir / "zzb_vs_bandwidth.npy", zzb_all)

np.save(result_dir / "pe_curves.npy", pe_curves)

# ==========================================================
# Figure 1
# ==========================================================
figure_dir = PROJECT_ROOT / "Figures" / "Experiment_06_Bandwidth_Sweep"
figure_dir.mkdir(exist_ok=True)

plt.figure(figsize=(7,5))

for i, bw in enumerate(BANDWIDTH_LIST):

    plt.semilogy(

        H_VALUES,

        pe_curves[i],

        label=f"{bw:.1f} MHz",

    )

plt.grid(True)

plt.xlabel("Delay perturbation h")

plt.ylabel("Average Pe")

plt.title("Average Pe under different Bandwidth")

plt.legend()

plt.tight_layout()

plt.savefig(
    figure_dir / "exp06_average_pe.png",
    dpi=300,
)
# ==========================================================
# Figure 2
# ==========================================================

plt.figure(figsize=(6,5))

plt.plot(

    BANDWIDTH_LIST,

    zzb_all,

    marker="o",

)

plt.grid(True)

plt.xlabel("Bandwidth (MHz)")

plt.ylabel("Bhattacharyya-relaxed ZZB")

plt.title("ZZB versus Bandwidth")

plt.tight_layout()

plt.savefig(
    figure_dir / "exp06_zzb_vs_bandwidth.png",
    dpi=300,
)
plt.show()