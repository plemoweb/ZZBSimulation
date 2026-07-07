"""
Experiment 05

SNR Sweep

Evaluate the Bhattacharyya-relaxed ZZB
under different SNR values.
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
# SNR Sweep
# ==========================================================

for i, snr in enumerate(SNR_LIST):

    print("=" * 60)
    print(f"SNR = {snr:.1f} dB ({i+1}/{len(SNR_LIST)})")
    print("=" * 60)

    config.snr_db = snr

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

result_dir = PROJECT_ROOT / "Results"

result_dir.mkdir(exist_ok=True)

np.save(result_dir / "snr_values.npy", SNR_LIST)

np.save(result_dir / "zzb_vs_snr.npy", zzb_all)

np.save(result_dir / "pe_curves.npy", pe_curves)

# ==========================================================
# Figure 1
# ==========================================================

plt.figure(figsize=(7,5))

for i, snr in enumerate(SNR_LIST):

    plt.semilogy(

        H_VALUES,

        pe_curves[i],

        label=f"{snr} dB",

    )

plt.grid(True)

plt.xlabel("Delay perturbation h")

plt.ylabel("Average Pe")

plt.title("Average Pe under different SNR")

plt.legend()

plt.tight_layout()

# ==========================================================
# Figure 2
# ==========================================================

plt.figure(figsize=(6,5))

plt.plot(

    SNR_LIST,

    zzb_all,

    marker="o",

)

plt.grid(True)

plt.xlabel("SNR (dB)")

plt.ylabel("Bhattacharyya-relaxed ZZB")

plt.title("ZZB versus SNR")

plt.tight_layout()

plt.show()