"""
Experiment 03

Monte Carlo approximation of

P_{e,B}^{(k)}(h)

from the ZZB paper.
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
from Src.random_targets import generate_random_targets

# ============================================================
# Parameters
# ============================================================

K = 3

TARGET_INDEX = 1

MAX_DELAY = int(T * FS)

MIN_SPACING = 20

N_TRIALS = 500

H_VALUES = np.arange(0, 41)



# ============================================================
# Monte Carlo
# ============================================================

average_pe = []

for h in H_VALUES:

    pe_sum = 0.0

    for _ in range(N_TRIALS):
        max_delay = int(T * FS) - h

        taus = generate_random_targets(
            K=K,
            max_delay=max_delay,
            min_spacing=MIN_SPACING
        )

        result = simulate_once(
            fs=FS,
            T=T,
            B=B,
            taus=taus,
            target_index=TARGET_INDEX,
            h=h,
        )

        pe_sum += result["Pe"]

    average_pe.append(pe_sum / N_TRIALS)

    print(
        f"h={h:2d}   Pe={average_pe[-1]:.6e}"
    )


average_pe = np.array(average_pe)


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(
    H_VALUES,
    average_pe,
    linewidth=2,
)

plt.grid(True)

plt.xlabel("Delay perturbation h (samples)")

plt.ylabel(r"$P_{e,B}^{(k)}(h)$")

plt.title("Average Bhattacharyya Error Probability")

plt.tight_layout()

# Save figure
figure_dir = PROJECT_ROOT / "Figures"
figure_dir.mkdir(exist_ok=True)

plt.savefig(
    figure_dir / "exp03_average_probability.png",
    dpi=300,
)

plt.show()