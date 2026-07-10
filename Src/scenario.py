"""
scenario.py

Target scenario generation.

This module generates target delay configurations
used in Monte Carlo simulations.

Currently supported
-------------------
- Random uniform targets

Future extensions
-----------------
- Fixed targets
- Evenly spaced targets
- User-defined targets
- Clustered targets
"""

from typing import Optional

import numpy as np

from .random_targets import generate_random_targets


def generate_scenario(
    *,
    K: int,
    max_delay: int,
    min_spacing: int = 0,
    mode: str = "random",
    fixed_taus: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate one target scenario.

    Parameters
    ----------
    K
        Number of targets.

    max_delay
        Maximum allowable delay.

    min_spacing
        Minimum spacing between targets.

    mode
        Scenario type.

        "random"
            Uniform random targets.

        "fixed"
            Use user-provided delays.

    fixed_taus
        Required when mode="fixed".

    Returns
    -------
    ndarray
        Sorted target delays.
    """

    if mode == "random":

        return generate_random_targets(
            K=K,
            max_delay=max_delay,
            min_spacing=min_spacing,
        )

    elif mode == "fixed":

        if fixed_taus is None:
            raise ValueError(
                "fixed_taus must be provided when mode='fixed'."
            )

        fixed_taus = np.asarray(fixed_taus, dtype=int)

        if len(fixed_taus) != K:
            raise ValueError(
                "Length of fixed_taus must equal K."
            )

        return np.sort(fixed_taus)

    else:

        raise ValueError(
            f"Unknown scenario mode: {mode}"
        )