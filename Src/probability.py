"""
probability.py

Probability functions used in the ZZB simulation.

Current implementation:
    - Bhattacharyya error probability
"""

import numpy as np


def pe_bhattacharyya(BD: float) -> float:
    """
    Compute the Bhattacharyya error probability.

    Paper equation:

        Pe_B = 1/2 * (1 - sqrt(1 - exp(-2*BD)))

    Parameters
    ----------
    BD : float
        Bhattacharyya distance.

    Returns
    -------
    float
        Bhattacharyya error probability.
    """

    # Numerical safety
    x = np.exp(-2.0 * BD)

    # Avoid tiny numerical errors causing x > 1
    x = np.clip(x, 0.0, 1.0)

    pe = 0.5 * (1.0 - np.sqrt(1.0 - x))

    return float(pe)


def similarity(BD: float) -> float:
    """
    Similarity coefficient

        rho = exp(-2*BD)

    Returns
    -------
    float
    """

    return float(np.exp(-2.0 * BD))