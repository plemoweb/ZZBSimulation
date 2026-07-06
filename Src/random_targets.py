"""
random_targets.py

Random target delay generation for ZZB simulations.

The target delays are assumed to follow a uniform prior,
subject to a minimum spacing constraint.
"""

import numpy as np


def generate_random_targets(
    K: int,
    max_delay: int,
    min_spacing: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate K uniformly distributed target delays.

    Parameters
    ----------
    K : int
        Number of targets.

    max_delay : int
        Maximum allowed delay (inclusive of the search range's upper bound
        is handled by randint's exclusive endpoint).

    min_spacing : int
        Minimum spacing between adjacent targets (samples).

    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    ndarray
        Sorted integer delays.
    """

    if rng is None:
        rng = np.random.default_rng()

    if K <= 0:
        raise ValueError("K must be positive.")

    if max_delay <= 0:
        raise ValueError("max_delay must be positive.")

    if min_spacing < 0:
        raise ValueError("min_spacing must be non-negative.")

    while True:

        taus = np.sort(
            rng.integers(
                low=0,
                high=max_delay,
                size=K,
            )
        )

        if K == 1:
            return taus

        if np.min(np.diff(taus)) >= min_spacing:
            return taus