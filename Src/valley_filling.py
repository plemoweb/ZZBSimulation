"""
valley_filling.py

Valley Filling Operator used in the Ziv-Zakai Bound.
"""

import numpy as np


def valley_filling(values: np.ndarray) -> np.ndarray:
    """
    Perform the valley-filling operation.

    For a discrete function f(h),

        V(f(h)) = max_{t >= h} f(t)

    Parameters
    ----------
    values : ndarray
        Function samples.

    Returns
    -------
    ndarray
        Valley-filled samples.
    """

    values = np.asarray(values, dtype=float)

    filled = np.empty_like(values)

    filled[-1] = values[-1]

    # Traverse from right to left
    for i in range(len(values) - 2, -1, -1):

        filled[i] = max(values[i], filled[i + 1])

    return filled