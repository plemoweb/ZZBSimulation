import numpy as np


def bhattacharyya_bound(bd):

    """
    Pairwise error probability upper bound

    Parameters
    ----------
    bd : float

    Returns
    -------
    Pe
    """

    return 0.5*np.exp(-bd)