import numpy as np

from .delay import delay_signal


def build_signal_matrix(signal, taus):
    """
    Construct signal matrix S.

    Parameters
    ----------
    signal : ndarray

    taus : list

    Returns
    -------
    S
    """

    columns = []

    for tau in taus:

        columns.append(

            delay_signal(signal, tau)

        )

    return np.column_stack(columns)