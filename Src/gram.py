import numpy as np


def gram_matrix(S):
    """
    Compute Gram matrix.
    """

    return S.conj().T @ S