"""
Bhattacharyya distance computation.

This module implements the Bhattacharyya distance used in
Section III of the ZZB paper.

Author:
"""

import numpy as np


def logdet_from_gram(
    G: np.ndarray,
    sigma_alpha: float = 1.0,
    sigma_noise: float = 1.0,
) -> float:
    """
    Compute log(det(R)) using the Matrix Determinant Lemma.

    R = sigma_noise^2 I + sigma_alpha^2 S S^H

    Parameters
    ----------
    G : ndarray
        Gram matrix S^H S

    sigma_alpha : float
        Target amplitude standard deviation.

    sigma_noise : float
        Noise standard deviation.

    Returns
    -------
    float
        log(det(R))
    """

    K = G.shape[0]

    A = np.eye(K) + (sigma_alpha**2 / sigma_noise**2) * G

    sign, logdet = np.linalg.slogdet(A)

    if sign <= 0:
        raise ValueError("Matrix determinant is not positive.")

    return logdet


def bhattacharyya_distance(
    G0: np.ndarray,
    G1: np.ndarray,
    Gavg: np.ndarray,
    sigma_alpha: float = 1.0,
    sigma_noise: float = 1.0,
) -> float:
    """
    Compute Bhattacharyya distance.

    Parameters
    ----------
    G0
        Gram matrix under H0

    G1
        Gram matrix under H1

    Gavg
        Gram matrix corresponding to average covariance

    Returns
    -------
    float
        Bhattacharyya distance
    """

    logdet0 = logdet_from_gram(
        G0,
        sigma_alpha,
        sigma_noise,
    )

    logdet1 = logdet_from_gram(
        G1,
        sigma_alpha,
        sigma_noise,
    )

    logdet_avg = logdet_from_gram(
        Gavg,
        sigma_alpha,
        sigma_noise,
    )

    BD = (
        0.5 * logdet_avg
        - 0.25 * logdet0
        - 0.25 * logdet1
    )

    return float(BD)


def similarity(BD: float) -> float:
    """
    Similarity coefficient.

    rho = exp(-BD)
    """

    return float(np.exp(-BD))