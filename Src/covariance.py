import numpy as np


def covariance_matrix(
    S,
    sigma_noise=1,
    sigma_alpha=1
):
    """
    Construct covariance matrix.

    R = sigma_n^2 I + sigma_alpha^2 SS^H
    """

    N = S.shape[0]

    R = sigma_noise**2 * np.eye(N, dtype=np.complex128)

    R += sigma_alpha**2 * (S @ S.conj().T)

    return R