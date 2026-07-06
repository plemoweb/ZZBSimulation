import numpy as np


def generate_lfm(fs, T, B):
    """
    Generate a complex baseband LFM waveform.

    Parameters
    ----------
    fs : float
        Sampling frequency.

    T : float
        Pulse duration.

    B : float
        Bandwidth.

    Returns
    -------
    t : ndarray
        Time axis.

    s : ndarray
        Complex waveform.
    """

    N = int(fs*T)

    t = np.arange(N)/fs

    t = t - T/2

    k = B/T

    s = np.exp(1j*np.pi*k*t**2)

    return t, s