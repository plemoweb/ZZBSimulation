import numpy as np


def delay_signal(signal, delay):
    """
    Integer delay with zero padding.
    """

    delayed = np.zeros_like(signal)

    delayed[delay:] = signal[:-delay]

    return delayed