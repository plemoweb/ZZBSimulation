import numpy as np

def delay_signal(signal, delay):
    """
    Delay a discrete-time signal by an integer number of samples.
    Samples shifted beyond the observation window are discarded,
    and new samples are zero-padded.
    """
    delayed = np.zeros_like(signal)

    if delay <= 0:
        delayed[:] = signal
        return delayed

    if delay >= len(signal):
        return delayed

    delayed[delay:] = signal[:-delay]

    return delayed