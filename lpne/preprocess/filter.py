"""
Filter LFP waveforms.

"""
__date__ = "October 2021 - February 2022"


import numpy as np
from scipy.signal import butter, lfilter, stft, iirnotch, freqz, welch


ORDER = 3 # Butterworth filter order
"""Butterworth filter order"""
LOWCUT = 0.5
"""Default lowcut for filtering (Hz)"""
HIGHCUT = 55.0
"""Default highcut for filtering (Hz)"""
Q = 1.5 # Notch filter parameter
"""Notch filter quality parameter"""



def filter_signal(x, fs, lowcut=LOWCUT, highcut=HIGHCUT, q=Q, order=ORDER):
    """
    Apply a bandpass filter and notch filters to the signal.

    Parameters
    ----------
    x : numpy.ndarray
    fs : float
    lowcut : float
    highcut : float
    q : float
    order : int

    Returns
    -------
    x : numpy.ndarray
    """
    assert x.ndim == 1, f"len({x.shape}) != 1"
    # Remove NaNs.
    nan_mask = np.isnan(x)
    x[nan_mask] = 0.0
    # Bandpass.
    x = _butter_bandpass_filter(x, lowcut, highcut, fs, order=ORDER)
    # Remove electrical noise at 60Hz and harmonics.
    for freq in range(60,int(highcut),60):
        b, a = iirnotch(freq, q, fs)
        x = lfilter(b, a, x)
    # Reintroduce NaNs.
    x[nan_mask] = np.nan
    return x


def filter_lfps(lfps, fs, lowcut=LOWCUT, highcut=HIGHCUT, q=Q, order=ORDER):
    """
    Apply a bandpass filter and notch filters to all the LFPs.

    Parameters
    ----------
    lfps : dict
        Maps channel names to waveforms.
    fs : float
    lowcut : float
    highcut : float
    q : float
    order : int

    Returns
    -------
    lfps : dict
    """
    for channel in list(lfps.keys()):
        lfps[channel] = filter_signal(
                lfps[channel],
                fs,
                lowcut=lowcut,
                highcut=highcut,
                q=q,
                order=order,
        )
    return lfps


def _butter_bandpass(lowcut, highcut, fs, order=ORDER):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def _butter_bandpass_filter(data, lowcut, highcut, fs, order=ORDER):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



if __name__ == '__main__':
    pass


###
