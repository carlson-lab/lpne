"""
Filter LFP and EMG waveforms.

"""
__date__ = "October 2021"


from scipy.signal import butter, lfilter, stft, iirnotch, freqz, welch


ORDER = 3 # Butterworth filter order
Q = 1.5 # Notch filter parameter



def filter_signal(x, fs, lowcut, highcut, q=Q, order=ORDER):
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
    # Bandpass.
    x = _butter_bandpass_filter(x, lowcut, highcut, fs, order=ORDER)
    # Remove electrical noise at 60Hz and harmonics.
    for freq in range(60,int(highcut),60):
        b, a = iirnotch(freq, q, fs)
        x = lfilter(b, a, x)
    return x


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
