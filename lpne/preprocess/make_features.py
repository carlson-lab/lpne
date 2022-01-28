"""
Make features

"""
__date__ = "July - August 2021"


import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, csd


EPSILON = 1e-6
DEFAULT_CSD_PARAMS = {
    'detrend': 'linear',
    'window': 'hann',
    'nperseg': 512,
    'noverlap': 256,
    'nfft': None,
}
"""Default parameters sent to `scipy.signal.csd`"""



def make_features(lfps, fs=1000, min_freq=0.0, max_freq=55.0,
    window_duration=5.0, window_step=None, max_n_windows=None, csd_params={}):
    """
    Main function: make features from an LFP waveform.

    For 0 <= j <= i < n, the cross power spectral density feature for ROI i and
    ROI j is stored at index i * (i + 1) // 2 + j, assuming both i and j are
    zero-indexed. When i == j, this is simply the power spectral density. The
    ROI order is sorted by the channel names.

    See `lpne.unsqueeze_triangular_array` and `lpne.squeeze_triangular_array`
    to convert the power between dense and redundant forms.

    Parameters
    ----------
    lfps : dict
        Maps region names to LFP waveforms.
    fs : int, optional
        LFP samplerate
    min_freq : float, optional
        Minimum frequency
    max_freq : float, optional
        Maximum frequency
    window_duration : float, optional
        Window duration, in seconds
    window_step : None or float, optional
        Time between consecutive window onsets, in seconds. If `None`, this is
        set to `window_duration`.
    max_n_windows : None or int, optional
        Maximum number of windows
    csd_params : dict, optional
        Parameters sent to `scipy.signal.csd`

    Returns
    -------
    res : dict
        'power' : numpy.ndarray
            Power features.
            Shape: [n_window, n_roi*(n_roi+1)//2, n_freq]
        'freq' : numpy.ndarray
            Frequency bins.
            Shape: [n_freq]
        'rois' : list of str
            Sorted list of grouped channel names.
    """
    if window_step is None:
        window_step = window_duration
    assert window_step > 0.0, f"Nonpositive window step: {window_step}"
    assert max_n_windows is None or max_n_windows > 0
    rois = sorted(lfps.keys())
    n = len(rois)
    assert n >= 1, f"{n} < 1"
    duration = len(lfps[rois[0]]) / fs
    assert duration >= window_duration, \
            f"LFPs are too short: {duration} < {window_duration}"
    window_samp = int(fs * window_duration)
    onsets = np.arange(
            0.0,
            duration - window_duration + EPSILON,
            window_step,
    )
    if max_n_windows is not None:
        onsets = onsets[:max_n_windows]

    # Make cross power spectral density features for each pair of ROIs.
    csd_params = {**DEFAULT_CSD_PARAMS, **csd_params}
    for i in range(len(rois)):
        lfp_i = lfps[rois[i]].flatten()
        for j in range(i+1):
            lfp_j = lfps[rois[j]].flatten()
            if i == 0 and j == 0:
                k = window_samp
                f, _ = csd(lfp_i[:k], lfp_j[:k], fs=fs, **csd_params)
                i1, i2 = np.searchsorted(f, [min_freq, max_freq])
                power = np.zeros((len(onsets), (n*(n+1))//2, i2-i1))
            idx = (i * (i + 1)) // 2 + j
            for k in range(len(onsets)):
                k1 = int(fs*onsets[k])
                k2 = k1 + window_samp
                _, Cxy = csd(lfp_i[k1:k2], lfp_j[k1:k2], fs=fs, **csd_params)
                power[k,idx] = np.abs(Cxy[i1:i2])

    # Get 1/f-scaled power features.
    freq = f[i1:i2]
    power[:,:] *= freq

    # Assemble and return features.
    return {
        'power': power,
        'freq': freq,
        'rois': rois,
    }



if __name__ == '__main__':
    pass


###
