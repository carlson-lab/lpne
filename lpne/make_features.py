"""
Make features

"""
__date__ = "July 2021"


import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, csd


FN = 'test_data/Mouse5_012514_Mouse5_Sleep_01_LFP.mat'
FS = 1000

SHARED_PARAMS = {
    'detrend': 'linear',
    'window': 'hann',
    'nperseg': 512,
    'noverlap': 256,
    'nfft': None,
}



def make_features(lfps, fs=1000, min_freq=0.0, max_freq=55.0,
    window_duration=10.0):
    """
    Main function: make features from an LFP waveform.

    For 0 <= j <= i < n, the cross power spectral density feature for ROI i and
    ROI j is stored at index i * (i + 1) // 2 + j, assuming both i and j are
    zero-indexed. When i == j, this is simply the power spectral density. The
    ROI order is sorted by the channel names.

    min/max frequency
    frequency scaling
    Include a JSON parameter file?
    normalization

    Parameters
    ----------
    lfps : dict
        Maps region names to LFP waveforms.
    fs : int, optional
        LFP samplerate
    min_freq : float, optional
        ...
    max_freq : float, optional
        ....
    window_duration : float, optional
        Window duration

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
    rois = sorted(lfps.keys())
    n = len(rois)
    window_samp = int(fs * window_duration)

    print("TEMP: TRUNCATING FILES")
    for key in lfps:
        lfps[key] = lfps[key].flatten()[:3*window_samp]

    # Make power features for each ROI.
    for i, roi in enumerate(rois):
        lfp_i = lfps[roi].flatten()
        if i == 0:
            n_window = int(np.floor(len(lfp_i) / window_samp))
            f, _ = welch(lfp_i, fs=fs, **SHARED_PARAMS)
            i1, i2 = np.searchsorted(f, [min_freq, max_freq])
            power = np.zeros((n_window, (n*(n+1))//2, i2-i1))
        idx = (i * (i + 1)) // 2 + i
        for k in range(n_window):
            k1, k2 = k*window_samp, (k+1)*window_samp
            _, Pxx = welch(lfp_i[k1:k2], fs=fs, **SHARED_PARAMS)
            power[k,idx] = Pxx[i1:i2]

    # Make cross power spectral density features for each pair of ROIs.
    for j in range(len(rois)-1):
        lfp_i = lfps[rois[i]].flatten()
        for i in range(j+1, len(rois)):
            lfp_j = lfps[rois[j]].flatten()
            idx = (i * (i + 1)) // 2 + j
            for k in range(n_window):
                k1, k2 = k*window_samp, (k+1)*window_samp
                _, Cxy = csd(lfp_i[k1:k2], lfp_j[k1:k2], fs=fs, **SHARED_PARAMS)
                power[k,idx] = np.abs(Cxy[i1:i2])

    # Get 1/f-scaled power features. 
    freq = f[i1:i2]
    power[:,:] *= freq

    # Assemble features.
    return {
        'power': power,
        'freq': freq,
        'rois': rois,
    }




if __name__ == '__main__':
    pass


###
