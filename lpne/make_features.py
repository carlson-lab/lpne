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
    'fs': FS,
    'window': 'hann',
    'nperseg': 512,
    'noverlap': 256,
    'nfft': None,
}



def make_features(lfps, min_freq=0.0, max_freq=55.0):
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
    min_freq : float, optional
        ...
    max_freq : float, optional
        ....

    Returns
    -------
    res : dict
        'power' : numpy.ndarray
            Power features.
        'freq' : numpy.ndarray
            Frequency bins.
        'rois' : list of str
            Sorted list of grouped channel names.
    """
    rois = sorted(lfps.keys())

    # Make power features for each ROI.
    for i, roi in enumerate(rois):
        lfp_i = lfps[roi].flatten()
        f, Pxx = welch(lfp_i, **SHARED_PARAMS)
        if i == 0:
            i1, i2 = np.searchsorted(f, [min_freq, max_freq])
            power = np.zeros(((n*(n+1))//2, i2-i1))
        idx = (i * (i + 1)) // 2 + i
        power[idx] = Pxx[i1:i2]

    # Make cross power spectral density features for each pair of ROIs.
    for j in range(len(rois)-1):
        lfp_i = lfps[rois[i]].flatten()
        for i in range(j+1, len(rois)):
            lfp_j = lfps[rois[j]].flatten()
            f, Cxy = csd(lfp_i, lfp_j, **SHARED_PARAMS)
            idx = (i * (i + 1)) // 2 + j
            power[idx] = np.abs(Cxy[i1:i2])

    # Assemble features.
    return {
        'power': power,
        'freq': f[i1:i2],
        'rois': rois,
    }




if __name__ == '__main__':
    pass


###
