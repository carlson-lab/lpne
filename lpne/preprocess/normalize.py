"""
Normalize features

TO DO
-----
* add more normalization methods
"""
__date__ = "July - October 2021"


import numpy as np

EPSILON = 1e-6



def normalize_features(power_features, partition, mode='max'):
    """
    Normalize the features.

    Parameters
    ----------
    power_features : numpy.ndarray
        LFP power features.
        Shape: [n_windows, n_roi*(n_roi+1)//2, n_freq]
    partition : dict
        'train': numpy.ndarray
            Train indices.
        'test': numpy.ndarray
            Test indices.
    mode : {'max'}, optional
        Normalization method.
        'max': normalize by the maximum value of the training set

    Returns
    -------
    normalized_power_features : numpy.ndarray
        Normalized LFP power features.
        Shape: [n_windows, n_roi*(n_roi+1)//2, n_freq]
    """
    if mode == 'max':
        max_val = np.max(power_features[partition['train']])
        power_features /= max_val
    else:
        raise NotImplementedError(f"Mode {mode} not implemented!")
    return power_features



def normalize_lfps(lfps, mode='zscore'):
    """
    Normalize the LFPs.

    This is an in-place operation!

    Parameters
    ----------
    lfps : dict
        Maps ROI names to waveforms.
    mode : str, optional
        Normalization method.
        'zscore': normalize by z-score

    Returns
    -------
    lfps : dict
        Maps ROI names to waveforms.
    """
    if mode == 'zscore':
        for channel in lfps:
            mean, std = np.mean(lfps[channel]), np.std(lfps[channel])
            lfps[channel] = (lfps[channel] - mean) / (std + EPSILON)
    else:
        raise NotImplementedError(f"Mode {mode} not implemented!")
    return lfps




if __name__ == '__main__':
    pass



###
