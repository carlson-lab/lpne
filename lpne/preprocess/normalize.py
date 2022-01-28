"""
Normalize LFPs and features.

"""
__date__ = "July 2021 - January 2022"


import numpy as np

EPSILON = 1e-6



def normalize_features(power_features, partition=None, mode='std'):
    """
    Normalize the features.

    This is an in-place operation!

    Parameters
    ----------
    power_features : numpy.ndarray
        LFP power features.
        Shape: ``[n_windows, n_roi*(n_roi+1)//2, n_freq]``
    partition : None or dict, optional
        If ``None``, use all the indices to calculate window statistics.
        Otherwise, use only the indices contained in ``partition['train']``.
        ``'train'``: numpy.ndarray
            Train indices.
        ``'test'``: numpy.ndarray
            Test indices.
    mode : {'max'}, optional
        Normalization method.
        ``'std'``: normalize by the standard deviation of the training set.
        ``'max'``: normalize by the maximum value of the training set, scaling
                   to [0,1].

    Returns
    -------
    normalized_power_features : numpy.ndarray
        Normalized LFP power features.
        Shape: ``[n_windows, n_roi*(n_roi+1)//2, n_freq]``
    """
    if partition is None:
        idx = np.arange(len(power_features))
    else:
        idx = partition['train']
    if mode == 'std':
        temp = np.std(power_features[idx])
        power_features /= temp
    elif mode == 'max':
        max_val = np.max(power_features[idx])
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
        ``'zscore'``: normalize by z-score

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
