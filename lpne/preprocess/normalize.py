"""
Normalize features

TO DO
-----
* add more normalization methods
"""
__date__ = "July 2021"


import numpy as np



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



if __name__ == '__main__':
    pass



###
