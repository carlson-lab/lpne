"""
Data utilities

"""
__date__ = "July 2021"


import numpy as np
from scipy.io import loadmat
import warnings



def load_lfps(fn):
    """

    Parameters
    ----------
    fn : str
        File containing LFP data.

    Returns
    -------
    lfps : dict
        Maps ROI names to LFP waveforms.
    """
    assert isinstance(fn, str)
    if fn.endswith('.mat'):
        # try:
        data = loadmat(fn)
        # except: for old .mat files...
    else:
        raise NotImplementedError(f"Cannot load file: {fn}")
    return data


def save_features(features, fn):
    """
    Save the features to the given filename.

    Raises
    ------
    * NotImplementedError if `fn` is an unsupported file type.

    Paramters
    ---------
    features : dict
        ...
    fn : str
        Where to save the data.
    """
    assert isinstance(fn, str)
    if fn.endswith('.npy'):
        np.save(fn, features)
    else:
        raise NotImplementedError(f"Unsupported file type: {fn}")


def load_features(fn):
    """
    Load the features saved in the given filename.

    Raises
    ------
    * NotImplementedError if `fn` is an unsupported file type.

    Paramters
    ---------
    fn : str
        Where the data is saved.

    Returns
    -------
    features : dict
        LFP features
    """
    assert isinstance(fn, str)
    if fn.endswith('.npy'):
        return np.load(fn, allow_pickle=True).item()
    else:
        raise NotImplementedError(f"Unsupported file type: {fn}")


def save_labels(labels, fn):
    """
    Save the labels to the given filename.

    Raises
    ------
    * NotImplementedError if `fn` is an unsupported file type.

    Paramters
    ---------
    labels : numpy.ndarray
        ...
    fn : str
        Where to save the data.
    """
    assert isinstance(fn, str)
    if fn.endswith('.npy'):
        np.save(fn, labels)
    else:
        raise NotImplementedError(f"Unsupported file type: {fn}")


def load_labels(fn):
    """
    Load the labels saved in the given filename.

    Raises
    ------
    * NotImplementedError if `fn` is an unsupported file type.

    Paramters
    ---------
    fn : str
        Where the data is saved.

    Returns
    -------
    labels : numpy.ndarray
        LFP window labels.
        Shape: [n_windows]
    """
    assert isinstance(fn, str)
    if fn.endswith('.npy'):
        labels = np.load(fn)
    else:
        raise NotImplementedError(f"Unsupported file type: {fn}")
    assert len(labels.shape) == 1, f"len({labels.shape}) != 1"
    return labels



if __name__ == '__main__':
    pass



###
