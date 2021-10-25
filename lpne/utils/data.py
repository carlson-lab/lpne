"""
Data utilities

"""
__date__ = "July - September 2021"


import numpy as np
import os
from scipy.io import loadmat
import warnings


IGNORED_KEYS = [
    '__header__',
    '__version__',
    '__globals__',
]
"""Ignored keys in the LFP data file"""



def load_lfps(fn):
    """
    Load LFPs from the given filename.

    Parameters
    ----------
    fn : str
        File containing LFP data. Supported file types: {'.mat'}

    Returns
    -------
    lfps : dict
        Maps ROI names to LFP waveforms.
    """
    assert isinstance(fn, str)
    if fn.endswith('.mat'):
        # try:
        lfps = loadmat(fn)
        # except: for old .mat files...
    else:
        raise NotImplementedError(f"Cannot load file: {fn}")
    # Make sure all the channels are 1D float arrays.
    for channel in list(lfps.keys()):
        if channel in IGNORED_KEYS:
            del lfps[channel]
            continue
        try:
            lfps[channel] = np.array(lfps[channel]).astype(np.float).flatten()
        except ValueError:
            warnings.warn(f"Unable to normalize channel: {channel}")
            del lfps[channel]
    return lfps


def save_features(features, fn):
    """
    Save the features to the given filename.

    Raises
    ------
    * NotImplementedError if `fn` is an unsupported file type.

    Parameters
    ----------
    features : dict
        ...
    fn : str
        Where to save the data. Supported file types: {'.npy'}
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
    * `NotImplementedError` if `fn` is an unsupported file type.

    Parameters
    ----------
    fn : str
        Where the data is saved. Supported file types: {'.npy'}

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


def save_labels(labels, fn, overwrite=True):
    """
    Save the labels to the given filename.

    Raises
    ------
    * NotImplementedError if `fn` is an unsupported file type.
    * AssertionError if `fn` exists and `not overwrite`.

    Parameters
    ----------
    labels : numpy.ndarray
        ...
    fn : str
        Where to save the data. Supported file types: {'.npy'}
    overwrite : bool, optional
        Whether to overwrite an existing file with the same name.
    """
    assert isinstance(fn, str), f"fn {fn} is not a string!"
    assert overwrite or not os.path.exists(fn), f"File {fn} exists!"
    if isinstance(labels, list):
        labels = np.array(labels)
    assert isinstance(labels, np.ndarray)
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

    Parameters
    ----------
    fn : str
        Where the data is saved. Supported file types: {'.npy'}

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
