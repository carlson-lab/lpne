"""
Data utilities

"""
__date__ = "July 2021 - January 2022"


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


def load_features(fns, return_counts=False):
    """
    Load the features saved in the given filenames.

    Parameters
    ----------
    fns : str or list of str
        Where the data is saved. Supported file types: {'.npy'}
    return_counts : bool, optional
        Return the number of windows for each file.

    Returns
    -------
    features : numpy.ndarray
        LFP power features
        Shape: [n_windows,feature_dim]
    rois : list of str
        ROI names
    counts : list of int
        Number of windows in each file. Returned if `return_counts` is `True`.
    """
    if isinstance(fns, str):
        fns = [fns]
    assert isinstance(fns, list)
    features, counts = [], []
    prev_rois = None
    for fn in fns:
        assert fn.endswith('.npy'), f"Unsupported file type: {fn}"
        temp = np.load(fn, allow_pickle=True).item()
        rois = temp['rois']
        if prev_rois is not None:
            assert prev_rois == rois, f"Inconsitent ROIs: {rois} != {prev_rois}"
        prev_rois = rois
        features.append(temp['power'])
        counts.append(len(features[-1]))
    features = np.concatenate(features, axis=0)
    if return_counts:
        return features, rois, counts
    return features, rois


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


def load_features_and_labels(feature_fns, label_fns, group_func=None,
    return_counts=False):
    """
    Load the features and labels.

    If `group_func` is specified, then groups are also returned.

    Parameters
    ----------
    feature_fns : list of str
        Feature filenames
    label_fns : list of str
        Corresponding label filenames
    group_func : None or function, optional
        If `None`, no groups are returned. Otherwise, groups are defined as
        `group_func(feature_fn)` for the corresponding feature filename of each
        window. `group_func` should map strings (filenames) to integers
        (groups).
    return_counts : bool, optional
        Return the number of windows for each file.

    Returns
    -------
    features : numpy.ndarray
        Shape: [n_windows,feature_dim]
    labels : numpy.ndarray
        Shape: [n_windows]
    rois : list of str
        Regions of interest, channel names
    groups : numpy.ndarray
        Returned if `group_func is not None`
        Shape: [n_windows]
    counts : list of int
        Number of windows in each file. Returned if `return_counts` is `True`.
    """
    assert group_func is None or isinstance(group_func, type(lambda x: x)), \
            "group_func must either be None or a function!"
    assert len(feature_fns) == len(label_fns), \
            f"Expected the same number of feature and label filenames. " \
            f"Found {len(feature_fns)} and {len(label_fns)}."
    # Collect everything.
    features, labels, groups, counts = [], [], [], []
    prev_rois = None
    for feature_fn, label_fn in zip(feature_fns, label_fns):
        power, rois = load_features(feature_fn)
        if prev_rois is not None:
            assert prev_rois == rois, f"Inconsitent ROIs: {rois} != {prev_rois}"
        prev_rois = rois
        features.append(power)
        counts.append(len(power))
        labels.append(load_labels(label_fn))
        if group_func is not None:
            groups.append([group_func(feature_fn)]*len(labels[-1]))
    # Concatenate and return.
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    res = (features, labels, rois)
    if group_func is not None:
        groups = np.concatenate(groups, axis=0)
        res += (groups,)
    if return_counts:
        res += (counts,)
    return res



if __name__ == '__main__':
    pass



###
