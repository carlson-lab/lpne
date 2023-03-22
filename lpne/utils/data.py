"""
Data utilities

"""
__date__ = "July 2021 - March 2023"
__all__ = [
    "load_channel_map",
    "load_features",
    "load_features_and_labels",
    "load_labels",
    "load_lfps",
    "save_features",
    "save_labels",
]


import h5py
import numpy as np
import os

PANDAS_INSTALLED = True
try:
    import pandas as pd
except ModuleNotFoundError:
    PANDAS_INSTALLED = False
from scipy.io import loadmat
import warnings


from .. import MATLAB_IGNORED_KEYS


def load_channel_map(fn):
    """
    Load the channel map from the file.

    Parameters
    ----------
    fn : str
        Filename of the channel map file

    Returns
    -------
    channel_map : dict
        Maps channel names (str) to region names (str)
    """
    if fn.endswith(".csv"):
        channel_map = {}
        with open(fn, "r") as f:
            for ln, line in enumerate(f):
                parts = [p.strip() for p in line.split(",") if len(p.strip()) > 0]
                if len(parts) == 0:
                    continue
                assert len(parts) == 2, f"Expected two columns on line {ln+1}!"
                assert parts[0] not in channel_map, f"Duplicate entry on line {ln+1}!"
                channel_map[parts[0]] = parts[1]
        return channel_map
    elif fn.endswith(".xlsx"):
        assert PANDAS_INSTALLED, f"Pandas needs to be installed to read: {fn}"
        f = pd.read_excel(fn, index_col=False, header=None)
        channel_map_full = f.set_index(0).to_dict()[1]
        return channel_map_full
    else:
        raise NotImplementedError(f"Cannot load the channel map file: {fn}")


def load_features(fns, return_counts=False, feature="power", return_freqs=False):
    """
    Load the features saved in the given filenames.

    Parameters
    ----------
    fns : str or list of str
        Where the data is saved. Supported file types: {'.npy'}
    return_counts : bool, optional
        Return the number of windows for each file.
    feature : str, optional
        Which feature in {"power","dir_spec"} to load.

    Returns
    -------
    features : numpy.ndarray
        LFP power features or directed spectrum features.
        Power Shape: ``[n_windows,(n_roi)*(n_roi+1)/2,n_freqs]``
        Dir Spec Shape: ``[n_windows,n_roi,n_roi,n_freqs]``
    rois : list of str
        ROI names
    counts : list of int
        Number of windows in each file. Returned if `return_counts` is `True`.
    frequencies : np.ndarray
        Feature frequencies. Returned if ``return_freqs`` is ``True``.
    """
    assert feature in ["power", "dir_spec"], f"Unsupported feature: {feature}"
    if isinstance(fns, str):
        fns = [fns]
    assert isinstance(fns, list)
    features, counts = [], []
    prev_rois, prev_freqs = None, None
    for fn in fns:
        assert fn.endswith(".npy"), f"Unsupported file type: {fn}"
        temp = np.load(fn, allow_pickle=True).item()
        rois = temp["rois"]
        freqs = temp["freq"]
        if prev_rois is not None:
            assert prev_rois == rois, f"Inconsitent ROIs: {rois} != {prev_rois}"
        if prev_freqs is not None:
            assert np.allclose(
                prev_freqs, freqs
            ), f"Inconsitent frequencies: {freqs} != {prev_freqs}"
        prev_rois = rois
        prev_freqs = freqs
        features.append(temp[feature])
        counts.append(len(features[-1]))
    features = np.concatenate(features, axis=0)
    res = (features, rois)
    if return_counts:
        res += (counts,)
    if return_freqs:
        res += (freqs,)
    return res


def load_features_and_labels(
    feature_fns,
    label_fns,
    group_func=None,
    group_map=None,
    return_counts=False,
    soft_labels=False,
    return_freqs=False,
):
    """
    Load the features and labels.

    If ``group_func`` or ``group_map`` is specified, then groups are also returned.

    Parameters
    ----------
    feature_fns : list of str
        Feature filenames
    label_fns : list of str
        Corresponding label filenames
    group_func : None or function, optional
        Groups are defined as ``group_func(feature_fn)`` for the corresponding feature
        filename of each window. ``group_func`` should map strings (filenames) to
        integers (groups).
    group_map : None or dict, optional
        An alternative way of specifying groups. Maps filenames to integers specifying
        groups.
    return_counts : bool, optional
        Whether to return the number of windows for each file
    soft_labels : bool, optional
        If labels are given as probabilities, don't perform an argmax operation.
    return_freqs : bool, optional
        Whether to return the frequencies associated with the features

    Returns
    -------
    features : numpy.ndarray
        Shape: [n_windows,feature_dim]
    labels : numpy.ndarray
        Shape: [n_windows]
    rois : list of str
        Regions of interest, channel names
    groups : numpy.ndarray
        Returned if ``group_func is not None``
        Shape: [n_windows]
    counts : list of int
        Number of windows in each file. Returned if ``return_counts`` is ``True``.
    frequencies : np.ndarray
        Feature frequencies. Returned if ``return_freqs`` is ``True``.
    """
    assert group_func is None or isinstance(
        group_func, type(lambda x: x)
    ), "group_func must either be None or a function!"
    assert group_map is None or isinstance(
        group_map, dict
    ), "group_map must either be None or a dict"
    assert (
        group_func is None or group_map is None
    ), "Only one of group_map and group_func can be specified!"
    assert len(feature_fns) == len(label_fns), (
        f"Expected the same number of feature and label filenames. "
        f"Found {len(feature_fns)} and {len(label_fns)}."
    )
    # Translate the group_map to a group_func.
    if group_map is not None:

        def group_func(fn):
            for key in group_map:
                if key in os.path.split(fn)[1]:
                    return group_map[key]
            raise NotImplementedError(fn)

    # Collect everything.
    features, labels, groups, counts = [], [], [], []
    prev_rois, prev_freqs = None, None
    for i, (feature_fn, label_fn) in enumerate(zip(feature_fns, label_fns)):
        power, rois, freqs = load_features(feature_fn, return_freqs=True)
        if prev_rois is not None:
            assert prev_rois == rois, (
                f"Inconsitent ROIs: {prev_rois} != {rois}"
                f"\n\tFile 1: {feature_fns[i-1]}"
                f"\n\tFile 2: {feature_fns[i]}"
            )
        prev_rois = rois
        if prev_freqs is not None:
            assert np.allclose(freqs, prev_freqs), (
                f"Inconsistent frequencies: {prev_freqs} != {freqs}"
                f"\n\tFile 1: {feature_fns[i-1]}"
                f"\n\tFile 2: {feature_fns[i]}"
            )
        prev_freqs = freqs
        features.append(power)
        counts.append(len(power))
        labels.append(load_labels(label_fn))
        assert len(power) == len(labels[-1]), (
            f"Number of windows doesn't match for feature and label file!"
            f"\n\tFeatures: {feature_fn} (len({power.shape}))"
            f"\n\tLabels: {label_fn} ({len(labels[-1])})"
        )
        if group_func is not None:
            groups.append([group_func(feature_fn)] * len(labels[-1]))
    # Concatenate and return.
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    if labels.ndim == 2 and not soft_labels:
        labels = np.argmax(labels, axis=1)
    res = (features, labels, rois)
    if group_func is not None:
        groups = np.concatenate(groups, axis=0)
        res += (groups,)
    if return_counts:
        res += (counts,)
    if return_freqs:
        res += (freqs,)
    return res


def load_labels(fn, soft_labels=False):
    """
    Load the labels saved in the given filename.

    Raises
    ------
    * NotImplementedError if ``fn`` is an unsupported file type.

    Parameters
    ----------
    fn : str
        Where the data is saved. Supported file types: {'.npy', '.csv'}
    soft_labels : bool, optional
        If labels are given as probabilities, don't perform an argmax operation.

    Returns
    -------
    labels : numpy.ndarray
        LFP window labels.
        Shape: [n_windows] or [n_windows,n_classes]
    """
    assert isinstance(fn, str)
    if fn.endswith(".npy"):
        labels = np.load(fn)
    elif fn.endswith(".csv"):
        labels = np.loadtxt(fn, delimiter=",")
    else:
        raise NotImplementedError(f"Unsupported file type: {fn}")
    if labels.ndim == 2 and not soft_labels:
        labels = np.argmax(labels, axis=1)
    elif labels.ndim == 1:
        labels = np.array(labels, dtype=int)
    return labels


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
    if fn.endswith(".mat"):
        try:
            lfps = loadmat(fn)
        except NotImplementedError:
            # This must be one of the older HDF5 mat files.
            lfps = dict(h5py.File(fn, "r"))
    else:
        raise NotImplementedError(f"Cannot load file: {fn}")
    # Make sure all the channels are 1D float arrays.
    for channel in list(lfps.keys()):
        if channel in MATLAB_IGNORED_KEYS:
            del lfps[channel]
            continue
        try:
            lfps[channel] = np.array(lfps[channel]).astype(np.float).flatten()
        except (ValueError, TypeError):
            warnings.warn(f"Unable to read channel: {channel}")
            del lfps[channel]
    return lfps


def save_features(features, fn):
    """
    Save the features to the given filename.

    Raises
    ------
    * NotImplementedError if ``fn`` is an unsupported file type.

    Parameters
    ----------
    features : dict
        Maps fields to data
    fn : str
        Where to save the data. Supported file types: {'.npy'}
    """
    assert isinstance(fn, str)
    if fn.endswith(".npy"):
        np.save(fn, features)
    else:
        raise NotImplementedError(f"Unsupported file type: {fn}")


def save_labels(labels, fn, overwrite=True):
    """
    Save the labels to the given filename.

    Raises
    ------
    * NotImplementedError if ``fn`` is an unsupported file type.
    * AssertionError if ``fn`` exists and ``not overwrite``.

    Parameters
    ----------
    labels : numpy.ndarray
        Shape: [n_window] or [n_window, n_classes]
    fn : str
        Where to save the data. Supported file types: {'.npy', '.csv'}
    overwrite : bool, optional
        Whether to overwrite an existing file with the same name.
    """
    assert isinstance(fn, str), f"fn {fn} is not a string!"
    assert overwrite or not os.path.exists(fn), f"File {fn} exists!"
    if isinstance(labels, list):
        labels = np.array(labels)
    assert isinstance(labels, np.ndarray)
    if fn.endswith(".npy"):
        np.save(fn, labels)
    elif fn.endswith(".csv"):
        fmt = "%.4e" if labels.ndim == 2 else "%d"
        np.savetxt(fn, labels, fmt=fmt, delimiter=",")
    else:
        raise NotImplementedError(f"Unsupported file type: {fn}")


if __name__ == "__main__":
    pass


###
