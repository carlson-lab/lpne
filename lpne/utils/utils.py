"""
Useful functions

"""
__date__ = "July 2021"


import numpy as np
import os
import warnings

from .data import load_features, save_labels


LFP_FN_SUFFIX = '_LFP.mat'
CHANS_FN_SUFFIX = '_CHANS.mat'
FEATURE_FN_SUFFIX = '.npy'
LABEL_FN_SUFFIX = '.npy'



def write_fake_labels(feature_dir, label_dir, n_label_types=2,
    label_format='.npy', seed=42):
    """
    Write fake behavioral labels.

    Parameters
    ----------
    feature_dir : str
    label_dir : str
    n_label_types : int, optional
    label_format : str, optional
    seed : int, optional
    """
    # Get filenames.
    feature_fns = get_feature_filenames(feature_dir)
    label_fns = get_label_filenames_from_feature_filenames(
            feature_fns,
            label_dir,
    )
    # Seed.
    if seed is not None:
        np.random.seed(seed)
    # For each file pair...
    for feature_fn, label_fn in zip(feature_fns, label_fns):
        # Make the appropiate number of labels and save them.
        features = load_features(feature_fn)
        n = len(features['power'])
        labels = np.random.randint(0,high=n_label_types,size=n)
        save_labels(labels, label_fn)
    # Undo the seed.
    if seed is not None:
        np.random.seed(None)


def get_lfp_filenames(lfp_dir):
    """
    Get the LFP filenames in the given directory.

    Raises
    ------
    * AssertionError if the feature directory doesn't exist.
    * UserWarning if there are no feature files.

    Paramters
    ---------
    lfp_dir : str
        LFP directory

    Returns
    -------
    lfp_fns : list of str
        Sorted list of LFP filenames
    """
    assert os.path.exists(lfp_dir), f"{lfp_dir} doesn't exist!"
    fns = [
            os.path.join(lfp_dir,fn) \
            for fn in sorted(os.listdir(lfp_dir)) \
            if fn.endswith(LFP_FN_SUFFIX)
    ]
    if len(fns) == 0:
        warnings.warn(f"No LFP files in {lfp_dir}!")
    return fns


def get_feature_filenames(feature_dir):
    """
    Get the feature filenames in the given directory.

    Raises
    ------
    * AssertionError if the feature directory doesn't exist.
    * UserWarning if there are no feature files.

    Parameters
    ----------
    feature_dir : str
        Feature directory.
    """
    assert os.path.exists(feature_dir), f"{feature_dir} doesn't exist!"
    fns = [
            os.path.join(feature_dir,fn) \
            for fn in sorted(os.listdir(feature_dir)) \
            if fn.endswith(FEATURE_FN_SUFFIX)
    ]
    if len(fns) == 0:
        warnings.warn(f"No feature files in {feature_dir}!")
    return fns


def get_label_filenames_from_feature_filenames(feature_fns, label_dir):
    """
    ...

    Raises
    ------
    *

    Parameters
    ----------
    feature_fns : list of str
        ...
    label_dir : str
        ...

    Returns
    -------
    label_fns : list of str
        Label filenames
    """
    return [
            os.path.join(label_dir, os.path.split(feature_fn)[-1]) \
            for feature_fn in feature_fns
    ]


def get_lfp_chans_filenames(lfp_dir, chans_dir):
    """
    Get the corresponding LFP and CHANS filenames.

    Parameters
    ----------
    lfp_dir : str
    chans_dir : str

    Returns
    -------
    lfp_filenames : list of str
        LFP filenames
    chans_filenames : list of str
        The corresponding CHANS filenames
    """
    assert os.path.exists(lfp_dir), f"{lfp_dir} doesn't exist!"
    assert os.path.exists(chans_dir), f"{chans_dir} doesn't exist!"
    lfp_fns = [
            os.path.join(lfp_dir,fn) \
            for fn in sorted(os.listdir(lfp_dir)) \
            if fn.endswith(LFP_FN_SUFFIX)
    ]
    if len(lfp_fns) == 0:
        warnings.warn(f"No LFP files in {lfp_fns}!")
    chans_fns = [
            os.path.join(chans_dir,fn) \
            for fn in sorted(os.listdir(chans_dir)) \
            if fn.endswith(CHANS_FN_SUFFIX)
    ]
    if len(chans_fns) == 0:
        warnings.warn(f"No CHANS files in {chans_dir}!")
    assert len(lfp_fns) == len(chans_fns), f"{len(lfp_fns)} != {len(chans_fns)}"
    for i in range(len(lfp_fns)):
        lfp_fn = os.path.split(lfp_fns[i])[-1][:-len(LFP_FN_SUFFIX)]
        chans_fn = os.path.split(chans_fns[i])[-1][:-len(CHANS_FN_SUFFIX)]
        assert lfp_fn == chans_fn, f"{lfp_fn} != {chans_fn}"
    return lfp_fns, chans_fns


def get_feature_label_filenames(feature_dir, label_dir):
    """
    Get the corresponding feature and label filenames.

    Parameters
    ----------
    feature_dir : str
    label_dir : str

    Returns
    -------
    feature_filenames : list of str
        Feature filenames
    label_filenames : list of str
        The corresponding label filenames
    """
    assert os.path.exists(feature_dir), f"{feature_dir} doesn't exist!"
    assert os.path.exists(label_dir), f"{label_dir} doesn't exist!"
    feature_fns = [
            os.path.join(feature_dir,fn) \
            for fn in sorted(os.listdir(feature_dir)) \
            if fn.endswith(FEATURE_FN_SUFFIX)
    ]
    if len(feature_fns) == 0:
        warnings.warn(f"No feature files in {feature_dir}!")
    label_fns = [
            os.path.join(label_dir,fn) \
            for fn in sorted(os.listdir(label_dir)) \
            if fn.endswith(LABEL_FN_SUFFIX)
    ]
    if len(label_fns) == 0:
        warnings.warn(f"No label files in {label_dir}!")
    assert len(feature_fns) == len(label_fns), \
            f"{len(feature_fns)} != {len(label_fns)}"
    for i in range(len(feature_fns)):
        feature_fn = os.path.split(feature_fns[i])[-1]
        label_fn = os.path.split(label_fns[i])[-1]
        assert feature_fn == label_fn, f"{feature_fn} != {label_fn}"
    return feature_fns, label_fns



if __name__ == '__main__':
    pass



###
