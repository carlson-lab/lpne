"""
File system-related utilities

TODO: clean this up and deprecate some functions
"""
__date__ = "July 2021 - November 2022"
__all__ = [
    "get_all_fns",
    "get_feature_filenames",
    "get_feature_label_filenames",
    "get_label_filenames_from_feature_filenames",
    "get_lfp_chans_filenames",
    "get_lfp_filenames",
    "infer_groups_from_fns",
]

import numpy as np
import os
import warnings

LFP_FN_SUFFIX = "_LFP.mat"
CHANS_FN_SUFFIX = "_CHANS.mat"
FEATURE_FN_SUFFIX = ".npy"
LABEL_FN_SUFFIX = ".npy"


def get_all_fns(
    exp_dir,
    chans_subdir=None,
    feature_subdir=None,
    label_subdir=None,
    lfp_subdir=None,
    chans_suffix=None,
    label_suffix=None,
    lfp_suffix=None,
    strict_checking=True,
    **params,
):
    """
    Return the corresponding CHANS, feature, label, and LFP filenames.

    Raises
    ------
    * AssertionError if any of the subdirectories, excluding the feature directory,
        don't exist.
    * AssertionError if the files in each directory don't match and ``strict_checking``.
        If ``not strict_checking``, a UserWarning is thrown instead.

    Parameters
    ----------
    exp_dir : str
        Experiment directory
    chans_subdir : str
        CHANS subdirectory
    feature_subdir : str
        Feature subdirectory
    label_subdir : str
        Label subdirectory
    lfp_subdirectory : str
        LFP subdirectory
    chans_suffix : str
        Common suffix for CHANS files
    label_suffix : str
        Common suffix for label files
    lfp_suffix : str
        Common suffix for LFP files
    strict_checking : bool, optional
        Toggles whether to throw an error or a warning if there are mismatches between
        the CHANS, label, and LFP directories. The feature directory is not checked to
        facilitate the case where the features haven't been calculated yet.

    Returns
    -------
    chans_fns : list of str
        CHANS filenames
    feature_fns : list of str
        Feature filenames
    label_fns : list of str
        Label filenames
    lfp_fns : list of str
        LFP filenames
    """
    # Make sure all the directories exist. Make the feature directory if it doesn't.
    assert chans_subdir is not None, "chans_subdir must be specified!"
    assert feature_subdir is not None, "feature_subdir must be specified!"
    assert label_subdir is not None, "label_subdir must be specified!"
    assert lfp_subdir is not None, "lfp_subdir must be specified!"
    assert os.path.exists(exp_dir), f"Experiment directory '{exp_dir}' doesn't exist!"
    chans_dir = os.path.join(exp_dir, chans_subdir)
    assert os.path.exists(chans_dir), f"CHANS directory '{chans_dir}' doesn't exist!"
    feature_dir = os.path.join(exp_dir, feature_subdir)
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    label_dir = os.path.join(exp_dir, label_subdir)
    assert os.path.exists(label_dir), f"Label directory '{label_dir}' doesn't exist!"
    lfp_dir = os.path.join(exp_dir, lfp_subdir)
    assert os.path.exists(label_dir), f"LFP directory '{lfp_dir}' doesn't exist!"
    assert isinstance(chans_suffix, str) and len(chans_suffix) > 0
    assert isinstance(label_suffix, str) and len(label_suffix) > 0
    assert isinstance(lfp_suffix, str) and len(lfp_suffix) > 0

    # Figure out the missing/extra files.
    j = len(chans_suffix)
    chans_fns = [i[:-j] for i in os.listdir(chans_dir) if i.endswith(chans_suffix)]
    chans_fns = np.array(chans_fns)
    j = len(label_suffix)
    label_fns = [i[:-j] for i in os.listdir(label_dir) if i.endswith(label_suffix)]
    label_fns = np.array(label_fns)
    j = len(lfp_suffix)
    lfp_fns = [i[:-j] for i in os.listdir(lfp_dir) if i.endswith(lfp_suffix)]
    lfp_fns = np.array(lfp_fns)
    kw = dict(invert=True, assume_unique=True)
    chan_label_f = chans_fns[np.isin(chans_fns, label_fns, **kw)]
    label_chan_f = label_fns[np.isin(label_fns, chans_fns, **kw)]
    chan_lfp_f = chans_fns[np.isin(chans_fns, lfp_fns, **kw)]
    lfp_chan_f = lfp_fns[np.isin(lfp_fns, chans_fns, **kw)]
    label_lfp_f = label_fns[np.isin(label_fns, lfp_fns, **kw)]
    lfp_label_f = lfp_fns[np.isin(lfp_fns, label_fns, **kw)]
    temp = [
        chan_label_f,
        label_chan_f,
        chan_lfp_f,
        lfp_chan_f,
        label_lfp_f,
        lfp_label_f,
    ]
    lens = [len(i) for i in temp]
    msg = ""
    if lens[0] > 0:
        msg += f"Found {lens[0]} files in CHANS not in labels! "
    if lens[1] > 0:
        msg += f"Found {lens[1]} files in labels not in CHANS! "
    if lens[2] > 0:
        msg += f"Found {lens[2]} files in CHANS not in LFPs! "
    if lens[3] > 0:
        msg += f"Found {lens[3]} files in LFPs not in CHANS! "
    if lens[4] > 0:
        msg += f"Found {lens[4]} files in labels not in LFPs! "
    if lens[5] > 0:
        msg += f"Found {lens[5]} files in LFPs not in labels! "
    msg = msg[:-1] if msg != "" else msg  # cut off the last space character
    if strict_checking:
        assert sum(lens) == 0, msg
    elif len(msg) > 0:
        warnings.warn(msg)

    # Figure out the common group of files.
    fns = np.intersect1d(chans_fns, label_fns)
    fns = np.intersect1d(fns, lfp_fns)  # guaranteed to be sorted
    assert len(fns) > 0, (
        f"Found no filenames in common between CHANS, label, and " f"LFP directories!"
    )
    chans_fns = [os.path.join(chans_dir, i + chans_suffix) for i in fns]
    feature_fns = [os.path.join(feature_dir, i + FEATURE_FN_SUFFIX) for i in fns]
    label_fns = [os.path.join(label_dir, i + label_suffix) for i in fns]
    lfp_fns = [os.path.join(lfp_dir, i + lfp_suffix) for i in fns]
    return chans_fns, feature_fns, label_fns, lfp_fns


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
        os.path.join(feature_dir, fn)
        for fn in sorted(os.listdir(feature_dir))
        if fn.endswith(FEATURE_FN_SUFFIX)
    ]
    if len(fns) == 0:
        warnings.warn(f"No feature files in {feature_dir}!")
    return fns


def get_feature_label_filenames(feature_dir, label_dir):
    """
    Get the corresponding feature and label filenames.

    Parameters
    ----------
    feature_dir : str
        Feature directory
    label_dir : str
        Label directory

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
        os.path.join(feature_dir, fn)
        for fn in sorted(os.listdir(feature_dir))
        if fn.endswith(FEATURE_FN_SUFFIX)
    ]
    if len(feature_fns) == 0:
        warnings.warn(f"No feature files in {feature_dir}!")
    label_fns = [
        os.path.join(label_dir, fn)
        for fn in sorted(os.listdir(label_dir))
        if fn.endswith(LABEL_FN_SUFFIX)
    ]
    if len(label_fns) == 0:
        warnings.warn(f"No label files in {label_dir}!")
    assert len(feature_fns) == len(label_fns), f"{len(feature_fns)} != {len(label_fns)}"
    for i in range(len(feature_fns)):
        feature_fn = os.path.split(feature_fns[i])[-1]
        label_fn = os.path.split(label_fns[i])[-1]
        assert feature_fn == label_fn, f"{feature_fn} != {label_fn}"
    return feature_fns, label_fns


def get_label_filenames_from_feature_filenames(feature_fns, label_dir):
    """
    Given features filenames, return corresponding label filenames.

    Parameters
    ----------
    feature_fns : list of str
        Feature filenames
    label_dir : str
        Label directory

    Returns
    -------
    label_fns : list of str
        Label filenames
    """
    return [
        os.path.join(label_dir, os.path.split(feature_fn)[-1])
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
        os.path.join(lfp_dir, fn)
        for fn in sorted(os.listdir(lfp_dir))
        if fn.endswith(LFP_FN_SUFFIX)
    ]
    if len(lfp_fns) == 0:
        warnings.warn(f"No LFP files in {lfp_fns}!")
    chans_fns = [
        os.path.join(chans_dir, fn)
        for fn in sorted(os.listdir(chans_dir))
        if fn.endswith(CHANS_FN_SUFFIX)
    ]
    if len(chans_fns) == 0:
        warnings.warn(f"No CHANS files in {chans_dir}!")
    assert len(lfp_fns) == len(chans_fns), f"{len(lfp_fns)} != {len(chans_fns)}"
    for i in range(len(lfp_fns)):
        lfp_fn = os.path.split(lfp_fns[i])[-1][: -len(LFP_FN_SUFFIX)]
        chans_fn = os.path.split(chans_fns[i])[-1][: -len(CHANS_FN_SUFFIX)]
        assert lfp_fn == chans_fn, f"{lfp_fn} != {chans_fn}"
    return lfp_fns, chans_fns


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
        os.path.join(lfp_dir, fn)
        for fn in sorted(os.listdir(lfp_dir))
        if fn.endswith(LFP_FN_SUFFIX)
    ]
    if len(fns) == 0:
        warnings.warn(f"No LFP files in {lfp_dir}!")
    return fns


def infer_groups_from_fns(fns):
    """
    Infer groups from the filenames.

    The expected filename format is: */Mouse<id>_<date>_*.mat

    Unique combinations of <id> and <date> are assigned to different groups. Both <id>
    and <date> are assumed to contain no underscores.

    Raises
    ------
    * AssertionError if any of the filename are improperly formatted.

    Parameters
    ----------
    fns : list of str
        Filenames

    Returns
    -------
    groups : list of int
        Inferred group for each filename. The groups are zero-indexed and sorted by the
        tuple ``(id, date)``.
    group_map : dict
        Maps group name to integer group
    """
    # First check to make sure all the filenames are in the correct format.
    fns = [os.path.split(fn)[1] for fn in fns]
    group_tuples = []
    for fn in fns:
        assert fn.startswith("Mouse"), f"Filename {fn} doesn't start with 'Mouse'!"
        temp = fn.split("_")
        assert len(temp) >= 3, f"Expected filename {fn} to have at least 2 underscores!"
        group_tuples.append((temp[0][len("Mouse") :] + "_" + temp[1]))
    # Sort the groups and make map.
    unique_group_tuples = np.unique(group_tuples).tolist()  # guaranteed sorted
    groups = [unique_group_tuples.index(tup) for tup in group_tuples]
    group_map = dict(zip(group_tuples, groups))
    return groups, group_map


if __name__ == "__main__":
    pass


###
