"""
Useful functions

"""
__date__ = "July 2021 - November 2022"


import numpy as np
import os
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import torch
import warnings

from .. import INVALID_LABEL
from .data import load_features, save_labels


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

    TODO: implement single CHANS file for multiple mouse files case

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


def write_fake_labels(
    feature_dir, label_dir, n_classes=2, label_format=".npy", seed=42
):
    """
    Write fake behavioral labels.

    Parameters
    ----------
    feature_dir : str
    label_dir : str
    n_classes : int, optional
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
        features = load_features(feature_fn)[0]
        n = len(features)
        labels = np.random.randint(0, high=n_classes, size=n)
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
        os.path.join(lfp_dir, fn)
        for fn in sorted(os.listdir(lfp_dir))
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
        os.path.join(feature_dir, fn)
        for fn in sorted(os.listdir(feature_dir))
        if fn.endswith(FEATURE_FN_SUFFIX)
    ]
    if len(fns) == 0:
        warnings.warn(f"No feature files in {feature_dir}!")
    return fns


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


def get_weights(labels, groups, invalid_label=INVALID_LABEL):
    """
    Get weights inversely proportional to the label and group frequency.

    The average weight is fixed at one. If `labels` or `groups` are PyTorch
    tensors, they are converted to NumPy arrays.

    Parameters
    ----------
    labels : numpy.ndarray or torch.Tensor
        Label array
        Shape: [n]
    groups : None or numpy.ndarray or torch.Tensor
        Group array
        Shape: [n]
    invalid_label : int, optional
        A label that should be exempted from this procedure. This label is
        given a weight of 1.

    Returns
    -------
    weights : numpy.ndarray
        Weights
        Shape: [n]
    """
    if groups is not None:
        assert len(labels) == len(groups), f"{len(labels)} != {len(groups)}"
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(groups, torch.Tensor):
        groups = groups.detach().cpu().numpy()
    ids = np.array(labels)
    idx = np.argwhere(ids == invalid_label).flatten()
    idx_comp = np.argwhere(ids != invalid_label).flatten()
    n = len(idx_comp)
    if groups is not None:
        ids = ids + (np.max(labels) + 1) * np.array(groups)
    ids_subset = ids[idx_comp]
    unique_ids = np.unique(ids_subset)
    id_counts = [len(np.argwhere(ids_subset == t_id).flatten()) for t_id in unique_ids]
    id_weights = n / (len(unique_ids) * np.array(id_counts))
    weights = np.ones(len(labels))
    for id, weight in zip(unique_ids, id_weights):
        weights[np.argwhere(ids == id).flatten()] = weight
    weights[idx] = 1.0
    return weights


def unsqueeze_triangular_array(arr, dim=0):
    """
    Transform a numpy array from condensed triangular form to symmetric form.

    Parameters
    ----------
    arr : numpy.ndarray
    dim : int
        Axis to expand

    Returns
    -------
    new_arr : numpy.ndarray
        Expanded array
    """
    n = int(round((-1 + np.sqrt(1 + 8 * arr.shape[dim])) / 2))
    assert (n * (n + 1)) // 2 == arr.shape[
        dim
    ], f"{(n * (n+1)) // 2} != {arr.shape[dim]}"
    arr = np.swapaxes(arr, dim, -1)
    new_shape = arr.shape[:-1] + (n, n)
    new_arr = np.zeros(new_shape, dtype=arr.dtype)
    for i in range(n):
        for j in range(i + 1):
            idx = (i * (i + 1)) // 2 + j
            new_arr[..., i, j] = arr[..., idx]
            if i != j:
                new_arr[..., j, i] = arr[..., idx]
    dim_list = list(range(new_arr.ndim - 2)) + [dim]
    dim_list = dim_list[:dim] + [-2, -1] + dim_list[dim + 1 :]
    new_arr = np.transpose(new_arr, dim_list)
    return new_arr


def squeeze_triangular_array(arr, dims=(0, 1)):
    """
    Inverse of `unsqueeze_triangular_array`.

    Parameters
    ----------
    arr : numpy.ndarray
    dims : tuple of int
        The two dimensions to contract to one. These should be contiguous.

    Returns
    -------
    new_arr : numpy.ndarray
        Contracted array
    """
    assert len(dims) == 2
    assert arr.ndim > np.max(dims)
    assert arr.shape[dims[0]] == arr.shape[dims[1]]
    assert dims[1] == dims[0] + 1
    n = arr.shape[dims[0]]
    dim_list = list(range(arr.ndim))
    dim_list = dim_list[: dims[0]] + dim_list[dims[1] + 1 :] + list(dims)
    arr = np.transpose(arr, dim_list)
    new_arr = np.zeros(arr.shape[:-2] + ((n * (n + 1)) // 2,))
    for i in range(n):
        for j in range(i + 1):
            idx = (i * (i + 1)) // 2 + j
            new_arr[..., idx] = arr[..., i, j]
    dim_list = list(range(new_arr.ndim))
    dim_list = dim_list[: dims[0]] + [-1] + dim_list[dims[0] : -1]
    new_arr = np.transpose(new_arr, dim_list)
    return new_arr


def get_outlier_summary(lfps, fs, window_duration, top_n=6):
    """
    Return a message summarizing the outliers found

    Parameters
    ----------
    lfps :
    fs : int
        Samplerate
    window_duration : float
        LFP window duration, in seconds
    top_n : int, optional
        Show stats for this many channels

    Returns
    -------
    message : str
        A description of the outliers found.
    """
    rois = sorted(list(lfps.keys()))
    roi_counts = np.zeros(len(rois), dtype=int)
    window_samples = int(fs * window_duration)
    n_windows = len(lfps[rois[0]]) // window_samples
    window_count = 0
    # Collect the number of windows each window is implicated in.
    for i in range(n_windows):
        flag = False
        i1 = int(fs * i * window_duration)
        i2 = i1 + window_samples
        for j, roi in enumerate(rois):
            if np.isnan(lfps[roi][i1:i2]).sum() > 0:
                roi_counts[j] += 1
                flag = True
        if flag:
            window_count += 1
    # Make the message.
    msg = (
        f"{window_count} of {n_windows} windows contain outliers "
        f"({100*window_count/n_windows:.2f}%)\n"
        f"Top offending channels:\n"
    )
    sorted_counts = np.sort(-roi_counts)
    sorted_rois = np.array(rois)[np.argsort(-roi_counts)]
    for i in range(min(len(rois), top_n)):
        roi = sorted_rois[i]
        numerator = -sorted_counts[i]
        percent = 100 * numerator / n_windows
        msg += f"  {i+1}) {roi}: {numerator}/{n_windows} ({percent:.2f}%)\n"
    return msg


def flatten_power_features(features, rois, f):
    """
    Returns a flattened cross power features congruous with the legacy format

    Parameters
    ----------
    features : numpy.ndarray
        LFP Directed Spectrum Features
        Shape: ``[n_windows,n_freqs,n_roi,n_roi]``

    rois : list of str
        Sorted list of ROI labels.

    f : np.ndarray
        Array of evaluated frequencies

    Returns
    -------
    flat_features : numpy.ndarray
        Flattened LFP Directed Spectrum Features sorted by feature type
        Feature Interpretation: ``[n_windows, power features + causality features]``
        Shape: ``[n_windows,n_roi*n_freqs + n_roi*(n_roi-1)*n_freqs]``

    feature_ids : list of str
        List mapping flat_features index to a feature label ``<feature_id> <freq>``
        Shape: ``[n_roi*n_freqs + n_roi*(n_roi-1)*n_freqs]

    """

    assert features.shape[-1] == len(rois), f"Shape mismatch between features and rois"
    assert features.shape[1] == len(f), f"Shape mismatch between features and f"

    n_rois = features.shape[-1]
    n_freqs = features.shape[1]
    n_flat_features = n_rois**2 * n_freqs

    flat_features = np.zeros((features.shape[0], n_flat_features))
    feature_ids = []

    # Assign the power features
    for roi_idx in range(n_rois):
        flat_features[:, roi_idx * n_freqs : (roi_idx + 1) * n_freqs] = features[
            :, :, roi_idx, roi_idx
        ]

        for freq in f:
            feature_id = f"{rois[roi_idx]} {freq}"
            feature_ids.append(feature_id)

    # Assign the directed spectrum causality features
    flat_idx = n_rois
    for roi_idx_1 in range(n_rois):
        for roi_idx_2 in range(n_rois):
            if roi_idx_1 == roi_idx_2:
                continue
            else:
                start_idx = flat_idx * n_freqs
                stop_idx = (flat_idx + 1) * n_freqs
                flat_features[:, start_idx:stop_idx] = features[
                    :, :, roi_idx_1, roi_idx_2
                ]

                for freq in f:
                    feature_id = f"{rois[roi_idx_1]}<->{rois[roi_idx_2]} {freq}"
                    feature_ids.append(feature_id)

    return flat_features, feature_ids


def flatten_dir_spec_features(features, rois, f):
    """
    Returns a flattened directed spectrum congruous with the legacy format.

    Parameters
    ----------
    features : numpy.ndarray
        LFP Directed Spectrum Features
        Shape: ``[n_windows,n_roi,n_roi,n_freqs]``

    rois : list of str
        Sorted list of ROI labels.

    f : np.ndarray
        Array of evaluated frequencies

    Returns
    -------
    flat_features : numpy.ndarray
        Flattened LFP Directed Spectrum Features sorted by feature type
        Feature Interpretation: ``[n_windows, power features + causality features]``
        Shape: ``[n_windows,n_roi*n_freqs + n_roi*(n_roi-1)*n_freqs]``

    feature_ids : list of str
        List mapping flat_features index to a feature label ``<feature_id> <freq>``
        Shape: ``[n_roi*n_freqs + n_roi*(n_roi-1)*n_freqs]
    """

    assert features.shape[1] == len(rois), f"Shape mismatch between features and rois"
    assert features.shape[-1] == len(f), f"Shape mismatch between features and f"

    n_rois = features.shape[1]
    n_freqs = features.shape[-1]
    n_flat_features = n_rois**2 * n_freqs

    flat_features = np.zeros((features.shape[0], n_flat_features))
    feature_ids = []

    # Assign the power features
    for roi_idx in range(n_rois):
        flat_features[:, roi_idx * n_freqs : (roi_idx + 1) * n_freqs] = features[
            :, roi_idx, roi_idx, :
        ]

        for freq in f:
            feature_id = f"{rois[roi_idx]} {freq}"
            feature_ids.append(feature_id)

    # Assign the directed spectrum causality features
    flat_idx = n_rois
    for roi_idx_1 in range(n_rois):
        for roi_idx_2 in range(n_rois):
            if roi_idx_1 == roi_idx_2:
                continue
            else:
                start_idx = flat_idx * n_freqs
                stop_idx = (flat_idx + 1) * n_freqs
                flat_features[:, start_idx:stop_idx] = features[
                    :, roi_idx_1, roi_idx_2, :
                ]

                for freq in f:
                    feature_id = f"{rois[roi_idx_1]}->{rois[roi_idx_2]} {freq}"
                    feature_ids.append(feature_id)

    return flat_features, feature_ids


def confusion_matrix(true_labels, pred_labels):
    """
    Return a confusion matrix with true labels on the rows.

    This is a wrapper around ``sklearn.metrics.confusion_matrix`` that
    disregards ``lpne.INVALID_LABEL``.

    Parameters
    ----------
    true_labels : numpy.ndarray
        True labels
        Shape: [n]
    pred_labels : numpy.ndarray
        Predicted labels
        Shape: [n]

    Returns
    -------
    confusion_matrix : numpy.ndarray
        Shape: [c,c]
    """
    idx1 = np.argwhere(true_labels != INVALID_LABEL).flatten()
    idx2 = np.argwhere(pred_labels != INVALID_LABEL).flatten()
    idx = np.intersect1d(idx1, idx2)
    return sk_confusion_matrix(true_labels[idx], pred_labels[idx])


if __name__ == "__main__":
    pass


###
