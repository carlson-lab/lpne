"""
Useful functions

"""
__date__ = "July 2021 - June 2022"


import numpy as np
import os
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import torch
import warnings

from .. import INVALID_LABEL
from .data import load_features, save_labels


LFP_FN_SUFFIX = '_LFP.mat'
CHANS_FN_SUFFIX = '_CHANS.mat'
FEATURE_FN_SUFFIX = '.npy'
LABEL_FN_SUFFIX = '.npy'



def write_fake_labels(feature_dir, label_dir, n_classes=2, label_format='.npy',
    seed=42):
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
        labels = np.random.randint(0,high=n_classes,size=n)
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
    idx = np.argwhere( ids == invalid_label).flatten()
    idx_comp = np.argwhere(ids != invalid_label).flatten()
    n = len(idx_comp)
    if groups is not None:
        ids = ids + (np.max(labels)+1) * np.array(groups)
    ids_subset = ids[idx_comp]
    unique_ids = np.unique(ids_subset)
    id_counts = [ \
            len(np.argwhere(ids_subset==t_id).flatten()) \
            for t_id in unique_ids \
    ]
    id_weights = n / (len(unique_ids) * np.array(id_counts))
    weights = np.ones(len(labels))
    for id, weight in zip(unique_ids, id_weights):
        weights[np.argwhere(ids==id).flatten()] = weight
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
    n = int(round((-1 + np.sqrt(1 + 8*arr.shape[dim])) / 2))
    assert (n * (n+1)) // 2 == arr.shape[dim], \
            f"{(n * (n+1)) // 2} != {arr.shape[dim]}"
    arr = np.swapaxes(arr, dim, -1)
    new_shape = arr.shape[:-1] + (n,n)
    new_arr = np.zeros(new_shape, dtype=arr.dtype)
    for i in range(n):
        for j in range(i+1):
            idx = (i * (i+1)) // 2 + j
            new_arr[..., i, j] = arr[..., idx]
            if i != j:
                new_arr[..., j, i] = arr[..., idx]
    dim_list = list(range(new_arr.ndim-2)) + [dim]
    dim_list = dim_list[:dim] + [-2,-1] + dim_list[dim+1:]
    new_arr = np.transpose(new_arr, dim_list)
    return new_arr


def squeeze_triangular_array(arr, dims=(0,1)):
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
    dim_list = dim_list[:dims[0]] + dim_list[dims[1]+1:] + list(dims)
    arr = np.transpose(arr, dim_list)
    new_arr = np.zeros(arr.shape[:-2] + ((n*(n+1))//2,))
    for i in range(n):
        for j in range(i+1):
            idx = (i * (i+1)) // 2 + j
            new_arr[..., idx] = arr[..., i, j]
    dim_list = list(range(new_arr.ndim))
    dim_list = dim_list[:dims[0]] + [-1] + dim_list[dims[0]:-1]
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
    msg = f"{window_count} of {n_windows} windows contain outliers " \
          f"({100*window_count/n_windows:.2f}%)\n" \
          f"Top offending channels:\n"
    sorted_counts = np.sort(-roi_counts)
    sorted_rois = np.array(rois)[np.argsort(-roi_counts)]
    for i in range(min(len(rois), top_n)):
        roi = sorted_rois[i]
        numerator = -sorted_counts[i]
        percent = 100 * numerator / n_windows
        msg += f"  {i+1}) {roi}: {numerator}/{n_windows} ({percent:.2f}%)\n"
    return msg


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



if __name__ == '__main__':
    pass



###
