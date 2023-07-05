"""
Useful functions that don't fit cleanly into another file

"""
__date__ = "July 2021 - November 2022"
__all__ = [
    "confusion_matrix",
    "get_outlier_summary",
    "get_weights",
    "write_fake_labels",
]


import numpy as np
import os
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import torch


from .. import INVALID_LABEL
from .data import load_features, save_labels
from .file_utils import *


def confusion_matrix(true_labels, pred_labels):
    """
    Return a confusion matrix with true labels on the rows.

    This is a wrapper around ``sklearn.metrics.confusion_matrix`` that
    disregards ``lpne.INVALID_LABEL``.

    Parameters
    ----------
    true_labels : numpy.ndarray
        True labels
        Shape: ``[n]``
    pred_labels : numpy.ndarray
        Predicted labels
        Shape: ``[n]``

    Returns
    -------
    confusion_matrix : numpy.ndarray
        Shape: ``[c,c]``
    """
    idx1 = np.argwhere(true_labels != INVALID_LABEL).flatten()
    idx2 = np.argwhere(pred_labels != INVALID_LABEL).flatten()
    idx = np.intersect1d(idx1, idx2)
    return sk_confusion_matrix(true_labels[idx], pred_labels[idx])


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


if __name__ == "__main__":
    pass


###
