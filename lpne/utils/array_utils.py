"""
Array-related utilities

"""
__date__ = "July 2021 - November 2022"
__all__ = [
    "flatten_dir_spec_features",
    "flatten_power_features",
    "squeeze_triangular_array",
    "unsqueeze_triangular_array",
]

import numpy as np


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


if __name__ == "__main__":
    pass


###
