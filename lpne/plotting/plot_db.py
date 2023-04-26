"""
Make decibel plots

"""
__date__ = "July 2022 - April 2023"


from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

from .. import INVALID_LABEL

EPS = 1e-7


def plot_db(
    features,
    freqs,
    labels,
    groups,
    relative_to=None,
    rois=None,
    colors=None,
    mode="abs",
    n_sem=2,
    min_quantile=0.05,
    sem_alpha=0.5,
    lw=1.0,
    x_ticks=None,
    y_ticks=None,
    figsize=(8, 8),
    fn="temp.pdf",
):
    """
    Plot cross power in decibels for the labels, averaged over groups.

    You can plot absolute decibels by passing ``mode='abs'``, and differences by passing
    ``mode='diff'``. If ``relative_to is None``, the differences are from the average
    across labels, and if ``relative_to`` is a label, the differences are from that
    label.

    Labels equal to ``lpne.INVALID_LABEL`` (``-1`` by default) are ignored.

    Parameters
    ----------
    features : numpy.ndarray
        Shape: [b,f,r,r]
    freqs : numpy.ndarray
        Shape: [f]
    labels : numpy.ndarray
        Shape: [b]
    groups : numpy.ndarray
        Shape: [b]
    relative_to : None or int, optional
        If ``mode == 'diff'``, plot decibels relative to this label. If ``None``, plot
        decibels relative to the mean across labels.
    rois : None or list of str
        ROI names. Defaults to ``None``.
    colors : None or list of str
        Defaults to ``None``. Colors correspond to unique labels.
    mode : {``'abs'``, ``'diff'``}, optional
        Whether to plot absolute dB values or differences from the mean.
        Defaults to ``'abs'``.
    n_sem : int or float, optional
        Number of standard errors of the mean to plot. Defaults to ``2``.
    min_quantile : float, optional
        Used to define y range. Defaults to ``0.05``.
    sem_alpha : float, optional
        Tranparency of uncertainty ranges. Defaults to ``0.5``.
    lw : float, optional
        Line width. Defaults to ``1``.
    x_ticks : None or list of float
        Frequency tick values
    y_ticks : None or list of float
        Decibel tick values
    figsize : tuple of float, optional
        Passed to ``plt.subplots``. Defaults to ``(8,8)``.
    fn : str, optional
        Image filename. Defaults to ``temp.pdf``.
    """
    assert len(features) == len(labels) and len(labels) == len(groups)
    assert features.shape[1] == len(freqs)
    assert features.shape[2] == features.shape[3]
    assert mode in ["diff", "abs"]

    # Figure out colors.
    if colors is None:
        colors = list(TABLEAU_COLORS)

    # Remove NaNs and invalid labels.
    idx1 = np.argwhere(np.isnan(features).sum(axis=(1, 2, 3)) == 0).flatten()
    idx2 = np.argwhere(labels != INVALID_LABEL).flatten()
    idx = np.intersect1d(idx1, idx2)
    features, labels, groups = features[idx], labels[idx], groups[idx]

    # Convert to decibels.
    db_features = _to_db(features, freqs)  # [w,f,r,r]

    # Get unique labels.
    unique_labels = np.unique(labels)
    if relative_to is not None:
        assert relative_to in unique_labels, f"{relative_to} not in {unique_labels}!"

    # Get the label averages and SEMs.
    label_avgs = np.zeros((len(unique_labels),) + features.shape[1:])
    label_sems = np.zeros((len(unique_labels),) + features.shape[1:])
    for i, label in enumerate(unique_labels):
        idx1 = np.argwhere(labels == label).flatten()
        label_groups = np.unique(groups[idx1])
        label_group_avgs = np.zeros((len(label_groups),) + features.shape[1:])
        for j, group in enumerate(label_groups):
            idx2 = np.argwhere(groups[idx1] == group).flatten()
            label_group_avgs[j] = np.mean(db_features[idx1][idx2], axis=0)
        label_avgs[i] = np.mean(label_group_avgs, axis=0)
        label_sems[i] = sem(label_group_avgs, axis=0)

    # Normalize in different ways.
    if mode == "diff":
        ymin = None
        if relative_to is None:
            label_avgs -= np.mean(label_avgs, axis=0, keepdims=True)
        else:
            temp = label_avgs[np.searchsorted(unique_labels, relative_to)]
            label_avgs -= temp.reshape((1,) + temp.shape)
    else:
        temp = np.min(label_avgs - n_sem * label_sems, axis=0)
        ymin = np.quantile(temp, min_quantile)

    # Plot.
    n_roi = features.shape[2]
    if rois is not None:
        pretty_rois = [roi.replace("_", " ") for roi in rois]
    _, axarr = plt.subplots(
        nrows=n_roi, ncols=n_roi, figsize=figsize, sharex=True, sharey=True
    )
    for i in range(n_roi):
        for j in range(n_roi):
            plt.sca(axarr[i, j])
            if mode == "diff" and relative_to is not None:
                plt.axhline(y=0, c="k", ls="--", lw=0.75)
            for k in range(len(unique_labels)):
                mean_val = label_avgs[k, :, i, j]
                err = n_sem * label_sems[k, :, i, j]
                axarr[i, j].plot(
                    freqs,
                    mean_val,
                    c=colors[k % len(colors)],
                    lw=lw,
                )
                axarr[i, j].fill_between(
                    freqs,
                    mean_val - err,
                    mean_val + err,
                    fc=colors[k % len(colors)],
                    alpha=sem_alpha,
                )
            for direction in ["top", "right"]:
                axarr[i, j].spines[direction].set_visible(False)
            if x_ticks is None:
                plt.xticks([])
            else:
                plt.xticks(x_ticks, size="xx-small")
                if i != n_roi - 1:
                    plt.setp(axarr[i, j].get_xticklabels(), visible=False)
            if y_ticks is None:
                plt.yticks([])
            else:
                plt.xticks(y_ticks, size="xx-small")
                if j != 0:
                    plt.setp(axarr[i, j].get_yticklabels(), visible=False)
            plt.ylim(ymin, None)
            if (rois is not None) and j == 0:
                plt.sca(axarr[i, j])
                plt.ylabel(pretty_rois[i], size="xx-small", rotation=30)
            if (rois is not None) and i == n_roi - 1:
                plt.sca(axarr[i, j])
                plt.xlabel(pretty_rois[j], size="xx-small", rotation=30)
    plt.savefig(fn)
    plt.close("all")


def _to_db(arr, freqs):
    db_arr = arr / (EPS + freqs.reshape(1, -1, 1, 1))
    db_arr = 10.0 * np.log10(db_arr + EPS)
    return db_arr


if __name__ == "__main__":
    pass


###
