"""
Plot 

"""
__date__ = "July 2022"


from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

EPS = 1e-7



def plot_db(features, freqs, labels, groups, rois=None, colors=None,
    quantile=0.05, fn='temp.pdf'):
    """
    Plot cross power in decibels for the labels, averaged over groups.
    
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
    rois : None or list of str
        ROI names
    colors : None or list of str
    fn : str, optional
        Image filename
    """
    assert len(features) == len(labels) and len(labels) == len(groups)
    assert features.shape[1] == len(freqs)
    assert features.shape[2] == features.shape[3]

    # Remove NaNs.
    idx = np.argwhere(np.isnan(features).sum(axis=(1,2,3)) == 0).flatten()
    features, labels, groups = features[idx], labels[idx], groups[idx]

    # Get unique groups and labels.
    unique_groups = np.unique(groups)
    unique_labels = np.unique(labels)

    # Figure out colors.
    if colors is None:
        colors = list(TABLEAU_COLORS)
    # Convert to decibels.
    db_features = _to_db(features, freqs)
    group_avgs = np.zeros(
            (len(unique_labels),len(unique_groups)) + features.shape[1:]
    )
    for i, group in enumerate(unique_groups):
        idx_1 = np.argwhere(groups == group).flatten()
        for j, label in enumerate(unique_labels):
            idx_2 = np.argwhere(labels == label).flatten()
            idx = np.intersect1d(idx_1, idx_2)
            if len(idx) == 0:
                continue
            group_avgs[j,i] = np.mean(db_features[idx], axis=0)
    ymin = np.quantile(group_avgs, quantile)
    n_roi = features.shape[2]
    if rois is not None:
        pretty_rois = [roi.replace('_', ' ') for roi in rois]
    _, axarr = plt.subplots(nrows=n_roi, ncols=n_roi)
    for i in range(n_roi):
        for j in range(n_roi):
            for k in range(len(unique_labels)):
                mean_val = np.mean(group_avgs[k], axis=0)[:,i,j]
                err = sem(group_avgs[k], axis=0)[:,i,j]
                axarr[i,j].plot(
                    freqs,
                    mean_val,
                    c=colors[k % len(colors)],
                )
                axarr[i,j].fill_between(
                    freqs,
                    mean_val-err,
                    mean_val+err,
                    fc=colors[k % len(colors)],
                    alpha=0.5,
                )
            for direction in ['top', 'right']:
                axarr[i,j].spines[direction].set_visible(False)
            plt.sca(axarr[i,j])
            plt.xticks([])
            plt.yticks([])
            plt.ylim(ymin, None)
            if (rois is not None) and j == 0:
                    plt.sca(axarr[i,j])
                    plt.ylabel(pretty_rois[i], size='xx-small', rotation=30)
            if (rois is not None) and i == n_roi-1:
                plt.sca(axarr[i,j])
                plt.xlabel(pretty_rois[j], size='xx-small', rotation=30)
    plt.savefig(fn)
    plt.close('all')



def _to_db(arr, freqs):
    db_arr = arr / (EPS + freqs.reshape(1,-1,1,1))
    db_arr = 10.0 * np.log10(db_arr + EPS)
    return db_arr



if __name__ == '__main__':
    pass



###