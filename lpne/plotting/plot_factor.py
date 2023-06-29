"""
Plot a linear cross power spectral density feature on a grid.

"""
__date__ = "July 2021 - June 2023"


from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
import numpy as np


def plot_factor(factor, rois, color=None, alpha=1.0, spines=[], fn="temp.pdf"):
    """
    Plot power feature factor in a square grid.

    Parameters
    ----------
    factor : numpy.ndarray
        Power features. Shape: ``[n_roi, n_roi, n_freq]``
    rois : list of str
        ROI names
    color : None or str, optional
        Plotting color
    alpha : float, optional
        Transparency
    spines : list of str, optional
        Which spines to plot. Should be a subset of
        ``['left', 'right', 'top', 'bottom']``.
    fn : str, optional
        Image filename
    """
    assert factor.ndim == 3, f"len({factor.shape}) != 3"
    plot_factors(
        factor[np.newaxis],
        rois,
        colors=color if color is None else [color],
        alpha=alpha,
        spines=spines,
        fn=fn,
    )


def plot_factors(factors, rois, colors=None, alpha=0.6, spines=[], fn="temp.pdf"):
    """
    Plot power feature factors in a square grid.

    Parameters
    ----------
    factors : numpy.ndarray
        Power features. Shape: ``[n_factors,n_roi,n_roi,n_freq]``
    rois : list of str
        ROI names
    fn : str, optional
        Image filename
    colors : None or list of str, optional
        The colors for each factor
    alpha : float, optional
        Transparency
    spines : list of str, optional
        Which spines to plot. Should be a subset of
        ``['left', 'right', 'top', bottom]``.
    fn : str, optional
        Image filename
    """
    assert factors.ndim == 4, f"len({factors.shape}) != 4"
    if colors is None:
        colors = list(TABLEAU_COLORS)
    pretty_rois = [roi.replace("_", " ") for roi in rois]
    temp = 1.05 * np.max(np.abs(factors))
    if np.min(factors) < 0.0:
        ylim = (-temp, temp)
    else:
        ylim = (0, temp)
    n = len(rois)
    freqs = np.arange(factors.shape[3])
    _, axarr = plt.subplots(n, n)
    for f in range(len(factors)):
        color = colors[f % len(colors)]
        for i in range(n):
            for j in range(n):
                axarr[i, j].fill_between(
                    freqs,
                    factors[f, i, j],
                    fc=color,
                    alpha=alpha,
                )
                for dir in axarr[i, j].spines:
                    if dir not in spines:
                        axarr[i, j].spines[dir].set_visible(False)
                plt.sca(axarr[i, j])
                plt.xticks([])
                plt.yticks([])
                plt.ylim(ylim)
                if j == 0:
                    plt.ylabel(pretty_rois[i], size="xx-small", rotation=30)
                if i == n - 1:
                    plt.xlabel(pretty_rois[j], size="xx-small", rotation=30)
    plt.savefig(fn)
    plt.close("all")


if __name__ == "__main__":
    pass


###
