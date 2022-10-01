"""
Plot a linear cross power spectral density feature on a grid.

"""
__date__ = "July 2021 - July 2022"


from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
import numpy as np


def plot_factor(factor, rois, color=None, alpha=1.0, fn="temp.pdf"):
    """
    Plot power feature factor in a square grid.

    Parameters
    ----------
    factor : numpy.ndarray
        Power features. Shape: ``[n_roi*(n_roi+1)//2, n_freq]``
    rois : list of str
        ROI names
    color : None or str, optional
        Plotting color
    alpha : float, optional
        Transparency
    fn : str, optional
        Image filename
    """
    assert factor.ndim == 2, f"len({factor.shape}) != 2"
    plot_factors(
        factor.reshape(1, factor.shape[0], factor.shape[1]),
        rois,
        colors=color if color is None else [color],
        alpha=alpha,
        fn=fn,
    )


def plot_factors(factors, rois, colors=None, alpha=0.6, fn="temp.pdf"):
    """
    Plot power feature factors in a square grid.

    Parameters
    ----------
    factors : numpy.ndarray
        Power features. Shape: ``[n_factors,n_roi*(n_roi+1)//2,n_freq]``
    rois : list of str
        ROI names
    fn : str, optional
        Image filename
    colors : None or list of str, optional
        The colors for each factor
    alpha : float, optional
        Transparency
    """
    assert factors.ndim == 3, f"len({factors.shape}) != 3"
    if colors is None:
        colors = list(TABLEAU_COLORS)
    pretty_rois = [roi.replace("_", " ") for roi in rois]
    temp = 1.05 * np.max(np.abs(factors))
    ylim = (-temp, temp)
    n = len(rois)
    factors = factors.reshape(len(factors), n * (n + 1) // 2, -1)
    freqs = np.arange(factors.shape[2])
    _, axarr = plt.subplots(n, n)
    for f in range(len(factors)):
        color = colors[f % len(colors)]
        for i in range(n):
            for j in range(i + 1):
                idx = (i * (i + 1)) // 2 + j
                axarr[i, j].fill_between(
                    freqs,
                    factors[f, idx],
                    fc=color,
                    alpha=alpha,
                )
                for dir in axarr[i, j].spines:
                    axarr[i, j].spines[dir].set_visible(False)
                plt.sca(axarr[i, j])
                plt.xticks([])
                plt.yticks([])
                plt.ylim(ylim)
                if j < i:
                    axarr[j, i].fill_between(
                        freqs,
                        factors[f, idx],
                        fc=color,
                        alpha=alpha,
                    )
                    for dir in axarr[i, j].spines:
                        axarr[j, i].spines[dir].set_visible(False)
                    plt.sca(axarr[j, i])
                    plt.xticks([])
                    plt.yticks([])
                    plt.ylim(ylim)
                if j == 0:
                    plt.sca(axarr[i, j])
                    plt.ylabel(pretty_rois[i], size="xx-small", rotation=30)
                if i == n - 1:
                    plt.sca(axarr[i, j])
                    plt.xlabel(pretty_rois[j], size="xx-small", rotation=30)
    plt.savefig(fn)
    plt.close("all")


if __name__ == "__main__":
    pass


###
