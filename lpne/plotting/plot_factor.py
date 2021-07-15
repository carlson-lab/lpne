"""
Plot a linear cross power spectral density feature on a grid.

TO DO
-----
* Make an animated version of the plot.
"""
__date__ = "July 2021"


import matplotlib.pyplot as plt
import numpy as np



def plot_factor(factor, rois, fn='temp.pdf'):
    """
    Plot power feature factor in a square grid.

    Parameters
    ----------
    factor : numpy.ndarray
        Power features. Shape: [n_roi*(n_roi+1)//2, n_freq]
    rois : list of str
        ROI names
    fn : str, optional
        Image filename
    """
    pretty_rois = [roi.replace('_', ' ') for roi in rois]
    # factor -= np.mean(factor)
    temp = 1.05 * np.max(np.abs(factor))
    ylim = (-temp, temp)
    n = len(rois)
    factor = factor.reshape(n*(n+1)//2,-1)
    fig, axarr = plt.subplots(n,n)
    for i in range(n):
        for j in range(i+1):
            idx = (i * (i+1)) // 2 + j
            axarr[i,j].fill_between(np.arange(len(factor[0])), factor[idx])
            for dir in axarr[i,j].spines:
                axarr[i,j].spines[dir].set_visible(False)
            plt.sca(axarr[i,j])
            plt.xticks([])
            plt.yticks([])
            plt.ylim(ylim)
            if j < i:
                axarr[j,i].fill_between(np.arange(len(factor[0])), factor[idx])
                for dir in axarr[i,j].spines:
                    axarr[j,i].spines[dir].set_visible(False)
                plt.sca(axarr[j,i])
                plt.xticks([])
                plt.yticks([])
                plt.ylim(ylim)
            if j == 0:
                plt.sca(axarr[i,j])
                plt.ylabel(pretty_rois[i], size='xx-small', rotation=30)
            if i == n-1:
                plt.sca(axarr[i,j])
                plt.xlabel(pretty_rois[j], size='xx-small', rotation=30)
    plt.savefig(fn)
    plt.close('all')



if __name__ == '__main__':
    pass



###
