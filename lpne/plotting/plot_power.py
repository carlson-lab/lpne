"""
Plot cross power spectral density features in a grid.

"""
__date__ = "July 2021"


import matplotlib.pyplot as plt
import numpy as np



def plot_power(power, rois, fn='temp.pdf'):
    """
    Plot power features in a square grid.

    Parameters
    ----------
    power : numpy.ndarray
        Power features. Shape: [n_roi*(n_roi+1)//2, n_freq]
    rois : list of str
        ROI names
    fn : str, optional
        Image filename
    """
    pretty_rois = [roi.replace('_', ' ') for roi in rois]
    ylim = (-0.05*np.max(power), 1.05*np.max(power))
    n = int(round((-1 + np.sqrt(1+8*power.shape[0]))/2))
    assert n == len(rois)
    fig, axarr = plt.subplots(n,n)
    for i in range(n):
        for j in range(i+1):
            idx = (i * (i+1)) // 2 + j
            axarr[i,j].fill_between(np.arange(len(power[0])), power[idx])
            for dir in axarr[i,j].spines:
                axarr[i,j].spines[dir].set_visible(False)
            plt.sca(axarr[i,j])
            plt.xticks([])
            plt.yticks([])
            plt.ylim(ylim)
            if j < i:
                axarr[j,i].fill_between(np.arange(len(power[0])), power[idx])
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
