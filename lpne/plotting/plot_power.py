"""
Plot cross power spectral density features in a grid.

TO DO
-----
* Add ROI names to the plot.
* Make an animated version of the plot.
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
    ylim = (-0.05*np.max(power), 1.05*np.max(power))
    n = int(np.floor(np.sqrt(2*power.shape[0]+0.5)))
    fig, axarr = plt.subplots(n,n)
    for i in range(n):
        for j in range(i,n):
            idx = (i * (i+1)) // 2 + j
            axarr[i,j].plot(power[idx])
            for dir in axarr[i,j].spines:
                axarr[i,j].spines[dir].set_visible(False)
            plt.sca(axarr[i,j])
            plt.xticks([])
            plt.yticks([])
            plt.ylim(ylim)
            if j > i:
                axarr[j,i].plot(power[idx])
                for dir in axarr[i,j].spines:
                    axarr[j,i].spines[dir].set_visible(False)
                plt.sca(axarr[j,i])
                plt.xticks([])
                plt.yticks([])
                plt.ylim(ylim)
    plt.savefig(fn)
    plt.close('all')



if __name__ == '__main__':
    pass



###
