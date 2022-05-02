"""
Plot LFPs in the time domain.

"""
__date__ = "August - September 2021"


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

import lpne



def plot_lfps(lfps, rois=None, t1=0.0, t2=None, fs=1000, y_space=4.0,
    alpha=0.85, lw=1.0, window_duration=None, fn='temp.pdf'):
    """
    Plot the LFPs in the specified time range.

    Parameters
    ----------
    lfps : dict
        Maps ROI names to LFPs.
    rois : None or list of str
        Which ROIs to plot. If `None`, all the ROIs are plotted.
    t1 : float
        Start time, in seconds.
    t2 : None or float, optional
        End time, in seconds. If `None`, this is taken to be the end
        of the LFP.
    fs : int
        Samplerate, in Hz.
    y_space : float, optional
        Vertical spacing between LFPs.
    alpha : float, optional
        Passed to `pyplot.plot`.
    lw : float, optional
        Passed to `pyplot.plot`.
    window_duration : None or float
        Window duration, in seconds. If `None`, no window markers are plotted.
    fn : str
        Image filename
    """
    # Figure out ROIs.
    if rois is None:
        rois = sorted([i for i in lfps.keys() if i not in lpne.IGNORED_KEYS])
    assert len(rois) > 0
    # Figure out times.
    i1 = int(fs * t1)
    if t2 is None:
        i2 = len(lfps[list(lfps.keys())[0]])
    else:
        i2 = int(fs * t2)
    t_vals = np.linspace(t1, t2, i2-i1)
    # Plot.
    lfp_num = 0
    nan_bar = None
    for roi in rois:
        trace = _zscore(lfps[roi].flatten()[i1:i2])
        plt.plot(t_vals, lfp_num*y_space+trace, lw=lw, alpha=alpha)
        if nan_bar is None:
            nan_bar = np.zeros(len(trace), dtype=int)
        nan_bar += np.isnan(trace)
        lfp_num += 1
    idx = np.argwhere(nan_bar > 0)
    if len(idx) > 0:
        plt.scatter(t_vals[idx], [-y_space]*len(idx), c='k')
    pretty_rois = [roi.replace('_', ' ') for roi in rois]
    plt.yticks(
            y_space*np.arange(len(rois)),
            pretty_rois,
            size='xx-small',
            rotation=30,
    )
    ax = plt.gca()
    for dir in ['top', 'left', 'right']:
        ax.spines[dir].set_visible(False)
    if window_duration is not None:
        i1 = int(np.ceil(t1/window_duration))
        i2 = 1 + int(np.floor(t2/window_duration))
        window_ts = [window_duration*i for i in range(i1,i2)]
        for window_t in window_ts:
            plt.axvline(x=window_t, c='k', lw=1.0, ls='--')
    plt.xlabel('Time (s)')
    plt.savefig(fn)
    plt.close('all')


def _zscore(trace):
    temp = np.copy(trace)
    nan_mask = np.isnan(temp)
    temp[np.isnan(temp)] = 0.0
    temp = zscore(temp)
    temp[nan_mask] = np.nan
    return temp


if __name__ == '__main__':
    pass


###
