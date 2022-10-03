"""
Plot LFPs in the time domain.

"""
__date__ = "August 2021 - June 2022"


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

import lpne


def plot_lfps(
    lfps,
    rois=None,
    t1=0.0,
    t2=None,
    fs=1000,
    y_space=4.0,
    alpha=0.85,
    lw=1.0,
    window_duration=None,
    show_windows=True,
    fn="temp.pdf",
):
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
    show_windows : bool, optional
        Whether to plot vertical lines separating the windows
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
        i2 = min(int(fs * t2), len(lfps[rois[0]]))
    t2 = i2 / fs
    t_vals = np.linspace(t1, t2, i2 - i1)
    # Plot.
    lfp_num = 0
    nan_bar = None
    for roi in rois:
        # Plot the single channel.
        trace = _zscore(lfps[roi][i1:i2])
        plt.plot(t_vals, lfp_num * y_space + trace, lw=lw, alpha=alpha)
        if nan_bar is None:
            nan_bar = np.zeros(len(trace), dtype=int)
        nan_trace = np.isnan(trace)
        nan_bar += nan_trace
        idx = np.argwhere(nan_trace > 0).flatten()
        # Plot the channel's outliers (NaNs) on top of that channel.
        if len(idx) > 0:
            y_vals = [lfp_num * y_space] * len(idx)
            plt.scatter(t_vals[idx], y_vals, c="k", s=4.0)
        lfp_num += 1
    # Plot the overall outliers.
    idx = np.argwhere(nan_bar > 0).flatten()
    if len(idx) > 0:
        if window_duration is None:
            # Plot as scatters if we don't have window information.
            plt.scatter(t_vals[idx], [-2 * y_space] * len(idx), c="k")
        else:
            # Otherwise mark out entire windows at a time.
            i = i1
            window_samp = int(fs * window_duration)
            temp_y = [-2 * y_space] * 2
            while i + window_samp <= i2:
                if nan_bar[i - i1 : i - i1 + window_samp].sum() > 0:
                    plt.plot(
                        [t_vals[i - i1], t_vals[i - i1 + window_samp]],
                        temp_y,
                        c="k",
                        solid_capstyle="butt",
                        lw=2,
                    )
                i += window_samp
    pretty_rois = [roi.replace("_", " ") for roi in rois]
    plt.yticks(
        y_space * np.arange(len(rois)),
        pretty_rois,
        size="xx-small",
        rotation=30,
    )
    ax = plt.gca()
    for dir in ["top", "left", "right"]:
        ax.spines[dir].set_visible(False)
    if show_windows and window_duration is not None:
        i1 = int(np.ceil(t1 / window_duration))
        i2 = 1 + int(np.floor(t2 / window_duration))
        window_ts = [window_duration * i for i in range(i1, i2)]
        for window_t in window_ts:
            plt.axvline(x=window_t, c="k", lw=1.0, ls="--")
    plt.xlabel("Time (s)")
    plt.savefig(fn)
    plt.close("all")


def _zscore(trace):
    """Z-score with NaN handling"""
    temp = np.copy(trace)
    nan_mask = np.argwhere(np.isnan(temp) > 0).flatten()
    temp[nan_mask] = 0.0
    temp = zscore(temp)
    temp[nan_mask] = np.nan
    return temp


if __name__ == "__main__":
    pass


###
