"""
Plot LFPs in the time domain.

"""
__date__ = "August 2021 - April 2023"

from matplotlib.colors import TABLEAU_COLORS
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
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
    colors=None,
    highlight_bands=[],
    highlight_colors=None,
    highlight_threshold=2.0,
    highlight_alpha=0.4,
    window_duration=None,
    show_windows=True,
    fn="temp.pdf",
):
    """
    Plot the LFPs in the specified time range.

    Optionally highlight times and channels where there is high power in particular
    frequency bands.

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
    colors : None or list of str, optional
        Colors to plot
    highlight_bands : list of tuples, optional
        Determines which frequency bands to highlight. Each tuple should contain two
        floats, the lowcut and highcut bandpass filter parameters.
    highlight_colors : None or list of str, optional
        Colors of the frequency band highlights
    highlight_threshold : float, optional
        Highlight when the norm of the analytic bandpassed signal is above this many
        standard deviations from the mean.
    highlight_alpha : float, optional
        Transparency of the highlights
    window_duration : None or float
        Window duration, in seconds. If `None`, no window markers are plotted.
    show_windows : bool, optional
        Whether to plot vertical lines separating the windows
    fn : str
        Image filename
    """
    # Figure out ROIs.
    if rois is None:
        rois = sorted([i for i in lfps.keys() if i not in lpne.MATLAB_IGNORED_KEYS])
    assert len(rois) > 0
    if colors is None:
        colors = list(TABLEAU_COLORS)
    if highlight_colors is None:
        highlight_colors = list(TABLEAU_COLORS)
    # Figure out times.
    i1 = int(fs * t1)
    if t2 is None:
        i2 = len(lfps[list(lfps.keys())[0]])
    else:
        i2 = min(int(fs * t2), len(lfps[rois[0]]))
    t2 = i2 / fs
    t_vals = np.linspace(t1, t2, i2 - i1)
    # Plot.
    ax = plt.gca()
    nan_bar = None
    for i, roi in enumerate(rois):
        # Plot the single channel.
        trace = _zscore(lfps[roi][i1:i2])
        c = colors[i % len(colors)]
        plt.plot(t_vals, i * y_space + trace, lw=lw, alpha=alpha, c=c)

        # Plot the channel's outliers (NaNs) on top of that channel.
        if nan_bar is None:
            nan_bar = np.zeros(len(trace), dtype=int)
        nan_trace = np.isnan(trace)
        nan_bar += nan_trace
        idx = np.argwhere(nan_trace > 0).flatten()
        if len(idx) > 0:
            y_vals = [i * y_space] * len(idx)
            plt.scatter(t_vals[idx], y_vals, c="k", s=4.0)

        # Plot the frequency band highlights.
        for j, (f1, f2) in enumerate(highlight_bands):
            temp_lfp = lfps[roi][:]
            temp_lfp[np.isnan(temp_lfp)] = 0.0
            lfp_filt = lpne.filter_signal(lfps[roi][:], fs, lowcut=f1, highcut=f2)
            lfp_filt = zscore(np.abs(hilbert(lfp_filt)))
            lfp_filt = lfp_filt[i1:i2]
            idx = np.argwhere(lfp_filt >= highlight_threshold).flatten()
            onsets, offsets = _indices_to_onsets_offsets(idx)
            y1, y2 = (i - 0.5) * y_space, (i + 0.5) * y_space
            for onset, offset in zip(onsets, offsets):
                x1, x2 = t1 + onset / fs, t1 + offset / fs
                patch = Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    alpha=highlight_alpha,
                    fc=highlight_colors[j % len(highlight_colors)],
                )
                ax.add_patch(patch)

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


def _indices_to_onsets_offsets(idx):
    if len(idx) == 0:
        return [], []
    arr = np.zeros(idx[-1] + 3, dtype=int)
    arr[idx + 1] = 1
    diff_arr = np.diff(arr)
    onsets = np.argwhere(diff_arr > 0).flatten()
    offsets = np.argwhere(diff_arr < 0).flatten()
    return onsets, offsets


if __name__ == "__main__":
    pass


###
