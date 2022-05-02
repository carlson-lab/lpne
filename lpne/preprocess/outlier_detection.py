"""
Remove artifacts in the LFPs.

"""
__date__ = "May 2022"


import numpy as np


DEFAULT_MAD_TRESHOLD = 15.0
"""Default median absolute deviation threshold for outlier detection"""



def mark_outliers(lfps, mad_threshold=DEFAULT_MAD_TRESHOLD):
    """
    Detect outlying samples in the LFPs.

    Parameters
    ----------
    lfps : dict
        Maps ROI names to LFP waveforms.
    mad_threshold : float, optional
        A median absolute deviation treshold used to determine whether a point
        is an outlier. A lower value marks more points as outliers.

    Returns
    -------
    lfps : dict
        Maps ROI names to LFP waveforms.
    """
    assert mad_threshold > 0.0, "mad_threshold must be positive!"

    for roi in lfps:
        # Subtract out the median.
        trace = np.abs(lfps[roi] - np.median(lfps[roi]))
        # Calculate the MAD and the treshold.
        mad = np.median(trace) # median absolute deviation
        thresh = mad_threshold * mad
        # Mark outlying samples.
        lfps[roi][trace > thresh] = np.nan

    return lfps



if __name__ == '__main__':
    pass


###