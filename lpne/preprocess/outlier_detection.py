"""
Remove artifacts in the LFPs.

"""
__date__ = "May 2022"


import numpy as np

from .filter import filter_signal


DEFAULT_MAD_TRESHOLD = 15.0
"""Default median absolute deviation threshold for outlier detection"""
LOWCUT = 30.0  # Butterworth bandpass filter parameter
"""Default lowcut for filtering (Hz)"""
HIGHCUT = 55.0  # Butterworth bandpass filter parameter
"""Default highcut for filtering (Hz)"""


def mark_outliers(
    lfps, fs, lowcut=LOWCUT, highcut=HIGHCUT, mad_threshold=DEFAULT_MAD_TRESHOLD
):
    """
    Detect outlying samples in the LFPs.

    Parameters
    ----------
    lfps : dict
        Maps ROI names to LFP waveforms.
    fs : int, optional
        Samplerate
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
        # Copy the signal.
        trace = np.copy(lfps[roi])
        # Filter the signal.
        trace = filter_signal(
            trace,
            fs,
            lowcut=lowcut,
            highcut=highcut,
            apply_notch_filters=False,
        )
        # Subtract out the median and rectify.
        trace = np.abs(trace - np.median(trace))
        # Calculate the MAD and the treshold.
        mad = np.median(trace)  # median absolute deviation
        thresh = mad_threshold * mad
        # Mark outlying samples.
        lfps[roi][trace > thresh] = np.nan
    return lfps


if __name__ == "__main__":
    pass


###
