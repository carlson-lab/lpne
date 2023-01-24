"""
Calculate the phase-slope index

"""
__date__ = "January 2023"
__all__ = ["get_psi"]


import numpy as np
from scipy.signal import csd

from .make_features import EPSILON, DEFAULT_CSD_PARAMS


def get_psi(
    lfps,
    fs=1000,
    min_freq=0.0,
    max_freq=55.0,
    window_duration=5.0,
    window_step=None,
    max_n_windows=None,
    csd_params={},
):
    """
    Calculate the Phase-Slope Index (PSI).

    This isn't summed over frequencies as in the original, found here:

    > Nolte, G., Ziehe, A., Nikulin, V. V., Schlögl, A., Krämer, N., Brismar, T., &
    > Müller, K. R. (2008). Robustly estimating the flow direction of information in
    > complex physical systems. Physical review letters, 100(23), 234101.

    Parameters
    ----------
    lfps : dict
        Maps region names to LFP waveforms.
    fs : int, optional
        LFP samplerate
    min_freq : float, optional
        Minimum frequency
    max_freq : float, optional
        Maximum frequency
    window_duration : float, optional
        Window duration, in seconds
    window_step : None or float, optional
        Time between consecutive window onsets, in seconds. If ``None``, this is
        set to ``window_duration``.
    max_n_windows : None or int, optional
        Maximum number of windows
    csd_params : dict, optional
        Parameters sent to ``scipy.signal.csd``

    Returns
    -------
    res : dict
        'psi' : numpy.ndarray
            Phase slope index features
            Shape: ``[n_window, n_roi, n_roi, n_freq]``
        'freq' : numpy.ndarray
            Frequency bins
            Shape: ``[n_freq]``
        'rois' : list of str
            Sorted list of grouped channel names
    """
    assert (
        window_step is None or window_step > 0.0
    ), f"Nonpositive window step: {window_step}"
    assert max_n_windows is None or max_n_windows > 0
    rois = sorted(lfps.keys())
    n = len(rois)
    assert n >= 1, f"{n} < 1"
    duration = len(lfps[rois[0]]) / fs
    assert (
        duration >= window_duration
    ), f"LFPs are too short: {duration} < {window_duration}"
    window_samp = int(fs * window_duration)
    csd_params = {**DEFAULT_CSD_PARAMS, **csd_params}

    # Stack the LFPs into a big array, X: [r,t]
    X = np.vstack([lfps[rois[i]].flatten() for i in range(len(rois))])
    if window_step is None:
        # No window overlap: reshape the data.
        idx = (X.shape[1] // window_samp) * window_samp
        X = X[:, :idx]  # [r,t]
        X = X.reshape(X.shape[0], -1, window_samp).transpose(1, 0, 2)  # [w,r,t]
        if max_n_windows is not None:
            X = X[:max_n_windows]  # [w,r,t]
    else:
        # Window overlap: copy the data.
        onsets = np.arange(
            0.0,
            duration - window_duration + EPSILON,
            window_step,
        )
        if max_n_windows is not None:
            onsets = onsets[:max_n_windows]
        temp_X = []
        for k in range(len(onsets)):
            k1 = int(fs * onsets[k])
            k2 = k1 + window_samp
            temp_X.append(X[:, k1:k2])
        X = np.stack(temp_X, axis=0)  # [w,r,t]
    assert X.ndim == 3, f"len({X.shape}) != 3"

    # Make cross power spectral density features for each pair of ROIs.
    # f: [f], cpsd: [w,r,r,f]
    nan_mask = np.sum(np.isnan(X), axis=(1, 2)) != 0
    X[nan_mask] = np.random.randn(*X[nan_mask].shape)
    f, cpsd = csd(
        X[:, :, np.newaxis],
        X[:, np.newaxis],
        fs=fs,
        **csd_params,
    )
    i1, i2 = np.searchsorted(f, [min_freq, max_freq])
    assert i2 < len(f), f"Need {i1 - len(f) + 1} more frequency bin(s)!"
    f = f[i1:i2]
    cpsd = cpsd[..., i1 : i2 + 1]  # [w,r,r,f]

    # Calculate the phase-slope index.
    amp = np.sqrt(np.diagonal(cpsd, 0, 1, 2).real + EPSILON)  # [w,f,r]
    amp = np.moveaxis(amp, 1, -1) # [w,r,f]
    coh = cpsd / (amp[:, np.newaxis] * amp[:, :, np.newaxis])  # [w,r,r,f]
    psi = np.imag(np.conj(coh[..., :-1]) * coh[..., 1:])  # [w,r,r,f]
    psi[nan_mask] = np.nan  # reintroduce NaNs

    res = {
        "psi": psi,
        "freq": f,
        "rois": rois,
    }
    return res


if __name__ == "__main__":
    pass


###
