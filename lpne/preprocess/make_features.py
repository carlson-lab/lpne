"""
Make features

"""
__date__ = "July 2021 - August 2024"


import numpy as np
from scipy.signal import csd

from .directed_measures import get_directed_spectral_measures
from .. import __commit__ as LPNE_COMMIT
from .. import __version__ as LPNE_VERSION
from ..utils.array_utils import squeeze_triangular_array


EPSILON = 1e-6
DEFAULT_CSD_PARAMS = {
    "detrend": "constant",
    "window": "hann",
    "nperseg": 512,
    "noverlap": 256,
    "nfft": None,
}
"""Default parameters sent to ``scipy.signal.csd``"""


def make_features(
    lfps,
    fs=1000,
    min_freq=0.0,
    max_freq=55.0,
    window_duration=5.0,
    window_step=None,
    max_n_windows=None,
    spectral_granger=False,
    directed_spectrum=False,
    pairwise=True,
    phases=False,
    csd_params={},
):
    """
    Main function: make features from an LFP waveform.

    For ``0 <= j <= i < n``, the cross power spectral density feature for ROI ``i`` and
    ROI ``j`` is stored at index ``i * (i + 1) // 2 + j``, assuming both ``i`` and ``j``
    are zero-indexed. When ``i == j``, this is simply the power spectral density of the
    ROI. The ROI order is sorted by the channel names.

    See ``lpne.unsqueeze_triangular_array`` and ``lpne.squeeze_triangular_array`` to
    convert the power between dense and symmetric forms.

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
    spectral_granger : bool, optional
        Whether to make spectral Granger features
    directed_spectrum : bool, optional
        Whether to make directed spectrum features
    pairwise : bool, optional
        Whether spectral Granger and directed spectrum should be pairwise
    phases : bool, optional
        Whether to make phase features
    csd_params : dict, optional
        Parameters sent to ``scipy.signal.csd``

    Returns
    -------
    res : dict
        'power' : numpy.ndarray
            Cross power spectral density features
            Shape: ``[n_window, n_roi*(n_roi+1)//2, n_freq]``
        'spectral_granger' : numpy.ndarray
            Spectral granger features. Only included if ``spectral_granger``.
            Shape: ``[n_window, n_roi, n_roi, n_freq]``
        'dir_spec' : numpy.ndarray
            Directed spectrum features. Only included if ``directed_spectrum``.
            Shape: ``[n_window, n_roi, n_roi, n_freq]``
        'phase' : numpy.ndarray
            Phase features. Only included if ``phases``.
            Shape: ``[n_window, n_roi*(n_roi+1)//2, n_freq]``
        'freq' : numpy.ndarray
            Frequency bins
            Shape: ``[n_freq]``
        'rois' : list of str
            Sorted list of grouped channel names
        '__commit__' : str
            Git commit of LPNE package
        '__version__' : str
            Version number of LPNE package
    """
    assert (
        window_step is None or window_step > 0.0
    ), f"Window step ({window_step}) must be greater than zero!"
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

    # Truncate the frequencies.
    i1, i2 = np.searchsorted(f, [min_freq, max_freq])
    f = f[i1:i2]
    cpsd = cpsd[..., i1:i2]

    # Condense the cross power to a symmetric form.
    cpsd = squeeze_triangular_array(cpsd, dims=(1, 2))  # [w,r*(r+1)//2,f]
    
    # Collect the phase information.
    if phases:
        phase = np.angle(cpsd) # [w,r*(r+1)//2,f]
        phase[nan_mask] = np.nan  # reintroduce NaNs

    # Get the frequency-scaled cross power.
    cpsd = np.abs(cpsd)
    cpsd[:, :] *= f  # scale the power features by frequency
    cpsd[nan_mask] = np.nan  # reintroduce NaNs

    # Assemble features.
    res = {
        "power": cpsd,
        "freq": f,
        "rois": rois,
        "__commit__": LPNE_COMMIT,
        "__version__": LPNE_VERSION,
    }

    # Add the phase features.
    if phases:
        res["phase"] = phase

    # Make directed spectrum features.
    if spectral_granger or directed_spectrum:
        # f_temp: [f], sg and ds: [w,f,r,r]
        temp_res = get_directed_spectral_measures(
            X,
            fs,
            return_spectral_granger=spectral_granger,
            return_directed_spectrum=directed_spectrum,
            pairwise=pairwise,
            csd_params=csd_params,
        )
        # Figure out frequencies.
        f_temp = temp_res[0]
        i1, i2 = np.searchsorted(f, [min_freq, max_freq])
        f_temp = f_temp[i1:i2]
        assert np.allclose(f, f_temp), f"Frequencies don't match:\n{f}\n{f_temp}"
        f_reshape = f_temp.reshape(1, -1, 1, 1)
        if spectral_granger:
            sg = temp_res[1][:, i1:i2] # don't scale by frequency
            sg = np.moveaxis(sg, 1, -1)  # [w,r,r,f]
            sg[nan_mask] = np.nan  # reintroduce NaNs
            res["spectral_granger"] = sg
        if directed_spectrum:
            ds = temp_res[-1][:, i1:i2] * f_reshape  # scale by frequency
            ds = np.moveaxis(ds, 1, -1)  # [w,r,r,f]
            ds[nan_mask] = np.nan  # reintroduce NaNs
            res["dir_spec"] = ds

    return res


if __name__ == "__main__":
    pass


###
