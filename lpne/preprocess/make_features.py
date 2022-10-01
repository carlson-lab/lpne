"""
Make features

"""
__date__ = "July 2021 - May 2022"


import numpy as np
from scipy.io import loadmat
from scipy.signal import csd

from .directed_spectrum import get_directed_spectrum
from .. import __commit__ as LPNE_COMMIT
from .. import __version__ as LPNE_VERSION
from ..utils.utils import squeeze_triangular_array


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
    directed_spectrum=False,
    csd_params={},
):
    """
    Main function: make features from an LFP waveform.

    For ``0 <= j <= i < n``, the cross power spectral density feature for ROI
    ``i`` and ROI ``j`` is stored at index ``i * (i + 1) // 2 + j``, assuming
    both ``i`` and ``j`` are zero-indexed. When ``i == j``, this is simply the
    power spectral density of the ROI. The ROI order is sorted by the channel
    names.

    See ``lpne.unsqueeze_triangular_array`` and
    ``lpne.squeeze_triangular_array`` to convert the power between dense and
    symmetric forms.

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
    directed_spectrum : bool, optional
        Whether to make directed spectrum features
    csd_params : dict, optional
        Parameters sent to ``scipy.signal.csd``

    Returns
    -------
    res : dict
        'power' : numpy.ndarray
            Cross power spectral density features
            Shape: ``[n_window, n_roi*(n_roi+1)//2, n_freq]``
        'dir_spec' : numpy.ndarray
            Directed spectrum features. Only included if ``directed_spectrum``
            is ``True``.
            Shape: ``[n_window, n_roi, n_roi, n_freq]``
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
    f = f[i1:i2]
    cpsd = np.abs(cpsd[..., i1:i2])
    cpsd = squeeze_triangular_array(cpsd, dims=(1, 2))  # [w,r*(r+1)//2,f]
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

    # Make directed spectrum features.
    if directed_spectrum:
        # f_temp: [f], dir_spec: [w,f,r,r]
        f_temp, dir_spec = get_directed_spectrum(X, fs, csd_params=csd_params)
        i1, i2 = np.searchsorted(f, [min_freq, max_freq])
        f_temp = f_temp[i1:i2]
        assert np.allclose(f, f_temp), f"Frequencies don't match:\n{f}\n{f_temp}"
        dir_spec = np.moveaxis(dir_spec[:, i1:i2], 1, -1)  # [w,r,r,f]
        dir_spec[nan_mask] = np.nan  # reintroduce NaNs
        res["dir_spec"] = dir_spec

    return res


if __name__ == "__main__":
    pass


###
