"""
Estimate the bispectrum using Welch's method

Most of this is adapted from ``scipy.signal``.

"""
__date__ = "December 2022"
__all__ = ["bispectrum"]

import numpy as np
from scipy import fft as sp_fft
from scipy.signal import detrend as sp_detrend
from scipy.signal import get_window


def bispectrum(
    x,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    average="mean",
):
    """

    Parameters
    ----------
    x : array_like
        Array or sequence containing the data to be analyzed.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross
        spectrum ('spectrum') where `Pxy` has units of V**2, if `x`
        and `y` are measured in V and `fs` is measured in Hz.
        Defaults to 'density'

    Returns
    -------
    freqs :
    bispectrum :

    """
    freqs, _, Pxy = _spectral_helper(
        x,
        x,
        fs,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        return_onesided,
        scaling,
        axis,
        mode="psd",
    )

    assert Pxy.ndim == x.ndim + 2
    # Average over windows.
    if len(Pxy.shape) >= 2 and Pxy.size > 0:
        if Pxy.shape[-3] > 1:
            if average == "median":
                # np.median must be passed real arrays for the desired result
                bias = _median_bias(Pxy.shape[-3])
                if np.iscomplexobj(Pxy):
                    Pxy = np.median(np.real(Pxy), axis=-3) + 1j * np.median(
                        np.imag(Pxy), axis=-3
                    )
                else:
                    Pxy = np.median(Pxy, axis=-3)
                Pxy /= bias
            elif average == "mean":
                Pxy = Pxy.mean(axis=-3)
            else:
                raise ValueError(
                    'average must be "median" or "mean", got %s' % (average,)
                )
        else:
            Pxy = np.reshape(Pxy, Pxy.shape[:-3] + Pxy.shape[-2:])

    return freqs, Pxy


def _spectral_helper(
    x,
    y,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    mode="psd",
    boundary=None,
    padded=False,
):
    """Calculate various forms of windowed FFTs for PSD, CSD, etc.
    This is a helper function that implements the commonality between
    the stft, psd, csd, and spectrogram functions. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Parameters
    ----------
    x : array_like
        Array or sequence containing the data to be analyzed.
    y : array_like
        Array or sequence containing the data to be analyzed. If this is
        the same object in memory as `x` (i.e. ``_spectral_helper(x,
        x, ...)``), the extra computations are spared.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross
        spectrum ('spectrum') where `Pxy` has units of V**2, if `x`
        and `y` are measured in V and `fs` is measured in Hz.
        Defaults to 'density'

    Returns
    -------
    freqs : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of times corresponding to each data segment
    result : ndarray
        Array of output data, contents dependent on *mode* kwarg.

    Notes
    -----
    Adapted from matplotlib.mlab

    .. versionadded:: 0.16.0
    """
    # Assertions so that I don't have to implement everything.
    assert mode == "psd"
    assert boundary is None
    assert return_onesided
    assert not padded
    assert x is y
    assert axis == -1
    assert isinstance(window, str)
    assert not np.iscomplexobj(x)
    assert scaling == "density"

    # Ensure we have np.arrays, get outdtype
    x = np.asarray(x)
    outdtype = np.result_type(x, np.complex64)

    if x.size == 0:
        return np.empty(x.shape), np.empty(x.shape), np.empty(x.shape)

    if nperseg is not None:  # if specified by user
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError("nperseg must be a positive integer")
    else:
        nperseg = 256

    # parse window; if array like, then set nperseg = win.shape
    if nperseg > x.shape[-1]:
        nperseg = x.shape[-1]
    win = get_window(window, nperseg)

    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError("nfft must be greater than or equal to nperseg.")
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg.")

    # Handle detrending and window functions
    if not detrend:

        def detrend_func(d):
            return d

    elif not hasattr(detrend, "__call__"):

        def detrend_func(d):
            return sp_detrend(d, type=detrend, axis=-1)

    else:
        detrend_func = detrend

    if np.result_type(win, np.complex64) != outdtype:
        win = win.astype(outdtype)

    # TODO: what scaling makes sense?
    if scaling == "density":
        scale = 1.0 / (fs * (win * win).sum())
    elif scaling == "spectrum":
        scale = 1.0 / win.sum() ** 2
    else:
        raise ValueError("Unknown scaling: %r" % scaling)

    freqs = sp_fft.rfftfreq(nfft, 1 / fs)

    # Perform the windowed FFTs
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft)
    print("result", result.shape, "x", x.shape)

    result /= (win * win * win).sum()

    # HERE!
    nfreq = result.shape[-1]
    nfreq2 = (nfreq + 1) // 2
    print("nfreq", nfreq)
    # if nfreq % 2 == 1:

    res = np.zeros(
        result.shape[:-1]
        + (
            nfreq2,
            nfreq,
        ),
        dtype=result.dtype,
    )
    for i in range(nfreq):
        for j in range(i, nfreq - i):
            temp = result[..., i] * result[..., j] * np.conj(result[..., i + j])
            res[..., i, j] = temp
            # res[...,i,j] = temp
    result = res

    print("result", result.shape)
    # quit()
    # # HERE!
    # result[...,-1] *= scale
    # if nfft % 2:
    #     result[..., 1:] *= 2
    # else:
    #     # Last point is unpaired Nyquist freq point, don't double
    #     result[..., 1:-1] *= 2

    result /= fs

    time = np.arange(
        nperseg / 2, x.shape[-1] - nperseg / 2 + 1, nperseg - noverlap
    ) / float(fs)

    # # Output is going to have new last axis for time/window index, so a
    # # negative axis index shifts down one
    # if axis < 0:
    #     axis -= 1

    # # Roll frequency axis back to axis where the data came from
    # result = np.moveaxis(result, -1, axis)

    return freqs, time, result


def _median_bias(n):
    """
    Returns the bias of the median of a set of periodograms relative to
    the mean.

    Copied from ``scipy.signal._spectral_py.py``.

    See Appendix B from [1]_ for details.

    Parameters
    ----------
    n : int
        Numbers of periodograms being averaged.

    Returns
    -------
    bias : float
        Calculated bias.

    References
    ----------
    .. [1] B. Allen, W.G. Anderson, P.R. Brady, D.A. Brown, J.D.E. Creighton.
           "FINDCHIRP: an algorithm for detection of gravitational waves from
           inspiraling compact binaries", Physical Review D 85, 2012,
           :arxiv:`gr-qc/0509116`
    """
    ii_2 = 2 * np.arange(1.0, (n - 1) // 2 + 1)
    return 1 + np.sum(1.0 / (ii_2 + 1) - 1.0 / ii_2)


def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft):
    """
    Calculate windowed FFT, for internal use by `scipy.signal._spectral_helper`.

    This is a helper function that does the main FFT calculation for
    `_spectral helper`. All input validation is performed there, and the
    data axis is assumed to be the last axis of x. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Returns
    -------
    result : ndarray
        Array of FFT data

    Notes
    -----
    Adapted from matplotlib.mlab

    .. versionadded:: 0.16.0
    """
    # Created strided array of data segments
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        # https://stackoverflow.com/a/5568169
        step = nperseg - noverlap
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
        strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # Detrend each data segment individually
    result = detrend_func(result)

    # Apply window by multiplication
    result = win * result

    # Perform the FFT. Acts on last axis by default. Zero-pads automatically
    result = result.real
    result = sp_fft.rfft(result, n=nfft)

    return result


if __name__ == "__main__":
    x = np.random.randn(2, 1001)

    N = 10001
    t = np.linspace(0, 100, N)
    fs = 1 / (t[1] - t[0])
    s1 = np.cos(2 * np.pi * 5 * t + 0.2)
    s2 = 3 * np.cos(2 * np.pi * 7 * t + 0.5)
    np.random.seed(0)
    noise = 5 * np.random.normal(0, 1, N)
    x = s1 + s2 + 0.5 * s1 * s2 + noise
    x = x.reshape(1, -1)

    freq, bispec = bispectrum(x, fs=1000, nperseg=64)
    print("freq", freq.shape)
    print("bispec", bispec.shape)

    print(bispec.dtype)
    bispec = np.abs(bispec).real

    import matplotlib.pyplot as plt

    plt.imshow(
        bispec[0],
        extent=[freq[0], freq[-1], freq[0], freq[-1] // 2],
        origin="lower",
        aspect="equal",
    )
    plt.colorbar()
    plt.show()


###
