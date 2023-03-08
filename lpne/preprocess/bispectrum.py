"""
Estimate the bispectrum and the bispectral power decomposition.

The bispectral power decomposition is described in Shields & Kim, "Simulation of higher
order stochastic processes by spectral representation" (2016).

A reference implementation is on Github: https://github.com/SURGroup/UQpy

"""
__date__ = "December 2022 - February 2023"
__all__ = ["bispectral_power_decomposition", "get_bicoherence", "get_bispectrum"]

import numpy as np
import scipy.fft as sp_fft


def get_bispectrum(
    x, fs=1000, min_freq=0.0, max_freq=55.0, complex=False, return_power=False
):
    """
    Calculate the bispectrum.

    Use ``lpne.squeeze_bispec_array`` and ``lpne.unsqueeze_bispec_array`` to convert to
    and from dense and sparse forms. This function returns the sparse form.

    Parameters
    ----------
    x : numpy.ndarray
        Shape: [n,w,t]
    fs : int, optional
        Samplerate
    max_freq_bins : None or int, optional
        Maximum number of frequency bins
    complex : bool, optional
        Whether to return the full complex bispectrum or its squared modulus
    return_power : bool, optional
        Whether to return the power spectrum

    Returns
    -------
    bispectrum : numpy.ndarray
        If ``complex``, this is the bispectrum. Otherwise it is the squared modulus of
        the bispectrum. Shape: [n,f,f']
    freq : numpy.ndarray
        Frequencies
        Shape: [f]
    power : numpy.ndarray
        Returned if ``return_power``. Shape: [n,f]
    """
    assert x.ndim == 3, f"len({x.shape}) != 3"
    # Remove the DC offset for each window.
    x -= np.mean(x, axis=-1, keepdims=True)

    # Do an FFT.
    freq = sp_fft.rfftfreq(x.shape[-1], d=1 / fs)  # [f]
    fft = sp_fft.rfft(x)  # [n,w,f]
    fft[..., 0] = 0.0  # manually set the zero-frequency component to 0
    power = np.mean(fft * np.conj(fft), axis=1).real  # [n,f]

    # Truncate frequencies.
    i1, i2 = np.searchsorted(freq, [min_freq, max_freq])
    freq = freq[i1:i2]
    power = power[:, i1:i2]

    idx1 = np.arange(len(freq))
    idx2 = idx1[: (len(freq) + 1) // 2]
    idx3a = idx1[:, None]
    idx3b = idx2[None, :]
    idx3 = idx3a + idx3b
    idx3[idx3 >= len(freq)] = 0
    idx3[idx3a < idx3b] = 0
    # [n,w,f,f']
    bispec = fft[..., idx1, None] * fft[..., None, idx2] * np.conj(fft[..., idx3])
    bispec = np.mean(bispec, axis=1)  # [n,f,f']
    if complex and return_power:
        return bispec, freq, power
    elif complex:
        return bispec, freq
    bispec = bispec.real**2 + bispec.imag**2  # [n,f,f']
    if return_power:
        return bispec, freq, power
    return bispec, freq


def get_bicoherence(
    x, fs=1000, min_freq=0.0, max_freq=55.0, return_power=False, eps=1e-8
):
    """
    Calculate the bicoherence.

    Use ``lpne.squeeze_bispec_array`` and ``lpne.unsqueeze_bispec_array`` to convert to
    and from dense and sparse forms. This function returns the sparse form.

    Parameters
    ----------
    x : numpy.ndarray
        Shape: [n,w,t]
    fs : int, optional
        Samplerate
    max_freq_bins : None or int, optional
        Maximum number of frequency bins
    return_power : bool, optional
        Whether to return the power spectrum
    eps : float, optional
        Regularization for the bicoherence denominator

    Returns
    -------
    bicoherence : numpy.ndarray
        If ``complex``, this is the bispectrum. Otherwise it is the squared modulus of
        the bispectrum. Shape: [n,f,f']
    freq : numpy.ndarray
        Frequencies
        Shape: [f]
    power : numpy.ndarray
        Returned if ``return_power``. Shape: [n,f]
    """
    assert x.ndim == 3, f"len({x.shape}) != 3"
    # Remove the DC offset for each window.
    x -= np.mean(x, axis=-1, keepdims=True)

    # Do an FFT.
    freq = sp_fft.rfftfreq(x.shape[-1], d=1 / fs)  # [f]
    fft = sp_fft.rfft(x)  # [n,w,f]
    fft[..., 0] = 0.0  # manually set the zero-frequency component to 0
    power = np.mean(fft * np.conj(fft), axis=1).real  # [n,f]

    # Truncate frequencies.
    i1, i2 = np.searchsorted(freq, [min_freq, max_freq])
    freq = freq[i1:i2]
    power = power[:, i1:i2]

    idx1 = np.arange(len(freq))
    idx2 = idx1[: (len(freq) + 1) // 2]
    idx3a = idx1[:, None]
    idx3b = idx2[None, :]
    idx3 = idx3a + idx3b
    idx3[idx3 >= len(freq)] = 0
    idx3[idx3a < idx3b] = 0
    # [n,w,f,f']
    bispec = fft[..., idx1, None] * fft[..., None, idx2] * np.conj(fft[..., idx3])
    bispec = np.mean(bispec, axis=1)  # [n,f,f']
    bispec = bispec.real**2 + bispec.imag**2  # [n,f,f']
    denom = np.mean(np.abs(fft[..., idx1, None] * fft[..., None, idx2]) ** 2, axis=1)
    denom = denom * power[..., idx3]
    bicoh = bispec / (denom + eps)
    if return_power:
        return bicoh, freq, power
    return bicoh, freq


def bispectral_power_decomposition(x, fs=1000, min_freq=0.0, max_freq=55.0):
    """
    Decompose the the signal into pure power and bispectral power.

    Use ``lpne.squeeze_bispec_array`` and ``lpne.unsqueeze_bispec_array`` to convert to
    and from dense and sparse forms. This function returns the sparse form.

    Parameters
    ----------
    x : numpy.ndarray
        Shape: [n,w,t]
    fs : int, optional
        Samplerate
    max_freq_bins : None or int, optional
        Maximum number of frequency bins

    Returns
    -------
    power_decomp : numpy.ndarray
        The sum along the antidiagonals gives the total power spectrum.
        Shape: [n,f,f']
    freq : numpy.ndarray
        Frequencies
        Shape: [f]
    """
    # Get the squared bispectrum and the corresponding power spectrum.
    bispec, freq, power = get_bispectrum(
        x, fs=fs, min_freq=min_freq, max_freq=max_freq, return_power=True
    )  # [n,f,f'], [n,f]
    f = power.shape[1]

    bc2 = np.zeros_like(bispec)  # squared partial bicoherence [n,f,f']
    sum_bc2 = np.zeros_like(power)  # [n,f]
    pure_power = np.zeros_like(power)  # [n,f]
    pure_power[:, :2] = power[:, :2]

    # Zero-pad the bispectrum.
    diff = f - bispec.shape[-1]
    bispec = np.pad(bispec, ((0, 0), (0, 0), (0, diff)))

    # Collect the squared partial bicoherence.
    for k in range(f):
        for j in range(k // 2 + 1):
            i = k - j
            for n in range(len(bispec)):
                if bispec[n, i, j] > 0 and pure_power[n, i] * pure_power[n, j] != 0:
                    denom = pure_power[n, i] * pure_power[n, j] * power[n, k]
                    bc2[n, i, j] = bispec[n, i, j] / (denom + 1e-8)
                    sum_bc2[n, k] += bc2[n, i, j]
        for n in range(len(bispec)):
            if sum_bc2[n, k] >= 1.0:
                for j in range(k // 2 + 1):
                    i = k - j
                    bc2[n, i, j] /= sum_bc2[n, k]
                sum_bc2[n, k] = 1.0
            if k > 1:
                pure_power[n, k] = power[n, k] * (1.0 - sum_bc2[n, k])

    # Convert the partial bicoherence into the power decomposition.
    power_decomp = np.zeros_like(bc2)
    for i in range(bc2.shape[0]):
        for j in range(bc2.shape[1]):
            k = i + j
            if k < len(power):
                power_decomp[:, i, j] = bc2[:, i, j] * power[:, k]

    # Place the pure power along the first zero-frequency axis and return.
    power_decomp[..., 0] = pure_power[:, : power_decomp.shape[1]]
    return power_decomp, freq


if __name__ == "__main__":
    pass


###
