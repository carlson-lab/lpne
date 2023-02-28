"""
Estimate the bispectrum

"""
__date__ = "December 2022 - February 2023"
__all__ = ["bispectral_power_decomposition"]

import numpy as np
import scipy.fft as sp_fft


def fft_power(x):
    # Remove the DC offset.
    x -= np.mean(x, axis=1, keepdims=True)
    f = sp_fft.rfftfreq(len(x))
    return np.abs(sp_fft.rfft(x)) ** 2, f


def fft_bispec(x, max_f):
    """
    Estimate the bispectrum.

    Parameters
    ----------
    x : numpy.ndarray
        Shape: [n,t]
    max_f : int
        HERE

    Returns
    -------
    bispec : numpy.ndarray
        Shape: [f,f]
    """
    # Remove the DC offset.
    x -= np.mean(x, axis=1, keepdims=True)
    f = sp_fft.rfftfreq(x.shape[1])
    fft = sp_fft.rfft(x)

    max_f = min(max_f, len(f))
    f, fft = f[:max_f], fft[:max_f]

    # NOTE: HERE!

    print("f", f.shape, "fft", fft.shape)
    idx = np.arange(len(f) // 2 + 1)
    idx2 = idx[:, None] + idx[None, :]
    bispec = fft[:, idx, None] * fft[:, None, idx] * np.conj(fft[:, idx2])
    bispec = np.mean(bispec, axis=0)
    return bispec


def get_bispectrum(x, max_freq_bins=None, complex=False, return_power=False):
    """
    Calculate the bispectrum

    Parameters
    ----------
    x : numpy.ndarray
        Shape: [n,t]
    max_freq_bins : None or int, optional
        Maximum number of frequency bins
    complex : bool, optional
        Whether to return the complex bispectrum or its squared modulus
    return_power : bool, optional
        Whether to return the power spectrum

    Returns
    -------
    bispectrum : numpy.ndarray
        If ``complex``, this is the bispectrum. Otherwise it is the squared modulus of
        the bispectrum. Shape: [f,f']
    power : numpy.ndarray
        Returned if ``return_power``. Shape: [f]
    """
    assert x.ndim == 2, f"len({x.shape}) != 2"
    # Remove the DC offset.
    x -= np.mean(x, axis=1, keepdims=True)

    # Do an FFT.
    freq = sp_fft.rfftfreq(x.shape[1])  # [f]
    fft = sp_fft.rfft(x)  # [n,f]
    power = np.mean(fft * np.conj(fft), axis=0).real  # [f]

    if max_freq_bins is None:
        f = fft.shape[1]
    else:
        f = min(max_freq_bins, fft.shape[1])
        power, freq = power[:f], freq[:f]
    if return_power:
        to_return = (power,)
    else:
        to_return = tuple()

    idx1 = np.arange(len(freq))
    idx2 = idx1[: (len(freq) + 1) // 2]
    idx3a = idx1[:, None]
    idx3b = idx2[None, :]
    idx3 = idx3a + idx3b
    idx3[idx3 >= len(freq)] = 0
    idx3[idx3a < idx3b] = 0
    bispec = fft[:, idx1, None] * fft[:, None, idx2] * np.conj(fft[:, idx3])  # [n,f,f']
    if complex:
        return (bispec,) + to_return
    bispec = np.mean(bispec, axis=0)  # [f,f']
    bispec = bispec.real**2 + bispec.imag**2  # [f,f']
    return (bispec,) + to_return


def bispectral_power_decomposition(x, max_freq_bins=None):
    """
    Decompose the the signal into pure power and bispectral power.

    Parameters
    ----------
    x : numpy.ndarray
        Shape: [n,t]
    max_freq_bins : None or int, optional
        Maximum number of frequency bins

    Returns
    -------
    power_decomp : numpy.ndarray
        The sum along the antidiagonals gives the total power spectrum.
        Shape: [f,f']
    """
    # Get the squared bispectrum and the corresponding power spectrum.
    bispec, power = get_bispectrum(
        x, max_freq_bins=max_freq_bins, return_power=True
    )  # [f,f'], [f]
    f = len(power)

    bc2 = np.zeros_like(bispec)  # squared partial bicoherence [f,f]
    sum_bc2 = np.zeros_like(power)  # [f]
    pure_power = np.zeros_like(power)  # [f]
    pure_power[:2] = power[:2]

    # Zero-pad the bispectrum.
    diff = f - bispec.shape[1]
    bispec = np.pad(bispec, ((0, 0), (0, diff)))

    # Collect the squared partial bicoherence.
    for k in range(f):
        for j in range(k // 2 + 1):
            i = k - j
            if bispec[i, j] > 0 and pure_power[i] * pure_power[j] != 0:
                denom = pure_power[i] * pure_power[j] * power[k]
                bc2[i, j] = bispec[i, j] / (denom + 1e-8)
                sum_bc2[k] += bc2[i, j]
        if sum_bc2[k] >= 1.0:
            for j in range(k // 2 + 1):
                i = k - j
                bc2[i, j] /= sum_bc2[k]
            sum_bc2[k] = 1.0
        if k > 1:
            pure_power[k] = power[k] * (1.0 - sum_bc2[k])

    # Convert the partial bicoherence into the power decomposition.
    power_decomp = np.zeros_like(bc2)
    for i in range(bc2.shape[0]):
        for j in range(bc2.shape[1]):
            k = i + j
            if k < len(power):
                power_decomp[i, j] = bc2[i, j] * power[k]

    # Place the pure power along the first zero-frequency axis and return.
    power_decomp[:, 0] = pure_power[: power_decomp.shape[0]]
    return power_decomp


if __name__ == "__main__":
    np.random.seed(42)

    N = 201
    t = np.linspace(0, 100, N)
    xs = []
    for i in range(50):
        s1 = np.cos(2 * np.pi * 5 * t + 0.2)
        s2 = 3 * np.cos(2 * np.pi * 7 * t + 0.5)
        noise = 5 * np.random.normal(0, 1, N)
        signal = s1 + s2 + 0.5 * s1 * s2 + noise
        # xs.append(signal)
        xs.append(noise)
    x = np.array(xs)

    decomp = bispectral_power_decomposition(x, max_freq_bins=50)

    import matplotlib.pyplot as plt

    plt.imshow(decomp.T, origin="lower")
    plt.colorbar()
    plt.savefig("temp.pdf")


###
